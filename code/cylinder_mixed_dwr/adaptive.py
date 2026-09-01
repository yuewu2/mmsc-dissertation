r"""Stages E--F: causal slab refinement and the full adaptive cylinder loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from firedrake import Function, FunctionSpace
from firedrake.petsc import PETSc

from automated_DWR.mark_refine import (
    inherit_refinement_marks,
    mark_spacetime_cells,
    mark_spacetime_vertex_patches,
    refine_marked_mesh,
)
from navier_stokes_cylinder_irksome_static_primal import _make_hierarchy

from .benchmark import drag_history_diagnostics, mean_drag_from_history
from .directional_split import compute_linear_directional_split
from .estimator import estimate_linear_dwr_global, estimate_symmetric_dwr
from .localisation import (
    localise_linear_dwr,
    localise_symmetric_dwr,
    localise_symmetric_two_term_dwr,
    uniform_global_only_localisation,
)
from .slabwise import (
    build_slab_transfers,
    interpolate_enriched_adjoint_to_low,
    solve_slabwise_adjoint,
    solve_slabwise_enriched_primal,
    solve_slabwise_primal,
    solve_temporally_refined_enriched_adjoint,
    project_enriched_adjoint_to_low,
)


@dataclass(frozen=True)
class CylinderAdaptiveConfig:
    goal_functional: str = "r5_surface_mean_drag"
    hierarchy_levels: int = 1
    geometry_degree: int = 2
    initial_time_slabs: int = 2
    final_time: float = 0.125
    viscosity: float = 1.0e-3
    primal_time_degree: int = 1
    enriched_time_degree: int = 2
    enriched_time_refinement_factor: int = 1
    max_iterations: int = 2
    tolerance: float = 1.0e-8
    true_goal_error_tolerance: float | None = None
    theta: float = 0.30
    time_marked_fraction: float = 0.05
    time_marking_strategy: str = "cell_fraction"
    time_score_source: str = "marked_fraction"
    time_fixed_rate: float = 0.20
    time_bulk_fraction: float = 0.20
    time_max_fraction: float = 0.15
    time_max_count: int = 15
    enable_space_refinement: bool = True
    enable_time_refinement: bool = True
    uniform_refinement: bool = False
    space_marking_strategy: str = "cellwise"
    space_refinement_mode: str = "independent"
    interface_transfer_mode: str = "stokes_l2"
    primal_space_family: str = "taylor_hood"
    enriched_velocity_degree: int = 4
    enriched_pressure_degree: int = 2
    dwr_identity_relative_tolerance: float = 1.0e-2
    dwr_stationarity_relative_tolerance: float = 1.0e-4
    dual_weight_mode: str = "enriched_minus_interpolant"
    dual_base_strategy: str = "interpolated_enriched"
    directional_split_diagnostic: bool = False
    include_cubic_remainder: bool = False
    estimator_strategy: str = "primal_only"
    three_term_marking_strategy: str = "full"
    # The reference dG estimator uses QGauss(high time degree + 5), which is
    # seven points for the enriched dG(2) trajectory used here.
    quadrature_points: int = 7
    recovery_space_degree: int = 2
    facet_recovery_degree: int = 2
    recovery_time_degree: int = 2
    maximum_recovery_gap_relative: float = 0.05
    report_every: int = 0
    reference_mean_drag: float | None = None

    def __post_init__(self):
        if self.goal_functional != "r5_surface_mean_drag":
            raise ValueError(
                "Only the R5 surface mean-drag goal is supported."
            )
        if self.hierarchy_levels < 0:
            raise ValueError("hierarchy_levels must be nonnegative.")
        if self.geometry_degree < 1:
            raise ValueError("geometry_degree must be at least one.")
        if self.initial_time_slabs < 1 or self.max_iterations < 1:
            raise ValueError("At least one time slab and adaptive iteration are required.")
        if self.final_time <= 0.0 or self.viscosity <= 0.0:
            raise ValueError("final_time and viscosity must be positive.")
        if (
            self.true_goal_error_tolerance is not None
            and self.true_goal_error_tolerance <= 0.0
        ):
            raise ValueError("true_goal_error_tolerance must be positive.")
        if self.enriched_time_degree < self.primal_time_degree:
            raise ValueError(
                "The enriched time degree must not be lower than the primal degree."
            )
        if self.enriched_time_refinement_factor < 1:
            raise ValueError(
                "enriched_time_refinement_factor must be at least one."
            )
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("theta must lie in (0,1].")
        if not 0.0 <= self.time_marked_fraction <= 1.0:
            raise ValueError("time_marked_fraction must lie in [0,1].")
        if self.time_marking_strategy not in {
            "cell_fraction",
            "cell_fraction_capped",
            "fixed_rate",
            "slab_bulk_capped",
        }:
            raise ValueError(
                "time_marking_strategy must be 'cell_fraction', "
                "'cell_fraction_capped', 'fixed_rate', or "
                "'slab_bulk_capped'."
            )
        if self.time_score_source not in {
            "marked_fraction",
            "combined_indicator",
            "directional_time",
        }:
            raise ValueError(
                "time_score_source must be 'marked_fraction', "
                "'combined_indicator', or 'directional_time'."
            )
        if self.time_score_source in {
            "combined_indicator", "directional_time"
        } and self.time_marking_strategy in {
            "cell_fraction", "cell_fraction_capped"
        }:
            raise ValueError(
                f"time_score_source={self.time_score_source!r} requires "
                "a score-driven strategy ('fixed_rate' or "
                "'slab_bulk_capped'); cell-fraction strategies only "
                "count spatially marked cells and ignore slab scores."
            )
        if self.time_score_source == "directional_time":
            if not self.directional_split_diagnostic:
                raise ValueError(
                    "time_score_source='directional_time' requires "
                    "directional_split_diagnostic=True: the temporal slab "
                    "scores are the localised eta_k of the R5 split."
                )
        if not 0.0 < self.time_fixed_rate <= 1.0:
            raise ValueError("time_fixed_rate must lie in (0,1].")
        if not 0.0 < self.time_bulk_fraction <= 1.0:
            raise ValueError("time_bulk_fraction must lie in (0,1].")
        if not 0.0 < self.time_max_fraction <= 1.0:
            raise ValueError("time_max_fraction must lie in (0,1].")
        if self.time_max_count < 1:
            raise ValueError("time_max_count must be positive.")
        if self.space_refinement_mode not in {"independent", "causal", "common"}:
            raise ValueError(
                "space_refinement_mode must be 'independent', 'causal', or 'common'."
            )
        if self.space_marking_strategy not in {"cellwise", "vertex_patch"}:
            raise ValueError(
                "space_marking_strategy must be 'cellwise' or 'vertex_patch'."
            )
        if self.interface_transfer_mode not in {"mass", "stokes_l2", "stokes_h1"}:
            raise ValueError(
                "Taylor--Hood interfaces support 'mass', 'stokes_l2', or "
                "'stokes_h1'."
            )
        if self.primal_space_family != "taylor_hood":
            raise ValueError("Only the Taylor--Hood CG2/CG1 primal is supported.")
        if self.enriched_velocity_degree < 3:
            raise ValueError("enriched_velocity_degree must be at least three.")
        if self.enriched_pressure_degree < 2:
            raise ValueError("enriched_pressure_degree must be at least two.")
        if self.enriched_velocity_degree <= self.enriched_pressure_degree:
            raise ValueError(
                "The enriched velocity degree must exceed the pressure degree."
            )
        if self.dwr_identity_relative_tolerance <= 0.0:
            raise ValueError("dwr_identity_relative_tolerance must be positive.")
        if self.dwr_stationarity_relative_tolerance <= 0.0:
            raise ValueError("dwr_stationarity_relative_tolerance must be positive.")
        if self.dual_weight_mode not in {
            "enriched_minus_numerical", "enriched_minus_interpolant"
        }:
            raise ValueError("Unsupported dual_weight_mode.")
        if self.dual_base_strategy not in {
            "numerical", "projected_enriched", "interpolated_enriched"
        }:
            raise ValueError(
                "dual_base_strategy must be 'numerical', "
                "'projected_enriched', or 'interpolated_enriched'."
            )
        if self.estimator_strategy == "primal_only":
            expected_base = (
                "numerical"
                if self.dual_weight_mode == "enriched_minus_numerical"
                else "interpolated_enriched"
            )
            if self.dual_base_strategy != expected_base:
                raise ValueError(
                    "dual_base_strategy is not an independent algorithmic "
                    "choice in the strict linear path: it must match "
                    f"dual_weight_mode (expected {expected_base!r})."
                )
        if (
            self.directional_split_diagnostic
            and self.dual_weight_mode != "enriched_minus_interpolant"
        ):
            raise ValueError(
                "The directional split requires the nodal "
                "enriched-minus-interpolant dual weight."
            )
        if self.estimator_strategy not in {
            "primal_only",
            "symmetric_three_term",
            "symmetric_two_term",
        }:
            raise ValueError("Unsupported estimator_strategy.")
        if self.estimator_strategy != "primal_only":
            raise ValueError(
                "The R5 surface mean-drag goal is currently enabled only "
                "for the strict linear/primal-residual-only estimator."
            )
        if (
            self.enriched_time_refinement_factor != 1
            and self.estimator_strategy != "primal_only"
        ):
            raise ValueError(
                "Independent enriched temporal refinement is currently "
                "implemented only for the primal-residual-only linear DWR path."
            )
        if self.three_term_marking_strategy not in {"full", "residual_pair"}:
            raise ValueError(
                "three_term_marking_strategy must be 'full' or 'residual_pair'."
            )
        if self.reference_mean_drag is not None and not np.isfinite(
            self.reference_mean_drag
        ):
            raise ValueError("reference_mean_drag must be finite when supplied.")


@dataclass
class GridRefinement:
    times: np.ndarray
    meshes: list[Any | None]
    causal_marks: list[np.ndarray | None]
    time_marked: set[int]
    parent_slab: list[int | None]


@dataclass
class CylinderAdaptiveResult:
    config: CylinderAdaptiveConfig
    history: list[dict[str, Any]]
    times: np.ndarray
    meshes: list[Any | None]
    primal: dict[str, Any]
    primal_enriched: dict[str, Any] | None
    dual_low: dict[str, Any]
    dual_enriched: dict[str, Any]
    estimate: Any
    localisation: Any


def _time_marks(
    marked_by_slab,
    threshold: float,
    *,
    strategy: str = "cell_fraction",
    fixed_rate: float = 0.20,
) -> tuple[set[int], list[float]]:
    marked: set[int] = set()
    fractions = [0.0] * len(marked_by_slab)
    for n in range(1, len(marked_by_slab)):
        mask = np.asarray(marked_by_slab[n], dtype=bool)
        fractions[n] = float(np.count_nonzero(mask)) / float(mask.size)
        if strategy in {"cell_fraction", "cell_fraction_capped"} and np.any(mask) and fractions[n] >= float(threshold):
            marked.add(n)
    if strategy == "fixed_rate":
        nslabs = len(marked_by_slab) - 1
        count = max(1, int(np.ceil(float(fixed_rate) * nslabs)))
        ranked = sorted(
            range(1, nslabs + 1), key=lambda n: (-fractions[n], n)
        )
        marked = set(ranked[:count])
    elif strategy not in {
        "cell_fraction", "cell_fraction_capped", "slab_bulk_capped"
    }:
        raise ValueError("Unsupported time marking strategy.")
    return marked, fractions


def _cell_fraction_capped_marks(
    fractions: list[float],
    *,
    threshold: float,
    max_fraction: float,
    max_count: int,
) -> set[int]:
    r"""Apply the thesis cell-fraction trigger with a growth cap.

    A slab remains eligible only when the fraction of spatially marked cells
    reaches ``threshold``.  If too many slabs are eligible, retain those with
    the largest marked fractions subject to

        #M <= min(ceil(max_fraction * N_t), max_count).

    The uncapped ``cell_fraction`` strategy is unchanged.
    """
    nslabs = len(fractions) - 1
    if nslabs <= 0:
        return set()
    cap = min(
        int(max_count),
        max(1, int(np.ceil(float(max_fraction) * nslabs))),
    )
    eligible = [
        n for n in range(1, nslabs + 1)
        if float(fractions[n]) >= float(threshold)
    ]
    ranked = sorted(eligible, key=lambda n: (-float(fractions[n]), n))
    return set(ranked[:cap])


def _slab_bulk_capped_marks(
    indicator_by_slab,
    *,
    bulk_fraction: float,
    max_fraction: float,
    max_count: int,
) -> tuple[set[int], list[float]]:
    """Dorfler-rank slab activities with an explicit per-iteration cap."""
    scores = _absolute_slab_scores(indicator_by_slab)
    nslabs = len(scores) - 1
    if nslabs <= 0 or sum(scores[1:]) <= 0.0:
        return set(), scores
    cap = min(
        int(max_count),
        max(1, int(np.ceil(float(max_fraction) * nslabs))),
    )
    target = float(bulk_fraction) * float(sum(scores[1:]))
    ranked = sorted(range(1, nslabs + 1), key=lambda n: (-scores[n], n))
    marked: set[int] = set()
    subtotal = 0.0
    for slab in ranked[:cap]:
        marked.add(slab)
        subtotal += scores[slab]
        if subtotal >= target:
            break
    return marked, scores


def _fixed_rate_score_marks(
    scores: list[float], rate: float
) -> set[int]:
    """Mark the ``ceil(rate * nslabs)`` slabs with the largest scores.

    This is R5's temporal fixed-number strategy (their 2D-3 runs bisect the
    top 75% of temporal elements every loop) applied to explicit slab
    scores instead of spatial marked-cell fractions.
    """
    nslabs = len(scores) - 1
    if nslabs <= 0 or sum(scores[1:]) <= 0.0:
        return set()
    count = max(1, int(np.ceil(float(rate) * nslabs)))
    ranked = sorted(range(1, nslabs + 1), key=lambda n: (-scores[n], n))
    return set(ranked[:count])


def _absolute_slab_scores(indicator_by_slab) -> list[float]:
    r"""Return ``E_n = sum_K abs(eta_Kn)`` from existing indicators.

    The leading zero preserves the one-based slab convention used by the
    adaptive driver.  This aggregation performs no additional estimator,
    adjoint, directional split, or localisation assembly.
    """
    return [0.0] + [
        float(np.abs(np.asarray(values, dtype=float)).sum())
        for values in indicator_by_slab[1:]
    ]


def refine_causal_slab_grid(
    meshes,
    times,
    marked_by_slab,
    *,
    time_marked_fraction: float,
    time_marking_strategy: str = "cell_fraction",
    time_fixed_rate: float = 0.20,
    time_marked_override: set[int] | None = None,
    enable_space_refinement: bool = True,
    enable_time_refinement: bool = True,
) -> GridRefinement:
    r"""Refine marked cells and inherit every earlier marked physical region.

    The causal invariant makes mesh ``T_{n+1}`` a refinement of ``T_n``.  It
    is the safe direction for the AS-to-DG embedding used by the verified
    marked-mesh transfer; no fine-to-coarse velocity trace is introduced.
    """
    nslabs = len(times) - 1
    if len(meshes) != nslabs + 1 or len(marked_by_slab) != nslabs + 1:
        raise ValueError("meshes, times, and marked_by_slab have incompatible lengths.")
    time_marked, _ = _time_marks(
        marked_by_slab,
        time_marked_fraction,
        strategy=time_marking_strategy,
        fixed_rate=time_fixed_rate,
    )
    if time_marked_override is not None:
        time_marked = set(time_marked_override)
    if not enable_time_refinement:
        time_marked = set()
    causal_marks: list[np.ndarray | None] = [None] * (nslabs + 1)
    refined: list[Any | None] = [None] * (nslabs + 1)
    # A mesh interface is a numerical operator, not merely bookkeeping.  If
    # the same parent mesh receives the same cumulative marks on adjacent
    # slabs, constructing two equivalent Netgen meshes would make the slab
    # solver apply P=QI instead of its exact-identity path.  Cache by parent
    # identity and the complete mark mask so equivalent causal grids share
    # the very same MeshGeometry object.
    refinement_cache: dict[tuple[int, tuple[int, ...], bytes], Any] = {}
    for n in range(1, nslabs + 1):
        local = np.asarray(marked_by_slab[n], dtype=bool)
        if n == 1:
            inherited = np.zeros_like(local)
        else:
            inherited = inherit_refinement_marks(
                meshes[n - 1], causal_marks[n - 1], meshes[n]
            )
        causal_marks[n] = np.logical_or(local, inherited)
        mesh = meshes[n]
        if enable_space_refinement and np.any(causal_marks[n]):
            mask = np.ascontiguousarray(causal_marks[n], dtype=np.bool_)
            cache_key = (id(mesh), mask.shape, mask.tobytes())
            cached = refinement_cache.get(cache_key)
            if cached is None:
                DG0 = FunctionSpace(mesh, "DG", 0)
                marker = Function(DG0, name=f"causal_cylinder_marker_slab_{n}")
                marker.dat.data[:] = mask.astype(marker.dat.data.dtype)
                cached = refine_marked_mesh(mesh, marker)
                refinement_cache[cache_key] = cached
            mesh = cached
        refined[n] = mesh

    next_times = [float(times[0])]
    next_meshes: list[Any | None] = [None]
    parent_slab: list[int | None] = [None]
    for n in range(1, nslabs + 1):
        if n in time_marked:
            next_times.append(0.5 * (float(times[n - 1]) + float(times[n])))
            next_meshes.append(refined[n])
            parent_slab.append(n)
        next_times.append(float(times[n]))
        next_meshes.append(refined[n])
        parent_slab.append(n)
    return GridRefinement(
        times=np.asarray(next_times, dtype=float),
        meshes=next_meshes,
        causal_marks=causal_marks,
        time_marked=time_marked,
        parent_slab=parent_slab,
    )


def refine_independent_slab_grid(
    meshes,
    times,
    marked_by_slab,
    *,
    time_marked_fraction: float,
    time_marking_strategy: str = "cell_fraction",
    time_fixed_rate: float = 0.20,
    time_marked_override: set[int] | None = None,
    enable_space_refinement: bool = True,
    enable_time_refinement: bool = True,
) -> GridRefinement:
    r"""Refine every time slab only with that slab's own DWR marks.

    No spatial marks are propagated forward in physical time.  Adjacent slab
    meshes may consequently be unrelated sibling refinements, or the target
    may be coarser than the source.  Their dG velocity trace is coupled by the
    arbitrary-mesh mixed-mass/supermesh transfer in ``slabwise.py``.
    """
    nslabs = len(times) - 1
    if len(meshes) != nslabs + 1 or len(marked_by_slab) != nslabs + 1:
        raise ValueError("meshes, times, and marked_by_slab have incompatible lengths.")
    time_marked, _ = _time_marks(
        marked_by_slab,
        time_marked_fraction,
        strategy=time_marking_strategy,
        fixed_rate=time_fixed_rate,
    )
    if time_marked_override is not None:
        time_marked = set(time_marked_override)
    if not enable_time_refinement:
        time_marked = set()

    local_marks: list[np.ndarray | None] = [None] * (nslabs + 1)
    refined: list[Any | None] = [None] * (nslabs + 1)
    refinement_cache: dict[tuple[int, tuple[int, ...], bytes], Any] = {}
    for n in range(1, nslabs + 1):
        local = np.ascontiguousarray(marked_by_slab[n], dtype=np.bool_)
        local_marks[n] = local.copy()
        mesh = meshes[n]
        if enable_space_refinement and np.any(local):
            cache_key = (id(mesh), local.shape, local.tobytes())
            cached = refinement_cache.get(cache_key)
            if cached is None:
                dg0 = FunctionSpace(mesh, "DG", 0)
                marker = Function(dg0, name=f"independent_marker_slab_{n}")
                marker.dat.data[:] = local.astype(marker.dat.data.dtype)
                cached = refine_marked_mesh(mesh, marker)
                refinement_cache[cache_key] = cached
            mesh = cached
        refined[n] = mesh

    next_times = [float(times[0])]
    next_meshes: list[Any | None] = [None]
    parent_slab: list[int | None] = [None]
    for n in range(1, nslabs + 1):
        if n in time_marked:
            next_times.append(0.5 * (float(times[n - 1]) + float(times[n])))
            next_meshes.append(refined[n])
            parent_slab.append(n)
        next_times.append(float(times[n]))
        next_meshes.append(refined[n])
        parent_slab.append(n)
    return GridRefinement(
        times=np.asarray(next_times, dtype=float),
        meshes=next_meshes,
        causal_marks=local_marks,
        time_marked=time_marked,
        parent_slab=parent_slab,
    )


def refine_common_slab_grid(
    meshes,
    times,
    marked_by_slab,
    *,
    time_marked_fraction: float,
    time_marking_strategy: str = "cell_fraction",
    time_fixed_rate: float = 0.20,
    time_marked_override: set[int] | None = None,
    enable_space_refinement: bool = True,
    enable_time_refinement: bool = True,
) -> GridRefinement:
    r"""Apply the space-time marked-cell union to one common spatial mesh.

    This keeps the global Dörfler selection and slabwise time marking, but it
    deliberately removes every within-iteration cross-mesh primal trace.  It
    is the robust fallback when a non-nested AS transfer is too perturbative
    for long-time nonlinear dynamics.
    """
    nslabs = len(times) - 1
    if len(meshes) != nslabs + 1 or len(marked_by_slab) != nslabs + 1:
        raise ValueError("meshes, times, and marked_by_slab have incompatible lengths.")
    base = meshes[1]
    if any(meshes[n] is not base for n in range(2, nslabs + 1)):
        raise ValueError(
            "Common spatial refinement requires one shared parent mesh on all slabs."
        )
    union = np.zeros_like(np.asarray(marked_by_slab[1], dtype=bool))
    for n in range(1, nslabs + 1):
        local = np.asarray(marked_by_slab[n], dtype=bool)
        if local.shape != union.shape:
            raise ValueError("Common-grid slab marks have incompatible shapes.")
        union |= local
    refined_mesh = base
    if enable_space_refinement and np.any(union):
        DG0 = FunctionSpace(base, "DG", 0)
        marker = Function(DG0, name="common_cylinder_marker")
        marker.dat.data[:] = union.astype(marker.dat.data.dtype)
        refined_mesh = refine_marked_mesh(base, marker)

    time_marked, _ = _time_marks(
        marked_by_slab,
        time_marked_fraction,
        strategy=time_marking_strategy,
        fixed_rate=time_fixed_rate,
    )
    if time_marked_override is not None:
        time_marked = set(time_marked_override)
    if not enable_time_refinement:
        time_marked = set()
    next_times = [float(times[0])]
    next_meshes: list[Any | None] = [None]
    parent_slab: list[int | None] = [None]
    for n in range(1, nslabs + 1):
        if n in time_marked:
            next_times.append(0.5 * (float(times[n - 1]) + float(times[n])))
            next_meshes.append(refined_mesh)
            parent_slab.append(n)
        next_times.append(float(times[n]))
        next_meshes.append(refined_mesh)
        parent_slab.append(n)
    return GridRefinement(
        times=np.asarray(next_times, dtype=float),
        meshes=next_meshes,
        causal_marks=[None] + [union.copy() for _ in range(nslabs)],
        time_marked=time_marked,
        parent_slab=parent_slab,
    )


class CylinderAdaptiveSolver:
    """Problem-specific SOLVE -> ESTIMATE -> MARK -> REFINE driver."""

    def __init__(
        self,
        config: CylinderAdaptiveConfig,
        *,
        iteration_callback=None,
        grid_callback=None,
        initial_times=None,
        initial_meshes=None,
        initial_history=None,
        start_iteration: int = 0,
        hierarchy=None,
        labels=None,
        refinement_lineage=None,
    ):
        self.config = config
        self.iteration_callback = iteration_callback
        self.grid_callback = grid_callback
        if hierarchy is None or labels is None:
            hierarchy, inlet, wall, cylinder, outlet = _make_hierarchy(
                int(config.hierarchy_levels),
                geometry_degree=int(config.geometry_degree),
            )
            labels = {
                "inlet": inlet,
                "wall": wall,
                "cylinder": cylinder,
                "outlet": outlet,
            }
        self.hierarchy = hierarchy
        self.labels = dict(labels)
        if initial_times is None or initial_meshes is None:
            base = hierarchy[-1]
            self.times = np.linspace(
                0.0, float(config.final_time), int(config.initial_time_slabs) + 1
            )
            self.meshes: list[Any | None] = [None] + [
                base for _ in range(int(config.initial_time_slabs))
            ]
        else:
            self.times = np.asarray(initial_times, dtype=float).copy()
            self.meshes = list(initial_meshes)
            if len(self.meshes) != len(self.times):
                raise ValueError("initial_times and initial_meshes have incompatible lengths.")
            if self.meshes[0] is not None:
                raise ValueError("The initial trace mesh entry must be None.")
            if not np.all(np.diff(self.times) > 0.0):
                raise ValueError("Checkpoint times must be strictly increasing.")
            if abs(float(self.times[-1]) - float(config.final_time)) > 1.0e-12:
                raise ValueError("Checkpoint final time does not match the configuration.")
        self.history = [dict(row) for row in (initial_history or [])]
        self.start_iteration = int(start_iteration)
        if self.start_iteration < 0:
            raise ValueError("start_iteration must be nonnegative.")
        self.refinement_lineage = list(refinement_lineage or [])
        self.last: dict[str, Any] = {}

    def _solve_estimate(self):
        transfers = build_slab_transfers(
            self.meshes,
            self.labels,
            mode=self.config.interface_transfer_mode,
            low_family=self.config.primal_space_family,
            enriched_velocity_degree=self.config.enriched_velocity_degree,
            enriched_pressure_degree=self.config.enriched_pressure_degree,
        )
        primal = solve_slabwise_primal(
            self.meshes,
            self.times,
            transfers,
            self.labels,
            viscosity=self.config.viscosity,
            time_degree=self.config.primal_time_degree,
            report_every=self.config.report_every,
        )
        primal_only = self.config.estimator_strategy == "primal_only"
        if primal_only:
            # Strict linear DWR, matching Firedrake's steady goal-adaptive
            # primal-residual path: A'(u_h)^* z+ = J'(u_h) is solved in the
            # enriched dual space.  There is no enriched-primal solve and no
            # low-adjoint solve for the interpolant-weight mode.
            primal_enriched = None
            if self.config.enriched_time_refinement_factor == 1:
                dual_enriched = solve_slabwise_adjoint(
                    primal,
                    transfers,
                    time_degree=self.config.enriched_time_degree,
                    spatially_enriched=True,
                    report_every=self.config.report_every,
                )
            else:
                PETSc.Sys.Print(
                    "[CYLINDER STRICT LINEAR] solving the enriched adjoint "
                    f"on {self.config.enriched_time_refinement_factor} "
                    "uniform temporal children per low slab."
                )
                dual_enriched = solve_temporally_refined_enriched_adjoint(
                    primal,
                    refinement_factor=(
                        self.config.enriched_time_refinement_factor
                    ),
                    time_degree=self.config.enriched_time_degree,
                    interface_transfer_mode=(
                        self.config.interface_transfer_mode
                    ),
                    low_family=self.config.primal_space_family,
                    enriched_velocity_degree=(
                        self.config.enriched_velocity_degree
                    ),
                    enriched_pressure_degree=(
                        self.config.enriched_pressure_degree
                    ),
                    report_every=self.config.report_every,
                )
            if self.config.dual_weight_mode == "enriched_minus_numerical":
                dual_low = solve_slabwise_adjoint(
                    primal,
                    transfers,
                    time_degree=self.config.primal_time_degree,
                    spatially_enriched=False,
                    report_every=self.config.report_every,
                )
            else:
                PETSc.Sys.Print(
                    "[CYLINDER STRICT LINEAR] constructing P2/P1-dG1 "
                    "nodal interpolant of the enriched adjoint."
                )
                dual_low = interpolate_enriched_adjoint_to_low(
                    dual_enriched,
                    transfers,
                    primal["labels"],
                    time_degree=self.config.primal_time_degree,
                )
            PETSc.Sys.Print(
                "[CYLINDER STRICT LINEAR] assembling global primal-residual "
                "estimator."
            )
            estimate = estimate_linear_dwr_global(
                primal,
                dual_enriched,
                dual_low,
                quadrature_points=self.config.quadrature_points,
                dual_weight_mode=self.config.dual_weight_mode,
            )
            if self.config.uniform_refinement:
                PETSc.Sys.Print(
                    "[CYLINDER STRICT LINEAR] uniform grid: skipping local "
                    "indicators and retaining only the global estimator."
                )
                localisation = uniform_global_only_localisation(
                    primal, estimate
                )
            else:
                PETSc.Sys.Print(
                    "[CYLINDER STRICT LINEAR] localising primal residual into "
                    "volume, spatial-facet, and temporal-jump contributions; "
                    "no space-time ridge or auxiliary recovery solve is used."
                )
                localisation = localise_linear_dwr(
                    primal,
                    dual_enriched,
                    dual_low,
                    estimate,
                    quadrature_points=self.config.quadrature_points,
                    dual_weight_mode=self.config.dual_weight_mode,
                    primal_recovery_degree=self.config.recovery_space_degree,
                    facet_recovery_degree=self.config.facet_recovery_degree,
                    recovery_time_degree=self.config.recovery_time_degree,
                )
            directional_split = None
            if self.config.directional_split_diagnostic:
                PETSc.Sys.Print(
                    "[CYLINDER DIRECTIONAL SPLIT] separately assembling "
                    "R5-style temporal and spatial dual-weight components; "
                    "production marking remains unchanged."
                )
                directional_split = compute_linear_directional_split(
                    primal,
                    dual_enriched,
                    dual_low,
                    estimate,
                    localisation,
                    primal_time_degree=self.config.primal_time_degree,
                    quadrature_points=self.config.quadrature_points,
                    primal_recovery_degree=(
                        self.config.recovery_space_degree
                    ),
                    facet_recovery_degree=(
                        self.config.facet_recovery_degree
                    ),
                    recovery_time_degree=self.config.recovery_time_degree,
                )
                PETSc.Sys.Print(
                    "[CYLINDER DIRECTIONAL SPLIT] "
                    f"eta_time={directional_split.eta_time:.16e}; "
                    f"eta_space={directional_split.eta_space:.16e}; "
                    f"sum={directional_split.eta_sum:.16e}; "
                    f"global_gap={directional_split.global_gap:.3e}; "
                    f"cell_gap_linf={directional_split.cell_gap_linf:.3e}."
                )
            PETSc.Sys.Print(
                "[CYLINDER STRICT LINEAR] estimator/localisation complete."
            )
            self.last = {
                "transfers": transfers,
                "primal": primal,
                "primal_enriched": None,
                "dual_low": dual_low,
                "dual_enriched": dual_enriched,
                "estimate": estimate,
                "stationarity_identity_scale": float("nan"),
                "stationarity_relative": {"primal": 0.0, "adjoint": 0.0},
                "adjoint_linearisation": "low_primal",
                "dual_weight_construction": dual_low.get(
                    "construction", "independent_low_adjoint_solve"
                ),
                "directional_split": directional_split,
            }
            return (
                transfers,
                primal,
                None,
                dual_low,
                dual_enriched,
                estimate,
                localisation,
            )

        primal_enriched = solve_slabwise_enriched_primal(
            primal,
            transfers,
            time_degree=self.config.enriched_time_degree,
            report_every=self.config.report_every,
        )
        dual_enriched = solve_slabwise_adjoint(
            primal_enriched,
            transfers,
            time_degree=self.config.enriched_time_degree,
            spatially_enriched=True,
            report_every=self.config.report_every,
        )
        # The interpolant-weighted linear estimator uses
        # Z^+ - Pi_h Z^+ and therefore does not require an independently
        # solved low-order adjoint.  Retain the numerical low adjoint only
        # for modes that actually use Z_h.
        if self.config.dual_weight_mode == "enriched_minus_numerical":
            dual_low = solve_slabwise_adjoint(
                primal,
                transfers,
                time_degree=self.config.primal_time_degree,
                spatially_enriched=False,
                report_every=self.config.report_every,
            )
        else:
            dual_low = project_enriched_adjoint_to_low(
                dual_enriched, transfers, primal["labels"]
            )
        estimate = estimate_symmetric_dwr(
            primal,
            primal_enriched,
            dual_enriched,
            dual_low,
            quadrature_points=self.config.quadrature_points,
            dual_weight_mode=self.config.dual_weight_mode,
            include_cubic_remainder=self.config.include_cubic_remainder,
        )
        if self.config.estimator_strategy == "symmetric_two_term":
            zero_cells = [None] + [
                np.zeros_like(estimate.eta_correction_cell_weak[n])
                for n in range(1, len(estimate.eta_correction_cell_weak))
            ]
            pair_cells = [None] + [
                0.5 * estimate.eta_primal_cell_weak[n]
                + 0.5 * estimate.eta_adjoint_cell_weak[n]
                + estimate.eta_cubic_remainder_cell_weak[n]
                for n in range(1, len(estimate.eta_primal_cell_weak))
            ]
            pair_core = (
                0.5 * estimate.eta_primal_residual
                + 0.5 * estimate.eta_adjoint_residual
            )
            pair_global = pair_core + estimate.eta_cubic_remainder
            estimate = replace(
                estimate,
                eta_global=pair_global,
                eta_galerkin_correction=0.0,
                eta_symmetric_core=pair_core,
                eta_cell_weak=pair_cells,
                eta_correction_cell_weak=zero_cells,
                weak_closure_gap=(
                    float(sum(values.sum() for values in pair_cells[1:]))
                    - pair_global
                ),
                observed_cubic_remainder=(
                    estimate.enriched_goal_difference - pair_core
                ),
                enriched_identity_gap=(
                    estimate.enriched_goal_difference - pair_global
                ),
            )
        if primal_only:
            zero_cells = [None] + [
                np.zeros_like(estimate.eta_primal_cell_weak[n])
                for n in range(1, len(estimate.eta_primal_cell_weak))
            ]
            estimate = replace(
                estimate,
                eta_global=estimate.eta_primal_residual,
                eta_adjoint_residual=0.0,
                eta_galerkin_correction=0.0,
                eta_symmetric_core=estimate.eta_primal_residual,
                eta_cell_weak=estimate.eta_primal_cell_weak,
                eta_adjoint_cell_weak=zero_cells,
                eta_correction_cell_weak=zero_cells,
                # Keep the measured enriched-minus-low QoI difference as a
                # diagnostic, although it is not part of the primal-only
                # estimator identity.
                enriched_primal_stationarity_defect=0.0,
                enriched_adjoint_stationarity_defect=0.0,
                enriched_primal_stationarity_volume_by_slab=[0.0] * len(self.times),
                enriched_primal_stationarity_jump_by_slab=[0.0] * len(self.times),
                enriched_adjoint_stationarity_volume_by_slab=[0.0] * len(self.times),
                enriched_adjoint_stationarity_jump_by_slab=[0.0] * len(self.times),
            )
        identity_scale = max(abs(estimate.enriched_goal_difference), 1.0e-14)
        identity_relative_gap = abs(estimate.enriched_identity_gap) / identity_scale
        primal_stationarity_relative = (
            abs(estimate.enriched_primal_stationarity_defect) / identity_scale
        )
        adjoint_stationarity_relative = (
            abs(estimate.enriched_adjoint_stationarity_defect) / identity_scale
        )
        self.last = {
            "transfers": transfers,
            "primal": primal,
            "primal_enriched": primal_enriched,
            "dual_low": dual_low,
            "dual_enriched": dual_enriched,
            "estimate": estimate,
            "stationarity_identity_scale": identity_scale,
            "stationarity_relative": {
                "primal": primal_stationarity_relative,
                "adjoint": adjoint_stationarity_relative,
            },
        }
        stationarity_diagnostic_passed = primal_only or max(
            primal_stationarity_relative, adjoint_stationarity_relative
        ) <= self.config.dwr_stationarity_relative_tolerance
        if not stationarity_diagnostic_passed:
            # Stationarity is an implementation diagnostic, not an acceptance
            # condition in the cited nonlinear-DWR adaptive algorithm.  Keep
            # the values visible in the history, but do not suppress an
            # otherwise computable estimator and localisation.
            PETSc.Sys.Print(
                "[CYLINDER STATIONARITY WARNING] "
                f"primal={primal_stationarity_relative:.6e}, "
                f"adjoint={adjoint_stationarity_relative:.6e}, diagnostic="
                f"{self.config.dwr_stationarity_relative_tolerance:.6e}."
            )
        if (not primal_only) and identity_relative_gap > self.config.dwr_identity_relative_tolerance:
            # With the optional cubic remainder disabled, agreement with the
            # enriched goal difference is a diagnostic, not a paper-defined
            # acceptance condition.  Report the mismatch and continue so the
            # three retained DWR terms can still be localised and inspected.
            PETSc.Sys.Print(
                "[CYLINDER DWR IDENTITY WARNING] "
                f"relative gap={identity_relative_gap:.6e}, allowed="
                f"{self.config.dwr_identity_relative_tolerance:.6e}; "
                f"Jplus-Jh={estimate.enriched_goal_difference:.16e}, "
                f"eta={estimate.eta_global:.16e}, "
                f"rich primal defect="
                f"{estimate.enriched_primal_stationarity_defect:.16e}, "
                f"rich adjoint defect="
                f"{estimate.enriched_adjoint_stationarity_defect:.16e}."
            )
        if self.config.estimator_strategy == "symmetric_two_term":
            PETSc.Sys.Print(
                "[CYLINDER NONLINEAR TWO-TERM] localising primal and adjoint "
                "residuals into volume, spatial-facet, and temporal-jump "
                "contributions; the optional cubic remainder is volume-only."
            )
            localisation = localise_symmetric_two_term_dwr(
                primal,
                primal_enriched,
                dual_enriched,
                dual_low,
                estimate,
                quadrature_points=self.config.quadrature_points,
                dual_weight_mode=self.config.dual_weight_mode,
            )
        else:
            localisation = localise_symmetric_dwr(
                primal,
                primal_enriched,
                dual_enriched,
                dual_low,
                estimate,
                quadrature_points=self.config.quadrature_points,
                dual_weight_mode=self.config.dual_weight_mode,
                primal_recovery_degree=self.config.recovery_space_degree,
                adjoint_recovery_degree=self.config.recovery_space_degree,
                facet_recovery_degree=self.config.facet_recovery_degree,
                recovery_time_degree=self.config.recovery_time_degree,
            )
        if primal_only:
            signed = localisation.eta_primal_bubble_cell
            local_sum = float(sum(values.sum() for values in signed[1:]))
            localisation = replace(
                localisation,
                eta_cell_signed=signed,
                eta_local_sum=local_sum,
                eta_marking_sum=float(sum(np.abs(values).sum() for values in signed[1:])),
                localisation_gap=local_sum - estimate.eta_global,
                eta_adjoint_recovered=0.0,
                eta_correction_recovered=0.0,
                adjoint_recovery_gap=0.0,
                correction_recovery_gap=0.0,
            )
        elif self.config.estimator_strategy == "symmetric_two_term":
            signed = [None] + [
                0.5 * localisation.eta_primal_bubble_cell[n]
                + 0.5 * localisation.eta_adjoint_bubble_cell[n]
                + localisation.eta_cubic_remainder_cell[n]
                for n in range(1, len(self.times))
            ]
            local_sum = float(sum(values.sum() for values in signed[1:]))
            localisation = replace(
                localisation,
                eta_cell_signed=signed,
                eta_local_sum=local_sum,
                eta_marking_sum=float(
                    sum(np.abs(values).sum() for values in signed[1:])
                ),
                localisation_gap=local_sum - estimate.eta_global,
                eta_correction_recovered=0.0,
                correction_recovery_gap=0.0,
            )
        return (
            transfers,
            primal,
            primal_enriched,
            dual_low,
            dual_enriched,
            estimate,
            localisation,
        )

    def solve(self) -> CylinderAdaptiveResult:
        PETSc.Sys.Print("[CYLINDER ADAPTIVE CONFIG] " + str(asdict(self.config)))
        if self.start_iteration >= int(self.config.max_iterations):
            raise ValueError(
                "The checkpoint grid iteration is not smaller than max_iterations; "
                "increase --max-it to continue."
            )
        if self.grid_callback is not None:
            self.grid_callback(self, self.start_iteration)
        final_objects = None
        for iteration in range(
            self.start_iteration, int(self.config.max_iterations)
        ):
            iteration_started = perf_counter()
            objects = self._solve_estimate()
            (
                transfers,
                primal,
                primal_enriched,
                dual_low,
                dual_enriched,
                estimate,
                localisation,
            ) = objects
            directional_split = self.last.get("directional_split")
            final_objects = objects
            recovery_relative = abs(localisation.localisation_gap) / max(
                abs(estimate.eta_global), np.finfo(float).eps
            )
            recovery_gate_passed = recovery_relative <= float(
                self.config.maximum_recovery_gap_relative
            )
            goal = mean_drag_from_history(
                primal, quadrature_points=self.config.quadrature_points
            )
            primal_drag_audit = drag_history_diagnostics(
                primal, quadrature_points=self.config.quadrature_points
            )
            if primal_enriched is None:
                enriched_drag_audit = {
                    key: None for key in primal_drag_audit
                }
            else:
                enriched_drag_audit = drag_history_diagnostics(
                    primal_enriched,
                    quadrature_points=self.config.quadrature_points,
                )
            reference = self.config.reference_mean_drag
            true_error = None if reference is None else float(reference - goal)
            enriched_goal_difference = float(estimate.enriched_goal_difference)
            enriched_goal = goal + enriched_goal_difference
            enriched_identity_ratio = (
                None
                if not np.isfinite(enriched_goal_difference)
                or abs(enriched_goal_difference) <= 1.0e-15
                else float(estimate.eta_global) / enriched_goal_difference
            )
            saturation_ratio = (
                None
                if true_error is None
                or abs(true_error) <= 1.0e-15
                or not np.isfinite(enriched_goal)
                else (float(reference) - enriched_goal) / true_error
            )
            effectivity_denominator = (
                None
                if true_error is None or abs(true_error) <= 1.0e-15
                else true_error
            )
            stationarity_relative = self.last["stationarity_relative"]
            primal_stationarity_relative = float(
                stationarity_relative["primal"]
            )
            adjoint_stationarity_relative = float(
                stationarity_relative["adjoint"]
            )
            stationarity_diagnostic_passed = (
                self.config.estimator_strategy == "primal_only"
                or max(
                    primal_stationarity_relative,
                    adjoint_stationarity_relative,
                ) <= self.config.dwr_stationarity_relative_tolerance
            )
            estimator_threshold = (
                abs(estimate.eta_global) <= float(self.config.tolerance)
            )
            true_goal_threshold = (
                self.config.true_goal_error_tolerance is not None
                and true_error is not None
                and abs(true_error)
                <= float(self.config.true_goal_error_tolerance)
            )
            threshold = estimator_threshold or true_goal_threshold
            final_iteration = iteration == int(self.config.max_iterations) - 1
            stop = threshold or final_iteration or not recovery_gate_passed
            # Even on the configured final iteration, retain the proposed
            # Dörfler set.  It is diagnostic output and permits a later
            # checkpoint continuation without recomputing this full DWR
            # cycle.  A failed recovery gate or a reached tolerance still
            # suppresses refinement proposals.
            marking_signed = localisation.eta_cell_signed
            if (
                self.config.estimator_strategy == "symmetric_three_term"
                and self.config.three_term_marking_strategy == "residual_pair"
            ):
                marking_signed = [None] + [
                    0.5 * localisation.eta_primal_bubble_cell[n]
                    + 0.5 * localisation.eta_adjoint_bubble_cell[n]
                    for n in range(1, len(self.times))
                ]
            marking_activity = float(sum(
                np.abs(marking_signed[n]).sum()
                for n in range(1, len(marking_signed))
            ))
            patch_activity = None
            patches_total = None
            patches_marked = None
            patch_signed = None
            patch_marked = None
            if threshold or not recovery_gate_passed:
                marked = [None] + [
                    np.zeros(
                        FunctionSpace(self.meshes[n], "DG", 0).node_count,
                        dtype=bool,
                    )
                    for n in range(1, len(self.times))
                ]
            elif self.config.uniform_refinement:
                marked = [None] + [
                    np.ones(
                        FunctionSpace(self.meshes[n], "DG", 0).node_count,
                        dtype=bool,
                    )
                    for n in range(1, len(self.times))
                ]
            else:
                if self.config.space_marking_strategy == "vertex_patch":
                    marked, patch_signed, patch_marked = (
                        mark_spacetime_vertex_patches(
                            marking_signed, self.meshes, self.config.theta
                        )
                    )
                    patch_activity = float(sum(
                        np.abs(patch_signed[n]).sum()
                        for n in range(1, len(patch_signed))
                    ))
                    patches_total = int(sum(
                        patch_signed[n].size
                        for n in range(1, len(patch_signed))
                    ))
                    patches_marked = int(sum(
                        np.count_nonzero(patch_marked[n])
                        for n in range(1, len(patch_marked))
                    ))
                else:
                    marked = mark_spacetime_cells(
                        marking_signed, self.config.theta
                    )
            _, fractions = _time_marks(
                marked,
                self.config.time_marked_fraction,
                strategy="cell_fraction",
                fixed_rate=self.config.time_fixed_rate,
            )
            time_scores = _absolute_slab_scores(marking_signed)
            if self.config.time_score_source == "directional_time":
                # Temporal refinement is decided by the localised temporal
                # component rho(U_h)(z+ - I_k z+) of the R5 split, not by
                # the combined space-time indicator: the combined weight is
                # dominated by its spatial part, which is exactly why the
                # cell_fraction trigger never fires while the true error
                # stagnates on the temporal floor.
                if directional_split is None:
                    raise RuntimeError(
                        "time_score_source='directional_time' needs the "
                        "directional split of this iteration."
                    )
                directional_cells = (
                    directional_split.time_localisation.eta_cell_signed
                )
                time_scores = [0.0] + [
                    float(np.abs(directional_cells[n]).sum())
                    for n in range(1, len(directional_cells))
                ]
                if self.config.time_marking_strategy == "slab_bulk_capped":
                    time_marked, time_scores = _slab_bulk_capped_marks(
                        directional_cells,
                        bulk_fraction=self.config.time_bulk_fraction,
                        max_fraction=self.config.time_max_fraction,
                        max_count=self.config.time_max_count,
                    )
                else:
                    time_marked = _fixed_rate_score_marks(
                        time_scores, self.config.time_fixed_rate
                    )
            elif self.config.time_score_source == "combined_indicator":
                # Reuse the production cell indicators exactly:
                # E_n = sum_K |eta_{K,n}|.  This path deliberately avoids
                # the optional directional split and its extra assemblies.
                if self.config.time_marking_strategy == "slab_bulk_capped":
                    time_marked, time_scores = _slab_bulk_capped_marks(
                        marking_signed,
                        bulk_fraction=self.config.time_bulk_fraction,
                        max_fraction=self.config.time_max_fraction,
                        max_count=self.config.time_max_count,
                    )
                else:
                    time_marked = _fixed_rate_score_marks(
                        time_scores, self.config.time_fixed_rate
                    )
            elif self.config.time_marking_strategy == "slab_bulk_capped":
                time_marked, time_scores = _slab_bulk_capped_marks(
                    marking_signed,
                    bulk_fraction=self.config.time_bulk_fraction,
                    max_fraction=self.config.time_max_fraction,
                    max_count=self.config.time_max_count,
                )
            elif (
                self.config.time_marking_strategy
                == "cell_fraction_capped"
            ):
                time_marked = _cell_fraction_capped_marks(
                    fractions,
                    threshold=self.config.time_marked_fraction,
                    max_fraction=self.config.time_max_fraction,
                    max_count=self.config.time_max_count,
                )
            else:
                time_marked, _ = _time_marks(
                    marked,
                    self.config.time_marked_fraction,
                    strategy=self.config.time_marking_strategy,
                    fixed_rate=self.config.time_fixed_rate,
                )
            if not self.config.enable_time_refinement:
                time_marked = set()
            row = {
                "iteration": iteration,
                "goal_functional": self.config.goal_functional,
                "mean_drag": goal,
                "corrected_mean_drag": goal + estimate.eta_global,
                "enriched_mean_drag": enriched_goal,
                "enriched_primal_solved": primal_enriched is not None,
                "adjoint_linearisation": self.last.get(
                    "adjoint_linearisation", "enriched_primal"
                ),
                "dual_weight_construction": self.last.get(
                    "dual_weight_construction",
                    dual_low.get("construction", "independent_low_adjoint_solve"),
                ),
                **{
                    f"primal_{key}": value
                    for key, value in primal_drag_audit.items()
                },
                **{
                    f"enriched_{key}": value
                    for key, value in enriched_drag_audit.items()
                },
                "reference_mean_drag": reference,
                "true_goal_error": true_error,
                "effectivity_global": (
                    None
                    if effectivity_denominator is None
                    else estimate.eta_global / effectivity_denominator
                ),
                "effectivity_global_absolute": (
                    None
                    if effectivity_denominator is None
                    else abs(estimate.eta_global / effectivity_denominator)
                ),
                "effectivity_localised": (
                    None
                    if effectivity_denominator is None
                    else localisation.eta_local_sum / effectivity_denominator
                ),
                "effectivity_localised_absolute": (
                    None
                    if effectivity_denominator is None
                    else abs(
                        localisation.eta_local_sum / effectivity_denominator
                    )
                ),
                "eta_global": estimate.eta_global,
                "enriched_goal_difference": enriched_goal_difference,
                "enriched_identity_ratio": enriched_identity_ratio,
                "saturation_ratio": saturation_ratio,
                "eta_symmetric_core": estimate.eta_symmetric_core,
                "eta_cubic_remainder": estimate.eta_cubic_remainder,
                "enriched_identity_gap": estimate.enriched_identity_gap,
                "enriched_primal_stationarity_defect": (
                    estimate.enriched_primal_stationarity_defect
                ),
                "enriched_adjoint_stationarity_defect": (
                    estimate.enriched_adjoint_stationarity_defect
                ),
                "primal_stationarity_relative": primal_stationarity_relative,
                "adjoint_stationarity_relative": adjoint_stationarity_relative,
                "stationarity_diagnostic_passed": (
                    stationarity_diagnostic_passed
                ),
                "eta_primal_residual": estimate.eta_primal_residual,
                "eta_adjoint_residual": estimate.eta_adjoint_residual,
                "eta_galerkin_correction": estimate.eta_galerkin_correction,
                "eta_local_sum": localisation.eta_local_sum,
                "eta_marking_sum": marking_activity,
                "space_marking_strategy": self.config.space_marking_strategy,
                "patch_marking_sum": patch_activity,
                "n_spacetime_patches": patches_total,
                "n_spacetime_patches_marked": patches_marked,
                "localisation_strategy": (
                    "uniform_global_only"
                    if self.config.uniform_refinement
                    and self.config.estimator_strategy == "primal_only"
                    else "three_part_strong_residual"
                    if self.config.estimator_strategy == "primal_only"
                    else "three_part_strong_residual_plus_cubic_volume"
                    if self.config.estimator_strategy == "symmetric_two_term"
                    and self.config.include_cubic_remainder
                    else "three_part_strong_residual"
                    if self.config.estimator_strategy == "symmetric_two_term"
                    else "tensor_bubble_cone_recovery"
                ),
                "localisation_entities": (
                    "none"
                    if self.config.uniform_refinement
                    and self.config.estimator_strategy == "primal_only"
                    else "volume+spatial_facet+temporal_jump"
                    if self.config.estimator_strategy in {
                        "primal_only", "symmetric_two_term"
                    }
                    and not self.config.include_cubic_remainder
                    else "volume+spatial_facet+temporal_jump+cubic_volume"
                    if self.config.estimator_strategy == "symmetric_two_term"
                    else "volume+spatial_facet+temporal_facet+spacetime_ridge"
                ),
                "localisation_auxiliary_solves_per_slab": (
                    0
                    if self.config.estimator_strategy in {
                        "primal_only", "symmetric_two_term"
                    }
                    else 9
                ),
                "eta_primal_volume_local_sum": float(sum(
                    values.sum()
                    for values in localisation.eta_primal_volume_cell[1:]
                )),
                "eta_primal_spatial_facet_local_sum": float(sum(
                    values.sum()
                    for values in localisation.eta_primal_spatial_cell[1:]
                )),
                "eta_primal_temporal_jump_local_sum": float(sum(
                    values.sum()
                    for values in localisation.eta_primal_temporal_cell[1:]
                )),
                "eta_primal_spacetime_ridge_local_sum": float(sum(
                    values.sum()
                    for values in localisation.eta_primal_mixed_ridge_cell[1:]
                )),
                "three_term_marking_strategy": (
                    self.config.three_term_marking_strategy
                ),
                "dual_base_strategy": self.config.dual_base_strategy,
                "enriched_time_refinement_factor": (
                    self.config.enriched_time_refinement_factor
                ),
                "localisation_gap_relative": recovery_relative,
                "recovery_gate_passed": recovery_gate_passed,
                "weak_closure_gap": estimate.weak_closure_gap,
                "adjoint_reverse_identity_gap": (
                    localisation.adjoint_reverse_identity_gap
                ),
                "n_time_slabs": len(self.times) - 1,
                "minimum_dt": float(np.min(np.diff(self.times))),
                "maximum_dt": float(np.max(np.diff(self.times))),
                "primal_spacetime_dofs": int(sum(
                    transfers.low_spaces[n][2].dim()
                    * (int(self.config.primal_time_degree) + 1)
                    for n in range(1, len(self.times))
                )),
                "enriched_spacetime_dofs": int(sum(
                    transfers.rich_spaces[n][2].dim()
                    * (int(self.config.enriched_time_degree) + 1)
                    for n in range(1, len(self.times))
                )),
                "adjoint_spacetime_dofs": int(sum(
                    transfers.rich_spaces[n][2].dim()
                    * (int(self.config.enriched_time_degree) + 1)
                    for n in range(1, len(self.times))
                ))
                if self.config.enriched_time_refinement_factor == 1
                else int(dual_enriched["fine_spacetime_dofs"]),
                "n_cells_total": int(sum(
                    FunctionSpace(self.meshes[n], "DG", 0).node_count
                    for n in range(1, len(self.times))
                )),
                "n_spacetime_marked": int(sum(
                    np.count_nonzero(marked[n])
                    for n in range(1, len(marked))
                )),
                "n_time_slabs_marked": len(time_marked),
                "marked_fraction_by_slab": fractions[1:],
                "time_score_source": self.config.time_score_source,
                "time_score_by_slab": time_scores[1:],
                "threshold_reached": threshold,
                "estimator_threshold_reached": estimator_threshold,
                "true_goal_error_threshold_reached": true_goal_threshold,
                "iteration_wall_seconds": perf_counter() - iteration_started,
            }
            if directional_split is not None:
                row.update({
                    "directional_split_enabled": True,
                    "corrected_mean_drag_directional": (
                        goal + directional_split.eta_sum
                    ),
                    "effectivity_directional": (
                        None
                        if effectivity_denominator is None
                        else directional_split.eta_sum
                        / effectivity_denominator
                    ),
                    "eta_directional_time": directional_split.eta_time,
                    "eta_directional_space": directional_split.eta_space,
                    "eta_directional_sum": directional_split.eta_sum,
                    "eta_directional_global_gap": (
                        directional_split.global_gap
                    ),
                    "eta_directional_global_gap_relative": (
                        directional_split.global_gap_relative
                    ),
                    "eta_directional_local_time_sum": (
                        directional_split.local_time_sum
                    ),
                    "eta_directional_local_space_sum": (
                        directional_split.local_space_sum
                    ),
                    "eta_directional_local_sum": directional_split.local_sum,
                    "eta_directional_local_gap": directional_split.local_gap,
                    "eta_directional_local_gap_relative": (
                        directional_split.local_gap_relative
                    ),
                    "eta_directional_cell_gap_linf": (
                        directional_split.cell_gap_linf
                    ),
                    "eta_directional_cell_gap_l1": (
                        directional_split.cell_gap_l1
                    ),
                    "eta_directional_time_by_slab": (
                        directional_split.eta_time_by_slab[1:]
                    ),
                    "eta_directional_space_by_slab": (
                        directional_split.eta_space_by_slab[1:]
                    ),
                })
            self.history.append(row)
            PETSc.Sys.Print(f"[CYLINDER ADAPTIVE {iteration}] {row}")
            self.last = {
                "transfers": transfers,
                "primal": primal,
                "primal_enriched": primal_enriched,
                "dual_low": dual_low,
                "dual_enriched": dual_enriched,
                "estimate": estimate,
                "localisation": localisation,
                "directional_split": directional_split,
                "marking_signed": marking_signed,
                "marked": marked,
                "patch_signed": patch_signed,
                "patch_marked": patch_marked,
            }
            if self.iteration_callback is not None:
                self.iteration_callback(self, row, self.last)
            if not recovery_gate_passed:
                raise RuntimeError(
                    "Adaptive marking was rejected because the tensor recovery "
                    f"gap is {recovery_relative:.3%}; the completed iteration "
                    "diagnostics were retained."
                )
            if stop:
                break
            refinement_function = {
                "common": refine_common_slab_grid,
                "causal": refine_causal_slab_grid,
                "independent": refine_independent_slab_grid,
            }[self.config.space_refinement_mode]
            refinement = refinement_function(
                self.meshes,
                self.times,
                marked,
                time_marked_fraction=self.config.time_marked_fraction,
                time_marking_strategy=self.config.time_marking_strategy,
                time_fixed_rate=self.config.time_fixed_rate,
                time_marked_override=time_marked,
                enable_space_refinement=self.config.enable_space_refinement,
                enable_time_refinement=self.config.enable_time_refinement,
            )
            self.refinement_lineage.append(
                {
                    "source_iteration": int(iteration),
                    "times_before": np.asarray(self.times, dtype=float).copy(),
                    "marked": [
                        None
                        if slab_mark is None
                        else np.asarray(slab_mark, dtype=bool).copy()
                        for slab_mark in marked
                    ],
                    "time_marked_fraction": float(
                        self.config.time_marked_fraction
                    ),
                    "time_marking_strategy": self.config.time_marking_strategy,
                    "time_score_source": self.config.time_score_source,
                    "time_fixed_rate": float(self.config.time_fixed_rate),
                    "time_marked_override": sorted(time_marked),
                    "enable_space_refinement": bool(
                        self.config.enable_space_refinement
                    ),
                    "enable_time_refinement": bool(
                        self.config.enable_time_refinement
                    ),
                    "uniform_refinement": bool(self.config.uniform_refinement),
                    "space_refinement_mode": self.config.space_refinement_mode,
                }
            )
            self.times = refinement.times
            self.meshes = refinement.meshes
            if self.grid_callback is not None:
                self.grid_callback(self, iteration + 1)

        if final_objects is None:
            raise RuntimeError("The adaptive loop performed no iteration.")
        return CylinderAdaptiveResult(
            config=self.config,
            history=self.history,
            times=self.times,
            meshes=self.meshes,
            primal=self.last["primal"],
            primal_enriched=self.last["primal_enriched"],
            dual_low=self.last["dual_low"],
            dual_enriched=self.last["dual_enriched"],
            estimate=self.last["estimate"],
            localisation=self.last["localisation"],
        )


__all__ = [
    "CylinderAdaptiveConfig",
    "CylinderAdaptiveResult",
    "CylinderAdaptiveSolver",
    "GridRefinement",
    "refine_causal_slab_grid",
    "refine_common_slab_grid",
    "refine_independent_slab_grid",
]
