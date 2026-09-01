"""Validated input data for the transient bubble-projection solver.

The layout deliberately mirrors Firedrake's ``GoalAdaptiveOptions``: values
which control the outer adaptive algorithm live in one object, while all
remaining PETSc dictionaries are passed unchanged to the inner linear solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class BubbleProjectionOptions:
    r"""Parameters for ``SOLVE -> ESTIMATE -> MARK -> REFINE``.

    The default DWR estimator is

    .. math::

       \eta = \rho(u_h)(z^\star),\qquad z^\star=z_{p+1,r}-z_{p,0},

    and the bubble projections split it into signed space--time contributions
    ``eta[K, n]``.  ``marking_strategy`` determines whether these values are
    marked globally, by separate space/time bulk criteria, or by the fixed-rate
    policy used in the Hartmann/Thiele--Wick comparison.
    For a nonlinear problem, ``nonlinear_error_identity=True`` instead uses
    the symmetric Lagrangian identity and reports the enriched goal difference
    minus the computed terms as an observed remainder.  The Galerkin correction
    is retained by default and can be omitted explicitly for a two-term study.
    """

    # The stop criterion is ``|eta_global| <= tolerance``.
    tolerance: float = 6.0e-4
    max_it: int = 6

    # Base trial space is ``CG(spatial_degree) x DG(primal_time_degree)``.
    spatial_degree: int = 1
    primal_time_degree: int = 0
    dual_extra_spatial_degree: int = 1
    dual_time_degree: int = 1
    # ``z*`` is either ``z_enriched-z_low`` or ``z_enriched-I_h(z_enriched)``.
    dual_weight_mode: str = "enriched_minus_numerical"
    # ``hierarchical_recovery`` is the thesis localisation.  The joint cell
    # partition is an algebraically closing DG0 cell/slab diagnostic.  The
    # legacy ``weak_cell_partition`` spelling remains an input alias, but it
    # is not the nodal cG(1)dG(0) PU of Thiele--Wick.
    localisation_mode: str = "hierarchical_recovery"
    # Preserve the signed hierarchical estimator while optionally using the
    # sum of absolute entity contributions as the positive marking score.
    bubble_marking_score: str = "signed_total"
    # Diagnostic ablation for nonlinear two-term indicators.  ``total`` is
    # the production method; the other choices drive marking with only one
    # of 1/2*rho(e_z) and 1/2*rho*(e_u), without changing either solve or the
    # reported estimator.
    diagnostic_marking_component: str = "total"
    nonlinear_adjoint_localisation: str = "bubble_recovery"
    # Diagnostic ablation only: the default retains the codimension-two
    # mixed ridge in the hierarchical recovery/localisation identity.
    include_mixed_ridge: bool = True
    # Use the symmetric nonlinear error identity with enriched primal and dual.
    nonlinear_error_identity: bool = False
    # Include ``-rho(u_h)(z_h)`` in both global and local nonlinear estimates.
    include_galerkin_correction: bool = True

    # Recovery represents cell residuals in ``DG(q_x) x P(q_t)``.
    recovery_space_degree: int = 1
    facet_recovery_degree: int = 1
    recovery_time_degree: int = 1
    recovery_quadrature_points: int = 5
    # Recovery acceptance is measured against total signed-indicator
    # activity, the scale used by Dörfler marking.  The gap/global ratio is
    # still reported separately and may be large under strong cancellation.
    localisation_closure_tolerance: float = 0.05

    # Dörfler bulk criterion: sum over marked ``|eta[K,n]|`` reaches theta.
    theta_spacetime: float = 0.2
    time_slab_marked_fraction: float = 0.05
    marking_strategy: str = "global_bulk_fraction_trigger"
    # For global space--time bulk marking, select temporal slabs either by the
    # original within-slab marked fraction or by their share of all marks.
    time_trigger_policy: str = "within_slab_fraction"
    enable_space_refinement: bool = True
    enable_time_refinement: bool = True
    # ``independent_slab`` preserves the original fully slab-local meshes.
    # ``causal_nested`` makes every later slab inherit earlier marked regions.
    # ``shared_time_mesh`` refines their union once on every slab as a robust
    # common-mesh baseline, so no fine-to-coarse transfer occurs.
    space_refinement_strategy: str = "independent_slab"

    # Output is independent from numerical assembly and may be disabled.
    write_vtk: bool = True
    # ``spacetime_only`` writes only the 2-D (x,t) adaptive-grid collection.
    vtk_output_mode: str = "all"
    # ``None`` preserves the legacy all-slab VTK output.  Otherwise only one
    # containing/nearest slab is written for each requested physical time.
    snapshot_times: tuple[float, ...] | None = None
    output_prefix: str = "output/comparison/heat_cone"
    verbose: bool = True

    # PETSc options are intentionally forwarded unchanged to Firedrake.
    solver_parameters: dict[str, Any] | None = None
    recovery_solver_parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject inputs that violate the finite-element construction."""
        if self.max_it < 1:
            raise ValueError("max_it must be at least one.")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.spatial_degree < 1:
            raise ValueError("spatial_degree must be at least one.")
        if self.primal_time_degree < 0 or self.dual_time_degree < 0:
            raise ValueError("DG time degrees must be nonnegative.")
        if self.dual_extra_spatial_degree < 1:
            raise ValueError("dual_extra_spatial_degree must be positive.")
        if self.dual_weight_mode not in {
            "enriched_minus_numerical",
            "enriched_minus_interpolant",
        }:
            raise ValueError(
                "dual_weight_mode must be 'enriched_minus_numerical' or "
                "'enriched_minus_interpolant'."
            )
        if self.localisation_mode not in {
            "hierarchical_recovery",
            "joint_cell_partition",
            "weak_cell_partition",
            "strong_residual_bound",
        }:
            raise ValueError(
                "localisation_mode must be 'hierarchical_recovery' or "
                "'joint_cell_partition', or 'strong_residual_bound' "
                "(legacy alias: 'weak_cell_partition')."
            )
        if self.bubble_marking_score not in {"signed_total", "componentwise_abs"}:
            raise ValueError(
                "bubble_marking_score must be 'signed_total' or 'componentwise_abs'."
            )
        if self.diagnostic_marking_component not in {"total", "primal", "adjoint"}:
            raise ValueError(
                "diagnostic_marking_component must be 'total', 'primal', or 'adjoint'."
            )
        if (
            self.diagnostic_marking_component != "total"
            and not self.nonlinear_error_identity
        ):
            raise ValueError(
                "primal/adjoint diagnostic marking requires nonlinear_error_identity=True."
            )
        if (
            self.diagnostic_marking_component != "total"
            and self.bubble_marking_score != "signed_total"
        ):
            raise ValueError(
                "primal/adjoint diagnostic marking requires bubble_marking_score='signed_total'."
            )
        if self.nonlinear_adjoint_localisation not in {
            "bubble_recovery", "cellwise_dg0"
        }:
            raise ValueError(
                "nonlinear_adjoint_localisation must be 'bubble_recovery' "
                "or 'cellwise_dg0'."
            )
        if (
            self.bubble_marking_score == "componentwise_abs"
            and self.localisation_mode != "hierarchical_recovery"
        ):
            raise ValueError(
                "componentwise_abs is defined only for hierarchical_recovery."
            )
        if (
            min(
                self.recovery_space_degree,
                self.facet_recovery_degree,
                self.recovery_time_degree,
            )
            < 0
        ):
            raise ValueError("Recovery polynomial degrees must be nonnegative.")
        if self.recovery_quadrature_points < 1:
            raise ValueError("recovery_quadrature_points must be at least one.")
        if not 0.0 < self.localisation_closure_tolerance < 1.0:
            raise ValueError("localisation_closure_tolerance must lie in (0, 1).")
        if not 0.0 < self.theta_spacetime <= 1.0:
            raise ValueError("theta_spacetime must lie in (0, 1].")
        if not 0.0 <= self.time_slab_marked_fraction <= 1.0:
            raise ValueError("time_slab_marked_fraction must lie in [0, 1].")
        if self.marking_strategy not in {
            "global_bulk_fraction_trigger",
            "separate_bulk",
            "fixed_rate",
            "uniform",
        }:
            raise ValueError(
                "marking_strategy must be 'global_bulk_fraction_trigger', "
                "'separate_bulk', 'fixed_rate', or 'uniform'."
            )
        if self.time_trigger_policy not in {
            "within_slab_fraction",
            "global_marked_share",
        }:
            raise ValueError(
                "time_trigger_policy must be 'within_slab_fraction' or "
                "'global_marked_share'."
            )
        if self.space_refinement_strategy not in {
            "independent_slab",
            "causal_nested",
            "shared_time_mesh",
        }:
            raise ValueError(
                "space_refinement_strategy must be 'independent_slab', "
                "'causal_nested', or 'shared_time_mesh'."
            )
        if self.vtk_output_mode not in {"all", "spacetime_only"}:
            raise ValueError("vtk_output_mode must be 'all' or 'spacetime_only'.")
        if self.snapshot_times is not None:
            if not self.snapshot_times:
                raise ValueError("snapshot_times must contain at least one time.")
            try:
                values = tuple(float(value) for value in self.snapshot_times)
            except (TypeError, ValueError) as exception:
                raise ValueError(
                    "snapshot_times must contain only real numbers."
                ) from exception
            if any(not isfinite(value) or value < 0.0 for value in values):
                raise ValueError("snapshot_times must be finite and nonnegative.")

    @property
    def dual_spatial_degree(self) -> int:
        """Return ``p+1`` (or the requested higher-order dual degree)."""
        return self.spatial_degree + self.dual_extra_spatial_degree

    def recovery_parameters(self) -> dict[str, Any]:
        """Give projections a robust default mass-matrix solver."""
        return self.recovery_solver_parameters or {
            "mat_type": "aij",
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-11,
            "ksp_atol": 1.0e-13,
            "ksp_max_it": 1000,
        }


@dataclass(frozen=True)
class NonstationaryDWRConfig:
    """Compact public configuration used by all problem input files."""

    nx: int = 8
    ny: int = 8
    nt: int = 8
    final_time: float = 0.5
    max_it: int = 6
    tolerance: float = 6.0e-4
    theta: float = 0.2
    time_fraction: float = 0.05
    marking_strategy: str = "global_bulk_fraction_trigger"
    time_trigger_policy: str = "within_slab_fraction"
    space_refinement_strategy: str = "independent_slab"
    spatial_degree: int = 1
    primal_time_degree: int = 0
    dual_extra_spatial_degree: int = 1
    dual_time_degree: int = 1
    dual_weight_mode: str = "enriched_minus_numerical"
    localisation_mode: str = "hierarchical_recovery"
    bubble_marking_score: str = "signed_total"
    diagnostic_marking_component: str = "total"
    nonlinear_adjoint_localisation: str = "bubble_recovery"
    include_mixed_ridge: bool = True
    recovery_space_degree: int = 1
    recovery_facet_degree: int = 1
    recovery_time_degree: int = 1
    quadrature_points: int = 5
    nonlinear_identity: bool = False
    include_galerkin_correction: bool = True
    write_vtk: bool = True
    vtk_output_mode: str = "all"
    snapshot_times: tuple[float, ...] | None = None
    output_prefix: str = "output/nonstationary_dwr/run"


__all__ = ["BubbleProjectionOptions", "NonstationaryDWRConfig"]
