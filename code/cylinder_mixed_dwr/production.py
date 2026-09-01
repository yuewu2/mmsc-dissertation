r"""Stage G production runner and durable experiment output.

The adaptive mathematics lives in :mod:`cylinder_mixed_dwr.adaptive`.  This
module adds a reproducible CLI and writes every *completed* outer iteration
before refinement starts.  A later nonlinear or linear failure therefore
does not erase the diagnostics from earlier cycles.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from firedrake import (
    Function,
    FunctionSpace,
    VTKFile,
    VectorFunctionSpace,
    div,
    project,
)
from firedrake.petsc import PETSc
from navier_stokes_cylinder_irksome_static_primal import _make_hierarchy

from .adaptive import CylinderAdaptiveConfig, CylinderAdaptiveSolver
from .benchmark import R5CylinderSpecification
from .checkpoint import CylinderCheckpointStore


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


class CylinderProductionOutput:
    """Persist convergence data, indicators, and optional physical fields."""

    def __init__(
        self,
        prefix: str,
        *,
        write_vtk: bool = False,
        checkpoint_root: str | None = None,
        resumed_from: str | None = None,
    ):
        self.prefix = Path(prefix)
        self.write_vtk = bool(write_vtk)
        self.history: list[dict[str, Any]] = []
        self.resumed_from = resumed_from
        self.checkpoint_store = (
            None
            if checkpoint_root is None
            else CylinderCheckpointStore(checkpoint_root)
        )
        self.latest_checkpoint: str | None = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.wall_started = perf_counter()
        self.config: dict[str, Any] | None = None
        self.prefix.parent.mkdir(parents=True, exist_ok=True)

    @property
    def history_path(self) -> Path:
        return Path(f"{self.prefix}_history.csv")

    @property
    def manifest_path(self) -> Path:
        return Path(f"{self.prefix}_manifest.json")

    def initialise(
        self,
        config: CylinderAdaptiveConfig,
        *,
        inherited_history=None,
    ) -> None:
        self.config = asdict(config)
        self.history = [dict(row) for row in (inherited_history or [])]
        self._write_history()
        self._write_manifest(status="running")

    def _atomic_text(self, path: Path, payload: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def _write_history(self) -> None:
        if not self.history:
            return
        temporary = self.history_path.with_suffix(".csv.tmp")
        fieldnames = list(self.history[0])
        known = set(fieldnames)
        for row in self.history[1:]:
            for key in row:
                if key not in known:
                    fieldnames.append(key)
                    known.add(key)
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        temporary.replace(self.history_path)

    def _write_manifest(self, *, status: str, failure: str | None = None) -> None:
        transfer_mode = self.config.get("interface_transfer_mode", "stokes_l2")
        if transfer_mode == "mass":
            primal_interface = "velocity_mass_riesz"
            adjoint_interface = "exact_L2_adjoint_of_velocity_mass_riesz"
        elif transfer_mode == "stokes_l2":
            primal_interface = "R5_interpolation_then_Taylor_Hood_Stokes_L2"
            adjoint_interface = "exact_L2_adjoint_Istar_Q0"
        elif transfer_mode == "stokes_h1":
            primal_interface = "interpolation_then_Taylor_Hood_Stokes_H1"
            adjoint_interface = "exact_L2_adjoint_Istar_QH1star"
        else:
            raise ValueError(f"Unsupported interface transfer: {transfer_mode!r}")
        manifest = {
            "stage": "G",
            "implementation": {
                "goal_functional": "R5_surface_traction_mean_drag",
                "goal_reference_T8": 1.6031368,
                "drag_normal": "obstacle_outward_equals_minus_Firedrake_fluid_normal",
                "same_mesh_interface": "exact_identity",
                "changed_mesh_primal_interface": primal_interface,
                "changed_mesh_adjoint_interface": adjoint_interface,
                "pressure_interface": "no_mass_trace_algebraic_initial_guess_only",
                "causal_mesh_identity": (
                    "reuse_same_parent_and_identical_cumulative_marks"
                ),
            },
            "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": perf_counter() - self.wall_started,
            "config": self.config,
            "completed_iterations": len(self.history),
            "resumed_from": self.resumed_from,
            "checkpoint_root": (
                None
                if self.checkpoint_store is None
                else str(self.checkpoint_store.root)
            ),
            "latest_checkpoint": self.latest_checkpoint,
            "history_file": str(self.history_path),
            "failure": failure,
        }
        self._atomic_text(
            self.manifest_path,
            json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n",
        )

    def _write_indicators(self, iteration: int, solver, objects) -> None:
        localisation = objects["localisation"]
        directional_split = objects.get("directional_split")
        marking_signed = objects.get(
            "marking_signed", localisation.eta_cell_signed
        )
        marked = objects["marked"]
        patch_signed = objects.get("patch_signed")
        patch_marked = objects.get("patch_marked")
        arrays: dict[str, np.ndarray] = {
            "times": np.asarray(solver.times, dtype=float),
        }
        for n in range(1, len(solver.times)):
            arrays[f"eta_signed_slab_{n}"] = np.asarray(
                localisation.eta_cell_signed[n], dtype=float
            )
            arrays[f"eta_marking_slab_{n}"] = np.asarray(
                marking_signed[n], dtype=float
            )
            arrays[f"eta_primal_slab_{n}"] = np.asarray(
                localisation.eta_primal_bubble_cell[n], dtype=float
            )
            arrays[f"eta_adjoint_slab_{n}"] = np.asarray(
                localisation.eta_adjoint_bubble_cell[n], dtype=float
            )
            arrays[f"eta_correction_slab_{n}"] = np.asarray(
                localisation.eta_correction_bubble_cell[n], dtype=float
            )
            for term in ("primal", "adjoint", "correction"):
                for entity in ("volume", "spatial", "temporal"):
                    arrays[f"eta_{term}_{entity}_slab_{n}"] = np.asarray(
                        getattr(localisation, f"eta_{term}_{entity}_cell")[n],
                        dtype=float,
                    )
                arrays[f"eta_{term}_ridge_slab_{n}"] = np.asarray(
                    getattr(localisation, f"eta_{term}_mixed_ridge_cell")[n],
                    dtype=float,
                )
            arrays[f"eta_cubic_remainder_slab_{n}"] = np.asarray(
                localisation.eta_cubic_remainder_cell[n], dtype=float
            )
            if directional_split is not None:
                eta_time = np.asarray(
                    directional_split.time_localisation.eta_cell_signed[n],
                    dtype=float,
                )
                eta_space = np.asarray(
                    directional_split.space_localisation.eta_cell_signed[n],
                    dtype=float,
                )
                arrays[f"eta_directional_time_slab_{n}"] = eta_time
                arrays[f"eta_directional_space_slab_{n}"] = eta_space
                arrays[f"eta_directional_sum_slab_{n}"] = (
                    eta_time + eta_space
                )
            arrays[f"marked_slab_{n}"] = np.asarray(marked[n], dtype=np.uint8)
            if patch_signed is not None:
                arrays[f"eta_vertex_patch_slab_{n}"] = np.asarray(
                    patch_signed[n], dtype=float
                )
                arrays[f"marked_vertex_patch_slab_{n}"] = np.asarray(
                    patch_marked[n], dtype=np.uint8
                )
        np.savez_compressed(f"{self.prefix}_iter_{iteration}_indicators.npz", **arrays)

    def _write_vtk(self, iteration: int, objects) -> None:
        if not self.write_vtk:
            return
        primal = objects["primal"]
        localisation = objects["localisation"]
        marking_signed = objects.get(
            "marking_signed", localisation.eta_cell_signed
        )
        for n in range(1, len(primal["slabs"])):
            slab = primal["slabs"][n]
            velocity = Function(
                slab["right_trace"].subfunctions[0].function_space(),
                name="Velocity",
            )
            velocity.assign(slab["right_trace"].subfunctions[0])
            pressure = Function(
                slab["right_trace"].subfunctions[1].function_space(),
                name="Pressure",
            )
            pressure.assign(slab["right_trace"].subfunctions[1])
            DG0 = FunctionSpace(slab["mesh"], "DG", 0)
            divergence = project(div(velocity), DG0, name="Divergence")
            eta_signed = Function(DG0, name="eta_Kn_signed")
            eta_signed.dat.data[:] = localisation.eta_cell_signed[n]
            eta_absolute = Function(DG0, name="eta_Kn_abs")
            eta_absolute.dat.data[:] = np.abs(localisation.eta_cell_signed[n])
            eta_marking = Function(DG0, name="eta_Kn_marking_signed")
            eta_marking.dat.data[:] = marking_signed[n]
            eta_marking_absolute = Function(DG0, name="eta_Kn_marking_abs")
            eta_marking_absolute.dat.data[:] = np.abs(marking_signed[n])
            VTKFile(f"{self.prefix}_iter_{iteration}_slab_{n}.pvd").write(
                velocity,
                pressure,
                divergence,
                eta_signed,
                eta_absolute,
                eta_marking,
                eta_marking_absolute,
                time=float(primal["times"][n]),
            )

    def completed_iteration(self, solver, row, objects) -> None:
        self.history.append(dict(row))
        self._write_history()
        self._write_indicators(int(row["iteration"]), solver, objects)
        self._write_vtk(int(row["iteration"]), objects)
        self._write_manifest(status="running")

    def checkpoint_grid(self, solver, grid_iteration: int) -> None:
        if self.checkpoint_store is None:
            return
        path = self.checkpoint_store.save_grid(solver, grid_iteration)
        self.latest_checkpoint = str(path)
        self._write_manifest(status="running")
        PETSc.Sys.Print(
            f"[CYLINDER CHECKPOINT] grid={grid_iteration}; path={path}"
        )

    def complete(self) -> None:
        self._write_manifest(status="complete")

    def fail(self, error: BaseException) -> None:
        self._write_manifest(
            status="failed", failure=f"{type(error).__name__}: {error}"
        )

    def write_stationarity_diagnostic(self, solver) -> Path | None:
        estimate = solver.last.get("estimate")
        if estimate is None:
            return None
        scale = float(solver.last.get("stationarity_identity_scale", float("nan")))
        rows = []
        for n in range(1, len(solver.times)):
            pv = float(estimate.enriched_primal_stationarity_volume_by_slab[n])
            pj = float(estimate.enriched_primal_stationarity_jump_by_slab[n])
            av = float(estimate.enriched_adjoint_stationarity_volume_by_slab[n])
            aj = float(estimate.enriched_adjoint_stationarity_jump_by_slab[n])
            rows.append({
                "slab": n,
                "t_left": float(solver.times[n - 1]),
                "t_right": float(solver.times[n]),
                "cells": int(solver.meshes[n].num_cells()),
                "same_mesh_as_previous": (
                    n == 1 or solver.meshes[n] is solver.meshes[n - 1]
                ),
                "primal_volume": pv,
                "primal_jump": pj,
                "primal_total": pv + pj,
                "adjoint_volume": av,
                "adjoint_jump": aj,
                "adjoint_total": av + aj,
            })
        payload = {
            "identity_scale": scale,
            "goal_difference": float(estimate.enriched_goal_difference),
            "global": {
                "primal": float(estimate.enriched_primal_stationarity_defect),
                "adjoint": float(estimate.enriched_adjoint_stationarity_defect),
                "primal_relative": abs(float(estimate.enriched_primal_stationarity_defect)) / scale,
                "adjoint_relative": abs(float(estimate.enriched_adjoint_stationarity_defect)) / scale,
            },
            "slabs": rows,
        }
        path = Path(f"{self.prefix}_stationarity_diagnostic.json")
        self._atomic_text(path, json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
        return path


def _reference_for(final_time: float, explicit: float | None):
    if explicit is not None:
        return float(explicit)
    specification = R5CylinderSpecification()
    if abs(float(final_time) - specification.final_time) <= 1.0e-12:
        return specification.nonlinear_mean_drag_reference
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument(
        "--geometry-degree",
        type=int,
        default=None,
        help=(
            "Polynomial degree of the Netgen/Firedrake coordinate mapping. "
            "The default is two, giving a curved cylinder boundary."
        ),
    )
    parser.add_argument("--nt", type=int, default=128)
    parser.add_argument("--T", type=float, default=8.0)
    parser.add_argument("--nu", type=float, default=1.0e-3)
    parser.add_argument("--primal-time-degree", type=int, default=1)
    parser.add_argument("--enriched-time-degree", type=int, default=2)
    parser.add_argument(
        "--enriched-time-refinement-factor",
        type=int,
        default=None,
        help=(
            "Uniform temporal children per low slab for the enriched adjoint. "
            "A value of two gives N_t^H=2*N_t and is available only for "
            "the primal-only linear estimator."
        ),
    )
    parser.add_argument(
        "--quadrature-points",
        type=int,
        default=None,
        help=(
            "Gauss--Legendre points used to evaluate the DWR estimator and "
            "goal functional. Fresh runs default to seven; resumed runs "
            "preserve the checkpoint value unless explicitly overridden."
        ),
    )
    parser.add_argument("--max-it", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--true-goal-error-tolerance",
        type=float,
        default=None,
        help=(
            "Stop after a completed iteration when abs(J_reference-J_h) is "
            "at most this value. Requires --reference-drag."
        ),
    )
    parser.add_argument("--theta", type=float, default=0.30)
    parser.add_argument(
        "--space-marking-strategy",
        choices=("cellwise", "vertex_patch"),
        default=None,
        help=(
            "Use ordinary cellwise abs(eta_Kn) Dörfler marking or first "
            "aggregate signed cell contributions on CG1 vertex patches."
        ),
    )
    parser.add_argument("--time-fraction", type=float, default=0.05)
    parser.add_argument(
        "--time-marking-strategy",
        choices=(
            "cell_fraction",
            "cell_fraction_capped",
            "fixed_rate",
            "slab_bulk_capped",
        ),
        default=None,
        help=(
            "Temporal marking policy for future adaptive steps. "
            "cell_fraction_capped preserves the thesis marked-cell trigger "
            "but limits the number of bisected slabs using "
            "--time-max-fraction and --time-max-count."
        ),
    )
    parser.add_argument(
        "--time-fixed-rate",
        type=float,
        default=None,
        help="Fraction of slabs bisected by fixed-rate time marking.",
    )
    parser.add_argument(
        "--time-score-source",
        choices=(
            "marked_fraction",
            "combined_indicator",
            "directional_time",
        ),
        default=None,
        help=(
            "Slab score feeding fixed_rate/slab_bulk_capped time marking: "
            "'marked_fraction' keeps the spatial-mark-count trigger; "
            "'combined_indicator' reuses the production localisation and "
            "ranks slabs by sum_K abs(eta_Kn), without a directional split; "
            "'directional_time' ranks slabs by the localised temporal "
            "component |rho(U_h)(z+ - I_k z+)| of the R5 split (requires "
            "--directional-split-diagnostic)."
        ),
    )
    parser.add_argument("--time-bulk-fraction", type=float, default=None)
    parser.add_argument("--time-max-fraction", type=float, default=None)
    parser.add_argument("--time-max-count", type=int, default=None)
    parser.add_argument("--dwr-identity-rtol", type=float, default=1.0e-2)
    parser.add_argument(
        "--dwr-stationarity-rtol",
        type=float,
        default=None,
        help=(
            "Relative stationarity gate. When resuming, omission preserves "
            "the checkpoint value; an explicit value records a controlled "
            "experimental override."
        ),
    )
    parser.add_argument("--recovery-gap", type=float, default=0.05)
    parser.add_argument("--recovery-space-degree", type=int, default=2)
    parser.add_argument("--facet-recovery-degree", type=int, default=2)
    parser.add_argument("--recovery-time-degree", type=int, default=2)
    parser.add_argument("--reference-drag", type=float, default=None)
    parser.add_argument("--output-prefix", default="output/cylinder_mixed_dwr/r5_adaptive")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from an immutable iter_NNNN checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Checkpoint directory (default: OUTPUT_PREFIX_checkpoints).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable outer-grid checkpoints.",
    )
    parser.add_argument("--write-vtk", action="store_true")
    parser.add_argument("--no-space-refinement", action="store_true")
    parser.add_argument("--no-time-refinement", action="store_true")
    parser.add_argument(
        "--uniform-refinement",
        action="store_true",
        help=(
            "Mark every spatial cell and, through the existing slabwise "
            "cell-fraction trigger, bisect every time slab after each "
            "accepted nonterminal iteration."
        ),
    )
    parser.add_argument(
        "--space-mode",
        choices=("independent", "causal", "common"),
        default=None,
        help=(
            "Spatial refinement policy for future adaptive steps. A resumed "
            "checkpoint is always restored exactly before this policy is applied."
        ),
    )
    parser.add_argument(
        "--interface-transfer",
        choices=("mass", "stokes_l2", "stokes_h1"),
        default=None,
        help=(
            "Cross-mesh Taylor--Hood velocity trace operator. stokes_l2 "
            "uses R5-style interpolation followed by a divergence-free "
            "target Stokes projection; stokes_h1 uses the H1-seminorm "
            "projection with its exact L2 jump-pairing adjoint."
        ),
    )
    parser.add_argument(
        "--primal-space",
        choices=("taylor_hood",),
        default=None,
        help="Low primal/adjoint mixed pair; Taylor--Hood means CG2/CG1.",
    )
    parser.add_argument(
        "--enriched-velocity-degree",
        type=int,
        default=None,
        help="Velocity degree in the enriched Taylor--Hood pair.",
    )
    parser.add_argument(
        "--enriched-pressure-degree",
        type=int,
        default=None,
        help="Pressure degree in the enriched Taylor--Hood pair.",
    )
    parser.add_argument(
        "--dual-weight-mode",
        choices=("enriched_minus_numerical", "enriched_minus_interpolant"),
        default=None,
        help="Adjoint sensitivity used in the primal-residual DWR estimator.",
    )
    parser.add_argument(
        "--dual-base-strategy",
        choices=(
            "numerical", "projected_enriched", "interpolated_enriched"
        ),
        default=None,
        help=(
            "Construct the low comparator by a separate adjoint solve, a "
            "constrained spatial projection, or nodal space-time interpolation."
        ),
    )
    parser.add_argument(
        "--estimator-strategy",
        choices=(
            "primal_only",
            "symmetric_three_term",
            "symmetric_two_term",
        ),
        default=None,
        help=(
            "Estimator path. symmetric_two_term removes the Galerkin "
            "correction from both the global estimator and localisation/marking."
        ),
    )
    parser.add_argument(
        "--three-term-marking",
        choices=("full", "residual_pair"),
        default=None,
        help=(
            "For the symmetric estimator, mark either the full localised "
            "three-term field or only 0.5*(primal+adjoint), retaining the "
            "Galerkin correction in the global estimator."
        ),
    )
    parser.add_argument(
        "--include-cubic-remainder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include the exact Navier--Stokes cubic Lagrangian remainder in "
            "the nonlinear symmetric estimator and its cell localisation. "
            "This does not add the Galerkin correction to symmetric_two_term."
        ),
    )
    parser.add_argument(
        "--directional-split-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Opt in to an R5-style temporal/spatial split of the linear "
            "dual weight. It writes separate diagnostics and leaves the "
            "production estimator/localisation unchanged; temporal marking "
            "uses it only when --time-score-source=directional_time."
        ),
    )
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Report initial-grid dimensions and a trajectory-storage lower bound, then exit.",
    )
    return parser


def _clear_petsc_cli_options(parser: argparse.ArgumentParser) -> None:
    options = PETSc.Options()
    for action in parser._actions:
        for flag in action.option_strings:
            if flag.startswith("--"):
                options.delValue(flag)


def run_from_args(args) -> object:
    loaded = (
        None
        if args.resume_from is None
        else CylinderCheckpointStore.load(args.resume_from)
    )
    if loaded is None:
        reference = _reference_for(args.T, args.reference_drag)
        config = CylinderAdaptiveConfig(
            hierarchy_levels=args.levels,
            geometry_degree=(
                2 if args.geometry_degree is None else int(args.geometry_degree)
            ),
            initial_time_slabs=args.nt,
            final_time=args.T,
            viscosity=args.nu,
            primal_time_degree=args.primal_time_degree,
            enriched_time_degree=args.enriched_time_degree,
            enriched_time_refinement_factor=(
                1
                if args.enriched_time_refinement_factor is None
                else int(args.enriched_time_refinement_factor)
            ),
            quadrature_points=(
                7 if args.quadrature_points is None
                else int(args.quadrature_points)
            ),
            max_iterations=args.max_it,
            tolerance=args.tolerance,
            true_goal_error_tolerance=args.true_goal_error_tolerance,
            theta=args.theta,
            time_marked_fraction=args.time_fraction,
            time_marking_strategy=args.time_marking_strategy or "cell_fraction",
            time_score_source=args.time_score_source or "marked_fraction",
            time_fixed_rate=(
                0.20 if args.time_fixed_rate is None else args.time_fixed_rate
            ),
            time_bulk_fraction=(
                0.20 if args.time_bulk_fraction is None else args.time_bulk_fraction
            ),
            time_max_fraction=(
                0.15 if args.time_max_fraction is None else args.time_max_fraction
            ),
            time_max_count=(
                15 if args.time_max_count is None else args.time_max_count
            ),
            dwr_identity_relative_tolerance=args.dwr_identity_rtol,
            dwr_stationarity_relative_tolerance=(
                1.0e-4
                if args.dwr_stationarity_rtol is None
                else float(args.dwr_stationarity_rtol)
            ),
            dual_weight_mode=(
                args.dual_weight_mode or "enriched_minus_interpolant"
            ),
            dual_base_strategy=(
                args.dual_base_strategy
                or (
                    "numerical"
                    if args.dual_weight_mode == "enriched_minus_numerical"
                    else "interpolated_enriched"
                )
            ),
            directional_split_diagnostic=bool(
                args.directional_split_diagnostic
            ),
            include_cubic_remainder=bool(args.include_cubic_remainder),
            estimator_strategy=args.estimator_strategy or "primal_only",
            three_term_marking_strategy=(
                args.three_term_marking or "full"
            ),
            enable_space_refinement=not args.no_space_refinement,
            enable_time_refinement=not args.no_time_refinement,
            uniform_refinement=bool(args.uniform_refinement),
            space_marking_strategy=(
                args.space_marking_strategy or "cellwise"
            ),
            space_refinement_mode=args.space_mode or "independent",
            interface_transfer_mode=args.interface_transfer or "stokes_l2",
            primal_space_family=args.primal_space or "taylor_hood",
            enriched_velocity_degree=(
                4 if args.enriched_velocity_degree is None
                else int(args.enriched_velocity_degree)
            ),
            enriched_pressure_degree=(
                2 if args.enriched_pressure_degree is None
                else int(args.enriched_pressure_degree)
            ),
            maximum_recovery_gap_relative=args.recovery_gap,
            recovery_space_degree=args.recovery_space_degree,
            facet_recovery_degree=args.facet_recovery_degree,
            recovery_time_degree=args.recovery_time_degree,
            report_every=args.report_every,
            reference_mean_drag=reference,
        )
    else:
        resumed_config = dict(loaded.saved_config)
        resumed_config.update(
            {
                "max_iterations": int(args.max_it),
                "tolerance": float(args.tolerance),
                "theta": float(args.theta),
                "time_marked_fraction": float(args.time_fraction),
                "enable_space_refinement": not args.no_space_refinement,
                "enable_time_refinement": not args.no_time_refinement,
                "uniform_refinement": bool(args.uniform_refinement),
                "maximum_recovery_gap_relative": float(args.recovery_gap),
                "report_every": int(args.report_every),
            }
        )
        if args.true_goal_error_tolerance is not None:
            resumed_config["true_goal_error_tolerance"] = float(
                args.true_goal_error_tolerance
            )
        if args.space_mode is not None:
            resumed_config["space_refinement_mode"] = args.space_mode
        if args.space_marking_strategy is not None:
            resumed_config["space_marking_strategy"] = (
                args.space_marking_strategy
            )
        if args.geometry_degree is not None:
            saved_degree = int(resumed_config.get("geometry_degree", 1))
            if int(args.geometry_degree) != saved_degree:
                raise ValueError(
                    "A resumed checkpoint cannot change geometry_degree: "
                    f"saved={saved_degree}, requested={args.geometry_degree}."
                )
        if args.time_marking_strategy is not None:
            resumed_config["time_marking_strategy"] = args.time_marking_strategy
        if args.time_score_source is not None:
            resumed_config["time_score_source"] = args.time_score_source
        if args.time_fixed_rate is not None:
            resumed_config["time_fixed_rate"] = float(args.time_fixed_rate)
        if args.time_bulk_fraction is not None:
            resumed_config["time_bulk_fraction"] = float(args.time_bulk_fraction)
        if args.time_max_fraction is not None:
            resumed_config["time_max_fraction"] = float(args.time_max_fraction)
        if args.time_max_count is not None:
            resumed_config["time_max_count"] = int(args.time_max_count)
        if args.interface_transfer is not None:
            resumed_config["interface_transfer_mode"] = args.interface_transfer
        if args.primal_space is not None:
            resumed_config["primal_space_family"] = args.primal_space
        if args.enriched_velocity_degree is not None:
            resumed_config["enriched_velocity_degree"] = int(
                args.enriched_velocity_degree
            )
        if args.enriched_pressure_degree is not None:
            resumed_config["enriched_pressure_degree"] = int(
                args.enriched_pressure_degree
            )
        if args.dual_weight_mode is not None:
            resumed_config["dual_weight_mode"] = args.dual_weight_mode
        if args.dual_base_strategy is not None:
            resumed_config["dual_base_strategy"] = args.dual_base_strategy
        if args.estimator_strategy is not None:
            resumed_config["estimator_strategy"] = args.estimator_strategy
        if args.three_term_marking is not None:
            resumed_config["three_term_marking_strategy"] = (
                args.three_term_marking
            )
        if args.include_cubic_remainder is not None:
            resumed_config["include_cubic_remainder"] = bool(
                args.include_cubic_remainder
            )
        if args.directional_split_diagnostic is not None:
            resumed_config["directional_split_diagnostic"] = bool(
                args.directional_split_diagnostic
            )
        if args.dwr_stationarity_rtol is not None:
            resumed_config["dwr_stationarity_relative_tolerance"] = float(
                args.dwr_stationarity_rtol
            )
        if args.quadrature_points is not None:
            resumed_config["quadrature_points"] = int(args.quadrature_points)
        if args.enriched_time_refinement_factor is not None:
            resumed_config["enriched_time_refinement_factor"] = int(
                args.enriched_time_refinement_factor
            )
        resumed_config["dwr_identity_relative_tolerance"] = float(
            args.dwr_identity_rtol
        )
        if args.reference_drag is not None:
            resumed_config["reference_mean_drag"] = float(args.reference_drag)
        config = CylinderAdaptiveConfig(**resumed_config)
        PETSc.Sys.Print(
            "[CYLINDER RESUME] structural discretisation restored from "
            f"{loaded.path}; future adaptive controls use the current CLI."
        )
    checkpoint_root = None
    if not args.no_checkpoint:
        checkpoint_root = (
            args.checkpoint_root
            if args.checkpoint_root is not None
            else f"{args.output_prefix}_checkpoints"
        )
    output = CylinderProductionOutput(
        args.output_prefix,
        write_vtk=args.write_vtk,
        checkpoint_root=checkpoint_root,
        resumed_from=None if loaded is None else str(loaded.path),
    )
    output.initialise(
        config,
        inherited_history=None if loaded is None else loaded.history,
    )
    solver = None
    try:
        solver = CylinderAdaptiveSolver(
            config,
            iteration_callback=output.completed_iteration,
            grid_callback=output.checkpoint_grid,
            initial_times=None if loaded is None else loaded.times,
            initial_meshes=None if loaded is None else loaded.meshes,
            initial_history=None if loaded is None else loaded.history,
            start_iteration=0 if loaded is None else loaded.grid_iteration,
            hierarchy=None if loaded is None else loaded.hierarchy,
            labels=None if loaded is None else loaded.labels,
            refinement_lineage=(
                None if loaded is None else loaded.refinement_lineage
            ),
        )
        result = solver.solve()
    except BaseException as error:
        if solver is not None:
            output.write_stationarity_diagnostic(solver)
        output.fail(error)
        raise
    output.complete()
    PETSc.Sys.Print(
        f"[CYLINDER STAGE G] history={output.history_path}; "
        f"manifest={output.manifest_path}"
    )
    return result


def run_preflight(args) -> dict[str, Any]:
    if args.resume_from is None:
        hierarchy, *_ = _make_hierarchy(
            args.levels,
            geometry_degree=(
                2 if args.geometry_degree is None else int(args.geometry_degree)
            ),
        )
        meshes = [None] + [hierarchy[-1]] * int(args.nt)
        nslabs = int(args.nt)
        primal_degree = int(args.primal_time_degree)
        enriched_degree = int(args.enriched_time_degree)
        hierarchy_levels = int(args.levels)
        primal_space_family = args.primal_space or "taylor_hood"
        enriched_velocity_degree = (
            4 if args.enriched_velocity_degree is None
            else int(args.enriched_velocity_degree)
        )
        enriched_pressure_degree = (
            2 if args.enriched_pressure_degree is None
            else int(args.enriched_pressure_degree)
        )
        enriched_time_factor = (
            1 if args.enriched_time_refinement_factor is None
            else int(args.enriched_time_refinement_factor)
        )
    else:
        loaded = CylinderCheckpointStore.load(args.resume_from)
        meshes = loaded.meshes
        nslabs = len(loaded.times) - 1
        primal_degree = int(loaded.saved_config["primal_time_degree"])
        enriched_degree = int(loaded.saved_config["enriched_time_degree"])
        hierarchy_levels = int(loaded.saved_config["hierarchy_levels"])
        primal_space_family = "taylor_hood"
        enriched_velocity_degree = (
            int(args.enriched_velocity_degree)
            if args.enriched_velocity_degree is not None
            else int(loaded.saved_config.get("enriched_velocity_degree", 4))
        )
        enriched_pressure_degree = (
            int(args.enriched_pressure_degree)
            if args.enriched_pressure_degree is not None
            else int(loaded.saved_config.get("enriched_pressure_degree", 2))
        )
        enriched_time_factor = (
            int(args.enriched_time_refinement_factor)
            if args.enriched_time_refinement_factor is not None
            else int(
                loaded.saved_config.get("enriched_time_refinement_factor", 1)
            )
        )

    dimension_cache: dict[int, tuple[int, int]] = {}
    low_dimensions = []
    rich_dimensions = []
    for mesh in meshes[1:]:
        key = id(mesh)
        dimensions = dimension_cache.get(key)
        if dimensions is None:
            low_velocity = VectorFunctionSpace(mesh, "CG", 2)
            low_pressure = FunctionSpace(mesh, "CG", 1)
            low_mixed = low_velocity * low_pressure
            rich_velocity = VectorFunctionSpace(
                mesh, "CG", enriched_velocity_degree
            )
            rich_pressure = FunctionSpace(
                mesh, "CG", enriched_pressure_degree
            )
            dimensions = (int(low_mixed.dim()), int((rich_velocity * rich_pressure).dim()))
            dimension_cache[key] = dimensions
        low_dimensions.append(dimensions[0])
        rich_dimensions.append(dimensions[1])
    low_dimension = int(low_dimensions[0])
    rich_dimension = int(rich_dimensions[0])
    low_dimension_sum = int(sum(low_dimensions))
    rich_dimension_sum = int(sum(rich_dimensions))
    # Each of primal/adjoint stores r+1 polynomial coefficients and two
    # traces.  This deliberately excludes nonlinear solver work vectors and
    # recovery entities, so it is a transparent lower bound rather than a
    # peak-memory promise.
    # The enriched adjoint trajectory lives on `enriched_time_factor`
    # uniform child slabs per low slab; its coefficient storage scales
    # with the factor, the low trajectory does not.
    scalar_dofs_per_slab = (
        2 * (primal_degree + 3) * low_dimension_sum
        + 2 * (enriched_degree + 3) * rich_dimension_sum * enriched_time_factor
    )
    result = {
        "hierarchy_levels": hierarchy_levels,
        "initial_cells": int(meshes[1].num_cells()),
        "time_slabs": nslabs,
        "unique_spatial_meshes": len(dimension_cache),
        "primal_space_family": primal_space_family,
        "primal_time_degree": primal_degree,
        "enriched_velocity_degree": enriched_velocity_degree,
        "enriched_pressure_degree": enriched_pressure_degree,
        "low_mixed_spatial_dimension": low_dimension,
        "rich_mixed_spatial_dimension": rich_dimension,
        "primal_spacetime_dofs": int(
            (primal_degree + 1) * low_dimension_sum
        ),
        "enriched_time_refinement_factor": enriched_time_factor,
        "enriched_spacetime_dofs": int(
            (enriched_degree + 1) * rich_dimension_sum * enriched_time_factor
        ),
        "stored_trajectory_lower_bound_gib": (
            scalar_dofs_per_slab * 8.0 / 1024.0**3
        ),
        "warning": (
            "The memory estimate excludes matrices, factorisations, work vectors, "
            "and tensor-recovery entities; peak memory is higher."
        ),
    }
    PETSc.Sys.Print("[CYLINDER STAGE G PREFLIGHT] " + json.dumps(result, indent=2))
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    _clear_petsc_cli_options(parser)
    if args.preflight:
        return run_preflight(args)
    return run_from_args(args)


if __name__ == "__main__":
    main()


__all__ = [
    "CylinderProductionOutput",
    "build_parser",
    "run_from_args",
    "run_preflight",
]
