"""Stationary-solver-style adaptive driver for transient bubble projection."""

from __future__ import annotations

# ``solver.py`` is a library module, not a stand-alone executable.  Relative
# imports require Python to load this directory as the ``nonstationary_dwr``
# package, so fail early with an actionable command.
if __name__ == "__main__" and not __package__:
    raise SystemExit("Run a top-level problem_*.py input file, not solver.py directly.")

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from firedrake import CellDiameter, Function, FunctionSpace, Mesh, assemble
from firedrake.petsc import PETSc

from .estimate import (
    BubbleEstimate,
    estimate_dwr_by_bubble_projection,
    estimate_dwr_by_joint_cell_partition,
    estimate_dwr_by_strong_residual_bound,
)
from .mark import (
    mark_each_slab,
    mark_slabs_by_activity,
    mark_spacetime_cells,
    marked_slab_global_shares,
    marked_slab_fractions,
)
from .refine import inherit_refinement_marks, refine_marked_mesh
from .output import AdaptiveOutput
from .options import BubbleProjectionOptions, NonstationaryDWRConfig
from .problem import TransientDWRProblem
from .time import IrksomeDGSolver, evaluate_goal, evaluate_goal_components
from .transfer import SlabInterfaceTransfer, build_slab_interface_transfer


class BubbleProjectionAdaptiveSolver:
    """Run ``SOLVE -> ESTIMATE -> MARK -> REFINE`` for one transient problem.

    Its user-facing shape follows Firedrake's adaptive variational solver:

    .. code-block:: python

       solver = BubbleProjectionAdaptiveSolver(
           problem, mesh, time_grid,
           solver_parameters={
               "goal_adaptive": {"tolerance": 1.0e-5, "theta_spacetime": 0.3},
               "ksp_type": "preonly", "pc_type": "lu",
           },
           post_iteration_callback=my_callback,
       )
       solver.solve()

    The nested ``goal_adaptive`` dictionary configures this outer loop; every
    other key is forwarded unchanged to primal and adjoint Irksome solves.
    Bubble-projection options may also be supplied directly as ``options``.
    """

    localisation_method = "weak_bubble_per_facet_cone_slabwise"
    options_heading = "Bubble projection options:"

    def __init__(
        self,
        problem: TransientDWRProblem,
        base_mesh: Mesh,
        time_grid: np.ndarray | list[float],
        *,
        options: BubbleProjectionOptions | dict[str, Any] | None = None,
        solver_parameters: dict[str, Any] | None = None,
        post_iteration_callback: (
            Callable[["BubbleProjectionAdaptiveSolver", int], None] | None
        ) = None,
    ):
        self.problem = problem
        self.options = self._make_options(options, solver_parameters)
        if self.options.localisation_mode == "strong_residual_bound":
            if not problem.supports_strong_residual_bound:
                raise ValueError(
                    "strong_residual_bound requires strong_residual= and "
                    "normal_flux= problem inputs."
                )
            self.localisation_method = "strong_residual_inequality_slabwise"
        elif self.options.localisation_mode in {
            "joint_cell_partition",
            "weak_cell_partition",
        }:
            self.localisation_method = "joint_cell_partition_dg0_slabwise"
        if base_mesh.comm.size != 1:
            raise NotImplementedError(
                "The first global space--time marker is serial; do not use mpiexec."
            )
        self.ts = np.asarray(time_grid, dtype=float)
        if self.ts.ndim != 1 or len(self.ts) < 2 or np.any(np.diff(self.ts) <= 0.0):
            raise ValueError(
                "time_grid must be a strictly increasing one-dimensional array."
            )
        self.final_time = float(self.ts[-1])
        if self.options.snapshot_times is not None and any(
            value < float(self.ts[0]) or value > self.final_time
            for value in self.options.snapshot_times
        ):
            raise ValueError(
                "Every snapshot time must lie inside the supplied time grid "
                f"[{self.ts[0]}, {self.final_time}]."
            )
        # ``meshes[n]`` is the spatial mesh T_n used solely on I_n.  The
        # legacy ``mesh`` attribute remains the terminal slab mesh for simple
        # callbacks which only inspect u_h(T).
        self.meshes: list[Mesh | None] = [None] + [
            base_mesh for _ in range(len(self.ts) - 1)
        ]
        self.mesh = base_mesh
        self.post_iteration_callback = post_iteration_callback
        self.time_solver = IrksomeDGSolver(problem, self.options.solver_parameters)
        self.output = AdaptiveOutput(
            self.options.output_prefix,
            self.options.write_vtk,
            self.options.vtk_output_mode,
            self.options.snapshot_times,
        )
        self.history: list[dict[str, Any]] = []
        self.last_primal: dict[str, Any] | None = None
        self.last_primal_enriched: dict[str, Any] | None = None
        self.last_dual_enriched: dict[str, Any] | None = None
        self.last_dual_low: dict[str, Any] | None = None
        self.last_estimate: BubbleEstimate | None = None
        self.last_goal_components: dict[str, Any] | None = None

    @staticmethod
    def _make_options(
        options: BubbleProjectionOptions | dict[str, Any] | None,
        solver_parameters: dict[str, Any] | None,
    ) -> BubbleProjectionOptions:
        """Split Firedrake-style unified parameters into outer and PETSc data."""
        if isinstance(options, BubbleProjectionOptions):
            if solver_parameters is not None:
                raise ValueError("Pass either options or solver_parameters, not both.")
            return options
        if options is not None:
            return BubbleProjectionOptions(**options)
        unified = dict(solver_parameters or {})
        adaptive = dict(unified.pop("goal_adaptive", {}))
        adaptive["solver_parameters"] = unified or None
        return BubbleProjectionOptions(**adaptive)

    @classmethod
    def uniform_time_grid(cls, final_time: float, nslabs: int) -> np.ndarray:
        """Create ``0=t_0<...<t_N=T`` for the common uniform-grid input."""
        if nslabs < 1:
            raise ValueError("nslabs must be at least one.")
        return np.linspace(0.0, float(final_time), int(nslabs) + 1)

    def _mesh_size_field_and_stats(
        self, mesh: Mesh, DG0
    ) -> tuple[Function, dict[str, float]]:
        """Return local ``h_K`` data for one slab mesh."""
        h_K = Function(DG0, name="h_K")
        h_K.interpolate(CellDiameter(mesh))
        values = np.asarray(h_K.dat.data_ro, dtype=float)
        return h_K, {
            "n_cells": float(values.size),
            "h_min": float(np.min(values)),
            "h_median": float(np.median(values)),
            "h_max": float(np.max(values)),
        }

    def _solve_and_estimate(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], BubbleEstimate]:
        """Solve on slabwise meshes, including assembled ``P`` and ``P_star``."""
        nslabs = len(self.ts) - 1
        V_by_slab: list[Any | None] = [None] + [
            FunctionSpace(self.meshes[n], "CG", self.options.spatial_degree)
            for n in range(1, nslabs + 1)
        ]
        V_dual_by_slab: list[Any | None] = [None] + [
            FunctionSpace(self.meshes[n], "CG", self.options.dual_spatial_degree)
            for n in range(1, nslabs + 1)
        ]
        transfers: list[SlabInterfaceTransfer | None] = [None] * (nslabs + 1)
        enriched_transfers: list[SlabInterfaceTransfer | None] = [None] * (nslabs + 1)
        for n in range(1, nslabs):
            transfers[n] = build_slab_interface_transfer(
                self.problem, V_by_slab[n], V_by_slab[n + 1]
            )
            enriched_transfers[n] = build_slab_interface_transfer(
                self.problem, V_dual_by_slab[n], V_dual_by_slab[n + 1]
            )
        primal = self.time_solver.solve_primal(
            V_by_slab, self.meshes, self.ts, transfers, self.options.primal_time_degree
        )
        primal_enriched = None
        if self.options.nonlinear_error_identity:
            if not self.problem.supports_nonlinear_error_identity:
                raise ValueError(
                    "nonlinear_error_identity=True requires problem-specific residual "
                    "and goal derivative actions."
                )
            primal_enriched = self.time_solver.solve_primal(
                V_dual_by_slab,
                self.meshes,
                self.ts,
                enriched_transfers,
                self.options.dual_time_degree,
            )
        if self.problem.has_goal_components:
            current_values = evaluate_goal_components(
                self.problem,
                primal,
                self.ts,
                self.options.recovery_quadrature_points,
            )
            exact_values = self.problem.goal_component_exact_values(self.final_time)
            if any(value is None for value in exact_values):
                reference_primal = primal_enriched if primal_enriched is not None else primal
                reference_values = evaluate_goal_components(
                    self.problem,
                    reference_primal,
                    self.ts,
                    self.options.recovery_quadrature_points,
                )
            else:
                reference_values = [float(value) for value in exact_values]
            weights = self.problem.update_signed_relative_goal(
                current_values, reference_values
            )
            self.last_goal_components = {
                "current": current_values,
                "reference": reference_values,
                "weights": weights,
            }
        dual_enriched = self.time_solver.solve_terminal_adjoint(
            V_dual_by_slab,
            self.meshes,
            self.ts,
            enriched_transfers,
            self.options.dual_time_degree,
            primal=primal_enriched if primal_enriched is not None else primal,
        )
        dual_low = self.time_solver.solve_terminal_adjoint(
            V_by_slab,
            self.meshes,
            self.ts,
            transfers,
            self.options.primal_time_degree,
            primal=primal,
        )
        if self.options.localisation_mode == "strong_residual_bound":
            estimate = estimate_dwr_by_strong_residual_bound(
                primal,
                dual_enriched,
                dual_low,
                self.ts,
                self.problem,
                self.options,
                primal_enriched=primal_enriched,
            )
        elif self.options.localisation_mode in {
            "joint_cell_partition",
            "weak_cell_partition",
        }:
            estimate = estimate_dwr_by_joint_cell_partition(
                primal,
                dual_enriched,
                dual_low,
                self.ts,
                self.problem,
                self.options,
                primal_enriched=primal_enriched,
            )
        else:
            estimate = estimate_dwr_by_bubble_projection(
                primal,
                dual_enriched,
                dual_low,
                self.ts,
                self.problem,
                self.options,
                primal_enriched=primal_enriched,
            )
        self.last_primal_enriched = primal_enriched
        return primal, dual_enriched, dual_low, estimate

    def step(self, iteration: int) -> None:
        """Execute exactly one adaptive cycle and raise ``StopIteration`` on completion.

        Refinement is performed only after output is written, ensuring a VTK
        dataset and CSV row always refer to the identical mesh/time grid used
        for the reported DWR estimate.
        """
        primal, dual_enriched, dual_low, estimate = self._solve_and_estimate()
        self.last_primal, self.last_dual_enriched, self.last_dual_low = (
            primal,
            dual_enriched,
            dual_low,
        )
        self.last_estimate = estimate
        nslabs = len(self.ts) - 1
        terminal_mesh = self.meshes[nslabs]
        terminal_state = primal["nodes"][-1]
        goal_h = evaluate_goal(
            self.problem, primal, self.ts, self.options.recovery_quadrature_points
        )
        initial_goal = (
            float(assemble(self.problem.terminal_goal_form(
                self.meshes[1], primal["nodes"][0]
            )))
            if self.problem.has_terminal_goal else 0.0
        )
        exact_goal = self.problem.exact_goal_value(self.final_time)
        true_error = None if exact_goal is None else float(exact_goal - goal_h)
        threshold_reached = abs(estimate.eta_global) <= self.options.tolerance
        closure_scale = max(abs(estimate.eta_global), np.finfo(float).eps)
        primal_closure_relative = abs(estimate.primal_closure_gap) / max(
            abs(estimate.eta_primal_global), closure_scale
        )
        adjoint_closure_relative = abs(estimate.adjoint_closure_gap) / max(
            abs(estimate.eta_adjoint_global), closure_scale
        )
        closure_activity = sum(
            np.abs(np.asarray(values, dtype=float)).sum()
            for values in estimate.eta_cell_slab_signed[1:]
        )
        localisation_gap_over_activity = abs(estimate.localisation_gap) / max(
            closure_activity, np.finfo(float).eps
        )
        closure_acceptable = (
            self.options.localisation_mode != "hierarchical_recovery"
            or not estimate.nonlinear_identity
            or localisation_gap_over_activity
            <= self.options.localisation_closure_tolerance
        )
        if not closure_acceptable:
            raise RuntimeError(
                "Bubble localisation closure failed before marking: "
                f"gap/activity={localisation_gap_over_activity:.3e}, "
                f"gap/global={estimate.localisation_gap_relative:.3e}, "
                f"primal={primal_closure_relative:.3e}, "
                f"adjoint={adjoint_closure_relative:.3e}, "
                f"limit={self.options.localisation_closure_tolerance:.3e}."
            )
        final_iteration = iteration == self.options.max_it - 1
        should_stop = threshold_reached or final_iteration

        # Marks remain vectors in the individual cell order of T_n.  Unlike
        # the former shared-mesh code, there is intentionally no spatial union.
        marks: list[np.ndarray | None] = [None] + [
            np.zeros(FunctionSpace(self.meshes[n], "DG", 0).node_count, dtype=bool)
            for n in range(1, nslabs + 1)
        ]
        time_marked: set[int] = set()
        slab_fractions = [0.0] * (nslabs + 1)
        if not should_stop:
            strategy = self.options.marking_strategy
            diagnostic_component = self.options.diagnostic_marking_component
            if diagnostic_component == "primal":
                marking_values = [None] + [
                    0.5 * np.asarray(values, dtype=float)
                    for values in estimate.eta_primal_cell[1:]
                ]
            elif diagnostic_component == "adjoint":
                marking_values = [None] + [
                    0.5 * np.asarray(values, dtype=float)
                    for values in estimate.eta_adjoint_cell[1:]
                ]
            else:
                marking_values = estimate.eta_cell_slab_marking
            diagnostic_slab_activity = [0.0] + [
                float(np.abs(values).sum()) for values in marking_values[1:]
            ]
            if strategy == "uniform":
                marks = [None] + [
                    np.ones_like(values, dtype=bool)
                    for values in estimate.eta_cell_slab_signed[1:]
                ]
                selected = set(range(1, nslabs + 1))
                _, slab_fractions = marked_slab_fractions(marks, 2.0)
            elif strategy == "global_bulk_fraction_trigger":
                marks = mark_spacetime_cells(
                    marking_values, self.options.theta_spacetime
                )
                if self.options.time_trigger_policy == "global_marked_share":
                    selected, slab_fractions = marked_slab_global_shares(
                        marks, self.options.time_slab_marked_fraction
                    )
                else:
                    selected, slab_fractions = marked_slab_fractions(
                        marks, self.options.time_slab_marked_fraction
                    )
            else:
                fixed_rate = strategy == "fixed_rate"
                marks = mark_each_slab(
                    marking_values,
                    self.options.theta_spacetime,
                    fixed_rate=fixed_rate,
                )
                selected = mark_slabs_by_activity(
                    diagnostic_slab_activity,
                    self.options.time_slab_marked_fraction,
                    fixed_rate=fixed_rate,
                )
                _, slab_fractions = marked_slab_fractions(marks, 2.0)
            if self.options.enable_time_refinement:
                time_marked = selected

        n_spacetime = int(sum(np.count_nonzero(mask) for mask in marks[1:]))
        n_time = len(time_marked)
        shared_space_marks: np.ndarray | None = None
        if self.options.space_refinement_strategy == "shared_time_mesh":
            # This diagnostic/stable mode deliberately keeps one spatial mesh
            # for every time slab.  Hence a forward interface is an identity
            # embedding, never the information-losing fine-to-coarse transfer
            # observed with independent slab-local meshes.
            common_mesh = self.meshes[1]
            if any(self.meshes[n] is not common_mesh for n in range(1, nslabs + 1)):
                raise RuntimeError(
                    "shared_time_mesh requires all current slabs to share one mesh. "
                    "Start a new adaptive run with this strategy rather than switching mid-run."
                )
            shared_space_marks = np.zeros_like(marks[1], dtype=bool)
            for slab_marks in marks[1:]:
                shared_space_marks |= slab_marks
        n_space_cells_marked = (
            int(np.count_nonzero(shared_space_marks))
            if shared_space_marks is not None
            else n_spacetime
        )
        all_h: list[np.ndarray] = []
        total_cells = 0
        primal_dofs = 0
        for n in range(1, nslabs + 1):
            DG0 = FunctionSpace(self.meshes[n], "DG", 0)
            _, slab_stats = self._mesh_size_field_and_stats(self.meshes[n], DG0)
            h = Function(DG0)
            h.interpolate(CellDiameter(self.meshes[n]))
            all_h.append(np.asarray(h.dat.data_ro, dtype=float))
            total_cells += DG0.node_count
            primal_dofs += FunctionSpace(
                self.meshes[n], "CG", self.options.spatial_degree
            ).dim()
        h_values = np.concatenate(all_h)
        mesh_stats = {
            "h_min": float(h_values.min()),
            "h_median": float(np.median(h_values)),
            "h_max": float(h_values.max()),
            "n_cells": float(total_cells),
        }
        terminal_V = FunctionSpace(terminal_mesh, "CG", self.options.spatial_degree)
        denominator = abs(true_error) if true_error is not None else None
        row = {
            "iteration": iteration,
            "method": self.localisation_method,
            "spatial_dofs_terminal": terminal_V.dim(),
            "n_time_slabs": nslabs,
            "primal_spacetime_dofs": primal_dofs,
            "recovery_unknowns_proxy": estimate.recovery_unknowns_proxy,
            "recovery_space_degree": self.options.recovery_space_degree,
            "recovery_facet_degree": self.options.facet_recovery_degree,
            "recovery_time_degree": self.options.recovery_time_degree,
            "recovery_quadrature_points": self.options.recovery_quadrature_points,
            "n_cells_total": total_cells,
            "goal_label": getattr(self.problem, "goal_label", "terminal_qoi"),
            "marking_strategy": self.options.marking_strategy,
            "time_trigger_policy": self.options.time_trigger_policy,
            "dual_weight_mode": self.options.dual_weight_mode,
            "localisation_mode": (
                "joint_cell_partition"
                if self.options.localisation_mode == "weak_cell_partition"
                else self.options.localisation_mode
            ),
            "bubble_marking_score": self.options.bubble_marking_score,
            "diagnostic_marking_component": self.options.diagnostic_marking_component,
            "include_mixed_ridge": self.options.include_mixed_ridge,
            "include_galerkin_correction": self.options.include_galerkin_correction,
            "indicator_semantics": estimate.indicator_semantics,
            "theta_space_or_global": self.options.theta_spacetime,
            "time_marking_parameter": self.options.time_slab_marked_fraction,
            "J_exact": exact_goal,
            "J_h": goal_h,
            "J_initial": initial_goal,
            "J_change": goal_h - initial_goal,
            "true_goal_error": true_error,
            "eta_global": estimate.eta_global,
            "nonlinear_identity": estimate.nonlinear_identity,
            "eta_primal_residual": estimate.eta_primal_global,
            "eta_adjoint_residual": estimate.eta_adjoint_global,
            "eta_galerkin_correction": estimate.eta_correction_global,
            "eta_primal_recovered": estimate.eta_primal_local,
            "eta_adjoint_recovered": estimate.eta_adjoint_local,
            "eta_correction_recovered": estimate.eta_correction_local,
            "primal_closure_gap": estimate.primal_closure_gap,
            "adjoint_closure_gap": estimate.adjoint_closure_gap,
            "correction_closure_gap": estimate.correction_closure_gap,
            "primal_closure_relative": primal_closure_relative,
            "adjoint_closure_relative": adjoint_closure_relative,
            "localisation_closure_acceptable": closure_acceptable,
            "J_enriched_minus_J_h": estimate.enriched_goal_difference,
            "R3_observed_enriched": estimate.observed_remainder,
            "eta_local_sum": estimate.eta_local_sum,
            "eta_marking_sum": estimate.eta_marking_sum,
            "diagnostic_marking_activity": (
                estimate.eta_marking_sum
                if self.options.diagnostic_marking_component == "total"
                else float(sum(
                    np.abs(np.asarray(values, dtype=float)).sum()
                    for values in (
                        estimate.eta_primal_cell[1:]
                        if self.options.diagnostic_marking_component == "primal"
                        else estimate.eta_adjoint_cell[1:]
                    )
                )) * 0.5
            ),
            "eta_signed_activity_sum": float(sum(
                np.abs(np.asarray(values, dtype=float)).sum()
                for values in estimate.eta_cell_slab_signed[1:]
            )),
            "localisation_gap": estimate.localisation_gap,
            "localisation_gap_relative": estimate.localisation_gap_relative,
            "localisation_gap_over_activity": (
                float("nan")
                if self.options.localisation_mode == "strong_residual_bound"
                else localisation_gap_over_activity
            ),
            "localisation_consistency_index": estimate.localisation_consistency_index,
            "eta_weak_cell_sum": estimate.eta_weak_cell_sum,
            "weak_cell_closure_gap": estimate.weak_cell_closure_gap,
            "hierarchical_minus_weak_activity": estimate.hierarchical_minus_weak_activity,
            "effectivity_global": (
                estimate.eta_global / true_error
                if denominator and denominator > 1.0e-15
                else float("nan")
            ),
            "effectivity_localised": (
                float("nan")
                if self.options.localisation_mode == "strong_residual_bound"
                else estimate.eta_local_sum / true_error
                if denominator and denominator > 1.0e-15
                else float("nan")
            ),
            "indicator_effectivity": (
                estimate.eta_marking_sum / denominator
                if denominator and denominator > 1.0e-15
                else float("nan")
            ),
            "target_tolerance": self.options.tolerance,
            "threshold_reached": threshold_reached,
            "n_spacetime_marked": n_spacetime,
            "n_space_cells_marked": n_space_cells_marked,
            "n_time_slabs_marked": n_time,
        }
        if self.last_goal_components is not None:
            for component, current, reference, weight in zip(
                self.problem.goal_components,
                self.last_goal_components["current"],
                self.last_goal_components["reference"],
                self.last_goal_components["weights"],
            ):
                label = component.label
                row[f"goal_component_{label}_reference"] = reference
                row[f"goal_component_{label}_h"] = current
                row[f"goal_component_{label}_error"] = reference - current
                row[f"goal_component_{label}_weight"] = weight
        row.update(self.problem.goal_diagnostics(
            goal_h,
            estimate.eta_global,
            true_error,
            symmetric_identity=estimate.nonlinear_identity,
        ))
        self.history.append(row)
        self._print_iteration(row, mesh_stats, slab_fractions)
        self.output.write_vtk(iteration, terminal_state, self.final_time)
        self.output.write_slabwise_vtk(iteration, primal, estimate, self.ts)
        self.output.write_slab_metadata(
            iteration,
            self.ts,
            estimate,
            marks,
            time_marked,
        )
        self.output.write_spacetime_vtk(iteration, primal, estimate, self.ts)
        self.output.write_history(self.history)
        if self.post_iteration_callback is not None:
            self.post_iteration_callback(self, iteration)
        if should_stop:
            if threshold_reached:
                self._print("Error estimate below tolerance; finished.")
            elif final_iteration:
                self._print(
                    f"Maximum iteration ({self.options.max_it}) reached; finished."
                )
            raise StopIteration

        # ``independent_slab`` is the original genuinely slab-local policy.
        # ``causal_nested`` retains local meshes but propagates a marked parent
        # region forward in time.  ``shared_time_mesh`` is the stronger common
        # mesh comparison.  If a time interval is split, both children retain
        # their parent's newly refined mesh.
        next_times = [float(self.ts[0])]
        next_meshes: list[Mesh | None] = [None]
        if self.options.space_refinement_strategy == "shared_time_mesh":
            common_mesh = self.meshes[1]
            refined_mesh = common_mesh
            if self.options.enable_space_refinement and np.any(shared_space_marks):
                DG0 = FunctionSpace(common_mesh, "DG", 0)
                markers = Function(DG0, name="space_markers_shared_time_mesh")
                markers.dat.data[:] = np.asarray(
                    shared_space_marks, dtype=markers.dat.data.dtype
                )
                if self.problem.spatial_refinement_mode in {
                    "uniform_slab",
                    "local_interval",
                    "local_periodic",
                }:
                    refined_mesh = self.problem.refine_slab_mesh(common_mesh, markers)
                else:
                    refined_mesh = refine_marked_mesh(common_mesh, markers)
            for n in range(1, nslabs + 1):
                if n in time_marked:
                    next_times.append(0.5 * (float(self.ts[n - 1]) + float(self.ts[n])))
                    next_meshes.append(refined_mesh)
                next_times.append(float(self.ts[n]))
                next_meshes.append(refined_mesh)
        elif self.options.space_refinement_strategy == "causal_nested":
            causal_marks: list[np.ndarray | None] = [None] * (nslabs + 1)
            causal_meshes: list[Mesh | None] = [None] * (nslabs + 1)
            for n in range(1, nslabs + 1):
                inherited = (
                    np.zeros_like(marks[n], dtype=bool)
                    if n == 1
                    else inherit_refinement_marks(
                        self.meshes[n - 1], causal_marks[n - 1], self.meshes[n]
                    )
                )
                # C_n = M_n union inherit_(n-1 to n)(C_(n-1)).  Thus an
                # earlier marked parent region cannot be coarsened away by a
                # subsequent slab's independent marking decision.
                causal_marks[n] = np.logical_or(marks[n], inherited)
                mesh = self.meshes[n]
                causal_meshes[n] = mesh
                if self.options.enable_space_refinement and np.any(causal_marks[n]):
                    DG0 = FunctionSpace(mesh, "DG", 0)
                    markers = Function(DG0, name=f"causal_space_markers_slab_{n}")
                    markers.dat.data[:] = np.asarray(
                        causal_marks[n], dtype=markers.dat.data.dtype
                    )
                    if self.problem.spatial_refinement_mode in {
                        "uniform_slab",
                        "local_interval",
                        "local_periodic",
                    }:
                        causal_meshes[n] = self.problem.refine_slab_mesh(mesh, markers)
                    else:
                        causal_meshes[n] = refine_marked_mesh(mesh, markers)
            for n in range(1, nslabs + 1):
                if n in time_marked:
                    next_times.append(0.5 * (float(self.ts[n - 1]) + float(self.ts[n])))
                    next_meshes.append(causal_meshes[n])
                next_times.append(float(self.ts[n]))
                next_meshes.append(causal_meshes[n])
        else:
            for n in range(1, nslabs + 1):
                mesh = self.meshes[n]
                if self.options.enable_space_refinement and np.any(marks[n]):
                    DG0 = FunctionSpace(mesh, "DG", 0)
                    markers = Function(DG0, name=f"space_markers_slab_{n}")
                    markers.dat.data[:] = np.asarray(
                        marks[n], dtype=markers.dat.data.dtype
                    )
                    if self.problem.spatial_refinement_mode in {
                        "uniform_slab",
                        "local_interval",
                        "local_periodic",
                    }:
                        refined_mesh = self.problem.refine_slab_mesh(mesh, markers)
                    else:
                        refined_mesh = refine_marked_mesh(mesh, markers)
                else:
                    refined_mesh = mesh
                if n in time_marked:
                    next_times.append(0.5 * (float(self.ts[n - 1]) + float(self.ts[n])))
                    next_meshes.append(refined_mesh)
                next_times.append(float(self.ts[n]))
                next_meshes.append(refined_mesh)
        self.ts = np.asarray(next_times, dtype=float)
        self.meshes = next_meshes
        self.mesh = self.meshes[-1]
        if time_marked:
            self._print(
                f"[Time refinement] bisected slabs {sorted(time_marked)}; Nt={len(self.ts) - 1}"
            )
        if n_spacetime:
            if self.options.space_refinement_strategy == "shared_time_mesh":
                self._print(
                    "[Shared-time spatial refinement] refined "
                    f"{n_space_cells_marked} union cells from {n_spacetime} "
                    "marked space-time cells; all slabs inherit the same mesh"
                )
                return
            if self.options.space_refinement_strategy == "causal_nested":
                self._print(
                    "[Causal nested spatial refinement] propagated each marked "
                    "parent region to all later slabs before their local refinement"
                )
                return
            if self.problem.spatial_refinement_mode == "uniform_slab":
                self._print(
                    "[Slabwise periodic refinement] uniformly refined the "
                    "slabs containing the marked space-time cells"
                )
            elif self.problem.spatial_refinement_mode == "local_interval":
                self._print(
                    "[Slabwise local interval refinement] bisected only the "
                    "marked spatial cells on their own slabs"
                )
            elif self.problem.spatial_refinement_mode == "local_periodic":
                self._print(
                    "[Slabwise local periodic refinement] bisected only the "
                    "marked cells while retaining the periodic seam"
                )
            else:
                self._print(
                    f"[Slabwise space refinement] bisected {n_spacetime} marked space-time cells on their own slabs"
                )

    def solve(self) -> "BubbleProjectionAdaptiveSolver":
        """Run the full adaptive loop, returning ``self`` for post-processing."""
        self._print(self.options_heading)
        self._print(asdict(self.options))
        for iteration in range(self.options.max_it):
            try:
                self.step(iteration)
            except StopIteration:
                break
        if self.options.vtk_output_mode == "all":
            self.output.write_iteration_collection(len(self.history), self.final_time)
        self.output.write_spacetime_collection(len(self.history), self.final_time)
        self.output.write_history(self.history)
        self._print_summary()
        return self

    def _print(self, *args) -> None:
        """Respect the outer-loop verbosity flag for all diagnostic output."""
        if self.options.verbose:
            PETSc.Sys.Print(*args)

    def _print_iteration(
        self, row: dict[str, Any], stats: dict[str, float], fractions: list[float]
    ) -> None:
        """Report estimator consistency, effectivity, and refinement decisions."""
        if self.options.localisation_mode == "strong_residual_bound":
            self._print_strong_iteration(row, stats, fractions)
            return
        iteration_label = (
            "DG0 JOINT CELL"
            if self.options.localisation_mode in {
                "joint_cell_partition",
                "weak_cell_partition",
            }
            else "BUBBLE PROJECTION"
        )
        self._print(
            f"\n---- [{iteration_label} ITER {row['iteration']}] "
            f"terminalDOFs={row['spatial_dofs_terminal']} | "
            f"space-time DOFs={row['primal_spacetime_dofs']} | "
            f"Nt={row['n_time_slabs']} | slabwise cells={row['n_cells_total']} ----"
        )
        self._print("  primal / low dual                = CG(p) x DG(r)")
        self._print(
            f"  enriched dual                    = CG({self.options.dual_spatial_degree}) x DG{self.options.dual_time_degree}"
        )
        if self.options.dual_weight_mode == "enriched_minus_numerical":
            self._print("  DWR weight                       = z_enriched - z_numerical")
        else:
            self._print(
                "  DWR weight                       = z_enriched - I_h(z_enriched)"
            )
        self._print(f"  J_h                              = {row['J_h']:+.6e}")
        if row["goal_label"] == "I2":
            self._print(f"  I2(u_h(0))                       = {row['J_initial']:+.6e}")
            self._print(f"  I2(u_h(T))-I2(u_h(0))            = {row['J_change']:+.6e}")
        if row["J_exact"] is not None:
            self._print(f"  J_exact                          = {row['J_exact']:+.6e}")
            self._print(
                f"  true goal error                  = {row['true_goal_error']:+.6e}"
            )
            self._print(
                f"  signed global effectivity I_eff  = "
                f"{row['effectivity_global']:8.4f}"
            )
            self._print(
                f"  signed local effectivity         = "
                f"{row['effectivity_localised']:8.4f}"
            )
        self._print(f"  eta_global                       = {row['eta_global']:+.6e}")
        if not self.options.include_mixed_ridge:
            self._print(
                "  mixed ridge                      = OMITTED (diagnostic ablation)"
            )
        if row["nonlinear_identity"]:
            correction_text = (
                f"; -rho(z_h) {-row['eta_galerkin_correction']:+.6e}"
                if row["include_galerkin_correction"]
                else "; Galerkin correction OMITTED"
            )
            self._print(
                "  nonlinear identity components     = "
                f"1/2*rho(e_z) {0.5*row['eta_primal_residual']:+.6e}; "
                f"1/2*rho* (e_u) {0.5*row['eta_adjoint_residual']:+.6e}"
                + correction_text
            )
            if self.options.localisation_mode == "hierarchical_recovery":
                self._print(
                    "  bubble closure criterion          = "
                    f"gap/activity {row['localisation_gap_over_activity']:.3e}; "
                    f"limit {self.options.localisation_closure_tolerance:.3e}; PASS"
                )
                self._print(
                    "  component closure diagnostics     = "
                    f"primal {row['primal_closure_relative']:.3e}; "
                    f"adjoint {row['adjoint_closure_relative']:.3e}"
                )
            self._print(
                f"  J(u_enriched)-J(u_h)             = {row['J_enriched_minus_J_h']:+.6e}"
            )
            remainder_label = (
                "observed R3 (enriched reference)"
                if row["include_galerkin_correction"]
                else "observed omitted remainder"
            )
            self._print(
                f"  {remainder_label:<33}= {row['R3_observed_enriched']:+.6e}"
            )
        self._print(f"  eta_local_sum                    = {row['eta_local_sum']:+.6e}")
        self._print(
            f"  sum |eta_Kn|                     = {row['eta_signed_activity_sum']:.6e}"
        )
        if self.options.diagnostic_marking_component != "total":
            self._print(
                "  DIAGNOSTIC marking component      = "
                f"{self.options.diagnostic_marking_component}; "
                f"activity {row['diagnostic_marking_activity']:.6e}"
            )
        if self.options.bubble_marking_score == "componentwise_abs":
            self._print(
                "  marking sum_j |eta_j|             = "
                f"{row['eta_marking_sum']:.6e}"
            )
        self._print(
            f"  localisation gap                 = {row['localisation_gap']:+.3e} (relative {row['localisation_gap_relative']:.3e})"
        )
        self._print(
            f"  marking                          = {row['n_spacetime_marked']} space-time; {row['n_space_cells_marked']} slab-local spatial; {row['n_time_slabs_marked']} temporal"
        )
        active = {
            n: fraction for n, fraction in enumerate(fractions) if n and fraction > 0.0
        }
        if active:
            label = (
                "share of all marked cells by slab"
                if self.options.marking_strategy == "global_bulk_fraction_trigger"
                and self.options.time_trigger_policy == "global_marked_share"
                else "marked cell fraction by slab"
            )
            self._print(f"  {label:<34}= {active}")
        self._print(
            f"  mesh h_K                         = min {stats['h_min']:.3e}, median {stats['h_median']:.3e}, max {stats['h_max']:.3e}"
        )

    def _print_strong_iteration(
        self, row: dict[str, Any], stats: dict[str, float], fractions: list[float]
    ) -> None:
        """Label a positive strong bound without calling it a signed PU sum."""
        self._print(
            f"\n---- [STRONG-RESIDUAL ITER {row['iteration']}] "
            f"space-time DOFs={row['primal_spacetime_dofs']} | "
            f"Nt={row['n_time_slabs']} | cells={row['n_cells_total']} ----"
        )
        self._print(f"  J_h                              = {row['J_h']:+.6e}")
        label = "symmetric signed DWR" if row["nonlinear_identity"] else "raw signed DWR"
        self._print(f"  eta_global ({label})       = {row['eta_global']:+.6e}")
        self._print(f"  sum strong m_Kn (positive)       = {row['eta_marking_sum']:.6e}")
        self._print(f"  weak DG0 closure gap             = {row['weak_cell_closure_gap']:+.3e}")
        if row["J_exact"] is not None:
            self._print(f"  signed global effectivity        = {row['effectivity_global']:8.4f}")
            self._print(f"  bound indicator effectivity      = {row['indicator_effectivity']:8.4f}")
        self._print(
            f"  marking                          = {row['n_spacetime_marked']} "
            f"space-time; {row['n_time_slabs_marked']} temporal"
        )
        self._print(
            f"  mesh h_K                         = min {stats['h_min']:.3e}, "
            f"median {stats['h_median']:.3e}, max {stats['h_max']:.3e}"
        )

    def _print_summary(self) -> None:
        """Print concise final convergence history and output locations."""
        self._print("\n==== Nonstationary DWR comparison summary ====")
        for row in self.history:
            effectivity = (
                f" Ieff={row['effectivity_global']:8.4f}"
                if row["J_exact"] is not None
                else ""
            )
            self._print(
                f"  it={row['iteration']:2d} terminalDOFs={row['spatial_dofs_terminal']:6d} "
                f"Nt={row['n_time_slabs']:4d} eta_global={row['eta_global']:+.6e} "
                f"eta_local={row['eta_local_sum']:+.6e}{effectivity}"
            )
        self._print(f"CSV history: {self.options.output_prefix}_history.csv")
        if self.options.write_vtk and not self.output.vtk_failed:
            candidates = []
            if self.options.snapshot_times is not None:
                candidates.extend(
                    Path(f"{self.options.output_prefix}_iter_{iteration}_snapshots.pvd")
                    for iteration in range(len(self.history))
                )
            candidates.extend([
                Path(f"{self.options.output_prefix}_iterations.pvd"),
                Path(f"{self.options.output_prefix}_spacetime_iterations.pvd"),
            ])
            for path in candidates:
                if path.exists():
                    self._print(f"ParaView collection: {path}")


class NonstationaryDWRSolver:
    """Stationary-style public interface for the full nonstationary loop."""

    def __init__(self, problem: TransientDWRProblem, config: NonstationaryDWRConfig):
        self.problem = problem
        self.config = config
        self.problem.set_final_time(config.final_time)
        mesh = problem.make_mesh(config.nx, config.ny)
        options = BubbleProjectionOptions(
            max_it=config.max_it,
            tolerance=config.tolerance,
            spatial_degree=config.spatial_degree,
            primal_time_degree=config.primal_time_degree,
            dual_extra_spatial_degree=config.dual_extra_spatial_degree,
            dual_time_degree=config.dual_time_degree,
            recovery_space_degree=config.recovery_space_degree,
            facet_recovery_degree=config.recovery_facet_degree,
            recovery_time_degree=config.recovery_time_degree,
            recovery_quadrature_points=config.quadrature_points,
            nonlinear_error_identity=config.nonlinear_identity,
            include_galerkin_correction=config.include_galerkin_correction,
            dual_weight_mode=config.dual_weight_mode,
            localisation_mode=config.localisation_mode,
            bubble_marking_score=config.bubble_marking_score,
            diagnostic_marking_component=config.diagnostic_marking_component,
            nonlinear_adjoint_localisation=config.nonlinear_adjoint_localisation,
            include_mixed_ridge=config.include_mixed_ridge,
            theta_spacetime=config.theta,
            time_slab_marked_fraction=config.time_fraction,
            marking_strategy=config.marking_strategy,
            time_trigger_policy=config.time_trigger_policy,
            space_refinement_strategy=config.space_refinement_strategy,
            write_vtk=config.write_vtk,
            vtk_output_mode=config.vtk_output_mode,
            snapshot_times=config.snapshot_times,
            output_prefix=config.output_prefix,
            solver_parameters=problem.solver_parameters,
        )
        if config.nonlinear_identity and not problem.supports_nonlinear_error_identity:
            raise ValueError(
                "nonlinear_identity=True requires nonlinear_identity=True in the problem."
            )
        self.adaptive_solver = BubbleProjectionAdaptiveSolver(
            problem,
            mesh,
            self.uniform_time_grid(config.final_time, config.nt),
            options=options,
        )

    @staticmethod
    def uniform_time_grid(final_time: float, nslabs: int) -> np.ndarray:
        if nslabs < 1:
            raise ValueError("nt must be at least one.")
        return np.linspace(0.0, float(final_time), int(nslabs) + 1)

    def solve(self) -> BubbleProjectionAdaptiveSolver:
        return self.adaptive_solver.solve()


# Compatibility with the name used by the first prototype.
NonstationaryDWRAdaptiveSolver = NonstationaryDWRSolver


__all__ = [
    "BubbleProjectionAdaptiveSolver",
    "NonstationaryDWRAdaptiveSolver",
    "NonstationaryDWRSolver",
]
