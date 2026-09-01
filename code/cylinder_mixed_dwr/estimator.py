r"""Symmetric nonlinear DWR identity for the mixed cylinder problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    TestFunction,
    assemble,
    div,
    dx,
    grad,
    inner,
    split,
)

from automated_DWR.time_solver import gauss_rule
from navier_stokes_cylinder_irksome_static_primal import evaluate_slab, evaluate_slab_dt

from .benchmark import drag_derivative_form, mean_drag_from_history
from .adapter import CylinderMixedDAEAdapter
from .slabwise import evaluate_temporally_composite_slab


@dataclass
class SymmetricDWREstimate:
    dual_weight_mode: str
    eta_global: float
    eta_primal_residual: float
    eta_adjoint_residual: float
    eta_galerkin_correction: float
    eta_symmetric_core: float
    eta_cubic_remainder: float
    eta_cell_weak: list[np.ndarray | None]
    eta_primal_cell_weak: list[np.ndarray | None]
    eta_adjoint_cell_weak: list[np.ndarray | None]
    eta_correction_cell_weak: list[np.ndarray | None]
    eta_cubic_remainder_cell_weak: list[np.ndarray | None]
    weak_closure_gap: float
    enriched_goal_difference: float
    observed_cubic_remainder: float
    enriched_identity_gap: float
    enriched_primal_stationarity_defect: float
    enriched_adjoint_stationarity_defect: float
    enriched_primal_stationarity_volume_by_slab: list[float]
    enriched_primal_stationarity_jump_by_slab: list[float]
    enriched_adjoint_stationarity_volume_by_slab: list[float]
    enriched_adjoint_stationarity_jump_by_slab: list[float]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "dual_weight_mode": self.dual_weight_mode,
            "eta_global": self.eta_global,
            "half_primal_residual": 0.5 * self.eta_primal_residual,
            "half_adjoint_residual": 0.5 * self.eta_adjoint_residual,
            "galerkin_correction": self.eta_galerkin_correction,
            "symmetric_core": self.eta_symmetric_core,
            "cubic_remainder": self.eta_cubic_remainder,
            "weak_closure_gap": self.weak_closure_gap,
            "enriched_goal_difference": self.enriched_goal_difference,
            "observed_cubic_remainder": self.observed_cubic_remainder,
            "enriched_identity_gap": self.enriched_identity_gap,
            "enriched_primal_stationarity_defect": (
                self.enriched_primal_stationarity_defect
            ),
            "enriched_adjoint_stationarity_defect": (
                self.enriched_adjoint_stationarity_defect
            ),
        }


@dataclass
class LinearDWRGlobalEstimate:
    """Scalar primal-residual DWR estimator without an enriched primal.

    The compatibility properties intentionally expose the diagnostics used by
    the production history writer.  Quantities that require an enriched
    primal are reported as ``nan`` rather than being inferred from the low
    trajectory.
    """

    dual_weight_mode: str
    eta_global: float
    eta_volume: float
    eta_temporal_jump: float
    eta_volume_by_slab: list[float]
    eta_temporal_jump_by_slab: list[float]

    @property
    def eta_primal_residual(self) -> float:
        return self.eta_global

    @property
    def eta_adjoint_residual(self) -> float:
        return 0.0

    @property
    def eta_galerkin_correction(self) -> float:
        return 0.0

    @property
    def eta_symmetric_core(self) -> float:
        return self.eta_global

    @property
    def eta_cubic_remainder(self) -> float:
        return 0.0

    @property
    def enriched_goal_difference(self) -> float:
        return float("nan")

    @property
    def observed_cubic_remainder(self) -> float:
        return float("nan")

    @property
    def enriched_identity_gap(self) -> float:
        return float("nan")

    @property
    def enriched_primal_stationarity_defect(self) -> float:
        return float("nan")

    @property
    def enriched_adjoint_stationarity_defect(self) -> float:
        return float("nan")

    @property
    def enriched_primal_stationarity_volume_by_slab(self) -> list[float]:
        return [float("nan")] * len(self.eta_volume_by_slab)

    @property
    def enriched_primal_stationarity_jump_by_slab(self) -> list[float]:
        return [float("nan")] * len(self.eta_volume_by_slab)

    @property
    def enriched_adjoint_stationarity_volume_by_slab(self) -> list[float]:
        return [float("nan")] * len(self.eta_volume_by_slab)

    @property
    def enriched_adjoint_stationarity_jump_by_slab(self) -> list[float]:
        return [float("nan")] * len(self.eta_volume_by_slab)

    @property
    def weak_closure_gap(self) -> float:
        return float("nan")


def _lift_mixed(value: Function, target_space, name: str) -> Function:
    lifted = Function(target_space, name=name)
    if value.function_space() == target_space:
        lifted.assign(value)
    else:
        for target, source in zip(lifted.subfunctions, value.subfunctions):
            target.interpolate(source)
    return lifted


def _difference(high: Function, low: Function, name: str) -> Function:
    result = Function(high.function_space(), name=name)
    result.assign(high)
    result -= _lift_mixed(low, high.function_space(), f"{name}_low_lift")
    return result


def dual_error_at(
    rich_slab,
    low_slab,
    point: float,
    *,
    mode: str = "enriched_minus_interpolant",
    essential_labels=(),
    stored_interpolant: bool = False,
) -> Function:
    r"""Return the selected goal-weight sensitivity at physical time ``point``.

    With ``stored_interpolant=True``, the already constructed low-space-time
    comparator is evaluated directly.  In the production strict-linear path
    it is R5's ``I_h I_k Z+`` interpolant: Gauss--Legendre support points for
    ``I_k`` and P2/P1 spatial nodal interpolation for ``I_h`` (without
    adopting R5's estimator splitting or marking strategy).
    """
    rich = evaluate_temporally_composite_slab(
        rich_slab, 1.0 - float(point), name="Z_rich_at_t"
    )
    if mode == "enriched_minus_numerical":
        comparator = evaluate_slab(low_slab, 1.0 - float(point), name="Z_low_at_t")
    elif mode == "enriched_minus_interpolant":
        if stored_interpolant:
            comparator = evaluate_slab(
                low_slab, 1.0 - float(point), name="I_h_Z_rich"
            )
        else:
            comparator = Function(
                low_slab["coeffs"][0].function_space(), name="Pi_h_Z_rich"
            )
            direct = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
            velocity_target, pressure_target = comparator.subfunctions
            velocity_target.project(
                rich.subfunctions[0], solver_parameters=direct
            )
            pressure_target.project(
                rich.subfunctions[1], solver_parameters=direct
            )
            if tuple(essential_labels):
                DirichletBC(
                    comparator.function_space().sub(0),
                    Constant((0.0, 0.0)),
                    tuple(essential_labels),
                ).apply(comparator)
    else:
        raise ValueError(
            "mode must be 'enriched_minus_numerical' or "
            "'enriched_minus_interpolant'."
        )
    return _difference(rich, comparator, "dual_error")


def primal_error_at(rich_slab, low_slab, point: float, step: float):
    high = evaluate_slab(rich_slab, point, name="U_rich_at_t")
    low = evaluate_slab(low_slab, point, name="U_low_at_t")
    error = _difference(high, low, "primal_error")
    high_dt = evaluate_slab_dt(rich_slab, point, step, name="U_rich_dt")
    low_dt = evaluate_slab_dt(low_slab, point, step, name="U_low_dt")
    error_dt = _difference(high_dt, low_dt, "primal_error_dt")
    return error, error_dt


def primal_trace_error(rich_trace: Function, low_trace: Function) -> Function:
    """Return the enriched-minus-low error of a saved dG interface state."""
    return _difference(rich_trace, low_trace, "primal_interface_error")


def _low_incoming_in_rich_operator(
    primal, primal_enriched, slab_number: int
) -> Function:
    r"""Evaluate the low previous trace with the enriched interface operator.

    On a changed slab mesh, ``I(P_h u_h^-)`` and ``P_+(Iu_h^-)`` are not the
    same.  The latter is required when the low trajectory is inserted into
    the enriched DWR operator; their difference is the GEO/interface
    consistency contribution.
    """
    n = int(slab_number)
    target_space = primal_enriched["slabs"][n]["coeffs"][0].function_space()
    incoming = Function(target_space, name=f"Iplus_Pplus_Ulow_interface_{n}")
    incoming.assign(0.0)
    if n == 1:
        lifted = _lift_mixed(
            primal["slabs"][n]["left_trace"], target_space,
            f"Iplus_Ulow_initial_{n}",
        )
        incoming.assign(lifted)
        return incoming

    source_space = primal_enriched["slabs"][n - 1]["coeffs"][0].function_space()
    source = _lift_mixed(
        primal["slabs"][n - 1]["right_trace"], source_space,
        f"Iplus_Ulow_right_{n - 1}",
    )
    transfer = primal_enriched["transfers"][n - 1]
    if transfer is None:
        incoming.assign(_lift_mixed(source, target_space, f"I_Ulow_incoming_{n}"))
    else:
        boundary_value = None
        if getattr(transfer, "divergence_preserving", False):
            adapter = CylinderMixedDAEAdapter(primal["labels"])
            boundary_value = adapter.inflow_value(
                target_space.mesh(), float(primal["times"][n - 1])
            )
        incoming.subfunctions[0].assign(
            transfer.forward(
                source.subfunctions[0],
                boundary_value=boundary_value,
                name=f"Pplus_Iplus_Ulow_velocity_{n}",
            )
        )
    return incoming


def primal_volume_residual_form(
    state, state_dt, test, viscosity, *, cell_weight=1.0, measure=dx
):
    """Return ``rho^V(U_h)(test)=-A^V(U_h)(test)``."""
    velocity, pressure = split(state)
    velocity_dt = split(state_dt)[0]
    test_velocity, test_pressure = split(test)
    return cell_weight * (
        -inner(velocity_dt, test_velocity)
        - inner(grad(velocity) * velocity, test_velocity)
        - viscosity * inner(grad(velocity), grad(test_velocity))
        + pressure * div(test_velocity)
        + div(velocity) * test_pressure
    ) * measure


def primal_jump_residual_form(state_left, previous_left_trace, test, *, cell_weight=1.0, measure=dx):
    jump_velocity = state_left.subfunctions[0] - previous_left_trace.subfunctions[0]
    return -cell_weight * inner(jump_velocity, split(test)[0]) * measure


def estimate_linear_dwr_global(
    primal: dict[str, Any],
    dual_enriched: dict[str, Any],
    dual_low: dict[str, Any],
    *,
    quadrature_points: int = 4,
    dual_weight_mode: str = "enriched_minus_numerical",
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> LinearDWRGlobalEstimate:
    r"""Assemble only the scalar linear DWR estimator.

    This deliberately skips DG0 weak-cell partitions and bubble/cone recovery.
    It is intended for frozen-grid enrichment/saturation audits where the
    decisive question is whether a richer adjoint changes the global
    primal-residual estimator.  It does not alter the production localisation
    or marking path.
    """
    times = primal["times"]
    horizon = float(times[-1] - times[0])
    if horizon <= 0.0:
        raise ValueError("The primal history must have a positive time horizon.")
    quadrature = gauss_rule(int(quadrature_points))
    nslabs = len(times) - 1
    volume_by_slab = [0.0] * (nslabs + 1)
    jump_by_slab = [0.0] * (nslabs + 1)
    essential = (
        tuple(primal["labels"]["inlet"])
        + tuple(primal["labels"]["wall"])
    )

    for n in range(1, nslabs + 1):
        step = float(times[n] - times[n - 1])
        low_slab = primal["slabs"][n]
        low_dual_slab = dual_low["slabs"][n]
        rich_dual_slab = dual_enriched["slabs"][n]
        temporal_factor = int(
            rich_dual_slab.get("temporal_refinement_factor", 1)
        )
        for child in range(temporal_factor):
            for child_point, child_weight in quadrature:
                point = (float(child) + float(child_point)) / temporal_factor
                weight = float(child_weight) / temporal_factor
                state = evaluate_slab(low_slab, point, name="U_low_global_dwr")
                state_dt = evaluate_slab_dt(
                    low_slab, point, step, name="U_low_dt_global_dwr"
                )
                dual_error = dual_error_at(
                    rich_dual_slab,
                    low_dual_slab,
                    point,
                    mode=dual_weight_mode,
                    essential_labels=essential,
                    stored_interpolant=(
                        dual_weight_mode == "enriched_minus_interpolant"
                    ),
                )
                volume_by_slab[n] += float(
                    assemble(
                        float(step)
                        * float(weight)
                        * primal_volume_residual_form(
                            state, state_dt, dual_error, primal["viscosity"]
                        )
                    )
                )

        state_left = evaluate_slab(
            low_slab, 0.0, name="U_low_left_global_dwr"
        )
        dual_error_left = dual_error_at(
            rich_dual_slab,
            low_dual_slab,
            0.0,
            mode=dual_weight_mode,
            essential_labels=essential,
            stored_interpolant=(
                dual_weight_mode == "enriched_minus_interpolant"
            ),
        )
        previous_trace = _low_incoming_in_rich_operator(primal, primal, n)
        jump_by_slab[n] = float(
            assemble(
                primal_jump_residual_form(
                    state_left, previous_trace, dual_error_left
                )
            )
        )
        if progress_callback is not None:
            progress_callback(n, volume_by_slab[n], jump_by_slab[n])

    eta_volume = float(sum(volume_by_slab[1:]))
    eta_jump = float(sum(jump_by_slab[1:]))
    return LinearDWRGlobalEstimate(
        dual_weight_mode=str(dual_weight_mode),
        eta_global=eta_volume + eta_jump,
        eta_volume=eta_volume,
        eta_temporal_jump=eta_jump,
        eta_volume_by_slab=volume_by_slab,
        eta_temporal_jump_by_slab=jump_by_slab,
    )


def adjoint_volume_residual_form(
    state,
    state_dt,
    primal_error,
    primal_error_dt,
    low_dual,
    viscosity,
    cylinder_labels,
    goal_horizon: float,
    *,
    cell_weight=1.0,
    measure=dx,
):
    r"""Return ``rho*(U_h,Z_h)(e_U)=J'(e_U)-A'(e_U,Z_h)``."""
    velocity, _ = split(state)
    error_velocity, error_pressure = split(primal_error)
    error_velocity_dt = split(primal_error_dt)[0]
    dual_velocity, dual_pressure = split(low_dual)
    form = cell_weight * (
        -inner(error_velocity_dt, dual_velocity)
        - inner(
            grad(error_velocity) * velocity + grad(velocity) * error_velocity,
            dual_velocity,
        )
        - viscosity * inner(grad(error_velocity), grad(dual_velocity))
        + error_pressure * div(dual_velocity)
        + div(error_velocity) * dual_pressure
    ) * measure
    form += (1.0 / float(goal_horizon)) * drag_derivative_form(
        error_velocity,
        error_pressure,
        viscosity,
        cylinder_labels,
        weight=cell_weight,
    )
    return form


def adjoint_jump_residual_form(
    primal_error_left,
    primal_error_previous,
    low_dual_left,
    *,
    cell_weight=1.0,
    measure=dx,
):
    error_jump = (
        primal_error_left.subfunctions[0]
        - primal_error_previous.subfunctions[0]
    )
    return -cell_weight * inner(error_jump, split(low_dual_left)[0]) * measure


def estimate_symmetric_dwr(
    primal: dict[str, Any],
    primal_enriched: dict[str, Any],
    dual_enriched: dict[str, Any],
    dual_low: dict[str, Any],
    *,
    quadrature_points: int = 4,
    dual_weight_mode: str = "enriched_minus_numerical",
    include_cubic_remainder: bool = False,
) -> SymmetricDWREstimate:
    r"""Assemble the nonlinear three-term estimator and exact DG0 partition.

    With ``rho=-A`` the Galerkin/iteration correction has a minus sign:

    ``eta = 1/2 rho(e_Z) + 1/2 rho*(e_U) - rho(Z_h)``.

    The optional cubic Lagrangian remainder is deliberately disabled by the
    cylinder production configuration.  It remains available only for a
    later ablation/identity study and is never inferred from the observed
    enriched goal difference.
    """
    times = primal["times"]
    if not np.array_equal(times, primal_enriched["times"]):
        raise ValueError("Low and enriched primal histories need the same time grid.")
    horizon = float(times[-1] - times[0])
    quadrature = gauss_rule(int(quadrature_points))
    nslabs = len(times) - 1
    primal_cells: list[np.ndarray | None] = [None] * (nslabs + 1)
    adjoint_cells: list[np.ndarray | None] = [None] * (nslabs + 1)
    correction_cells: list[np.ndarray | None] = [None] * (nslabs + 1)
    remainder_cells: list[np.ndarray | None] = [None] * (nslabs + 1)
    combined_cells: list[np.ndarray | None] = [None] * (nslabs + 1)
    eta_primal = eta_adjoint = eta_correction = eta_remainder = 0.0
    rich_primal_defect = rich_adjoint_defect = 0.0
    rich_primal_volume = [0.0] * (nslabs + 1)
    rich_primal_jump = [0.0] * (nslabs + 1)
    rich_adjoint_volume = [0.0] * (nslabs + 1)
    rich_adjoint_jump = [0.0] * (nslabs + 1)

    for n in range(1, nslabs + 1):
        step = float(times[n] - times[n - 1])
        low_slab = primal["slabs"][n]
        mesh = low_slab.get("mesh", primal.get("mesh"))
        if mesh is None:
            raise ValueError(f"Primal slab {n} does not identify its mesh.")
        rich_primal_slab = primal_enriched["slabs"][n]
        low_dual_slab = dual_low["slabs"][n]
        rich_dual_slab = dual_enriched["slabs"][n]
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_test = TestFunction(DG0)
        primal_values = np.zeros(DG0.node_count)
        adjoint_values = np.zeros(DG0.node_count)
        correction_values = np.zeros(DG0.node_count)
        remainder_values = np.zeros(DG0.node_count)

        for point, weight in quadrature:
            factor = float(step) * float(weight)
            state = evaluate_slab(low_slab, point, name="U_low_estimator")
            state_dt = evaluate_slab_dt(low_slab, point, step, name="U_low_dt_estimator")
            dual_error = dual_error_at(
                rich_dual_slab,
                low_dual_slab,
                point,
                mode=dual_weight_mode,
                essential_labels=(
                    tuple(primal["labels"]["inlet"])
                    + tuple(primal["labels"]["wall"])
                ),
            )
            low_dual_value = evaluate_slab(
                low_dual_slab, 1.0 - float(point), name="Z_low_estimator"
            )
            primal_error, primal_error_dt = primal_error_at(
                rich_primal_slab, low_slab, point, step
            )
            rich_state = evaluate_slab(
                rich_primal_slab, point, name="U_rich_stationarity"
            )
            rich_state_dt = evaluate_slab_dt(
                rich_primal_slab,
                point,
                step,
                name="U_rich_dt_stationarity",
            )
            full_dual_error = dual_error_at(
                rich_dual_slab,
                low_dual_slab,
                point,
                mode="enriched_minus_numerical",
            )
            rich_dual_value = evaluate_slab(
                rich_dual_slab,
                1.0 - float(point),
                name="Z_rich_stationarity",
            )

            primal_form = factor * primal_volume_residual_form(
                state, state_dt, dual_error, primal["viscosity"]
            )
            adjoint_form = factor * adjoint_volume_residual_form(
                state,
                state_dt,
                primal_error,
                primal_error_dt,
                low_dual_value,
                primal["viscosity"],
                primal["labels"]["cylinder"],
                horizon,
            )
            # Dissertation (2.54): the Galerkin correction is
            # ``-rho(u_h)(z_h)``.
            correction_form = -factor * primal_volume_residual_form(
                state, state_dt, low_dual_value, primal["viscosity"]
            )
            eta_primal += float(assemble(primal_form))
            eta_adjoint += float(assemble(adjoint_form))
            eta_correction += float(assemble(correction_form))
            error_velocity = split(primal_error)[0]
            error_dual_velocity = split(full_dual_error)[0]
            if include_cubic_remainder:
                remainder_form = 0.5 * factor * inner(
                    grad(error_velocity) * error_velocity,
                    error_dual_velocity,
                ) * dx
                eta_remainder += float(assemble(remainder_form))
            primal_stationarity_value = float(assemble(
                factor * primal_volume_residual_form(
                    rich_state,
                    rich_state_dt,
                    full_dual_error,
                    primal["viscosity"],
                )
            ))
            adjoint_stationarity_value = float(assemble(
                factor * adjoint_volume_residual_form(
                    rich_state,
                    rich_state_dt,
                    primal_error,
                    primal_error_dt,
                    rich_dual_value,
                    primal["viscosity"],
                    primal["labels"]["cylinder"],
                    horizon,
                )
            ))
            rich_primal_defect += primal_stationarity_value
            rich_adjoint_defect += adjoint_stationarity_value
            rich_primal_volume[n] += primal_stationarity_value
            rich_adjoint_volume[n] += adjoint_stationarity_value
            primal_values += np.asarray(
                assemble(factor * primal_volume_residual_form(
                    state, state_dt, dual_error, primal["viscosity"], cell_weight=cell_test
                )).dat.data_ro,
                dtype=float,
            )
            adjoint_values += np.asarray(
                assemble(factor * adjoint_volume_residual_form(
                    state,
                    state_dt,
                    primal_error,
                    primal_error_dt,
                    low_dual_value,
                    primal["viscosity"],
                    primal["labels"]["cylinder"],
                    horizon,
                    cell_weight=cell_test,
                )).dat.data_ro,
                dtype=float,
            )
            correction_values -= np.asarray(
                assemble(factor * primal_volume_residual_form(
                    state, state_dt, low_dual_value, primal["viscosity"], cell_weight=cell_test
                )).dat.data_ro,
                dtype=float,
            )
            if include_cubic_remainder:
                remainder_values += np.asarray(
                    assemble(
                        0.5
                        * factor
                        * inner(
                            grad(error_velocity) * error_velocity,
                            error_dual_velocity,
                        )
                        * cell_test
                        * dx
                    ).dat.data_ro,
                    dtype=float,
                )

        state_left = evaluate_slab(low_slab, 0.0, name="U_low_left_estimator")
        dual_error_left = dual_error_at(
            rich_dual_slab,
            low_dual_slab,
            0.0,
            mode=dual_weight_mode,
            essential_labels=(
                tuple(primal["labels"]["inlet"])
                + tuple(primal["labels"]["wall"])
            ),
        )
        low_dual_left = evaluate_slab(low_dual_slab, 1.0, name="Z_low_left_estimator")
        rich_error_left, _ = primal_error_at(rich_primal_slab, low_slab, 0.0, step)
        rich_previous = rich_primal_slab["left_trace"]
        low_previous_rich_operator = _low_incoming_in_rich_operator(
            primal, primal_enriched, n
        )
        previous_error = _difference(
            rich_previous,
            low_previous_rich_operator,
            "primal_previous_error",
        )

        primal_jump = primal_jump_residual_form(
            state_left, low_previous_rich_operator, dual_error_left
        )
        adjoint_jump = adjoint_jump_residual_form(
            rich_error_left, previous_error, low_dual_left
        )
        # Galerkin correction rho(U_h)(Z_h): use the exact incoming trace
        # that was supplied to the low primal slab solve.  Reusing the
        # enriched interface operator here evaluates a different discrete
        # residual once neighbouring slabs own different meshes.
        low_previous_exact = low_slab["left_trace"]
        correction_jump = -primal_jump_residual_form(
            state_left, low_previous_exact, low_dual_left
        )
        eta_primal += float(assemble(primal_jump))
        eta_adjoint += float(assemble(adjoint_jump))
        eta_correction += float(assemble(correction_jump))
        rich_state_left = evaluate_slab(
            rich_primal_slab, 0.0, name="U_rich_left_stationarity"
        )
        full_dual_error_left = dual_error_at(
            rich_dual_slab,
            low_dual_slab,
            0.0,
            mode="enriched_minus_numerical",
        )
        rich_dual_left = evaluate_slab(
            rich_dual_slab, 1.0, name="Z_rich_left_stationarity"
        )
        primal_stationarity_jump = float(assemble(primal_jump_residual_form(
            rich_state_left,
            rich_primal_slab["left_trace"],
            full_dual_error_left,
        )))
        adjoint_stationarity_jump = float(assemble(adjoint_jump_residual_form(
            rich_error_left,
            previous_error,
            rich_dual_left,
        )))
        rich_primal_defect += primal_stationarity_jump
        rich_adjoint_defect += adjoint_stationarity_jump
        rich_primal_jump[n] = primal_stationarity_jump
        rich_adjoint_jump[n] = adjoint_stationarity_jump
        primal_values += np.asarray(
            assemble(primal_jump_residual_form(
                state_left, low_previous_rich_operator, dual_error_left, cell_weight=cell_test
            )).dat.data_ro,
            dtype=float,
        )
        adjoint_values += np.asarray(
            assemble(adjoint_jump_residual_form(
                rich_error_left, previous_error, low_dual_left, cell_weight=cell_test
            )).dat.data_ro,
            dtype=float,
        )
        correction_values -= np.asarray(
            assemble(primal_jump_residual_form(
                state_left, low_previous_exact, low_dual_left, cell_weight=cell_test
            )).dat.data_ro,
            dtype=float,
        )
        primal_cells[n] = primal_values
        adjoint_cells[n] = adjoint_values
        correction_cells[n] = correction_values
        remainder_cells[n] = remainder_values
        combined_cells[n] = (
            0.5 * primal_values
            + 0.5 * adjoint_values
            + correction_values
            + remainder_values
        )

    eta_core = 0.5 * eta_primal + 0.5 * eta_adjoint + eta_correction
    eta = eta_core + eta_remainder
    local_sum = float(sum(values.sum() for values in combined_cells[1:]))
    goal_difference = (
        mean_drag_from_history(primal_enriched, quadrature_points=quadrature_points)
        - mean_drag_from_history(primal, quadrature_points=quadrature_points)
    )
    return SymmetricDWREstimate(
        dual_weight_mode=str(dual_weight_mode),
        eta_global=eta,
        eta_primal_residual=eta_primal,
        eta_adjoint_residual=eta_adjoint,
        eta_galerkin_correction=eta_correction,
        eta_symmetric_core=eta_core,
        eta_cubic_remainder=eta_remainder,
        eta_cell_weak=combined_cells,
        eta_primal_cell_weak=primal_cells,
        eta_adjoint_cell_weak=adjoint_cells,
        eta_correction_cell_weak=correction_cells,
        eta_cubic_remainder_cell_weak=remainder_cells,
        weak_closure_gap=local_sum - eta,
        enriched_goal_difference=goal_difference,
        observed_cubic_remainder=goal_difference - eta_core,
        enriched_identity_gap=goal_difference - eta,
        enriched_primal_stationarity_defect=rich_primal_defect,
        enriched_adjoint_stationarity_defect=rich_adjoint_defect,
        enriched_primal_stationarity_volume_by_slab=rich_primal_volume,
        enriched_primal_stationarity_jump_by_slab=rich_primal_jump,
        enriched_adjoint_stationarity_volume_by_slab=rich_adjoint_volume,
        enriched_adjoint_stationarity_jump_by_slab=rich_adjoint_jump,
    )


__all__ = [
    "LinearDWRGlobalEstimate",
    "SymmetricDWREstimate",
    "adjoint_jump_residual_form",
    "adjoint_volume_residual_form",
    "dual_error_at",
    "estimate_linear_dwr_global",
    "estimate_symmetric_dwr",
    "primal_error_at",
    "primal_trace_error",
    "primal_jump_residual_form",
    "primal_volume_residual_form",
]
