"""Irksome DG-in-time solves and slab-polynomial evaluation.

All primal and dual states are saved slab by slab.  Bubble localisation needs
``u_h(t)`` and ``partial_t u_h(t)`` at quadrature points, so retaining only the
time-node values would be mathematically insufficient.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from firedrake import Constant, Function, Mesh, TestFunction, assemble, dx
from irksome import DiscontinuousGalerkinScheme, TimeStepper
from ufl.algorithms.map_integrands import map_integrands

from .problem import TransientDWRProblem
from .transfer import SlabInterfaceTransfer


def time_nodes(degree: int) -> np.ndarray:
    """Return equispaced reference nodes of ``P_degree([0,1])``."""
    return np.asarray([0.5]) if int(degree) == 0 else np.linspace(0.0, 1.0, int(degree) + 1)


def lagrange_values(nodes: np.ndarray, s: float) -> np.ndarray:
    """Evaluate Lagrange basis ``ell_i(s)`` used to reconstruct one DG slab."""
    if len(nodes) == 1:
        return np.asarray([1.0])
    return np.asarray([
        np.prod([(float(s) - xj) / (xi - xj) for j, xj in enumerate(nodes) if i != j])
        for i, xi in enumerate(nodes)
    ])


def lagrange_derivatives(nodes: np.ndarray, s: float) -> np.ndarray:
    """Evaluate ``d ell_i/ds`` for ``partial_t u_h=k_n^{-1}d_su_h``."""
    if len(nodes) == 1:
        return np.asarray([0.0])
    values: list[float] = []
    for i, xi in enumerate(nodes):
        derivative_value = 0.0
        for m, xm in enumerate(nodes):
            if m == i:
                continue
            product = 1.0 / (xi - xm)
            for j, xj in enumerate(nodes):
                if j != i and j != m:
                    product *= (float(s) - xj) / (xi - xj)
            derivative_value += product
        values.append(derivative_value)
    return np.asarray(values)


def linear_combination(coefficients: Sequence, weights: Sequence):
    """Build the UFL expression ``sum_i weights[i] coefficients[i]``.

    ``weights`` may be floating-point values or UFL expressions.  The latter
    is needed when a nonlinear adjoint evaluates a saved primal DG polynomial
    at Irksome's reverse-time stage locations.
    """
    expression = Constant(0.0) * coefficients[0]
    for weight, coefficient in zip(weights, coefficients):
        expression += weight * coefficient
    return expression


def evaluate_slab(slab: dict[str, Any], s: float):
    """Evaluate saved forward or reversed DG polynomial at physical ``s``."""
    s_eval = 1.0 - float(s) if slab.get("orientation") == "reverse" else float(s)
    return linear_combination(slab["coeffs"], lagrange_values(time_nodes(slab["degree"]), s_eval))


def evaluate_slab_dt(slab: dict[str, Any], s: float, step: float):
    """Evaluate ``partial_t`` of a saved slab, including the reversal sign."""
    reverse = slab.get("orientation") == "reverse"
    s_eval = 1.0 - float(s) if reverse else float(s)
    sign = -1.0 if reverse else 1.0
    return linear_combination(
        slab["coeffs"],
        sign * lagrange_derivatives(time_nodes(slab["degree"]), s_eval) / float(step),
    )


def lagrange_expression_values(nodes: np.ndarray, s):
    """Evaluate Lagrange basis functions when ``s`` is a UFL expression."""
    if len(nodes) == 1:
        return [1.0]
    values = []
    for i, xi in enumerate(nodes):
        value = 1.0
        for j, xj in enumerate(nodes):
            if i != j:
                value *= (s - float(xj)) / float(xi - xj)
        values.append(value)
    return values


def evaluate_forward_slab_at_physical_time(
    slab: dict[str, Any], physical_time, time_left: float, step: float,
):
    r"""Evaluate a saved forward DG polynomial at symbolic physical time.

    This keeps the nonlinear adjoint linearised at ``U_h(t)`` at each Irksome
    reverse-time stage, rather than freezing the coefficient at a slab end.
    """
    if slab.get("orientation", "forward") != "forward":
        raise ValueError("Primal linearisation requires a forward-oriented slab.")
    reference_time = (physical_time - Constant(float(time_left))) / Constant(float(step))
    return linear_combination(
        slab["coeffs"],
        lagrange_expression_values(time_nodes(slab["degree"]), reference_time),
    )


def gauss_rule(npoints: int) -> list[tuple[float, float]]:
    """Map Gauss--Legendre quadrature from ``[-1,1]`` to a time slab."""
    points, weights = np.polynomial.legendre.leggauss(max(1, int(npoints)))
    return [(float(0.5 * (s + 1.0)), float(0.5 * w)) for s, w in zip(points, weights)]


def evaluate_goal(
    problem: TransientDWRProblem,
    primal: dict[str, Any],
    ts: np.ndarray,
    quadrature_points: int,
) -> float:
    r"""Assemble ``J_T(u(T)) + integral_0^T j(u,t) dt`` on saved slabs."""
    nslabs = len(ts) - 1
    value = 0.0
    if problem.has_terminal_goal:
        terminal = problem.terminal_goal_form(
            primal["slabs"][nslabs]["mesh"], primal["nodes"][nslabs], measure=dx
        )
        value += float(assemble(terminal))
    if problem.has_running_goal:
        for n in range(1, nslabs + 1):
            step = float(ts[n] - ts[n - 1])
            slab = primal["slabs"][n]
            for reference_time, weight in gauss_rule(quadrature_points):
                physical_time = Constant(float(ts[n - 1] + step * reference_time))
                form = problem.running_goal_form(
                    slab["mesh"], evaluate_slab(slab, reference_time), physical_time,
                    measure=dx,
                )
                weighted = map_integrands(
                    lambda integrand: Constant(step * weight) * integrand, form
                )
                value += float(assemble(weighted))
    return value


def evaluate_goal_components(
    problem: TransientDWRProblem,
    primal: dict[str, Any],
    ts: np.ndarray,
    quadrature_points: int,
) -> list[float]:
    """Assemble every unweighted component of a signed-relative goal."""
    nslabs = len(ts) - 1
    values = [0.0] * len(problem.goal_components)
    for index, component in enumerate(problem.goal_components):
        if component.terminal_goal is not None:
            form = problem.terminal_goal_component_form(
                index,
                primal["slabs"][nslabs]["mesh"],
                primal["nodes"][nslabs],
                measure=dx,
            )
            values[index] += float(assemble(form))
        if component.running_goal is not None:
            for n in range(1, nslabs + 1):
                step = float(ts[n] - ts[n - 1])
                slab = primal["slabs"][n]
                for reference_time, weight in gauss_rule(quadrature_points):
                    physical_time = Constant(
                        float(ts[n - 1] + step * reference_time)
                    )
                    form = problem.running_goal_component_form(
                        index,
                        slab["mesh"],
                        evaluate_slab(slab, reference_time),
                        physical_time,
                        measure=dx,
                    )
                    weighted = map_integrands(
                        lambda integrand: Constant(step * weight) * integrand,
                        form,
                    )
                    values[index] += float(assemble(weighted))
    return values


def _copy(function: Function, name: str) -> Function:
    """Detach a function from Irksome's mutable stage storage."""
    copied = Function(function.function_space(), name=name)
    copied.assign(function)
    return copied


def _interpolate(V, expression, name: str) -> Function:
    """Store a UFL expression in a named Firedrake function."""
    value = Function(V, name=name)
    value.interpolate(expression)
    return value


def _dg_scheme(degree: int) -> DiscontinuousGalerkinScheme:
    """Use the same strong-form Irksome DG scheme for every time solve."""
    return DiscontinuousGalerkinScheme(
        int(degree),
        basis_type="equispaced",
        quadrature_degree=max(2 * int(degree) + 3, 3),
        deriv_type="strong",
    )


class IrksomeDGSolver:
    """Solve primal and terminal adjoint on independently refinable slabs."""

    def __init__(self, problem: TransientDWRProblem, solver_parameters: dict[str, Any] | None = None):
        self.problem = problem
        self.solver_parameters = solver_parameters or {"ksp_type": "preonly", "pc_type": "lu"}

    def _initialise_state(
        self,
        state: Function,
        initial_data,
        transfer: SlabInterfaceTransfer | None,
        reverse_transfer: bool,
        name: str,
        boundary_time,
        adjoint_problem: bool,
    ) -> Function:
        """Set the incoming state by an expression, ``P``, or ``P_star``."""
        boundary_conditions = (
            self.problem.adjoint_boundary_conditions
            if adjoint_problem
            else lambda V: self.problem.boundary_conditions(V, boundary_time)
        )
        if isinstance(initial_data, Function):
            if transfer is None:
                if initial_data.function_space() != state.function_space():
                    raise ValueError("A direct function initial state must belong to the slab space.")
                state.assign(initial_data)
                for bc in boundary_conditions(state.function_space()):
                    bc.apply(state)
                return state
            return (
                transfer.adjoint(initial_data, name)
                if reverse_transfer
                else transfer.forward(initial_data, name, boundary_time=boundary_time)
            )
        state.interpolate(initial_data)
        for bc in boundary_conditions(state.function_space()):
            bc.apply(state)
        return state

    def _solve_one_slab(
        self,
        V,
        mesh: Mesh,
        time_left: float,
        step: float,
        initial_data,
        degree: int,
        residual_builder,
        name_prefix: str,
        *,
        orientation: str = "forward",
        transfer: SlabInterfaceTransfer | None = None,
        reverse_transfer: bool = False,
        adjoint_problem: bool = False,
    ) -> dict[str, Any]:
        r"""Solve one DG slab on its own mesh and save its polynomial.

        ``orientation='reverse'`` means that the solve proceeds forward in
        ``tau=T-t``.  The saved coefficients are nevertheless evaluated in
        physical reference time by :func:`evaluate_slab`.
        """
        time = Constant(float(time_left))
        dt = Constant(float(step))
        state = Function(V, name=f"{name_prefix}_solution")
        state = self._initialise_state(
            state,
            initial_data,
            transfer,
            reverse_transfer,
            f"{name_prefix}_incoming",
            time,
            adjoint_problem,
        )
        previous_right = _copy(state, f"{name_prefix}_previous_right")
        residual = residual_builder(state, TestFunction(V), time)
        stepper = TimeStepper(
            residual,
            _dg_scheme(degree),
            time,
            dt,
            state,
            bcs=(
                self.problem.adjoint_boundary_conditions(V)
                if adjoint_problem
                else self.problem.boundary_conditions(V, time)
            ),
            solver_parameters=self.solver_parameters,
        )
        stepper.advance()
        coefficients = [
            _copy(coefficient, f"{name_prefix}_DG{degree}_coefficient_{i}")
            for i, coefficient in enumerate(stepper.stages.subfunctions)
        ]
        right = _interpolate(
            V,
            linear_combination(coefficients, lagrange_values(time_nodes(degree), 1.0)),
            f"{name_prefix}_right",
        )
        return {
            "degree": int(degree), "orientation": orientation,
            "coeffs": coefficients, "prev_right": previous_right,
            "right": right, "mesh": mesh,
        }

    def solve_primal(
        self,
        V_by_slab: list[Any | None],
        meshes: list[Mesh | None],
        ts: np.ndarray,
        transfers: list[SlabInterfaceTransfer | None],
        degree: int,
    ) -> dict[str, Any]:
        """Solve forward with ``u_n^+=P_{n-1}u_{n-1}^-`` at mesh changes."""
        nslabs = len(ts) - 1
        nodes: list[Function | None] = [None] * (nslabs + 1)
        slabs: list[dict[str, Any] | None] = [None] * (nslabs + 1)
        incoming: Function | None = None
        for n in range(1, nslabs + 1):
            mesh, V = meshes[n], V_by_slab[n]
            initial = self.problem.initial_condition(mesh) if n == 1 else incoming
            slab = self._solve_one_slab(
                V, mesh, float(ts[n - 1]), float(ts[n] - ts[n - 1]), initial,
                degree, self.problem.primal_residual, f"U_slab_{n}",
                transfer=None if n == 1 else transfers[n - 1],
            )
            slabs[n] = slab
            nodes[n] = _copy(slab["right"], f"U_node_{n}")
            incoming = slab["right"]
        nodes[0] = _interpolate(V_by_slab[1], self.problem.initial_condition(meshes[1]), "U_node_0")
        return {"nodes": nodes, "slabs": slabs}

    def solve_terminal_adjoint(
        self,
        V_by_slab: list[Any | None],
        meshes: list[Mesh | None],
        ts: np.ndarray,
        transfers: list[SlabInterfaceTransfer | None],
        degree: int,
        primal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        r"""Solve backward with ``z_n^-=P_n^*z_{n+1}^+`` at interfaces.

        A nonlinear problem receives its saved forward DG polynomial on the
        same slab, so its reverse-time form can use ``A'_u(U_h(t))^*`` at the
        actual time-stage locations.
        """
        nslabs = len(ts) - 1
        final_time = float(ts[-1])
        needs_primal = (
            self.problem.requires_primal_linearisation or self.problem.has_running_goal
        )
        if needs_primal and primal is None:
            raise ValueError(
                "A nonlinear operator or running goal requires the saved primal slab polynomials."
            )
        nodes: list[Function | None] = [None] * (nslabs + 1)
        slabs: list[dict[str, Any] | None] = [None] * (nslabs + 1)
        incoming: Function | None = None
        for n in range(nslabs, 0, -1):
            mesh, V = meshes[n], V_by_slab[n]
            initial = (
                self.problem.terminal_adjoint_state(
                    V, f"Z_terminal_DG{degree}",
                    terminal_primal=None if primal is None else primal["nodes"][nslabs],
                )
                if n == nslabs else incoming
            )
            residual_builder = self.problem.adjoint_residual
            if needs_primal:
                primal_slab = primal["slabs"][n]
                slab_left = float(ts[n - 1])
                slab_step = float(ts[n] - ts[n - 1])

                def residual_builder(dual, test, reverse_time):
                    physical_time = Constant(final_time) - reverse_time
                    linearisation_state = evaluate_forward_slab_at_physical_time(
                        primal_slab, physical_time, slab_left, slab_step
                    )
                    return self.problem.adjoint_residual(
                        dual, test, reverse_time,
                        linearisation_state=linearisation_state,
                    )
            slab = self._solve_one_slab(
                V, mesh, final_time - float(ts[n]), float(ts[n] - ts[n - 1]), initial,
                degree, residual_builder, f"Z_DG{degree}_slab_{n}",
                orientation="reverse", transfer=None if n == nslabs else transfers[n],
                reverse_transfer=(n != nslabs),
                adjoint_problem=True,
            )
            slabs[n] = slab
            nodes[n] = _interpolate(V, evaluate_slab(slab, 1.0), f"Z_node_{n}")
            incoming = slab["right"]
        nodes[0] = _interpolate(V_by_slab[1], evaluate_slab(slabs[1], 0.0), "Z_node_0")
        return {"nodes": nodes, "slabs": slabs}
