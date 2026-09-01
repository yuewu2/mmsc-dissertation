"""Stationary-style UFL input contract for nonstationary DWR problems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from firedrake import (
    Constant,
    Function,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    adjoint,
    derivative,
    dx,
    inner,
    solve,
)
from irksome import Dt
from ufl import replace
from ufl.algorithms import extract_coefficients
from ufl.algorithms.map_integrands import map_integrands


@dataclass(frozen=True)
class GoalComponent:
    """One terminal/running contribution to a signed-relative multi-goal."""

    label: str
    exact_goal: Callable[[float], float] | float | None
    terminal_goal: Callable[..., Any] | None = None
    running_goal: Callable[..., Any] | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.terminal_goal is None and self.running_goal is None:
            raise ValueError("A goal component needs a terminal or running form.")
        if self.weight <= 0.0:
            raise ValueError("Goal-component weights must be positive.")


def _space_from_expressions(*expressions):
    for expression in expressions:
        if hasattr(expression, "function_space"):
            return expression.function_space()
        for coefficient in extract_coefficients(expression):
            if hasattr(coefficient, "function_space"):
                return coefficient.function_space()
    raise ValueError("Could not infer a Firedrake space from the supplied expressions.")


def _multiply_integrands(form, weight):
    return map_integrands(lambda integrand: weight * integrand, form)


class NonstationaryProblem:
    r"""Minimal user input for the generic adaptive solver.

    The user provides the mesh, initial value, weak spatial residual
    ``A(u;v,t)``, one or both of a terminal goal ``J_T(u(T))`` and a running
    goal ``integral j(u,t) dt``, and boundary conditions.  The class constructs
    the full primal residual, Firedrake/UFL adjoint and all derivative actions
    used by the DWR identity.
    """

    def __init__(
        self,
        *,
        mesh: Callable[[int, int], Mesh],
        initial_condition: Callable[[Mesh], Any],
        spatial_residual: Callable[..., Any],
        goal: Callable[..., Any] | None = None,
        terminal_goal: Callable[..., Any] | None = None,
        running_goal: Callable[..., Any] | None = None,
        goal_components: list[GoalComponent] | tuple[GoalComponent, ...] | None = None,
        boundary_conditions: Callable[[Any], Any] | None = None,
        adjoint_boundary_conditions: Callable[[Any], Any] | None = None,
        time_mass: Callable[..., Any] | None = None,
        exact_goal: Callable[[float], float] | float | None = None,
        goal_diagnostics: Callable[[float, float, float | None], dict[str, float]] | None = None,
        strong_residual: Callable[..., Any] | None = None,
        normal_flux: Callable[..., Any] | None = None,
        refine: Callable[[Mesh, Any], Mesh] | None = None,
        name: str = "nonstationary_problem",
        goal_label: str = "terminal_qoi",
        nonlinear: bool = False,
        nonlinear_identity: bool = False,
        spatial_refinement_mode: str = "marked",
        solver_parameters: dict[str, Any] | None = None,
    ):
        self._mesh_input = mesh
        self._initial_input = initial_condition
        self._spatial_residual_input = spatial_residual
        if goal_components and any(
            value is not None for value in (goal, terminal_goal, running_goal)
        ):
            raise ValueError(
                "Pass either goal_components= or terminal/running goal callbacks."
            )
        if goal is not None and terminal_goal is not None:
            raise ValueError("Pass either goal= or terminal_goal=, not both.")
        self._goal_components = tuple(goal_components or ())
        self._goal_component_weights = [1.0] * len(self._goal_components)
        self._terminal_goal_input = terminal_goal if terminal_goal is not None else goal
        self._running_goal_input = running_goal
        if (
            self._terminal_goal_input is None
            and self._running_goal_input is None
            and not self._goal_components
        ):
            raise ValueError("Provide terminal_goal=, running_goal=, or the goal= alias.")
        self._bcs_input = boundary_conditions
        self._adjoint_bcs_input = adjoint_boundary_conditions
        self._bcs_accept_time = self._accepts_time_argument(boundary_conditions)
        self._time_mass_input = time_mass
        self._exact_goal_input = exact_goal
        self._goal_diagnostics_input = goal_diagnostics
        self._strong_residual_input = strong_residual
        self._normal_flux_input = normal_flux
        self._refine_input = refine
        self.name = str(name)
        self.goal_label = str(goal_label)
        self.spatial_operator_is_linear = not bool(nonlinear)
        self.requires_primal_linearisation = bool(nonlinear)
        self.supports_nonlinear_error_identity = bool(nonlinear_identity)
        self.spatial_refinement_mode = str(spatial_refinement_mode)
        self.solver_parameters = solver_parameters
        self._final_time: float | None = None

    @property
    def supports_strong_residual_bound(self) -> bool:
        """Whether scalar strong-volume and normal-flux inputs are available."""
        return (
            self._strong_residual_input is not None
            and self._normal_flux_input is not None
        )

    def strong_residual(self, mesh, state, state_dt, time):
        if self._strong_residual_input is None:
            raise ValueError(
                "strong_residual_bound requires strong_residual= in the problem."
            )
        return self._strong_residual_input(mesh, state, state_dt, time)

    def normal_flux(self, state, normal):
        if self._normal_flux_input is None:
            raise ValueError(
                "strong_residual_bound requires normal_flux= in the problem."
            )
        return self._normal_flux_input(state, normal)

    def set_final_time(self, final_time: float) -> None:
        self._final_time = float(final_time)

    # Backwards-compatible name used by the earlier public wrapper.
    set_dwr_final_time = set_final_time

    def make_mesh(self, nx: int, ny: int) -> Mesh:
        return self._mesh_input(int(nx), int(ny))

    def initial_condition(self, mesh: Mesh):
        return self._initial_input(mesh)

    def goal_diagnostics(
        self, goal_value: float, estimator: float, true_error: float | None,
        *, symmetric_identity: bool = False,
    ) -> dict[str, float]:
        """Return optional problem-specific derived quantities for the CSV."""
        if self._goal_diagnostics_input is None:
            return {}
        callback = self._goal_diagnostics_input
        if "symmetric_identity" in signature(callback).parameters:
            return dict(callback(
                goal_value, estimator, true_error,
                symmetric_identity=symmetric_identity,
            ))
        return dict(callback(goal_value, estimator, true_error))

    def spatial_residual(self, state, test, time, measure=dx):
        return self._spatial_residual_input(state, test, time, measure=measure)

    @property
    def has_terminal_goal(self) -> bool:
        return self._terminal_goal_input is not None or any(
            component.terminal_goal is not None for component in self._goal_components
        )

    @property
    def has_running_goal(self) -> bool:
        return self._running_goal_input is not None or any(
            component.running_goal is not None for component in self._goal_components
        )

    @property
    def has_goal_components(self) -> bool:
        return bool(self._goal_components)

    @property
    def goal_components(self) -> tuple[GoalComponent, ...]:
        return self._goal_components

    @property
    def goal_component_weights(self) -> tuple[float, ...]:
        return tuple(self._goal_component_weights)

    def terminal_goal_form(self, mesh: Mesh, terminal_state, *, measure=dx):
        if self._goal_components:
            form = 0
            for component, weight in zip(
                self._goal_components, self._goal_component_weights
            ):
                if component.terminal_goal is not None:
                    form += Constant(weight) * component.terminal_goal(
                        mesh, terminal_state, measure=measure
                    )
            return form
        if self._terminal_goal_input is None:
            return None
        return self._terminal_goal_input(mesh, terminal_state, measure=measure)

    def running_goal_form(self, mesh: Mesh, state, time, *, measure=dx):
        if self._goal_components:
            form = 0
            for component, weight in zip(
                self._goal_components, self._goal_component_weights
            ):
                if component.running_goal is not None:
                    form += Constant(weight) * component.running_goal(
                        mesh, state, time, measure=measure
                    )
            return form
        if self._running_goal_input is None:
            return None
        return self._running_goal_input(mesh, state, time, measure=measure)

    def terminal_goal_component_form(
        self, index: int, mesh: Mesh, terminal_state, *, measure=dx
    ):
        component = self._goal_components[index]
        if component.terminal_goal is None:
            return None
        return component.terminal_goal(mesh, terminal_state, measure=measure)

    def running_goal_component_form(
        self, index: int, mesh: Mesh, state, time, *, measure=dx
    ):
        component = self._goal_components[index]
        if component.running_goal is None:
            return None
        return component.running_goal(mesh, state, time, measure=measure)

    def goal_component_exact_values(self, final_time: float) -> list[float | None]:
        values: list[float | None] = []
        for component in self._goal_components:
            exact = component.exact_goal
            values.append(
                None
                if exact is None
                else float(exact(float(final_time)) if callable(exact) else exact)
            )
        return values

    def update_signed_relative_goal(
        self,
        current_values: list[float],
        reference_values: list[float],
    ) -> list[float]:
        """Freeze signed relative weights for the current adaptive iteration."""
        if len(current_values) != len(self._goal_components):
            raise ValueError("Goal-component value count does not match the problem.")
        weights: list[float] = []
        for component, current, reference in zip(
            self._goal_components, current_values, reference_values
        ):
            scale = max(abs(float(reference)), 1.0e-15)
            error = float(reference) - float(current)
            sign = 1.0 if error >= 0.0 else -1.0
            weights.append(float(component.weight) * sign / scale)
        self._goal_component_weights[:] = weights
        return weights

    # Backwards-compatible terminal-goal names used by the original examples.
    def goal_form(self, mesh: Mesh, terminal_state, *, measure=dx):
        form = self.terminal_goal_form(mesh, terminal_state, measure=measure)
        if form is None:
            raise ValueError("This problem has no terminal goal form.")
        return form

    def goal_functional(self, mesh: Mesh, terminal_state):
        return self.goal_form(mesh, terminal_state, measure=dx)

    @staticmethod
    def _accepts_time_argument(callback) -> bool:
        """Return whether ``callback(V, time)`` is a supported call shape."""
        if callback is None:
            return False
        parameters = list(signature(callback).parameters.values())
        return any(p.kind == Parameter.VAR_POSITIONAL for p in parameters) or len(
            [
                p
                for p in parameters
                if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            ]
        ) >= 2

    @staticmethod
    def _as_bc_list(values) -> list[Any]:
        if values is None:
            return []
        return list(values) if isinstance(values, (list, tuple)) else [values]

    def boundary_conditions(self, V, time=None) -> list[Any]:
        """Return primal BCs, optionally evaluated at Irksome's time Constant."""
        if self._bcs_input is None:
            return []
        values = (
            self._bcs_input(V, time)
            if self._bcs_accept_time
            else self._bcs_input(V)
        )
        return self._as_bc_list(values)

    def adjoint_boundary_conditions(self, V) -> list[Any]:
        """Return adjoint BCs; homogeneous primal BCs remain the default."""
        callback = self._adjoint_bcs_input
        if callback is None:
            if self._bcs_accept_time:
                raise ValueError(
                    "A time-dependent primal Dirichlet condition requires an explicit "
                    "adjoint_boundary_conditions= callback (normally homogeneous)."
                )
            callback = self._bcs_input
        return self._as_bc_list(None if callback is None else callback(V))

    def exact_goal_value(self, final_time: float) -> float | None:
        if self._goal_components:
            exact_values = self.goal_component_exact_values(final_time)
            if any(value is None for value in exact_values):
                return None
            return float(sum(
                weight * value
                for weight, value in zip(self._goal_component_weights, exact_values)
            ))
        if self._exact_goal_input is None:
            return None
        if callable(self._exact_goal_input):
            return float(self._exact_goal_input(float(final_time)))
        return float(self._exact_goal_input)

    def refine_slab_mesh(self, mesh: Mesh, markers):
        if self._refine_input is None:
            raise RuntimeError("This problem did not provide a custom mesh refiner.")
        return self._refine_input(mesh, markers)

    def time_mass_action(self, trial, test, measure=dx):
        if self._time_mass_input is None:
            return inner(trial, test) * measure
        return self._time_mass_input(trial, test, measure=measure)

    def primal_residual(self, state: Function, test, time: Constant):
        return self.time_mass_action(Dt(state), test, dx) + self.spatial_residual(
            state, test, time, measure=dx
        )

    def volume_residual_action(self, state, state_dt, test, time: Constant, measure=dx):
        return -self.time_mass_action(state_dt, test, measure) - self.spatial_residual(
            state, test, time, measure=measure
        )

    def temporal_residual_action(self, state_left, previous_right, test, measure=dx):
        return -self.time_mass_action(state_left - previous_right, test, measure)

    def _physical_adjoint_time(self, reverse_time):
        if self._final_time is None:
            raise RuntimeError("The solver must set the final time before solving.")
        return Constant(self._final_time) - reverse_time

    def adjoint_residual(
        self,
        dual: Function,
        test,
        reverse_time: Constant,
        *,
        linearisation_state=None,
    ):
        V = dual.function_space()
        placeholder = Function(V, name="automatic_adjoint_state")
        primal_test = TestFunction(V)
        increment = TrialFunction(V)
        spatial = self.spatial_residual(
            placeholder,
            primal_test,
            self._physical_adjoint_time(reverse_time),
            measure=dx,
        )
        transposed = action(adjoint(derivative(spatial, placeholder, increment)), dual)
        if linearisation_state is not None:
            transposed = replace(transposed, {placeholder: linearisation_state})
        elif self.requires_primal_linearisation:
            raise ValueError("A nonlinear adjoint needs the saved primal slab state.")
        residual = self.time_mass_action(Dt(dual), test, dx) + transposed
        if self.has_running_goal:
            physical_time = self._physical_adjoint_time(reverse_time)
            goal_state = placeholder if linearisation_state is None else linearisation_state
            goal_source = self.running_goal_derivative_action(
                V.mesh(), goal_state, test, physical_time, measure=dx
            )
            if (
                linearisation_state is None
                and placeholder in extract_coefficients(goal_source)
            ):
                raise ValueError("A nonlinear running goal needs the saved primal slab state.")
            residual -= goal_source
        return residual

    def terminal_adjoint_state(
        self,
        V,
        name: str,
        *,
        terminal_primal: Function | None = None,
    ) -> Function:
        terminal_dual = Function(V, name=name)
        if not self.has_terminal_goal:
            terminal_dual.assign(0.0)
            for bc in self.adjoint_boundary_conditions(V):
                bc.apply(terminal_dual)
            return terminal_dual
        mesh = V.mesh()
        placeholder = Function(V, name="automatic_terminal_goal_state")
        test, trial = TestFunction(V), TrialFunction(V)
        rhs = derivative(
            self.terminal_goal_form(mesh, placeholder, measure=dx), placeholder, test
        )
        if placeholder in extract_coefficients(rhs):
            if terminal_primal is None:
                raise ValueError("A nonlinear goal needs the terminal primal state.")
            terminal_on_V = Function(V, name="terminal_primal_on_dual_space")
            if terminal_primal.function_space() == V:
                terminal_on_V.assign(terminal_primal)
            else:
                terminal_on_V.interpolate(terminal_primal)
            rhs = replace(rhs, {placeholder: terminal_on_V})
        solve(
            self.time_mass_action(trial, test, dx) == rhs,
            terminal_dual,
            bcs=self.adjoint_boundary_conditions(V),
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        )
        return terminal_dual

    def volume_residual_derivative_action(
        self,
        state,
        state_dt,
        increment,
        increment_dt,
        dual,
        time: Constant,
        *,
        cell_weight=1.0,
        measure=dx,
    ):
        V = _space_from_expressions(state, dual, increment)
        placeholder = Function(V, name="automatic_volume_state")
        dt_placeholder = Function(V, name="automatic_volume_state_dt")
        variation = TestFunction(V)
        residual = self.volume_residual_action(
            placeholder, dt_placeholder, dual, time, measure=measure
        )
        replacements = {
            variation: increment,
            placeholder: state,
            dt_placeholder: state_dt,
        }
        linearised = replace(derivative(residual, placeholder, variation), replacements)
        linearised += replace(
            derivative(residual, dt_placeholder, variation),
            {**replacements, variation: increment_dt},
        )
        return _multiply_integrands(linearised, cell_weight)

    def temporal_residual_derivative_action(
        self,
        increment_left,
        increment_previous_right,
        dual,
        *,
        cell_weight=1.0,
        measure=dx,
    ):
        V = _space_from_expressions(dual, increment_left)
        left = Function(V, name="automatic_jump_left")
        previous = Function(V, name="automatic_jump_previous")
        variation = TestFunction(V)
        residual = self.temporal_residual_action(left, previous, dual, measure=measure)
        placeholders = {left: increment_left, previous: increment_previous_right}
        linearised = replace(
            derivative(residual, left, variation),
            {**placeholders, variation: increment_left},
        )
        linearised += replace(
            derivative(residual, previous, variation),
            {**placeholders, variation: increment_previous_right},
        )
        return _multiply_integrands(linearised, cell_weight)

    def terminal_goal_derivative_action(
        self,
        mesh,
        terminal_state,
        terminal_increment,
        *,
        cell_weight=1.0,
        measure=dx,
    ):
        if not self.has_terminal_goal:
            return None
        V = _space_from_expressions(terminal_state, terminal_increment)
        placeholder = Function(V, name="automatic_goal_state")
        variation = TestFunction(V)
        differentiated = replace(
            derivative(
                self.terminal_goal_form(mesh, placeholder, measure=measure),
                placeholder,
                variation,
            ),
            {variation: terminal_increment, placeholder: terminal_state},
        )
        return _multiply_integrands(differentiated, cell_weight)

    # Backwards-compatible name: historically every goal was terminal.
    goal_derivative_action = terminal_goal_derivative_action

    def running_goal_derivative_action(
        self,
        mesh,
        state,
        increment,
        time,
        *,
        cell_weight=1.0,
        measure=dx,
    ):
        if not self.has_running_goal:
            return None
        V = _space_from_expressions(state, increment)
        placeholder = Function(V, name="automatic_running_goal_state")
        variation = TestFunction(V)
        differentiated = replace(
            derivative(
                self.running_goal_form(mesh, placeholder, time, measure=measure),
                placeholder,
                variation,
            ),
            {variation: increment, placeholder: state},
        )
        return _multiply_integrands(differentiated, cell_weight)


# Internal modules retain this descriptive type name.
TransientDWRProblem = NonstationaryProblem


__all__ = ["GoalComponent", "NonstationaryProblem", "TransientDWRProblem"]
