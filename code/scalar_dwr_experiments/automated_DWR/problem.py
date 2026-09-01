"""PDE input contract and the manufactured heat-equation implementation.

To change PDE, create one small subclass of :class:`TransientDWRProblem`.
The outer DWR algorithm never contains heat-specific constants, source terms,
or goal-functional definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from firedrake import (
    Constant,
    DirichletBC,
    Function,
    IntervalMesh,
    Mesh,
    PeriodicIntervalMesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    action,
    adjoint,
    derivative,
    dx,
    exp,
    grad,
    inner,
    pi,
    cos,
    sin,
    solve,
)
from irksome import Dt
from ufl import And, conditional

from .mesh import create_adaptive_unit_square_mesh


class TransientDWRProblem(ABC):
    r"""Input required by the generic transient DWR solver.

    A forward residual has the weak form

    .. math::

       F(u;v)=\langle \partial_tu,v\rangle+A(u;v)=0.

    ``adjoint_residual`` must describe its time-reversed counterpart
    ``<partial_tau z, w> + A'_u(u)^*(z,w) = 0``.  For a non-self-adjoint or
    nonlinear PDE, this is the only problem-specific adjoint method to amend.
    The helper :func:`firedrake_spatial_adjoint_form` constructs the spatial
    ``A'_u(u)^*`` term using Firedrake/UFL's ``derivative``, ``adjoint``, and
    ``action`` operators, exactly the form-level approach used by the cited
    stationary adaptive solver.
    """

    name = "transient_dwr_problem"
    # Linear scalar inputs use the default UFL-derived spatial adjoint below.
    spatial_operator_is_linear = False
    # A nonlinear adjoint must be supplied with the saved primal polynomial on
    # the physical slab where it is being solved.
    requires_primal_linearisation = False
    # Most meshes use Firedrake/Netgen marked refinement.  A periodic interval
    # does not expose that refiner, so BBM refines only the marked *slabs*
    # uniformly while preserving periodicity.
    spatial_refinement_mode = "marked"
    supports_nonlinear_error_identity = False

    @abstractmethod
    def make_mesh(self, nx: int, ny: int) -> Mesh:
        """Return the coarse mesh on which the first adaptive cycle starts."""

    @abstractmethod
    def initial_condition(self, mesh: Mesh):
        """Return ``u_0`` in ``u(0)=u_0`` as a UFL expression."""

    def primal_residual(self, state: Function, test, time: Constant):
        r"""Build the complete primal residual from the two minimal inputs.

        A scalar conforming PDE normally supplies only

        .. math:: m(\partial_tu,v)+A(u;v,t)=0.

        This default inserts Irksome's ``Dt`` in the problem-defined time
        mass and calls :meth:`spatial_residual`; it removes the need to write
        a duplicate full residual in every experiment input.
        """
        return self.time_mass_action(Dt(state), test) + self.spatial_residual(
            state, test, time, measure=dx
        )

    @abstractmethod
    def spatial_residual(self, state, test, time: Constant, measure=dx):
        r"""Return the non-time part ``A(u;v,t)`` of the weak PDE residual.

        The value must be a UFL one-form.  For example, scalar
        advection--diffusion supplies
        ``eps*(grad(u),grad(v)) + (b dot grad(u),v) - (f,v)``.
        """

    def time_mass_action(self, trial, test, measure=dx):
        r"""Return the bilinear form paired with the PDE time derivative.

        For standard parabolic equations this is the ``L2`` mass

        .. math:: m(u,v)=(u,v)_\Omega.

        Slabwise mesh transfer uses this form to build
        ``P_star=M_old^{-1} P.T M_new``.  BBM must override it with its
        ``H1`` time mass ``(u,v)+(grad(u),grad(v))``; mixed systems similarly
        provide the mass only on components carrying a time derivative.
        """
        return inner(trial, test) * measure

    def volume_residual_action(self, state, state_dt, test, time: Constant, measure=dx):
        r"""Return the slab-interior primal residual action ``rho_n^V(test)``.

        Every scalar PDE enters the global DWR estimator and the bubble/cone
        recovery through this one weak residual action.  On a slab ``I_n``,

        .. math::

           \rho_n^V(v)=\int_\Omega \bigl[f-u_{h,t}\bigr]v
           -\kappa\nabla u_h\cdot\nabla v\,\mathrm dx

        for heat, but an advection--diffusion or BBM problem supplies its own
        form here.  ``state_dt`` is the already reconstructed DG derivative
        ``partial_t u_h``; consequently this method must *not* use
        :class:`irksome.Dt`.

        The generic estimator never assumes a particular strong residual such
        as ``f-u_t+kappa*Delta(u)``.  It only evaluates this weak action on
        bubble and facet-cone test functions.
        """
        return -self.time_mass_action(state_dt, test, measure) - self.spatial_residual(
            state, test, time, measure=measure
        )

    def temporal_residual_action(self, state_left, previous_right, test, measure=dx):
        r"""Return the left-DG-interface action ``rho_n^T(test)``.

        For a standard parabolic mass term this is

        .. math::

           \rho_n^T(v)=-\bigl(u_h(t_{n-1}^+)-u_h(t_{n-1}^-),v\bigr)_\Omega.

        Equations with a different time mass operator, such as BBM with
        ``(I-partial_xx)u_t``, override this action.  This is essential: using
        the heat jump for BBM would give a mathematically incorrect local DWR
        decomposition.
        """
        return -self.time_mass_action(state_left - previous_right, test, measure)

    def volume_residual_derivative_action(
        self, state, state_dt, increment, increment_dt, dual, time: Constant,
        *, cell_weight=1.0, measure=dx,
    ):
        r"""Return ``rho'_u(u)(increment, dual)`` on slab interiors.

        This extra contract is required only by the symmetric nonlinear DWR
        identity.  It is deliberately explicit: differentiating a weak form
        after inserting a discontinuous cell marker would also differentiate
        that marker and would no longer represent the cell partition of the
        original functional.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the nonlinear DWR derivative action."
        )

    def temporal_residual_derivative_action(
        self, increment_left, increment_previous_right, dual,
        *, cell_weight=1.0, measure=dx,
    ):
        """Return the derivative of the left-DG-interface residual."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the nonlinear temporal derivative action."
        )

    def goal_derivative_action(
        self, mesh: Mesh, terminal_state, terminal_increment,
        *, cell_weight=1.0, measure=dx,
    ):
        """Return ``J'(terminal_state)(terminal_increment)`` cellwise."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the goal derivative action."
        )

    def adjoint_residual(
        self,
        dual: Function,
        test,
        time: Constant,
        *,
        linearisation_state=None,
    ):
        r"""Build a linear scalar adjoint through UFL form differentiation.

        The default is valid when ``A(u;v)`` is linear in ``u``.  Nonlinear
        PDEs must override it with a Jacobian evaluated at the saved primal
        state, or use a fully annotated firedrake-adjoint workflow.
        """
        if not self.spatial_operator_is_linear:
            raise NotImplementedError(
                "A nonlinear PDE needs an adjoint linearised at its primal state."
            )
        return self.time_mass_action(Dt(dual), test) + firedrake_spatial_adjoint_form(
            self.spatial_residual, dual, dual, time
        )

    @abstractmethod
    def terminal_adjoint_condition(self, mesh: Mesh):
        """Return terminal datum ``z(T)=J'_u(u(T))`` for the chosen goal."""

    def terminal_adjoint_state(self, V, name: str, *, terminal_primal: Function | None = None) -> Function:
        r"""Return the terminal dual coefficient in the PDE time-mass pairing.

        For the standard parabolic ``L2`` time mass this is simply the
        interpolated terminal datum.  A Sobolev-in-time-mass equation such as
        BBM overrides this method and computes the corresponding Riesz
        representative instead.
        """
        state = Function(V, name=name)
        state.interpolate(self.terminal_adjoint_condition(V.mesh()))
        for bc in self.boundary_conditions(V):
            bc.apply(state)
        return state

    @abstractmethod
    def goal_functional(self, mesh: Mesh, terminal_state: Function):
        """Return scalar terminal goal ``J(u_h)`` as a UFL zero-form."""

    def boundary_conditions(self, V) -> list[Any]:
        """Return primal/adjoint homogeneous boundary conditions by default."""
        return []

    def exact_goal_value(self, final_time: float) -> float | None:
        """Return analytic ``J(u)`` when available; ``None`` omits effectivity."""
        return None

    def source(self, mesh: Mesh, time: Constant):
        """Return forcing when the problem chooses to expose one explicitly."""
        return Constant(0.0)


def firedrake_spatial_adjoint_form(
    spatial_residual: Callable[[Function, Any, Constant], Any],
    linearisation_state: Function,
    dual: Function,
    time: Constant,
):
    r"""Build ``A'_u(u)^* z`` with Firedrake's variational-adjoint operators.

    If ``A(u;v)`` is supplied by ``spatial_residual``, the returned one-form is

    .. math::

       \operatorname{action}\left((\partial_uA(u))^*,z\right).

    It is intentionally a form helper rather than a separate hand-derived
    matrix.  A nonlinear problem may evaluate it with a slab-specific primal
    state before passing it to its Irksome adjoint residual.
    """
    V = linearisation_state.function_space()
    primal_test = TestFunction(V)
    primal_increment = TrialFunction(V)
    spatial_form = spatial_residual(linearisation_state, primal_test, time)
    return action(adjoint(derivative(spatial_form, linearisation_state, primal_increment)), dual)


class HeatEquationProblem(TransientDWRProblem):
    r"""Manufactured heat problem with a regularised interior point QoI.

    The heat equation is ``u_t-kappa Delta(u)=f`` with exact state
    ``u=e^{-t}sin(pi*x)sin(pi*y)``.  Its terminal functional is

    .. math::

       J_\epsilon(u)=\int_\Omega \psi_\epsilon u(T)\,dx,
       \qquad
       \psi_\epsilon=\frac{1}{\pi\epsilon^2}
       \exp[-|x-x_0|^2/\epsilon^2].

    It is a bounded, mesh-independent approximation to point evaluation at
    ``(x0,y0)``.  Consequently, it produces a sharply localised terminal dual
    while retaining an analytic-reference QoI for effectivity studies.
    """

    name = "manufactured_heat"
    spatial_operator_is_linear = True

    def __init__(
        self,
        kappa: float = 1.0,
        sensor_x: float = 0.72,
        sensor_y: float = 0.68,
        sensor_radius: float = 0.06,
    ):
        if kappa <= 0.0 or sensor_radius <= 0.0:
            raise ValueError("kappa and sensor_radius must be positive.")
        if not 0.0 < sensor_x < 1.0 or not 0.0 < sensor_y < 1.0:
            raise ValueError("The sensor centre must lie strictly inside the unit square.")
        self.kappa = float(kappa)
        self.sensor_x = float(sensor_x)
        self.sensor_y = float(sensor_y)
        self.sensor_radius = float(sensor_radius)

    def make_mesh(self, nx: int, ny: int) -> Mesh:
        """Create a Netgen mesh so marked slab-local refinement is available."""
        return create_adaptive_unit_square_mesh(nx, ny)

    def exact_solution(self, mesh: Mesh, time: Constant):
        """Return ``u(x,t)=e^{-t}sin(pi x)sin(pi y)``."""
        x, y = SpatialCoordinate(mesh)
        return exp(-time) * sin(pi * x) * sin(pi * y)

    def source(self, mesh: Mesh, time: Constant):
        """Return ``f=(2*kappa*pi^2-1)e^{-t}sin(pi x)sin(pi y)``."""
        x, y = SpatialCoordinate(mesh)
        return Constant(2.0 * self.kappa * pi**2 - 1.0) * exp(-time) * sin(pi * x) * sin(pi * y)

    def sensor(self, mesh: Mesh):
        r"""Return the normalised Gaussian terminal weight ``psi_epsilon``."""
        x, y = SpatialCoordinate(mesh)
        radius = Constant(self.sensor_radius)
        distance_squared = (
            (x - Constant(self.sensor_x)) ** 2
            + (y - Constant(self.sensor_y)) ** 2
        )
        return exp(-distance_squared / radius**2) / (pi * radius**2)

    def initial_condition(self, mesh: Mesh):
        """Use the manufactured state at ``t=0`` as the initial condition."""
        return self.exact_solution(mesh, Constant(0.0))

    def spatial_residual(self, state: Function, test, time: Constant, measure=dx):
        """Return ``A(u;v)=(kappa grad u,grad v)-(f,v)`` without ``u_t``."""
        # ``state`` may be a reconstructed DG-in-time UFL sum rather than a
        # concrete Function.  The recovery test always retains its domain.
        mesh = test.ufl_domain()
        return (
            Constant(self.kappa) * inner(grad(state), grad(test)) * measure
            - inner(self.source(mesh, time), test) * measure
        )

    def adjoint_residual(
        self,
        dual: Function,
        test,
        time: Constant,
        *,
        linearisation_state=None,
    ):
        """Return the reversed-time heat adjoint ``z_tau-kappa Delta z=0``.

        The spatial term is constructed through Firedrake/UFL
        ``derivative -> adjoint -> action`` rather than being separately
        transcribed.  For heat it reduces to the familiar symmetric gradient
        form; another PDE can replace ``spatial_residual`` and reuse the same
        variational-adjoint pattern.
        """
        return (
            inner(Dt(dual), test) * dx
            + firedrake_spatial_adjoint_form(self.spatial_residual, dual, dual, time)
        )

    def boundary_conditions(self, V) -> list[Any]:
        """Impose ``u=z=0`` on the boundary, consistent with the exact state."""
        return [DirichletBC(V, 0.0, "on_boundary")]

    def terminal_adjoint_condition(self, mesh: Mesh):
        r"""Set ``z(T)=J'_\epsilon(u)=psi_epsilon`` for the terminal QoI."""
        return self.sensor(mesh)

    def goal_functional(self, mesh: Mesh, terminal_state: Function):
        r"""Return ``J_epsilon(u_h)=int_Omega psi_epsilon*u_h(T) dx``."""
        return self.sensor(mesh) * terminal_state * dx

    def exact_goal_value(self, final_time: float) -> float:
        """Evaluate the smooth fixed QoI against the exact terminal state."""
        nodes, weights = np.polynomial.legendre.leggauss(100)
        coordinates = 0.5 * (nodes + 1.0)
        quadrature = 0.5 * weights
        x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
        weight_2d = np.outer(quadrature, quadrature)
        sensor = np.exp(
            -((x - self.sensor_x) ** 2 + (y - self.sensor_y) ** 2)
            / self.sensor_radius**2
        ) / (np.pi * self.sensor_radius**2)
        exact_terminal = (
            np.exp(-final_time) * np.sin(np.pi * x) * np.sin(np.pi * y)
        )
        return float(np.sum(weight_2d * sensor * exact_terminal))


class BBMTravellingWaveProblem(TransientDWRProblem):
    r"""Forced periodic BBM with a smooth terminal sensor.

    The manufactured state is ``sin(2*pi*(x-c*t))``.  It makes the nonlinear
    DWR effectivity check reproducible while retaining BBM's non-standard
    ``H1`` time mass and nonlinear transport term.
    """

    name = "bbm_forced_periodic_wave"
    requires_primal_linearisation = True
    spatial_refinement_mode = "uniform_slab"
    supports_nonlinear_error_identity = True

    def __init__(self, speed: float = 0.35, sensor_center: float = 0.65, sensor_radius: float = 0.06):
        if sensor_radius <= 0.0:
            raise ValueError("sensor_radius must be positive.")
        self.speed = float(speed)
        self.sensor_center = float(sensor_center)
        self.sensor_radius = float(sensor_radius)

    def make_mesh(self, nx: int, ny_ignored: int = 1) -> Mesh:
        """Return a periodic one-dimensional mesh; ``ny_ignored`` preserves the common API."""
        return PeriodicIntervalMesh(int(nx), 1.0)

    def refine_slab_mesh(self, mesh: Mesh, markers) -> Mesh:
        r"""Uniformly refine one marked periodic slab while preserving its seam.

        Firedrake's local Netgen refiner is unavailable for
        :class:`PeriodicIntervalMesh`.  Refining the complete marked slab is
        the conservative periodic alternative: adjacent slabs may still have
        different meshes, and their interface is still handled by the BBM
        ``H1``-mass ``P/P_star`` pair.
        """
        cells = int(markers.function_space().node_count)
        return PeriodicIntervalMesh(2 * cells, 1.0)

    def exact_solution(self, mesh: Mesh, time: Constant):
        x, = SpatialCoordinate(mesh)
        return sin(2.0 * pi * (x - Constant(self.speed) * time))

    def source(self, mesh: Mesh, time: Constant):
        r"""Return the forcing obtained by substituting the exact travelling wave."""
        x, = SpatialCoordinate(mesh)
        wave_number = 2.0 * pi
        theta = wave_number * (x - Constant(self.speed) * time)
        return (
            Constant(wave_number * (1.0 - self.speed - wave_number**2 * self.speed)) * cos(theta)
            + Constant(wave_number) * sin(theta) * cos(theta)
        )

    def initial_condition(self, mesh: Mesh):
        return self.exact_solution(mesh, Constant(0.0))

    def time_mass_action(self, trial, test, measure=dx):
        r"""Use the BBM ``H1`` time mass in the primal, jumps, and ``P_star``."""
        return inner(trial, test) * measure + inner(grad(trial), grad(test)) * measure

    def spatial_residual(self, state, test, time: Constant, measure=dx):
        r"""Return ``-(u+u^2/2,v_x)-(f,v)`` with the nonlinear flux intact."""
        mesh = test.ufl_domain()
        return (
            -(state + 0.5 * state**2) * test.dx(0) * measure
            - inner(self.source(mesh, time), test) * measure
        )

    def volume_residual_action(self, state, state_dt, test, time: Constant, measure=dx):
        r"""Evaluate the full nonlinear BBM residual in the global/local DWR terms."""
        mesh = test.ufl_domain()
        return (
            inner(self.source(mesh, time) - state_dt, test) * measure
            - inner(grad(state_dt), grad(test)) * measure
            + (state + 0.5 * state**2) * test.dx(0) * measure
        )

    def volume_residual_derivative_action(
        self, state, state_dt, increment, increment_dt, dual, time: Constant,
        *, cell_weight=1.0, measure=dx,
    ):
        r"""Differentiate the BBM primal residual in its state argument."""
        return cell_weight * (
            -inner(increment_dt, dual)
            - inner(grad(increment_dt), grad(dual))
            + (1.0 + state) * increment * dual.dx(0)
        ) * measure

    def temporal_residual_derivative_action(
        self, increment_left, increment_previous_right, dual,
        *, cell_weight=1.0, measure=dx,
    ):
        """Differentiate BBM's H1-mass DG jump residual."""
        jump = increment_left - increment_previous_right
        return -cell_weight * (
            inner(jump, dual) + inner(grad(jump), grad(dual))
        ) * measure

    def goal_derivative_action(
        self, mesh: Mesh, terminal_state, terminal_increment,
        *, cell_weight=1.0, measure=dx,
    ):
        """Derivative of the terminal sensor functional."""
        return cell_weight * self.sensor(mesh) * terminal_increment * measure

    def sensor(self, mesh: Mesh):
        x, = SpatialCoordinate(mesh)
        return exp(-((x - Constant(self.sensor_center)) / Constant(self.sensor_radius)) ** 2)

    def terminal_adjoint_condition(self, mesh: Mesh):
        """Return the terminal QoI derivative in the ordinary ``L2`` pairing."""
        return self.sensor(mesh)

    def terminal_adjoint_state(self, V, name: str, *, terminal_primal: Function | None = None) -> Function:
        r"""Compute the ``H1`` Riesz representative of the terminal QoI derivative.

        The reverse-time BBM solve is initialised by ``z_T`` satisfying
        ``m_BBM(z_T,w)=(psi,w)``, rather than by interpolating ``psi``.
        """
        trial, test = TrialFunction(V), TestFunction(V)
        state = Function(V, name=name)
        solve(
            self.time_mass_action(trial, test) == inner(self.sensor(V.mesh()), test) * dx,
            state,
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        )
        return state

    def adjoint_residual(
        self,
        dual: Function,
        test,
        time: Constant,
        *,
        linearisation_state=None,
    ):
        r"""Return the reverse-time adjoint linearised at ``U_h(t)``.

        ``A'_u(U)^*z=-(1+U)z_x`` for BBM, so the weak reverse-time form is
        ``m_BBM(z_tau,w)-((1+U_h(t))*z_x,w)=0``.
        """
        if linearisation_state is None:
            raise ValueError("BBM adjoint requires the saved primal slab polynomial.")
        return self.time_mass_action(Dt(dual), test) - (
            (1.0 + linearisation_state) * dual.dx(0) * test
        ) * dx

    def goal_functional(self, mesh: Mesh, terminal_state: Function):
        return self.sensor(mesh) * terminal_state * dx

    def exact_goal_value(self, final_time: float) -> float:
        """Integrate the manufactured terminal wave with high-order 1D quadrature."""
        nodes, weights = np.polynomial.legendre.leggauss(128)
        x = 0.5 * (nodes + 1.0)
        sensor = np.exp(-((x - self.sensor_center) / self.sensor_radius) ** 2)
        state = np.sin(2.0 * np.pi * (x - self.speed * float(final_time)))
        return float(0.5 * np.dot(weights, sensor * state))


class BBMSolitaryWaveProblem(BBMTravellingWaveProblem):
    r"""Unforced periodic BBM solitary wave used in Irksome's official demo.

    This input preserves the official physical configuration--a right-moving
    BBM solitary wave on a long periodic interval--but supplies it through the
    generic bubble-DWR interface.  The outer solver intentionally remains
    DG-in-time: its temporal jump recovery and independent slab refinement are
    the contribution tested here, whereas the official forward-only demo uses
    continuous Petrov--Galerkin time stepping to study invariant preservation.
    """

    name = "bbm_unforced_solitary_wave"
    spatial_refinement_mode = "local_periodic"

    def __init__(
        self,
        *,
        length: float = 100.0,
        amplitude_parameter: float = 0.5,
        initial_center: float = 30.0,
        sensor_center: float = 54.0,
        sensor_radius: float = 1.0,
        goal_mode: str = "terminal_sensor",
    ):
        if length <= 0.0 or not 0.0 < amplitude_parameter < 1.0:
            raise ValueError("Require length>0 and a BBM amplitude parameter in (0, 1).")
        if sensor_radius <= 0.0:
            raise ValueError("sensor_radius must be positive.")
        if goal_mode not in {"terminal_sensor", "invariant_i2"}:
            raise ValueError("goal_mode must be 'terminal_sensor' or 'invariant_i2'.")
        self.length = float(length)
        self.amplitude_parameter = float(amplitude_parameter)
        self.initial_center = float(initial_center)
        self.sensor_center = float(sensor_center)
        self.sensor_radius = float(sensor_radius)
        self.goal_mode = str(goal_mode)

    @property
    def wave_speed(self) -> float:
        r"""Return the exact solitary-wave speed ``1/(1-c^2)``."""
        return 1.0 / (1.0 - self.amplitude_parameter**2)

    @property
    def goal_label(self) -> str:
        return "I2" if self.goal_mode == "invariant_i2" else "terminal_sensor"

    def make_mesh(self, nx: int, ny_ignored: int = 1) -> Mesh:
        """Use a long periodic interval so the solitary-wave tails miss its seam."""
        return PeriodicIntervalMesh(int(nx), self.length)

    def refine_slab_mesh(self, mesh: Mesh, markers) -> Mesh:
        """Locally bisect marked cells while preserving the periodic seam."""
        from .mark_refine import refine_marked_periodic_interval_mesh

        return refine_marked_periodic_interval_mesh(mesh, markers, self.length)

    def exact_solution(self, mesh: Mesh, time: Constant):
        """Return Irksome's exact right-moving BBM solitary wave."""
        x, = SpatialCoordinate(mesh)
        c = Constant(self.amplitude_parameter)
        phase = 0.5 * (
            c * x
            - c * time / (1.0 - c**2)
            - c * Constant(self.initial_center)
        )
        sech = 2.0 / (exp(phase) + exp(-phase))
        return 3.0 * c**2 / (1.0 - c**2) * sech**2

    def source(self, mesh: Mesh, time: Constant):
        """The solitary wave solves the unforced BBM equation."""
        return Constant(0.0)

    def terminal_adjoint_state(self, V, name: str, *, terminal_primal: Function | None = None) -> Function:
        r"""Return the BBM-mass Riesz representative of the selected QoI.

        For ``I2=integral(u^2+u_x^2)``, ``I2'(u)[v]=2*m_BBM(u,v)``.
        Therefore its terminal dual datum is exactly ``2*u_h(T)`` in the
        BBM ``H1`` mass pairing--no auxiliary Riesz solve is needed.
        """
        if self.goal_mode == "terminal_sensor":
            return super().terminal_adjoint_state(V, name, terminal_primal=terminal_primal)
        if terminal_primal is None:
            raise ValueError("The I2 terminal adjoint requires the computed terminal primal state.")
        state = Function(V, name=name)
        state.interpolate(2.0 * terminal_primal)
        for bc in self.boundary_conditions(V):
            bc.apply(state)
        return state

    def goal_functional(self, mesh: Mesh, terminal_state: Function):
        if self.goal_mode == "terminal_sensor":
            return super().goal_functional(mesh, terminal_state)
        return (terminal_state**2 + inner(grad(terminal_state), grad(terminal_state))) * dx

    def goal_derivative_action(
        self, mesh: Mesh, terminal_state, terminal_increment,
        *, cell_weight=1.0, measure=dx,
    ):
        if self.goal_mode == "terminal_sensor":
            return super().goal_derivative_action(
                mesh, terminal_state, terminal_increment,
                cell_weight=cell_weight, measure=measure,
            )
        return 2.0 * cell_weight * (
            terminal_state * terminal_increment
            + inner(grad(terminal_state), grad(terminal_increment))
        ) * measure

    def exact_goal_value(self, final_time: float) -> float:
        """Compute the selected exact terminal QoI by high-order quadrature."""
        nodes, weights = np.polynomial.legendre.leggauss(512)
        x = 0.5 * self.length * (nodes + 1.0)
        c = self.amplitude_parameter
        phase = 0.5 * (
            c * x - c * float(final_time) / (1.0 - c**2) - c * self.initial_center
        )
        state = 3.0 * c**2 / (1.0 - c**2) * (2.0 / (np.exp(phase) + np.exp(-phase)))**2
        if self.goal_mode == "invariant_i2":
            derivative_x = -c * state * np.tanh(phase)
            return float(0.5 * self.length * np.dot(weights, state**2 + derivative_x**2))
        sensor = np.exp(-((x - self.sensor_center) / self.sensor_radius) ** 2)
        return float(0.5 * self.length * np.dot(weights, sensor * state))


class BBMFiniteIntervalSolitaryWaveProblem(BBMSolitaryWaveProblem):
    r"""Solitary BBM on a long interval with quiet far-field boundaries.

    This is the locally-refinable counterpart of the official periodic BBM
    configuration.  On ``[0,100]`` over ``0<=t<=18`` the chosen solitary
    wave remains far from both endpoints, so homogeneous Dirichlet data model
    an undisturbed far field while allowing marked 1D cells to be bisected.
    """

    name = "bbm_solitary_finite_interval"
    spatial_refinement_mode = "local_interval"

    def make_mesh(self, nx: int, ny_ignored: int = 1) -> Mesh:
        return IntervalMesh(int(nx), self.length, reorder=False)

    def boundary_conditions(self, V) -> list[Any]:
        """Use zero elevation at the two far-field interval endpoints."""
        return [DirichletBC(V, 0.0, "on_boundary")]

    def refine_slab_mesh(self, mesh: Mesh, markers) -> Mesh:
        """Bisect only the marked cells on this slab's nonperiodic mesh."""
        from .mark_refine import refine_marked_interval_mesh

        return refine_marked_interval_mesh(mesh, markers)

    def exact_goal_value(self, final_time: float) -> float | None:
        r"""Avoid a false periodic reference for the finite-interval ``I2`` run.

        Homogeneous Dirichlet data conserve ``I2`` for a smooth compatible
        finite-interval BBM solution.  The imported solitary wave, however,
        has exponentially small but nonzero endpoint traces before the
        discrete Dirichlet conditions clamp them.  Its periodic/infinite-line
        analytic value is therefore not the exact QoI of this modified
        initial-boundary-value problem.  The CSV's ``J_change`` is the honest
        numerical conservation diagnostic in this local-refinement mode.
        """
        if self.goal_mode == "invariant_i2":
            return None
        return super().exact_goal_value(final_time)
