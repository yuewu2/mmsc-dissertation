from __future__ import annotations

"""Automated DWR localisation for a moving 1D pulse.

The PDE is

    u_t - epsilon*u_xx + velocity*u_x = f  on (0, 1) x (0, T),
    u(0, t) = u(1, t) = 0.

The manufactured solution is a Gaussian pulse moving from left to right,
multiplied by x(1-x) so that the homogeneous boundary condition is exact.

Discretisation
--------------
* primal and numerical dual: CG(p) in x, Irksome DG0 in t;
* enriched dual: CG(p+1) in x, Irksome DG1 in t;
* recovered cell residual: DG1 in x, P1 in reference time;
* recovered endpoint residual: broken DG1 endpoint values, P1 in time;
* recovered temporal-interface residual: DG1 in x.

The code writes both ordinary Firedrake PVD output of the terminal state and
a two-dimensional x-t VTU file.  Each time slab owns its own spatial mesh, so
the displayed rectangles K x I_n are genuinely slab-local rather than the
extrusion of one globally refined spatial mesh.

The terminal QoI can be a hard collection window, one smooth Gaussian
detector, or two smooth Gaussian detectors.  The latter two choices make it
easy to separate physical detector sensitivity from hard-window edge effects.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from firedrake import (
    COMM_WORLD,
    CellDiameter,
    Cofunction,
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    Mesh,
    MixedFunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    VTKFile,
    assemble,
    dS,
    ds,
    dx,
    exp,
    grad,
    interpolate,
    inner,
    solve,
)
from firedrake.adjoint import (
    Control,
    compute_derivative,
    continue_annotation,
    pause_annotation,
)
from firedrake.mesh import plex_from_cell_list
from firedrake.petsc import PETSc
from irksome import Dt, TimeStepper
from pyadjoint import Block, annotate_tape, get_working_tape, stop_annotating
from ufl import And, conditional

from automated_DWR.problem import TransientDWRProblem

from heat_dwr_irksome import (
    _copy_function,
    _dg_scheme,
    _expr_linear_combination,
    _gauss_rule,
    _interpolate_expr,
    _lagrange_values,
    _time_nodes,
    eval_slab_dt_expr,
    eval_slab_expr,
)
from heat_dwr_mesh import mark_spacetime_cells


@dataclass(frozen=True)
class MovingPulseOptions:
    max_it: int = 8
    tolerance: float = 1.0e-4

    p: int = 1
    enriched_time_degree: int = 1
    recovery_space_degree: int = 1
    recovery_time_degree: int = 1
    time_quadrature_points: int = 5
    spatial_quadrature_degree: int = 24

    epsilon: float = 2.0e-2
    velocity: float = 0.6
    beta: float = 100.0
    pulse_x0: float = 0.2

    goal_a: float = 0.75
    goal_b: float = 0.90
    qoi: str = "terminal-window"
    sensor_center: float = 0.80
    sensor_radius: float = 0.05
    left_sensor_center: float = 0.70
    right_sensor_center: float = 0.90

    theta_spacetime: float = 0.2
    time_slab_marked_fraction: float = 0.05
    enable_space_refinement: bool = True
    enable_time_refinement: bool = True

    write_vtk: bool = True
    output_prefix: str = "output/advection_diffusion_1d/automated"
    adjoint_backend: str = "firedrake-adjoint"
    solver_parameters: dict[str, Any] | None = None
    recovery_solver_parameters: dict[str, Any] | None = None


class MovingPulseProblem(TransientDWRProblem):
    r"""The legacy moving-pulse data expressed through the generic input API.

    The class proves that the one-dimensional endpoint backend does not need
    to know the advection--diffusion formula.  A new compatible one-dimensional
    linear PDE subclasses :class:`TransientDWRProblem` instead of editing the
    recovery, marking, or refinement code.
    """

    name = "moving_advection_diffusion_pulse"
    spatial_operator_is_linear = True

    def __init__(self, options: MovingPulseOptions):
        self.options = options

    def make_mesh(self, nx: int, ny_ignored: int = 1) -> Mesh:
        """Return the initial interval mesh used by the slabwise driver."""
        return create_interval_mesh(np.linspace(0.0, 1.0, int(nx) + 1))

    def initial_condition(self, mesh: Mesh):
        """Return the manufactured Gaussian pulse at ``t=0``."""
        return exact_solution_expr(mesh, Constant(0.0), self.options)

    def source(self, mesh: Mesh, time: Constant):
        """Return the manufactured forcing for this particular experiment."""
        return source_expr(mesh, time, self.options)

    def spatial_residual(self, state, test, time: Constant, measure=dx):
        r"""Return ``eps*(u_x,v_x)+(b*u_x,v)-(f,v)``."""
        mesh = test.ufl_domain()
        return (
            Constant(self.options.epsilon) * inner(grad(state), grad(test)) * measure
            + Constant(self.options.velocity) * grad(state)[0] * test * measure
            - self.source(mesh, time) * test * measure
        )

    def terminal_adjoint_condition(self, mesh: Mesh):
        """Return the terminal datum associated with the selected QoI."""
        return goal_weight_expr(mesh, self.options)

    def goal_functional(self, mesh: Mesh, terminal_state):
        """Return the selected terminal detector/collection QoI."""
        return goal_weight_expr(mesh, self.options) * terminal_state * dx

    def boundary_conditions(self, V) -> list[DirichletBC]:
        """Apply the homogeneous traces used by primal and adjoint solves."""
        return [DirichletBC(V, 0.0, "on_boundary")]

    def exact_goal_value(self, final_time: float) -> float:
        """Return the manufactured QoI used for effectivity verification."""
        return exact_goal_value(final_time, self.options)


def _validate_options(opts: MovingPulseOptions) -> None:
    if opts.max_it < 1:
        raise ValueError("max_it must be at least one.")
    if opts.tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if opts.p < 1:
        raise ValueError("p must be at least one.")
    if opts.enriched_time_degree < 1:
        raise ValueError("The enriched dual must be at least DG1 in time.")
    if opts.recovery_space_degree < 0 or opts.recovery_time_degree < 0:
        raise ValueError("Recovery degrees must be nonnegative.")
    if opts.time_quadrature_points < 1:
        raise ValueError("time_quadrature_points must be positive.")
    if opts.spatial_quadrature_degree < 1:
        raise ValueError("spatial_quadrature_degree must be positive.")
    if opts.adjoint_backend not in {"firedrake-adjoint", "ufl"}:
        raise ValueError(
            "adjoint_backend must be 'firedrake-adjoint' or 'ufl'."
        )
    if opts.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if opts.beta <= 0.0:
        raise ValueError("beta must be positive.")
    if not 0.0 < opts.goal_a < opts.goal_b < 1.0:
        raise ValueError("Require 0 < goal_a < goal_b < 1.")
    if opts.qoi not in {
        "terminal-window",
        "terminal-gaussian",
        "terminal-double-gaussian",
    }:
        raise ValueError("Unknown terminal QoI selection.")
    if opts.sensor_radius <= 0.0:
        raise ValueError("sensor_radius must be positive.")
    if not all(
        0.0 < centre < 1.0
        for centre in (
            opts.sensor_center,
            opts.left_sensor_center,
            opts.right_sensor_center,
        )
    ):
        raise ValueError("All Gaussian sensor centres must lie in (0, 1).")
    if not 0.0 < opts.theta_spacetime <= 1.0:
        raise ValueError("theta_spacetime must lie in (0, 1].")
    if not 0.0 <= opts.time_slab_marked_fraction <= 1.0:
        raise ValueError("time_slab_marked_fraction must lie in [0, 1].")


def create_interval_mesh(x_nodes: np.ndarray) -> Mesh:
    """Build a nonuniform interval mesh from sorted physical coordinates."""
    nodes = np.asarray(x_nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("x_nodes must contain at least two coordinates.")
    if not np.all(np.diff(nodes) > 0.0):
        raise ValueError("x_nodes must be strictly increasing.")
    if abs(nodes[0]) > 1.0e-14 or abs(nodes[-1] - 1.0) > 1.0e-14:
        raise ValueError("The interval mesh must cover [0, 1].")

    cells = np.column_stack(
        [np.arange(nodes.size - 1), np.arange(1, nodes.size)]
    ).astype(np.int32)
    coordinates = nodes.reshape((-1, 1))
    plex = plex_from_cell_list(1, cells, coordinates, COMM_WORLD)
    return Mesh(plex, name="moving_pulse_interval")


def exact_solution_expr(mesh: Mesh, t, opts: MovingPulseOptions):
    x = SpatialCoordinate(mesh)[0]
    r = x - Constant(opts.pulse_x0) - Constant(opts.velocity) * t
    return x * (1.0 - x) * exp(-Constant(opts.beta) * r * r)


def source_expr(mesh: Mesh, t, opts: MovingPulseOptions):
    """Manufactured f=u_t-epsilon*u_xx+velocity*u_x."""
    x = SpatialCoordinate(mesh)[0]
    beta = Constant(opts.beta)
    velocity = Constant(opts.velocity)
    epsilon = Constant(opts.epsilon)
    r = x - Constant(opts.pulse_x0) - velocity * t
    g = x * (1.0 - x)
    E = exp(-beta * r * r)

    u_t = 2.0 * beta * velocity * r * g * E
    u_x = E * ((1.0 - 2.0 * x) - 2.0 * beta * r * g)
    u_xx = E * (
        -2.0
        - 2.0 * beta * g
        - 4.0 * beta * r * (1.0 - 2.0 * x)
        + 4.0 * beta * beta * r * r * g
    )
    return u_t - epsilon * u_xx + velocity * u_x


def goal_weight_expr(mesh: Mesh, opts: MovingPulseOptions):
    """Return the selected hard-window or smooth terminal sensor weight."""
    x = SpatialCoordinate(mesh)[0]
    if opts.qoi == "terminal-window":
        inside = And(x >= Constant(opts.goal_a), x <= Constant(opts.goal_b))
        return conditional(inside, 1.0, 0.0)

    def gaussian(centre: float):
        return exp(-((x - Constant(centre)) / Constant(opts.sensor_radius)) ** 2)

    if opts.qoi == "terminal-gaussian":
        return gaussian(opts.sensor_center)
    return gaussian(opts.left_sensor_center) + gaussian(opts.right_sensor_center)


def exact_goal_value(T: float, opts: MovingPulseOptions) -> float:
    """Accurate one-dimensional Gauss integration of the manufactured goal."""
    points, weights = np.polynomial.legendre.leggauss(100)
    if opts.qoi == "terminal-window":
        x = 0.5 * (opts.goal_b - opts.goal_a) * points + 0.5 * (
            opts.goal_a + opts.goal_b
        )
        w = 0.5 * (opts.goal_b - opts.goal_a) * weights
        qoi_weight = np.ones_like(x)
    else:
        x = 0.5 * (points + 1.0)
        w = 0.5 * weights
        qoi_weight = np.exp(-((x - opts.sensor_center) / opts.sensor_radius) ** 2)
        if opts.qoi == "terminal-double-gaussian":
            qoi_weight = np.exp(
                -((x - opts.left_sensor_center) / opts.sensor_radius) ** 2
            ) + np.exp(-((x - opts.right_sensor_center) / opts.sensor_radius) ** 2)
    centre = opts.pulse_x0 + opts.velocity * float(T)
    values = x * (1.0 - x) * np.exp(-opts.beta * (x - centre) ** 2)
    return float(np.dot(w, qoi_weight * values))


def solve_forward_advection_diffusion(
    V,
    ts: np.ndarray,
    mesh: Mesh,
    initial_expr,
    degree: int,
    opts: MovingPulseOptions,
    source_factory=None,
    adjoint_operator: bool = False,
    project_initial: bool = False,
    name_prefix: str = "U",
) -> dict[str, Any]:
    """Irksome DG-in-time solve for the primal or reversed adjoint operator."""
    N = len(ts) - 1
    if N < 1:
        raise ValueError("At least one time slab is required.")
    params = opts.solver_parameters or {"ksp_type": "preonly", "pc_type": "lu"}

    t = Constant(float(ts[0]))
    dt = Constant(float(ts[1] - ts[0]))
    y = Function(V, name=f"{name_prefix}_solution")
    v = TestFunction(V)
    bcs = [DirichletBC(V, 0.0, "on_boundary")]
    if project_initial:
        initial_trial = TrialFunction(V)
        solve(
            inner(initial_trial, v) * dx == inner(initial_expr, v) * dx,
            y,
            bcs=bcs,
            solver_parameters=params,
        )
    else:
        y.interpolate(initial_expr)
    epsilon = Constant(opts.epsilon)
    velocity = Constant(opts.velocity)

    F = inner(Dt(y), v) * dx + epsilon * inner(grad(y), grad(v)) * dx
    if adjoint_operator:
        # a(v,y)=epsilon*(v_x,y_x)+velocity*(v_x,y), the weak adjoint.
        F += velocity * grad(v)[0] * y * dx
    else:
        F += velocity * grad(y)[0] * v * dx
    if source_factory is not None:
        F -= source_factory(mesh, t, opts) * v * dx

    time_nodes = _time_nodes(int(degree))
    solution_nodes: list[Function | None] = [None] * (N + 1)
    solution_nodes[0] = _copy_function(y, f"{name_prefix}_node_0")
    slabs: list[dict[str, Any] | None] = [None] * (N + 1)

    for n in range(1, N + 1):
        k_n = float(ts[n] - ts[n - 1])
        t.assign(float(ts[n - 1]))
        dt.assign(k_n)
        previous_right = _copy_function(y, f"{name_prefix}_previous_right_{n}")

        stepper = TimeStepper(
            F,
            _dg_scheme(int(degree)),
            t,
            dt,
            y,
            bcs=bcs,
            solver_parameters=params,
        )
        stepper.advance()

        coefficients = [
            _copy_function(
                coefficient,
                f"{name_prefix}_DG{degree}_slab_{n}_coefficient_{i}",
            )
            for i, coefficient in enumerate(stepper.stages.subfunctions)
        ]
        right_expr = _expr_linear_combination(
            coefficients,
            _lagrange_values(time_nodes, 1.0),
        )
        right_value = _interpolate_expr(V, right_expr, f"{name_prefix}_node_{n}")
        solution_nodes[n] = _copy_function(
            right_value,
            f"{name_prefix}_node_{n}_saved",
        )
        slabs[n] = {
            "degree": int(degree),
            "orientation": "forward",
            "coeffs": coefficients,
            "prev_right": previous_right,
            "right": solution_nodes[n],
        }
    return {"U_nodes": solution_nodes, "slabs": slabs}


@dataclass
class SlabInterfaceTransfer:
    r"""The forward interpolation and exact discrete reverse transfer.

    The forward state transfer is the assembled interpolation matrix

    .. math:: \boldsymbol u_{n+1}^+=P_n\boldsymbol u_n^-.

    Since the temporal residual is paired in the spatial ``L2`` inner
    product, the corresponding discrete-adjoint operation is **not** ``P_n``
    applied backward.  It is

    .. math::

       \boldsymbol z_n^-
       =M_n^{-1}P_n^T M_{n+1}\boldsymbol z_{n+1}^+,

    where ``M_n`` is the mass matrix on ``V_n``.  Storing these operators once
    per interface makes the primal and dual mesh changes mathematically
    consistent.
    """

    source_space: Any
    target_space: Any
    interpolation_matrix: Any
    source_mass: Any
    target_mass: Any
    source_mass_ksp: Any
    source_mass_bc: DirichletBC

    def forward(self, source: Function, name: str) -> Function:
        """Apply ``u_target=P*u_source`` and enforce homogeneous traces."""
        target = Function(self.target_space, name=name)
        with source.dat.vec_ro as source_vec, target.dat.vec_wo as target_vec:
            self.interpolation_matrix.petscmat.mult(source_vec, target_vec)
        DirichletBC(self.target_space, 0.0, "on_boundary").apply(target)
        return target

    def adjoint(self, target_dual: Function, name: str) -> Function:
        """Apply the mass-inner-product adjoint ``M_s^{-1} P^T M_t z``."""
        mass_weighted_target = Function(
            self.target_space, name=f"{name}_mass_weighted_target"
        )
        source_rhs = Function(self.source_space, name=f"{name}_rhs")
        source_dual = Function(self.source_space, name=name)
        with (
            target_dual.dat.vec_ro as target_vec,
            mass_weighted_target.dat.vec_wo as weighted_vec,
        ):
            self.target_mass.petscmat.mult(target_vec, weighted_vec)
        with (
            mass_weighted_target.dat.vec_ro as weighted_vec,
            source_rhs.dat.vec_wo as rhs_vec,
        ):
            self.interpolation_matrix.petscmat.multTranspose(weighted_vec, rhs_vec)
        # The source state has homogeneous essential traces.  Solve the mass
        # Riesz map on that constrained space; zeroing the result *after* an
        # unconstrained mass solve would not be the discrete adjoint.
        self.source_mass_bc.apply(source_rhs)
        with source_rhs.dat.vec_ro as rhs_vec, source_dual.dat.vec_wo as source_vec:
            self.source_mass_ksp.solve(rhs_vec, source_vec)
        DirichletBC(self.source_space, 0.0, "on_boundary").apply(source_dual)
        return source_dual


class SlabInterfaceTransferBlock(Block):
    r"""Teach ``pyadjoint`` the exact mesh-change operation ``P``.

    The forward mesh transfer is assembled outside UFL, so it is invisible to
    ``firedrake-adjoint`` unless we register this block explicitly.  Its
    reverse action returns the covector corresponding to
    ``M_s^{-1} P^T M_t`` rather than a reverse nodal interpolation.
    """

    def __init__(self, transfer: SlabInterfaceTransfer, source: Function):
        super().__init__()
        self.transfer = transfer
        self.source_boundary_nodes = DirichletBC(
            transfer.source_space, 0.0, "on_boundary"
        ).nodes
        self.target_boundary_nodes = DirichletBC(
            transfer.target_space, 0.0, "on_boundary"
        ).nodes
        self.add_dependency(source)

    def evaluate_adj_component(
        self,
        inputs,
        adj_inputs,
        block_variable,
        idx,
        prepared=None,
    ):
        # pyadjoint propagates algebraic cotangents, not L2-representing
        # functions.  Thus its reverse operation is P.T on covectors.  The
        # equivalent L2 field operation is M_s^-1 P.T M_t = P_star.
        #
        # The forward transfer fixes boundary dofs to zero.  Those dofs are
        # not differentiable inputs or outputs, hence their cotangent entries
        # must be discarded before and after P.T.
        target_cotangent = adj_inputs[0].copy(deepcopy=True)
        target_cotangent.dat.data[self.target_boundary_nodes] = 0.0
        source_cotangent = Cofunction(self.transfer.source_space.dual())
        with (
            target_cotangent.dat.vec_ro as target_vec,
            source_cotangent.dat.vec_wo as source_vec,
        ):
            self.transfer.interpolation_matrix.petscmat.multTranspose(
                target_vec, source_vec
            )
        source_cotangent.dat.data[self.source_boundary_nodes] = 0.0
        return source_cotangent

    def evaluate_tlm_component(
        self,
        inputs,
        tlm_inputs,
        block_variable,
        idx,
        prepared=None,
    ):
        if tlm_inputs[0] is None:
            return None
        return self.transfer.forward(tlm_inputs[0], "tape_slab_interface_tlm")

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        return self.transfer.forward(inputs[0], "tape_slab_interface_recompute")


def tape_slab_interface_forward(
    transfer: SlabInterfaceTransfer,
    source: Function,
    name: str,
) -> Function:
    """Apply ``P`` and register its exact discrete adjoint with the tape."""
    target = transfer.forward(source, name)
    if annotate_tape():
        block = SlabInterfaceTransferBlock(transfer, source)
        get_working_tape().add_block(block)
        block.add_output(target.create_block_variable())
    return target


def build_slab_interface_transfer(
    source_space,
    target_space,
    problem: TransientDWRProblem,
) -> SlabInterfaceTransfer:
    r"""Assemble transfer operators using the PDE's own time mass ``m``.

    For heat/advection--diffusion this reduces to the ``L2`` mass.  Keeping
    ``m`` here makes the discrete transfer adjoint applicable to any linear
    first-order evolution equation accepted by the input contract.
    """
    if source_space.mesh().comm.size != 1:
        raise NotImplementedError(
            "The slabwise 1D demonstrator currently supports one MPI rank."
        )
    interpolation_matrix = assemble(
        interpolate(TrialFunction(source_space), target_space), mat_type="aij"
    )
    source_trial = TrialFunction(source_space)
    source_test = TestFunction(source_space)
    target_trial = TrialFunction(target_space)
    target_test = TestFunction(target_space)
    source_mass_bc = DirichletBC(source_space, 0.0, "on_boundary")
    source_mass = assemble(
        problem.time_mass_action(source_trial, source_test),
        bcs=source_mass_bc,
        mat_type="aij",
    )
    target_mass = assemble(
        problem.time_mass_action(target_trial, target_test), mat_type="aij"
    )
    mass_ksp = PETSc.KSP().create(source_space.mesh().comm)
    mass_ksp.setOperators(source_mass.petscmat)
    mass_ksp.setType("preonly")
    mass_ksp.getPC().setType("lu")
    mass_ksp.setFromOptions()
    mass_ksp.setUp()
    return SlabInterfaceTransfer(
        source_space=source_space,
        target_space=target_space,
        interpolation_matrix=interpolation_matrix,
        source_mass=source_mass,
        target_mass=target_mass,
        source_mass_ksp=mass_ksp,
        source_mass_bc=source_mass_bc,
    )


def _initialise_slab_state(
    state: Function,
    initial_data,
    test,
    bcs: list[DirichletBC],
    project_initial: bool,
    solver_parameters: dict[str, Any],
    problem: TransientDWRProblem,
    name: str,
    interface_transfer: SlabInterfaceTransfer | None = None,
    reverse_transfer: bool = False,
) -> Function:
    """Set a slab's incoming state, including the terminal dual projection."""
    V = state.function_space()
    if isinstance(initial_data, Function):
        if interface_transfer is None:
            raise ValueError("A mesh-changing state needs its interface transfer.")
        if reverse_transfer:
            return interface_transfer.adjoint(initial_data, name)
        return interface_transfer.forward(initial_data, name)
    if project_initial:
        trial = TrialFunction(V)
        solve(
            problem.time_mass_action(trial, test)
            == problem.time_mass_action(initial_data, test),
            state,
            bcs=bcs,
            solver_parameters=solver_parameters,
        )
    else:
        state.interpolate(initial_data)
    return state


def solve_advection_diffusion_slab(
    V,
    mesh: Mesh,
    t_left: float,
    dt_value: float,
    initial_data,
    degree: int,
    opts: MovingPulseOptions,
    problem: TransientDWRProblem,
    *,
    adjoint_operator: bool = False,
    project_initial: bool = False,
    interface_transfer: SlabInterfaceTransfer | None = None,
    reverse_transfer: bool = False,
    name_prefix: str,
) -> dict[str, Any]:
    """Solve one Irksome DG slab on its own spatial mesh.

    The input object supplies the complete primal weak form
    ``m(Dt(u),v)+A(u;v,t)``.  For a reverse slab the generic UFL-derived
    adjoint is used instead.  ``orientation='reverse'`` makes
    :func:`eval_slab_expr` interpret the saved polynomial in physical time.
    """
    params = opts.solver_parameters or {"ksp_type": "preonly", "pc_type": "lu"}
    t = Constant(float(t_left))
    dt = Constant(float(dt_value))
    y = Function(V, name=f"{name_prefix}_solution")
    v = TestFunction(V)
    bcs = problem.boundary_conditions(V)
    y = _initialise_slab_state(
        y,
        initial_data,
        v,
        bcs,
        project_initial,
        params,
        problem,
        f"{name_prefix}_incoming",
        interface_transfer,
        reverse_transfer,
    )
    previous_right = _copy_function(y, f"{name_prefix}_previous_right")

    F = (
        problem.adjoint_residual(y, v, t)
        if adjoint_operator
        else problem.primal_residual(y, v, t)
    )

    stepper = TimeStepper(
        F,
        _dg_scheme(int(degree)),
        t,
        dt,
        y,
        bcs=bcs,
        solver_parameters=params,
    )
    stepper.advance()
    coefficients = [
        _copy_function(coefficient, f"{name_prefix}_coefficient_{i}")
        for i, coefficient in enumerate(stepper.stages.subfunctions)
    ]
    right_expr = _expr_linear_combination(
        coefficients,
        _lagrange_values(_time_nodes(int(degree)), 1.0),
    )
    return {
        "degree": int(degree),
        "orientation": "reverse" if adjoint_operator else "forward",
        "coeffs": coefficients,
        "prev_right": previous_right,
        "right": _interpolate_expr(V, right_expr, f"{name_prefix}_right"),
        "mesh": mesh,
    }


def solve_primal(
    V_by_slab: list[Any | None],
    ts: np.ndarray,
    meshes: list[Mesh | None],
    transfers: list[SlabInterfaceTransfer | None],
    opts: MovingPulseOptions,
    problem: TransientDWRProblem,
) -> dict[str, Any]:
    """March forward with one independently refinable mesh per time slab."""
    N = len(ts) - 1
    solution_nodes: list[Function | None] = [None] * (N + 1)
    slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    incoming: Function | None = None
    for n in range(1, N + 1):
        mesh = meshes[n]
        V = V_by_slab[n]
        initial = (
            problem.initial_condition(mesh)
            if n == 1
            else incoming
        )
        slab = solve_advection_diffusion_slab(
            V,
            mesh,
            float(ts[n - 1]),
            float(ts[n] - ts[n - 1]),
            initial,
            degree=0,
            opts=opts,
            problem=problem,
            interface_transfer=None if n == 1 else transfers[n - 1],
            name_prefix=f"U_slab_{n}",
        )
        slabs[n] = slab
        solution_nodes[n] = _copy_function(slab["right"], f"U_node_{n}")
        incoming = slab["right"]
    solution_nodes[0] = _interpolate_expr(
        V_by_slab[1],
        problem.initial_condition(meshes[1]),
        "U_node_0",
    )
    return {"U_nodes": solution_nodes, "slabs": slabs}


def solve_dual(
    V_by_slab: list[Any | None],
    ts: np.ndarray,
    meshes: list[Mesh | None],
    transfers: list[SlabInterfaceTransfer | None],
    degree: int,
    opts: MovingPulseOptions,
    problem: TransientDWRProblem,
) -> dict[str, Any]:
    """March the adjoint backward, transferring it to each preceding slab.

    With ``tau=T-t``, each call solves the forward-in-``tau`` adjoint on
    ``T_n``.  The resulting polynomial is tagged as reverse so estimator
    evaluations at reference coordinate ``s`` represent physical time
    ``t_{n-1}+s k_n``.
    """
    N = len(ts) - 1
    T = float(ts[-1])
    Z_nodes: list[Function | None] = [None] * (N + 1)
    Z_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    incoming: Function | None = None
    for n in range(N, 0, -1):
        mesh = meshes[n]
        V = V_by_slab[n]
        initial = problem.terminal_adjoint_condition(mesh) if n == N else incoming
        slab = solve_advection_diffusion_slab(
            V,
            mesh,
            T - float(ts[n]),
            float(ts[n] - ts[n - 1]),
            initial,
            degree=degree,
            opts=opts,
            problem=problem,
            adjoint_operator=True,
            project_initial=(n == N),
            interface_transfer=None if n == N else transfers[n],
            reverse_transfer=(n != N),
            name_prefix=f"Z_DG{degree}_slab_{n}",
        )
        Z_slabs[n] = slab
        Z_nodes[n] = _interpolate_expr(
            V,
            eval_slab_expr(slab, 1.0),
            f"Z_node_{n}",
        )
        # The tau-right value is the physical left trace and becomes the
        # incoming data for slab n-1 after nodal transfer there.
        incoming = slab["right"]
    Z_nodes[0] = _interpolate_expr(
        V_by_slab[1], eval_slab_expr(Z_slabs[1], 0.0), "Z_node_0"
    )
    return {"Z_nodes": Z_nodes, "slabs": Z_slabs}


def solve_primal_and_tape_dual(
    V_by_slab: list[Any | None],
    ts: np.ndarray,
    meshes: list[Mesh | None],
    transfers: list[SlabInterfaceTransfer | None],
    degree: int,
    opts: MovingPulseOptions,
    problem: TransientDWRProblem,
    *,
    name_prefix: str,
    return_primal: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    r"""Run a forward DG trajectory and obtain its discrete dual from the tape.

    ``firedrake-adjoint`` differentiates the actual Irksome solve block on
    every slab.  The auxiliary forward solve used for the enriched dual has
    the enriched space/time degree, so its tape adjoint is the desired
    enriched discrete dual.  Mesh changes are represented by
    :class:`SlabInterfaceTransferBlock`, whose reverse action is ``P*``.
    """
    N = len(ts) - 1
    tape = get_working_tape()
    pause_annotation()
    tape.clear_tape()
    initial_control = Function(V_by_slab[1], name=f"{name_prefix}_initial")
    initial_control.interpolate(problem.initial_condition(meshes[1]))

    raw_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    continue_annotation()
    incoming: Function | None = None
    try:
        for n in range(1, N + 1):
            mesh = meshes[n]
            V = V_by_slab[n]
            if n == 1:
                state = Function(V, name=f"{name_prefix}_slab_{n}_solution")
                state.assign(initial_control)
            else:
                transferred_incoming = tape_slab_interface_forward(
                    transfers[n - 1],
                    incoming,
                    f"{name_prefix}_slab_{n}_incoming",
                )
                # Preserve the transfer output as its own tape variable.
                # Irksome updates ``state`` in place during ``advance()``, so
                # using it directly as the P-output would overwrite the
                # checkpoint needed by the transfer block's reverse action.
                state = Function(V, name=f"{name_prefix}_slab_{n}_solution")
                state.assign(transferred_incoming)
            with stop_annotating():
                previous_right = _copy_function(
                    state, f"{name_prefix}_slab_{n}_previous_right"
                )

            t = Constant(float(ts[n - 1]))
            dt = Constant(float(ts[n] - ts[n - 1]))
            v = TestFunction(V)
            stepper = TimeStepper(
                problem.primal_residual(state, v, t),
                _dg_scheme(int(degree)),
                t,
                dt,
                state,
                bcs=problem.boundary_conditions(V),
                solver_parameters=opts.solver_parameters
                or {"ksp_type": "preonly", "pc_type": "lu"},
            )
            first_block = len(tape.get_blocks())
            stepper.advance()
            solve_blocks = [
                block
                for block in tape.get_blocks()[first_block:]
                if type(block).__name__ == "NonlinearVariationalSolveBlock"
            ]
            if len(solve_blocks) != 1:
                raise RuntimeError(
                    "Expected exactly one Firedrake-adjoint solve block per slab; "
                    f"found {len(solve_blocks)} on slab {n}."
                )
            raw_slabs[n] = {
                "mesh": mesh,
                "state": state,
                "previous_right": previous_right,
                "stages": list(stepper.stages.subfunctions),
                "solve_block": solve_blocks[0],
            }
            incoming = state

        terminal_mesh = meshes[N]
        J_h = assemble(problem.goal_functional(terminal_mesh, incoming))
        # Keep the algebraic derivative with respect to the initial state.
        # It is a representation-independent scalar check for another
        # implementation of the physical adjoint, unlike raw Irksome stage
        # multipliers whose coordinates depend on the time formulation.
        initial_derivative = compute_derivative(
            J_h, Control(initial_control), apply_riesz=False
        )
    finally:
        pause_annotation()

    primal_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    dual_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    U_nodes: list[Function | None] = [None] * (N + 1)
    Z_nodes: list[Function | None] = [None] * (N + 1)
    nodes = _time_nodes(int(degree))
    endpoint_weights = _lagrange_values(nodes, 1.0)
    left_weights = _lagrange_values(nodes, 0.0)

    with stop_annotating():
        for n in range(1, N + 1):
            raw = raw_slabs[n]
            V = V_by_slab[n]
            primal_coeffs = [
                _copy_function(value, f"{name_prefix}_U_slab_{n}_coefficient_{i}")
                for i, value in enumerate(raw["stages"])
            ]
            primal_right = _interpolate_expr(
                V,
                _expr_linear_combination(primal_coeffs, endpoint_weights),
                f"{name_prefix}_U_slab_{n}_right",
            )
            primal_slabs[n] = {
                "degree": int(degree),
                "orientation": "forward",
                "coeffs": primal_coeffs,
                "prev_right": raw["previous_right"],
                "right": primal_right,
                "mesh": raw["mesh"],
            }
            U_nodes[n] = _copy_function(primal_right, f"{name_prefix}_U_node_{n}")

            adjoint_stages = raw["solve_block"].adj_sol
            if adjoint_stages is None:
                raise RuntimeError(
                    f"Tape adjoint was not computed for slab {n}."
                )
            dual_coeffs = [
                _copy_function(value, f"{name_prefix}_Z_slab_{n}_coefficient_{i}")
                for i, value in enumerate(adjoint_stages.subfunctions)
            ]
            dual_right = _interpolate_expr(
                V,
                _expr_linear_combination(dual_coeffs, endpoint_weights),
                f"{name_prefix}_Z_slab_{n}_right",
            )
            dual_slabs[n] = {
                "degree": int(degree),
                "orientation": "forward",
                "coeffs": dual_coeffs,
                "prev_right": _interpolate_expr(
                    V,
                    _expr_linear_combination(dual_coeffs, left_weights),
                    f"{name_prefix}_Z_slab_{n}_left",
                ),
                "right": dual_right,
                "mesh": raw["mesh"],
            }
            Z_nodes[n] = _copy_function(dual_right, f"{name_prefix}_Z_node_{n}")

        U_nodes[0] = _copy_function(initial_control, f"{name_prefix}_U_node_0")
        Z_nodes[0] = _interpolate_expr(
            V_by_slab[1],
            _expr_linear_combination(dual_slabs[1]["coeffs"], left_weights),
            f"{name_prefix}_Z_node_0",
        )

    primal = {"U_nodes": U_nodes, "slabs": primal_slabs}
    dual = {
        "Z_nodes": Z_nodes,
        "slabs": dual_slabs,
        "initial_derivative": initial_derivative,
    }
    return (primal if return_primal else None), dual


def _recovery_parameters(opts: MovingPulseOptions) -> dict[str, Any]:
    return opts.recovery_solver_parameters or {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }


def _rename_subfunctions(function: Function, prefix: str) -> list[Function]:
    coefficients = list(function.subfunctions)
    for i, coefficient in enumerate(coefficients):
        coefficient.rename(f"{prefix}_{i}")
    return coefficients


def recover_residual_on_slab(
    slab_number: int,
    primal_slab: dict[str, Any],
    ts: np.ndarray,
    mesh: Mesh,
    opts: MovingPulseOptions,
    problem: TransientDWRProblem,
) -> dict[str, Any]:
    """Recover cell, spatial-endpoint, and left-time-interface residuals."""
    n = int(slab_number)
    k_n = float(ts[n] - ts[n - 1])
    q_degree = int(opts.spatial_quadrature_degree)
    dx_q = dx(metadata={"quadrature_degree": q_degree})
    ds_q = ds(metadata={"quadrature_degree": q_degree})
    dS_q = dS(metadata={"quadrature_degree": q_degree})
    quadrature = _gauss_rule(opts.time_quadrature_points)
    time_nodes = _time_nodes(opts.recovery_time_degree)
    n_time_coefficients = len(time_nodes)
    solver_parameters = _recovery_parameters(opts)

    Q = FunctionSpace(
        mesh,
        "DG",
        opts.recovery_space_degree,
        variant="integral",
    )
    bubble_space = FunctionSpace(mesh, "B", 2, variant="integral")
    bubble = Function(
        bubble_space,
        name=f"cell_bubble_slab_{n}",
    ).assign(1.0)

    # In one spatial dimension each facet is a point.  A broken DG1 function
    # has one independent trace value at each end of every interval.  Its two
    # nodal shapes are exactly the left and right endpoint cone functions.
    Q_endpoint = FunctionSpace(mesh, "DG", 1, variant="equispaced")

    # Cell-interior recovery in DG(p_rec) x P(r_rec).
    W_volume = MixedFunctionSpace([Q] * n_time_coefficients)
    r_volume_trial = TrialFunctions(W_volume)
    q_volume = TestFunctions(W_volume)
    a_volume = 0
    L_volume = 0

    for s_q, w_q in quadrature:
        lagrange = _lagrange_values(time_nodes, s_q)
        temporal_bubble = 4.0 * s_q * (1.0 - s_q)
        R_volume_q = _expr_linear_combination(r_volume_trial, lagrange)
        U_q = eval_slab_expr(primal_slab, s_q)
        dU_dt_q = eval_slab_dt_expr(primal_slab, s_q, k_n)
        t_q = Constant(float(ts[n - 1] + k_n * s_q))
        for j, q_j in enumerate(q_volume):
            test = Constant(temporal_bubble * lagrange[j]) * bubble * q_j
            weight = Constant(k_n * w_q)
            a_volume += weight * R_volume_q * test * dx_q
            L_volume += problem.volume_residual_action(
                U_q, dU_dt_q, weight * test, t_q, measure=dx_q
            )

    R_volume = Function(W_volume, name=f"R_volume_slab_{n}")
    solve(a_volume == L_volume, R_volume, solver_parameters=solver_parameters)
    volume_coefficients = _rename_subfunctions(R_volume, f"R_volume_slab_{n}")

    # Endpoint-cone recovery.  Boundary integration in 1D is point evaluation.
    W_endpoint = MixedFunctionSpace([Q_endpoint] * n_time_coefficients)
    r_endpoint_trial = TrialFunctions(W_endpoint)
    q_endpoint = TestFunctions(W_endpoint)
    a_endpoint = 0
    L_endpoint = 0

    for s_q, w_q in quadrature:
        lagrange = _lagrange_values(time_nodes, s_q)
        temporal_bubble = 4.0 * s_q * (1.0 - s_q)
        R_volume_q = _expr_linear_combination(volume_coefficients, lagrange)
        R_endpoint_q = _expr_linear_combination(r_endpoint_trial, lagrange)
        U_q = eval_slab_expr(primal_slab, s_q)
        dU_dt_q = eval_slab_dt_expr(primal_slab, s_q, k_n)
        t_q = Constant(float(ts[n - 1] + k_n * s_q))
        weight = Constant(k_n * w_q)

        for j, q_j in enumerate(q_endpoint):
            test = Constant(temporal_bubble * lagrange[j]) * q_j
            a_endpoint += weight * (
                R_endpoint_q * test * ds_q
                + (
                    R_endpoint_q("+") * test("+")
                    + R_endpoint_q("-") * test("-")
                )
                * dS_q
            )
            L_endpoint += problem.volume_residual_action(
                U_q, dU_dt_q, weight * test, t_q, measure=dx_q
            ) - weight * R_volume_q * test * dx_q

    R_endpoint = Function(W_endpoint, name=f"R_endpoint_slab_{n}")
    solve(a_endpoint == L_endpoint, R_endpoint, solver_parameters=solver_parameters)
    endpoint_coefficients = _rename_subfunctions(
        R_endpoint,
        f"R_endpoint_slab_{n}",
    )

    # Left temporal interface.  The spatial bubble removes endpoint terms.
    r_temporal_trial = TrialFunction(Q)
    q_temporal = TestFunction(Q)
    a_temporal = r_temporal_trial * bubble * q_temporal * dx_q
    L_temporal = 0

    for s_q, w_q in quadrature:
        lagrange = _lagrange_values(time_nodes, s_q)
        temporal_cone = 1.0 - s_q
        R_volume_q = _expr_linear_combination(volume_coefficients, lagrange)
        U_q = eval_slab_expr(primal_slab, s_q)
        dU_dt_q = eval_slab_dt_expr(primal_slab, s_q, k_n)
        t_q = Constant(float(ts[n - 1] + k_n * s_q))
        test = Constant(temporal_cone) * bubble * q_temporal
        weight = Constant(k_n * w_q)
        L_temporal += problem.volume_residual_action(
            U_q, dU_dt_q, weight * test, t_q, measure=dx_q
        ) - weight * R_volume_q * test * dx_q

    # ``prev_right`` already equals P_{n-1} u_{n-1}^- on this slab's mesh.
    # Hence this is the mesh-changing DG interface residual
    # u_n(t_{n-1}^+) - P_{n-1} u_{n-1}(t_{n-1}^-), paired with the same
    # transfer whose L2-adjoint is used in the backward solve.
    jump_left = eval_slab_expr(primal_slab, 0.0) - primal_slab["prev_right"]
    L_temporal += problem.temporal_residual_action(
        eval_slab_expr(primal_slab, 0.0),
        primal_slab["prev_right"],
        bubble * q_temporal,
        measure=dx_q,
    )
    R_temporal = Function(Q, name=f"R_temporal_slab_{n}")
    solve(a_temporal == L_temporal, R_temporal, solver_parameters=solver_parameters)

    return {
        "time_nodes": time_nodes,
        "volume": R_volume,
        "volume_coeffs": volume_coefficients,
        "endpoint": R_endpoint,
        "endpoint_coeffs": endpoint_coefficients,
        "temporal": R_temporal,
        "jump_left": jump_left,
    }


def estimate_automated_dwr(
    primal: dict[str, Any],
    dual_enriched: dict[str, Any],
    dual_numerical: dict[str, Any],
    ts: np.ndarray,
    opts: MovingPulseOptions,
    problem: TransientDWRProblem,
) -> dict[str, Any]:
    """Compute global DWR and signed ``eta[K,n]`` on slabwise meshes."""
    if primal["slabs"][1]["mesh"].comm.size != 1:
        raise NotImplementedError("Run this visual demonstration without mpiexec.")

    N = len(ts) - 1
    quadrature = _gauss_rule(opts.time_quadrature_points)
    q_degree = int(opts.spatial_quadrature_degree)
    dx_q = dx(metadata={"quadrature_degree": q_degree})
    ds_q = ds(metadata={"quadrature_degree": q_degree})
    dS_q = dS(metadata={"quadrature_degree": q_degree})

    eta_cell_slab_signed: list[np.ndarray | None] = [None] * (N + 1)
    eta_volume_cell: list[np.ndarray | None] = [None] * (N + 1)
    eta_endpoint_cell: list[np.ndarray | None] = [None] * (N + 1)
    eta_temporal_cell: list[np.ndarray | None] = [None] * (N + 1)
    eta_slab_signed = [0.0] * (N + 1)
    recovered_entities: list[dict[str, Any] | None] = [None] * (N + 1)
    eta_K_abs_by_slab: list[Function | None] = [None] * (N + 1)
    eta_global = 0.0

    for n in range(1, N + 1):
        primal_slab = primal["slabs"][n]
        enriched_slab = dual_enriched["slabs"][n]
        numerical_slab = dual_numerical["slabs"][n]
        mesh = primal_slab["mesh"]
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_test = TestFunction(DG0)
        k_n = float(ts[n] - ts[n - 1])
        recovered = recover_residual_on_slab(
            n, primal_slab, ts, mesh, opts, problem
        )
        recovered_entities[n] = recovered
        recovery_nodes = recovered["time_nodes"]

        volume_values = np.zeros(DG0.node_count, dtype=float)
        endpoint_values = np.zeros(DG0.node_count, dtype=float)

        for s_q, w_q in quadrature:
            lagrange = _lagrange_values(recovery_nodes, s_q)
            R_volume_q = _expr_linear_combination(
                recovered["volume_coeffs"],
                lagrange,
            )
            R_endpoint_q = _expr_linear_combination(
                recovered["endpoint_coeffs"],
                lagrange,
            )
            U_q = eval_slab_expr(primal_slab, s_q)
            dU_dt_q = eval_slab_dt_expr(primal_slab, s_q, k_n)
            z_star_q = (
                eval_slab_expr(enriched_slab, s_q)
                - eval_slab_expr(numerical_slab, s_q)
            )
            t_q = Constant(float(ts[n - 1] + k_n * s_q))
            weight = Constant(k_n * w_q)

            eta_global += float(
                assemble(problem.volume_residual_action(
                    U_q, dU_dt_q, weight * z_star_q, t_q, measure=dx_q
                ))
            )

            volume_vector: Cofunction = assemble(
                weight * R_volume_q * z_star_q * cell_test * dx_q
            )
            volume_values += np.asarray(volume_vector.dat.data_ro, dtype=float)

            endpoint_vector: Cofunction = assemble(
                weight
                * (
                    R_endpoint_q * z_star_q * cell_test * ds_q
                    + (
                        R_endpoint_q("+")
                        * z_star_q("+")
                        * cell_test("+")
                        + R_endpoint_q("-")
                        * z_star_q("-")
                        * cell_test("-")
                    )
                    * dS_q
                )
            )
            endpoint_values += np.asarray(endpoint_vector.dat.data_ro, dtype=float)

        z_star_left = (
            eval_slab_expr(enriched_slab, 0.0)
            - eval_slab_expr(numerical_slab, 0.0)
        )
        eta_global += float(assemble(problem.temporal_residual_action(
            eval_slab_expr(primal_slab, 0.0),
            primal_slab["prev_right"],
            z_star_left,
            measure=dx_q,
        )))
        temporal_vector: Cofunction = assemble(
            recovered["temporal"] * z_star_left * cell_test * dx_q
        )
        temporal_values = np.asarray(
            temporal_vector.dat.data_ro,
            dtype=float,
        ).copy()

        eta_values = volume_values + endpoint_values + temporal_values
        eta_cell_slab_signed[n] = eta_values
        eta_volume_cell[n] = volume_values
        eta_endpoint_cell[n] = endpoint_values
        eta_temporal_cell[n] = temporal_values
        eta_slab_signed[n] = float(eta_values.sum())
        eta_K_abs_by_slab[n] = Function(DG0, name=f"eta_K_abs_slab_{n}")
        eta_K_abs_by_slab[n].dat.data[:] = np.abs(eta_values)

    eta_local_sum = float(sum(eta_slab_signed[1:]))
    eta_marking_sum = float(
        sum(np.abs(values).sum() for values in eta_cell_slab_signed[1:])
    )
    localisation_gap = eta_local_sum - eta_global
    gap_relative = abs(localisation_gap) / max(abs(eta_global), np.finfo(float).eps)
    consistency = (
        eta_local_sum / eta_global
        if abs(eta_global) > np.finfo(float).eps
        else float("nan")
    )

    return {
        "eta_global": float(eta_global),
        "eta_local_sum": eta_local_sum,
        "eta_marking_sum": eta_marking_sum,
        "localisation_gap": float(localisation_gap),
        "localisation_gap_relative": float(gap_relative),
        "localisation_consistency_index": float(consistency),
        "eta_cell_slab_signed": eta_cell_slab_signed,
        "eta_volume_cell": eta_volume_cell,
        "eta_endpoint_cell": eta_endpoint_cell,
        "eta_temporal_cell": eta_temporal_cell,
        "eta_slab_signed": eta_slab_signed,
        "eta_K_abs_by_slab": eta_K_abs_by_slab,
        "recovered_entities": recovered_entities,
    }


def spatial_cell_order(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return Firedrake-cell indices sorted from left to right and their centres."""
    DG0 = FunctionSpace(mesh, "DG", 0)
    centres = Function(DG0, name="cell_centre")
    centres.interpolate(SpatialCoordinate(mesh)[0])
    values = np.asarray(centres.dat.data_ro, dtype=float).copy()
    order = np.argsort(values)
    return order, values[order]


def bisect_marked_intervals(
    x_nodes: np.ndarray,
    marked_left_to_right: np.ndarray,
) -> np.ndarray:
    """Bisect marked intervals and preserve every unmarked coordinate."""
    nodes = np.asarray(x_nodes, dtype=float)
    marked = np.asarray(marked_left_to_right, dtype=bool)
    if marked.size != nodes.size - 1:
        raise ValueError("One spatial marker is required for each interval.")
    refined = [float(nodes[0])]
    for i, is_marked in enumerate(marked):
        if is_marked:
            refined.append(0.5 * (float(nodes[i]) + float(nodes[i + 1])))
        refined.append(float(nodes[i + 1]))
    return np.asarray(refined, dtype=float)


def _write_ascii_vtu(
    path: Path,
    x_nodes_by_slab: list[np.ndarray | None],
    ts: np.ndarray,
    cell_data: dict[str, np.ndarray],
) -> None:
    """Write the nonconforming slabwise ``K x I_n`` grid as VTU.

    Adjacent slabs may have different spatial partitions.  We therefore give
    every slab its own set of vertices at its top and bottom edges.  Duplicate
    points on a time interface are intentional: they let ParaView display the
    actual locally refined rectangles without imposing a false conforming
    space--time mesh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = np.asarray(ts, dtype=float)
    nt = ts.size - 1
    if len(x_nodes_by_slab) != nt + 1:
        raise ValueError("One spatial node array is required for every slab.")
    ncells = int(sum(len(x_nodes_by_slab[n]) - 1 for n in range(1, nt + 1)))
    for name, values in cell_data.items():
        if np.asarray(values).size != ncells:
            raise ValueError(f"Cell array {name} has the wrong size.")

    points: list[str] = []
    connectivity: list[str] = []
    offsets: list[str] = []
    types: list[str] = []
    offset = 0
    point_offset = 0
    for n in range(nt):
        nodes = np.asarray(x_nodes_by_slab[n + 1], dtype=float)
        for t in (ts[n], ts[n + 1]):
            for x in nodes:
                points.append(f"{x:.17g} {t:.17g} 0")
        n_nodes = nodes.size
        for i in range(n_nodes - 1):
            p00 = point_offset + i
            p10 = p00 + 1
            p01 = point_offset + n_nodes + i
            p11 = p01 + 1
            connectivity.append(f"{p00} {p10} {p11} {p01}")
            offset += 4
            offsets.append(str(offset))
            types.append("9")  # VTK_QUAD
        point_offset += 2 * n_nodes

    with open(path, "w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n')
        stream.write(
            '<VTKFile type="UnstructuredGrid" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        stream.write("  <UnstructuredGrid>\n")
        stream.write(
            f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{ncells}">\n'
        )
        stream.write("      <Points>\n")
        stream.write(
            '        <DataArray type="Float64" NumberOfComponents="3" '
            'format="ascii">\n          '
        )
        stream.write("\n          ".join(points))
        stream.write("\n        </DataArray>\n      </Points>\n")
        stream.write("      <Cells>\n")
        stream.write(
            '        <DataArray type="Int32" Name="connectivity" '
            'format="ascii">\n          '
        )
        stream.write("\n          ".join(connectivity))
        stream.write("\n        </DataArray>\n")
        stream.write(
            '        <DataArray type="Int32" Name="offsets" format="ascii">\n'
            "          "
        )
        stream.write(" ".join(offsets))
        stream.write("\n        </DataArray>\n")
        stream.write(
            '        <DataArray type="UInt8" Name="types" format="ascii">\n'
            "          "
        )
        stream.write(" ".join(types))
        stream.write("\n        </DataArray>\n      </Cells>\n")
        stream.write("      <CellData>\n")
        for name, values in cell_data.items():
            stream.write(
                f'        <DataArray type="Float64" Name="{name}" '
                'format="ascii">\n          '
            )
            stream.write(
                " ".join(f"{float(value):.17g}" for value in np.asarray(values))
            )
            stream.write("\n        </DataArray>\n")
        stream.write("      </CellData>\n")
        stream.write("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")


class OneDimensionalLinearAdaptiveSolver:
    r"""Dimension-one adaptive driver with endpoint-cone localisation.

    This class is PDE independent within the scalar linear contract
    ``m(u_t,v)+A(u;v,t)=0``.  The only dimension-specific part is the recovery
    of ``R^{partial K}`` at the two endpoints of each interval.
    """

    def __init__(
        self,
        nx0: int,
        nt0: int,
        T: float,
        options: MovingPulseOptions | dict[str, Any] | None = None,
        problem: TransientDWRProblem | None = None,
    ):
        self.options = (
            MovingPulseOptions(**options)
            if isinstance(options, dict)
            else (options or MovingPulseOptions())
        )
        _validate_options(self.options)
        self.problem = problem or MovingPulseProblem(self.options)
        if not self.problem.spatial_operator_is_linear:
            raise ValueError("The endpoint-cone driver accepts linear scalar PDEs only.")
        if nx0 < 2 or nt0 < 1 or T <= 0.0:
            raise ValueError("Require nx0>=2, nt0>=1, and T>0.")
        self.ts = np.linspace(0.0, float(T), int(nt0) + 1)
        self.T = float(T)
        initial_nodes = np.linspace(0.0, 1.0, int(nx0) + 1)
        # ``x_nodes_by_slab[n]`` is the mesh T_n used only on I_n.  Keeping
        # these arrays separate is the key distinction from the former
        # shared-mesh code, where one spatial refinement affected every slab.
        self.x_nodes_by_slab: list[np.ndarray | None] = [None] + [
            initial_nodes.copy() for _ in range(int(nt0))
        ]
        self.history: list[dict[str, Any]] = []
        self._vtk_failed = False

    def _write_spatial_vtk(
        self,
        iteration: int,
        U_T: Function,
    ) -> None:
        """Write the terminal state on the final slab's spatial mesh.

        There is no longer a single spatial mesh on which it would be honest
        to write a time-summed indicator.  The complete local information is
        instead written to the slabwise x--t VTU file.
        """
        u_out = _copy_function(U_T, "u_h_T")
        VTKFile(f"{self.options.output_prefix}_{iteration}.pvd").write(u_out)

    def _write_spacetime_output(
        self,
        iteration: int,
        primal: dict[str, Any],
        estimate: dict[str, Any],
        marked_by_slab: list[np.ndarray | None],
    ) -> None:
        u_h: list[float] = []
        eta_signed: list[float] = []
        eta_volume: list[float] = []
        eta_endpoint: list[float] = []
        eta_temporal: list[float] = []
        marked: list[float] = []
        h_x: list[float] = []
        k_t: list[float] = []
        rows: list[dict[str, Any]] = []

        for n in range(1, len(self.ts)):
            mesh = primal["slabs"][n]["mesh"]
            cell_order, _ = spatial_cell_order(mesh)
            x_nodes = np.asarray(self.x_nodes_by_slab[n], dtype=float)
            # The primal discretisation is DG0 in time.  Store its value at
            # each spatial cell midpoint, giving one representative value on
            # every displayed space--time rectangle K x I_n.
            u_h_n = Function(
                FunctionSpace(mesh, "DG", 0),
                name="u_h_spacetime",
            ).interpolate(eval_slab_expr(primal["slabs"][n], 0.5))
            u_h_ordered = np.asarray(u_h_n.dat.data_ro, dtype=float)[cell_order]
            signed_n = np.asarray(
                estimate["eta_cell_slab_signed"][n],
                dtype=float,
            )[cell_order]
            volume_n = np.asarray(estimate["eta_volume_cell"][n], dtype=float)[
                cell_order
            ]
            endpoint_n = np.asarray(
                estimate["eta_endpoint_cell"][n],
                dtype=float,
            )[cell_order]
            temporal_n = np.asarray(
                estimate["eta_temporal_cell"][n],
                dtype=float,
            )[cell_order]
            marked_n = np.asarray(marked_by_slab[n], dtype=bool)[cell_order]
            k_n = float(self.ts[n] - self.ts[n - 1])
            for i in range(x_nodes.size - 1):
                u_h.append(float(u_h_ordered[i]))
                eta_signed.append(float(signed_n[i]))
                eta_volume.append(float(volume_n[i]))
                eta_endpoint.append(float(endpoint_n[i]))
                eta_temporal.append(float(temporal_n[i]))
                marked.append(float(marked_n[i]))
                h_i = float(x_nodes[i + 1] - x_nodes[i])
                h_x.append(h_i)
                k_t.append(k_n)
                rows.append(
                    {
                        "iteration": iteration,
                        "time_slab": n,
                        "space_cell": i,
                        "x_left": float(x_nodes[i]),
                        "x_right": float(x_nodes[i + 1]),
                        "t_left": float(self.ts[n - 1]),
                        "t_right": float(self.ts[n]),
                        "u_h": float(u_h_ordered[i]),
                        "eta_signed": float(signed_n[i]),
                        "eta_abs": abs(float(signed_n[i])),
                        "eta_volume": float(volume_n[i]),
                        "eta_endpoint": float(endpoint_n[i]),
                        "eta_temporal": float(temporal_n[i]),
                        "marked": int(marked_n[i]),
                        "h_x": h_i,
                        "k_t": k_n,
                    }
                )

        eta_signed_array = np.asarray(eta_signed, dtype=float)
        output_path = Path(
            f"{self.options.output_prefix}_spacetime_iter_{iteration}.vtu"
        )
        _write_ascii_vtu(
            output_path,
            self.x_nodes_by_slab,
            self.ts,
            {
                "u_h": np.asarray(u_h),
                "eta_signed": eta_signed_array,
                "eta_abs": np.abs(eta_signed_array),
                "eta_volume": np.asarray(eta_volume),
                "eta_endpoint": np.asarray(eta_endpoint),
                "eta_temporal": np.asarray(eta_temporal),
                "marked": np.asarray(marked),
                "h_x": np.asarray(h_x),
                "k_t": np.asarray(k_t),
            },
        )

        csv_path = Path(
            f"{self.options.output_prefix}_spacetime_iter_{iteration}.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_collections(self) -> None:
        prefix = Path(self.options.output_prefix)
        output_dir = prefix.parent
        prefix_name = prefix.name

        spatial_path = output_dir / f"{prefix_name}_iterations.pvd"
        spatial_entries: list[tuple[int, Path]] = []
        for iteration in range(len(self.history)):
            folder = output_dir / f"{prefix_name}_{iteration}"
            for vtu in sorted(folder.glob("*.vtu")):
                spatial_entries.append((iteration, vtu))
        if spatial_entries:
            with open(spatial_path, "w", encoding="utf-8") as stream:
                stream.write('<?xml version="1.0"?>\n')
                stream.write(
                    '<VTKFile type="Collection" version="0.1" '
                    'byte_order="LittleEndian">\n  <Collection>\n'
                )
                for iteration, vtu in spatial_entries:
                    relative = vtu.relative_to(output_dir)
                    stream.write(
                        f'    <DataSet timestep="{iteration}" group="" part="0" '
                        f'file="{relative.as_posix()}"/>\n'
                    )
                stream.write("  </Collection>\n</VTKFile>\n")

        spacetime_path = output_dir / f"{prefix_name}_spacetime_iterations.pvd"
        with open(spacetime_path, "w", encoding="utf-8") as stream:
            stream.write('<?xml version="1.0"?>\n')
            stream.write(
                '<VTKFile type="Collection" version="0.1" '
                'byte_order="LittleEndian">\n  <Collection>\n'
            )
            for iteration in range(len(self.history)):
                vtu = output_dir / f"{prefix_name}_spacetime_iter_{iteration}.vtu"
                stream.write(
                    f'    <DataSet timestep="{iteration}" group="" part="0" '
                    f'file="{vtu.name}"/>\n'
                )
            stream.write("  </Collection>\n</VTKFile>\n")

    def _write_history(self) -> None:
        if not self.history:
            return
        path = Path(f"{self.options.output_prefix}_history.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)

    def step(self, iteration: int) -> bool:
        opts = self.options
        ts = self.ts
        N = len(ts) - 1
        meshes: list[Mesh | None] = [None] + [
            create_interval_mesh(self.x_nodes_by_slab[n])
            for n in range(1, N + 1)
        ]
        V_by_slab: list[Any | None] = [None] + [
            FunctionSpace(meshes[n], "CG", opts.p) for n in range(1, N + 1)
        ]
        V_enriched_by_slab: list[Any | None] = [None] + [
            FunctionSpace(meshes[n], "CG", opts.p + 1)
            for n in range(1, N + 1)
        ]
        # Each primal/numerical-dual interface uses P_n forward and P_n^*
        # backward.  The enriched dual lives in a different polynomial space,
        # so it receives its own degree-(p+1) transfer pair.
        transfers: list[SlabInterfaceTransfer | None] = [None] * (N + 1)
        enriched_transfers: list[SlabInterfaceTransfer | None] = [None] * (N + 1)
        for n in range(1, N):
            transfers[n] = build_slab_interface_transfer(
                V_by_slab[n], V_by_slab[n + 1], self.problem
            )
            enriched_transfers[n] = build_slab_interface_transfer(
                V_enriched_by_slab[n], V_enriched_by_slab[n + 1], self.problem
            )

        if opts.adjoint_backend == "firedrake-adjoint":
            primal, dual_numerical = solve_primal_and_tape_dual(
                V_by_slab,
                ts,
                meshes,
                transfers,
                0,
                opts,
                self.problem,
                name_prefix="Tape_DG0",
                return_primal=True,
            )
            _, dual_enriched = solve_primal_and_tape_dual(
                V_enriched_by_slab,
                ts,
                meshes,
                enriched_transfers,
                opts.enriched_time_degree,
                opts,
                self.problem,
                name_prefix=f"Tape_DG{opts.enriched_time_degree}",
                return_primal=False,
            )
        else:
            primal = solve_primal(
                V_by_slab, ts, meshes, transfers, opts, self.problem
            )
            dual_enriched = solve_dual(
                V_enriched_by_slab,
                ts,
                meshes,
                enriched_transfers,
                opts.enriched_time_degree,
                opts,
                self.problem,
            )
            dual_numerical = solve_dual(
                V_by_slab, ts, meshes, transfers, 0, opts, self.problem
            )
        estimate = estimate_automated_dwr(
            primal,
            dual_enriched,
            dual_numerical,
            ts,
            opts,
            self.problem,
        )

        U_T = primal["U_nodes"][-1]
        terminal_mesh = meshes[N]
        J_h = float(assemble(self.problem.goal_functional(terminal_mesh, U_T)))
        J_exact = self.problem.exact_goal_value(self.T)
        true_error = None if J_exact is None else J_exact - J_h
        eta_global = float(estimate["eta_global"])
        eta_local = float(estimate["eta_local_sum"])
        effectivity = (
            eta_global / true_error
            if true_error is not None and abs(true_error) > 1.0e-15
            else float("nan")
        )
        should_stop = abs(eta_global) <= opts.tolerance or iteration == opts.max_it - 1

        marked_by_slab: list[np.ndarray | None] = [None]
        for n in range(1, N + 1):
            n_cells = self.x_nodes_by_slab[n].size - 1
            marked_by_slab.append(np.zeros(n_cells, dtype=bool))
        time_marked: set[int] = set()

        if not should_stop:
            marked_by_slab = mark_spacetime_cells(
                estimate["eta_cell_slab_signed"],
                opts.theta_spacetime,
            )
            if not opts.enable_space_refinement:
                marked_by_slab = [
                    None,
                    *[
                        np.zeros_like(marked_by_slab[n], dtype=bool)
                        for n in range(1, N + 1)
                    ],
                ]
            if opts.enable_time_refinement:
                for n in range(1, N + 1):
                    fraction = float(np.count_nonzero(marked_by_slab[n])) / float(
                        marked_by_slab[n].size
                    )
                    if (
                        np.any(marked_by_slab[n])
                        and fraction >= opts.time_slab_marked_fraction
                    ):
                        time_marked.add(n)

        n_spacetime_marked = int(
            sum(np.count_nonzero(mask) for mask in marked_by_slab[1:])
        )
        n_space_marked = n_spacetime_marked
        h_values = np.concatenate(
            [
                np.diff(np.asarray(self.x_nodes_by_slab[n], dtype=float))
                for n in range(1, N + 1)
            ]
        )
        primal_spacetime_dofs = int(sum(V_by_slab[n].dim() for n in range(1, N + 1)))
        row = {
            "iteration": iteration,
            "adjoint_backend": opts.adjoint_backend,
            "spatial_dofs_terminal": V_by_slab[N].dim(),
            "n_space_cells_terminal": self.x_nodes_by_slab[N].size - 1,
            "n_space_cells_total": int(
                sum(self.x_nodes_by_slab[n].size - 1 for n in range(1, N + 1))
            ),
            "n_time_slabs": N,
            "primal_spacetime_dofs": primal_spacetime_dofs,
            "J_exact": J_exact,
            "J_h": J_h,
            "true_error": float("nan") if true_error is None else true_error,
            "eta_global": eta_global,
            "eta_local_sum": eta_local,
            "eta_marking_sum": estimate["eta_marking_sum"],
            "effectivity_global": effectivity,
            "localisation_consistency_index": estimate[
                "localisation_consistency_index"
            ],
            "localisation_gap_relative": estimate["localisation_gap_relative"],
            "n_spacetime_marked": n_spacetime_marked,
            "n_space_marked": n_space_marked,
            "n_time_marked": len(time_marked),
            "h_min": float(h_values.min()),
            "h_max": float(h_values.max()),
            "k_min": float(np.diff(ts).min()),
            "k_max": float(np.diff(ts).max()),
            "threshold_reached": abs(eta_global) <= opts.tolerance,
        }
        self.history.append(row)

        PETSc.Sys.Print(
            f"\n---- [1D AUTOMATED ITER {iteration}] "
            f"terminalDOFs={V_by_slab[N].dim()} "
            f"totalSpaceTimeDOFs={primal_spacetime_dofs} Nt={N} ----"
        )
        PETSc.Sys.Print(f"  PDE                              = {self.problem.name}")
        PETSc.Sys.Print("  time solver                      = Irksome DG-in-time")
        PETSc.Sys.Print(f"  adjoint backend                  = {opts.adjoint_backend}")
        PETSc.Sys.Print("  primal / numerical dual          = CG(p) x DG0")
        PETSc.Sys.Print(
            f"  enriched dual                    = CG({opts.p + 1}) "
            f"x DG{opts.enriched_time_degree}"
        )
        PETSc.Sys.Print("  localisation                     = automated cell bubble + endpoint cones")
        PETSc.Sys.Print(
            f"  J_exact                          = {J_exact:+.6e}"
            if J_exact is not None
            else "  J_exact                          = unavailable"
        )
        PETSc.Sys.Print(f"  J_h                              = {J_h:+.6e}")
        PETSc.Sys.Print(
            f"  true goal error                  = {true_error:+.6e}"
            if true_error is not None
            else "  true goal error                  = unavailable"
        )
        PETSc.Sys.Print(f"  eta_global                       = {eta_global:+.6e}")
        PETSc.Sys.Print(f"  eta_local_sum                    = {eta_local:+.6e}")
        PETSc.Sys.Print(f"  signed effectivity               = {effectivity:8.4f}")
        PETSc.Sys.Print(
            f"  localisation consistency         = "
            f"{estimate['localisation_consistency_index']:8.5f}"
        )
        PETSc.Sys.Print(
            f"  marked                           = {n_spacetime_marked} space-time, "
            f"{n_space_marked} slab-local space, {len(time_marked)} time"
        )
        PETSc.Sys.Print(
            f"  h range                          = "
            f"[{h_values.min():.3e}, {h_values.max():.3e}]"
        )
        PETSc.Sys.Print(
            f"  k range                          = "
            f"[{np.diff(ts).min():.3e}, {np.diff(ts).max():.3e}]"
        )

        if opts.write_vtk and not self._vtk_failed:
            try:
                self._write_spatial_vtk(
                    iteration,
                    U_T,
                )
                self._write_spacetime_output(
                    iteration,
                    primal,
                    estimate,
                    marked_by_slab,
                )
            except ModuleNotFoundError as exc:
                self._vtk_failed = True
                PETSc.Sys.Print(f"  [VTK output skipped] {exc}")
        self._write_history()

        if should_stop:
            return True

        # Build the next slabwise mesh.  A marked K in I_n is bisected only
        # in T_n; if I_n is also split in time, both children inherit that
        # locally refined T_n.  No union over all time slabs is taken.
        next_ts = [float(ts[0])]
        next_nodes: list[np.ndarray | None] = [None]
        for n in range(1, N + 1):
            cell_order, _ = spatial_cell_order(meshes[n])
            markers_left_to_right = np.asarray(marked_by_slab[n], dtype=bool)[
                cell_order
            ]
            refined_nodes = (
                bisect_marked_intervals(
                    self.x_nodes_by_slab[n], markers_left_to_right
                )
                if opts.enable_space_refinement
                else np.asarray(self.x_nodes_by_slab[n], dtype=float).copy()
            )
            if n in time_marked:
                next_ts.append(0.5 * (float(ts[n - 1]) + float(ts[n])))
                next_nodes.append(refined_nodes.copy())
            next_ts.append(float(ts[n]))
            next_nodes.append(refined_nodes.copy())
        self.ts = np.asarray(next_ts, dtype=float)
        self.x_nodes_by_slab = next_nodes
        if time_marked:
            PETSc.Sys.Print(
                f"  [Time refinement] bisected slabs {sorted(time_marked)}; "
                f"Nt={len(self.ts)-1}"
            )
        if n_space_marked > 0:
            PETSc.Sys.Print(
                f"  [Slabwise space refinement] bisected {n_space_marked} "
                "marked space-time intervals only on their own slabs"
            )
        return False

    def solve(self) -> "OneDimensionalLinearAdaptiveSolver":
        for iteration in range(self.options.max_it):
            if self.step(iteration):
                break
        if self.options.write_vtk and not self._vtk_failed:
            self._write_collections()
        self._write_history()

        PETSc.Sys.Print("\n==== 1D endpoint-cone automated summary ====")
        for row in self.history:
            PETSc.Sys.Print(
                f"  it={row['iteration']:2d} "
                f"terminalDOFs={row['spatial_dofs_terminal']:5d} "
                f"Nt={row['n_time_slabs']:4d} "
                f"eta={row['eta_global']:+.6e} "
                f"Ieff={row['effectivity_global']:8.4f}"
            )
        if self.options.write_vtk and not self._vtk_failed:
            PETSc.Sys.Print(
                f"\nSpace-time ParaView file: "
                f"{self.options.output_prefix}_spacetime_iterations.pvd"
            )
        else:
            PETSc.Sys.Print("\nSpace-time ParaView output was disabled.")
        PETSc.Sys.Print(f"History CSV: {self.options.output_prefix}_history.csv")
        return self


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--nt", type=int, default=8)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--max-it", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--epsilon", type=float, default=2.0e-2)
    parser.add_argument("--velocity", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument(
        "--qoi",
        choices=(
            "terminal-window",
            "terminal-gaussian",
            "terminal-double-gaussian",
        ),
        default="terminal-window",
        help="Final-time collection window, one detector, or two detectors.",
    )
    parser.add_argument("--sensor-center", type=float, default=0.80)
    parser.add_argument("--sensor-radius", type=float, default=0.05)
    parser.add_argument("--left-sensor-center", type=float, default=0.70)
    parser.add_argument("--right-sensor-center", type=float, default=0.90)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--time-fraction", type=float, default=0.05)
    parser.add_argument("--quadrature-points", type=int, default=5)
    parser.add_argument("--spatial-quadrature-degree", type=int, default=24)
    parser.add_argument(
        "--adjoint-backend",
        choices=("firedrake-adjoint", "ufl"),
        default="firedrake-adjoint",
    )
    parser.add_argument(
        "--output-prefix",
        default="output/advection_diffusion_1d/automated",
    )
    parser.add_argument("--no-vtk", action="store_true")
    arguments = parser.parse_args()
    petsc_options = PETSc.Options()
    for action in parser._actions:
        for option_string in action.option_strings:
            if option_string.startswith("--"):
                petsc_options.delValue(option_string)
    return arguments


MovingPulseAutomatedSolver = OneDimensionalLinearAdaptiveSolver
"""Backward-compatible name for the supplied moving-pulse input."""


def main() -> OneDimensionalLinearAdaptiveSolver:
    args = _parse_arguments()
    options = MovingPulseOptions(
        max_it=args.max_it,
        tolerance=args.tolerance,
        epsilon=args.epsilon,
        velocity=args.velocity,
        beta=args.beta,
        qoi=args.qoi,
        sensor_center=args.sensor_center,
        sensor_radius=args.sensor_radius,
        left_sensor_center=args.left_sensor_center,
        right_sensor_center=args.right_sensor_center,
        theta_spacetime=args.theta,
        time_slab_marked_fraction=args.time_fraction,
        time_quadrature_points=args.quadrature_points,
        spatial_quadrature_degree=args.spatial_quadrature_degree,
        adjoint_backend=args.adjoint_backend,
        write_vtk=not args.no_vtk,
        output_prefix=args.output_prefix,
    )
    PETSc.Sys.Print("1D moving-pulse automated options:")
    PETSc.Sys.Print(asdict(options))
    return OneDimensionalLinearAdaptiveSolver(
        nx0=args.nx,
        nt0=args.nt,
        T=args.T,
        options=options,
    ).solve()


if __name__ == "__main__":
    main()
