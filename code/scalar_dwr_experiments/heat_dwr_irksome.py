from __future__ import annotations

"""Shared heat-problem data and the single Irksome DG-in-time solver.

Both strong-residual and bubble/cone localisation use this module.  Keeping
the time discretisation here prevents the two estimators from silently using
different primal or adjoint solvers.
"""

from typing import Any, Sequence

import numpy as np

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    dx,
    exp,
    grad,
    inner,
    pi,
    sin,
)
from irksome import DiscontinuousGalerkinScheme, Dt, TimeStepper
from ufl import And, conditional


KAPPA = 1.0
A_BOX = 0.25
B_BOX = 0.75


def exact_solution_expr(mesh: Mesh, t, scale=1.0):
    """Manufactured solution u=exp(-t) sin(pi*x) sin(pi*y)."""
    x, y = SpatialCoordinate(mesh)
    return scale * exp(-t) * sin(pi * x) * sin(pi * y)


def source_expr(mesh: Mesh, t, kappa: float = KAPPA, scale=1.0):
    """Right-hand side for u_t-kappa*Delta(u)=f."""
    x, y = SpatialCoordinate(mesh)
    factor = Constant(2.0 * kappa * pi**2 - 1.0)
    return scale * factor * exp(-t) * sin(pi * x) * sin(pi * y)


def chi_omega_expr(mesh: Mesh):
    """Indicator of omega=[0.25,0.75]^2 in the terminal goal functional."""
    x, y = SpatialCoordinate(mesh)
    inside_x = And(x >= A_BOX, x <= B_BOX)
    inside_y = And(y >= A_BOX, y <= B_BOX)
    return conditional(And(inside_x, inside_y), 1.0, 0.0)


def exact_goal_value(T: float) -> float:
    """Exact J(u)=integral_omega u(x,T) dx for the manufactured solution."""
    integral_sin = (np.cos(np.pi * A_BOX) - np.cos(np.pi * B_BOX)) / np.pi
    return float(np.exp(-T) * integral_sin**2)


def _copy_function(u: Function, name: str) -> Function:
    out = Function(u.function_space(), name=name)
    out.assign(u)
    return out


def _interpolate_expr(V, expr, name: str) -> Function:
    out = Function(V, name=name)
    out.interpolate(expr)
    return out


def _time_nodes(degree: int) -> np.ndarray:
    if int(degree) == 0:
        return np.asarray([0.5], dtype=float)
    return np.linspace(0.0, 1.0, int(degree) + 1)


def _lagrange_values(nodes: np.ndarray, s: float) -> np.ndarray:
    if len(nodes) == 1:
        return np.asarray([1.0], dtype=float)
    values: list[float] = []
    for i, xi in enumerate(nodes):
        value = 1.0
        for j, xj in enumerate(nodes):
            if i != j:
                value *= (float(s) - xj) / (xi - xj)
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _lagrange_derivatives(nodes: np.ndarray, s: float) -> np.ndarray:
    if len(nodes) == 1:
        return np.asarray([0.0], dtype=float)
    derivatives: list[float] = []
    for i, xi in enumerate(nodes):
        total = 0.0
        for m, xm in enumerate(nodes):
            if m == i:
                continue
            product = 1.0 / (xi - xm)
            for j, xj in enumerate(nodes):
                if j != i and j != m:
                    product *= (float(s) - xj) / (xi - xj)
            total += product
        derivatives.append(float(total))
    return np.asarray(derivatives, dtype=float)


def _expr_linear_combination(coefficients: Sequence, weights: np.ndarray):
    expr = Constant(0.0) * coefficients[0]
    for weight, coefficient in zip(weights, coefficients):
        expr += Constant(float(weight)) * coefficient
    return expr


def eval_slab_expr(slab: dict[str, Any], s: float):
    """Evaluate a saved Irksome DG polynomial at reference time s in [0,1]."""
    nodes = _time_nodes(int(slab["degree"]))
    if slab.get("orientation", "forward") == "reverse":
        s_eval = 1.0 - float(s)
    else:
        s_eval = float(s)
    return _expr_linear_combination(slab["coeffs"], _lagrange_values(nodes, s_eval))


def eval_slab_dt_expr(slab: dict[str, Any], s: float, k_n: float):
    """Evaluate d_t of a saved forward or reverse Irksome DG polynomial."""
    nodes = _time_nodes(int(slab["degree"]))
    if slab.get("orientation", "forward") == "reverse":
        s_eval = 1.0 - float(s)
        sign = -1.0
    else:
        s_eval = float(s)
        sign = 1.0
    weights = sign * _lagrange_derivatives(nodes, s_eval) / float(k_n)
    return _expr_linear_combination(slab["coeffs"], weights)


def _gauss_rule(npoints: int) -> list[tuple[float, float]]:
    """Quadrature used only to integrate estimators over a solved time slab."""
    points, weights = np.polynomial.legendre.leggauss(max(1, int(npoints)))
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    return [(float(s), float(w)) for s, w in zip(points, weights)]


def _dg_scheme(degree: int) -> DiscontinuousGalerkinScheme:
    """Irksome DG scheme used for every primal and adjoint time slab."""
    return DiscontinuousGalerkinScheme(
        int(degree),
        basis_type="equispaced",
        quadrature_degree=max(2 * int(degree) + 3, 3),
        deriv_type="strong",
    )


def solve_forward_dg_in_time(
    V,
    ts: np.ndarray,
    mesh: Mesh,
    initial_expr,
    degree: int,
    source_factory=None,
    kappa: float = KAPPA,
    solver_parameters: dict[str, Any] | None = None,
    name_prefix: str = "U",
) -> dict[str, Any]:
    """Solve y_t-kappa*Delta(y)=source with Irksome DG in time."""
    N = len(ts) - 1
    params = solver_parameters or {"ksp_type": "preonly", "pc_type": "lu"}
    if N < 1:
        raise ValueError("At least one time slab is required.")

    t = Constant(float(ts[0]))
    dt = Constant(float(ts[1] - ts[0]))
    y = Function(V, name=f"{name_prefix}_solution")
    y.interpolate(initial_expr)
    v = TestFunction(V)
    bcs = [DirichletBC(V, 0.0, "on_boundary")]

    F = (
        inner(Dt(y), v) * dx
        + Constant(float(kappa)) * inner(grad(y), grad(v)) * dx
    )
    if source_factory is not None:
        F -= inner(source_factory(mesh, t, kappa), v) * dx

    time_nodes = _time_nodes(int(degree))
    U_nodes: list[Function | None] = [None] * (N + 1)
    U_nodes[0] = _copy_function(y, f"{name_prefix}_node_0")
    slabs: list[dict[str, Any] | None] = [None] * (N + 1)

    for n in range(1, N + 1):
        k_n = float(ts[n] - ts[n - 1])
        t.assign(float(ts[n - 1]))
        dt.assign(k_n)
        previous_for_slab = _copy_function(
            y,
            f"{name_prefix}_previous_right_{n}",
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
        U_nodes[n] = _copy_function(right_value, f"{name_prefix}_node_{n}_saved")
        slabs[n] = {
            "degree": int(degree),
            "orientation": "forward",
            "coeffs": coefficients,
            "prev_right": previous_for_slab,
            "right": U_nodes[n],
        }
    return {"U_nodes": U_nodes, "slabs": slabs}


def solve_primal_dg0(
    V,
    ts: np.ndarray,
    mesh: Mesh,
    u0_expr,
    kappa: float = KAPPA,
    solver_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the primal heat equation in CG(p) x DG0 using Irksome."""
    return solve_forward_dg_in_time(
        V,
        ts,
        mesh,
        u0_expr,
        degree=0,
        source_factory=source_expr,
        kappa=kappa,
        solver_parameters=solver_parameters,
        name_prefix="U",
    )


def solve_dual_uniform_dg(
    V_dual,
    ts: np.ndarray,
    mesh: Mesh,
    degree: int,
    kappa: float = KAPPA,
    solver_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the terminal adjoint with Irksome in reversed time."""
    N = len(ts) - 1
    T = float(ts[-1])
    tau_ts = T - ts[::-1]
    forward = solve_forward_dg_in_time(
        V_dual,
        tau_ts,
        mesh,
        chi_omega_expr(mesh),
        degree=degree,
        source_factory=None,
        kappa=kappa,
        solver_parameters=solver_parameters,
        name_prefix=f"W_DG{degree}",
    )
    tau_slabs = forward["slabs"]

    Z_nodes: list[Function | None] = [None] * (N + 1)
    Z_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    for n in range(1, N + 1):
        j = N - n + 1
        Z_slabs[n] = {
            "degree": int(degree),
            "orientation": "reverse",
            "coeffs": tau_slabs[j]["coeffs"],
        }
        Z_nodes[n] = _interpolate_expr(
            V_dual,
            eval_slab_expr(Z_slabs[n], 1.0),
            f"Z_node_{n}",
        )
    Z_nodes[0] = _interpolate_expr(
        V_dual,
        eval_slab_expr(Z_slabs[1], 0.0),
        "Z_node_0",
    )
    return {
        "Z_nodes": Z_nodes,
        "slabs": Z_slabs,
        "tau_slabs": tau_slabs,
        "tau_ts": tau_ts,
    }
