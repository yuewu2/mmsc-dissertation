from __future__ import annotations

"""Goal-oriented h-refinement for a manufactured heat equation.

* spatial refinement: marked-cell h-refinement using Netgen/Firedrake;
* temporal refinement: DG0 time slabs are bisected.

Adaptive order in each iteration
--------------------------------
1. Solve the primal heat equation on the current mesh and time grid.
2. Solve the enriched dual in CG(p+1) x DG1.
3. Solve the numerical dual in the current space CG(p) x DG0.
4. Compute the signed DWR estimate and positive local marking indicators.
5. Mark space cells and time slabs from the same current estimator.
6. Write the current state to PVD/VTU for ParaView.
7. Apply temporal h-refinement first, then spatial h-refinement.

Main computed quantities
------------------------
u_h_T:
    numerical primal solution at final time T, written to ParaView.
z_numerical:
    numerical dual in the current space CG(p) x DG0.
z_enriched:
    enriched dual in CG(p+1) x DG1.
z_star:
    DWR weight z_enriched - z_numerical.
eta_global_signed:
    signed global DWR estimate, used for the reported effectivity index.
eta_K_abs:
    positive cell indicator used for spatial Dörfler marking.
eta_n_total:
    positive time-slab indicator used for temporal bisection.
h_K:
    cell diameter, used to visualise the mesh-size distribution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from firedrake import Constant, DirichletBC, Function, FunctionSpace, TestFunction, VTKFile, assemble
from firedrake.mesh import Mesh
from firedrake.petsc import PETSc
from irksome import Dt, DiscontinuousGalerkinScheme, TimeStepper
from ufl import And, CellDiameter, FacetNormal, SpatialCoordinate, conditional, dS, div, ds, dx, exp, grad, inner, pi, sin


# =============================================================================
# Problem data
# =============================================================================

KAPPA = 1.0
A_BOX = 0.25
B_BOX = 0.75


@dataclass(frozen=True)
class HeatHRefinementOptions:
    """User-facing parameters for the h-refinement run."""

    max_it: int = 8
    tolerance: float = 1.0e-8

    # Current primal/numerical-dual space is CG(p) in space and DG0 in time.
    p: int = 1

    # Enriched dual space is CG(p+1) in space and DG1 in time.
    enriched_time_degree: int = 1

    # Dörfler fractions.  Larger values mark more cells/slabs.
    dorfler_alpha_space: float = 0.2
    dorfler_alpha_time: float = 0.6

    # Decide whether the space/time indicators are large enough to act on.
    space_refine_fraction: float = 0.25
    time_refine_fraction: float = 0.01
    enable_space_refinement: bool = True
    enable_time_refinement: bool = True

    write_vtk: bool = True
    output_prefix: str = "output/heat_h_refinement"
    kappa: float = KAPPA
    solver_parameters: dict[str, Any] | None = None


def exact_solution_expr(mesh, t, scale=1.0):
    """Manufactured exact solution: u = exp(-t) sin(pi x) sin(pi y)."""
    x, y = SpatialCoordinate(mesh)
    return scale * exp(-t) * sin(pi * x) * sin(pi * y)


def source_expr(mesh, t, kappa: float = KAPPA, scale=1.0):
    """RHS f for u_t - kappa Delta u = f."""
    x, y = SpatialCoordinate(mesh)
    factor = Constant(2.0 * kappa * pi**2 - 1.0)
    return scale * factor * exp(-t) * sin(pi * x) * sin(pi * y)


def chi_omega_expr(mesh):
    """Indicator of the terminal observation box omega = [0.25, 0.75]^2."""
    x, y = SpatialCoordinate(mesh)
    inside_x = And(x >= A_BOX, x <= B_BOX)
    inside_y = And(y >= A_BOX, y <= B_BOX)
    return conditional(And(inside_x, inside_y), 1.0, 0.0)


def exact_goal_value(T: float) -> float:
    """Exact value of J(u)=int_omega u(x,T) dx for the manufactured solution."""
    integral_sin = (np.cos(np.pi * A_BOX) - np.cos(np.pi * B_BOX)) / np.pi
    return float(np.exp(-T) * integral_sin**2)


def create_box_fitted_mesh(nx=8, ny=8) -> Mesh:
    """Create a Netgen-backed mesh with the goal box fitted as an internal boundary."""
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle(p1=(0.0, 0.0), p2=(1.0, 1.0), bc="boundary", leftdomain=1, rightdomain=0)
    geo.AddRectangle(p1=(A_BOX, A_BOX), p2=(B_BOX, B_BOX), bc="box_interface", leftdomain=2, rightdomain=1)
    geo.SetMaterial(1, "outer")
    geo.SetMaterial(2, "inner")
    ngmesh = geo.GenerateMesh(maxh=1.0 / max(nx, ny))
    return Mesh(ngmesh)


# =============================================================================
# Numerical helpers
# =============================================================================


def _copy_function(u: Function, name: str) -> Function:
    out = Function(u.function_space(), name=name)
    out.assign(u)
    return out


def _interpolate_expr(V, expr, name: str) -> Function:
    out = Function(V, name=name)
    out.interpolate(expr)
    return out


def _positive_sqrt_array(values) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(values, dtype=float), 0.0))


def _time_nodes(degree: int) -> np.ndarray:
    if int(degree) == 0:
        return np.asarray([0.5], dtype=float)
    return np.linspace(0.0, 1.0, int(degree) + 1)


def _lagrange_values(nodes: np.ndarray, s: float) -> np.ndarray:
    if len(nodes) == 1:
        return np.asarray([1.0], dtype=float)
    values = []
    for i, xi in enumerate(nodes):
        value = 1.0
        for j, xj in enumerate(nodes):
            if i != j:
                value *= (s - xj) / (xi - xj)
        values.append(value)
    return np.asarray(values, dtype=float)


def _lagrange_derivatives(nodes: np.ndarray, s: float) -> np.ndarray:
    if len(nodes) == 1:
        return np.asarray([0.0], dtype=float)
    derivs = []
    for i, xi in enumerate(nodes):
        total = 0.0
        for m, xm in enumerate(nodes):
            if m == i:
                continue
            prod = 1.0 / (xi - xm)
            for j, xj in enumerate(nodes):
                if j != i and j != m:
                    prod *= (s - xj) / (xi - xj)
            total += prod
        derivs.append(total)
    return np.asarray(derivs, dtype=float)


def _expr_linear_combination(coeffs: list[Function], weights: np.ndarray):
    expr = Constant(0.0) * coeffs[0]
    for weight, coeff in zip(weights, coeffs):
        expr += Constant(float(weight)) * coeff
    return expr


def eval_slab_expr(slab: dict[str, Any], s: float):
    """Evaluate a saved DG-in-time polynomial at reference time s in [0,1]."""
    nodes = _time_nodes(int(slab["degree"]))
    if slab.get("orientation", "forward") == "reverse":
        s_eval = 1.0 - float(s)
    else:
        s_eval = float(s)
    return _expr_linear_combination(slab["coeffs"], _lagrange_values(nodes, s_eval))


def eval_slab_dt_expr(slab: dict[str, Any], s: float, k_n: float):
    """Evaluate d_t of a saved forward-time DG polynomial."""
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
    pts, wts = np.polynomial.legendre.leggauss(max(1, int(npoints)))
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts
    return [(float(s), float(w)) for s, w in zip(pts, wts)]


def _dg_scheme(degree: int) -> DiscontinuousGalerkinScheme:
    return DiscontinuousGalerkinScheme(
        int(degree),
        basis_type="equispaced",
        quadrature_degree=max(2 * int(degree) + 3, 3),
        deriv_type="strong",
    )


def mark_by_bulk(values: np.ndarray, fraction: float) -> np.ndarray:
    """Dörfler marking: mark largest positive values until fraction of total is reached."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    marked = np.zeros(values.shape, dtype=bool)
    if total <= 0.0:
        return marked

    order = np.argsort(values)[::-1]
    target = float(fraction) * total
    subtotal = 0.0
    for i in order:
        marked[i] = True
        subtotal += float(values[i])
        if subtotal >= target:
            break
    return marked


def set_adaptive_cell_markers(eta_K: Function, fraction: float) -> Function:
    """Convert cell indicators eta_K into a DG0 marker field for Netgen refinement."""
    markers = Function(eta_K.function_space(), name="space_markers")
    markers.assign(0.0)
    marked = mark_by_bulk(np.asarray(eta_K.dat.data_ro, dtype=float), fraction)
    markers.dat.data[:] = marked.astype(markers.dat.data.dtype)
    return markers


def refine_marked_mesh(mesh: Mesh, markers: Function) -> Mesh:
    """Refine marked cells using Firedrake's Netgen-backed marked refinement."""
    if hasattr(mesh, "refine_marked_elements"):
        return mesh.refine_marked_elements(markers)
    raise RuntimeError("The current mesh does not support refine_marked_elements.")


def refine_time_grid(ts: np.ndarray, marked_slabs: set[int]) -> np.ndarray:
    """Bisect marked time slabs."""
    if not marked_slabs:
        return np.asarray(ts, dtype=float)

    new_ts = [float(ts[0])]
    for n in range(1, len(ts)):
        if n in marked_slabs:
            new_ts.append(0.5 * (float(ts[n - 1]) + float(ts[n])))
        new_ts.append(float(ts[n]))
    return np.asarray(new_ts, dtype=float)


def choose_refinement_actions(eta_space: float, eta_time: float, opts: HeatHRefinementOptions) -> tuple[bool, bool]:
    """Decide whether to act in space, time, or both from positive indicators."""
    total = abs(eta_space) + abs(eta_time)
    if total <= 0.0:
        return False, False

    do_space = opts.enable_space_refinement and abs(eta_space) > opts.space_refine_fraction * total
    do_time = opts.enable_time_refinement and abs(eta_time) > opts.time_refine_fraction * total

    if not do_space and not do_time:
        if opts.enable_space_refinement and abs(eta_space) >= abs(eta_time):
            do_space = True
        elif opts.enable_time_refinement:
            do_time = True
    return do_space, do_time


def mark_time_slabs(eta_n: list[float], fraction: float) -> set[int]:
    """Dörfler-mark time slabs for bisection using eta_n_total."""
    values = np.asarray([abs(eta_n[n]) for n in range(1, len(eta_n))], dtype=float)
    mask = mark_by_bulk(values, fraction)
    return {i + 1 for i, marked in enumerate(mask) if marked}


# =============================================================================
# Irksome primal and adjoint solvers
# =============================================================================


def solve_primal_irksome_dg0(
    V,
    ts: np.ndarray,
    mesh: Mesh,
    u0_expr,
    kappa: float = KAPPA,
    solver_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve u_t - kappa Delta u = f with CG(p) in space and DG0 in time."""
    N = len(ts) - 1
    params = solver_parameters or {"ksp_type": "preonly", "pc_type": "lu"}
    bcs = [DirichletBC(V, 0.0, "on_boundary")]

    u = Function(V, name="u_h")
    u.interpolate(u0_expr)

    v = TestFunction(V)
    t = Constant(float(ts[0]))
    dt = Constant(float(ts[1] - ts[0]) if N > 0 else 0.0)
    F = inner(Dt(u), v) * dx + Constant(kappa) * inner(grad(u), grad(v)) * dx - inner(source_expr(mesh, t, kappa), v) * dx

    U_nodes: list[Function | None] = [None] * (N + 1)
    U_nodes[0] = _copy_function(u, "U_node_0")
    slabs: list[dict[str, Any] | None] = [None] * (N + 1)

    for n in range(1, N + 1):
        k_n = float(ts[n] - ts[n - 1])
        t.assign(float(ts[n - 1]))
        dt.assign(k_n)

        prev_right = _copy_function(u, f"U_prev_right_{n}")
        stepper = TimeStepper(F, _dg_scheme(0), t, dt, u, bcs=bcs, solver_parameters=params)
        stepper.advance()

        coeffs = [_copy_function(c, f"U_DG0_slab_{n}_coef_{i}") for i, c in enumerate(stepper.stages.subfunctions)]
        U_nodes[n] = _copy_function(u, f"U_node_{n}")
        slabs[n] = {
            "degree": 0,
            "orientation": "forward",
            "coeffs": coeffs,
            "prev_right": prev_right,
            "right": U_nodes[n],
        }

    return {"U_nodes": U_nodes, "slabs": slabs}


def solve_dual_irksome_uniform_dg(
    V_dual,
    ts: np.ndarray,
    mesh: Mesh,
    degree: int,
    kappa: float = KAPPA,
    solver_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve the terminal adjoint in reversed time.

    For J(u)=int_omega u(T) dx, the adjoint is
        -z_t - kappa Delta z = 0, z(T)=chi_omega.
    With tau=T-t, this becomes a forward heat equation for w(tau)=z(T-t).
    """
    N = len(ts) - 1
    T = float(ts[-1])
    tau_ts = T - ts[::-1]
    params = solver_parameters or {"ksp_type": "preonly", "pc_type": "lu"}
    bcs = [DirichletBC(V_dual, 0.0, "on_boundary")]

    w_fun = Function(V_dual, name=f"w_dual_DG{degree}")
    w_fun.interpolate(chi_omega_expr(mesh))

    v = TestFunction(V_dual)
    tau = Constant(float(tau_ts[0]))
    dtau = Constant(float(tau_ts[1] - tau_ts[0]) if N > 0 else 0.0)
    F = inner(Dt(w_fun), v) * dx + Constant(kappa) * inner(grad(w_fun), grad(v)) * dx

    tau_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    W_nodes: list[Function | None] = [None] * (N + 1)
    W_nodes[0] = _copy_function(w_fun, "W_tau_node_0")

    for j in range(1, N + 1):
        k_j = float(tau_ts[j] - tau_ts[j - 1])
        tau.assign(float(tau_ts[j - 1]))
        dtau.assign(k_j)

        stepper = TimeStepper(F, _dg_scheme(degree), tau, dtau, w_fun, bcs=bcs, solver_parameters=params)
        stepper.advance()

        coeffs = [_copy_function(c, f"W_DG{degree}_tau_slab_{j}_coef_{i}") for i, c in enumerate(stepper.stages.subfunctions)]
        W_nodes[j] = _copy_function(w_fun, f"W_tau_node_{j}")
        tau_slabs[j] = {
            "degree": int(degree),
            "orientation": "forward",
            "coeffs": coeffs,
            "right": W_nodes[j],
        }

    Z_nodes: list[Function | None] = [None] * (N + 1)
    Z_slabs: list[dict[str, Any] | None] = [None] * (N + 1)
    for n in range(1, N + 1):
        j = N - n + 1
        Z_slabs[n] = {
            "degree": int(degree),
            "orientation": "reverse",
            "coeffs": tau_slabs[j]["coeffs"],
        }
        Z_nodes[n] = _interpolate_expr(V_dual, eval_slab_expr(Z_slabs[n], 1.0), f"Z_node_{n}")
    Z_nodes[0] = _interpolate_expr(V_dual, eval_slab_expr(Z_slabs[1], 0.0), "Z_node_0")

    return {"Z_nodes": Z_nodes, "slabs": Z_slabs, "tau_slabs": tau_slabs, "tau_ts": tau_ts}


# =============================================================================
# DWR estimator and local positive indicators
# =============================================================================


def estimate_dwr_with_enriched_minus_numerical(
    primal_current: dict[str, Any],
    dual_enriched: dict[str, Any],
    dual_numerical: dict[str, Any],
    ts: np.ndarray,
    mesh: Mesh,
    kappa: float = KAPPA,
) -> dict[str, Any]:
    """Compute DWR using z_star = z_enriched - z_numerical.

    Returned variables:
    * eta_global_signed: signed DWR estimate R(u_h)(z_enr-z_num);
    * eta_K_abs: positive cell indicator used for spatial marking;
    * eta_n_total: positive slab indicator used for temporal bisection.
    """
    N = len(ts) - 1
    DG0 = FunctionSpace(mesh, "DG", 0)
    cell_test = TestFunction(DG0)
    normal = FacetNormal(mesh)

    eta_n_space_time = [0.0] * (N + 1)
    eta_n_jump = [0.0] * (N + 1)
    eta_n_total = [0.0] * (N + 1)

    h_field = Function(DG0, name="h_K")
    h_field.interpolate(CellDiameter(mesh))
    h_values = np.asarray(h_field.dat.data_ro, dtype=float)
    h_safe = np.maximum(h_values, np.finfo(float).eps)

    enriched_degree = int(dual_enriched["slabs"][1]["degree"])
    quad = _gauss_rule(max(4, 2 * enriched_degree + 3))
    eta_global_signed = 0.0
    eta_K_values = np.zeros_like(h_values, dtype=float)

    for n in range(1, N + 1):
        slab_u = primal_current["slabs"][n]
        slab_z_enr = dual_enriched["slabs"][n]
        slab_z_num = dual_numerical["slabs"][n]
        k_n = float(ts[n] - ts[n - 1])
        k_c = Constant(k_n)

        R2_K = np.zeros_like(h_values, dtype=float)
        z2_K = np.zeros_like(h_values, dtype=float)
        r2_dK = np.zeros_like(h_values, dtype=float)
        z2_dK = np.zeros_like(h_values, dtype=float)

        for s_q, w_q in quad:
            t_q = Constant(float(ts[n - 1] + k_n * s_q))
            U_q = eval_slab_expr(slab_u, s_q)
            dU_dt_q = eval_slab_dt_expr(slab_u, s_q, k_n)
            z_star_q = eval_slab_expr(slab_z_enr, s_q) - eval_slab_expr(slab_z_num, s_q)
            f_q = source_expr(mesh, t_q, kappa)

            eta_global_signed += float(
                assemble(
                    k_c
                    * Constant(w_q)
                    * (f_q * z_star_q - inner(dU_dt_q, z_star_q) - Constant(kappa) * inner(grad(U_q), grad(z_star_q)))
                    * dx
                )
            )

            R_q = f_q + div(Constant(kappa) * grad(U_q)) - dU_dt_q
            R2_K += float(k_n * w_q) * np.asarray(assemble(R_q**2 * cell_test * dx).dat.data_ro, dtype=float)
            z2_K += float(k_n * w_q) * np.asarray(assemble(z_star_q**2 * cell_test * dx).dat.data_ro, dtype=float)

            r_facet_q = 0.5 * inner(Constant(kappa) * (grad(U_q)("+") - grad(U_q)("-")), normal("+"))
            r2_dK += float(k_n * w_q) * np.asarray(
                assemble(r_facet_q**2 * (cell_test("+") + cell_test("-")) * dS).dat.data_ro,
                dtype=float,
            )
            z2_dK += float(k_n * w_q) * np.asarray(
                assemble(z_star_q**2 * cell_test * ds + z_star_q("+") ** 2 * (cell_test("+") + cell_test("-")) * dS).dat.data_ro,
                dtype=float,
            )

        jump_left = eval_slab_expr(slab_u, 0.0) - slab_u["prev_right"]
        z_star_left = eval_slab_expr(slab_z_enr, 0.0) - eval_slab_expr(slab_z_num, 0.0)
        eta_global_signed -= float(assemble(jump_left * z_star_left * dx))

        jump2_K = assemble(jump_left**2 * cell_test * dx)
        z_left2_K = assemble(z_star_left**2 * cell_test * dx)

        R_norm = _positive_sqrt_array(R2_K)
        z_norm = _positive_sqrt_array(z2_K)
        r_norm = _positive_sqrt_array(r2_dK)
        z_boundary_norm = _positive_sqrt_array(z2_dK)
        jump_norm = _positive_sqrt_array(jump2_K.dat.data_ro)
        z_left_norm = _positive_sqrt_array(z_left2_K.dat.data_ro)

        eta_space_time_n = (R_norm + h_safe ** (-0.5) * r_norm) * (z_norm + h_safe**0.5 * z_boundary_norm)
        eta_jump_n = (k_n ** (-0.5) * jump_norm) * (k_n**0.5 * z_left_norm)
        eta_total_n = eta_space_time_n + eta_jump_n

        eta_K_values += eta_total_n
        eta_n_space_time[n] = float(eta_space_time_n.sum())
        eta_n_jump[n] = float(eta_jump_n.sum())
        eta_n_total[n] = float(eta_total_n.sum())

    eta_K_abs = Function(DG0, name="eta_K_abs")
    eta_K_abs.dat.data[:] = eta_K_values

    return {
        "eta_global_signed": eta_global_signed,
        "eta_space": float(np.sum(eta_K_abs.dat.data_ro)),
        "eta_time": float(sum(abs(v) for v in eta_n_total)),
        "eta_K_abs": eta_K_abs,
        "eta_n_space_time": eta_n_space_time,
        "eta_n_jump": eta_n_jump,
        "eta_n_total": eta_n_total,
    }

# global signed estimator
#     eta_global_signed

# positive marking indicators
#     eta_K_abs = [eta_K1, eta_K2, eta_K3, ...]      spatial cell indicator
#     eta_n_total = [0, eta_1, eta_2, eta_3, ...]    time slab indicator

# diagnostic split in time direction
#     eta_n_space_time
#     eta_n_jump

# summed totals
#     eta_space
#     eta_time


# =============================================================================
# Adaptive driver
# =============================================================================


class HeatHRefinementSolver:
    def __init__(
        self,
        base_mesh: Mesh,
        Nt0: int,
        T: float,
        options: HeatHRefinementOptions | dict[str, Any] | None = None,
    ):
        self.options = HeatHRefinementOptions(**options) if isinstance(options, dict) else (options or HeatHRefinementOptions())
        if self.options.enriched_time_degree < 1:
            raise ValueError("enriched_time_degree must be at least 1 for DG0 -> DG1 temporal enrichment.")

        self.mesh = base_mesh
        self.ts = np.linspace(0.0, float(T), int(Nt0) + 1)
        self.T = float(T)

        self.spatial_dofs_vec: list[int] = []
        self.nt_vec: list[int] = []
        self.eta_global_vec: list[float] = []
        self.true_error_vec: list[float] = []
        self.I_eff_vec: list[float] = []

    def mesh_size_field_and_stats(self, DG0: FunctionSpace) -> tuple[Function, dict[str, float]]:
        h_K = Function(DG0, name="h_K")
        h_K.interpolate(CellDiameter(self.mesh))
        h_values = np.asarray(h_K.dat.data_ro, dtype=float)
        return h_K, {
            "n_cells": float(h_values.size),
            "h_min": float(np.min(h_values)),
            "h_median": float(np.median(h_values)),
            "h_max": float(np.max(h_values)),
        }

    def write_vtk(
        self,
        it: int,
        U_T,
        est: dict[str, Any],
        h_K: Function,
    ) -> None:
        """Write the main fields needed to inspect solution, indicator and mesh size."""
        opts = self.options
        Path(opts.output_prefix).parent.mkdir(parents=True, exist_ok=True)

        # Fixed field names are useful when ParaView reads all iterations as one animation.
        u_out = Function(U_T.function_space(), name="u_h_T")
        u_out.assign(U_T)

        VTKFile(f"{opts.output_prefix}_{it}.pvd").write(
            u_out,
            est["eta_K_abs"],
            h_K,
        )

    def write_iteration_collection_pvd(self) -> None:
        """Collect per-iteration VTU files into one ParaView time series."""
        prefix = Path(self.options.output_prefix)
        output_dir = prefix.parent
        prefix_name = prefix.name
        output_path = output_dir / f"{prefix_name}_iterations.pvd"

        vtu_files: list[tuple[int, Path]] = []
        for it in range(len(self.I_eff_vec)):
            folder = output_dir / f"{prefix_name}_{it}"
            for vtu in sorted(folder.glob("*.vtu")):
                vtu_files.append((it, vtu))

        if not vtu_files:
            return

        with open(output_path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write("  <Collection>\n")
            for it, vtu in vtu_files:
                rel = vtu.relative_to(output_dir)
                f.write(f'    <DataSet timestep="{it}" group="" part="0" file="{escape(str(rel))}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")

    def step(self, it: int) -> float:
        opts = self.options
        mesh = self.mesh
        ts = self.ts

        V = FunctionSpace(mesh, "CG", opts.p)
        V_enr = FunctionSpace(mesh, "CG", opts.p + 1)

        primal_current = solve_primal_irksome_dg0(
            V,
            ts,
            mesh,
            exact_solution_expr(mesh, Constant(0.0)),
            kappa=opts.kappa,
            solver_parameters=opts.solver_parameters,
        )
        dual_enriched = solve_dual_irksome_uniform_dg(
            V_enr,
            ts,
            mesh,
            opts.enriched_time_degree,
            kappa=opts.kappa,
            solver_parameters=opts.solver_parameters,
        )
        dual_numerical = solve_dual_irksome_uniform_dg(
            V,
            ts,
            mesh,
            0,
            kappa=opts.kappa,
            solver_parameters=opts.solver_parameters,
        )

        est = estimate_dwr_with_enriched_minus_numerical(
            primal_current,
            dual_enriched,
            dual_numerical,
            ts,
            mesh,
            kappa=opts.kappa,
        )

        U_nodes = primal_current["U_nodes"]
        J_h = float(assemble(chi_omega_expr(mesh) * U_nodes[-1] * dx))
        J_exact = exact_goal_value(self.T)
        true_error = float(J_exact - J_h)
        eta_global = float(est["eta_global_signed"])
        I_eff = eta_global / true_error if abs(true_error) > 1.0e-15 else float("nan")

        eta_space = float(est["eta_space"])
        eta_time = float(est["eta_time"])
        should_stop = abs(eta_global) < opts.tolerance or it == opts.max_it - 1

        DG0 = FunctionSpace(mesh, "DG", 0)
        h_K, mesh_stats = self.mesh_size_field_and_stats(DG0)
        space_markers = Function(DG0, name="space_markers")
        space_markers.assign(0.0)
        time_marked: set[int] = set()
        do_space = False
        do_time = False

        if not should_stop:
            do_space, do_time = choose_refinement_actions(eta_space, eta_time, opts)
            if do_space:
                space_markers = set_adaptive_cell_markers(est["eta_K_abs"], opts.dorfler_alpha_space)
            if do_time:
                time_marked = mark_time_slabs(est["eta_n_total"], opts.dorfler_alpha_time)

        n_space_marked = int(np.count_nonzero(space_markers.dat.data_ro))

        self.spatial_dofs_vec.append(V.dim())
        self.nt_vec.append(len(ts) - 1)
        self.eta_global_vec.append(eta_global)
        self.true_error_vec.append(true_error)
        self.I_eff_vec.append(I_eff)

        PETSc.Sys.Print(
            f"\n---- [HEAT H-REFINEMENT ITER {it}] "
            f"spatialDOFs={V.dim()} | Nt={len(ts)-1} | cells={int(mesh_stats['n_cells'])} ----"
        )
        PETSc.Sys.Print("  time method                      = DG0, h-bisection only")
        PETSc.Sys.Print(f"  enriched dual                    = CG({opts.p + 1}) x DG{opts.enriched_time_degree}")
        PETSc.Sys.Print("  DWR weight                       = z_enriched - z_numerical")
        PETSc.Sys.Print(f"  J_exact                          = {J_exact:+.6e}")
        PETSc.Sys.Print(f"  J_h                              = {J_h:+.6e}")
        PETSc.Sys.Print(f"  true_error                       = {true_error:+.6e}")
        # PETSc.Sys.Print(f"  eta_global signed DWR            = {eta_global:+.6e}")
        PETSc.Sys.Print(f"  signed effectivity               = {I_eff:8.4f}")
        PETSc.Sys.Print(f"  positive indicators              = space {eta_space:.6e}, time {eta_time:.6e}")
        PETSc.Sys.Print(
            f"  mesh h_K                         = min {mesh_stats['h_min']:.3e}, "
            f"median {mesh_stats['h_median']:.3e}, max {mesh_stats['h_max']:.3e}"
        )

        if opts.write_vtk:
            self.write_vtk(it, U_nodes[-1], est, h_K)

        if should_stop:
            return eta_space + eta_time

        # Refinement is applied after writing the current iteration.
        # Both marks were computed before either grid is changed.
        if do_time and time_marked:
            self.ts = refine_time_grid(self.ts, time_marked)
            PETSc.Sys.Print(f"  [Time h-refinement] bisected slabs {sorted(time_marked)}; Nt={len(self.ts)-1}")
        elif opts.enable_time_refinement:
            PETSc.Sys.Print("  [Time h-refinement skipped]")

        if do_space and n_space_marked > 0:
            self.mesh = refine_marked_mesh(mesh, space_markers)
            PETSc.Sys.Print(f"  [Space h-refinement] marked {n_space_marked} cells")
        elif opts.enable_space_refinement:
            PETSc.Sys.Print("  [Space h-refinement skipped]")

        return eta_space + eta_time

    def solve(self):
        for it in range(self.options.max_it):
            self.step(it)

        if self.options.write_vtk:
            self.write_iteration_collection_pvd()

        PETSc.Sys.Print("\n==== Heat DWR h-refinement summary ====")
        for it, I_eff in enumerate(self.I_eff_vec):
            PETSc.Sys.Print(
                f"  it={it:2d} spatialDOFs={self.spatial_dofs_vec[it]:6d} "
                f"Nt={self.nt_vec[it]:4d} eta_global={self.eta_global_vec[it]:+.6e} "
                f"true_error={self.true_error_vec[it]:+.6e} I_eff={I_eff:8.4f}"
            )
        PETSc.Sys.Print(f"\nParaView animation file: {self.options.output_prefix}_iterations.pvd")
        return self


if __name__ == "__main__":
    base_mesh = create_box_fitted_mesh(nx=8, ny=8)
    solver = HeatHRefinementSolver(
        base_mesh,
        Nt0=8,
        T=0.5,
        options=HeatHRefinementOptions(
            max_it=8,
            tolerance=1.0e-8,
            p=1,
            enriched_time_degree=1,
            dorfler_alpha_space=0.2,
            dorfler_alpha_time=0.6,
            space_refine_fraction=0.25,
            time_refine_fraction=0.01,
            enable_space_refinement=True,
            enable_time_refinement=True,
            write_vtk=True,
            output_prefix="output/heat_h_refinement",
        ),
    )
    solver.solve()
