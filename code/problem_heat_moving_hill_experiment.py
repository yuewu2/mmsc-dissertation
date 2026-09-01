"""Rotating-hill heat benchmark of Hartmann/Thiele--Wick.

The published quantity is ``||u_exact-u_h||_L2(Q)``.  The solver-facing goal
below is one half of its square.  Its adjoint and every signed local indicator
differ from the norm-goal versions only by one positive iteration-wide scalar,
so Dörfler marking selects exactly the same cells and slabs.  From a CSV row,

    L2_error = sqrt(2*J_h)
    eta_L2   = -eta_global/L2_error
    Ieff_L2  = 0.5*effectivity_global

up to the estimator sign convention used in the row.
"""

from __future__ import annotations

import argparse
import math

from firedrake import (
    Constant,
    DirichletBC,
    SpatialCoordinate,
    cos,
    div,
    grad,
    inner,
    pi,
    sin,
)

from moving_hill_dwr_experiments import (
    NonstationaryDWRSolver,
    NonstationaryProblem,
    add_common_arguments,
    config_from_arguments,
    create_adaptive_unit_square_mesh,
    parse_arguments,
)


def build_problem(args) -> NonstationaryProblem:
    """Return the PDE, exact data, goal, and boundary inputs only."""
    if args.hill_sharpness <= 0.0 or args.kappa <= 0.0:
        raise ValueError("hill-sharpness and kappa must be positive.")

    sharpness = Constant(args.hill_sharpness)
    kappa = Constant(args.kappa)

    def exact_solution(mesh, time):
        x, y = SpatialCoordinate(mesh)
        centre_x = Constant(0.5) + Constant(0.25) * cos(2.0 * pi * time)
        centre_y = Constant(0.5) + Constant(0.25) * sin(2.0 * pi * time)
        radius_squared = (x - centre_x) ** 2 + (y - centre_y) ** 2
        return 1.0 / (1.0 + sharpness * radius_squared)

    def initial_condition(mesh):
        return exact_solution(mesh, Constant(0.0))

    def manufactured_source(mesh, time):
        """Return the source obtained from the rotating exact hill."""
        x, y = SpatialCoordinate(mesh)
        centre_x = Constant(0.5) + Constant(0.25) * cos(2.0 * pi * time)
        centre_y = Constant(0.5) + Constant(0.25) * sin(2.0 * pi * time)
        offset_x, offset_y = x - centre_x, y - centre_y
        radius_squared = offset_x**2 + offset_y**2
        denominator = 1.0 + sharpness * radius_squared
        time_derivative = -sharpness * pi * (
            offset_x * sin(2.0 * pi * time)
            - offset_y * cos(2.0 * pi * time)
        ) / denominator**2
        laplacian = (
            -4.0 * sharpness / denominator**2
            + 8.0 * sharpness**2 * radius_squared / denominator**3
        )
        return time_derivative - kappa * laplacian

    def spatial_residual(state, test, time, *, measure):
        source = manufactured_source(test.ufl_domain(), time)
        return (
            kappa * inner(grad(state), grad(test)) * measure
            - source * test * measure
        )

    def running_goal(mesh, state, time, *, measure):
        error = state - exact_solution(mesh, time)
        return Constant(0.5) * error**2 * measure

    def strong_residual(mesh, state, state_dt, time):
        """Manufactured ``f + div(kappa grad(u_h)) - dt(u_h)`` input."""
        source = manufactured_source(mesh, time)
        return source + div(kappa * grad(state)) - state_dt

    def normal_flux(state, normal):
        return inner(kappa * grad(state), normal)

    def primal_boundary_conditions(V, time):
        return [DirichletBC(V, exact_solution(V.mesh(), time), "on_boundary")]

    def adjoint_boundary_conditions(V):
        return [DirichletBC(V, 0.0, "on_boundary")]

    def goal_diagnostics(goal_value, estimator, true_error, *, symmetric_identity=False):
        l2_error = math.sqrt(max(0.0, 2.0 * float(goal_value)))
        if symmetric_identity:
            # eta estimates J(uexact)-J(uh)=-1/2 ||u-uh||^2 for this goal.
            eta_l2 = math.sqrt(max(0.0, -2.0 * float(estimator)))
        else:
            # Legacy raw primal residual: eta ~= -||u-uh||^2.
            eta_l2 = -float(estimator) / l2_error if l2_error > 1.0e-15 else float("nan")
        return {
            "L2_space_time_error": l2_error,
            "eta_for_L2_error": eta_l2,
            "effectivity_L2": (
                eta_l2 / l2_error if l2_error > 1.0e-15 else float("nan")
            ),
        }

    return NonstationaryProblem(
        name="hartmann_rotating_hill_heat",
        goal_label="half_squared_space_time_L2_error",
        mesh=create_adaptive_unit_square_mesh,
        initial_condition=initial_condition,
        spatial_residual=spatial_residual,
        running_goal=running_goal,
        boundary_conditions=primal_boundary_conditions,
        adjoint_boundary_conditions=adjoint_boundary_conditions,
        exact_goal=0.0,
        goal_diagnostics=goal_diagnostics,
        strong_residual=strong_residual,
        normal_flux=normal_flux,
        nonlinear_identity=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            __doc__
            + "\nThis isolated entry point does not modify/import the dissertation's "
            "nonstationary_dwr package."
        )
    )
    add_common_arguments(
        parser,
        default_T=1.0,
        default_nx=8,
        default_nt=40,
        default_output="output/nonstationary_dwr/heat_moving_hill",
        default_max_it=5,
        default_theta=0.4,
        default_time_fraction=0.05,
    )
    # Retain both left and right space--time ridge remainders so the primal
    # and reverse-adjoint bubble decompositions are complete.
    parser.set_defaults(
        omit_mixed_ridge=False,
        # Use the practical symmetric nonlinear two-term estimator:
        # 1/2 rho(e_z) + 1/2 rho*(e_u).  The converged-solve Galerkin
        # correction is not part of either the global or local estimator.
        omit_galerkin_correction=True,
    )
    parser.add_argument("--hill-sharpness", type=float, default=50.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parse_arguments(parser)
    return NonstationaryDWRSolver(
        build_problem(args), config_from_arguments(args, nonlinear_identity=True)
    ).solve()


if __name__ == "__main__":
    main()
