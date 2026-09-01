"""L-shaped regularised parabolic p-Laplace DWR benchmark.

This is the slabwise counterpart of Section 6.4 of ``[10]Review of DWR.pdf``.
It uses the same regularised p-Laplace operator and the paper's two goals

    J_1(u) = integral_0.5^0.75 integral_Omega_I |grad u|^p dx dt,
    J_2(u) = integral_Omega u(x, T) dx.

The review prints a manufactured factor
``(1-x)^2 (1-y)^2 r^(2/3) sin(2 theta/3)`` together with homogeneous
Dirichlet data on the L-domain ``(-1,1)^2 \\ [0,1]x[-1,0]``.  Those two
statements are not compatible on every outer boundary segment.  The default
below therefore uses the homogeneous-boundary correction
``(1-x^2)^2 (1-y^2)^2`` while retaining the same corner singularity and
``sin(t)`` time dependence.  This makes the manufactured source, boundary
data, exact goal and reported effectivity mutually consistent.

``--goal combined`` uses one automatic adjoint for the signed, relatively
normalised multi-goal functional.  The solver updates its weights after every
primal solve from the exact/reference component errors; the signs prevent the
oppositely signed J_1 and J_2 errors from cancelling.
"""

from __future__ import annotations

import argparse

import numpy as np
from firedrake import (
    Constant,
    DirichletBC,
    SpatialCoordinate,
    atan2,
    conditional,
    grad,
    inner,
    pi,
    sin,
    cos,
    sqrt,
)

from moving_hill_dwr_experiments import (
    GoalComponent,
    NonstationaryDWRSolver,
    NonstationaryProblem,
    add_common_arguments,
    config_from_arguments,
    create_adaptive_l_shaped_mesh,
    parse_arguments,
)


def build_problem(args) -> NonstationaryProblem:
    """Build the nonlinear problem using only UFL PDE/QoI inputs."""
    if args.p <= 1.0:
        raise ValueError("p must be greater than one.")
    if args.regularisation <= 0.0:
        raise ValueError("regularisation must be positive.")

    def polar_angle(x, y):
        raw = atan2(y, x)
        return conditional(raw < 0.0, raw + 2.0 * pi, raw)

    def spatial_profile(domain):
        x, y = SpatialCoordinate(domain)
        radius = sqrt(x**2 + y**2)
        angle = polar_angle(x, y)
        boundary_factor = (1.0 - x**2) ** 2 * (1.0 - y**2) ** 2
        return (
            Constant(1.5)
            * boundary_factor
            * radius ** (2.0 / 3.0)
            * sin(Constant(2.0 / 3.0) * angle)
        )

    def exact_solution(domain, time):
        return spatial_profile(domain) * sin(time)

    def exact_flux(domain, time):
        profile = spatial_profile(domain)
        exact = profile * sin(time)
        exact_gradient = grad(exact)
        # Keep the user-supplied exponent as a literal UFL value.  In the
        # paper's p=4 case it is exactly one, allowing TSFC to simplify the
        # flux instead of treating a Constant exponent as a generic power.
        exponent = 0.5 * (args.p - 2.0)
        regularised_norm = inner(exact_gradient, exact_gradient) + Constant(
            args.regularisation**2
        )
        return regularised_norm**exponent * exact_gradient

    def spatial_residual(state, test, time, *, measure):
        state_gradient = grad(state)
        exponent = 0.5 * (args.p - 2.0)
        regularised_norm = inner(state_gradient, state_gradient) + Constant(
            args.regularisation**2
        )
        flux = regularised_norm**exponent * state_gradient
        domain = test.ufl_domain()
        exact_time_derivative = spatial_profile(domain) * cos(time)
        # Assemble the manufactured right-hand side as a dual-space action,
        # <f,v> = (u_exact,t,v) + (F(u_exact), grad(v)).  The corner mode is
        # admissible in the weak energy space, whereas explicitly forming
        # div(F(u_exact)) creates a non-L1 expression at the re-entrant corner.
        return (
            inner(flux, grad(test)) * measure
            - exact_time_derivative * test * measure
            - inner(exact_flux(domain, time), grad(test)) * measure
        )

    def terminal_average(domain, terminal_state, *, measure):
        return terminal_state * measure

    def windowed_energy(domain, state, time, *, measure):
        x, y = SpatialCoordinate(domain)
        in_x = conditional(
            x < Constant(-15.0 / 16.0), 0.0,
            conditional(x <= Constant(-11.0 / 16.0), 1.0, 0.0),
        )
        in_y = conditional(
            y < Constant(11.0 / 16.0), 0.0,
            conditional(y <= Constant(15.0 / 16.0), 1.0, 0.0),
        )
        in_time = conditional(
            time < Constant(0.5), 0.0,
            conditional(time <= Constant(0.75), 1.0, 0.0),
        )
        return (
            in_x * in_y * in_time
            * inner(grad(state), grad(state)) ** (0.5 * args.p)
            * measure
        )

    def exact_terminal_average(final_time):
        # The L-domain used by create_adaptive_l_shaped_mesh is the disjoint
        # union [-1,0]x[-1,1] and [0,1]x[0,1], up to measure-zero interfaces.
        nodes, weights = np.polynomial.legendre.leggauss(args.reference_quadrature)

        def integrate_rectangle(x_left, x_right, y_left, y_right):
            x_values = 0.5 * ((x_right - x_left) * nodes + x_right + x_left)
            y_values = 0.5 * ((y_right - y_left) * nodes + y_right + y_left)
            x, y = np.meshgrid(x_values, y_values, indexing="ij")
            radius = np.sqrt(x**2 + y**2)
            angle = np.mod(np.arctan2(y, x), 2.0 * np.pi)
            profile = (
                1.5
                * (1.0 - x**2) ** 2
                * (1.0 - y**2) ** 2
                * radius ** (2.0 / 3.0)
                * np.sin((2.0 / 3.0) * angle)
            )
            jacobian = 0.25 * (x_right - x_left) * (y_right - y_left)
            return jacobian * np.sum(np.outer(weights, weights) * profile)

        spatial_integral = integrate_rectangle(-1.0, 0.0, -1.0, 1.0)
        spatial_integral += integrate_rectangle(0.0, 1.0, 0.0, 1.0)
        return np.sin(final_time) * spatial_integral

    def exact_windowed_energy(final_time):
        time_right = min(float(final_time), 0.75)
        if time_right <= 0.5:
            return 0.0
        nodes, weights = np.polynomial.legendre.leggauss(args.reference_quadrature)

        def mapped_interval(left, right):
            points = 0.5 * ((right - left) * nodes + right + left)
            mapped_weights = 0.5 * (right - left) * weights
            return points, mapped_weights

        x_values, x_weights = mapped_interval(-15.0 / 16.0, -11.0 / 16.0)
        y_values, y_weights = mapped_interval(11.0 / 16.0, 15.0 / 16.0)
        x, y = np.meshgrid(x_values, y_values, indexing="ij")
        radius = np.sqrt(x**2 + y**2)
        angle = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        alpha = 2.0 / 3.0
        singular = radius**alpha * np.sin(alpha * angle)
        singular_x = alpha * radius ** (alpha - 1.0) * np.sin((alpha - 1.0) * angle)
        singular_y = alpha * radius ** (alpha - 1.0) * np.cos((alpha - 1.0) * angle)
        boundary = (1.0 - x**2) ** 2 * (1.0 - y**2) ** 2
        boundary_x = -4.0 * x * (1.0 - x**2) * (1.0 - y**2) ** 2
        boundary_y = -4.0 * y * (1.0 - y**2) * (1.0 - x**2) ** 2
        profile_x = 1.5 * (boundary_x * singular + boundary * singular_x)
        profile_y = 1.5 * (boundary_y * singular + boundary * singular_y)
        spatial = np.sum(
            np.outer(x_weights, y_weights)
            * (profile_x**2 + profile_y**2) ** (0.5 * args.p)
        )
        time_values, time_weights = mapped_interval(0.5, time_right)
        temporal = np.sum(time_weights * np.sin(time_values) ** args.p)
        return float(spatial * temporal)

    goal_components = None
    if args.goal == "j1":
        terminal_goal, running_goal = None, windowed_energy
        exact_goal = exact_windowed_energy
        goal_label = "windowed_energy_J1"
    elif args.goal == "j2":
        terminal_goal, running_goal = terminal_average, None
        exact_goal = exact_terminal_average
        goal_label = "terminal_average_J2"
    else:
        terminal_goal = running_goal = exact_goal = None
        goal_components = [
            GoalComponent(
                "J1",
                exact_windowed_energy,
                running_goal=windowed_energy,
                weight=args.j1_weight,
            ),
            GoalComponent(
                "J2",
                exact_terminal_average,
                terminal_goal=terminal_average,
                weight=args.j2_weight,
            ),
        ]
        goal_label = "signed_relative_J1_plus_J2"

    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    return NonstationaryProblem(
        name="lshape_regularised_parabolic_plaplace",
        goal_label=goal_label,
        mesh=lambda nx, ny: create_adaptive_l_shaped_mesh(max(nx, ny)),
        initial_condition=lambda domain: exact_solution(domain, Constant(0.0)),
        spatial_residual=spatial_residual,
        terminal_goal=terminal_goal,
        running_goal=running_goal,
        goal_components=goal_components,
        boundary_conditions=lambda V: [DirichletBC(V, 0.0, "on_boundary")],
        exact_goal=exact_goal,
        nonlinear=True,
        nonlinear_identity=True,
        solver_parameters=solver_parameters,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_T=1.0,
        default_nx=8,
        default_nt=8,
        default_output="output/nonstationary_dwr/lshape_parabolic_plaplace",
        default_max_it=8,
        default_theta=0.35,
        default_time_fraction=0.10,
    )
    parser.set_defaults(tolerance=1.0e-5)
    parser.add_argument("--p", type=float, default=4.0)
    parser.add_argument("--regularisation", type=float, default=1.0e-2)
    parser.add_argument("--goal", choices=("j1", "j2", "combined"), default="j2")
    parser.add_argument("--j1-weight", type=float, default=1.0)
    parser.add_argument("--j2-weight", type=float, default=1.0)
    parser.add_argument("--reference-quadrature", type=int, default=220)
    args = parse_arguments(parser)
    if args.reference_quadrature < 8:
        raise ValueError("reference-quadrature must be at least eight.")
    if args.goal in {"j1", "combined"} and args.T < 0.75:
        raise ValueError("J1 uses the complete time window [0.5, 0.75], so require T >= 0.75.")
    if args.output_prefix == "output/nonstationary_dwr/lshape_parabolic_plaplace":
        args.output_prefix += f"_{args.goal}"
    return NonstationaryDWRSolver(
        build_problem(args), config_from_arguments(args, nonlinear_identity=True)
    ).solve()


if __name__ == "__main__":
    main()
