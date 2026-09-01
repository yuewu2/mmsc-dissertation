"""Mesh-fitted single-goal L-shaped J1 bubble-DWR experiment.

The quantity of interest is the windowed gradient energy from Section 6.4
of [10] Review of DWR,

    J1(u) = int_0.5^0.75 int_Omega_I |grad u|^p dx dt,

where

    Omega_I = (-15/16, -11/16) x (11/16, 15/16).

Unlike the earlier coordinate-indicator implementation, the four sides of
Omega_I are now mesh facets and the observation box is material subdomain 2.
The value of J1, its derivative in the adjoint problem, and the nonlinear DWR
identity consequently use exactly the same fitted measure and quadrature.

The PDE is the regularised parabolic p-Laplace manufactured problem from the
L-shaped benchmark.  This entry point keeps the current bubble projection,
global space-time marking, and independent slab-local spatial refinement.
VTK output is enabled by default.
"""

from __future__ import annotations

import argparse

import numpy as np
from firedrake import (
    Constant,
    DirichletBC,
    Mesh,
    SpatialCoordinate,
    atan2,
    conditional,
    cos,
    dx,
    grad,
    inner,
    pi,
    sin,
    sqrt,
)

from moving_hill_dwr_experiments import (
    NonstationaryDWRSolver,
    NonstationaryProblem,
    add_common_arguments,
    config_from_arguments,
    parse_arguments,
)


DEFAULT_OUTPUT = "output/thesis/lshape/j1_fitted_configurable"
OBSERVATION_SUBDOMAIN = 2


def create_fitted_l_shaped_mesh(n: int = 8) -> Mesh:
    """Create the L-domain with the J1 observation rectangle as material 2."""
    try:
        from netgen.geom2d import SplineGeometry
    except ImportError as exc:
        raise RuntimeError(
            "The fitted adaptive J1 benchmark requires netgen.geom2d. "
            "Run it in the Firedrake environment."
        ) from exc

    geometry = SplineGeometry()
    points = [
        geometry.AppendPoint(-1.0, -1.0),
        geometry.AppendPoint(0.0, -1.0),
        geometry.AppendPoint(0.0, 0.0),
        geometry.AppendPoint(1.0, 0.0),
        geometry.AppendPoint(1.0, 1.0),
        geometry.AppendPoint(-1.0, 1.0),
    ]
    for start, end in zip(points, points[1:] + points[:1]):
        geometry.Append(
            ["line", start, end],
            leftdomain=1,
            rightdomain=0,
            bc="boundary",
        )

    geometry.AddRectangle(
        p1=(-15.0 / 16.0, 11.0 / 16.0),
        p2=(-11.0 / 16.0, 15.0 / 16.0),
        bc="j1_observation_interface",
        leftdomain=2,
        rightdomain=1,
    )
    geometry.SetMaterial(1, "l_domain")
    geometry.SetMaterial(2, "j1_observation")
    return Mesh(geometry.GenerateMesh(maxh=2.0 / max(1, int(n))))


def build_problem(args) -> NonstationaryProblem:
    """Build the fitted nonlinear p-Laplace J1 problem from UFL inputs."""
    if args.p <= 1.0:
        raise ValueError("p must be greater than one")
    if args.regularisation <= 0.0:
        raise ValueError("regularisation must be positive")

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
        # Keep the singular manufactured load in V*: do not form the
        # non-integrable strong divergence at the re-entrant corner.
        return (
            inner(flux, grad(test)) * measure
            - exact_time_derivative * test * measure
            - inner(exact_flux(domain, time), grad(test)) * measure
        )

    def windowed_energy(domain, state, time, *, measure):
        # Deliberately use one explicit measure for every call site.  The
        # solver differentiates this same form to construct J1'(u)(phi).
        del measure
        goal_measure = dx(
            OBSERVATION_SUBDOMAIN,
            domain=domain,
            metadata={"quadrature_degree": args.goal_quadrature_degree},
        )
        in_time = conditional(
            time < Constant(0.5),
            0.0,
            conditional(time <= Constant(0.75), 1.0, 0.0),
        )
        return (
            in_time
            * inner(grad(state), grad(state)) ** (0.5 * args.p)
            * goal_measure
        )

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
        singular_x = alpha * radius ** (alpha - 1.0) * np.sin(
            (alpha - 1.0) * angle
        )
        singular_y = alpha * radius ** (alpha - 1.0) * np.cos(
            (alpha - 1.0) * angle
        )
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
        name="lshape_regularised_parabolic_plaplace_j1_fitted",
        goal_label="windowed_energy_J1_fitted",
        mesh=lambda nx, ny: create_fitted_l_shaped_mesh(max(nx, ny)),
        initial_condition=lambda domain: exact_solution(domain, Constant(0.0)),
        spatial_residual=spatial_residual,
        running_goal=windowed_energy,
        boundary_conditions=lambda V: [DirichletBC(V, 0.0, "on_boundary")],
        exact_goal=exact_windowed_energy,
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
        default_output=DEFAULT_OUTPUT,
        default_max_it=8,
        default_theta=0.30,
        default_time_fraction=0.10,
    )
    parser.set_defaults(
        tolerance=1.0e-8,
        localisation_mode="hierarchical_recovery",
        bubble_marking_score="signed_total",
        nonlinear_adjoint_localisation="bubble_recovery",
        space_refinement_strategy="independent_slab",
        omit_galerkin_correction=True,
        snapshot_times=(0.5, 0.625, 0.75, 1.0),
        vtk_output_mode="all",
    )
    parser.add_argument("--p", type=float, default=4.0)
    parser.add_argument("--regularisation", type=float, default=1.0e-2)
    parser.add_argument(
        "--reference-quadrature",
        type=int,
        default=220,
        help="Gauss order for the manufactured reference value of J1.",
    )
    parser.add_argument(
        "--goal-quadrature-degree",
        type=int,
        default=16,
        help="One spatial quadrature degree shared by J1, J1', and DWR forms.",
    )
    parser.add_argument(
        "--all-slab-vtk",
        action="store_true",
        help=(
            "Write one ParaView collection for every time slab instead of "
            "only the four prescribed physical snapshot times."
        ),
    )
    args = parse_arguments(parser)

    if args.all_slab_vtk:
        args.snapshot_times = None

    if args.T < 0.75:
        parser.error("J1 requires T >= 0.75 to include its full time window")
    if args.space_refinement_strategy != "independent_slab":
        parser.error(
            "this experiment uses only independent_slab refinement; "
            "nested/shared-mesh variants are intentionally disabled"
        )
    if args.reference_quadrature < 8:
        parser.error("--reference-quadrature must be at least 8")
    if args.goal_quadrature_degree < 4:
        parser.error("--goal-quadrature-degree must be at least 4")

    problem = build_problem(args)
    config = config_from_arguments(args, nonlinear_identity=True)
    return NonstationaryDWRSolver(problem, config).solve()


if __name__ == "__main__":
    main()
