"""Command-line bubble-DWR experiments for nonlinear solitary-wave BBM.

``--case solitary`` uses the unforced long-domain solitary wave from the
official Irksome BBM demo.  Its default far-field Dirichlet interval supports
marked local 1D refinement; ``--boundary periodic`` retains the official
periodic baseline.  Both use the DG-in-time discretisation needed by this
project's temporal-jump recovery and slabwise refinement.
``--case manufactured`` retains the forced sine wave with an analytic terminal
goal value for a controlled effectivity check.
"""

from __future__ import annotations

import argparse

from firedrake.petsc import PETSc

from .parameters import BubbleProjectionOptions
from .problem import (
    BBMFiniteIntervalSolitaryWaveProblem,
    BBMSolitaryWaveProblem,
    BBMTravellingWaveProblem,
)
from .solver import BubbleProjectionAdaptiveSolver


def parse_arguments() -> argparse.Namespace:
    """Read BBM and adaptive inputs without leaking them into PETSc options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("solitary", "manufactured"), default="solitary",
        help="Use the physical unforced solitary wave or the forced sine verification case.",
    )
    parser.add_argument("--nx", type=int, default=64, help="Initial spatial cell count.")
    parser.add_argument("--nt", type=int, default=4, help="Initial number of DG time slabs.")
    parser.add_argument("--T", type=float, default=18.0, help="Final physical time.")
    parser.add_argument("--length", type=float, default=100.0, help="Domain length for --case solitary.")
    parser.add_argument(
        "--boundary", choices=("dirichlet", "periodic"), default="periodic",
        help="Locally refinable far-field Dirichlet interval or the official periodic baseline.",
    )
    parser.add_argument("--amplitude-parameter", type=float, default=0.5,
                        help="Solitary-wave parameter c in (0,1); speed is 1/(1-c^2).")
    parser.add_argument("--initial-centre", type=float, default=30.0,
                        help="Initial solitary-wave centre for --case solitary.")
    parser.add_argument("--speed", type=float, default=0.35, help="Manufactured-wave speed.")
    parser.add_argument("--sensor-centre", type=float, default=None,
                        help="Terminal Gaussian-sensor centre; by default follows the solitary-wave centre.")
    parser.add_argument("--sensor-radius", type=float, default=None,
                        help="Positive terminal-sensor radius (1 for solitary, 0.06 for manufactured).")
    parser.add_argument(
        "--goal", choices=("terminal_sensor", "invariant_i2"), default="terminal_sensor",
        help="Terminal Gaussian observation or BBM H1 invariant I2 at final time.",
    )
    parser.add_argument("--max-it", type=int, default=4, help="Maximum adaptive cycles.")
    parser.add_argument("--tolerance", type=float, default=1.0e-5, help="Stop if |eta_global| is below this value.")
    parser.add_argument("--theta", type=float, default=0.25, help="Global space-time Doerfler fraction.")
    parser.add_argument("--time-fraction", type=float, default=0.20, help="Marked-cell fraction which bisects a time slab.")
    parser.add_argument(
        "--recovery-space-degree", type=int, default=1,
        help="Spatial polynomial degree of recovered cell residual densities.",
    )
    parser.add_argument(
        "--recovery-time-degree", type=int, default=1,
        help="Reference-time polynomial degree of recovered residual densities.",
    )
    parser.add_argument(
        "--recovery-quadrature-points", type=int, default=5,
        help="Gauss points per time slab used by global and recovered DWR terms.",
    )
    parser.add_argument(
        "--space-refinement-strategy",
        choices=("independent_slab", "causal_nested", "shared_time_mesh"),
        default="independent_slab",
        help="How marked spatial regions are transferred between time slabs.",
    )
    parser.add_argument(
        "--vtk-output", choices=("spacetime_only", "all"), default="spacetime_only",
        help="Write only the adaptive (x,t) PVD collection, or all auxiliary VTK outputs.",
    )
    parser.add_argument("--output-prefix", default="output/bbm_dwr/bbm", help="ParaView/CSV output prefix.")
    parser.add_argument("--no-vtk", action="store_true", help="Disable ParaView output but keep the CSV history.")
    parser.add_argument(
        "--checkpoint-prefix", default=None,
        help="Prefix for restartable adaptive-grid checkpoints.",
    )
    parser.add_argument(
        "--restart-from", default=None,
        help="Checkpoint prefix (or metadata JSON) from which to continue.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1,
        help="Write a restart state after this many completed refinements.",
    )
    args = parser.parse_args()
    petsc = PETSc.Options()
    for action in parser._actions:
        for flag in action.option_strings:
            if flag.startswith("--"):
                petsc.delValue(flag)
    return args


def main() -> BubbleProjectionAdaptiveSolver:
    """Build one nonlinear BBM input and execute the common slabwise driver."""
    args = parse_arguments()
    if args.case == "solitary":
        wave_speed = 1.0 / (1.0 - args.amplitude_parameter**2)
        sensor_centre = (
            args.initial_centre + wave_speed * args.T
            if args.sensor_centre is None else args.sensor_centre
        )
        problem_type = (
            BBMFiniteIntervalSolitaryWaveProblem
            if args.boundary == "dirichlet"
            else BBMSolitaryWaveProblem
        )
        problem = problem_type(
            length=args.length,
            amplitude_parameter=args.amplitude_parameter,
            initial_center=args.initial_centre,
            sensor_center=sensor_centre,
            sensor_radius=1.0 if args.sensor_radius is None else args.sensor_radius,
            goal_mode=args.goal,
        )
    else:
        if args.goal != "terminal_sensor":
            raise ValueError("--goal invariant_i2 is defined only for --case solitary.")
        problem = BBMTravellingWaveProblem(
            speed=args.speed,
            sensor_center=0.65 if args.sensor_centre is None else args.sensor_centre,
            sensor_radius=0.06 if args.sensor_radius is None else args.sensor_radius,
        )
    options = BubbleProjectionOptions(
        max_it=args.max_it,
        tolerance=args.tolerance,
        theta_spacetime=args.theta,
        time_slab_marked_fraction=args.time_fraction,
        recovery_space_degree=args.recovery_space_degree,
        recovery_time_degree=args.recovery_time_degree,
        recovery_quadrature_points=args.recovery_quadrature_points,
        space_refinement_strategy=args.space_refinement_strategy,
        vtk_output_mode=args.vtk_output,
        nonlinear_error_identity=True,
        # The forward BBM stage system is nonlinear; the two reverse solves
        # are linear after each has been linearised at its saved primal slab.
        solver_parameters={
            "mat_type": "aij",
            "snes_type": "newtonls",
            "snes_rtol": 1.0e-10,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        write_vtk=not args.no_vtk,
        output_prefix=args.output_prefix,
        checkpoint_prefix=args.checkpoint_prefix,
        restart_from=args.restart_from,
        checkpoint_every=args.checkpoint_every,
    )
    return BubbleProjectionAdaptiveSolver(
        problem,
        problem.make_mesh(args.nx),
        BubbleProjectionAdaptiveSolver.uniform_time_grid(args.T, args.nt),
        options=options,
    ).solve()


if __name__ == "__main__":
    main()
