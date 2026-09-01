"""Paper-ready single-goal L-shaped J2 nonlinear DWR experiment.

The quantity of interest is the terminal spatial average from Section 6.4 of
[10] Review of DWR,

    J2(u) = integral_Omega u(x, T) dx,   T = 1.

The PDE, corrected homogeneous-boundary manufactured solution, p=4,
regularisation epsilon=1e-2, symmetric nonlinear DWR identity, hierarchical
recovery, global Dörfler marking, and independent slab-local refinement agree
with the current J1 experiment.  Unlike J1, J2 needs no fitted observation
box because its integration region is the whole L-shaped domain.
"""

from __future__ import annotations

import argparse

from moving_hill_dwr_experiments import (
    NonstationaryDWRSolver,
    add_common_arguments,
    config_from_arguments,
    parse_arguments,
)
from problem_lshape_parabolic_plaplace import build_problem


DEFAULT_OUTPUT = "output/thesis/lshape/j2_terminal_average_theta030_time010"


class _CurrentProblemAdapter:
    """Adapt the shared benchmark input to the current solver diagnostics."""

    def __init__(self, problem):
        self._problem = problem

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def goal_diagnostics(
        self, goal_value, estimator, true_error, *, symmetric_identity=False
    ):
        del symmetric_identity
        return self._problem.goal_diagnostics(goal_value, estimator, true_error)


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
        snapshot_times=(0.0, 0.5, 0.75, 1.0),
        vtk_output_mode="all",
        goal="j2",
        j1_weight=1.0,
        j2_weight=1.0,
    )
    parser.add_argument("--p", type=float, default=4.0)
    parser.add_argument("--regularisation", type=float, default=1.0e-2)
    parser.add_argument(
        "--reference-quadrature",
        type=int,
        default=220,
        help="Gauss order for the exact terminal-average reference value.",
    )
    parser.add_argument(
        "--all-slab-vtk",
        action="store_true",
        help="Write one ParaView collection for every time slab.",
    )
    args = parse_arguments(parser)

    if args.all_slab_vtk:
        args.snapshot_times = None
    if args.space_refinement_strategy != "independent_slab":
        parser.error("J2 uses the current independent_slab refinement only")
    if args.reference_quadrature < 8:
        parser.error("--reference-quadrature must be at least 8")

    problem = _CurrentProblemAdapter(build_problem(args))
    config = config_from_arguments(args, nonlinear_identity=True)
    return NonstationaryDWRSolver(problem, config).solve()


if __name__ == "__main__":
    main()
