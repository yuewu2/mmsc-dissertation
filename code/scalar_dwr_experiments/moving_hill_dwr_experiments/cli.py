"""Shared command-line interface for the small problem-input files."""

from __future__ import annotations

import argparse

from firedrake.petsc import PETSc

from .options import NonstationaryDWRConfig


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_T: float,
    default_nx: int,
    default_nt: int,
    default_output: str,
    default_max_it: int = 4,
    default_theta: float = 0.2,
    default_time_fraction: float = 0.05,
) -> None:
    """Add controls shared by every transient example."""
    parser.add_argument("--nx", type=int, default=default_nx)
    parser.add_argument(
        "--ny", type=int, default=None,
        help="Second spatial resolution; defaults to the value passed with --nx.",
    )
    parser.add_argument("--nt", type=int, default=default_nt)
    parser.add_argument("--T", type=float, default=default_T)
    parser.add_argument("--max-it", type=int, default=default_max_it)
    parser.add_argument("--tolerance", type=float, default=6.0e-4)
    parser.add_argument(
        "--theta", type=float, default=default_theta,
        help=(
            "Global/space Dörfler bulk parameter.  With fixed_rate this is "
            "instead the fraction of cells marked on every slab."
        ),
    )
    parser.add_argument(
        "--time-fraction", type=float, default=default_time_fraction,
        help=(
            "For global_bulk_fraction_trigger: minimum marked-cell count fraction "
            "specified by --time-trigger-policy; for separate_bulk: temporal "
            "Dörfler bulk; for fixed_rate: fraction of slabs selected by count."
        ),
    )
    parser.add_argument(
        "--time-trigger-policy",
        choices=("within_slab_fraction", "global_marked_share"),
        default="within_slab_fraction",
        help=(
            "Temporal trigger used with global_bulk_fraction_trigger.  The "
            "default retains the original marked_count[n]/cell_count[n] rule; "
            "global_marked_share uses marked_count[n]/marked_count[all slabs]."
        ),
    )
    parser.add_argument(
        "--marking-strategy",
        choices=(
            "global_bulk_fraction_trigger", "separate_bulk", "fixed_rate", "uniform"
        ),
        default="global_bulk_fraction_trigger",
    )
    parser.add_argument(
        "--space-refinement-strategy",
        choices=("independent_slab", "causal_nested", "shared_time_mesh"),
        default="independent_slab",
    )
    parser.add_argument("--spatial-degree", type=int, default=1)
    parser.add_argument("--primal-time-degree", type=int, default=0)
    parser.add_argument("--dual-extra-spatial-degree", type=int, default=1)
    parser.add_argument("--dual-time-degree", type=int, default=1)
    parser.add_argument(
        "--dual-weight-mode",
        choices=("enriched_minus_numerical", "enriched_minus_interpolant"),
        default="enriched_minus_numerical",
    )
    parser.add_argument(
        "--localisation-mode",
        choices=(
            "hierarchical_recovery",
            "joint_cell_partition",
            "weak_cell_partition",
            "strong_residual_bound",
        ),
        default="hierarchical_recovery",
        help=(
            "Use the thesis hierarchical recovery, or an algebraically "
            "closing DG0 joint cell/slab partition.  weak_cell_partition is "
            "a compatibility alias; neither name denotes the paper's nodal "
            "cG(1)dG(0) split PU estimator.  strong_residual_bound uses a "
            "positive manual residual bound for marking while retaining the "
            "same signed global DWR estimate and goal."
        ),
    )
    parser.add_argument(
        "--bubble-marking-score",
        choices=("signed_total", "componentwise_abs"),
        default="signed_total",
        help=(
            "For hierarchical_recovery, mark with |sum_j eta_j| (legacy) or "
            "sum_j |eta_j| while preserving the signed estimator and closure."
        ),
    )
    parser.add_argument(
        "--diagnostic-marking-component",
        choices=("total", "primal", "adjoint"),
        default="total",
        help=(
            "Diagnostic nonlinear ablation: keep the solves and reported DWR "
            "estimator unchanged, but drive refinement with the complete two-term "
            "indicator, only rho(e_z), or only rho*(e_u).  The production default "
            "is total."
        ),
    )
    parser.add_argument(
        "--nonlinear-adjoint-localisation",
        choices=("bubble_recovery", "cellwise_dg0"),
        default="bubble_recovery",
        help=(
            "For the symmetric nonlinear DWR identity, localise the adjoint-"
            "residual action by a second bubble--cone recovery (default), or "
            "retain the earlier DG0 algebraic partition for compatibility."
        ),
    )
    parser.add_argument(
        "--omit-mixed-ridge",
        action="store_true",
        help=(
            "Diagnostic hierarchical-bubble ablation: omit the mixed-ridge "
            "contribution from localisation and marking.  This is not the "
            "complete bubble--cone identity."
        ),
    )
    parser.add_argument(
        "--omit-galerkin-correction",
        action="store_true",
        help=(
            "For the nonlinear identity use only 1/2*rho(e_z) + "
            "1/2*rho*(e_u), omitting -rho(z_h) globally and locally. "
            "The default preserves the complete three-term estimator."
        ),
    )
    parser.add_argument("--recovery-space-degree", type=int, default=1)
    parser.add_argument("--recovery-facet-degree", type=int, default=1)
    parser.add_argument("--recovery-time-degree", type=int, default=1)
    parser.add_argument(
        "--recovery-degree",
        type=int,
        default=None,
        help="Compatibility shortcut: set all three recovery degrees together.",
    )
    parser.add_argument("--quadrature-points", type=int, default=5)
    parser.add_argument(
        "--symmetric-dwr-identity",
        action="store_true",
        help=(
            "Use the symmetric DWR identity for a nonlinear goal functional. "
            "This solves an enriched primal too; the heat PDE itself remains linear."
        ),
    )
    parser.add_argument("--output-prefix", default=default_output)
    parser.add_argument(
        "--vtk-output-mode", choices=("all", "spacetime_only"), default="all"
    )
    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Write only the containing/nearest physical slab for each requested "
            "time (for example: --snapshot-times 0.25 0.5 0.75 1.0).  Omitting "
            "this option preserves the legacy all-slab VTK output."
        ),
    )
    parser.add_argument("--no-vtk", action="store_true")


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse argparse flags and remove them from PETSc's option database."""
    args = parser.parse_args()
    petsc_options = PETSc.Options()
    for parser_action in parser._actions:
        for option in parser_action.option_strings:
            if option.startswith("--"):
                petsc_options.delValue(option)
    return args


def config_from_arguments(
    args: argparse.Namespace,
    *,
    nonlinear_identity: bool = False,
) -> NonstationaryDWRConfig:
    """Translate common CLI inputs into the public solver configuration."""
    shortcut = args.recovery_degree
    return NonstationaryDWRConfig(
        nx=args.nx,
        ny=args.nx if args.ny is None else args.ny,
        nt=args.nt,
        final_time=args.T,
        max_it=args.max_it,
        tolerance=args.tolerance,
        theta=args.theta,
        time_fraction=args.time_fraction,
        marking_strategy=args.marking_strategy,
        time_trigger_policy=args.time_trigger_policy,
        space_refinement_strategy=args.space_refinement_strategy,
        spatial_degree=args.spatial_degree,
        primal_time_degree=args.primal_time_degree,
        dual_extra_spatial_degree=args.dual_extra_spatial_degree,
        dual_time_degree=args.dual_time_degree,
        dual_weight_mode=args.dual_weight_mode,
        localisation_mode=args.localisation_mode,
        bubble_marking_score=args.bubble_marking_score,
        diagnostic_marking_component=args.diagnostic_marking_component,
        nonlinear_adjoint_localisation=args.nonlinear_adjoint_localisation,
        include_mixed_ridge=not args.omit_mixed_ridge,
        recovery_space_degree=(
            args.recovery_space_degree if shortcut is None else shortcut
        ),
        recovery_facet_degree=(
            args.recovery_facet_degree if shortcut is None else shortcut
        ),
        recovery_time_degree=(
            args.recovery_time_degree if shortcut is None else shortcut
        ),
        quadrature_points=args.quadrature_points,
        nonlinear_identity=nonlinear_identity or args.symmetric_dwr_identity,
        include_galerkin_correction=not args.omit_galerkin_correction,
        write_vtk=not args.no_vtk,
        vtk_output_mode=args.vtk_output_mode,
        snapshot_times=(
            None if args.snapshot_times is None else tuple(args.snapshot_times)
        ),
        output_prefix=args.output_prefix,
    )


__all__ = ["add_common_arguments", "config_from_arguments", "parse_arguments"]
