r"""Opt-in R5-style temporal/spatial split of the linear DWR weight.

This module decomposes

    z+ - I_h I_k z+ = (z+ - I_k z+) + (I_k z+ - I_h I_k z+)

and applies the existing global estimator and three-part strong-residual
localisation to both summands.  The signed sums must recover the unchanged
production estimator and local indicators up to assembly roundoff.

It never replaces the production estimator.  Its marking role depends on
the configuration: by default it is purely diagnostic, but when the
adaptive configuration selects ``time_score_source='directional_time'``
the per-slab absolute mass of the temporal summand drives the temporal
marking decision.  That mode is an explicit algorithmic extension of the
dissertation's cell-fraction trigger, not part of the original method,
and must be reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from firedrake import Constant, DirichletBC, Function

from .estimator import LinearDWRGlobalEstimate, estimate_linear_dwr_global
from .localisation import BubbleConeLocalisation, localise_linear_dwr
from .slabwise import r5_temporal_interpolant_coefficients


@dataclass
class LinearDirectionalSplit:
    r"""Temporal/spatial components of one unchanged linear estimator."""

    eta_time: float
    eta_space: float
    eta_sum: float
    global_gap: float
    global_gap_relative: float
    eta_time_by_slab: list[float]
    eta_space_by_slab: list[float]
    local_time_sum: float
    local_space_sum: float
    local_sum: float
    local_gap: float
    local_gap_relative: float
    cell_gap_linf: float
    cell_gap_l1: float
    time_localisation: BubbleConeLocalisation
    space_localisation: BubbleConeLocalisation


def interpolate_enriched_adjoint_in_time(
    dual_enriched: dict[str, Any],
    labels: dict[str, Any],
    *,
    time_degree: int,
) -> dict[str, Any]:
    r"""Construct R5's ``I_k z+`` while retaining enriched spatial degree.

    The reverse-time enriched polynomial is interpolated at Gauss--Legendre
    support points and converted to Irksome's equispaced coefficient storage.
    This is the temporal half of the same tensor-product interpolation used
    by the production ``I_h I_k z+`` comparator.
    """
    degree = int(time_degree)
    if degree < 0:
        raise ValueError("time_degree must be nonnegative.")
    essential = tuple(labels["inlet"]) + tuple(labels["wall"])
    slabs: list[dict[str, Any] | None] = [None] * len(dual_enriched["slabs"])
    mixed_spaces: list[Any | None] = [None] * len(slabs)

    for n in range(1, len(slabs)):
        coefficients: list[Function] = []
        rich_space = dual_enriched["mixed_spaces"][n]
        mixed_spaces[n] = rich_space
        rich_coefficients = r5_temporal_interpolant_coefficients(
            dual_enriched["slabs"][n],
            degree,
            prefix=f"I_k_Z_rich_time_split_slab_{n}",
        )
        for stage, rich in enumerate(rich_coefficients):
            coefficient = Function(
                rich_space,
                name=f"I_k_Z_rich_time_split_slab_{n}_stage_{stage}",
            )
            coefficient.assign(rich)
            DirichletBC(
                rich_space.sub(0), Constant((0.0, 0.0)), essential
            ).apply(coefficient)
            coefficients.append(coefficient)
        slabs[n] = {
            "mesh": dual_enriched["slabs"][n]["mesh"],
            "degree": degree,
            "coeffs": coefficients,
        }

    return {
        "degree": degree,
        "spatially_enriched": True,
        "mixed_space": mixed_spaces[1],
        "mixed_spaces": mixed_spaces,
        "slabs": slabs,
        "construction": "R5_Gauss_Legendre_temporal_interpolation_of_enriched_adjoint",
    }


def compute_linear_directional_split(
    primal: dict[str, Any],
    dual_enriched: dict[str, Any],
    dual_low: dict[str, Any],
    baseline_estimate: LinearDWRGlobalEstimate,
    baseline_localisation: BubbleConeLocalisation,
    *,
    primal_time_degree: int,
    quadrature_points: int,
    primal_recovery_degree: int,
    facet_recovery_degree: int,
    recovery_time_degree: int,
) -> LinearDirectionalSplit:
    r"""Evaluate the R5 directional split without changing any marks."""
    temporal_interpolant = interpolate_enriched_adjoint_in_time(
        dual_enriched,
        primal["labels"],
        time_degree=int(primal_time_degree),
    )

    common_global = {
        "quadrature_points": int(quadrature_points),
        "dual_weight_mode": "enriched_minus_interpolant",
    }
    time_estimate = estimate_linear_dwr_global(
        primal,
        dual_enriched,
        temporal_interpolant,
        **common_global,
    )
    space_estimate = estimate_linear_dwr_global(
        primal,
        temporal_interpolant,
        dual_low,
        **common_global,
    )

    common_local = {
        "quadrature_points": int(quadrature_points),
        "dual_weight_mode": "enriched_minus_interpolant",
        "primal_recovery_degree": int(primal_recovery_degree),
        "facet_recovery_degree": int(facet_recovery_degree),
        "recovery_time_degree": int(recovery_time_degree),
    }
    time_localisation = localise_linear_dwr(
        primal,
        dual_enriched,
        temporal_interpolant,
        time_estimate,
        **common_local,
    )
    space_localisation = localise_linear_dwr(
        primal,
        temporal_interpolant,
        dual_low,
        space_estimate,
        **common_local,
    )

    eta_sum = float(time_estimate.eta_global + space_estimate.eta_global)
    global_gap = eta_sum - float(baseline_estimate.eta_global)
    global_scale = max(
        abs(float(baseline_estimate.eta_global)), np.finfo(float).eps
    )

    local_time_sum = float(time_localisation.eta_local_sum)
    local_space_sum = float(space_localisation.eta_local_sum)
    local_sum = local_time_sum + local_space_sum
    local_gap = local_sum - float(baseline_localisation.eta_local_sum)
    local_scale = max(
        abs(float(baseline_localisation.eta_local_sum)), np.finfo(float).eps
    )

    cell_gap_linf = 0.0
    cell_gap_l1 = 0.0
    for n in range(1, len(primal["times"])):
        recovered = (
            np.asarray(time_localisation.eta_cell_signed[n], dtype=float)
            + np.asarray(space_localisation.eta_cell_signed[n], dtype=float)
        )
        gap = recovered - np.asarray(
            baseline_localisation.eta_cell_signed[n], dtype=float
        )
        if gap.size:
            cell_gap_linf = max(cell_gap_linf, float(np.max(np.abs(gap))))
            cell_gap_l1 += float(np.sum(np.abs(gap)))

    eta_time_by_slab = [
        float(time_estimate.eta_volume_by_slab[n])
        + float(time_estimate.eta_temporal_jump_by_slab[n])
        for n in range(len(time_estimate.eta_volume_by_slab))
    ]
    eta_space_by_slab = [
        float(space_estimate.eta_volume_by_slab[n])
        + float(space_estimate.eta_temporal_jump_by_slab[n])
        for n in range(len(space_estimate.eta_volume_by_slab))
    ]
    return LinearDirectionalSplit(
        eta_time=float(time_estimate.eta_global),
        eta_space=float(space_estimate.eta_global),
        eta_sum=eta_sum,
        global_gap=global_gap,
        global_gap_relative=abs(global_gap) / global_scale,
        eta_time_by_slab=eta_time_by_slab,
        eta_space_by_slab=eta_space_by_slab,
        local_time_sum=local_time_sum,
        local_space_sum=local_space_sum,
        local_sum=local_sum,
        local_gap=local_gap,
        local_gap_relative=abs(local_gap) / local_scale,
        cell_gap_linf=cell_gap_linf,
        cell_gap_l1=cell_gap_l1,
        time_localisation=time_localisation,
        space_localisation=space_localisation,
    )


__all__ = [
    "LinearDirectionalSplit",
    "compute_linear_directional_split",
    "interpolate_enriched_adjoint_in_time",
]
