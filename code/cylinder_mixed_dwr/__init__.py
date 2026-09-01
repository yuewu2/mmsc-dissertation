"""Standalone mixed-DAE support for the R5 cylinder benchmark.

This package intentionally does not modify or subclass ``nonstationary_dwr``.
It implements the Taylor--Hood mixed-specific operations required by the
independent slabwise DWR stages.
"""

from .adapter import CylinderMixedDAEAdapter
from .benchmark import (
    R5CylinderSpecification,
    drag_coefficient_form,
    drag_derivative_form,
    mean_drag_from_history,
    solve_primal,
    variational_drag_integrand_form,
    variational_drag_lift,
    variational_drag_spatial_derivative_form,
)
from .adjoint import solve_enriched_adjoint, solve_low_adjoint
from .enrichment import solve_enriched_primal
from .estimator import (
    LinearDWRGlobalEstimate,
    SymmetricDWREstimate,
    estimate_linear_dwr_global,
    estimate_symmetric_dwr,
)
from .localisation import (
    localise_linear_dwr,
    localise_symmetric_dwr,
    mark_localisation,
)
from .slabwise import (
    build_slab_transfers,
    interpolate_enriched_adjoint_to_low,
    solve_slabwise_adjoint,
    solve_slabwise_enriched_primal,
    solve_slabwise_primal,
    verify_transfer_pairing,
)
from .adaptive import (
    CylinderAdaptiveConfig,
    CylinderAdaptiveResult,
    CylinderAdaptiveSolver,
    refine_causal_slab_grid,
    refine_common_slab_grid,
    refine_independent_slab_grid,
)
from .checkpoint import CylinderCheckpointStore, LoadedCylinderGrid

__all__ = [
    "CylinderMixedDAEAdapter",
    "R5CylinderSpecification",
    "drag_coefficient_form",
    "drag_derivative_form",
    "mean_drag_from_history",
    "variational_drag_integrand_form",
    "variational_drag_lift",
    "variational_drag_spatial_derivative_form",
    "solve_primal",
    "solve_low_adjoint",
    "solve_enriched_adjoint",
    "solve_enriched_primal",
    "SymmetricDWREstimate",
    "LinearDWRGlobalEstimate",
    "estimate_linear_dwr_global",
    "estimate_symmetric_dwr",
    "localise_linear_dwr",
    "localise_symmetric_dwr",
    "mark_localisation",
    "build_slab_transfers",
    "interpolate_enriched_adjoint_to_low",
    "solve_slabwise_adjoint",
    "solve_slabwise_enriched_primal",
    "solve_slabwise_primal",
    "verify_transfer_pairing",
    "CylinderAdaptiveConfig",
    "CylinderAdaptiveResult",
    "CylinderAdaptiveSolver",
    "refine_causal_slab_grid",
    "refine_common_slab_grid",
    "refine_independent_slab_grid",
    "CylinderCheckpointStore",
    "LoadedCylinderGrid",
]
