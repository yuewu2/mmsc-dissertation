"""Validated input data for the transient bubble-projection solver.

The layout deliberately mirrors Firedrake's ``GoalAdaptiveOptions``: values
which control the outer adaptive algorithm live in one object, while all
remaining PETSc dictionaries are passed unchanged to the inner linear solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BubbleProjectionOptions:
    r"""Parameters for ``SOLVE -> ESTIMATE -> MARK -> REFINE``.

    The default DWR estimator is

    .. math::

       \eta = \rho(u_h)(z^\star),\qquad z^\star=z_{p+1,r}-z_{p,0},

    and the bubble projections split it into signed space--time contributions
    ``eta[K, n]``.  ``theta_spacetime`` applies global Dörfler marking to
    ``abs(eta[K, n])`` rather than independently marking each time slab.
    For a nonlinear problem, ``nonlinear_error_identity=True`` instead uses
    the practical symmetric two-term Lagrangian estimator and reports the
    enriched goal difference minus the computed terms as an observed cubic
    remainder diagnostic.
    """

    # The stop criterion is ``|eta_global| <= tolerance``.
    tolerance: float = 6.0e-4
    max_it: int = 6

    # Base trial space is ``CG(spatial_degree) x DG(primal_time_degree)``.
    spatial_degree: int = 1
    primal_time_degree: int = 0
    dual_extra_spatial_degree: int = 1
    dual_time_degree: int = 1
    # ``z*`` is either ``z_enriched-z_low`` or ``z_enriched-I_h(z_enriched)``.
    dual_weight_mode: str = "enriched_minus_numerical"
    # Use the symmetric nonlinear error identity with enriched primal and dual.
    nonlinear_error_identity: bool = False

    # Recovery represents cell residuals in ``DG(q_x) x P(q_t)``.
    recovery_space_degree: int = 1
    facet_recovery_degree: int = 1
    recovery_time_degree: int = 1
    recovery_quadrature_points: int = 5
    # Refuse adaptive marking when a recovered nonlinear residual does not
    # reproduce its global counterpart to this relative tolerance.
    localisation_closure_tolerance: float = 0.05

    # Dörfler bulk criterion: sum over marked ``|eta[K,n]|`` reaches theta.
    theta_spacetime: float = 0.2
    time_slab_marked_fraction: float = 0.05
    enable_space_refinement: bool = True
    enable_time_refinement: bool = True
    # ``independent_slab`` preserves the original fully slab-local meshes.
    # ``causal_nested`` makes every later slab inherit earlier marked regions.
    # ``shared_time_mesh`` refines their union once on every slab as a robust
    # common-mesh baseline, so no fine-to-coarse transfer occurs.
    space_refinement_strategy: str = "independent_slab"

    # Output is independent from numerical assembly and may be disabled.
    write_vtk: bool = True
    # ``spacetime_only`` writes only the 2-D (x,t) adaptive-grid collection.
    vtk_output_mode: str = "all"
    output_prefix: str = "output/comparison/heat_cone"
    verbose: bool = True

    # Adaptive-grid checkpointing.  A restart resolves the complete primal and
    # adjoint problems on the saved grid, so transient stage vectors are not
    # part of the checkpoint.
    checkpoint_prefix: str | None = None
    restart_from: str | None = None
    checkpoint_every: int = 1

    # PETSc options are intentionally forwarded unchanged to Firedrake.
    solver_parameters: dict[str, Any] | None = None
    recovery_solver_parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject inputs that violate the finite-element construction."""
        if self.max_it < 1:
            raise ValueError("max_it must be at least one.")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.spatial_degree < 1:
            raise ValueError("spatial_degree must be at least one.")
        if self.primal_time_degree < 0 or self.dual_time_degree < 0:
            raise ValueError("DG time degrees must be nonnegative.")
        if self.dual_extra_spatial_degree < 1:
            raise ValueError("dual_extra_spatial_degree must be positive.")
        if self.dual_weight_mode not in {
            "enriched_minus_numerical",
            "enriched_minus_interpolant",
        }:
            raise ValueError(
                "dual_weight_mode must be 'enriched_minus_numerical' or "
                "'enriched_minus_interpolant'."
            )
        if min(
            self.recovery_space_degree,
            self.facet_recovery_degree,
            self.recovery_time_degree,
        ) < 0:
            raise ValueError("Recovery polynomial degrees must be nonnegative.")
        if self.recovery_quadrature_points < 1:
            raise ValueError("recovery_quadrature_points must be at least one.")
        if not 0.0 < self.localisation_closure_tolerance < 1.0:
            raise ValueError("localisation_closure_tolerance must lie in (0, 1).")
        if not 0.0 < self.theta_spacetime <= 1.0:
            raise ValueError("theta_spacetime must lie in (0, 1].")
        if not 0.0 <= self.time_slab_marked_fraction <= 1.0:
            raise ValueError("time_slab_marked_fraction must lie in [0, 1].")
        if self.space_refinement_strategy not in {
            "independent_slab",
            "causal_nested",
            "shared_time_mesh",
        }:
            raise ValueError(
                "space_refinement_strategy must be 'independent_slab', "
                "'causal_nested', or 'shared_time_mesh'."
            )
        if self.vtk_output_mode not in {"all", "spacetime_only"}:
            raise ValueError("vtk_output_mode must be 'all' or 'spacetime_only'.")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least one.")

    @property
    def dual_spatial_degree(self) -> int:
        """Return ``p+1`` (or the requested higher-order dual degree)."""
        return self.spatial_degree + self.dual_extra_spatial_degree

    def recovery_parameters(self) -> dict[str, Any]:
        """Give projections a robust default mass-matrix solver."""
        return self.recovery_solver_parameters or {
            "mat_type": "aij",
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-11,
            "ksp_atol": 1.0e-13,
            "ksp_max_it": 1000,
        }
