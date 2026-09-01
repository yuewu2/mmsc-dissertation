"""Bubble/cone residual projection and DWR global/local error estimates.

The recovery follows the bubble-projection localisation identity: an interior
cell bubble recovers volume residuals and broken facet cones recover the two
independent traces required on an interior spatial facet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from finat.ufl import BrokenElement, FiniteElement
from firedrake import (
    Cofunction,
    Constant,
    Function,
    FunctionSpace,
    Mesh,
    MixedFunctionSpace,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    assemble,
    dS,
    ds,
    dx,
    solve,
)
from ufl import avg

from .parameters import BubbleProjectionOptions
from .problem import TransientDWRProblem
from .time_solver import (
    evaluate_slab,
    evaluate_slab_dt,
    gauss_rule,
    lagrange_derivatives,
    lagrange_values,
    linear_combination,
    time_nodes,
)


@dataclass
class BubbleEstimate:
    """One global DWR estimate and its signed space--time localisations.

    ``eta_cell_slab_signed[n][K]`` is ``eta_{K,n}``.  Its signed sum should
    reproduce ``eta_global`` up to numerical recovery error, whereas the sum
    of absolute values is the quantity used for stable Dörfler marking.
    """

    eta_global: float
    eta_local_sum: float
    eta_marking_sum: float
    eta_cell_slab_signed: list[np.ndarray | None]
    eta_volume_cell: list[np.ndarray | None]
    eta_spatial_facet_cell: list[np.ndarray | None]
    eta_temporal_facet_cell: list[np.ndarray | None]
    eta_mixed_ridge_cell: list[np.ndarray | None]
    eta_slab_signed: list[float]
    slab_activity: list[float]
    eta_K_abs_by_slab: list[Function | None]
    recovered_entities: list[dict[str, Any] | None]
    recovery_unknowns_proxy: int
    localisation_gap: float
    localisation_gap_relative: float
    localisation_consistency_index: float
    nonlinear_identity: bool = False
    eta_primal_global: float = 0.0
    eta_adjoint_global: float = 0.0
    eta_correction_global: float = 0.0
    eta_primal_local: float = 0.0
    eta_adjoint_local: float = 0.0
    eta_correction_local: float = 0.0
    primal_closure_gap: float = 0.0
    adjoint_closure_gap: float = 0.0
    correction_closure_gap: float = 0.0
    mixed_ridge_sum: float = 0.0
    localisation_gap_without_mixed: float = 0.0
    eta_primal_cell: list[np.ndarray | None] | None = None
    eta_adjoint_cell: list[np.ndarray | None] | None = None
    eta_correction_cell: list[np.ndarray | None] | None = None
    enriched_goal_difference: float | None = None
    observed_remainder: float | None = None


def _rename_subfunctions(function: Function, prefix: str) -> list[Function]:
    """Give each mixed-time coefficient a unique ParaView/debugging name."""
    coefficients = list(function.subfunctions)
    for index, coefficient in enumerate(coefficients):
        coefficient.rename(f"{prefix}_{index}")
    return coefficients


def dual_weight_on_slab(
    enriched_slab: dict[str, Any],
    low_slab: dict[str, Any],
    reference_time: float,
    mode: str,
):
    r"""Return the configurable DWR weight ``z_star`` on one time slab.

    The two supported weights are

    .. math::

       z^\star=z_h^+-z_h
       \quad\hbox{or}\quad
       z^\star=z_h^+-I_hz_h^+.

    In the interpolation mode, ``I_h`` is Firedrake interpolation into the
    numerical-dual/primal spatial space on the *same* physical time slab.
    The enriched dual remains the first term in both cases, so the residual
    action can be shared by every PDE.
    """
    enriched = evaluate_slab(enriched_slab, reference_time)
    if mode == "enriched_minus_numerical":
        return enriched - evaluate_slab(low_slab, reference_time)
    if mode == "enriched_minus_interpolant":
        low_space = low_slab["coeffs"][0].function_space()
        interpolant = Function(low_space, name="I_h_z_enriched")
        interpolant.interpolate(enriched)
        return enriched - interpolant
    raise ValueError(f"Unknown dual-weight mode: {mode!r}")


def recover_residual_entities_on_slab(
    slab_number: int,
    primal_slab: dict[str, Any],
    ts: np.ndarray,
    mesh: Mesh,
    problem: TransientDWRProblem,
    options: BubbleProjectionOptions,
    *,
    volume_action: Callable[..., Any] | None = None,
    temporal_action: Callable[..., Any] | None = None,
    temporal_action_right: Callable[..., Any] | None = None,
    name_prefix: str = "R",
) -> dict[str, Any]:
    r"""Recover volume, spatial-facet, and left-time-facet residual densities.

    The three reconstructed terms satisfy the weak residual partition

    .. math::

       \rho_n(v) \simeq (R_n,v)_{I_n\times\Omega}
       +(R_{\partial K,n},v)_{I_n\times\partial K}
       -(R_{t,n},v(t_{n-1}^+))_\Omega.

    ``Rhat_spatial`` stores ``cone * R_spatial``.  Dividing by a cone only
    occurs symbolically inside integration, avoiding an invalid pointwise
    division at a cone's zero set.

    A custom ``volume_action`` receives
    ``(tau, phi, partial_t_phi, time, measure)``.  This is used for nonlinear
    adjoint recovery, where the probe is a primal variation and the weak
    functional depends on both the variation and its physical time
    derivative.  The factor ``1/k_n`` in ``partial_t_phi`` is the chain-rule
    scaling from reference to physical slab time.
    """
    n = int(slab_number)
    step = float(ts[n] - ts[n - 1])
    dimension = mesh.topological_dimension
    if dimension == 1:
        return recover_residual_entities_on_1d_slab(
            n, primal_slab, ts, mesh, problem, options,
            volume_action=volume_action,
            temporal_action=temporal_action,
            temporal_action_right=temporal_action_right,
            name_prefix=name_prefix,
        )
    density_space = FunctionSpace(mesh, "DG", options.recovery_space_degree, variant="integral")
    recovery_nodes = time_nodes(options.recovery_time_degree)
    n_time_coefficients = len(recovery_nodes)
    quadrature = gauss_rule(options.recovery_quadrature_points)

    # ``B_K=1`` in the bubble space represents the canonical cell bubble.
    bubble = Function(FunctionSpace(mesh, "B", dimension + 1, variant="integral"), name=f"cell_bubbles_slab_{n}").assign(1.0)
    # A broken facet-cone density has independent traces from both neighbour cells.
    cones = Function(FunctionSpace(mesh, "FB", dimension, variant="integral"), name=f"facet_cones_slab_{n}").assign(1.0)
    facet_element = BrokenElement(FiniteElement(
        "FB", cell=mesh.ufl_cell(), degree=options.facet_recovery_degree + dimension, variant="integral",
    ))
    facet_space = FunctionSpace(mesh, facet_element)
    q_degree = max(2 * options.recovery_space_degree + 6, 2 * (options.spatial_degree + 1) + 4)
    dx_q = dx(metadata={"quadrature_degree": q_degree})
    ds_q = ds(metadata={"quadrature_degree": q_degree})
    dS_q = dS(metadata={"quadrature_degree": q_degree})
    solver_parameters = options.recovery_parameters()

    def apply_volume(s_q, local_test, local_test_dt, time, measure):
        if volume_action is not None:
            return volume_action(s_q, local_test, local_test_dt, time, measure)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        return problem.volume_residual_action(
            state, state_dt, local_test, time, measure=measure
        )

    def apply_temporal(local_test, measure):
        if temporal_action is not None:
            return temporal_action(local_test, measure)
        return problem.temporal_residual_action(
            evaluate_slab(primal_slab, 0.0), primal_slab["prev_right"],
            local_test, measure=measure,
        )

    # (1) Volume projection: solve ``(R, B_K q) = rho(B_K q)``.
    volume_space = MixedFunctionSpace([density_space] * n_time_coefficients)
    volume_trial = TrialFunctions(volume_space)
    volume_test = TestFunctions(volume_space)
    a_volume = 0
    L_volume = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        basis_dtau = lagrange_derivatives(recovery_nodes, s_q)
        temporal_bubble = 4.0 * s_q * (1.0 - s_q)
        temporal_bubble_dtau = 4.0 * (1.0 - 2.0 * s_q)
        residual_density = linear_combination(volume_trial, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        for index, test in enumerate(volume_test):
            local_test = Constant(temporal_bubble * basis[index]) * bubble * test
            local_test_dt = Constant(
                (temporal_bubble_dtau * basis[index]
                 + temporal_bubble * basis_dtau[index]) / step
            ) * bubble * test
            slab_weight = Constant(step * weight_q)
            a_volume += slab_weight * residual_density * local_test * dx_q
            L_volume += apply_volume(
                s_q, slab_weight * local_test, slab_weight * local_test_dt,
                time, dx_q,
            )
    volume = Function(volume_space, name=f"{name_prefix}_volume_slab_{n}")
    solve(a_volume == L_volume, volume, solver_parameters=solver_parameters)
    volume_coefficients = _rename_subfunctions(volume, f"{name_prefix}_volume_slab_{n}")

    # (2) Broken spatial traces: subtract volume residual then project each cell side.
    spatial_space = MixedFunctionSpace([facet_space] * n_time_coefficients)
    spatial_trial = TrialFunctions(spatial_space)
    spatial_test = TestFunctions(spatial_space)
    a_spatial = 0
    L_spatial = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        basis_dtau = lagrange_derivatives(recovery_nodes, s_q)
        temporal_bubble = 4.0 * s_q * (1.0 - s_q)
        temporal_bubble_dtau = 4.0 * (1.0 - 2.0 * s_q)
        volume_density = linear_combination(volume_coefficients, basis)
        spatial_density = linear_combination(spatial_trial, basis) / cones
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        slab_weight = Constant(step * weight_q)
        for index, test in enumerate(spatial_test):
            local_test = Constant(temporal_bubble * basis[index]) * test
            local_test_dt = Constant(
                (temporal_bubble_dtau * basis[index]
                 + temporal_bubble * basis_dtau[index]) / step
            ) * test
            # ``R(+)*v(+)+R(-)*v(-)`` retains separate cell-side facet data.
            a_spatial += slab_weight * (
                spatial_density * local_test * ds_q
                + (spatial_density("+") * local_test("+") + spatial_density("-") * local_test("-")) * dS_q
            )
            L_spatial += apply_volume(
                s_q, slab_weight * local_test, slab_weight * local_test_dt,
                time, dx_q,
            ) - slab_weight * volume_density * local_test * dx_q
    spatial_facet_hat = Function(spatial_space, name=f"{name_prefix}hat_spatial_facet_slab_{n}")
    solve(a_spatial == L_spatial, spatial_facet_hat, solver_parameters=solver_parameters)
    spatial_facet_coefficients = _rename_subfunctions(
        spatial_facet_hat, f"{name_prefix}hat_spatial_facet_slab_{n}"
    )

    # (3) The left temporal trace represents the DG jump ``u(t_{n-1}^+)-u(t_{n-1}^-)``.
    temporal_trial = TrialFunction(density_space)
    temporal_test = TestFunction(density_space)
    a_temporal = temporal_trial * bubble * temporal_test * dx_q
    L_temporal = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        temporal_cone = 1.0 - s_q
        volume_density = linear_combination(volume_coefficients, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        local_test = Constant(temporal_cone) * bubble * temporal_test
        local_test_dt = Constant(-1.0 / step) * bubble * temporal_test
        slab_weight = Constant(step * weight_q)
        L_temporal += apply_volume(
            s_q, slab_weight * local_test, slab_weight * local_test_dt,
            time, dx_q,
        ) - slab_weight * volume_density * local_test * dx_q
    jump_left = evaluate_slab(primal_slab, 0.0) - primal_slab["prev_right"]
    L_temporal += apply_temporal(bubble * temporal_test, dx_q)
    temporal_facet = Function(density_space, name=f"{name_prefix}_temporal_facet_slab_{n}")
    solve(a_temporal == L_temporal, temporal_facet, solver_parameters=solver_parameters)

    return {
        "space": density_space, "facet_space": facet_space, "cones": cones,
        "time_nodes": recovery_nodes, "volume": volume, "volume_coeffs": volume_coefficients,
        "spatial_facet_hat": spatial_facet_hat,
        "spatial_facet_hat_coeffs": spatial_facet_coefficients,
        "temporal_facet": temporal_facet, "jump_left": jump_left,
    }


def recover_residual_entities_on_1d_slab(
    slab_number: int,
    primal_slab: dict[str, Any],
    ts: np.ndarray,
    mesh: Mesh,
    problem: TransientDWRProblem,
    options: BubbleProjectionOptions,
    *,
    volume_action: Callable[..., Any] | None = None,
    temporal_action: Callable[..., Any] | None = None,
    temporal_action_right: Callable[..., Any] | None = None,
    name_prefix: str = "R",
) -> dict[str, Any]:
    r"""Recover volume, endpoint, temporal, and endpoint--time residuals.

    A one-dimensional cell has two point facets.  A broken DG1 field supplies
    their independent left/right traces, so it replaces the two-dimensional
    ``FB`` facet-cone element used by :func:`recover_residual_entities_on_slab`.
    """
    n = int(slab_number)
    step = float(ts[n] - ts[n - 1])
    density_space = FunctionSpace(mesh, "DG", options.recovery_space_degree, variant="integral")
    endpoint_space = FunctionSpace(mesh, "DG", 1, variant="equispaced")
    recovery_nodes = time_nodes(options.recovery_time_degree)
    n_time_coefficients = len(recovery_nodes)
    quadrature = gauss_rule(options.recovery_quadrature_points)
    bubble = Function(FunctionSpace(mesh, "B", 2, variant="integral"), name=f"cell_bubbles_slab_{n}").assign(1.0)
    q_degree = max(2 * options.recovery_space_degree + 6, 2 * (options.spatial_degree + 1) + 4)
    dx_q = dx(metadata={"quadrature_degree": q_degree})
    ds_q = ds(metadata={"quadrature_degree": q_degree})
    dS_q = dS(metadata={"quadrature_degree": q_degree})
    solver_parameters = options.recovery_parameters()

    def apply_volume(s_q, local_test, local_test_dt, time, measure):
        if volume_action is not None:
            return volume_action(s_q, local_test, local_test_dt, time, measure)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        return problem.volume_residual_action(
            state, state_dt, local_test, time, measure=measure
        )

    def apply_temporal(local_test, measure):
        if temporal_action is not None:
            return temporal_action(local_test, measure)
        return problem.temporal_residual_action(
            evaluate_slab(primal_slab, 0.0), primal_slab["prev_right"],
            local_test, measure=measure,
        )

    def apply_temporal_right(local_test, measure):
        if temporal_action_right is not None:
            return temporal_action_right(local_test, measure)
        return Constant(0.0) * local_test * measure

    volume_space = MixedFunctionSpace([density_space] * n_time_coefficients)
    volume_trial = TrialFunctions(volume_space)
    volume_test = TestFunctions(volume_space)
    a_volume = 0
    L_volume = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        basis_dtau = lagrange_derivatives(recovery_nodes, s_q)
        temporal_bubble = 4.0 * s_q * (1.0 - s_q)
        temporal_bubble_dtau = 4.0 * (1.0 - 2.0 * s_q)
        volume_density = linear_combination(volume_trial, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        slab_weight = Constant(step * weight_q)
        for index, test in enumerate(volume_test):
            local_test = Constant(temporal_bubble * basis[index]) * bubble * test
            local_test_dt = Constant(
                (temporal_bubble_dtau * basis[index]
                 + temporal_bubble * basis_dtau[index]) / step
            ) * bubble * test
            a_volume += slab_weight * volume_density * local_test * dx_q
            L_volume += apply_volume(
                s_q, slab_weight * local_test, slab_weight * local_test_dt,
                time, dx_q,
            )
    volume = Function(volume_space, name=f"{name_prefix}_volume_slab_{n}")
    solve(a_volume == L_volume, volume, solver_parameters=solver_parameters)
    volume_coefficients = _rename_subfunctions(volume, f"{name_prefix}_volume_slab_{n}")

    endpoint_mixed_space = MixedFunctionSpace([endpoint_space] * n_time_coefficients)
    endpoint_trial = TrialFunctions(endpoint_mixed_space)
    endpoint_test = TestFunctions(endpoint_mixed_space)
    a_endpoint = 0
    L_endpoint = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        basis_dtau = lagrange_derivatives(recovery_nodes, s_q)
        temporal_bubble = 4.0 * s_q * (1.0 - s_q)
        temporal_bubble_dtau = 4.0 * (1.0 - 2.0 * s_q)
        volume_density = linear_combination(volume_coefficients, basis)
        endpoint_density = linear_combination(endpoint_trial, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        slab_weight = Constant(step * weight_q)
        for index, test in enumerate(endpoint_test):
            local_test = Constant(temporal_bubble * basis[index]) * test
            local_test_dt = Constant(
                (temporal_bubble_dtau * basis[index]
                 + temporal_bubble * basis_dtau[index]) / step
            ) * test
            a_endpoint += slab_weight * (
                endpoint_density * local_test * ds_q
                + (endpoint_density("+") * local_test("+")
                   + endpoint_density("-") * local_test("-")) * dS_q
            )
            L_endpoint += apply_volume(
                s_q, slab_weight * local_test, slab_weight * local_test_dt,
                time, dx_q,
            ) - slab_weight * volume_density * local_test * dx_q
    endpoint = Function(endpoint_mixed_space, name=f"{name_prefix}_endpoint_slab_{n}")
    solve(a_endpoint == L_endpoint, endpoint, solver_parameters=solver_parameters)
    endpoint_coefficients = _rename_subfunctions(endpoint, f"{name_prefix}_endpoint_slab_{n}")

    temporal_trial = TrialFunction(density_space)
    temporal_test = TestFunction(density_space)
    a_temporal = temporal_trial * bubble * temporal_test * dx_q
    L_temporal = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        volume_density = linear_combination(volume_coefficients, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        local_test = Constant(1.0 - s_q) * bubble * temporal_test
        local_test_dt = Constant(-1.0 / step) * bubble * temporal_test
        slab_weight = Constant(step * weight_q)
        L_temporal += apply_volume(
            s_q, slab_weight * local_test, slab_weight * local_test_dt,
            time, dx_q,
        ) - slab_weight * volume_density * local_test * dx_q
    state_left = evaluate_slab(primal_slab, 0.0)
    L_temporal += apply_temporal(bubble * temporal_test, dx_q)
    temporal = Function(density_space, name=f"{name_prefix}_temporal_slab_{n}")
    solve(a_temporal == L_temporal, temporal, solver_parameters=solver_parameters)

    # The nonlinear adjoint residual is tested with the enriched-primal
    # defect, which need not vanish at the outgoing end of a slab.  Complete
    # the temporal decomposition with the right cone ``s_q``.  There is no
    # physical dG jump on this outgoing face; this entity is the remainder of
    # the slab-volume functional after the volume projection.
    L_temporal_right = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        volume_density = linear_combination(volume_coefficients, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        local_test = Constant(s_q) * bubble * temporal_test
        local_test_dt = Constant(1.0 / step) * bubble * temporal_test
        slab_weight = Constant(step * weight_q)
        L_temporal_right += apply_volume(
            s_q, slab_weight * local_test, slab_weight * local_test_dt,
            time, dx_q,
        ) - slab_weight * volume_density * local_test * dx_q
    L_temporal_right += apply_temporal_right(bubble * temporal_test, dx_q)
    temporal_right = Function(
        density_space, name=f"{name_prefix}_temporal_right_slab_{n}"
    )
    solve(
        a_temporal == L_temporal_right, temporal_right,
        solver_parameters=solver_parameters,
    )

    # The temporal cone does not vanish on a spatial endpoint.  Consequently
    # its complement after the volume and temporal-cell projections contains
    # an endpoint x left-time-interface functional.  In 1+1 dimensions this
    # is the point-like analogue of the edge x time-interface ridge in 2+1D.
    mixed_trial = TrialFunction(endpoint_space)
    mixed_test = TestFunction(endpoint_space)
    a_mixed = (
        mixed_trial * mixed_test * ds_q
        + (mixed_trial("+") * mixed_test("+")
           + mixed_trial("-") * mixed_test("-")) * dS_q
    )
    L_mixed = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        volume_density = linear_combination(volume_coefficients, basis)
        endpoint_density = linear_combination(endpoint_coefficients, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        local_test = Constant(1.0 - s_q) * mixed_test
        local_test_dt = Constant(-1.0 / step) * mixed_test
        slab_weight = Constant(step * weight_q)
        L_mixed += apply_volume(
            s_q, slab_weight * local_test, slab_weight * local_test_dt,
            time, dx_q,
        ) - slab_weight * volume_density * local_test * dx_q
        L_mixed -= slab_weight * (
            endpoint_density * local_test * ds_q
            + (endpoint_density("+") * local_test("+")
               + endpoint_density("-") * local_test("-")) * dS_q
        )
    L_mixed += apply_temporal(mixed_test, dx_q) - temporal * mixed_test * dx_q
    mixed_ridge = Function(endpoint_space, name=f"{name_prefix}_mixed_ridge_slab_{n}")
    solve(a_mixed == L_mixed, mixed_ridge, solver_parameters=solver_parameters)

    # Spatial endpoint x outgoing-time-face remainder.  This is the right
    # partner of ``mixed_ridge`` and is required whenever the DWR weight has a
    # non-zero right trace (notably the nonlinear adjoint weight e_u).
    L_mixed_right = 0
    for s_q, weight_q in quadrature:
        basis = lagrange_values(recovery_nodes, s_q)
        volume_density = linear_combination(volume_coefficients, basis)
        endpoint_density = linear_combination(endpoint_coefficients, basis)
        state = evaluate_slab(primal_slab, s_q)
        state_dt = evaluate_slab_dt(primal_slab, s_q, step)
        time = Constant(float(ts[n - 1] + step * s_q))
        local_test = Constant(s_q) * mixed_test
        local_test_dt = Constant(1.0 / step) * mixed_test
        slab_weight = Constant(step * weight_q)
        L_mixed_right += apply_volume(
            s_q, slab_weight * local_test, slab_weight * local_test_dt,
            time, dx_q,
        ) - slab_weight * volume_density * local_test * dx_q
        L_mixed_right -= slab_weight * (
            endpoint_density * local_test * ds_q
            + (endpoint_density("+") * local_test("+")
               + endpoint_density("-") * local_test("-")) * dS_q
        )
    L_mixed_right += (
        apply_temporal_right(mixed_test, dx_q)
        - temporal_right * mixed_test * dx_q
    )
    mixed_ridge_right = Function(
        endpoint_space, name=f"{name_prefix}_mixed_ridge_right_slab_{n}"
    )
    solve(
        a_mixed == L_mixed_right, mixed_ridge_right,
        solver_parameters=solver_parameters,
    )

    return {
        "space": density_space,
        "endpoint_space": endpoint_space,
        "time_nodes": recovery_nodes,
        "volume": volume,
        "volume_coeffs": volume_coefficients,
        "endpoint": endpoint,
        "endpoint_coeffs": endpoint_coefficients,
        "temporal_facet": temporal,
        "temporal_facet_right": temporal_right,
        "mixed_ridge": mixed_ridge,
        "mixed_ridge_right": mixed_ridge_right,
        "jump_left": state_left - primal_slab["prev_right"],
    }


def estimate_dwr_by_bubble_projection(
    primal: dict[str, Any],
    dual_enriched: dict[str, Any],
    dual_low: dict[str, Any],
    ts: np.ndarray,
    problem: TransientDWRProblem,
    options: BubbleProjectionOptions,
    *,
    primal_enriched: dict[str, Any] | None = None,
) -> BubbleEstimate:
    r"""Compute global DWR estimate and signed bubble-projected ``eta[K,n]``.

    The globally assembled quantity is

    .. math::

       \eta=\sum_n\left[\int_{I_n}\rho_n^V(z^\star)\,dt
       +\rho_n^T(z^\star(t_{n-1}^+))\right],

    where ``z_star=z_enriched-z_low``.  Each corresponding local quantity is
    evaluated with recovered volume, spatial-trace, and temporal-trace terms.
    The heat, advection--diffusion, and BBM equations therefore share this
    estimator equation while supplying their own two residual actions.
    """
    nonlinear_identity = bool(options.nonlinear_error_identity)
    if nonlinear_identity and primal_enriched is None:
        raise ValueError("The nonlinear DWR identity requires an enriched primal solution.")
    if primal["slabs"][1]["mesh"].comm.size != 1:
        raise NotImplementedError("Global space--time Dörfler sorting is currently serial.")
    nslabs = len(ts) - 1
    quadrature = gauss_rule(options.recovery_quadrature_points)
    q_degree = max(2 * (options.spatial_degree + 1) + 6, 10)
    dx_q = dx(metadata={"quadrature_degree": q_degree})
    ds_q = ds(metadata={"quadrature_degree": q_degree})
    dS_q = dS(metadata={"quadrature_degree": q_degree})

    signed: list[np.ndarray | None] = [None] * (nslabs + 1)
    volume_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    spatial_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    temporal_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    mixed_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    primal_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    adjoint_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    correction_by_cell: list[np.ndarray | None] = [None] * (nslabs + 1)
    slab_signed = [0.0] * (nslabs + 1)
    slab_activity = [0.0] * (nslabs + 1)
    recovered_entities: list[dict[str, Any] | None] = [None] * (nslabs + 1)
    absolute_indicators: list[Function | None] = [None] * (nslabs + 1)
    eta_primal_global = 0.0
    eta_adjoint_global = 0.0
    eta_correction_global = 0.0

    for n in range(1, nslabs + 1):
        step = float(ts[n] - ts[n - 1])
        primal_slab = primal["slabs"][n]
        enriched_slab = dual_enriched["slabs"][n]
        low_slab = dual_low["slabs"][n]
        enriched_primal_slab = None if primal_enriched is None else primal_enriched["slabs"][n]
        mesh = primal_slab["mesh"]
        is_one_dimensional = mesh.topological_dimension == 1
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_test = TestFunction(DG0)
        recovered = recover_residual_entities_on_slab(n, primal_slab, ts, mesh, problem, options)
        recovered_entities[n] = recovered
        recovered_adjoint = None
        if nonlinear_identity:
            def adjoint_reverse_volume_action(
                s_q, local_variation, local_variation_dt, time, measure,
            ):
                state = evaluate_slab(primal_slab, s_q)
                dual_low_value = evaluate_slab(low_slab, s_q)
                dual_low_dt = evaluate_slab_dt(low_slab, s_q, step)
                return (
                    problem.time_mass_action(
                        dual_low_dt, local_variation, measure=measure
                    )
                    + (1.0 + state) * local_variation
                    * dual_low_value.dx(0) * measure
                )

            def adjoint_left_propagation_action(local_variation, measure):
                dual_low_left = evaluate_slab(low_slab, 0.0)
                return problem.time_mass_action(
                    dual_low_left, local_variation, measure=measure
                )

            def adjoint_right_outgoing_action(local_variation, measure):
                dual_low_right = evaluate_slab(low_slab, 1.0)
                return -problem.time_mass_action(
                    dual_low_right, local_variation, measure=measure
                )

            recovered_adjoint = recover_residual_entities_on_slab(
                n, primal_slab, ts, mesh, problem, options,
                volume_action=adjoint_reverse_volume_action,
                temporal_action=adjoint_left_propagation_action,
                temporal_action_right=adjoint_right_outgoing_action,
                name_prefix="Rstar",
            )
            recovered["adjoint"] = recovered_adjoint
        volume_values = np.zeros(DG0.node_count, dtype=float)
        spatial_values = np.zeros(DG0.node_count, dtype=float)
        correction_volume_values = np.zeros(DG0.node_count, dtype=float)
        correction_spatial_values = np.zeros(DG0.node_count, dtype=float)
        adjoint_volume_values = np.zeros(DG0.node_count, dtype=float)
        adjoint_spatial_values = np.zeros(DG0.node_count, dtype=float)

        for s_q, weight_q in quadrature:
            recovery_basis = lagrange_values(recovered["time_nodes"], s_q)
            volume_density = linear_combination(recovered["volume_coeffs"], recovery_basis)
            if is_one_dimensional:
                spatial_density = linear_combination(recovered["endpoint_coeffs"], recovery_basis)
            else:
                spatial_density = (
                    linear_combination(recovered["spatial_facet_hat_coeffs"], recovery_basis)
                    / recovered["cones"]
                )
            state = evaluate_slab(primal_slab, s_q)
            state_dt = evaluate_slab_dt(primal_slab, s_q, step)
            dual_error = dual_weight_on_slab(
                enriched_slab, low_slab, s_q, options.dual_weight_mode
            )
            time = Constant(float(ts[n - 1] + step * s_q))
            slab_weight = Constant(step * weight_q)

            eta_primal_global += float(assemble(problem.volume_residual_action(
                state, state_dt, slab_weight * dual_error, time, measure=dx_q
            )))
            volume_vector: Cofunction = assemble(slab_weight * volume_density * dual_error * cell_test * dx_q)
            volume_values += np.asarray(volume_vector.dat.data_ro, dtype=float)
            if is_one_dimensional:
                spatial_vector: Cofunction = assemble(
                    slab_weight * (
                        spatial_density * dual_error * cell_test * ds_q
                        + (
                            spatial_density("+") * dual_error("+") * cell_test("+")
                            + spatial_density("-") * dual_error("-") * cell_test("-")
                        ) * dS_q
                    )
                )
            else:
                spatial_vector = assemble(
                    slab_weight * (
                        spatial_density * dual_error * cell_test * ds_q
                        + avg(spatial_density * dual_error) * (cell_test("+") + cell_test("-")) * dS_q
                    )
                )
            spatial_values += np.asarray(spatial_vector.dat.data_ro, dtype=float)

            if nonlinear_identity:
                dual_low_value = evaluate_slab(low_slab, s_q)
                eta_correction_global += float(assemble(problem.volume_residual_action(
                    state, state_dt, slab_weight * dual_low_value, time, measure=dx_q
                )))
                correction_volume_vector: Cofunction = assemble(
                    slab_weight * volume_density * dual_low_value * cell_test * dx_q
                )
                correction_volume_values += np.asarray(
                    correction_volume_vector.dat.data_ro, dtype=float
                )
                if is_one_dimensional:
                    correction_spatial_vector: Cofunction = assemble(
                        slab_weight * (
                            spatial_density * dual_low_value * cell_test * ds_q
                            + (
                                spatial_density("+") * dual_low_value("+") * cell_test("+")
                                + spatial_density("-") * dual_low_value("-") * cell_test("-")
                            ) * dS_q
                        )
                    )
                else:
                    correction_spatial_vector = assemble(
                        slab_weight * (
                            spatial_density * dual_low_value * cell_test * ds_q
                            + avg(spatial_density * dual_low_value)
                            * (cell_test("+") + cell_test("-")) * dS_q
                        )
                    )
                correction_spatial_values += np.asarray(
                    correction_spatial_vector.dat.data_ro, dtype=float
                )

                primal_error = evaluate_slab(enriched_primal_slab, s_q) - state
                primal_error_dt = (
                    evaluate_slab_dt(enriched_primal_slab, s_q, step) - state_dt
                )
                adjoint_form = problem.volume_residual_derivative_action(
                    state, state_dt, primal_error, primal_error_dt,
                    dual_low_value, time, cell_weight=slab_weight, measure=dx_q,
                )
                eta_adjoint_global += float(assemble(adjoint_form))
                adjoint_volume_density = linear_combination(
                    recovered_adjoint["volume_coeffs"], recovery_basis
                )
                adjoint_volume_vector: Cofunction = assemble(
                    slab_weight * adjoint_volume_density * primal_error
                    * cell_test * dx_q
                )
                adjoint_volume_values += np.asarray(
                    adjoint_volume_vector.dat.data_ro, dtype=float
                )
                if is_one_dimensional:
                    adjoint_spatial_density = linear_combination(
                        recovered_adjoint["endpoint_coeffs"], recovery_basis
                    )
                    adjoint_spatial_vector: Cofunction = assemble(
                        slab_weight * (
                            adjoint_spatial_density * primal_error * cell_test * ds_q
                            + (
                                adjoint_spatial_density("+") * primal_error("+") * cell_test("+")
                                + adjoint_spatial_density("-") * primal_error("-") * cell_test("-")
                            ) * dS_q
                        )
                    )
                else:
                    adjoint_spatial_density = (
                        linear_combination(
                            recovered_adjoint["spatial_facet_hat_coeffs"], recovery_basis
                        ) / recovered_adjoint["cones"]
                    )
                    adjoint_spatial_vector = assemble(
                        slab_weight * (
                            adjoint_spatial_density * primal_error * cell_test * ds_q
                            + avg(adjoint_spatial_density * primal_error)
                            * (cell_test("+") + cell_test("-")) * dS_q
                        )
                    )
                adjoint_spatial_values += np.asarray(
                    adjoint_spatial_vector.dat.data_ro, dtype=float
                )

        dual_error_left = dual_weight_on_slab(
            enriched_slab, low_slab, 0.0, options.dual_weight_mode
        )
        eta_primal_global += float(assemble(problem.temporal_residual_action(
            evaluate_slab(primal_slab, 0.0), primal_slab["prev_right"], dual_error_left, measure=dx_q
        )))
        temporal_vector: Cofunction = assemble(recovered["temporal_facet"] * dual_error_left * cell_test * dx_q)
        temporal_values = np.asarray(temporal_vector.dat.data_ro, dtype=float).copy()
        dual_error_right = dual_weight_on_slab(
            enriched_slab, low_slab, 1.0, options.dual_weight_mode
        )
        if is_one_dimensional and "temporal_facet_right" in recovered:
            temporal_right_vector: Cofunction = assemble(
                recovered["temporal_facet_right"] * dual_error_right
                * cell_test * dx_q
            )
            temporal_values += np.asarray(
                temporal_right_vector.dat.data_ro, dtype=float
            )
        if is_one_dimensional and "mixed_ridge" in recovered:
            mixed_vector: Cofunction = assemble(
                recovered["mixed_ridge"] * dual_error_left * cell_test * ds_q
                + (
                    recovered["mixed_ridge"]("+") * dual_error_left("+") * cell_test("+")
                    + recovered["mixed_ridge"]("-") * dual_error_left("-") * cell_test("-")
                ) * dS_q
            )
            mixed_values = np.asarray(mixed_vector.dat.data_ro, dtype=float).copy()
            if "mixed_ridge_right" in recovered:
                mixed_right_vector: Cofunction = assemble(
                    recovered["mixed_ridge_right"] * dual_error_right
                    * cell_test * ds_q
                    + (
                        recovered["mixed_ridge_right"]("+")
                        * dual_error_right("+") * cell_test("+")
                        + recovered["mixed_ridge_right"]("-")
                        * dual_error_right("-") * cell_test("-")
                    ) * dS_q
                )
                mixed_values += np.asarray(
                    mixed_right_vector.dat.data_ro, dtype=float
                )
        else:
            mixed_values = np.zeros_like(temporal_values)

        if nonlinear_identity:
            dual_low_left = evaluate_slab(low_slab, 0.0)
            eta_correction_global += float(assemble(problem.temporal_residual_action(
                evaluate_slab(primal_slab, 0.0), primal_slab["prev_right"],
                dual_low_left, measure=dx_q,
            )))
            correction_temporal_vector: Cofunction = assemble(
                recovered["temporal_facet"] * dual_low_left * cell_test * dx_q
            )
            correction_temporal_values = np.asarray(
                correction_temporal_vector.dat.data_ro, dtype=float
            ).copy()
            dual_low_right = evaluate_slab(low_slab, 1.0)
            if is_one_dimensional and "temporal_facet_right" in recovered:
                correction_temporal_right_vector: Cofunction = assemble(
                    recovered["temporal_facet_right"] * dual_low_right
                    * cell_test * dx_q
                )
                correction_temporal_values += np.asarray(
                    correction_temporal_right_vector.dat.data_ro, dtype=float
                )
            if is_one_dimensional and "mixed_ridge" in recovered:
                correction_mixed_vector: Cofunction = assemble(
                    recovered["mixed_ridge"] * dual_low_left * cell_test * ds_q
                    + (
                        recovered["mixed_ridge"]("+") * dual_low_left("+") * cell_test("+")
                        + recovered["mixed_ridge"]("-") * dual_low_left("-") * cell_test("-")
                    ) * dS_q
                )
                correction_mixed_values = np.asarray(
                    correction_mixed_vector.dat.data_ro, dtype=float
                ).copy()
                if "mixed_ridge_right" in recovered:
                    correction_mixed_right_vector: Cofunction = assemble(
                        recovered["mixed_ridge_right"] * dual_low_right
                        * cell_test * ds_q
                        + (
                            recovered["mixed_ridge_right"]("+")
                            * dual_low_right("+") * cell_test("+")
                            + recovered["mixed_ridge_right"]("-")
                            * dual_low_right("-") * cell_test("-")
                        ) * dS_q
                    )
                    correction_mixed_values += np.asarray(
                        correction_mixed_right_vector.dat.data_ro, dtype=float
                    )
            else:
                correction_mixed_values = np.zeros_like(correction_temporal_values)

            primal_error_left = (
                evaluate_slab(enriched_primal_slab, 0.0)
                - evaluate_slab(primal_slab, 0.0)
            )
            primal_error_previous = (
                enriched_primal_slab["prev_right"] - primal_slab["prev_right"]
            )
            primal_error_jump = primal_error_left - primal_error_previous
            adjoint_temporal_vector_recovered: Cofunction = assemble(
                recovered_adjoint["temporal_facet"] * primal_error_previous
                * cell_test * dx_q
            )
            adjoint_temporal_values = np.asarray(
                adjoint_temporal_vector_recovered.dat.data_ro, dtype=float
            ).copy()
            primal_error_right = (
                evaluate_slab(enriched_primal_slab, 1.0)
                - evaluate_slab(primal_slab, 1.0)
            )
            if (
                is_one_dimensional
                and "temporal_facet_right" in recovered_adjoint
            ):
                adjoint_temporal_right_vector: Cofunction = assemble(
                    recovered_adjoint["temporal_facet_right"]
                    * primal_error_right * cell_test * dx_q
                )
                adjoint_temporal_values += np.asarray(
                    adjoint_temporal_right_vector.dat.data_ro, dtype=float
                )
            if is_one_dimensional and "mixed_ridge" in recovered_adjoint:
                adjoint_mixed_vector: Cofunction = assemble(
                    recovered_adjoint["mixed_ridge"] * primal_error_previous
                    * cell_test * ds_q
                    + (
                        recovered_adjoint["mixed_ridge"]("+")
                        * primal_error_previous("+") * cell_test("+")
                        + recovered_adjoint["mixed_ridge"]("-")
                        * primal_error_previous("-") * cell_test("-")
                    ) * dS_q
                )
                adjoint_mixed_values = np.asarray(
                    adjoint_mixed_vector.dat.data_ro, dtype=float
                ).copy()
                if "mixed_ridge_right" in recovered_adjoint:
                    adjoint_mixed_right_vector: Cofunction = assemble(
                        recovered_adjoint["mixed_ridge_right"]
                        * primal_error_right * cell_test * ds_q
                        + (
                            recovered_adjoint["mixed_ridge_right"]("+")
                            * primal_error_right("+") * cell_test("+")
                            + recovered_adjoint["mixed_ridge_right"]("-")
                            * primal_error_right("-") * cell_test("-")
                        ) * dS_q
                    )
                    adjoint_mixed_values += np.asarray(
                        adjoint_mixed_right_vector.dat.data_ro, dtype=float
                    )
            else:
                adjoint_mixed_values = np.zeros_like(adjoint_temporal_values)
            adjoint_temporal_form = problem.temporal_residual_derivative_action(
                primal_error_left, primal_error_previous, dual_low_left,
                measure=dx_q,
            )
            eta_adjoint_global += float(assemble(adjoint_temporal_form))

            if n == nslabs:
                terminal_state = primal["nodes"][nslabs]
                terminal_error = (
                    primal_enriched["nodes"][nslabs] - terminal_state
                )
                goal_form = problem.goal_derivative_action(
                    mesh, terminal_state, terminal_error, measure=dx_q,
                )
                eta_adjoint_global += float(assemble(goal_form))
                goal_vector: Cofunction = assemble(problem.goal_derivative_action(
                    mesh, terminal_state, terminal_error,
                    cell_weight=cell_test, measure=dx_q,
                ))
                # The terminal QoI is an outgoing temporal-face functional.
                # It has a known smooth density, so it is integrated on that
                # face directly rather than treated as a slab-volume term.
                adjoint_temporal_values += np.asarray(
                    goal_vector.dat.data_ro, dtype=float
                )

            primal_values = volume_values + spatial_values + temporal_values + mixed_values
            correction_values = (
                correction_volume_values + correction_spatial_values
                + correction_temporal_values + correction_mixed_values
            )
            adjoint_values = (
                adjoint_volume_values + adjoint_spatial_values
                + adjoint_temporal_values + adjoint_mixed_values
            )
            # Practical nonlinear two-term estimator: the algebraic/Galerkin
            # correction -rho(u_h)(z_h) is intentionally omitted.  The
            # discrete primal solve is converged tightly enough that this is
            # treated as iteration error, not as a discretisation indicator.
            eta_values = 0.5 * primal_values + 0.5 * adjoint_values
            volume_by_cell[n] = (
                0.5 * volume_values + 0.5 * adjoint_volume_values
            )
            spatial_by_cell[n] = (
                0.5 * spatial_values + 0.5 * adjoint_spatial_values
            )
            temporal_by_cell[n] = (
                0.5 * temporal_values + 0.5 * adjoint_temporal_values
            )
            mixed_by_cell[n] = (
                0.5 * mixed_values + 0.5 * adjoint_mixed_values
            )
            primal_by_cell[n] = primal_values
            adjoint_by_cell[n] = adjoint_values.copy()
            correction_by_cell[n] = correction_values
        else:
            eta_values = volume_values + spatial_values + temporal_values + mixed_values
            volume_by_cell[n], spatial_by_cell[n] = volume_values, spatial_values
            temporal_by_cell[n], mixed_by_cell[n] = temporal_values, mixed_values
            primal_by_cell[n] = eta_values.copy()
            adjoint_by_cell[n] = np.zeros_like(eta_values)
            correction_by_cell[n] = np.zeros_like(eta_values)
        signed[n] = eta_values
        slab_signed[n] = float(eta_values.sum())
        slab_activity[n] = float(np.abs(eta_values).sum())
        absolute_indicators[n] = Function(DG0, name=f"eta_K_abs_slab_{n}")
        absolute_indicators[n].dat.data[:] = np.abs(eta_values)

    eta_global = (
        0.5 * eta_primal_global + 0.5 * eta_adjoint_global
        if nonlinear_identity else eta_primal_global
    )
    eta_local_sum = float(sum(slab_signed[1:]))
    eta_marking_sum = float(sum(slab_activity[1:]))
    localisation_gap = eta_local_sum - eta_global
    localisation_gap_relative = abs(localisation_gap) / max(abs(eta_global), np.finfo(float).eps)
    consistency = eta_local_sum / eta_global if abs(eta_global) > np.finfo(float).eps else float("nan")
    eta_primal_local = float(sum(values.sum() for values in primal_by_cell[1:]))
    eta_adjoint_local = float(sum(values.sum() for values in adjoint_by_cell[1:]))
    eta_correction_local = float(sum(values.sum() for values in correction_by_cell[1:]))
    mixed_ridge_sum = float(sum(values.sum() for values in mixed_by_cell[1:]))
    localisation_gap_without_mixed = localisation_gap - mixed_ridge_sum
    recovery_unknowns = sum(
        entity["volume"].function_space().dim()
        + (entity["endpoint"].function_space().dim()
           if "endpoint" in entity else entity["spatial_facet_hat"].function_space().dim())
        + entity["temporal_facet"].function_space().dim()
        + (entity["temporal_facet_right"].function_space().dim()
           if "temporal_facet_right" in entity else 0)
        + (entity["mixed_ridge"].function_space().dim() if "mixed_ridge" in entity else 0)
        + (entity["mixed_ridge_right"].function_space().dim()
           if "mixed_ridge_right" in entity else 0)
        + (
            entity["adjoint"]["volume"].function_space().dim()
            + (entity["adjoint"]["endpoint"].function_space().dim()
               if "endpoint" in entity["adjoint"]
               else entity["adjoint"]["spatial_facet_hat"].function_space().dim())
            + entity["adjoint"]["temporal_facet"].function_space().dim()
            + (entity["adjoint"]["temporal_facet_right"].function_space().dim()
               if "temporal_facet_right" in entity["adjoint"] else 0)
            + (entity["adjoint"]["mixed_ridge"].function_space().dim()
               if "mixed_ridge" in entity["adjoint"] else 0)
            + (entity["adjoint"]["mixed_ridge_right"].function_space().dim()
               if "mixed_ridge_right" in entity["adjoint"] else 0)
            if "adjoint" in entity else 0
        )
        for entity in recovered_entities[1:]
    )
    enriched_goal_difference = None
    observed_remainder = None
    if nonlinear_identity:
        goal_low = float(assemble(problem.goal_functional(
            primal["slabs"][nslabs]["mesh"], primal["nodes"][nslabs]
        )))
        goal_high = float(assemble(problem.goal_functional(
            primal_enriched["slabs"][nslabs]["mesh"],
            primal_enriched["nodes"][nslabs],
        )))
        enriched_goal_difference = goal_high - goal_low
        observed_remainder = enriched_goal_difference - eta_global
    return BubbleEstimate(
        eta_global=eta_global, eta_local_sum=eta_local_sum, eta_marking_sum=eta_marking_sum,
        eta_cell_slab_signed=signed, eta_volume_cell=volume_by_cell,
        eta_spatial_facet_cell=spatial_by_cell, eta_temporal_facet_cell=temporal_by_cell,
        eta_mixed_ridge_cell=mixed_by_cell,
        eta_slab_signed=slab_signed, slab_activity=slab_activity,
        eta_K_abs_by_slab=absolute_indicators,
        recovered_entities=recovered_entities, recovery_unknowns_proxy=int(recovery_unknowns),
        localisation_gap=localisation_gap, localisation_gap_relative=localisation_gap_relative,
        localisation_consistency_index=consistency,
        nonlinear_identity=nonlinear_identity,
        eta_primal_global=eta_primal_global,
        eta_adjoint_global=eta_adjoint_global,
        eta_correction_global=eta_correction_global,
        eta_primal_local=eta_primal_local,
        eta_adjoint_local=eta_adjoint_local,
        eta_correction_local=eta_correction_local,
        primal_closure_gap=eta_primal_local - eta_primal_global,
        adjoint_closure_gap=eta_adjoint_local - eta_adjoint_global,
        correction_closure_gap=eta_correction_local - eta_correction_global,
        mixed_ridge_sum=mixed_ridge_sum,
        localisation_gap_without_mixed=localisation_gap_without_mixed,
        eta_primal_cell=primal_by_cell,
        eta_adjoint_cell=adjoint_by_cell,
        eta_correction_cell=correction_by_cell,
        enriched_goal_difference=enriched_goal_difference,
        observed_remainder=observed_remainder,
    )
