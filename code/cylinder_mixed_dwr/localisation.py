r"""Tensor-product bubble/cone localisation for the mixed cylinder DWR problem.

The recovery is performed on a complete space--time slab. Momentum and
continuity volume densities share one temporal polynomial basis; the velocity
facet density uses the same basis. The remaining physical-left (or
reverse-left) temporal trace and the spatial-facet x temporal-interface ridge
are recovered after subtracting the preceding entities.

Pressure is algebraic: it occurs in the mixed volume block, but never in a
temporal jump or mixed ridge. No weak DG0 partition contributes to marking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from finat.ufl import BrokenElement, FiniteElement, VectorElement
from firedrake import (
    Constant,
    FacetNormal,
    Function,
    FunctionSpace,
    Identity,
    MixedFunctionSpace,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    VectorFunctionSpace,
    avg,
    assemble,
    as_vector,
    div,
    dot,
    ds,
    dS,
    dx,
    grad,
    inner,
    jump,
    outer,
    solve,
    split,
)

from automated_DWR.mark_refine import mark_spacetime_cells
from automated_DWR.time_solver import (
    gauss_rule,
    lagrange_values,
    time_nodes,
)
from navier_stokes_cylinder_irksome_static_primal import (
    evaluate_slab,
    evaluate_slab_dt,
)


def _direct_parameters() -> dict[str, Any]:
    """Direct solver used by the slab-local recovery projections.

    Keep this small configuration local: the former static-localisation
    prototype is not part of the production package and must not be a runtime
    dependency of the adaptive cylinder experiment.
    """
    return {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

from .benchmark import drag_derivative_form
from .estimator import (
    LinearDWRGlobalEstimate,
    SymmetricDWREstimate,
    _low_incoming_in_rich_operator,
    dual_error_at,
    primal_error_at,
    primal_trace_error,
)


@dataclass
class BubbleConeLocalisation:
    eta_cell_signed: list[np.ndarray | None]
    eta_primal_bubble_cell: list[np.ndarray | None]
    eta_adjoint_bubble_cell: list[np.ndarray | None]
    eta_correction_bubble_cell: list[np.ndarray | None]
    eta_cubic_remainder_cell: list[np.ndarray | None]
    eta_primal_mixed_ridge_cell: list[np.ndarray | None]
    eta_adjoint_mixed_ridge_cell: list[np.ndarray | None]
    eta_correction_mixed_ridge_cell: list[np.ndarray | None]
    eta_primal_volume_cell: list[np.ndarray | None]
    eta_primal_spatial_cell: list[np.ndarray | None]
    eta_primal_temporal_cell: list[np.ndarray | None]
    eta_adjoint_volume_cell: list[np.ndarray | None]
    eta_adjoint_spatial_cell: list[np.ndarray | None]
    eta_adjoint_temporal_cell: list[np.ndarray | None]
    eta_correction_volume_cell: list[np.ndarray | None]
    eta_correction_spatial_cell: list[np.ndarray | None]
    eta_correction_temporal_cell: list[np.ndarray | None]
    eta_local_sum: float
    eta_marking_sum: float
    localisation_gap: float
    eta_primal_recovered: float
    eta_adjoint_recovered: float
    eta_correction_recovered: float
    primal_recovery_gap: float
    adjoint_recovery_gap: float
    correction_recovery_gap: float
    eta_adjoint_reverse_weak: float
    adjoint_reverse_identity_gap: float
    dual_weight_mode: str
    fields: list[Function | None]
    primal_slab_recovery_gap: list[float]
    adjoint_slab_recovery_gap: list[float]
    correction_slab_recovery_gap: list[float]
    recovered_entities: list[dict[str, Any] | None]
    recovery_time_degree: int


@dataclass
class MarkingDecision:
    marked_by_slab: list[np.ndarray | None]
    spatial_union: np.ndarray
    spatial_marker: Function
    time_slabs: set[int]
    marked_fraction_by_slab: list[float]
    selected_activity: float
    total_activity: float


def _rename(function: Function, prefix: str) -> list[Function]:
    coefficients = list(function.subfunctions)
    for index, coefficient in enumerate(coefficients):
        coefficient.rename(f"{prefix}_{index}")
    return coefficients


def linear_combination(coefficients, weights):
    """Combine time modes without in-place addition on UFL vector tensors."""
    return sum(
        (float(weight) * coefficient
         for weight, coefficient in zip(weights, coefficients)),
        0.0 * coefficients[0],
    )


def _vector_blocks(blocks):
    """Normalise Python component tuples while retaining UFL ListTensors."""
    return tuple(
        as_vector(block) if isinstance(block, (tuple, list)) else block
        for block in blocks
    )


def _facet_space(mesh, degree: int):
    dimension = mesh.topological_dimension
    scalar = FiniteElement(
        "FB", cell=mesh.ufl_cell(), degree=int(degree) + dimension,
        variant="integral",
    )
    return FunctionSpace(
        mesh,
        VectorElement(BrokenElement(scalar), dim=int(mesh.geometric_dimension)),
    )


def _facet_pair(density, test, dS_q, ds_outlet):
    return (
        inner(density, test) * ds_outlet
        + (
            inner(density("+"), test("+"))
            + inner(density("-"), test("-"))
        ) * dS_q
    )


def _primal_action(
    state, state_dt, velocity_test, pressure_test, viscosity, *, measure,
):
    """Return the mixed primal residual ``rho=-A`` on arbitrary blocks."""
    velocity, pressure = split(state)
    velocity_dt = split(state_dt)[0]
    residual = 0
    if velocity_test is not None:
        residual += (
            -inner(velocity_dt, velocity_test)
            - inner(grad(velocity) * velocity, velocity_test)
            - viscosity * inner(grad(velocity), grad(velocity_test))
            + pressure * div(velocity_test)
        )
    if pressure_test is not None:
        residual += div(velocity) * pressure_test
    return residual * measure


def _adjoint_action(
    primal_velocity, dual_state, dual_tau_derivative,
    velocity_variation, pressure_variation, viscosity, *, measure,
):
    r"""Return ``-B^V(Z_h)(delta U)`` in reverse time, without ``J'``."""
    dual_velocity, dual_pressure = split(dual_state)
    dual_velocity_tau = split(dual_tau_derivative)[0]
    residual = 0
    if velocity_variation is not None:
        residual += (
            -inner(dual_velocity_tau, velocity_variation)
            - inner(
                grad(velocity_variation) * primal_velocity
                + grad(primal_velocity) * velocity_variation,
                dual_velocity,
            )
            - viscosity * inner(grad(velocity_variation), grad(dual_velocity))
            + div(velocity_variation) * dual_pressure
        )
    if pressure_variation is not None:
        residual += pressure_variation * div(dual_velocity)
    return residual * measure


def _primal_jump(primal_slab, incoming_trace=None) -> Function:
    state_left = evaluate_slab(primal_slab, 0.0, name="U_primal_left_plus")
    jump = Function(
        state_left.subfunctions[0].function_space(), name="primal_velocity_jump"
    )
    incoming_velocity = (
        primal_slab["left_trace"] if incoming_trace is None else incoming_trace
    ).subfunctions[0]
    difference = state_left.subfunctions[0] - incoming_velocity
    if incoming_velocity.function_space() == jump.function_space():
        jump.assign(difference)
    else:
        jump.project(difference, solver_parameters=_direct_parameters())
    return jump


def _dual_physical_right_trace(dual_slab) -> Function:
    reverse_left = evaluate_slab(dual_slab, 0.0, name="Z_reverse_left_plus")
    trace = Function(
        reverse_left.subfunctions[0].function_space(),
        name="dual_velocity_physical_right_trace",
    )
    trace.assign(reverse_left.subfunctions[0])
    return trace


def _recovery_geometry(
    mesh, outlet_labels, slab_number: int, prefix: str,
    recovery_space_degree: int, facet_recovery_degree: int,
):
    n = int(slab_number)
    q_degree = max(12, 2 * int(recovery_space_degree) + 8)
    dx_q = dx(metadata={"quadrature_degree": q_degree})
    dS_q = dS(metadata={"quadrature_degree": q_degree})
    # The broken FB element has exterior-facet modes but no pointwise trace
    # nodes on which Firedrake can impose a DirichletBC.  Recover on every
    # exterior facet to keep the mass matrix nonsingular.  All DWR velocity
    # weights vanish on essential boundaries, so only the outlet survives
    # when the recovered density is evaluated.
    ds_outlet = ds(metadata={"quadrature_degree": q_degree})
    bubble = Function(
        FunctionSpace(mesh, "B", 3, variant="integral"),
        name=f"{prefix}_cell_bubble_slab_{n}",
    ).assign(1.0)
    cone = Function(
        FunctionSpace(mesh, "FB", 2, variant="integral"),
        name=f"{prefix}_facet_cone_slab_{n}",
    ).assign(1.0)
    momentum_space = VectorFunctionSpace(
        mesh, "DG", int(recovery_space_degree), variant="integral"
    )
    scalar_space = FunctionSpace(
        mesh, "DG", int(recovery_space_degree), variant="integral"
    )
    return {
        "dx": dx_q,
        "dS": dS_q,
        "ds_outlet": ds_outlet,
        "bubble": bubble,
        "cone": cone,
        "momentum_space": momentum_space,
        "scalar_space": scalar_space,
        "facet_space": _facet_space(mesh, int(facet_recovery_degree)),
    }


def _recover_primal_slab(
    slab_number: int, primal_slab, step: float, viscosity, mesh,
    outlet_labels, essential_labels, quadrature, *,
    recovery_space_degree: int, facet_recovery_degree: int,
    recovery_time_degree: int, incoming_trace=None,
):
    """Recover the four primal space--time residual entities on one slab."""
    n = int(slab_number)
    nodes = time_nodes(int(recovery_time_degree))
    count = len(nodes)
    geo = _recovery_geometry(
        mesh, outlet_labels, n, "primal", recovery_space_degree,
        facet_recovery_degree,
    )
    dx_q, dS_q, ds_outlet = geo["dx"], geo["dS"], geo["ds_outlet"]
    bubble, cone = geo["bubble"], geo["cone"]
    momentum_space, scalar_space = geo["momentum_space"], geo["scalar_space"]
    facet_space = geo["facet_space"]
    parameters = _direct_parameters()

    # (1) Mixed block [momentum coefficients, continuity coefficients].
    volume_space = MixedFunctionSpace(
        [momentum_space] * count + [scalar_space] * count
    )
    trials, tests = TrialFunctions(volume_space), TestFunctions(volume_space)
    momentum_trials = _vector_blocks(trials[:count])
    continuity_trials = trials[count:]
    momentum_tests = _vector_blocks(tests[:count])
    continuity_tests = tests[count:]
    a_volume = 0
    L_volume = 0
    for point, weight in quadrature:
        basis = lagrange_values(nodes, point)
        temporal_bubble = 4.0 * float(point) * (1.0 - float(point))
        momentum_density = linear_combination(momentum_trials, basis)
        continuity_density = linear_combination(continuity_trials, basis)
        state = evaluate_slab(primal_slab, point, name=f"U_volume_slab_{n}")
        state_dt = evaluate_slab_dt(
            primal_slab, point, step, name=f"U_t_volume_slab_{n}"
        )
        factor = Constant(float(step) * float(weight))
        for index in range(count):
            scalar = Constant(temporal_bubble * float(basis[index]))
            local_velocity = scalar * bubble * momentum_tests[index]
            local_pressure = scalar * bubble * continuity_tests[index]
            a_volume += factor * (
                inner(momentum_density, local_velocity)
                + continuity_density * local_pressure
            ) * dx_q
            L_volume += factor * _primal_action(
                state, state_dt, local_velocity, local_pressure, viscosity,
                measure=dx_q,
            )
    volume = Function(volume_space, name=f"R_primal_volume_slab_{n}")
    solve(a_volume == L_volume, volume, solver_parameters=parameters)
    volume_coefficients = _rename(volume, f"R_primal_volume_slab_{n}")
    momentum_coefficients = volume_coefficients[:count]
    continuity_coefficients = volume_coefficients[count:]

    # (2) Broken spatial-facet velocity flux.
    spatial_space = MixedFunctionSpace([facet_space] * count)
    spatial_trials = _vector_blocks(TrialFunctions(spatial_space))
    spatial_tests = _vector_blocks(TestFunctions(spatial_space))
    a_spatial = 0
    L_spatial = 0
    for point, weight in quadrature:
        basis = lagrange_values(nodes, point)
        temporal_bubble = 4.0 * float(point) * (1.0 - float(point))
        momentum_density = linear_combination(momentum_coefficients, basis)
        facet_density = linear_combination(spatial_trials, basis) / cone
        state = evaluate_slab(primal_slab, point, name=f"U_space_slab_{n}")
        state_dt = evaluate_slab_dt(
            primal_slab, point, step, name=f"U_t_space_slab_{n}"
        )
        factor = Constant(float(step) * float(weight))
        for index, test in enumerate(spatial_tests):
            local_test = Constant(
                temporal_bubble * float(basis[index])
            ) * test
            a_spatial += factor * _facet_pair(
                facet_density, local_test, dS_q, ds_outlet
            )
            L_spatial += factor * _primal_action(
                state, state_dt, local_test, None, viscosity, measure=dx_q
            )
            L_spatial -= factor * inner(momentum_density, local_test) * dx_q
    spatial = Function(spatial_space, name=f"Rhat_primal_space_slab_{n}")
    solve(
        a_spatial == L_spatial, spatial,
        solver_parameters=parameters,
    )
    spatial_coefficients = _rename(spatial, f"Rhat_primal_space_slab_{n}")

    # (3) Physical-left temporal facet; pressure has no jump.
    temporal_trial = TrialFunction(momentum_space)
    temporal_test = TestFunction(momentum_space)
    a_temporal = inner(temporal_trial, bubble * temporal_test) * dx_q
    L_temporal = 0
    for point, weight in quadrature:
        basis = lagrange_values(nodes, point)
        momentum_density = linear_combination(momentum_coefficients, basis)
        state = evaluate_slab(primal_slab, point, name=f"U_time_slab_{n}")
        state_dt = evaluate_slab_dt(
            primal_slab, point, step, name=f"U_t_time_slab_{n}"
        )
        local_test = Constant(1.0 - float(point)) * bubble * temporal_test
        factor = Constant(float(step) * float(weight))
        L_temporal += factor * _primal_action(
            state, state_dt, local_test, None, viscosity, measure=dx_q
        )
        L_temporal -= factor * inner(momentum_density, local_test) * dx_q
    jump = _primal_jump(primal_slab, incoming_trace)
    L_temporal -= inner(jump, bubble * temporal_test) * dx_q
    temporal = Function(momentum_space, name=f"R_primal_time_slab_{n}")
    solve(a_temporal == L_temporal, temporal, solver_parameters=parameters)

    # (4) Spatial facet x physical-left interface complement.
    ridge_trial, ridge_test = TrialFunction(facet_space), TestFunction(facet_space)
    ridge_density = ridge_trial / cone
    a_ridge = _facet_pair(ridge_density, ridge_test, dS_q, ds_outlet)
    L_ridge = 0
    for point, weight in quadrature:
        basis = lagrange_values(nodes, point)
        momentum_density = linear_combination(momentum_coefficients, basis)
        facet_density = linear_combination(spatial_coefficients, basis) / cone
        state = evaluate_slab(primal_slab, point, name=f"U_ridge_slab_{n}")
        state_dt = evaluate_slab_dt(
            primal_slab, point, step, name=f"U_t_ridge_slab_{n}"
        )
        local_test = Constant(1.0 - float(point)) * ridge_test
        factor = Constant(float(step) * float(weight))
        L_ridge += factor * _primal_action(
            state, state_dt, local_test, None, viscosity, measure=dx_q
        )
        L_ridge -= factor * inner(momentum_density, local_test) * dx_q
        L_ridge -= factor * _facet_pair(
            facet_density, local_test, dS_q, ds_outlet
        )
    L_ridge -= inner(jump, ridge_test) * dx_q
    L_ridge -= inner(temporal, ridge_test) * dx_q
    ridge = Function(facet_space, name=f"Rhat_primal_ridge_slab_{n}")
    solve(
        a_ridge == L_ridge, ridge,
        solver_parameters=parameters,
    )
    return {
        "time_nodes": nodes,
        "momentum_coeffs": momentum_coefficients,
        "continuity_coeffs": continuity_coefficients,
        "spatial_coeffs": spatial_coefficients,
        "temporal": temporal,
        "ridge_hat": ridge,
        "cone": cone,
        "jump": jump,
        "dx": dx_q,
        "dS": dS_q,
        "ds_outlet": ds_outlet,
        "volume": volume,
        "spatial": spatial,
    }


def _recover_adjoint_slab(
    slab_number: int, primal_slab, dual_slab, step: float, viscosity, mesh,
    outlet_labels, essential_labels, quadrature, *,
    recovery_space_degree: int, facet_recovery_degree: int,
    recovery_time_degree: int,
):
    """Recover the reverse-adjoint operator residual on a full time slab."""
    n = int(slab_number)
    nodes = time_nodes(int(recovery_time_degree))
    count = len(nodes)
    geo = _recovery_geometry(
        mesh, outlet_labels, n, "adjoint", recovery_space_degree,
        facet_recovery_degree,
    )
    dx_q, dS_q, ds_outlet = geo["dx"], geo["dS"], geo["ds_outlet"]
    bubble, cone = geo["bubble"], geo["cone"]
    momentum_space, scalar_space = geo["momentum_space"], geo["scalar_space"]
    facet_space = geo["facet_space"]
    parameters = _direct_parameters()

    volume_space = MixedFunctionSpace(
        [momentum_space] * count + [scalar_space] * count
    )
    trials, tests = TrialFunctions(volume_space), TestFunctions(volume_space)
    momentum_trials = _vector_blocks(trials[:count])
    constraint_trials = trials[count:]
    momentum_tests = _vector_blocks(tests[:count])
    constraint_tests = tests[count:]
    a_volume = 0
    L_volume = 0
    for reverse_point, weight in quadrature:
        basis = lagrange_values(nodes, reverse_point)
        temporal_bubble = 4.0 * float(reverse_point) * (1.0 - float(reverse_point))
        momentum_density = linear_combination(momentum_trials, basis)
        constraint_density = linear_combination(constraint_trials, basis)
        physical_point = 1.0 - float(reverse_point)
        state = evaluate_slab(primal_slab, physical_point)
        dual = evaluate_slab(dual_slab, reverse_point)
        dual_tau = evaluate_slab_dt(dual_slab, reverse_point, step)
        factor = Constant(float(step) * float(weight))
        for index in range(count):
            scalar = Constant(temporal_bubble * float(basis[index]))
            local_velocity = scalar * bubble * momentum_tests[index]
            local_pressure = scalar * bubble * constraint_tests[index]
            a_volume += factor * (
                inner(momentum_density, local_velocity)
                + constraint_density * local_pressure
            ) * dx_q
            L_volume += factor * _adjoint_action(
                state.subfunctions[0], dual, dual_tau,
                local_velocity, local_pressure, viscosity, measure=dx_q,
            )
    volume = Function(volume_space, name=f"Rstar_adjoint_volume_slab_{n}")
    solve(a_volume == L_volume, volume, solver_parameters=parameters)
    volume_coefficients = _rename(volume, f"Rstar_adjoint_volume_slab_{n}")
    momentum_coefficients = volume_coefficients[:count]
    constraint_coefficients = volume_coefficients[count:]

    spatial_space = MixedFunctionSpace([facet_space] * count)
    spatial_trials = _vector_blocks(TrialFunctions(spatial_space))
    spatial_tests = _vector_blocks(TestFunctions(spatial_space))
    a_spatial = 0
    L_spatial = 0
    for reverse_point, weight in quadrature:
        basis = lagrange_values(nodes, reverse_point)
        temporal_bubble = 4.0 * float(reverse_point) * (1.0 - float(reverse_point))
        momentum_density = linear_combination(momentum_coefficients, basis)
        facet_density = linear_combination(spatial_trials, basis) / cone
        physical_point = 1.0 - float(reverse_point)
        state = evaluate_slab(primal_slab, physical_point)
        dual = evaluate_slab(dual_slab, reverse_point)
        dual_tau = evaluate_slab_dt(dual_slab, reverse_point, step)
        factor = Constant(float(step) * float(weight))
        for index, test in enumerate(spatial_tests):
            local_test = Constant(
                temporal_bubble * float(basis[index])
            ) * test
            a_spatial += factor * _facet_pair(
                facet_density, local_test, dS_q, ds_outlet
            )
            L_spatial += factor * _adjoint_action(
                state.subfunctions[0], dual, dual_tau,
                local_test, None, viscosity, measure=dx_q,
            )
            L_spatial -= factor * inner(momentum_density, local_test) * dx_q
    spatial = Function(spatial_space, name=f"Rstar_hat_adjoint_space_slab_{n}")
    solve(
        a_spatial == L_spatial, spatial,
        solver_parameters=parameters,
    )
    spatial_coefficients = _rename(
        spatial, f"Rstar_hat_adjoint_space_slab_{n}"
    )

    temporal_trial, temporal_test = (
        TrialFunction(momentum_space), TestFunction(momentum_space)
    )
    a_temporal = inner(temporal_trial, bubble * temporal_test) * dx_q
    L_temporal = 0
    for reverse_point, weight in quadrature:
        basis = lagrange_values(nodes, reverse_point)
        momentum_density = linear_combination(momentum_coefficients, basis)
        physical_point = 1.0 - float(reverse_point)
        state = evaluate_slab(primal_slab, physical_point)
        dual = evaluate_slab(dual_slab, reverse_point)
        dual_tau = evaluate_slab_dt(dual_slab, reverse_point, step)
        local_test = Constant(1.0 - float(reverse_point)) * bubble * temporal_test
        factor = Constant(float(step) * float(weight))
        L_temporal += factor * _adjoint_action(
            state.subfunctions[0], dual, dual_tau,
            local_test, None, viscosity, measure=dx_q,
        )
        L_temporal -= factor * inner(momentum_density, local_test) * dx_q
    jump = _dual_physical_right_trace(dual_slab)
    L_temporal -= inner(jump, bubble * temporal_test) * dx_q
    temporal = Function(momentum_space, name=f"Rstar_adjoint_time_slab_{n}")
    solve(a_temporal == L_temporal, temporal, solver_parameters=parameters)

    # After temporal integration by parts the second endpoint entity is
    # +(e_in, z_left).  Keeping both endpoints on this slab avoids subtracting
    # finite-element traces that live on different slab meshes.
    dual_physical_left = evaluate_slab(
        dual_slab, 1.0, name=f"Z_physical_left_slab_{n}"
    )
    propagation_trial = TrialFunction(momentum_space)
    propagation_test = TestFunction(momentum_space)
    propagation = Function(
        momentum_space, name=f"Rstar_adjoint_propagation_slab_{n}"
    )
    solve(
        inner(propagation_trial, bubble * propagation_test) * dx_q
        == inner(
            dual_physical_left.subfunctions[0], bubble * propagation_test
        ) * dx_q,
        propagation,
        solver_parameters=parameters,
    )

    ridge_trial, ridge_test = TrialFunction(facet_space), TestFunction(facet_space)
    ridge_density = ridge_trial / cone
    a_ridge = _facet_pair(ridge_density, ridge_test, dS_q, ds_outlet)
    L_ridge = 0
    for reverse_point, weight in quadrature:
        basis = lagrange_values(nodes, reverse_point)
        momentum_density = linear_combination(momentum_coefficients, basis)
        facet_density = linear_combination(spatial_coefficients, basis) / cone
        physical_point = 1.0 - float(reverse_point)
        state = evaluate_slab(primal_slab, physical_point)
        dual = evaluate_slab(dual_slab, reverse_point)
        dual_tau = evaluate_slab_dt(dual_slab, reverse_point, step)
        local_test = Constant(1.0 - float(reverse_point)) * ridge_test
        factor = Constant(float(step) * float(weight))
        L_ridge += factor * _adjoint_action(
            state.subfunctions[0], dual, dual_tau,
            local_test, None, viscosity, measure=dx_q,
        )
        L_ridge -= factor * inner(momentum_density, local_test) * dx_q
        L_ridge -= factor * _facet_pair(
            facet_density, local_test, dS_q, ds_outlet
        )
    L_ridge -= inner(jump, ridge_test) * dx_q
    L_ridge -= inner(temporal, ridge_test) * dx_q
    ridge = Function(facet_space, name=f"Rstar_hat_adjoint_ridge_slab_{n}")
    solve(
        a_ridge == L_ridge, ridge,
        solver_parameters=parameters,
    )
    return {
        "time_nodes": nodes,
        "momentum_coeffs": momentum_coefficients,
        "constraint_coeffs": constraint_coefficients,
        "spatial_coeffs": spatial_coefficients,
        "temporal": temporal,
        "propagation": propagation,
        "ridge_hat": ridge,
        "cone": cone,
        "jump": jump,
        "dx": dx_q,
        "dS": dS_q,
        "ds_outlet": ds_outlet,
        "volume": volume,
        "spatial": spatial,
    }


def _volume_cell_form(recovered, weight, point: float, cell_test):
    basis = lagrange_values(recovered["time_nodes"], point)
    momentum = linear_combination(recovered["momentum_coeffs"], basis)
    scalar_coefficients = recovered.get("continuity_coeffs")
    if scalar_coefficients is None:
        scalar_coefficients = recovered["constraint_coeffs"]
    scalar = linear_combination(scalar_coefficients, basis)
    velocity_weight, pressure_weight = split(weight)
    return (
        inner(momentum, velocity_weight) + scalar * pressure_weight
    ) * cell_test * recovered["dx"]


def _spatial_cell_form(recovered, weight, point: float, cell_test):
    basis = lagrange_values(recovered["time_nodes"], point)
    density = linear_combination(recovered["spatial_coeffs"], basis) / recovered[
        "cone"
    ]
    velocity_weight = split(weight)[0]
    return (
        inner(density, velocity_weight) * cell_test * recovered["ds_outlet"]
        + (
            inner(density("+"), velocity_weight("+")) * cell_test("+")
            + inner(density("-"), velocity_weight("-")) * cell_test("-")
        ) * recovered["dS"]
    )


def _temporal_cell_form(recovered, weight, cell_test):
    return (
        inner(recovered["temporal"], split(weight)[0])
        * cell_test * recovered["dx"]
    )


def _propagation_cell_form(recovered, trace_defect, cell_test):
    return (
        inner(recovered["propagation"], split(trace_defect)[0])
        * cell_test * recovered["dx"]
    )


def _ridge_cell_form(recovered, weight, cell_test):
    density = recovered["ridge_hat"] / recovered["cone"]
    velocity_weight = split(weight)[0]
    return (
        inner(density, velocity_weight) * cell_test * recovered["ds_outlet"]
        + (
            inner(density("+"), velocity_weight("+")) * cell_test("+")
            + inner(density("-"), velocity_weight("-")) * cell_test("-")
        ) * recovered["dS"]
    )


def _cell_values(form) -> np.ndarray:
    return np.asarray(assemble(form).dat.data_ro, dtype=float).copy()


def _linear_strong_volume_cell_form(
    state, state_dt, weight, viscosity, cell_test, measure,
):
    r"""Cell-interior part of ``rho(U_h)(weight)`` after spatial IBP."""
    velocity, pressure = split(state)
    velocity_dt = split(state_dt)[0]
    velocity_weight, pressure_weight = split(weight)
    momentum = (
        -velocity_dt
        - grad(velocity) * velocity
        + viscosity * div(grad(velocity))
        - grad(pressure)
    )
    return cell_test * (
        inner(momentum, velocity_weight)
        + div(velocity) * pressure_weight
    ) * measure


def _linear_strong_spatial_cell_form(
    state, weight, viscosity, cell_test, interior_measure, exterior_measure,
):
    r"""Spatial stress-flux residual, split equally across interior facets."""
    velocity, pressure = split(state)
    velocity_weight = split(weight)[0]
    mesh = state.function_space().mesh()
    normal = FacetNormal(mesh)
    # This is the boundary flux of rho=-A after elementwise integration by
    # parts: (p I - nu grad(u)) n.  ``avg(cell_test)`` assigns one half of an
    # interior-facet contribution to each adjacent cell without introducing
    # a separate space-time ridge entity.
    residual_flux = pressure * Identity(int(mesh.geometric_dimension)) - (
        viscosity * grad(velocity)
    )
    return (
        inner(jump(residual_flux, normal), avg(velocity_weight))
        * avg(cell_test)
        * interior_measure
        + inner(dot(residual_flux, normal), velocity_weight)
        * cell_test
        * exterior_measure
    )


def _adjoint_strong_volume_cell_form(
    state, dual, dual_tau, primal_error, viscosity, cell_test, measure,
):
    r"""Cell-interior reverse-adjoint residual weighted by ``e_U``.

    This is ``rho*(U_h, Z_h)(e_U)`` after integration by parts in physical
    time and elementwise integration by parts in space.  ``dual_tau`` is the
    derivative in reverse time, so the volume time term is ``-Z_tau``.
    """
    velocity = split(state)[0]
    dual_velocity, dual_pressure = split(dual)
    dual_velocity_tau = split(dual_tau)[0]
    error_velocity, error_pressure = split(primal_error)
    momentum = (
        -dual_velocity_tau
        + grad(dual_velocity) * velocity
        + div(velocity) * dual_velocity
        - grad(velocity).T * dual_velocity
        + viscosity * div(grad(dual_velocity))
        - grad(dual_pressure)
    )
    return cell_test * (
        inner(momentum, error_velocity)
        + div(dual_velocity) * error_pressure
    ) * measure


def _adjoint_strong_spatial_cell_form(
    state,
    dual,
    primal_error,
    viscosity,
    cylinder_labels,
    horizon,
    cell_test,
    interior_measure,
    exterior_measure,
):
    r"""Spatial-flux and drag-boundary pieces of the adjoint residual."""
    velocity = split(state)[0]
    dual_velocity, dual_pressure = split(dual)
    error_velocity, error_pressure = split(primal_error)
    mesh = state.function_space().mesh()
    normal = FacetNormal(mesh)
    # Boundary term generated by moving spatial derivatives from the primal
    # error onto the low adjoint.  The goal derivative remains an exact
    # cylinder-boundary contribution and is assigned to the adjacent cell.
    residual_flux = (
        dual_pressure * Identity(int(mesh.geometric_dimension))
        - viscosity * grad(dual_velocity)
        - outer(dual_velocity, velocity)
    )
    form = (
        inner(jump(residual_flux, normal), avg(error_velocity))
        * avg(cell_test)
        * interior_measure
        + inner(dot(residual_flux, normal), error_velocity)
        * cell_test
        * exterior_measure
    )
    form += Constant(1.0 / float(horizon)) * drag_derivative_form(
        error_velocity,
        error_pressure,
        viscosity,
        cylinder_labels,
        weight=cell_test,
    )
    return form


def localise_linear_dwr(
    primal, dual_enriched, dual_low,
    estimate: LinearDWRGlobalEstimate, *, quadrature_points: int = 4,
    dual_weight_mode: str = "enriched_minus_interpolant",
    primal_recovery_degree: int = 2, facet_recovery_degree: int = 2,
    recovery_time_degree: int = 2,
) -> BubbleConeLocalisation:
    r"""Three-part strong-residual localisation of the strict linear DWR.

    The retained entities are cell volume, spatial stress facet, and dG-time
    jump.  There is deliberately no tensor-product space-time ridge entity
    and no auxiliary recovery solve.  Elementwise spatial integration by
    parts makes this an algebraic decomposition of the same weak global
    primal residual rather than a truncated bubble/cone recovery.
    """
    if dual_weight_mode != estimate.dual_weight_mode:
        raise ValueError(
            "Global estimator and localisation need the same dual weight mode."
        )
    if int(recovery_time_degree) < 0:
        raise ValueError("recovery_time_degree must be nonnegative.")
    times = primal["times"]
    quadrature = gauss_rule(int(quadrature_points))
    shape = len(times)
    primal_cells: list[np.ndarray | None] = [None] * shape
    zero_cells: list[np.ndarray | None] = [None] * shape
    primal_ridge_cells: list[np.ndarray | None] = [None] * shape
    zero_ridge_cells: list[np.ndarray | None] = [None] * shape
    primal_volume_cells: list[np.ndarray | None] = [None] * shape
    primal_spatial_cells: list[np.ndarray | None] = [None] * shape
    primal_temporal_cells: list[np.ndarray | None] = [None] * shape
    zero_volume_cells: list[np.ndarray | None] = [None] * shape
    zero_spatial_cells: list[np.ndarray | None] = [None] * shape
    zero_temporal_cells: list[np.ndarray | None] = [None] * shape
    fields: list[Function | None] = [None] * shape
    entities: list[dict[str, Any] | None] = [None] * shape
    primal_slab_gaps = [0.0] * shape
    zero_slab_gaps = [0.0] * shape
    essential = tuple(primal["labels"]["inlet"]) + tuple(
        primal["labels"]["wall"]
    )

    for n in range(1, shape):
        step = float(times[n] - times[n - 1])
        low_primal_slab = primal["slabs"][n]
        rich_dual_slab = dual_enriched["slabs"][n]
        low_dual_slab = dual_low["slabs"][n]
        mesh = low_primal_slab.get("mesh", primal.get("mesh"))
        if mesh is None:
            raise ValueError(f"Primal slab {n} does not identify its mesh.")
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_test = TestFunction(DG0)
        zeros = np.zeros(DG0.node_count, dtype=float)
        primal_volume = zeros.copy()
        primal_space = zeros.copy()
        q_degree = max(12, 2 * int(primal_recovery_degree) + 8)
        dx_q = dx(metadata={"quadrature_degree": q_degree})
        dS_q = dS(metadata={"quadrature_degree": q_degree})
        ds_q = ds(metadata={"quadrature_degree": q_degree})
        entities[n] = {
            "primal": {
                "strategy": "three_part_strong_residual",
                "entities": ("volume", "spatial_facet", "temporal_jump"),
            }
        }

        temporal_factor = int(
            rich_dual_slab.get("temporal_refinement_factor", 1)
        )
        for child in range(temporal_factor):
            for child_point, child_weight in quadrature:
                point = (float(child) + float(child_point)) / temporal_factor
                factor = Constant(
                    float(step) * float(child_weight) / temporal_factor
                )
                state = evaluate_slab(
                    low_primal_slab, point, name=f"U_strong_slab_{n}"
                )
                state_dt = evaluate_slab_dt(
                    low_primal_slab, point, step,
                    name=f"U_t_strong_slab_{n}",
                )
                goal_weight = dual_error_at(
                    rich_dual_slab, low_dual_slab, point,
                    mode=dual_weight_mode, essential_labels=essential,
                    stored_interpolant=(
                        dual_weight_mode == "enriched_minus_interpolant"
                    ),
                )
                primal_volume += _cell_values(
                    factor * _linear_strong_volume_cell_form(
                        state, state_dt, goal_weight, primal["viscosity"],
                        cell_test, dx_q,
                    )
                )
                primal_space += _cell_values(
                    factor * _linear_strong_spatial_cell_form(
                        state, goal_weight, primal["viscosity"], cell_test,
                        dS_q, ds_q,
                    )
                )

        goal_weight_left = dual_error_at(
            rich_dual_slab, low_dual_slab, 0.0,
            mode=dual_weight_mode, essential_labels=essential,
                stored_interpolant=(
                    dual_weight_mode == "enriched_minus_interpolant"
                ),
            )
        state_left = evaluate_slab(
            low_primal_slab, 0.0, name=f"U_strong_left_slab_{n}"
        )
        incoming_trace = _low_incoming_in_rich_operator(primal, primal, n)
        jump_velocity = (
            state_left.subfunctions[0] - incoming_trace.subfunctions[0]
        )
        primal_temporal = _cell_values(
            -cell_test
            * inner(jump_velocity, split(goal_weight_left)[0])
            * dx_q
        )
        primal_ridge = zeros.copy()
        primal_values = primal_volume + primal_space + primal_temporal
        primal_cells[n] = primal_values
        primal_ridge_cells[n] = primal_ridge
        primal_volume_cells[n] = primal_volume
        primal_spatial_cells[n] = primal_space
        primal_temporal_cells[n] = primal_temporal
        zero_cells[n] = zeros.copy()
        zero_ridge_cells[n] = zeros.copy()
        zero_volume_cells[n] = zeros.copy()
        zero_spatial_cells[n] = zeros.copy()
        zero_temporal_cells[n] = zeros.copy()
        field = Function(DG0, name=f"eta_linear_primal_abs_slab_{n}")
        field.dat.data[:] = np.abs(primal_values)
        fields[n] = field
        exact_slab = (
            estimate.eta_volume_by_slab[n]
            + estimate.eta_temporal_jump_by_slab[n]
        )
        primal_slab_gaps[n] = float(primal_values.sum()) - exact_slab

    primal_sum = float(sum(values.sum() for values in primal_cells[1:]))
    marking_sum = float(sum(np.abs(values).sum() for values in primal_cells[1:]))
    return BubbleConeLocalisation(
        eta_cell_signed=primal_cells,
        eta_primal_bubble_cell=primal_cells,
        eta_adjoint_bubble_cell=zero_cells,
        eta_correction_bubble_cell=zero_cells,
        eta_cubic_remainder_cell=zero_cells,
        eta_primal_mixed_ridge_cell=primal_ridge_cells,
        eta_adjoint_mixed_ridge_cell=zero_ridge_cells,
        eta_correction_mixed_ridge_cell=zero_ridge_cells,
        eta_primal_volume_cell=primal_volume_cells,
        eta_primal_spatial_cell=primal_spatial_cells,
        eta_primal_temporal_cell=primal_temporal_cells,
        eta_adjoint_volume_cell=zero_volume_cells,
        eta_adjoint_spatial_cell=zero_spatial_cells,
        eta_adjoint_temporal_cell=zero_temporal_cells,
        eta_correction_volume_cell=zero_volume_cells,
        eta_correction_spatial_cell=zero_spatial_cells,
        eta_correction_temporal_cell=zero_temporal_cells,
        eta_local_sum=primal_sum,
        eta_marking_sum=marking_sum,
        localisation_gap=primal_sum - estimate.eta_global,
        eta_primal_recovered=primal_sum,
        eta_adjoint_recovered=0.0,
        eta_correction_recovered=0.0,
        primal_recovery_gap=primal_sum - estimate.eta_global,
        adjoint_recovery_gap=0.0,
        correction_recovery_gap=0.0,
        eta_adjoint_reverse_weak=0.0,
        adjoint_reverse_identity_gap=0.0,
        dual_weight_mode=str(dual_weight_mode),
        fields=fields,
        primal_slab_recovery_gap=primal_slab_gaps,
        adjoint_slab_recovery_gap=zero_slab_gaps,
        correction_slab_recovery_gap=zero_slab_gaps,
        recovered_entities=entities,
        recovery_time_degree=int(recovery_time_degree),
    )


def uniform_global_only_localisation(
    primal, estimate: LinearDWRGlobalEstimate,
) -> BubbleConeLocalisation:
    r"""Zero-cost localisation placeholder for predetermined uniform grids.

    Uniform refinement does not consume local indicators.  The global DWR
    estimate remains available for effectivity, while every local array is
    explicitly zero and labelled as skipped.  Setting the scalar closure to
    the already assembled global estimate avoids turning a deliberately
    omitted localisation into a false recovery-gate failure.
    """
    shape = len(primal["times"])
    zero_cells: list[np.ndarray | None] = [None] * shape
    fields: list[Function | None] = [None] * shape
    entities: list[dict[str, Any] | None] = [None] * shape
    for n in range(1, shape):
        slab = primal["slabs"][n]
        mesh = slab.get("mesh", primal.get("mesh"))
        DG0 = FunctionSpace(mesh, "DG", 0)
        zero_cells[n] = np.zeros(DG0.node_count, dtype=float)
        entities[n] = {
            "strategy": "uniform_global_only",
            "entities": (),
            "localisation_skipped": True,
        }
    zero_slab_gaps = [0.0] * shape
    return BubbleConeLocalisation(
        eta_cell_signed=zero_cells,
        eta_primal_bubble_cell=zero_cells,
        eta_adjoint_bubble_cell=zero_cells,
        eta_correction_bubble_cell=zero_cells,
        eta_cubic_remainder_cell=zero_cells,
        eta_primal_mixed_ridge_cell=zero_cells,
        eta_adjoint_mixed_ridge_cell=zero_cells,
        eta_correction_mixed_ridge_cell=zero_cells,
        eta_primal_volume_cell=zero_cells,
        eta_primal_spatial_cell=zero_cells,
        eta_primal_temporal_cell=zero_cells,
        eta_adjoint_volume_cell=zero_cells,
        eta_adjoint_spatial_cell=zero_cells,
        eta_adjoint_temporal_cell=zero_cells,
        eta_correction_volume_cell=zero_cells,
        eta_correction_spatial_cell=zero_cells,
        eta_correction_temporal_cell=zero_cells,
        eta_local_sum=float(estimate.eta_global),
        eta_marking_sum=0.0,
        localisation_gap=0.0,
        eta_primal_recovered=float(estimate.eta_global),
        eta_adjoint_recovered=0.0,
        eta_correction_recovered=0.0,
        primal_recovery_gap=0.0,
        adjoint_recovery_gap=0.0,
        correction_recovery_gap=0.0,
        eta_adjoint_reverse_weak=0.0,
        adjoint_reverse_identity_gap=0.0,
        dual_weight_mode=str(estimate.dual_weight_mode),
        fields=fields,
        primal_slab_recovery_gap=zero_slab_gaps,
        adjoint_slab_recovery_gap=zero_slab_gaps,
        correction_slab_recovery_gap=zero_slab_gaps,
        recovered_entities=entities,
        recovery_time_degree=-1,
    )


def localise_symmetric_two_term_dwr(
    primal,
    primal_enriched,
    dual_enriched,
    dual_low,
    estimate: SymmetricDWREstimate,
    *,
    quadrature_points: int = 4,
    dual_weight_mode: str = "enriched_minus_numerical",
    strong_quadrature_degree: int = 12,
) -> BubbleConeLocalisation:
    r"""Exact three-entity localisation of the nonlinear two-term estimator.

    The marked field is

    ``0.5*rho(U_h)(Z+ - Z_h) + 0.5*rho*(U_h,Z_h)(U+ - U_h)``.

    Both residuals are decomposed algebraically into cell volume, spatial
    facet, and dG temporal-jump contributions.  When requested by the global
    estimator, the independently assembled cubic Navier--Stokes remainder is
    added as a volume-cell contribution.  No Galerkin correction,
    tensor-product ridge, recovery projection, or interpolated dual base is
    used by this path.
    """
    if dual_weight_mode != "enriched_minus_numerical":
        raise ValueError(
            "Nonlinear two-term three-part localisation requires independently "
            "solved low and enriched adjoints."
        )
    if dual_weight_mode != estimate.dual_weight_mode:
        raise ValueError(
            "Global estimator and localisation need the same dual weight mode."
        )
    if abs(float(estimate.eta_galerkin_correction)) > 1.0e-14:
        raise ValueError("The nonlinear two-term estimate must exclude correction.")
    times = primal["times"]
    horizon = float(times[-1] - times[0])
    quadrature = gauss_rule(int(quadrature_points))
    shape = len(times)
    primal_cells: list[np.ndarray | None] = [None] * shape
    adjoint_cells: list[np.ndarray | None] = [None] * shape
    zero_cells: list[np.ndarray | None] = [None] * shape
    primal_volume_cells: list[np.ndarray | None] = [None] * shape
    primal_spatial_cells: list[np.ndarray | None] = [None] * shape
    primal_temporal_cells: list[np.ndarray | None] = [None] * shape
    adjoint_volume_cells: list[np.ndarray | None] = [None] * shape
    adjoint_spatial_cells: list[np.ndarray | None] = [None] * shape
    adjoint_temporal_cells: list[np.ndarray | None] = [None] * shape
    zero_component_cells: list[np.ndarray | None] = [None] * shape
    zero_ridge_cells: list[np.ndarray | None] = [None] * shape
    signed: list[np.ndarray | None] = [None] * shape
    fields: list[Function | None] = [None] * shape
    entities: list[dict[str, Any] | None] = [None] * shape
    primal_slab_gaps = [0.0] * shape
    adjoint_slab_gaps = [0.0] * shape
    zero_slab_gaps = [0.0] * shape
    essential = tuple(primal["labels"]["inlet"]) + tuple(
        primal["labels"]["wall"]
    )

    for n in range(1, shape):
        step = float(times[n] - times[n - 1])
        low_primal_slab = primal["slabs"][n]
        rich_primal_slab = primal_enriched["slabs"][n]
        rich_dual_slab = dual_enriched["slabs"][n]
        low_dual_slab = dual_low["slabs"][n]
        mesh = low_primal_slab.get("mesh", primal.get("mesh"))
        if mesh is None:
            raise ValueError(f"Primal slab {n} does not identify its mesh.")
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_test = TestFunction(DG0)
        zeros = np.zeros(DG0.node_count, dtype=float)
        primal_volume = zeros.copy()
        primal_space = zeros.copy()
        adjoint_volume = zeros.copy()
        adjoint_space = zeros.copy()
        q_degree = max(12, int(strong_quadrature_degree))
        dx_q = dx(metadata={"quadrature_degree": q_degree})
        dS_q = dS(metadata={"quadrature_degree": q_degree})
        ds_q = ds(metadata={"quadrature_degree": q_degree})
        entities[n] = {
            "primal": {
                "strategy": "three_part_strong_residual",
                "entities": ("volume", "spatial_facet", "temporal_jump"),
            },
            "adjoint": {
                "strategy": "three_part_strong_residual",
                "entities": ("volume", "spatial_facet", "temporal_jump"),
            },
            "cubic_remainder": {
                "strategy": "weak_volume_cell",
                "entities": ("volume",),
            },
        }

        for point, weight in quadrature:
            factor = Constant(float(step) * float(weight))
            state = evaluate_slab(
                low_primal_slab, point, name=f"U_two_term_slab_{n}"
            )
            state_dt = evaluate_slab_dt(
                low_primal_slab,
                point,
                step,
                name=f"U_t_two_term_slab_{n}",
            )
            goal_weight = dual_error_at(
                rich_dual_slab,
                low_dual_slab,
                point,
                mode=dual_weight_mode,
                essential_labels=essential,
            )
            primal_volume += _cell_values(
                factor
                * _linear_strong_volume_cell_form(
                    state,
                    state_dt,
                    goal_weight,
                    primal["viscosity"],
                    cell_test,
                    dx_q,
                )
            )
            primal_space += _cell_values(
                factor
                * _linear_strong_spatial_cell_form(
                    state,
                    goal_weight,
                    primal["viscosity"],
                    cell_test,
                    dS_q,
                    ds_q,
                )
            )

            reverse_point = 1.0 - float(point)
            low_dual_value = evaluate_slab(
                low_dual_slab,
                reverse_point,
                name=f"Z_low_two_term_slab_{n}",
            )
            low_dual_tau = evaluate_slab_dt(
                low_dual_slab,
                reverse_point,
                step,
                name=f"Z_tau_two_term_slab_{n}",
            )
            primal_error, _ = primal_error_at(
                rich_primal_slab, low_primal_slab, point, step
            )
            adjoint_volume += _cell_values(
                factor
                * _adjoint_strong_volume_cell_form(
                    state,
                    low_dual_value,
                    low_dual_tau,
                    primal_error,
                    primal["viscosity"],
                    cell_test,
                    dx_q,
                )
            )
            adjoint_space += _cell_values(
                factor
                * _adjoint_strong_spatial_cell_form(
                    state,
                    low_dual_value,
                    primal_error,
                    primal["viscosity"],
                    primal["labels"]["cylinder"],
                    horizon,
                    cell_test,
                    dS_q,
                    ds_q,
                )
            )

        goal_weight_left = dual_error_at(
            rich_dual_slab,
            low_dual_slab,
            0.0,
            mode=dual_weight_mode,
            essential_labels=essential,
        )
        state_left = evaluate_slab(
            low_primal_slab, 0.0, name=f"U_two_term_left_slab_{n}"
        )
        low_incoming_rich_operator = _low_incoming_in_rich_operator(
            primal, primal_enriched, n
        )
        primal_jump_velocity = (
            state_left.subfunctions[0]
            - low_incoming_rich_operator.subfunctions[0]
        )
        primal_temporal = _cell_values(
            -cell_test
            * inner(primal_jump_velocity, split(goal_weight_left)[0])
            * dx_q
        )

        primal_error_right, _ = primal_error_at(
            rich_primal_slab, low_primal_slab, 1.0, step
        )
        incoming_error = primal_trace_error(
            rich_primal_slab["left_trace"], low_incoming_rich_operator
        )
        dual_physical_right = evaluate_slab(
            low_dual_slab, 0.0, name=f"Z_low_right_slab_{n}"
        )
        dual_physical_left = evaluate_slab(
            low_dual_slab, 1.0, name=f"Z_low_left_slab_{n}"
        )
        adjoint_temporal = _cell_values(
            cell_test
            * (
                -inner(
                    split(primal_error_right)[0],
                    split(dual_physical_right)[0],
                )
                + inner(
                    split(incoming_error)[0],
                    split(dual_physical_left)[0],
                )
            )
            * dx_q
        )

        primal_values = primal_volume + primal_space + primal_temporal
        adjoint_values = adjoint_volume + adjoint_space + adjoint_temporal
        remainder_values = np.asarray(
            estimate.eta_cubic_remainder_cell_weak[n], dtype=float
        )
        combined = (
            0.5 * primal_values
            + 0.5 * adjoint_values
            + remainder_values
        )
        primal_cells[n] = primal_values
        adjoint_cells[n] = adjoint_values
        signed[n] = combined
        primal_volume_cells[n] = primal_volume
        primal_spatial_cells[n] = primal_space
        primal_temporal_cells[n] = primal_temporal
        adjoint_volume_cells[n] = adjoint_volume
        adjoint_spatial_cells[n] = adjoint_space
        adjoint_temporal_cells[n] = adjoint_temporal
        zero_cells[n] = zeros.copy()
        zero_component_cells[n] = zeros.copy()
        zero_ridge_cells[n] = zeros.copy()
        field = Function(DG0, name=f"eta_nonlinear_two_term_abs_slab_{n}")
        field.dat.data[:] = np.abs(combined)
        fields[n] = field
        primal_exact_slab = float(estimate.eta_primal_cell_weak[n].sum())
        adjoint_exact_slab = float(estimate.eta_adjoint_cell_weak[n].sum())
        primal_slab_gaps[n] = float(primal_values.sum()) - primal_exact_slab
        adjoint_slab_gaps[n] = float(adjoint_values.sum()) - adjoint_exact_slab

    primal_sum = float(sum(values.sum() for values in primal_cells[1:]))
    adjoint_sum = float(sum(values.sum() for values in adjoint_cells[1:]))
    local_sum = float(sum(values.sum() for values in signed[1:]))
    marking_sum = float(sum(np.abs(values).sum() for values in signed[1:]))
    return BubbleConeLocalisation(
        eta_cell_signed=signed,
        eta_primal_bubble_cell=primal_cells,
        eta_adjoint_bubble_cell=adjoint_cells,
        eta_correction_bubble_cell=zero_cells,
        eta_cubic_remainder_cell=estimate.eta_cubic_remainder_cell_weak,
        eta_primal_mixed_ridge_cell=zero_ridge_cells,
        eta_adjoint_mixed_ridge_cell=zero_ridge_cells,
        eta_correction_mixed_ridge_cell=zero_ridge_cells,
        eta_primal_volume_cell=primal_volume_cells,
        eta_primal_spatial_cell=primal_spatial_cells,
        eta_primal_temporal_cell=primal_temporal_cells,
        eta_adjoint_volume_cell=adjoint_volume_cells,
        eta_adjoint_spatial_cell=adjoint_spatial_cells,
        eta_adjoint_temporal_cell=adjoint_temporal_cells,
        eta_correction_volume_cell=zero_component_cells,
        eta_correction_spatial_cell=zero_component_cells,
        eta_correction_temporal_cell=zero_component_cells,
        eta_local_sum=local_sum,
        eta_marking_sum=marking_sum,
        localisation_gap=local_sum - estimate.eta_global,
        eta_primal_recovered=primal_sum,
        eta_adjoint_recovered=adjoint_sum,
        eta_correction_recovered=0.0,
        primal_recovery_gap=primal_sum - estimate.eta_primal_residual,
        adjoint_recovery_gap=adjoint_sum - estimate.eta_adjoint_residual,
        correction_recovery_gap=0.0,
        eta_adjoint_reverse_weak=adjoint_sum,
        adjoint_reverse_identity_gap=(
            adjoint_sum - estimate.eta_adjoint_residual
        ),
        dual_weight_mode=str(dual_weight_mode),
        fields=fields,
        primal_slab_recovery_gap=primal_slab_gaps,
        adjoint_slab_recovery_gap=adjoint_slab_gaps,
        correction_slab_recovery_gap=zero_slab_gaps,
        recovered_entities=entities,
        recovery_time_degree=-1,
    )


def localise_symmetric_dwr(
    primal, primal_enriched, dual_enriched, dual_low,
    estimate: SymmetricDWREstimate, *, quadrature_points: int = 4,
    dual_weight_mode: str = "enriched_minus_interpolant",
    primal_recovery_degree: int = 2, adjoint_recovery_degree: int = 2,
    facet_recovery_degree: int = 2, recovery_time_degree: int = 2,
) -> BubbleConeLocalisation:
    r"""Recover all three nonlinear DWR terms with slab tensor products."""
    if dual_weight_mode != estimate.dual_weight_mode:
        raise ValueError(
            "Global estimator and localisation need the same dual weight mode."
        )
    if int(recovery_time_degree) < 0:
        raise ValueError("recovery_time_degree must be nonnegative.")
    times = primal["times"]
    horizon = float(times[-1] - times[0])
    quadrature = gauss_rule(int(quadrature_points))
    nslabs = len(times) - 1
    shape = nslabs + 1
    primal_cells: list[np.ndarray | None] = [None] * shape
    adjoint_cells: list[np.ndarray | None] = [None] * shape
    correction_cells: list[np.ndarray | None] = [None] * shape
    primal_ridge_cells: list[np.ndarray | None] = [None] * shape
    adjoint_ridge_cells: list[np.ndarray | None] = [None] * shape
    correction_ridge_cells: list[np.ndarray | None] = [None] * shape
    primal_volume_cells: list[np.ndarray | None] = [None] * shape
    primal_spatial_cells: list[np.ndarray | None] = [None] * shape
    primal_temporal_cells: list[np.ndarray | None] = [None] * shape
    adjoint_volume_cells: list[np.ndarray | None] = [None] * shape
    adjoint_spatial_cells: list[np.ndarray | None] = [None] * shape
    adjoint_temporal_cells: list[np.ndarray | None] = [None] * shape
    correction_volume_cells: list[np.ndarray | None] = [None] * shape
    correction_spatial_cells: list[np.ndarray | None] = [None] * shape
    correction_temporal_cells: list[np.ndarray | None] = [None] * shape
    signed: list[np.ndarray | None] = [None] * shape
    fields: list[Function | None] = [None] * shape
    entities: list[dict[str, Any] | None] = [None] * shape
    primal_slab_gaps = [0.0] * shape
    adjoint_slab_gaps = [0.0] * shape
    correction_slab_gaps = [0.0] * shape
    essential = tuple(primal["labels"]["inlet"]) + tuple(
        primal["labels"]["wall"]
    )
    adjoint_reverse_weak = 0.0

    for n in range(1, shape):
        step = float(times[n] - times[n - 1])
        low_primal_slab = primal["slabs"][n]
        rich_primal_slab = primal_enriched["slabs"][n]
        rich_dual_slab = dual_enriched["slabs"][n]
        low_dual_slab = dual_low["slabs"][n]
        mesh = low_primal_slab.get("mesh", primal.get("mesh"))
        if mesh is None:
            raise ValueError(f"Primal slab {n} does not identify its mesh.")
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_test = TestFunction(DG0)
        zeros = np.zeros(DG0.node_count, dtype=float)
        primal_volume, primal_space = zeros.copy(), zeros.copy()
        correction_volume, correction_space = zeros.copy(), zeros.copy()
        adjoint_volume, adjoint_space = zeros.copy(), zeros.copy()
        direct_reverse_slab = 0.0

        recovered_primal = _recover_primal_slab(
            n, low_primal_slab, step, primal["viscosity"], mesh,
            primal["labels"]["outlet"], essential, quadrature,
            recovery_space_degree=int(primal_recovery_degree),
            facet_recovery_degree=int(facet_recovery_degree),
            recovery_time_degree=int(recovery_time_degree),
            incoming_trace=_low_incoming_in_rich_operator(
                primal, primal_enriched, n
            ),
        )
        recovered_adjoint = _recover_adjoint_slab(
            n, low_primal_slab, low_dual_slab, step, primal["viscosity"],
            mesh, primal["labels"]["outlet"], essential, quadrature,
            recovery_space_degree=int(adjoint_recovery_degree),
            facet_recovery_degree=int(facet_recovery_degree),
            recovery_time_degree=int(recovery_time_degree),
        )
        entities[n] = {"primal": recovered_primal, "adjoint": recovered_adjoint}

        for point, weight in quadrature:
            factor = Constant(float(step) * float(weight))
            goal_weight = dual_error_at(
                rich_dual_slab, low_dual_slab, point,
                mode=dual_weight_mode, essential_labels=essential,
            )
            low_dual_value = evaluate_slab(
                low_dual_slab, 1.0 - float(point), name="Z_low_recovery"
            )
            primal_volume += _cell_values(
                factor * _volume_cell_form(
                    recovered_primal, goal_weight, point, cell_test
                )
            )
            primal_space += _cell_values(
                factor * _spatial_cell_form(
                    recovered_primal, goal_weight, point, cell_test
                )
            )
            # Dissertation (2.57)--(2.62): the correction enters as
            # ``-eta_c``.  Store the signed DWR contribution here.
            correction_volume -= _cell_values(
                factor * _volume_cell_form(
                    recovered_primal, low_dual_value, point, cell_test
                )
            )
            correction_space -= _cell_values(
                factor * _spatial_cell_form(
                    recovered_primal, low_dual_value, point, cell_test
                )
            )

        goal_weight_left = dual_error_at(
            rich_dual_slab, low_dual_slab, 0.0,
            mode=dual_weight_mode, essential_labels=essential,
        )
        low_dual_left = evaluate_slab(
            low_dual_slab, 1.0, name="Z_low_primal_time_weight"
        )
        primal_temporal = _cell_values(
            _temporal_cell_form(recovered_primal, goal_weight_left, cell_test)
        )
        correction_temporal = -_cell_values(
            _temporal_cell_form(recovered_primal, low_dual_left, cell_test)
        )
        primal_ridge = _cell_values(
            _ridge_cell_form(recovered_primal, goal_weight_left, cell_test)
        )
        correction_ridge = -_cell_values(
            _ridge_cell_form(recovered_primal, low_dual_left, cell_test)
        )

        # Reverse-time adjoint action against the physical-time primal error.
        for reverse_point, weight in quadrature:
            physical_point = 1.0 - float(reverse_point)
            factor_value = float(step) * float(weight)
            factor = Constant(factor_value)
            primal_error, _ = primal_error_at(
                rich_primal_slab, low_primal_slab, physical_point, step
            )
            adjoint_volume += _cell_values(
                factor * _volume_cell_form(
                    recovered_adjoint, primal_error, reverse_point, cell_test
                )
            )
            adjoint_space += _cell_values(
                factor * _spatial_cell_form(
                    recovered_adjoint, primal_error, reverse_point, cell_test
                )
            )
            error_velocity, error_pressure = split(primal_error)
            adjoint_volume += _cell_values(
                Constant(factor_value / horizon) * drag_derivative_form(
                    error_velocity, error_pressure, primal["viscosity"],
                    primal["labels"]["cylinder"], weight=cell_test,
                )
            )
            state = evaluate_slab(low_primal_slab, physical_point)
            dual = evaluate_slab(low_dual_slab, reverse_point)
            dual_tau = evaluate_slab_dt(low_dual_slab, reverse_point, step)
            direct_form = factor * _adjoint_action(
                state.subfunctions[0], dual, dual_tau,
                error_velocity, error_pressure, primal["viscosity"],
                measure=recovered_adjoint["dx"],
            )
            direct_form += Constant(factor_value / horizon) * drag_derivative_form(
                error_velocity, error_pressure, primal["viscosity"],
                primal["labels"]["cylinder"],
            )
            direct_reverse_slab += float(assemble(direct_form))

        primal_error_right, _ = primal_error_at(
            rich_primal_slab, low_primal_slab, 1.0, step
        )
        incoming_error = primal_trace_error(
            rich_primal_slab["left_trace"],
            _low_incoming_in_rich_operator(primal, primal_enriched, n),
        )
        adjoint_temporal = _cell_values(
            _temporal_cell_form(
                recovered_adjoint, primal_error_right, cell_test
            )
        )
        adjoint_temporal += _cell_values(
            _propagation_cell_form(
                recovered_adjoint, incoming_error, cell_test
            )
        )
        adjoint_ridge = _cell_values(
            _ridge_cell_form(recovered_adjoint, primal_error_right, cell_test)
        )
        direct_reverse_slab += float(assemble(
            -inner(
                recovered_adjoint["jump"], split(primal_error_right)[0]
            )
            * recovered_adjoint["dx"]
        ))
        direct_reverse_slab += float(assemble(
            inner(
                split(evaluate_slab(low_dual_slab, 1.0))[0],
                split(incoming_error)[0],
            ) * recovered_adjoint["dx"]
        ))
        adjoint_reverse_weak += direct_reverse_slab

        primal_values = primal_volume + primal_space + primal_temporal + primal_ridge
        correction_values = (
            correction_volume + correction_space
            + correction_temporal + correction_ridge
        )
        adjoint_values = (
            adjoint_volume + adjoint_space + adjoint_temporal + adjoint_ridge
        )
        primal_cells[n], correction_cells[n], adjoint_cells[n] = (
            primal_values, correction_values, adjoint_values
        )
        primal_ridge_cells[n] = primal_ridge
        correction_ridge_cells[n] = correction_ridge
        adjoint_ridge_cells[n] = adjoint_ridge
        primal_volume_cells[n] = primal_volume
        primal_spatial_cells[n] = primal_space
        primal_temporal_cells[n] = primal_temporal
        adjoint_volume_cells[n] = adjoint_volume
        adjoint_spatial_cells[n] = adjoint_space
        adjoint_temporal_cells[n] = adjoint_temporal
        correction_volume_cells[n] = correction_volume
        correction_spatial_cells[n] = correction_space
        correction_temporal_cells[n] = correction_temporal
        remainder_values = np.asarray(
            estimate.eta_cubic_remainder_cell_weak[n], dtype=float
        )
        values = (
            0.5 * primal_values
            + 0.5 * adjoint_values
            + correction_values
            + remainder_values
        )
        signed[n] = values
        field = Function(DG0, name=f"eta_tensor_bubble_cone_abs_slab_{n}")
        field.dat.data[:] = np.abs(values)
        fields[n] = field

        primal_exact_slab = float(estimate.eta_primal_cell_weak[n].sum())
        correction_exact_slab = float(estimate.eta_correction_cell_weak[n].sum())
        primal_slab_gaps[n] = float(primal_values.sum()) - primal_exact_slab
        correction_slab_gaps[n] = (
            float(correction_values.sum()) - correction_exact_slab
        )
        adjoint_slab_gaps[n] = float(adjoint_values.sum()) - direct_reverse_slab

    primal_sum = float(sum(values.sum() for values in primal_cells[1:]))
    adjoint_sum = float(sum(values.sum() for values in adjoint_cells[1:]))
    correction_sum = float(sum(values.sum() for values in correction_cells[1:]))
    local_sum = float(sum(values.sum() for values in signed[1:]))
    marking_sum = float(sum(np.abs(values).sum() for values in signed[1:]))
    return BubbleConeLocalisation(
        eta_cell_signed=signed,
        eta_primal_bubble_cell=primal_cells,
        eta_adjoint_bubble_cell=adjoint_cells,
        eta_correction_bubble_cell=correction_cells,
        eta_cubic_remainder_cell=estimate.eta_cubic_remainder_cell_weak,
        eta_primal_mixed_ridge_cell=primal_ridge_cells,
        eta_adjoint_mixed_ridge_cell=adjoint_ridge_cells,
        eta_correction_mixed_ridge_cell=correction_ridge_cells,
        eta_primal_volume_cell=primal_volume_cells,
        eta_primal_spatial_cell=primal_spatial_cells,
        eta_primal_temporal_cell=primal_temporal_cells,
        eta_adjoint_volume_cell=adjoint_volume_cells,
        eta_adjoint_spatial_cell=adjoint_spatial_cells,
        eta_adjoint_temporal_cell=adjoint_temporal_cells,
        eta_correction_volume_cell=correction_volume_cells,
        eta_correction_spatial_cell=correction_spatial_cells,
        eta_correction_temporal_cell=correction_temporal_cells,
        eta_local_sum=local_sum,
        eta_marking_sum=marking_sum,
        localisation_gap=local_sum - estimate.eta_global,
        eta_primal_recovered=primal_sum,
        eta_adjoint_recovered=adjoint_sum,
        eta_correction_recovered=correction_sum,
        primal_recovery_gap=primal_sum - estimate.eta_primal_residual,
        adjoint_recovery_gap=adjoint_sum - estimate.eta_adjoint_residual,
        correction_recovery_gap=correction_sum - estimate.eta_galerkin_correction,
        eta_adjoint_reverse_weak=adjoint_reverse_weak,
        adjoint_reverse_identity_gap=(
            adjoint_reverse_weak - estimate.eta_adjoint_residual
        ),
        dual_weight_mode=str(dual_weight_mode),
        fields=fields,
        primal_slab_recovery_gap=primal_slab_gaps,
        adjoint_slab_recovery_gap=adjoint_slab_gaps,
        correction_slab_recovery_gap=correction_slab_gaps,
        recovered_entities=entities,
        recovery_time_degree=int(recovery_time_degree),
    )


def mark_localisation(
    primal, localisation: BubbleConeLocalisation, *, theta: float,
    time_slab_marked_fraction: float = 0.05,
) -> MarkingDecision:
    """Apply one global space-time Dörfler decision to ``abs(eta[K,n])``."""
    if not 0.0 < float(theta) <= 1.0:
        raise ValueError("theta must lie in (0, 1].")
    if not 0.0 <= float(time_slab_marked_fraction) <= 1.0:
        raise ValueError("time_slab_marked_fraction must lie in [0, 1].")
    marked = mark_spacetime_cells(localisation.eta_cell_signed, float(theta))
    masks = [np.asarray(mask, dtype=bool) for mask in marked[1:]]
    spatial_union = (
        np.logical_or.reduce(masks) if masks else np.zeros(0, dtype=bool)
    )
    DG0 = FunctionSpace(primal["mesh"], "DG", 0)
    marker = Function(DG0, name="tensor_bubble_cone_spatial_marker")
    marker.dat.data[:] = spatial_union.astype(marker.dat.data.dtype)
    fractions = [0.0]
    time_slabs: set[int] = set()
    for n, mask in enumerate(masks, start=1):
        fraction = float(np.count_nonzero(mask)) / float(mask.size)
        fractions.append(fraction)
        if np.any(mask) and fraction >= float(time_slab_marked_fraction):
            time_slabs.add(n)
    selected = float(sum(
        np.abs(localisation.eta_cell_signed[n])[marked[n]].sum()
        for n in range(1, len(marked))
    ))
    return MarkingDecision(
        marked_by_slab=marked,
        spatial_union=spatial_union,
        spatial_marker=marker,
        time_slabs=time_slabs,
        marked_fraction_by_slab=fractions,
        selected_activity=selected,
        total_activity=localisation.eta_marking_sum,
    )


__all__ = [
    "BubbleConeLocalisation",
    "MarkingDecision",
    "localise_linear_dwr",
    "localise_symmetric_dwr",
    "mark_localisation",
]
