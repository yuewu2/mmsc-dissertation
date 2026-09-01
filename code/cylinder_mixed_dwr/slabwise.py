r"""Slabwise mixed solves and velocity-only P/P* interface transfers.

This module is intentionally problem-specific.  It keeps the public scalar
``nonstationary_dwr`` interface unchanged while allowing every cylinder time
slab to own an independent Netgen mesh.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    TestFunction,
    TestFunctions,
    VectorFunctionSpace,
    assemble,
    dx,
    inner,
)
from firedrake.petsc import PETSc
from irksome import DiscontinuousGalerkinScheme, TimeStepper

from navier_stokes_cylinder_irksome_static_adjoint import (
    _direct_linear_parameters,
)
from navier_stokes_cylinder_irksome_static_primal import (
    StaticCylinderParameters,
    _copy,
    _direct_newton_parameters,
    evaluate_slab,
    _primal_residual,
    _irksome_dg_nodes,
    _stage_coefficients,
)

from .adapter import CylinderMixedDAEAdapter
from .adjoint import reverse_adjoint_residual


@dataclass
class SlabTransferBundle:
    low_spaces: list[tuple[Any, Any, Any] | None]
    rich_spaces: list[tuple[Any, Any, Any] | None]
    low: list[Any | None]
    rich: list[Any | None]


def _low_spaces(mesh):
    velocity = VectorFunctionSpace(mesh, "CG", 2)
    pressure = FunctionSpace(mesh, "CG", 1)
    return velocity, pressure, velocity * pressure


def build_slab_transfers(
    meshes, labels, *, mode: str = "mass", low_family: str = "taylor_hood",
    enriched_velocity_degree: int = 4,
    enriched_pressure_degree: int = 2,
) -> SlabTransferBundle:
    """Construct all low/rich spaces and forward-interface transfer pairs."""
    if mode not in {"mass", "stokes_l2", "stokes_h1"} or low_family != "taylor_hood":
        raise ValueError(
            "The independent solver supports Taylor--Hood with mass, "
            "Stokes-L2, or Stokes-H1 transfer."
        )
    if int(enriched_velocity_degree) < 3:
        raise ValueError("The enriched velocity degree must be at least three.")
    if int(enriched_pressure_degree) < 2:
        raise ValueError("The enriched pressure degree must be at least two.")
    if int(enriched_velocity_degree) <= int(enriched_pressure_degree):
        raise ValueError(
            "The enriched velocity degree must exceed the pressure degree."
        )
    nslabs = len(meshes) - 1
    adapter = CylinderMixedDAEAdapter(labels)
    low_spaces = [None] + [
        _low_spaces(meshes[n]) for n in range(1, nslabs + 1)
    ]
    rich_spaces = [None]
    for n in range(1, nslabs + 1):
        velocity = VectorFunctionSpace(
            meshes[n], "CG", int(enriched_velocity_degree)
        )
        pressure = FunctionSpace(
            meshes[n], "CG", int(enriched_pressure_degree)
        )
        rich_spaces.append((velocity, pressure, velocity * pressure))
    low = [None] * (nslabs + 1)
    rich = [None] * (nslabs + 1)
    for n in range(1, nslabs):
        if meshes[n] is meshes[n + 1]:
            # The discrete interface operator on an unchanged space is the
            # identity.  None denotes this exact path for both P and P*.
            continue
        low[n] = adapter.build_velocity_transfer(
            low_spaces[n][0], low_spaces[n + 1][0], mode=mode,
            source_pressure_space=low_spaces[n][1],
            target_pressure_space=low_spaces[n + 1][1],
        )
        rich[n] = adapter.build_velocity_transfer(
            rich_spaces[n][0], rich_spaces[n + 1][0], mode=mode,
            source_pressure_space=rich_spaces[n][1],
            target_pressure_space=rich_spaces[n + 1][1],
        )
    return SlabTransferBundle(low_spaces, rich_spaces, low, rich)


def _initial_state(adapter, mixed_space, physical_time: float, name: str):
    state = Function(mixed_space, name=name)
    state.assign(0.0)
    for bc in adapter.primal_boundary_conditions(
        mixed_space, Constant(float(physical_time))
    ):
        bc.apply(state)
    return state


def _projected_inflow_mean(time, left: float, step: float, degree: int):
    """Common slabwise L2 lifting of the sinusoidal inlet trace.

    Using this same polynomial for low and enriched solves makes their
    difference an admissible homogeneous Dirichlet variation, as required by
    the nonlinear DWR identity.
    """
    degree = int(degree)
    nodes = _irksome_dg_nodes(degree)
    points, weights = np.polynomial.legendre.leggauss(max(8, degree + 3))
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    basis = np.ones((degree + 1, len(points)))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                basis[i] *= (points - node_j) / (node_i - node_j)
    exact = 1.5 * np.sin(np.pi * (float(left) + float(step) * points) / 8.0)
    mass = (basis * weights) @ basis.T
    rhs = basis @ (weights * exact)
    coefficients = np.linalg.solve(mass, rhs)
    reference_time = (time - Constant(float(left))) / Constant(float(step))
    value = 0.0
    for i, node_i in enumerate(nodes):
        shape = 1.0
        for j, node_j in enumerate(nodes):
            if i != j:
                shape *= (reference_time - float(node_j)) / (
                    float(node_i) - float(node_j)
                )
        value += float(coefficients[i]) * shape
    return value


def _solve_primal_family(
    meshes,
    times,
    spaces,
    transfers,
    labels,
    viscosity,
    time_degree: int,
    *,
    name_prefix: str,
    boundary_time_degree: int | None = None,
    report_every: int = 0,
):
    adapter = CylinderMixedDAEAdapter(labels, viscosity=float(viscosity))
    nslabs = len(times) - 1
    slabs: list[dict[str, Any] | None] = [None] * (nslabs + 1)
    previous_state = None
    for n in range(1, nslabs + 1):
        mixed_space = spaces[n][2]
        interface_diagnostics = None
        if previous_state is None:
            state = _initial_state(
                adapter, mixed_space, float(times[n - 1]), f"{name_prefix}_{n}"
            )
        elif transfers[n - 1] is None:
            # Pressure is copied only as a nonlinear algebraic initial guess;
            # it still has no mass or dG interface term.
            state = _copy(previous_state, f"I_{name_prefix}_interface_{n}")
        else:
            state = adapter.forward_interface_state(
                transfers[n - 1],
                previous_state,
                mixed_space,
                float(times[n - 1]),
                name=f"P_{name_prefix}_interface_{n}",
            )
            interface_diagnostics = {
                "source_slab": int(n - 1),
                "target_slab": int(n),
                "physical_time": float(times[n - 1]),
                "source_cells": int(meshes[n - 1].num_cells()),
                "target_cells": int(meshes[n].num_cells()),
                **dict(transfers[n - 1].last_forward_diagnostics),
            }
        left_trace = _copy(state, f"{name_prefix}_left_trace_{n}")
        time = Constant(float(times[n - 1]))
        step = float(times[n] - times[n - 1])
        lifting_degree = (
            int(time_degree)
            if boundary_time_degree is None
            else int(boundary_time_degree)
        )
        inflow_mean = _projected_inflow_mean(
            time, float(times[n - 1]), step, lifting_degree
        )
        bcs = adapter.primal_boundary_conditions(
            mixed_space, time, inflow_mean=inflow_mean
        )
        stepper = TimeStepper(
            _primal_residual(
                state, TestFunctions(mixed_space), Constant(float(viscosity))
            ),
            DiscontinuousGalerkinScheme(
                int(time_degree), quadrature_degree=3 * int(time_degree)
            ),
            time,
            Constant(step),
            state,
            bcs=bcs,
            solver_parameters=_direct_newton_parameters(),
        )
        stepper.advance()
        time.assign(float(times[n]))
        slabs[n] = {
            "mesh": meshes[n],
            "degree": int(time_degree),
            "coeffs": adapter.pack_irksome_stages(
                stepper,
                mixed_space,
                int(time_degree),
                f"{name_prefix}_slab_{n}",
            ),
            "left_trace": left_trace,
            "right_trace": _copy(state, f"{name_prefix}_right_trace_{n}"),
            "incoming_transfer_diagnostics": interface_diagnostics,
        }
        previous_state = slabs[n]["right_trace"]
        del stepper
        gc.collect()
        if report_every and (
            n == 1 or n == nslabs or n % int(report_every) == 0
        ):
            PETSc.Sys.Print(
                f"[CYLINDER {name_prefix.upper()}] slab {n}/{nslabs}; "
                f"cells={meshes[n].num_cells()}."
            )
    return slabs


def solve_slabwise_primal(
    meshes,
    times,
    transfers: SlabTransferBundle,
    labels,
    *,
    viscosity: float = 1.0e-3,
    time_degree: int = 1,
    report_every: int = 0,
):
    slabs = _solve_primal_family(
        meshes,
        times,
        transfers.low_spaces,
        transfers.low,
        labels,
        viscosity,
        time_degree,
        name_prefix="primal_low",
        boundary_time_degree=int(time_degree),
        report_every=report_every,
    )
    parameters = StaticCylinderParameters(
        hierarchy_levels=0,
        time_steps=len(times) - 1,
        final_time=float(times[-1]),
        time_degree=int(time_degree),
        viscosity=float(viscosity),
        store_stages=True,
        report_every=int(report_every),
        slabwise_steppers=True,
    )
    return {
        "parameters": parameters,
        "hierarchy": None,
        "mesh": meshes[1],
        "meshes": list(meshes),
        "mixed_space": transfers.low_spaces[1][2],
        "mixed_spaces": [None] + [space[2] for space in transfers.low_spaces[1:]],
        "viscosity": Constant(float(viscosity)),
        "times": np.asarray(times, dtype=float),
        "slabs": slabs,
        "labels": labels,
        "transfers": transfers.low,
    }


def solve_slabwise_enriched_primal(
    primal,
    transfers: SlabTransferBundle,
    *,
    time_degree: int,
    boundary_time_degree: int | None = None,
    report_every: int = 0,
):
    lifting_degree = (
        int(primal["parameters"].time_degree)
        if boundary_time_degree is None
        else int(boundary_time_degree)
    )
    slabs = _solve_primal_family(
        primal["meshes"],
        primal["times"],
        transfers.rich_spaces,
        transfers.rich,
        primal["labels"],
        float(primal["viscosity"]),
        int(time_degree),
        name_prefix="primal_rich",
        boundary_time_degree=lifting_degree,
        report_every=report_every,
    )
    return {
        "parameters": StaticCylinderParameters(
            hierarchy_levels=0,
            time_steps=len(primal["times"]) - 1,
            final_time=float(primal["times"][-1]),
            time_degree=int(time_degree),
            viscosity=float(primal["viscosity"]),
            store_stages=True,
            report_every=int(report_every),
            slabwise_steppers=True,
        ),
        "hierarchy": None,
        "mesh": primal["meshes"][1],
        "meshes": primal["meshes"],
        "mixed_space": transfers.rich_spaces[1][2],
        "mixed_spaces": [None] + [space[2] for space in transfers.rich_spaces[1:]],
        "viscosity": primal["viscosity"],
        "times": primal["times"],
        "slabs": slabs,
        "labels": primal["labels"],
        "degree": int(time_degree),
        "boundary_time_degree": lifting_degree,
        "transfers": transfers.rich,
        "enrichment": (
            "slabwise CG"
            f"{transfers.rich_spaces[1][0].ufl_element().degree()}/"
            f"CG{transfers.rich_spaces[1][1].ufl_element().degree()}"
        ),
    }


def solve_slabwise_adjoint(
    primal,
    transfers: SlabTransferBundle,
    *,
    time_degree: int,
    spatially_enriched: bool,
    report_every: int = 0,
):
    meshes = primal["meshes"]
    times = primal["times"]
    spaces = transfers.rich_spaces if spatially_enriched else transfers.low_spaces
    interface_transfers = transfers.rich if spatially_enriched else transfers.low
    nslabs = len(times) - 1
    adapter = CylinderMixedDAEAdapter(
        primal["labels"], viscosity=float(primal["viscosity"])
    )
    slabs: list[dict[str, Any] | None] = [None] * (nslabs + 1)
    incoming_state = None
    final_time = float(times[-1])
    horizon = float(times[-1] - times[0])
    for n in range(nslabs, 0, -1):
        mixed_space = spaces[n][2]
        if incoming_state is None:
            state = Function(mixed_space, name=f"Z_terminal_slab_{n}").assign(0.0)
        elif interface_transfers[n] is None:
            # As above, the pressure component is merely a linear-solver
            # initial guess.  The adjoint interface mass acts on velocity only.
            state = _copy(incoming_state, f"Istar_Z_interface_{n}")
        else:
            state = adapter.adjoint_interface_state(
                interface_transfers[n],
                incoming_state,
                mixed_space,
                name=f"Pstar_Z_interface_{n}",
            )
        for bc in adapter.adjoint_boundary_conditions(mixed_space):
            bc.apply(state)
        incoming_trace = _copy(state, f"Z_reverse_incoming_slab_{n}")
        step = float(times[n] - times[n - 1])
        reverse_time = Constant(final_time - float(times[n]))
        residual = reverse_adjoint_residual(
            state,
            TestFunctions(mixed_space),
            reverse_time,
            primal["slabs"][n],
            slab_left=float(times[n - 1]),
            step=step,
            final_time=final_time,
            goal_horizon=horizon,
            viscosity=primal["viscosity"],
            cylinder_labels=primal["labels"]["cylinder"],
        )
        stepper = TimeStepper(
            residual,
            DiscontinuousGalerkinScheme(
                int(time_degree), quadrature_degree=3 * int(time_degree)
            ),
            reverse_time,
            Constant(step),
            state,
            bcs=adapter.adjoint_boundary_conditions(mixed_space),
            solver_parameters=_direct_linear_parameters(),
        )
        stepper.advance()
        slabs[n] = {
            "mesh": meshes[n],
            "degree": int(time_degree),
            "coeffs": adapter.pack_irksome_stages(
                stepper,
                mixed_space,
                int(time_degree),
                f"Z_{'rich' if spatially_enriched else 'low'}_slab_{n}",
            ),
            "incoming_trace": incoming_trace,
            "outgoing_trace": _copy(state, f"Z_reverse_outgoing_slab_{n}"),
        }
        incoming_state = slabs[n]["outgoing_trace"]
        del stepper
        gc.collect()
        if report_every and (
            n == 1 or n == nslabs or n % int(report_every) == 0
        ):
            PETSc.Sys.Print(
                f"[CYLINDER {'RICH' if spatially_enriched else 'LOW'} ADJOINT] "
                f"slab {n}/{nslabs}."
            )
    return {
        "degree": int(time_degree),
        "spatially_enriched": bool(spatially_enriched),
        "mixed_space": spaces[1][2],
        "mixed_spaces": [None] + [space[2] for space in spaces[1:]],
        "slabs": slabs,
        "transfers": interface_transfers,
    }


def _restrict_primal_to_uniform_time_children(primal, factor: int):
    r"""Represent every saved low dG polynomial exactly on uniform children."""
    factor = int(factor)
    if factor < 1:
        raise ValueError("The temporal refinement factor must be positive.")
    if factor == 1:
        return primal
    degree = int(primal["parameters"].time_degree)
    nodes = _irksome_dg_nodes(degree)
    child_times = [float(primal["times"][0])]
    child_meshes: list[Any | None] = [None]
    child_slabs: list[dict[str, Any] | None] = [None]
    for parent in range(1, len(primal["times"])):
        left = float(primal["times"][parent - 1])
        right = float(primal["times"][parent])
        parent_slab = primal["slabs"][parent]
        for child in range(factor):
            child_left = left + (right - left) * child / factor
            child_right = left + (right - left) * (child + 1) / factor
            coefficients = [
                evaluate_slab(
                    parent_slab,
                    (float(child) + float(node)) / factor,
                    name=(
                        f"U_low_parent_{parent}_child_{child + 1}_stage_{stage}"
                    ),
                )
                for stage, node in enumerate(nodes)
            ]
            child_times.append(child_right)
            child_meshes.append(primal["meshes"][parent])
            child_slabs.append(
                {
                    "mesh": primal["meshes"][parent],
                    "degree": degree,
                    "coeffs": coefficients,
                    "left_trace": evaluate_slab(
                        parent_slab,
                        float(child) / factor,
                        name=f"U_low_parent_{parent}_child_{child + 1}_left",
                    ),
                    "right_trace": evaluate_slab(
                        parent_slab,
                        float(child + 1) / factor,
                        name=f"U_low_parent_{parent}_child_{child + 1}_right",
                    ),
                    "parent_slab": parent,
                    "child_number": child + 1,
                }
            )
    restricted = dict(primal)
    restricted.update(
        {
            "times": np.asarray(child_times, dtype=float),
            "meshes": child_meshes,
            "slabs": child_slabs,
            "mesh": child_meshes[1],
            "temporal_parent_times": np.asarray(primal["times"], dtype=float),
            "temporal_refinement_factor": factor,
        }
    )
    return restricted


def evaluate_temporally_composite_slab(
    slab: dict[str, Any], reverse_point: float, *, name: str
) -> Function:
    r"""Evaluate an ordinary or uniformly child-refined reverse-time slab."""
    children = slab.get("piecewise_children")
    if children is None:
        return evaluate_slab(slab, float(reverse_point), name=name)
    factor = len(children)
    if factor < 1:
        raise ValueError("A composite temporal slab must contain children.")
    # The stored adjoint polynomials use reverse time.  Convert the parent
    # reverse coordinate to physical orientation, choose the child, then
    # convert back to that child's reverse coordinate.
    physical = min(max(1.0 - float(reverse_point), 0.0), 1.0)
    scaled = physical * factor
    child_number = min(int(np.floor(scaled)), factor - 1)
    child_physical = scaled - child_number
    child_reverse = 1.0 - child_physical
    return evaluate_slab(
        children[child_number], child_reverse,
        name=f"{name}_child_{child_number + 1}",
    )


def _gauss_legendre_time_nodes(degree: int) -> np.ndarray:
    r"""R5 support points for the temporal interpolation ``I_k``.

    Irksome stores a dG polynomial in an equispaced Lagrange basis.  That
    storage choice must not silently define the DWR interpolation operator:
    R5 uses ``degree + 1`` Gauss--Legendre support points on each slab.
    """
    points, _ = np.polynomial.legendre.leggauss(int(degree) + 1)
    return 0.5 * (np.asarray(points, dtype=float) + 1.0)


def _lagrange_weights(nodes: np.ndarray, point: float) -> np.ndarray:
    values = np.ones(len(nodes), dtype=float)
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                values[i] *= (float(point) - float(node_j)) / (
                    float(node_i) - float(node_j)
                )
    return values


def r5_temporal_interpolant_coefficients(
    rich_slab: dict[str, Any], degree: int, *, prefix: str
) -> list[Function]:
    r"""Return equispaced storage coefficients of R5's GL-node interpolant."""
    degree = int(degree)
    support_nodes = _gauss_legendre_time_nodes(degree)
    storage_nodes = _irksome_dg_nodes(degree)
    samples = [
        evaluate_temporally_composite_slab(
            rich_slab, float(node), name=f"{prefix}_GL_sample_{sample}"
        )
        for sample, node in enumerate(support_nodes)
    ]
    coefficients: list[Function] = []
    for stage, node in enumerate(storage_nodes):
        weights = _lagrange_weights(support_nodes, float(node))
        coefficient = Function(
            samples[0].function_space(), name=f"{prefix}_storage_{stage}"
        )
        coefficient.assign(0.0)
        for target in coefficient.subfunctions:
            target.assign(0.0)
        for sample, weight in zip(samples, weights):
            for target, source in zip(
                coefficient.subfunctions, sample.subfunctions
            ):
                target.dat.data[:] += float(weight) * source.dat.data_ro
        coefficients.append(coefficient)
    return coefficients


def solve_temporally_refined_enriched_adjoint(
    primal,
    *,
    refinement_factor: int,
    time_degree: int,
    interface_transfer_mode: str,
    low_family: str,
    enriched_velocity_degree: int,
    enriched_pressure_degree: int,
    report_every: int = 0,
):
    r"""Solve the rich adjoint on uniform child slabs of the low trajectory."""
    factor = int(refinement_factor)
    if factor < 1:
        raise ValueError("refinement_factor must be at least one.")
    if factor == 1:
        raise ValueError(
            "Use solve_slabwise_adjoint for an unrefined enriched time grid."
        )
    fine_primal = _restrict_primal_to_uniform_time_children(primal, factor)
    fine_transfers = build_slab_transfers(
        fine_primal["meshes"],
        primal["labels"],
        mode=interface_transfer_mode,
        low_family=low_family,
        enriched_velocity_degree=int(enriched_velocity_degree),
        enriched_pressure_degree=int(enriched_pressure_degree),
    )
    fine_dual = solve_slabwise_adjoint(
        fine_primal,
        fine_transfers,
        time_degree=int(time_degree),
        spatially_enriched=True,
        report_every=int(report_every),
    )
    parent_slabs: list[dict[str, Any] | None] = [None]
    for parent in range(1, len(primal["times"])):
        first = (parent - 1) * factor + 1
        children = [fine_dual["slabs"][first + j] for j in range(factor)]
        parent_slabs.append(
            {
                "mesh": primal["meshes"][parent],
                "degree": int(time_degree),
                "piecewise_children": children,
                "temporal_refinement_factor": factor,
            }
        )
    fine_dofs = int(sum(
        fine_transfers.rich_spaces[n][2].dim() * (int(time_degree) + 1)
        for n in range(1, len(fine_primal["times"]))
    ))
    return {
        "degree": int(time_degree),
        "spatially_enriched": True,
        "mixed_space": fine_transfers.rich_spaces[1][2],
        "mixed_spaces": [None]
        + [
            fine_transfers.rich_spaces[(parent - 1) * factor + 1][2]
            for parent in range(1, len(primal["times"]))
        ],
        "slabs": parent_slabs,
        "fine_times": fine_primal["times"],
        "fine_slabs": fine_dual["slabs"],
        "fine_spacetime_dofs": fine_dofs,
        "temporal_refinement_factor": factor,
        "construction": "temporally_refined_enriched_adjoint",
    }


def project_enriched_adjoint_to_low(dual_enriched, transfers, labels):
    r"""Build ``Pi_h z+`` coefficientwise without a low-adjoint solve.

    The velocity projection is the constrained L2 Riesz representation in
    the homogeneous Taylor--Hood test space.  Pressure is projected with its
    ordinary L2 map.  Keeping the enriched dG time coefficients makes this a
    purely spatial projection of the complete enriched adjoint trajectory.
    """
    essential = tuple(labels["inlet"]) + tuple(labels["wall"])
    direct = _direct_linear_parameters()
    slabs: list[dict[str, Any] | None] = [None] * len(dual_enriched["slabs"])
    for n in range(1, len(slabs)):
        low_mixed = transfers.low_spaces[n][2]
        low_velocity = transfers.low_spaces[n][0]
        velocity_test = TestFunction(low_velocity)
        coefficients = []
        for stage, rich in enumerate(dual_enriched["slabs"][n]["coeffs"]):
            projected = Function(
                low_mixed, name=f"Pi_h_Z_rich_slab_{n}_stage_{stage}"
            )
            velocity_functional = assemble(
                inner(rich.subfunctions[0], velocity_test) * dx
            )
            velocity_bc = DirichletBC(
                low_velocity, Constant((0.0, 0.0)), essential
            )
            projected.subfunctions[0].assign(
                velocity_functional.riesz_representation(
                    "L2", bcs=velocity_bc, solver_options=direct
                )
            )
            projected.subfunctions[1].project(
                rich.subfunctions[1], solver_parameters=direct
            )
            coefficients.append(projected)
        slabs[n] = {
            "mesh": dual_enriched["slabs"][n]["mesh"],
            "degree": int(dual_enriched["slabs"][n]["degree"]),
            "coeffs": coefficients,
        }
    return {
        "degree": int(dual_enriched["degree"]),
        "spatially_enriched": False,
        "mixed_space": transfers.low_spaces[1][2],
        "mixed_spaces": [None]
        + [space[2] for space in transfers.low_spaces[1:]],
        "slabs": slabs,
        "transfers": transfers.low,
        "construction": "constrained_spatial_L2_projection_of_enriched_adjoint",
    }


def interpolate_enriched_adjoint_to_low(
    dual_enriched, transfers, labels, *, time_degree: int
):
    r"""Build R5's ``I_h I_k z+`` in the low space-time space.

    The operator is taken from the mixed-order construction in R5 Section
    3.6, but is used here inside the project's one-term linear estimator.
    ``I_k`` is defined at ``degree + 1`` Gauss--Legendre support points, as in
    R5's preferred variant.  Its polynomial is then converted to Irksome's
    equispaced coefficient storage.  Each field is interpolated at the low
    spatial DOFs.  No low-order adjoint equation is solved.
    """
    time_degree = int(time_degree)
    if time_degree < 0:
        raise ValueError("time_degree must be nonnegative.")
    essential = tuple(labels["inlet"]) + tuple(labels["wall"])
    slabs: list[dict[str, Any] | None] = [None] * len(dual_enriched["slabs"])
    for n in range(1, len(slabs)):
        low_mixed = transfers.low_spaces[n][2]
        coefficients = []
        rich_coefficients = r5_temporal_interpolant_coefficients(
            dual_enriched["slabs"][n],
            time_degree,
            prefix=f"I_k_Z_rich_slab_{n}",
        )
        for stage, rich in enumerate(rich_coefficients):
            interpolated = Function(
                low_mixed, name=f"I_h_I_k_Z_rich_slab_{n}_stage_{stage}"
            )
            for target, source in zip(
                interpolated.subfunctions, rich.subfunctions
            ):
                target.interpolate(source)
            DirichletBC(
                low_mixed.sub(0), Constant((0.0, 0.0)), essential
            ).apply(interpolated)
            coefficients.append(interpolated)
        slabs[n] = {
            "mesh": dual_enriched["slabs"][n]["mesh"],
            "degree": time_degree,
            "coeffs": coefficients,
        }
    return {
        "degree": time_degree,
        "spatially_enriched": False,
        "mixed_space": transfers.low_spaces[1][2],
        "mixed_spaces": [None]
        + [space[2] for space in transfers.low_spaces[1:]],
        "slabs": slabs,
        "transfers": transfers.low,
        "construction": (
            "R5_Gauss_Legendre_time_and_spatial_nodal_interpolation_"
            "of_enriched_adjoint"
        ),
    }


def verify_transfer_pairing(transfer, source_space, target_space, essential):
    """Check ``(Pv,z)_target=(v,P*z)_source`` for one interface."""
    source = Function(source_space, name="P_pairing_source")
    target = Function(target_space, name="P_pairing_target")
    source_values = source.dat.data.reshape(-1)
    target_values = target.dat.data.reshape(-1)
    source_values[:] = np.sin(np.arange(source_values.size, dtype=float))
    target_values[:] = np.cos(np.arange(target_values.size, dtype=float))
    DirichletBC(
        source_space, Constant((0.0, 0.0)), tuple(essential)
    ).apply(source)
    DirichletBC(
        target_space, Constant((0.0, 0.0)), tuple(essential)
    ).apply(target)
    left = float(assemble(inner(transfer.forward(source), target) * dx))
    right = float(assemble(inner(source, transfer.adjoint(target)) * dx))
    return {"left": left, "right": right, "gap": left - right}


__all__ = [
    "SlabTransferBundle",
    "build_slab_transfers",
    "solve_slabwise_adjoint",
    "solve_temporally_refined_enriched_adjoint",
    "solve_slabwise_enriched_primal",
    "solve_slabwise_primal",
    "r5_temporal_interpolant_coefficients",
    "interpolate_enriched_adjoint_to_low",
    "evaluate_temporally_composite_slab",
    "project_enriched_adjoint_to_low",
    "verify_transfer_pairing",
]
