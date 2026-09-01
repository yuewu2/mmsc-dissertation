r"""Reusable static-mesh primal for the Irksome cylinder benchmark.

This module intentionally preserves the stable Section 5.3 discretisation from
``code/cylinder.py``: a fixed, nested Netgen ``MeshHierarchy``, the
Alfeld--Sorokina/Alfeld pressure pair, and Newton--Krylov with monolithic
Vanka multigrid.  Unlike the original plotting driver, it retains the left
trace and every dG stage coefficient in memory.  The resulting history is the
input needed by a reverse adjoint and a global DWR check.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
from firedrake import (
    Constant, DirichletBC, DistributedMeshOverlapType, FacetNormal, Function,
    FunctionSpace, Mesh, MeshHierarchy, SpatialCoordinate, TestFunctions,
    as_vector, assemble, div, dot, ds, dx, grad, inner, pi, sin, split, sqrt,
)
from firedrake.petsc import PETSc
from irksome import DiscontinuousGalerkinScheme, Dt, TimeStepper
from mpi4py import MPI
from netgen.occ import Circle, OCCGeometry, Pnt, Rectangle, X
from automated_DWR.time_solver import gauss_rule


@dataclass(frozen=True)
class StaticCylinderParameters:
    """Parameters for the fixed-mesh Section 5.3 forward solve."""

    hierarchy_levels: int = 1
    time_steps: int = 128
    final_time: float = 8.0
    time_degree: int = 1
    viscosity: float = 1.0e-3
    store_stages: bool = True
    report_every: int = 16
    slabwise_steppers: bool = False


def _copy(value: Function, name: str) -> Function:
    result = Function(value.function_space(), name=name)
    result.assign(value)
    return result


def _make_hierarchy(
    levels: int,
    geometry_degree: int = 1,
) -> tuple[MeshHierarchy, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Create exactly the static nested hierarchy used in ``code/cylinder.py``."""
    distribution_parameters = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
    }
    # Nonstationary DFG/Roth cylinder benchmark domain:
    # (0, 2.2) x (0, 0.41) with the circular obstacle removed.
    # Keep this physical length in sync with R5CylinderSpecification and the
    # dissertation's Section 3.5 definition of Omega.
    rectangle = Rectangle(2.2, 0.41).Face()
    rectangle.edges.name = "wall"
    cylinder = Circle(Pnt(0.2, 0.2), 0.05).Face()
    cylinder.edges.name = "cyl"
    shape = rectangle - cylinder
    shape.edges.Min(X).name = "inlet"
    shape.edges.Max(X).name = "outlet"
    ngmesh = OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.1)
    mesh0 = Mesh(ngmesh, distribution_parameters=distribution_parameters)
    if int(geometry_degree) < 1:
        raise ValueError("geometry_degree must be at least one.")
    hierarchy = MeshHierarchy(
        mesh0,
        int(levels),
        netgen_flags={
            "degree": int(geometry_degree),
            "snap_to": "geometry",
        },
    )
    hierarchy.nested = True
    labels_inlet = tuple(
        index + 1 for index, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "inlet"
    )
    labels_wall = tuple(
        index + 1 for index, name in enumerate(ngmesh.GetRegionNames(codim=1))
        if name in {"wall", "cyl"}
    )
    labels_cylinder = tuple(
        index + 1 for index, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "cyl"
    )
    labels_outlet = tuple(
        index + 1 for index, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "outlet"
    )
    return hierarchy, labels_inlet, labels_wall, labels_cylinder, labels_outlet


def _spaces(mesh: Mesh):
    """The pointwise-divergence-free Alfeld--Sorokina pair from Section 5.3."""
    velocity = FunctionSpace(mesh, "AS", 2)
    pressure = FunctionSpace(mesh, "CG", 1, variant="alfeld")
    return velocity, pressure, velocity * pressure


def _solver_parameters(time_degree: int) -> dict[str, Any]:
    pressure_indices = ",".join(str(2 * stage + 1) for stage in range(int(time_degree) + 1))
    return {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_converged_reason": None,
        "snes_linesearch_type": "l2",
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "ksp_max_it": 30,
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-10,
        "snes_ksp_ew": None,
        "pc_type": "mg",
        "pc_mg_type": "multiplicative",
        "pc_mg_cycles": "v",
        "mg_levels": {
            "ksp_type": "gmres",
            "ksp_max_it": 3,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_construct_dim": 0,
            "pc_vanka_sub_sub_pc_type": "lu",
            "pc_vanka_sub_sub_pc_factor_mat_solver_type": "umfpack",
            "pc_vanka_exclude_subspaces": pressure_indices,
        },
        "mg_coarse": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
        },
    }


def _direct_newton_parameters() -> dict[str, Any]:
    """Robust nonlinear fallback for a genuinely locally refined mesh.

    The Section 5.3 Vanka multigrid requires a nested geometric hierarchy.
    A mesh returned by ``refine_marked_elements`` has no such hierarchy, so
    the first marked-mesh verification deliberately uses a monolithic direct
    linear solve inside Newton rather than pretending that the old hierarchy
    still applies.
    """
    return {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_converged_reason": None,
        "snes_linesearch_type": "l2",
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


def _primal_residual(state, tests, viscosity: Constant):
    velocity, pressure = split(state)
    velocity_test, pressure_test = tests
    return (
        inner(Dt(velocity), velocity_test) * dx
        + inner(dot(grad(velocity), velocity), velocity_test) * dx
        + viscosity * inner(grad(velocity), grad(velocity_test)) * dx
        - inner(pressure, div(velocity_test)) * dx
        - inner(div(velocity), pressure_test) * dx
    )


def _stage_coefficients(stepper, mixed_space, degree: int, name: str) -> list[Function]:
    """Copy Irksome's flattened stage vector as mixed dG coefficients."""
    stages = list(stepper.stages.subfunctions)
    expected = len(mixed_space) * (int(degree) + 1)
    if len(stages) != expected:
        raise RuntimeError(f"Unexpected dG stage layout: got {len(stages)}, expected {expected}.")
    coefficients: list[Function] = []
    fields = len(mixed_space)
    for stage_number in range(int(degree) + 1):
        coefficient = Function(mixed_space, name=f"{name}_{stage_number}")
        for target, source in zip(
            coefficient.subfunctions,
            stages[fields * stage_number: fields * (stage_number + 1)],
        ):
            target.assign(source)
        coefficients.append(coefficient)
    return coefficients


def _irksome_dg_nodes(degree: int) -> np.ndarray:
    """FIAT ordering of default discontinuous equispaced Lagrange nodes."""
    degree = int(degree)
    if degree == 0:
        return np.asarray([0.5])
    if degree == 1:
        return np.asarray([0.0, 1.0])
    return np.asarray([0.0, 1.0, *[i / degree for i in range(1, degree)]])


def _lagrange_values(degree: int, point: float) -> np.ndarray:
    nodes = _irksome_dg_nodes(degree)
    values = np.ones(len(nodes))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                values[i] *= (float(point) - node_j) / (node_i - node_j)
    return values


def _lagrange_derivatives(degree: int, point: float) -> np.ndarray:
    """Derivatives of the equispaced Lagrange basis on the unit interval."""
    if int(degree) == 0:
        return np.zeros(1)
    nodes = _irksome_dg_nodes(degree)
    derivatives = np.zeros(len(nodes))
    for i, node_i in enumerate(nodes):
        for omitted in range(len(nodes)):
            if omitted == i:
                continue
            value = 1.0 / (node_i - nodes[omitted])
            for j, node_j in enumerate(nodes):
                if j != i and j != omitted:
                    value *= (float(point) - node_j) / (node_i - node_j)
            derivatives[i] += value
    return derivatives


def evaluate_slab(slab: dict[str, Any], point: float, name: str = "dg_evaluation") -> Function:
    """Evaluate a saved equispaced dG polynomial at a reference time point."""
    coefficients = slab["coeffs"]
    result = Function(coefficients[0].function_space(), name=name)
    for component, target in enumerate(result.subfunctions):
        target.assign(0.0)
        for coefficient, weight in zip(coefficients, _lagrange_values(slab["degree"], point)):
            target.dat.data[:] += float(weight) * coefficient.subfunctions[component].dat.data_ro
    return result


def evaluate_slab_dt(
    slab: dict[str, Any], point: float, step: float, name: str = "dg_time_derivative",
) -> Function:
    """Evaluate the physical-time derivative of a saved dG polynomial."""
    coefficients = slab["coeffs"]
    result = Function(coefficients[0].function_space(), name=name)
    weights = _lagrange_derivatives(slab["degree"], point) / float(step)
    for component, target in enumerate(result.subfunctions):
        target.assign(0.0)
        for coefficient, weight in zip(coefficients, weights):
            target.dat.data[:] += float(weight) * coefficient.subfunctions[component].dat.data_ro
    return result


def mean_drag(primal: dict[str, Any]) -> float:
    """Time-average drag, matching the adjoint/DWR quantity of interest."""
    total = 0.0
    mesh = primal["mesh"]
    normal = FacetNormal(mesh)
    for slab_number in range(1, len(primal["times"])):
        step = float(primal["times"][slab_number] - primal["times"][slab_number - 1])
        for point, weight in gauss_rule(4):
            state = evaluate_slab(primal["slabs"][slab_number], point, name="U_mean_drag")
            velocity, pressure = state.subfunctions
            tangent = dot(velocity, as_vector((normal[1], -normal[0])))
            drag = -20.0 * assemble(
                (primal["viscosity"] * dot(grad(tangent), normal) * normal[1] - pressure * normal[0])
                * ds(primal["labels"]["cylinder"])
            )
            total += step * float(weight) * float(drag)
    return total / float(primal["times"][-1])


def solve_static_primal(
    parameters: StaticCylinderParameters, *, mesh: Mesh | None = None,
    labels: dict[str, tuple[int, ...]] | None = None,
    start_time: float = 0.0, initial_state: Function | None = None,
    solver_parameters_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance a dG primal and retain its slab history.

    With no ``mesh`` argument this is exactly the stable Section 5.3 Vanka-MG
    path.  Supplying a locally marked mesh restarts the forward solve at
    ``t=0`` on that mesh and uses direct Newton/MUMPS; no cross-mesh time
    transfer is introduced.
    """
    if parameters.hierarchy_levels < 1 or parameters.time_steps < 1:
        raise ValueError("hierarchy_levels and time_steps must be positive.")
    if parameters.final_time <= float(start_time) or parameters.viscosity <= 0.0:
        raise ValueError("final_time must exceed start_time and viscosity must be positive.")
    if parameters.time_degree < 0:
        raise ValueError("time_degree must be non-negative.")

    hierarchy = None
    if mesh is None:
        hierarchy, labels_inlet, labels_wall, labels_cylinder, labels_outlet = _make_hierarchy(
            parameters.hierarchy_levels
        )
        mesh = hierarchy[-1]
        labels = {
            "inlet": labels_inlet,
            "wall": labels_wall,
            "cylinder": labels_cylinder,
            "outlet": labels_outlet,
        }
        solver_parameters = _solver_parameters(parameters.time_degree)
    else:
        required_labels = {"inlet", "wall", "cylinder", "outlet"}
        if labels is None or not required_labels.issubset(labels):
            raise ValueError("A marked mesh requires inlet, wall, cylinder and outlet labels.")
        solver_parameters = _direct_newton_parameters()
    if solver_parameters_override is not None:
        solver_parameters = solver_parameters_override
    _, _, mixed_space = _spaces(mesh)
    viscosity = Constant(float(parameters.viscosity))
    step = (float(parameters.final_time) - float(start_time)) / int(parameters.time_steps)
    time = Constant(float(start_time))
    state = Function(mixed_space, name="U")
    if initial_state is not None:
        if initial_state.function_space() != mixed_space:
            raise ValueError("initial_state must belong to the requested mixed space.")
        state.assign(initial_state)
    velocity, pressure = split(state)
    coordinates = SpatialCoordinate(mesh)
    y = coordinates[1]
    inflow_mean = 1.5 * sin(pi * time / 8.0)
    inflow = as_vector((4.0 * inflow_mean * y * (0.41 - y) / 0.41**2, 0.0))
    bcs = [
        DirichletBC(mixed_space.sub(0), inflow, labels["inlet"]),
        DirichletBC(mixed_space.sub(0), Constant((0.0, 0.0)), labels["wall"]),
    ]
    normal = FacetNormal(mesh)
    tangent_velocity = dot(velocity, as_vector((normal[1], -normal[0])))
    drag_form = (viscosity * dot(grad(tangent_velocity), normal) * normal[1] - pressure * normal[0]) * ds(labels["cylinder"])
    lift_form = (viscosity * inner(grad(tangent_velocity), normal) * normal[0] + pressure * normal[1]) * ds(labels["cylinder"])
    divergence_form = inner(div(velocity), div(velocity)) * dx

    stepper = None
    if not parameters.slabwise_steppers:
        stepper = TimeStepper(
            _primal_residual(state, TestFunctions(mixed_space), viscosity),
            DiscontinuousGalerkinScheme(
                int(parameters.time_degree),
                quadrature_degree=3 * int(parameters.time_degree),
            ), time, Constant(step), state,
            bcs=bcs, solver_parameters=solver_parameters,
        )
    slabs: list[dict[str, Any] | None] = [None] * (int(parameters.time_steps) + 1)
    times = np.linspace(float(start_time), float(parameters.final_time), int(parameters.time_steps) + 1)
    drag = np.zeros(len(times))
    lift = np.zeros(len(times))
    divergence = np.zeros(len(times))

    for slab_number in range(1, len(times)):
        left_trace = _copy(state, f"U_left_{slab_number}")
        if parameters.slabwise_steppers:
            stepper = TimeStepper(
                _primal_residual(state, TestFunctions(mixed_space), viscosity),
                DiscontinuousGalerkinScheme(
                    int(parameters.time_degree),
                    quadrature_degree=3 * int(parameters.time_degree),
                ), time, Constant(step), state,
                bcs=bcs, solver_parameters=solver_parameters,
            )
        stepper.advance()
        time.assign(float(times[slab_number]))
        coefficients = (
            _stage_coefficients(stepper, mixed_space, parameters.time_degree, f"U_slab_{slab_number}")
            if parameters.store_stages else []
        )
        slabs[slab_number] = {
            "degree": int(parameters.time_degree),
            "coeffs": coefficients,
            "left_trace": left_trace,
            "right_trace": _copy(state, f"U_right_{slab_number}"),
        }
        if parameters.slabwise_steppers:
            del stepper
            stepper = None
            # A slabwise mesh-change run creates one PETSc solver per slab.
            # The retained dG coefficients are sufficient history, so release
            # each operator/factorisation before constructing the next slab.
            gc.collect()
        drag[slab_number] = -20.0 * float(assemble(drag_form))
        lift[slab_number] = 20.0 * float(assemble(lift_form))
        divergence[slab_number] = float(sqrt(assemble(divergence_form)))
        if parameters.report_every and (
            slab_number == 1
            or slab_number % int(parameters.report_every) == 0
            or slab_number == len(times) - 1
        ):
            PETSc.Sys.Print(
                f"[STATIC PRIMAL] slab {slab_number}/{len(times) - 1} t={times[slab_number]:.6g} "
                f"drag={drag[slab_number]:+.6e} div={divergence[slab_number]:.3e}."
            )

    return {
        "parameters": parameters,
        "hierarchy": hierarchy,
        "mesh": mesh,
        "mixed_space": mixed_space,
        "viscosity": viscosity,
        "times": times,
        "slabs": slabs,
        "drag": drag,
        "lift": lift,
        "divergence": divergence,
        "solver_stats": None if stepper is None else stepper.solver_stats(),
        "communicator": MPI.COMM_WORLD,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lvls", type=int, default=1)
    parser.add_argument("--nt", type=int, default=128)
    parser.add_argument("--T", type=float, default=8.0)
    parser.add_argument("--time-degree", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument("--slabwise-steppers", action="store_true")
    args = parser.parse_args()
    petsc_options = PETSc.Options()
    for action in parser._actions:
        for flag in action.option_strings:
            if flag.startswith("--"):
                petsc_options.delValue(flag)
    result = solve_static_primal(StaticCylinderParameters(
        hierarchy_levels=args.lvls, time_steps=args.nt, final_time=args.T,
        time_degree=args.time_degree, report_every=args.report_every,
        slabwise_steppers=args.slabwise_steppers,
    ))
    PETSc.Sys.Print(
        f"[STATIC PRIMAL] saved {len(result['slabs']) - 1} in-memory dG slabs; "
        f"final drag={result['drag'][-1]:+.6e}, mean drag={mean_drag(result):+.6e}, "
        f"final div={result['divergence'][-1]:.3e}."
    )


if __name__ == "__main__":
    main()
