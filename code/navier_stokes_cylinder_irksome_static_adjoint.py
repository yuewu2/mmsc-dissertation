r"""Reverse adjoints for the stable static cylinder primal.

This first adjoint layer enriches *time only*: the Alfeld--Sorokina spatial
space is held fixed and the reverse dual can use dG(2) while the primal uses
dG(1).  It is deliberately a global temporal-DWR building block.  A spatially
enriched dual must be designed separately for the Alfeld--Sorokina pair before
claiming bubble/PU spatial localisation.
"""

from __future__ import annotations

import argparse
import gc
from typing import Any

from firedrake import Constant, DirichletBC, FacetNormal, Function, FunctionSpace, TestFunctions, VectorFunctionSpace, as_vector, div, dot, ds, dx, grad, inner, split
from firedrake.petsc import PETSc
from irksome import DiscontinuousGalerkinScheme, Dt, TimeStepper

from automated_DWR.time_solver import linear_combination
from navier_stokes_cylinder_irksome_static_primal import _stage_coefficients
from navier_stokes_cylinder_irksome_static_primal import _irksome_dg_nodes


def _saved_primal_at(primal_slab: dict[str, Any], physical_time, left: float, step: float):
    """Return symbolic-in-time velocity and pressure from saved dG coefficients."""
    degree = int(primal_slab["degree"])
    reference_time = (physical_time - Constant(float(left))) / Constant(float(step))
    nodes = _irksome_dg_nodes(degree)
    basis = []
    for i, node_i in enumerate(nodes):
        value = 1.0
        for j, node_j in enumerate(nodes):
            if i != j:
                value *= (reference_time - float(node_j)) / (float(node_i) - float(node_j))
        basis.append(value)
    return (
        linear_combination([coefficient.subfunctions[0] for coefficient in primal_slab["coeffs"]], basis),
        linear_combination([coefficient.subfunctions[1] for coefficient in primal_slab["coeffs"]], basis),
    )


def _linear_solver_parameters(primal_parameters, degree: int) -> dict[str, Any]:
    """Reuse the verified monolithic hierarchy with a linear outer solve."""
    from navier_stokes_cylinder_irksome_static_primal import _solver_parameters

    parameters = _solver_parameters(degree)
    parameters["snes_type"] = "ksponly"
    return parameters


def _mean_drag_derivative(velocity_variation, pressure_variation, viscosity, normal, cylinder_labels):
    """Derivative of the exact drag convention used by ``code/cylinder.py``."""
    tangent_variation = dot(velocity_variation, as_vector((normal[1], -normal[0])))
    force_variation = (
        viscosity * dot(grad(tangent_variation), normal) * normal[1]
        - pressure_variation * normal[0]
    ) * ds(cylinder_labels)
    return -20.0 * force_variation


def _reverse_dual_residual(
    dual, tests, reverse_time, primal_slab, left: float, step: float,
    physical_final_time: float, qoi_final_time: float, viscosity, mesh, cylinder_labels,
):
    """Forward-in-tau form of ``A'(U)(delta U, Z) - J'(U)(delta U)``.

    ``tau = T-t``.  The last continuity term has a minus sign because the
    stable Section 5.3 primal uses ``-(div(u), q)``.
    """
    dual_velocity, dual_pressure = split(dual)
    velocity_variation, pressure_variation = tests
    physical_time = Constant(float(physical_final_time)) - reverse_time
    primal_velocity, _ = _saved_primal_at(primal_slab, physical_time, left, step)
    return (
        inner(Dt(dual_velocity), velocity_variation) * dx
        + inner(
            dot(grad(velocity_variation), primal_velocity)
            + dot(grad(primal_velocity), velocity_variation),
            dual_velocity,
        ) * dx
        + viscosity * inner(grad(velocity_variation), grad(dual_velocity)) * dx
        - pressure_variation * div(dual_velocity) * dx
        - div(velocity_variation) * dual_pressure * dx
        - (1.0 / float(qoi_final_time))
        * _mean_drag_derivative(
            velocity_variation, pressure_variation, viscosity, FacetNormal(mesh), cylinder_labels
        )
    )


def solve_static_adjoint(
    primal: dict[str, Any], degree: int = 2, report_every: int = 16,
    terminal_state: Function | None = None, qoi_final_time: float | None = None,
) -> dict[str, Any]:
    """Solve the mean-drag adjoint backwards on the static primal mesh.

    ``degree=1`` supplies the numerical dual and ``degree=2`` supplies the
    time-enriched dual used in the first global temporal DWR comparison.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if not primal["parameters"].store_stages:
        raise ValueError("The primal must be solved with store_stages=True.")
    mesh = primal["mesh"]
    mixed_space = primal["mixed_space"]
    times = primal["times"]
    state = Function(mixed_space, name=f"Z_dg{degree}")
    essential = tuple(primal["labels"]["inlet"]) + tuple(primal["labels"]["wall"])
    bcs = [DirichletBC(mixed_space.sub(0), Constant((0.0, 0.0)), essential)]
    for bc in bcs:
        bc.apply(state)
    slabs: list[dict[str, Any] | None] = [None] * len(times)
    physical_final_time = float(times[-1])
    if terminal_state is not None:
        if terminal_state.function_space() != mixed_space:
            raise ValueError("terminal_state must belong to the primal mixed space.")
        state.assign(terminal_state)
        for bc in bcs:
            bc.apply(state)
    if qoi_final_time is None:
        qoi_final_time = physical_final_time
    # The initial Section 5.3 mesh carries a nested hierarchy and can use the
    # verified Vanka multigrid.  A mesh produced by marked refinement cannot:
    # it must use the same robust direct linear solve as the CG4/CG2 dual.
    solver_parameters = (
        _linear_solver_parameters(primal["parameters"], degree)
        if primal["hierarchy"] is not None
        else _direct_linear_parameters()
    )
    for slab_number in range(len(times) - 1, 0, -1):
        step = float(times[slab_number] - times[slab_number - 1])
        reverse_time = Constant(physical_final_time - float(times[slab_number]))
        residual = _reverse_dual_residual(
            state, TestFunctions(mixed_space), reverse_time, primal["slabs"][slab_number],
            float(times[slab_number - 1]), step, physical_final_time, float(qoi_final_time),
            primal["viscosity"], mesh,
            primal["labels"]["cylinder"],
        )
        stepper = TimeStepper(
            residual, DiscontinuousGalerkinScheme(int(degree)), reverse_time, Constant(step), state,
            bcs=bcs, solver_parameters=solver_parameters,
        )
        stepper.advance()
        slabs[slab_number] = {
            "degree": int(degree),
            "coeffs": _stage_coefficients(stepper, mixed_space, degree, f"Z_slab_{slab_number}"),
        }
        # Each slab owns an assembled PETSc operator.  Retain only its copied
        # dG coefficients: otherwise a long reverse solve can retain every
        # previous solver/factorisation until WSL's OOM killer intervenes.
        del stepper
        gc.collect()
        if report_every and (
            slab_number == len(times) - 1 or slab_number % int(report_every) == 0 or slab_number == 1
        ):
            PETSc.Sys.Print(
                f"[STATIC ADJOINT dG{degree}] solved slab {slab_number}/{len(times) - 1} "
                f"at physical t={times[slab_number - 1]:.6g}."
            )
    return {"degree": int(degree), "slabs": slabs, "final_state": state}


def _spatially_enriched_space(mesh):
    """CG4/CG2 Taylor--Hood candidate for the first spatially rich dual.

    The primal remains the Section 5.3 Alfeld--Sorokina pair.  This is not
    assumed to be a nested injection; the numerical AS/Alfeld dual will be
    explicitly interpolated when forming the DWR weight.
    """
    return VectorFunctionSpace(mesh, "CG", 4) * FunctionSpace(mesh, "CG", 2)


def _direct_linear_parameters() -> dict[str, Any]:
    """Robust first solve for the new CG4/CG2 adjoint space.

    A Vanka hierarchy for this different pair should be designed only after
    the adjoint and estimator are verified.  On the initial static meshes,
    direct MUMPS isolates that algebraic optimisation from correctness.
    """
    return {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


def solve_static_spatially_enriched_adjoint(
    primal: dict[str, Any], time_degree: int = 2, report_every: int = 16,
    qoi_final_time: float | None = None,
) -> dict[str, Any]:
    """Compute a CG4/CG2, dG(time_degree) reverse adjoint on the static mesh.

    This is the first *candidate* spatial enrichment.  It is intentionally
    verified first on a short horizon and with a direct solve; it has not yet
    been connected to bubble/PU localisation or marked refinement.
    """
    if time_degree < 0:
        raise ValueError("time_degree must be non-negative.")
    if not primal["parameters"].store_stages:
        raise ValueError("The primal must be solved with store_stages=True.")
    mesh = primal["mesh"]
    mixed_space = _spatially_enriched_space(mesh)
    times = primal["times"]
    state = Function(mixed_space, name=f"Z_CG4CG2_dg{time_degree}")
    essential = tuple(primal["labels"]["inlet"]) + tuple(primal["labels"]["wall"])
    bcs = [DirichletBC(mixed_space.sub(0), Constant((0.0, 0.0)), essential)]
    for bc in bcs:
        bc.apply(state)
    slabs: list[dict[str, Any] | None] = [None] * len(times)
    physical_final_time = float(times[-1])
    if qoi_final_time is None:
        qoi_final_time = physical_final_time
    for slab_number in range(len(times) - 1, 0, -1):
        step = float(times[slab_number] - times[slab_number - 1])
        reverse_time = Constant(physical_final_time - float(times[slab_number]))
        residual = _reverse_dual_residual(
            state, TestFunctions(mixed_space), reverse_time, primal["slabs"][slab_number],
            float(times[slab_number - 1]), step, physical_final_time, float(qoi_final_time),
            primal["viscosity"], mesh,
            primal["labels"]["cylinder"],
        )
        stepper = TimeStepper(
            residual, DiscontinuousGalerkinScheme(int(time_degree)), reverse_time, Constant(step), state,
            bcs=bcs, solver_parameters=_direct_linear_parameters(),
        )
        stepper.advance()
        slabs[slab_number] = {
            "degree": int(time_degree),
            "coeffs": _stage_coefficients(stepper, mixed_space, time_degree, f"Zrich_slab_{slab_number}"),
        }
        # MUMPS factorisations are the dominant memory consumers here.  The
        # saved coefficient Functions above are sufficient for DWR, so make
        # the stepper and its assembled matrices collectible immediately.
        del stepper
        gc.collect()
        if report_every and (
            slab_number == len(times) - 1 or slab_number % int(report_every) == 0 or slab_number == 1
        ):
            PETSc.Sys.Print(
                f"[STATIC SPATIAL-RICH ADJOINT] solved slab {slab_number}/{len(times) - 1} "
                f"at physical t={times[slab_number - 1]:.6g}."
            )
    return {"degree": int(time_degree), "slabs": slabs, "final_state": state, "mixed_space": mixed_space}


def main() -> None:
    """Small executable check before adding the global estimator."""
    from navier_stokes_cylinder_irksome_static_primal import StaticCylinderParameters, solve_static_primal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lvls", type=int, default=1)
    parser.add_argument("--nt", type=int, default=4)
    parser.add_argument("--T", type=float, default=0.25)
    parser.add_argument("--dual-degree", type=int, default=2)
    parser.add_argument(
        "--spatial-rich", action="store_true",
        help="Solve the candidate CG4/CG2 enriched adjoint instead of the AS/Alfeld adjoint.",
    )
    args = parser.parse_args()
    petsc_options = PETSc.Options()
    for action in parser._actions:
        for flag in action.option_strings:
            if flag.startswith("--"):
                petsc_options.delValue(flag)
    primal = solve_static_primal(StaticCylinderParameters(
        hierarchy_levels=args.lvls, time_steps=args.nt, final_time=args.T, report_every=args.nt,
    ))
    solve = solve_static_spatially_enriched_adjoint if args.spatial_rich else solve_static_adjoint
    dual = solve(primal, time_degree=args.dual_degree, report_every=args.nt) if args.spatial_rich else solve(
        primal, degree=args.dual_degree, report_every=args.nt
    )
    PETSc.Sys.Print(
        f"[STATIC ADJOINT] completed {'CG4/CG2 ' if args.spatial_rich else ''}dG({dual['degree']}) "
        f"reverse solve over {len(dual['slabs']) - 1} slabs."
    )


if __name__ == "__main__":
    main()
