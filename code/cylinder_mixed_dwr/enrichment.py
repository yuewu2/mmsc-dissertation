r"""Spatio-temporally enriched primal reconstruction on the static mesh."""

from __future__ import annotations

import gc
from dataclasses import replace
from typing import Any

from firedrake import Constant, Function, TestFunctions
from irksome import DiscontinuousGalerkinScheme, TimeStepper

from navier_stokes_cylinder_irksome_static_adjoint import (
    _spatially_enriched_space,
)
from navier_stokes_cylinder_irksome_static_primal import (
    _copy,
    _direct_newton_parameters,
    _primal_residual,
)

from .adapter import CylinderMixedDAEAdapter


def solve_enriched_primal(
    primal: dict[str, Any],
    *,
    time_degree: int | None = None,
    report_every: int = 16,
) -> dict[str, Any]:
    """Solve CG4/CG2 on the same mesh and time partition as ``primal``.

    The enriched solve is an estimator reconstruction, not a replacement for
    the verified AS/Alfeld primal trajectory.  A direct Newton/MUMPS solve is
    used because the AS Vanka hierarchy is not valid for CG4/CG2.
    """
    degree = (
        int(primal["parameters"].time_degree) + 1
        if time_degree is None
        else int(time_degree)
    )
    if degree <= int(primal["parameters"].time_degree):
        raise ValueError("The enriched primal time degree must exceed the primal degree.")
    mesh = primal["mesh"]
    mixed_space = _spatially_enriched_space(mesh)
    adapter = CylinderMixedDAEAdapter(
        primal["labels"], viscosity=float(primal["viscosity"])
    )
    state = Function(mixed_space, name=f"U_rich_dG{degree}")
    times = primal["times"]
    time = Constant(float(times[0]))
    bcs = adapter.primal_boundary_conditions(mixed_space, time)
    for bc in bcs:
        bc.apply(state)
    slabs: list[dict[str, Any] | None] = [None] * len(times)
    solver_parameters = _direct_newton_parameters()

    for slab_number in range(1, len(times)):
        step = float(times[slab_number] - times[slab_number - 1])
        left_trace = _copy(state, f"U_rich_left_{slab_number}")
        stepper = TimeStepper(
            _primal_residual(
                state, TestFunctions(mixed_space), primal["viscosity"]
            ),
            DiscontinuousGalerkinScheme(
                degree, quadrature_degree=3 * degree
            ),
            time,
            Constant(step),
            state,
            bcs=bcs,
            solver_parameters=solver_parameters,
        )
        stepper.advance()
        time.assign(float(times[slab_number]))
        slabs[slab_number] = {
            "degree": degree,
            "coeffs": adapter.pack_irksome_stages(
                stepper,
                mixed_space,
                degree,
                f"U_rich_slab_{slab_number}",
            ),
            "left_trace": left_trace,
            "right_trace": _copy(state, f"U_rich_right_{slab_number}"),
        }
        del stepper
        gc.collect()
        if report_every and (
            slab_number == 1
            or slab_number == len(times) - 1
            or slab_number % int(report_every) == 0
        ):
            from firedrake.petsc import PETSc

            PETSc.Sys.Print(
                f"[MIXED ENRICHED PRIMAL] slab {slab_number}/{len(times) - 1}."
            )
    return {
        "parameters": replace(primal["parameters"], time_degree=degree),
        "hierarchy": None,
        "mesh": mesh,
        "mixed_space": mixed_space,
        "viscosity": primal["viscosity"],
        "times": times,
        "slabs": slabs,
        "labels": primal["labels"],
        "degree": degree,
        "enrichment": "CG4/CG2 x dG(primal_degree+1)",
    }


__all__ = ["solve_enriched_primal"]
