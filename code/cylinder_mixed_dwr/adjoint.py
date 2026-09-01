r"""Low and enriched reverse adjoints for the mixed cylinder DAE.

The implementation is isolated from ``nonstationary_dwr``.  It reuses the
verified AS/Alfeld solver parameters and Irksome stage packing.  The goal
derivative is the R5 surface-traction mean drag.  Only velocity carries a
reverse time derivative; dual pressure is algebraic on every slab.
"""

from __future__ import annotations

import gc
from typing import Any

from firedrake import Constant, Function, TestFunctions, div, dot, dx, grad, inner, split
from firedrake.petsc import PETSc
from irksome import DiscontinuousGalerkinScheme, Dt, TimeStepper

from navier_stokes_cylinder_irksome_static_adjoint import (
    _direct_linear_parameters,
    _linear_solver_parameters,
    _saved_primal_at,
    _spatially_enriched_space,
)

from .adapter import CylinderMixedDAEAdapter
from .benchmark import drag_derivative_form


def reverse_adjoint_residual(
    dual,
    variations,
    reverse_time,
    primal_slab,
    *,
    slab_left: float,
    step: float,
    final_time: float,
    goal_horizon: float,
    viscosity,
    cylinder_labels,
):
    r"""Return the forward-in-``tau=T-t`` mixed adjoint residual.

    This is ``A'(U_h)(delta U,Z)-J'(U_h)(delta U)`` after reversing time.
    The continuity signs match the stable primal form
    ``-(div(u),q)``.
    """
    dual_velocity, dual_pressure = split(dual)
    velocity_variation, pressure_variation = variations
    physical_time = Constant(float(final_time)) - reverse_time
    primal_velocity, _ = _saved_primal_at(
        primal_slab, physical_time, float(slab_left), float(step)
    )
    residual = (
        inner(Dt(dual_velocity), velocity_variation) * dx
        + inner(
            dot(grad(velocity_variation), primal_velocity)
            + dot(grad(primal_velocity), velocity_variation),
            dual_velocity,
        ) * dx
        + viscosity * inner(grad(velocity_variation), grad(dual_velocity)) * dx
        - pressure_variation * div(dual_velocity) * dx
        - div(velocity_variation) * dual_pressure * dx
    )
    # The reverse adjoint equation is A'(U_h)(delta U, Z)-J'(U_h)(delta U)=0.
    # R5's surface drag is linear in velocity/pressure, so its derivative is
    # state independent and has no temporal endpoint contribution.
    residual -= (1.0 / float(goal_horizon)) * drag_derivative_form(
        velocity_variation,
        pressure_variation,
        viscosity,
        cylinder_labels,
    )
    return residual


def solve_adjoint(
    primal: dict[str, Any],
    *,
    time_degree: int,
    spatially_enriched: bool,
    report_every: int = 16,
) -> dict[str, Any]:
    """March the R5 surface-mean-drag adjoint backwards over saved slabs."""
    if int(time_degree) < 0:
        raise ValueError("time_degree must be non-negative.")
    if not primal["parameters"].store_stages:
        raise ValueError("The primal must retain its dG stage coefficients.")
    mesh = primal["mesh"]
    mixed_space = (
        _spatially_enriched_space(mesh)
        if spatially_enriched
        else primal["mixed_space"]
    )
    adapter = CylinderMixedDAEAdapter(
        primal["labels"], viscosity=float(primal["viscosity"])
    )
    state = Function(
        mixed_space,
        name=("Z_rich" if spatially_enriched else "Z_low")
        + f"_dG{int(time_degree)}",
    )
    # The time-averaged surface drag has no terminal contribution.
    state.assign(0.0)
    bcs = adapter.adjoint_boundary_conditions(mixed_space)
    for bc in bcs:
        bc.apply(state)

    times = primal["times"]
    final_time = float(times[-1])
    horizon = float(times[-1] - times[0])
    if horizon <= 0.0:
        raise ValueError("The primal history must have a positive time horizon.")
    if spatially_enriched or primal["hierarchy"] is None:
        solver_parameters = _direct_linear_parameters()
    else:
        solver_parameters = _linear_solver_parameters(
            primal["parameters"], int(time_degree)
        )

    slabs: list[dict[str, Any] | None] = [None] * len(times)
    for slab_number in range(len(times) - 1, 0, -1):
        step = float(times[slab_number] - times[slab_number - 1])
        reverse_time = Constant(final_time - float(times[slab_number]))
        incoming_trace = Function(
            mixed_space, name=f"Z_reverse_incoming_slab_{slab_number}"
        )
        incoming_trace.assign(state)
        residual = reverse_adjoint_residual(
            state,
            TestFunctions(mixed_space),
            reverse_time,
            primal["slabs"][slab_number],
            slab_left=float(times[slab_number - 1]),
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
            bcs=bcs,
            solver_parameters=solver_parameters,
        )
        stepper.advance()
        slabs[slab_number] = {
            "degree": int(time_degree),
            "coeffs": adapter.pack_irksome_stages(
                stepper,
                mixed_space,
                int(time_degree),
                f"Z_{'rich' if spatially_enriched else 'low'}_slab_{slab_number}",
            ),
            # In reverse time this is the trace arriving from the physical
            # right of the slab.  Only its velocity participates in the dG
            # adjoint jump recovery.
            "incoming_trace": incoming_trace,
        }
        del stepper
        gc.collect()
        if report_every and (
            slab_number == len(times) - 1
            or slab_number == 1
            or slab_number % int(report_every) == 0
        ):
            PETSc.Sys.Print(
                f"[MIXED ADJOINT {'RICH' if spatially_enriched else 'LOW'}] "
                f"slab {slab_number}/{len(times) - 1}."
            )
    return {
        "degree": int(time_degree),
        "spatially_enriched": bool(spatially_enriched),
        "mixed_space": mixed_space,
        "slabs": slabs,
        "final_state": state,
    }


def solve_low_adjoint(primal: dict[str, Any], *, report_every: int = 16):
    """Solve in the primal AS/Alfeld space and primal dG time degree."""
    return solve_adjoint(
        primal,
        time_degree=int(primal["parameters"].time_degree),
        spatially_enriched=False,
        report_every=report_every,
    )


def solve_enriched_adjoint(
    primal: dict[str, Any], *, time_degree: int | None = None, report_every: int = 16
):
    """Solve in the configured enriched mixed space and higher dG degree."""
    degree = (
        int(primal["parameters"].time_degree) + 1
        if time_degree is None
        else int(time_degree)
    )
    return solve_adjoint(
        primal,
        time_degree=degree,
        spatially_enriched=True,
        report_every=report_every,
    )


__all__ = [
    "reverse_adjoint_residual",
    "solve_adjoint",
    "solve_enriched_adjoint",
    "solve_low_adjoint",
]
