r"""Physical data and mean-drag goals for DFG 2D-3.

The production quantity of interest is the surface-traction mean drag used
in R5,

.. math::

   J_{\rm drag}(U)=\frac{20}{T}\int_0^T\int_{\Gamma_{\rm circle}}
   \sigma(U)n_{\rm obstacle}\cdot e_1\,ds\,dt.

Firedrake's boundary normal on the hole points out of the fluid and is the
negative of ``n_obstacle``; :func:`drag_coefficient_form` therefore contains
one leading minus sign.  The variational volume drag of Bruchhäuser,
Margenberg and Bause (2026), Eq. (5.9),

.. math::

   J_{\rm drag}(u)=-\int_I\{(\partial_t v,\widehat\psi_d)
   +a(u)(\widehat\phi_d)\}\,dt,

where ``widehat(phi)_d=(widehat(psi)_d,0)`` and the velocity lift equals
``(20/T, 0)`` on the cylinder and zero on the remaining boundary.  In the
implementation the factor ``20`` is placed in the spatial lift and ``1/T``
is applied after time integration.  This is the same physical mean drag as
the boundary-traction expression for the exact solution, but it is a
different functional on a discrete solution and must therefore be used
consistently in the goal evaluation and adjoint right-hand side.

is retained as an explicitly named diagnostic.  The two functionals agree
for an exact solution but are different on a discrete solution and must not
share an adjoint or reference value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from firedrake import (
    Constant,
    DirichletBC,
    FacetNormal,
    Function,
    VectorFunctionSpace,
    assemble,
    as_vector,
    div,
    dot,
    ds,
    dx,
    grad,
    inner,
    split,
)

from automated_DWR.time_solver import gauss_rule
from navier_stokes_cylinder_irksome_static_primal import (
    StaticCylinderParameters,
    evaluate_slab,
    evaluate_slab_dt,
    mean_drag as legacy_mean_drag,
    solve_static_primal,
)


@dataclass(frozen=True)
class R5CylinderSpecification:
    """DFG 2D-3 physical data and the R5 surface-drag reference."""

    final_time: float = 8.0
    viscosity: float = 1.0e-3
    drag_scale: float = 20.0
    nonlinear_mean_drag_reference: float = 1.6031368
    channel_length: float = 2.2
    channel_height: float = 0.41
    cylinder_centre: tuple[float, float] = (0.2, 0.2)
    cylinder_radius: float = 0.05


def _state_fields(state):
    fields = split(state)
    if len(fields) != 2:
        raise ValueError("The cylinder state must contain velocity and pressure.")
    return fields


def variational_drag_lift(
    mixed_space,
    cylinder_labels,
    *,
    drag_scale: float = 20.0,
    name: str = "variational_drag_lift",
) -> Function:
    r"""Build the fixed CG2 velocity lift ``widehat(psi)_d`` on one mesh.

    The velocity function is initially zero, so applying the cylinder
    Dirichlet values produces a finite-element extension that is zero on all
    other boundaries and in unconstrained degrees of freedom.  The choice of
    interior extension is immaterial for the exact solution but is part of
    the discrete functional.  Always constructing it in CG2 ensures the same
    lift is used for low and enriched states on a given mesh; it does not
    silently change when the adjoint polynomial degree changes.
    """
    # A mixed Firedrake space may expose a MeshSequenceGeometry through
    # ``mixed_space.mesh()``.  Its velocity subspace always carries the
    # concrete slab mesh required to construct an independent CG2 lift.
    mesh = mixed_space.sub(0).mesh()
    velocity_space = VectorFunctionSpace(mesh, "CG", 2)
    lift = Function(velocity_space, name=name)
    lift.assign(0.0)
    DirichletBC(
        velocity_space,
        Constant((float(drag_scale), 0.0)),
        tuple(cylinder_labels),
    ).apply(lift)
    return lift


def variational_drag_integrand_form(state, state_dt, lift, viscosity, *, measure=dx):
    r"""Return the unnormalised integrand in the 2026 drag functional.

    This includes the leading minus sign in Eq. (5.9), but not the final
    division by the time horizon.
    """
    velocity, pressure = _state_fields(state)
    velocity_dt = _state_fields(state_dt)[0]
    lift_velocity = lift
    return -(
        inner(velocity_dt, lift_velocity)
        + inner(dot(grad(velocity), velocity), lift_velocity)
        + viscosity * inner(grad(velocity), grad(lift_velocity))
        - pressure * div(lift_velocity)
    ) * measure


def variational_drag_spatial_derivative_form(
    state,
    velocity_variation,
    pressure_variation,
    lift,
    viscosity,
    *,
    weight=1.0,
    measure=dx,
):
    r"""Return the spatial part of ``A'(u)(delta u,widehat phi_d)``.

    The derivative of the goal itself is the negative of this form together
    with ``-(partial_t delta v,widehat psi_d)``.  The reverse adjoint residual
    contains ``A'-J'`` and therefore adds the form returned here.
    """
    velocity = _state_fields(state)[0]
    lift_velocity = lift
    return weight * (
        inner(
            dot(grad(velocity_variation), velocity)
            + dot(grad(velocity), velocity_variation),
            lift_velocity,
        )
        + viscosity * inner(grad(velocity_variation), grad(lift_velocity))
        - pressure_variation * div(lift_velocity)
    ) * measure


def drag_coefficient_form(
    state,
    viscosity,
    cylinder_labels,
    *,
    drag_scale: float = 20.0,
):
    """Return the instantaneous R5 surface-traction drag coefficient."""
    velocity, pressure = _state_fields(state)
    normal = FacetNormal(state.function_space().mesh())
    stress_times_fluid_normal = -pressure * normal + viscosity * dot(grad(velocity), normal)
    return (
        -float(drag_scale)
        * dot(stress_times_fluid_normal, as_vector((1.0, 0.0)))
        * ds(cylinder_labels)
    )


def drag_derivative_form(
    velocity_variation,
    pressure_variation,
    viscosity,
    cylinder_labels,
    *,
    drag_scale: float = 20.0,
    weight=1.0,
):
    """Frechet derivative of :func:`drag_coefficient_form`.

    The drag functional is linear in ``(u,p)``, so this derivative is state
    independent and can be used directly as the distributed adjoint source.
    """
    mesh = velocity_variation.ufl_domain()
    normal = FacetNormal(mesh)
    stress_variation_times_fluid_normal = (
        -pressure_variation * normal
        + viscosity * dot(grad(velocity_variation), normal)
    )
    return (
        -float(drag_scale)
        * weight
        * dot(stress_variation_times_fluid_normal, as_vector((1.0, 0.0)))
        * ds(cylinder_labels)
    )


def variational_mean_drag_from_history(
    primal: dict[str, Any],
    *,
    quadrature_points: int = 4,
    drag_scale: float = 20.0,
) -> float:
    """Evaluate the 2026 variational volume drag for diagnostics only."""
    times = primal["times"]
    horizon = float(times[-1] - times[0])
    if horizon <= 0.0:
        raise ValueError("The primal history must span a positive time interval.")
    total = 0.0
    for slab_number in range(1, len(times)):
        step = float(times[slab_number] - times[slab_number - 1])
        slab = primal["slabs"][slab_number]
        lift = variational_drag_lift(
            slab["coeffs"][0].function_space(),
            primal["labels"]["cylinder"],
            drag_scale=drag_scale,
            name=f"variational_drag_lift_{slab_number}",
        )
        for point, weight in gauss_rule(quadrature_points):
            state = evaluate_slab(
                slab, point, name=f"U_variational_drag_slab_{slab_number}",
            )
            state_dt = evaluate_slab_dt(
                slab,
                point,
                step,
                name=f"U_t_variational_drag_slab_{slab_number}",
            )
            form = variational_drag_integrand_form(
                state, state_dt, lift, primal["viscosity"]
            )
            total += step * float(weight) * float(assemble(form))
    return total / horizon


def mean_drag_from_history(
    primal: dict[str, Any],
    *,
    quadrature_points: int = 4,
    drag_scale: float = 20.0,
) -> float:
    r"""Evaluate the R5 surface mean drag on the saved dG trajectory.

    This is the same UFL functional whose derivative is returned by
    :func:`drag_derivative_form`; keeping both definitions here prevents a
    goal/adjoint mismatch.
    """
    times = np.asarray(primal["times"], dtype=float)
    horizon = float(times[-1] - times[0])
    if horizon <= 0.0:
        raise ValueError("The primal history must span a positive time interval.")
    total = 0.0
    for slab_number in range(1, len(times)):
        step = float(times[slab_number] - times[slab_number - 1])
        slab = primal["slabs"][slab_number]
        for point, weight in gauss_rule(int(quadrature_points)):
            state = evaluate_slab(
                slab, point, name=f"U_r5_surface_drag_slab_{slab_number}"
            )
            total += step * float(weight) * float(assemble(
                drag_coefficient_form(
                    state,
                    primal["viscosity"],
                    primal["labels"]["cylinder"],
                    drag_scale=drag_scale,
                )
            ))
    return total / horizon


def variational_drag_history_diagnostics(
    primal: dict[str, Any],
    *,
    quadrature_points: int = 4,
    drag_scale: float = 20.0,
) -> dict[str, float]:
    r"""Report the four contributions to the variational diagnostic."""
    times = np.asarray(primal["times"], dtype=float)
    horizon = float(times[-1] - times[0])
    viscosity = primal["viscosity"]

    def components(state, state_dt, lift):
        velocity, pressure = _state_fields(state)
        velocity_dt = _state_fields(state_dt)[0]
        lift_velocity = lift
        temporal_part = -inner(velocity_dt, lift_velocity) * dx
        convective_part = -inner(
            dot(grad(velocity), velocity), lift_velocity
        ) * dx
        viscous_part = -viscosity * inner(
            grad(velocity), grad(lift_velocity)
        ) * dx
        pressure_part = pressure * div(lift_velocity) * dx
        temporal_value = float(assemble(temporal_part))
        convective_value = float(assemble(convective_part))
        pressure_value = float(assemble(pressure_part))
        viscous_value = float(assemble(viscous_part))
        return temporal_value, convective_value, pressure_value, viscous_value

    polynomial_values: list[float] = []
    temporal_total = 0.0
    convective_total = 0.0
    pressure_total = 0.0
    viscous_total = 0.0
    for n in range(1, len(times)):
        step = float(times[n] - times[n - 1])
        slab = primal["slabs"][n]
        lift = variational_drag_lift(
            slab["coeffs"][0].function_space(),
            primal["labels"]["cylinder"],
            drag_scale=drag_scale,
            name=f"variational_drag_audit_lift_{n}",
        )
        for point, weight in gauss_rule(quadrature_points):
            state = evaluate_slab(
                slab, point, name=f"U_drag_audit_slab_{n}"
            )
            state_dt = evaluate_slab_dt(
                slab, point, step, name=f"U_t_drag_audit_slab_{n}"
            )
            values = components(state, state_dt, lift)
            temporal_value, convective_value, pressure_value, viscous_value = values
            polynomial_values.append(sum(values))
            temporal_total += step * float(weight) * temporal_value
            convective_total += step * float(weight) * convective_value
            pressure_total += step * float(weight) * pressure_value
            viscous_total += step * float(weight) * viscous_value

    endpoint_values_list: list[float] = []
    for n in range(1, len(times)):
        slab = primal["slabs"][n]
        step = float(times[n] - times[n - 1])
        lift = variational_drag_lift(
            slab["coeffs"][0].function_space(),
            primal["labels"]["cylinder"],
            drag_scale=drag_scale,
            name=f"variational_drag_endpoint_lift_{n}",
        )
        if n == 1:
            state = evaluate_slab(slab, 0.0, name="U_drag_endpoint_0")
            state_dt = evaluate_slab_dt(
                slab, 0.0, step, name="U_t_drag_endpoint_0"
            )
            endpoint_values_list.append(sum(components(state, state_dt, lift)))
        state = evaluate_slab(slab, 1.0, name=f"U_drag_endpoint_{n}")
        state_dt = evaluate_slab_dt(
            slab, 1.0, step, name=f"U_t_drag_endpoint_{n}"
        )
        endpoint_values_list.append(sum(components(state, state_dt, lift)))
    endpoint_values = np.asarray(endpoint_values_list, dtype=float)
    trapezoid = float(np.trapezoid(endpoint_values, times) / horizon)
    polynomial_values_array = np.asarray(polynomial_values, dtype=float)
    return {
        "drag_temporal_mean": temporal_total / horizon,
        "drag_convective_mean": convective_total / horizon,
        "drag_pressure_mean": pressure_total / horizon,
        "drag_viscous_mean": viscous_total / horizon,
        "drag_polynomial_min": float(np.min(polynomial_values_array)),
        "drag_polynomial_max": float(np.max(polynomial_values_array)),
        "drag_endpoint_min": float(np.min(endpoint_values)),
        "drag_endpoint_max": float(np.max(endpoint_values)),
        "drag_endpoint_trapezoid_mean": trapezoid,
    }


def drag_history_diagnostics(
    primal: dict[str, Any],
    *,
    quadrature_points: int = 4,
    drag_scale: float = 20.0,
) -> dict[str, float]:
    """Report pressure/viscous contributions to the R5 surface drag."""
    times = np.asarray(primal["times"], dtype=float)
    horizon = float(times[-1] - times[0])
    if horizon <= 0.0:
        raise ValueError("The primal history must span a positive time interval.")
    pressure_total = 0.0
    viscous_total = 0.0
    polynomial_values: list[float] = []

    def components(state):
        velocity, pressure = _state_fields(state)
        normal = FacetNormal(state.function_space().mesh())
        direction = as_vector((1.0, 0.0))
        pressure_form = (
            float(drag_scale) * pressure * dot(normal, direction)
            * ds(primal["labels"]["cylinder"])
        )
        viscous_form = (
            -float(drag_scale)
            * primal["viscosity"]
            * dot(dot(grad(velocity), normal), direction)
            * ds(primal["labels"]["cylinder"])
        )
        return float(assemble(pressure_form)), float(assemble(viscous_form))

    for n in range(1, len(times)):
        step = float(times[n] - times[n - 1])
        slab = primal["slabs"][n]
        for point, weight in gauss_rule(int(quadrature_points)):
            state = evaluate_slab(slab, point, name=f"U_r5_drag_audit_slab_{n}")
            pressure_value, viscous_value = components(state)
            polynomial_values.append(pressure_value + viscous_value)
            pressure_total += step * float(weight) * pressure_value
            viscous_total += step * float(weight) * viscous_value

    endpoint_values: list[float] = []
    for n in range(1, len(times)):
        slab = primal["slabs"][n]
        if n == 1:
            endpoint_values.append(sum(components(evaluate_slab(
                slab, 0.0, name="U_r5_drag_endpoint_0"
            ))))
        endpoint_values.append(sum(components(evaluate_slab(
            slab, 1.0, name=f"U_r5_drag_endpoint_{n}"
        ))))
    endpoints = np.asarray(endpoint_values, dtype=float)
    samples = np.asarray(polynomial_values, dtype=float)
    return {
        "drag_pressure_mean": pressure_total / horizon,
        "drag_viscous_mean": viscous_total / horizon,
        "drag_polynomial_min": float(np.min(samples)),
        "drag_polynomial_max": float(np.max(samples)),
        "drag_endpoint_min": float(np.min(endpoints)),
        "drag_endpoint_max": float(np.max(endpoints)),
        "drag_endpoint_trapezoid_mean": float(
            np.trapezoid(endpoints, times) / horizon
        ),
    }


def compare_drag_conventions(primal: dict[str, Any]) -> dict[str, float]:
    """Compare R5 surface, legacy surface, and variational conventions."""
    surface_value = mean_drag_from_history(primal)
    legacy_surface_value = float(legacy_mean_drag(primal))
    variational_value = variational_mean_drag_from_history(primal)
    return {
        "variational_mean_drag": variational_value,
        "surface_mean_drag": surface_value,
        "legacy_surface_mean_drag": legacy_surface_value,
        "difference": variational_value - surface_value,
        "surface_implementation_difference": surface_value - legacy_surface_value,
    }


def solve_primal(
    *,
    hierarchy_levels: int = 1,
    time_steps: int = 128,
    final_time: float = 8.0,
    time_degree: int = 1,
    viscosity: float = 1.0e-3,
    report_every: int = 16,
    store_stages: bool = True,
    slabwise_steppers: bool = False,
) -> dict[str, Any]:
    """Call the verified AS/Alfeld primal without changing its algorithm."""
    parameters = StaticCylinderParameters(
        hierarchy_levels=int(hierarchy_levels),
        time_steps=int(time_steps),
        final_time=float(final_time),
        time_degree=int(time_degree),
        viscosity=float(viscosity),
        store_stages=bool(store_stages),
        report_every=int(report_every),
        slabwise_steppers=bool(slabwise_steppers),
    )
    return solve_static_primal(parameters)


__all__ = [
    "R5CylinderSpecification",
    "compare_drag_conventions",
    "drag_coefficient_form",
    "drag_derivative_form",
    "drag_history_diagnostics",
    "mean_drag_from_history",
    "variational_drag_history_diagnostics",
    "variational_mean_drag_from_history",
    "solve_primal",
    "variational_drag_integrand_form",
    "variational_drag_lift",
    "variational_drag_spatial_derivative_form",
]
