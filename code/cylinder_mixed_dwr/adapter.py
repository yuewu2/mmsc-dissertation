r"""Mixed-DAE operations needed by the cylinder DWR implementation.

The existing scalar ``nonstationary_dwr`` solver is deliberately left alone.
This adapter makes the non-generic parts explicit:

* velocity is differential and pressure is algebraic;
* the time mass and dG jump contain velocity only;
* Irksome's flattened mixed stages are repacked by time mode;
* forward and adjoint mesh-interface transfers act on velocity only;
* primal pressure is reset to a zero initial guess after a mesh change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunctions, VectorFunctionSpace, as_vector, dx, inner, pi, sin, split,
)

from navier_stokes_cylinder_irksome_static_primal import (
    _copy,
    _primal_residual,
    _stage_coefficients,
)

from .transfer import (
    MassConsistentVelocityTransfer,
    StokesH1VelocityTransfer,
    StokesL2VelocityTransfer,
)


@dataclass(frozen=True)
class CylinderBoundaryLabels:
    inlet: tuple[int, ...]
    wall: tuple[int, ...]
    cylinder: tuple[int, ...]
    outlet: tuple[int, ...]

    @classmethod
    def from_mapping(cls, labels: dict[str, tuple[int, ...]]):
        required = {"inlet", "wall", "cylinder", "outlet"}
        missing = required.difference(labels)
        if missing:
            raise ValueError(f"Missing cylinder boundary labels: {sorted(missing)}")
        return cls(**{name: tuple(labels[name]) for name in sorted(required)})


class CylinderMixedDAEAdapter:
    """Standalone contract for the Taylor--Hood velocity-pressure system."""

    differential_field_indices = (0,)
    algebraic_field_indices = (1,)
    velocity_field = 0
    pressure_field = 1

    def __init__(
        self,
        labels: CylinderBoundaryLabels | dict[str, tuple[int, ...]],
        *,
        viscosity: float = 1.0e-3,
    ):
        self.labels = (
            labels
            if isinstance(labels, CylinderBoundaryLabels)
            else CylinderBoundaryLabels.from_mapping(labels)
        )
        self.viscosity = Constant(float(viscosity))

    @staticmethod
    def make_spaces(mesh):
        """Return the Taylor--Hood CG2/CG1 velocity-pressure spaces."""
        velocity = VectorFunctionSpace(mesh, "CG", 2)
        pressure = FunctionSpace(mesh, "CG", 1)
        return velocity, pressure, velocity * pressure

    def primal_residual(self, state, tests=None):
        """Return the unchanged Section 5.3 mixed Irksome residual."""
        if tests is None:
            tests = TestFunctions(state.function_space())
        return _primal_residual(state, tests, self.viscosity)

    @staticmethod
    def _test_fields(test):
        if isinstance(test, (tuple, list)):
            if len(test) != 2:
                raise ValueError("A mixed test must contain velocity and pressure parts.")
            return test
        fields = split(test)
        if len(fields) != 2:
            raise ValueError("A mixed test must contain velocity and pressure parts.")
        return fields

    @staticmethod
    def velocity_time_mass_action(trial_velocity, test_velocity, measure=dx):
        """The nonsingular mass pairing on the differential subspace."""
        return inner(trial_velocity, test_velocity) * measure

    def mixed_time_mass_action(self, trial, test, measure=dx):
        """Singular mixed mass form, exposed only as an action.

        Callers must never invert this mixed form.  Transfer and terminal
        operations use the velocity mass through
        :meth:`velocity_time_mass_action` instead.
        """
        trial_velocity = split(trial)[self.velocity_field]
        test_velocity = self._test_fields(test)[self.velocity_field]
        return self.velocity_time_mass_action(trial_velocity, test_velocity, measure)

    def temporal_residual_action(
        self, state_left, incoming_velocity, test, *, measure=dx,
    ):
        """Return ``-(u_left-P u_previous, v_left)`` with no pressure jump."""
        left_velocity = split(state_left)[self.velocity_field]
        test_velocity = self._test_fields(test)[self.velocity_field]
        if hasattr(incoming_velocity, "subfunctions") and len(incoming_velocity.subfunctions) == 2:
            incoming_velocity = incoming_velocity.subfunctions[self.velocity_field]
        return -inner(left_velocity - incoming_velocity, test_velocity) * measure

    def inflow_value(self, mesh, physical_time, *, mean=None):
        """R5 inflow ``(sin(pi*t/8)*6*y*(0.41-y)/0.41^2, 0)``."""
        y = SpatialCoordinate(mesh)[1]
        if mean is None:
            mean = 1.5 * sin(pi * physical_time / 8.0)
        return as_vector((4.0 * mean * y * (0.41 - y) / 0.41**2, 0.0))

    def primal_boundary_conditions(self, mixed_space, physical_time, *, inflow_mean=None):
        return [
            DirichletBC(
                mixed_space.sub(self.velocity_field),
                self.inflow_value(
                    mixed_space.mesh(), physical_time, mean=inflow_mean
                ),
                self.labels.inlet,
            ),
            DirichletBC(
                mixed_space.sub(self.velocity_field), Constant((0.0, 0.0)),
                self.labels.wall,
            ),
        ]

    def adjoint_boundary_conditions(self, mixed_space):
        essential = tuple(self.labels.inlet) + tuple(self.labels.wall)
        return [
            DirichletBC(
                mixed_space.sub(self.velocity_field), Constant((0.0, 0.0)), essential,
            )
        ]

    @staticmethod
    def pack_irksome_stages(stepper, mixed_space, degree: int, name: str):
        """Repack ``(u_0,p_0,u_1,p_1,...)`` as mixed time coefficients."""
        return _stage_coefficients(stepper, mixed_space, int(degree), name)

    @staticmethod
    def velocity_trace(state, name: str = "velocity_trace") -> Function:
        """Copy only the differential trace from a mixed state."""
        return _copy(state.subfunctions[0], name)

    def state_from_velocity(
        self,
        mixed_space,
        velocity,
        *,
        name: str = "mixed_state_from_velocity",
        physical_time=None,
    ) -> Function:
        """Create an incoming mixed state and reset algebraic pressure to zero."""
        state = Function(mixed_space, name=name)
        target_velocity, pressure = state.subfunctions
        if velocity.function_space() == target_velocity.function_space():
            target_velocity.assign(velocity)
        else:
            target_velocity.interpolate(velocity)
        pressure.assign(0.0)
        if physical_time is not None:
            for bc in self.primal_boundary_conditions(mixed_space, physical_time):
                bc.apply(state)
        return state

    def build_velocity_transfer(
        self, source_velocity_space, target_velocity_space, *, mode="mass",
        embedding_degree: int = 2, source_pressure_space=None,
        target_pressure_space=None,
    ):
        """Build a velocity-only interface operator and its L2 adjoint."""
        if mode == "mass":
            return MassConsistentVelocityTransfer(
                source_velocity_space,
                target_velocity_space,
                source_pressure_space=source_pressure_space,
                target_pressure_space=target_pressure_space,
            )
        if mode == "stokes_l2":
            if target_pressure_space is None:
                raise ValueError(
                    "The Stokes-L2 transfer requires the target pressure space."
                )
            return StokesL2VelocityTransfer(
                source_velocity_space,
                target_velocity_space,
                source_pressure_space=source_pressure_space,
                target_pressure_space=target_pressure_space,
                inlet_labels=self.labels.inlet,
                wall_labels=self.labels.wall,
            )
        if mode == "stokes_h1":
            if target_pressure_space is None:
                raise ValueError(
                    "The Stokes-H1 transfer requires the target pressure space."
                )
            return StokesH1VelocityTransfer(
                source_velocity_space,
                target_velocity_space,
                source_pressure_space=source_pressure_space,
                target_pressure_space=target_pressure_space,
                inlet_labels=self.labels.inlet,
                wall_labels=self.labels.wall,
            )
        raise ValueError(f"Unsupported Taylor--Hood transfer mode: {mode!r}.")

    def forward_interface_state(
        self,
        transfer: MassConsistentVelocityTransfer,
        source_state,
        target_mixed_space,
        physical_time,
        *,
        name: str = "P_U_interface",
    ) -> Function:
        """Apply ``P`` to velocity, impose the trace, and set pressure to zero."""
        target_velocity = transfer.forward(
            source_state.subfunctions[self.velocity_field],
            boundary_value=self.inflow_value(target_mixed_space.mesh(), physical_time),
            name=f"{name}_velocity",
        )
        # The mass-consistent u0 is a Riesz representation of the old trace
        # functional.  It is not a new-slab trial function, so stage boundary
        # conditions must not be applied to it.  The Stokes transfer already
        # imposes the physical target trace; applying the same values while
        # packing the mixed state is idempotent.
        trace_time = None if getattr(transfer, "mass_consistent", False) else physical_time
        return self.state_from_velocity(
            target_mixed_space, target_velocity, name=name, physical_time=trace_time,
        )

    def adjoint_interface_state(
        self,
        transfer: MassConsistentVelocityTransfer,
        target_dual_state,
        source_mixed_space,
        *,
        name: str = "Pstar_Z_interface",
    ) -> Function:
        """Apply ``P_star`` to dual velocity; dual pressure has no trace."""
        source_velocity = transfer.adjoint(
            target_dual_state.subfunctions[self.velocity_field],
            name=f"{name}_velocity",
        )
        return self.state_from_velocity(source_mixed_space, source_velocity, name=name)

    def contract_summary(self) -> dict[str, Any]:
        return {
            "state_fields": ("velocity", "pressure"),
            "differential_fields": ("velocity",),
            "algebraic_fields": ("pressure",),
            "time_mass": "velocity_L2",
            "forward_interface": "supermesh_mass_velocity_only",
            "adjoint_interface": "velocity_L2_adjoint_only",
            "pressure_interface": "zero_initial_guess_no_trace",
        }


__all__ = ["CylinderBoundaryLabels", "CylinderMixedDAEAdapter"]
