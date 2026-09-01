r"""Taylor--Hood velocity trace transfers for independent slab meshes."""

from __future__ import annotations

from firedrake import (
    Cofunction,
    Constant,
    DirichletBC,
    Function,
    TestFunction,
    TestFunctions,
    TrialFunction,
    assemble,
    div,
    dx,
    grad,
    inner,
    interpolate,
    solve,
    split,
)
from firedrake.supermeshing import assemble_mixed_mass_matrix


_DIRECT = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


class MassConsistentVelocityTransfer:
    r"""Apply the arbitrary-mesh dG velocity mass map and its exact adjoint.

    If ``B_ji=(phi_i^source,phi_j^target)``, then
    ``P=M_target^{-1}B`` and ``P*=M_source^{-1}B.T``.  Pressure is algebraic
    and therefore has no interface transfer.
    """

    mass_consistent = True

    def __init__(
        self,
        source_velocity_space,
        target_velocity_space,
        *,
        source_pressure_space=None,
        target_pressure_space=None,
    ):
        self.source_space = source_velocity_space
        self.target_space = target_velocity_space
        self.source_pressure_space = source_pressure_space
        self.target_pressure_space = target_pressure_space
        try:
            self.mixed_mass = assemble_mixed_mass_matrix(
                source_velocity_space, target_velocity_space
            )
        except Exception as error:
            raise RuntimeError(
                "Independent slabwise Taylor--Hood transfer requires a valid "
                "Firedrake supermesh mixed-mass matrix."
            ) from error

    @staticmethod
    def _continuity_residual_l2(velocity, pressure_space):
        """L2 norm of the pressure-space projection of ``div(velocity)``."""
        if pressure_space is None:
            return float("nan")
        pressure_test = TestFunction(pressure_space)
        functional = assemble(inner(div(velocity), pressure_test) * dx)
        representative = functional.riesz_representation(
            "L2", solver_options=_DIRECT
        )
        return max(
            float(assemble(inner(representative, representative) * dx)), 0.0
        ) ** 0.5

    def forward(self, source_velocity, boundary_value=None, name="P_mass"):
        target_functional = Cofunction(
            self.target_space.dual(), name=f"{name}_functional"
        )
        with (
            source_velocity.dat.vec_ro as source_vector,
            target_functional.dat.vec_wo as target_vector,
        ):
            self.mixed_mass.mult(source_vector, target_vector)
        result = target_functional.riesz_representation(
            "L2", solver_options=_DIRECT
        )
        result.rename(name)
        source_squared = max(
            float(assemble(inner(source_velocity, source_velocity) * dx)), 0.0
        )
        target_squared = max(
            float(assemble(inner(result, result) * dx)), 0.0
        )
        with (
            target_functional.dat.vec_ro as functional_vector,
            result.dat.vec_ro as result_vector,
        ):
            cross_inner_product = float(functional_vector.dot(result_vector))
        correction_squared = max(
            source_squared + target_squared - 2.0 * cross_inner_product, 0.0
        )
        source_norm = source_squared**0.5
        target_norm = target_squared**0.5
        correction_norm = correction_squared**0.5
        self.last_forward_diagnostics = {
            "mode": "velocity_mass_supermesh",
            "source_l2": source_norm,
            "projected_l2": target_norm,
            "l2_norm_ratio": target_norm / max(source_norm, 1.0e-30),
            "cross_inner_product": cross_inner_product,
            "correction_l2": correction_norm,
            "correction_relative_l2": correction_norm / max(source_norm, 1.0e-30),
            "kinetic_energy_change": 0.5 * (target_squared - source_squared),
            "kinetic_energy_change_relative": (
                (target_squared - source_squared) / max(source_squared, 1.0e-30)
            ),
            "source_continuity_residual_l2": self._continuity_residual_l2(
                source_velocity, self.source_pressure_space
            ),
            "projected_continuity_residual_l2": self._continuity_residual_l2(
                result, self.target_pressure_space
            ),
            "projected_divergence_l2": max(
                float(assemble(inner(div(result), div(result)) * dx)), 0.0
            ) ** 0.5,
        }
        return result

    def adjoint(self, target_dual, name="Pstar_mass"):
        source_functional = Cofunction(
            self.source_space.dual(), name=f"{name}_functional"
        )
        with (
            target_dual.dat.vec_ro as target_vector,
            source_functional.dat.vec_wo as source_vector,
        ):
            self.mixed_mass.multTranspose(target_vector, source_vector)
        result = source_functional.riesz_representation(
            "L2", solver_options=_DIRECT
        )
        result.rename(name)
        return result


class StokesL2VelocityTransfer:
    r"""R5-style interpolation followed by a target Stokes projection.

    The forward map is ``P = Q I``.  ``I`` is the assembled cross-mesh
    finite-element interpolation from the old velocity space to the new one.
    ``Q`` is the target Taylor--Hood ``L2`` projection onto the discretely
    divergence-free velocities.  The physical inlet/wall trace makes the
    primal map affine; its derivative and the reverse dG interface use the
    homogeneous-trace map ``Q_0 I``.

    With velocity mass matrices ``M_s`` and ``M_t``, the reverse map is the
    exact mass adjoint

    ``P* = M_{s,0}^{-1} I.T M_t Q_0``, where the source Riesz solve is
    restricted to the homogeneous essential-velocity subspace.  Solving with
    the full source mass matrix and zeroing boundary coefficients afterwards
    is not equivalent and breaks the transfer pairing.

    This matches the short, robust transfer advocated for incompressible flow
    and the interpolation-plus-divergence-free-projection pipeline in R5.
    Pressure remains an algebraic multiplier and is not transferred.
    """

    mass_consistent = False
    divergence_preserving = True

    def __init__(
        self,
        source_velocity_space,
        target_velocity_space,
        *,
        source_pressure_space=None,
        target_pressure_space,
        inlet_labels,
        wall_labels,
    ):
        self.source_space = source_velocity_space
        self.target_space = target_velocity_space
        self.source_pressure_space = source_pressure_space
        self.target_pressure_space = target_pressure_space
        self.inlet = tuple(inlet_labels)
        self.wall = tuple(wall_labels)
        self.mixed_space = target_velocity_space * target_pressure_space
        try:
            self.interpolation_matrix = assemble(
                interpolate(
                    TrialFunction(source_velocity_space), target_velocity_space
                ),
                mat_type="aij",
            )
        except Exception as error:
            raise RuntimeError(
                "Could not assemble the cross-mesh Taylor--Hood interpolation "
                "required by the Stokes-L2 transfer."
            ) from error

    def _interpolate(self, source_velocity, name):
        seed = Function(self.target_space, name=f"{name}_interpolated")
        with (
            source_velocity.dat.vec_ro as source_vector,
            seed.dat.vec_wo as target_vector,
        ):
            self.interpolation_matrix.petscmat.mult(
                source_vector, target_vector
            )
        return seed

    def _project(self, seed, inlet_value, name):
        state = Function(self.mixed_space, name=f"{name}_mixed")
        velocity, multiplier = split(state)
        velocity_test, multiplier_test = TestFunctions(self.mixed_space)
        residual = (
            inner(velocity - seed, velocity_test)
            - multiplier * div(velocity_test)
            - div(velocity) * multiplier_test
        ) * dx
        bcs = [
            DirichletBC(self.mixed_space.sub(0), inlet_value, self.inlet),
            DirichletBC(
                self.mixed_space.sub(0), Constant((0.0, 0.0)), self.wall
            ),
        ]
        solve(
            residual == 0,
            state,
            bcs=bcs,
            solver_parameters=_DIRECT,
        )
        result = Function(self.target_space, name=name)
        result.assign(state.subfunctions[0])
        return result


    @staticmethod
    def _continuity_residual_l2(velocity, pressure_space):
        return MassConsistentVelocityTransfer._continuity_residual_l2(
            velocity, pressure_space
        )

    def forward(self, source_velocity, boundary_value=None, name="P_stokes_l2"):
        if boundary_value is None:
            boundary_value = Constant((0.0, 0.0))
        seed = self._interpolate(source_velocity, name)
        result = self._project(seed, boundary_value, name)
        source_squared = max(
            float(assemble(inner(source_velocity, source_velocity) * dx)), 0.0
        )
        seed_squared = max(float(assemble(inner(seed, seed) * dx)), 0.0)
        target_squared = max(
            float(assemble(inner(result, result) * dx)), 0.0
        )
        correction_squared = max(
            float(assemble(inner(result - seed, result - seed) * dx)), 0.0
        )
        source_norm = source_squared**0.5
        seed_norm = seed_squared**0.5
        target_norm = target_squared**0.5
        correction_norm = correction_squared**0.5
        self.last_forward_diagnostics = {
            "mode": "velocity_interpolation_stokes_l2",
            "source_l2": source_norm,
            "seed_l2": seed_norm,
            "projected_l2": target_norm,
            "interpolation_l2_norm_ratio": seed_norm
            / max(source_norm, 1.0e-30),
            "l2_norm_ratio": target_norm / max(source_norm, 1.0e-30),
            "correction_l2": correction_norm,
            "correction_relative_l2": correction_norm
            / max(seed_norm, 1.0e-30),
            "kinetic_energy_change": 0.5 * (target_squared - source_squared),
            "kinetic_energy_change_relative": (
                (target_squared - source_squared)
                / max(source_squared, 1.0e-30)
            ),
            "source_continuity_residual_l2": self._continuity_residual_l2(
                source_velocity, self.source_pressure_space
            ),
            "seed_continuity_residual_l2": self._continuity_residual_l2(
                seed, self.target_pressure_space
            ),
            "projected_continuity_residual_l2": self._continuity_residual_l2(
                result, self.target_pressure_space
            ),
            "projected_divergence_l2": max(
                float(assemble(inner(div(result), div(result)) * dx)), 0.0
            )
            ** 0.5,
        }
        return result

    def adjoint(self, target_dual, name="Pstar_stokes_l2"):
        projected = self._project(
            target_dual, Constant((0.0, 0.0)), f"{name}_projected"
        )
        target_functional = Cofunction(
            self.target_space.dual(), name=f"{name}_target_functional"
        )
        target_functional.assign(
            inner(projected, TestFunction(self.target_space)) * dx
        )
        source_functional = Cofunction(
            self.source_space.dual(), name=f"{name}_source_functional"
        )
        with (
            target_functional.dat.vec_ro as target_vector,
            source_functional.dat.vec_wo as source_vector,
        ):
            self.interpolation_matrix.petscmat.multTranspose(
                target_vector, source_vector
            )
        source_bcs = DirichletBC(
            self.source_space,
            Constant((0.0, 0.0)),
            tuple(self.inlet) + tuple(self.wall),
        )
        result = source_functional.riesz_representation(
            "L2", bcs=source_bcs, solver_options=_DIRECT
        )
        result.rename(name)
        return result


class StokesH1VelocityTransfer(StokesL2VelocityTransfer):
    r"""Interpolation followed by an H1-seminorm Stokes projection.

    The forward derivative is ``P = Q_H1 I``.  Because ``Q_H1`` is not
    self-adjoint in the velocity L2 pairing used by the dG jump term, the
    reverse operator is *not* obtained by applying the same H1 projection.
    Instead, ``adjoint`` solves the transposed target Stokes system with an
    L2 right-hand side, applies the target stiffness functional, then uses
    ``I.T`` and a source-side constrained L2 Riesz representation.  Hence it
    satisfies ``(P u, z)_target = (u, P* z)_source``.
    """

    def _project(self, seed, inlet_value, name):
        state = Function(self.mixed_space, name=f"{name}_mixed")
        velocity, multiplier = split(state)
        velocity_test, multiplier_test = TestFunctions(self.mixed_space)
        residual = (
            inner(grad(velocity - seed), grad(velocity_test))
            - multiplier * div(velocity_test)
            - div(velocity) * multiplier_test
        ) * dx
        bcs = [
            DirichletBC(self.mixed_space.sub(0), inlet_value, self.inlet),
            DirichletBC(
                self.mixed_space.sub(0), Constant((0.0, 0.0)), self.wall
            ),
        ]
        solve(residual == 0, state, bcs=bcs, solver_parameters=_DIRECT)
        result = Function(self.target_space, name=name)
        result.assign(state.subfunctions[0])
        return result

    def forward(self, source_velocity, boundary_value=None, name="P_stokes_h1"):
        result = super().forward(source_velocity, boundary_value, name)
        self.last_forward_diagnostics["mode"] = (
            "velocity_interpolation_stokes_h1"
        )
        return result

    def adjoint(self, target_dual, name="Pstar_stokes_h1"):
        # A_H1 is symmetric, so this solves A_H1 y = [M_t z, 0].
        adjoint_state = Function(self.mixed_space, name=f"{name}_mixed")
        velocity, multiplier = split(adjoint_state)
        velocity_test, multiplier_test = TestFunctions(self.mixed_space)
        residual = (
            inner(grad(velocity), grad(velocity_test))
            - multiplier * div(velocity_test)
            - div(velocity) * multiplier_test
            - inner(target_dual, velocity_test)
        ) * dx
        target_bcs = [
            DirichletBC(
                self.mixed_space.sub(0), Constant((0.0, 0.0)), self.inlet
            ),
            DirichletBC(
                self.mixed_space.sub(0), Constant((0.0, 0.0)), self.wall
            ),
        ]
        solve(
            residual == 0,
            adjoint_state,
            bcs=target_bcs,
            solver_parameters=_DIRECT,
        )
        adjoint_velocity = adjoint_state.subfunctions[0]

        # K_t y is Q_H1.T M_t z, a functional on the interpolated seed.
        target_functional = Cofunction(
            self.target_space.dual(), name=f"{name}_target_functional"
        )
        target_functional.assign(
            inner(grad(adjoint_velocity), grad(TestFunction(self.target_space)))
            * dx
        )
        source_functional = Cofunction(
            self.source_space.dual(), name=f"{name}_source_functional"
        )
        with (
            target_functional.dat.vec_ro as target_vector,
            source_functional.dat.vec_wo as source_vector,
        ):
            self.interpolation_matrix.petscmat.multTranspose(
                target_vector, source_vector
            )
        source_bcs = DirichletBC(
            self.source_space,
            Constant((0.0, 0.0)),
            tuple(self.inlet) + tuple(self.wall),
        )
        result = source_functional.riesz_representation(
            "L2", bcs=source_bcs, solver_options=_DIRECT
        )
        result.rename(name)
        return result


__all__ = [
    "MassConsistentVelocityTransfer",
    "StokesL2VelocityTransfer",
    "StokesH1VelocityTransfer",
]
