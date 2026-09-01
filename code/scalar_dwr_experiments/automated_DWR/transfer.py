"""Mesh-change operators shared by primal, adjoint, and DWR interfaces.

For every transition from the spatial space ``V_n`` on slab ``I_n`` to the
space ``V_{n+1}`` on ``I_{n+1}``, Firedrake assembles the interpolation matrix
``P_n``.  The forward state uses ``u_{n+1}^+ = P_n u_n^-``.  The numerical
adjoint must use the adjoint with respect to the PDE's time-mass pairing,
not simply interpolate in reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from firedrake import (
    Function, SpatialCoordinate, TestFunction, TrialFunction, assemble, interpolate,
)
from firedrake.exceptions import DofNotDefinedError
from firedrake.petsc import PETSc


@dataclass
class SlabInterfaceTransfer:
    r"""One forward mesh transfer ``P`` and its time-mass adjoint ``P^*``.

    If the temporal derivative in the weak PDE is represented by a mass form
    ``m_n(u,v)``, the transfer operators are

    .. math::

       \boldsymbol u_{n+1}^{+}=P_n\boldsymbol u_n^{-},
       \qquad
       P_n^*=M_n^{-1}P_n^TM_{n+1}.

    ``M_n`` is assembled from the problem-defined time-mass form.  For Heat
    this is the ordinary ``L2`` mass; BBM instead supplies its ``H1`` mass.
    """

    source_space: Any
    target_space: Any
    interpolation_matrix: Any
    source_mass: Any
    target_mass: Any
    source_mass_ksp: Any
    boundary_conditions: Callable[[Any], list[Any]]

    @staticmethod
    def _petsc_matrix(matrix):
        """Accept either a Firedrake assembled matrix or a raw PETSc matrix."""
        return getattr(matrix, "petscmat", matrix)

    def forward(self, source: Function, name: str) -> Function:
        """Return ``P_n source`` in the receiving slab space."""
        target = Function(self.target_space, name=name)
        with source.dat.vec_ro as source_vec, target.dat.vec_wo as target_vec:
            self._petsc_matrix(self.interpolation_matrix).mult(source_vec, target_vec)
        for bc in self.boundary_conditions(self.target_space):
            bc.apply(target)
        return target

    def adjoint(self, target_dual: Function, name: str) -> Function:
        """Return ``P_n^* target_dual`` in the preceding slab space."""
        weighted_target = Function(self.target_space, name=f"{name}_M_target")
        source_rhs = Function(self.source_space, name=f"{name}_rhs")
        source_dual = Function(self.source_space, name=name)
        with target_dual.dat.vec_ro as dual_vec, weighted_target.dat.vec_wo as weighted_vec:
            self.target_mass.petscmat.mult(dual_vec, weighted_vec)
        with weighted_target.dat.vec_ro as weighted_vec, source_rhs.dat.vec_wo as rhs_vec:
            self._petsc_matrix(self.interpolation_matrix).multTranspose(weighted_vec, rhs_vec)
        with source_rhs.dat.vec_ro as rhs_vec, source_dual.dat.vec_wo as source_vec:
            self.source_mass_ksp.solve(rhs_vec, source_vec)
        for bc in self.boundary_conditions(self.source_space):
            bc.apply(source_dual)
        return source_dual


def _periodic_interval_nodal_interpolation(source_space, target_space):
    r"""Build the exact nodal interpolation matrix on a serial periodic interval.

    Firedrake's general point-search interpolation treats the two copies of
    the seam of separately constructed :class:`PeriodicIntervalMesh` objects
    as different physical points.  A slab that is uniformly refined therefore
    cannot transfer its final trace to an unrefined neighbouring slab.  On a
    uniform periodic interval, however, the continuous Lagrange degrees of
    freedom are a uniform cyclic lattice.  This construction evaluates the
    source-cell Lagrange basis at every target node, with the cell index taken
    modulo the number of source cells.  It supports the project's CG1 primal
    and CG2 enriched-adjoint spaces without dropping seam degrees of freedom.

    The generic cross-mesh construction remains the default.  This is only a
    narrow fallback for the periodic 1D case where Firedrake's point locator
    raises ``DofNotDefinedError``.
    """
    source_mesh = source_space.mesh()
    target_mesh = target_space.mesh()
    if source_mesh.topological_dimension != 1 or target_mesh.topological_dimension != 1:
        raise RuntimeError("Periodic nodal transfer is only defined for one-dimensional meshes.")
    if source_mesh.comm.size != 1 or target_mesh.comm.size != 1:
        raise RuntimeError("Periodic nodal transfer currently requires a serial Firedrake run.")

    source_element = source_space.ufl_element()
    target_element = target_space.ufl_element()
    if source_element.family() != "Lagrange" or target_element.family() != "Lagrange":
        raise RuntimeError("Periodic nodal transfer requires continuous Lagrange spaces.")
    source_degree = int(source_element.degree())
    target_degree = int(target_element.degree())
    if source_degree < 1 or target_degree < 1:
        raise RuntimeError("Periodic nodal transfer requires positive polynomial degrees.")

    source_cells = int(source_mesh.num_cells())
    target_cells = int(target_mesh.num_cells())
    source_dofs = int(source_space.dim())
    target_dofs = int(target_space.dim())
    if source_dofs != source_cells * source_degree or target_dofs != target_cells * target_degree:
        raise RuntimeError("The periodic mesh does not have the expected cyclic Lagrange dof layout.")

    # The coordinate table gives the global dof permutation.  Sorting it puts
    # its entries in cyclic physical order; the omitted endpoint is recovered
    # by modulo arithmetic below.
    source_coordinates = Function(source_space, name="periodic_source_dof_coordinates")
    target_coordinates = Function(target_space, name="periodic_target_dof_coordinates")
    source_coordinates.interpolate(SpatialCoordinate(source_mesh)[0])
    target_coordinates.interpolate(SpatialCoordinate(target_mesh)[0])
    source_x = np.asarray(source_coordinates.dat.data_ro, dtype=float).copy()
    target_x = np.asarray(target_coordinates.dat.data_ro, dtype=float).copy()
    source_order = np.argsort(source_x)
    target_order = np.argsort(target_x)
    if len(source_order) != source_dofs or len(target_order) != target_dofs:
        raise RuntimeError("Could not obtain all periodic nodal coordinates.")

    # On PeriodicIntervalMesh, the last physical node L is identified with 0.
    # The uniform dof lattice hence determines L without relying on the absent
    # endpoint coordinate.
    source_sorted = source_x[source_order]
    spacing = float(np.min(np.diff(source_sorted)))
    length = spacing * source_dofs
    if not np.isfinite(length) or length <= 0.0:
        raise RuntimeError("Could not recover the periodic interval length from its dof lattice.")

    matrix = PETSc.Mat().createAIJ(
        size=(target_dofs, source_dofs), nnz=source_degree + 1, comm=source_mesh.comm
    )
    cell_width = length / source_cells
    reference_nodes = np.arange(source_degree + 1, dtype=float) / source_degree
    tolerance = 128.0 * np.finfo(float).eps * length
    for physical_target_index in target_order:
        x = float(np.mod(target_x[physical_target_index], length))
        if abs(x - length) <= tolerance:
            x = 0.0
        cell = int(np.floor((x + tolerance) / cell_width)) % source_cells
        xi = (x - cell * cell_width) / cell_width
        if xi < 0.0 and abs(xi) <= tolerance / cell_width:
            xi = 0.0
        if xi > 1.0 and abs(xi - 1.0) <= tolerance / cell_width:
            cell = (cell + 1) % source_cells
            xi = 0.0
        columns = [int(source_order[(cell * source_degree + j) % source_dofs])
                   for j in range(source_degree + 1)]
        values = []
        for j, node in enumerate(reference_nodes):
            value = 1.0
            for k, other_node in enumerate(reference_nodes):
                if k != j:
                    value *= (xi - other_node) / (node - other_node)
            values.append(value)
        matrix.setValues(int(physical_target_index), columns, values)
    matrix.assemblyBegin()
    matrix.assemblyEnd()
    return matrix


def build_slab_interface_transfer(problem, source_space, target_space) -> SlabInterfaceTransfer:
    """Assemble a sparse cross-mesh interpolation matrix and mass adjoint.

    ``interpolate(TrialFunction(V_old), V_new)`` is Firedrake's general
    finite-element construction of ``P``.  It works for arbitrary continuous
    Lagrange degree and dimensions; no node-average rule is handwritten here.
    """
    try:
        interpolation_matrix = assemble(
            interpolate(TrialFunction(source_space), target_space), mat_type="aij"
        )
    except DofNotDefinedError:
        interpolation_matrix = _periodic_interval_nodal_interpolation(
            source_space, target_space
        )
    source_trial, source_test = TrialFunction(source_space), TestFunction(source_space)
    target_trial, target_test = TrialFunction(target_space), TestFunction(target_space)
    source_mass = assemble(problem.time_mass_action(source_trial, source_test), mat_type="aij")
    target_mass = assemble(problem.time_mass_action(target_trial, target_test), mat_type="aij")
    mass_ksp = PETSc.KSP().create(source_space.mesh().comm)
    mass_ksp.setOperators(source_mass.petscmat)
    mass_ksp.setType("preonly")
    mass_ksp.getPC().setType("lu")
    mass_ksp.setFromOptions()
    mass_ksp.setUp()
    return SlabInterfaceTransfer(
        source_space=source_space,
        target_space=target_space,
        interpolation_matrix=interpolation_matrix,
        source_mass=source_mass,
        target_mass=target_mass,
        source_mass_ksp=mass_ksp,
        boundary_conditions=problem.boundary_conditions,
    )
