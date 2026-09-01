"""Spatial and temporal refinement on independently adapted time slabs."""

from __future__ import annotations

import numpy as np
from firedrake import Function, FunctionSpace, IntervalMesh, Mesh, PeriodicIntervalMesh


def refine_marked_mesh(mesh: Mesh, markers: Function) -> Mesh:
    """Request Firedrake/Netgen h-refinement of marked cells."""
    if not hasattr(mesh, "refine_marked_elements"):
        raise RuntimeError(
            "The current Firedrake mesh does not support marked h-refinement."
        )
    try:
        return mesh.refine_marked_elements(markers)
    except (ImportError, ValueError) as exception:
        raise RuntimeError(
            "Marked spatial refinement needs a Netgen-backed mesh."
        ) from exception


def refine_marked_interval_mesh(mesh: Mesh, markers: Function) -> Mesh:
    """Bisect precisely the marked cells of a serial interval mesh."""
    if mesh.topological_dimension != 1:
        raise ValueError("Marked interval refinement requires a one-dimensional mesh.")
    if mesh.comm.size != 1:
        raise NotImplementedError("Marked interval refinement is currently serial.")
    DG0 = FunctionSpace(mesh, "DG", 0)
    if markers.function_space() != DG0:
        raise ValueError("markers must belong to DG(0) on the mesh being refined.")
    coordinate_dofs = mesh.coordinates.function_space().cell_node_map().values
    cell_dofs = DG0.cell_node_map().values[:, 0]
    coordinates = np.asarray(mesh.coordinates.dat.data_ro, dtype=float).reshape(-1)
    endpoints = coordinates[coordinate_dofs]
    left, right = endpoints.min(axis=1), endpoints.max(axis=1)
    if np.any(right <= left):
        raise RuntimeError("Encountered a degenerate interval cell during refinement.")
    order = np.argsort(left)
    marker_values = np.asarray(markers.dat.data_ro, dtype=float)
    vertices = [float(left[order[0]])]
    for cell in order:
        a, b = float(left[cell]), float(right[cell])
        if marker_values[cell_dofs[cell]] > 0.5:
            vertices.append(0.5 * (a + b))
        vertices.append(b)
    vertices = np.asarray(vertices, dtype=float)
    if np.any(np.diff(vertices) <= 0.0):
        raise RuntimeError("Interval refinement produced unordered vertices.")
    refined = IntervalMesh(
        len(vertices) - 1, float(vertices[0]), float(vertices[-1]), reorder=False
    )
    refined.coordinates.dat.data.reshape(-1)[:] = vertices
    return refined


def refine_marked_periodic_interval_mesh(
    mesh: Mesh, markers: Function, length: float
) -> Mesh:
    """Bisect marked cells while retaining periodic DMPlex topology."""
    if mesh.topological_dimension != 1 or mesh.comm.size != 1:
        raise NotImplementedError("Periodic marked refinement requires serial 1D.")
    DG0 = FunctionSpace(mesh, "DG", 0)
    coordinate_dofs = mesh.coordinates.function_space().cell_node_map().values
    cell_dofs = DG0.cell_node_map().values[:, 0]
    coordinates = np.asarray(mesh.coordinates.dat.data_ro, dtype=float).reshape(-1)
    endpoints = coordinates[coordinate_dofs]
    left, right = endpoints.min(axis=1), endpoints.max(axis=1)
    order = np.argsort(left)
    marker_values = np.asarray(markers.dat.data_ro, dtype=float)
    vertices = [float(left[order[0]])]
    for cell in order:
        a, b = float(left[cell]), float(right[cell])
        if marker_values[cell_dofs[cell]] > 0.5:
            vertices.append(0.5 * (a + b))
        vertices.append(b)
    vertices = np.asarray(vertices, dtype=float)
    if abs(vertices[0]) > 1.0e-11 or abs(vertices[-1] - float(length)) > 1.0e-11:
        raise RuntimeError("Periodic refinement lost the physical endpoints.")

    refined = PeriodicIntervalMesh(len(vertices) - 1, float(length), reorder=False)
    target_dofs = refined.coordinates.function_space().cell_node_map().values
    target_coordinates = refined.coordinates.dat.data.reshape(-1)
    target_endpoints = target_coordinates[target_dofs]
    target_order = np.argsort(target_endpoints.min(axis=1))
    for index, cell in enumerate(target_order):
        dofs = target_dofs[cell]
        old = target_coordinates[dofs].copy()
        lower, upper = float(old.min()), float(old.max())
        for dof, value in zip(dofs, old):
            target_coordinates[dof] = (
                vertices[index]
                if abs(value - lower) <= abs(value - upper)
                else vertices[index + 1]
            )
    return refined


def inherit_refinement_marks(
    source_mesh: Mesh,
    source_marks: np.ndarray,
    target_mesh: Mesh,
    *,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    """Transfer a marked triangular parent region to a nested target mesh."""
    source_marks = np.asarray(source_marks, dtype=bool)
    source_dg0 = FunctionSpace(source_mesh, "DG", 0)
    target_dg0 = FunctionSpace(target_mesh, "DG", 0)
    source_cell_dofs = source_dg0.cell_node_map().values[:, 0]
    target_cell_dofs = target_dg0.cell_node_map().values[:, 0]
    if source_marks.size != source_dg0.node_count:
        raise ValueError("source_marks must be stored in source DG(0) dof order.")
    if source_mesh is target_mesh:
        return source_marks.copy()
    source_nodes = source_mesh.coordinates.function_space().cell_node_map().values
    target_nodes = target_mesh.coordinates.function_space().cell_node_map().values
    if source_nodes.shape[1] != 3 or target_nodes.shape[1] != 3:
        raise NotImplementedError("Causal inheritance supports triangular 2D meshes.")
    source_triangles = source_mesh.coordinates.dat.data_ro[source_nodes]
    target_triangles = target_mesh.coordinates.dat.data_ro[target_nodes]
    marked_rows = np.flatnonzero(source_marks[source_cell_dofs])
    inherited_rows = np.zeros(target_triangles.shape[0], dtype=bool)
    if marked_rows.size == 0:
        return np.zeros(target_dg0.node_count, dtype=bool)
    target_centres = target_triangles.mean(axis=1)
    for row in marked_rows:
        triangle = source_triangles[row]
        lower = triangle.min(axis=0) - tolerance
        upper = triangle.max(axis=0) + tolerance
        candidates = np.flatnonzero(
            np.logical_and(target_centres >= lower, target_centres <= upper).all(axis=1)
        )
        origin = triangle[0]
        edge_1, edge_2 = triangle[1] - origin, triangle[2] - origin
        determinant = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if abs(float(determinant)) <= tolerance:
            raise RuntimeError("Degenerate source triangle during causal inheritance.")
        offsets = target_centres[candidates] - origin
        b1 = (offsets[:, 0] * edge_2[1] - offsets[:, 1] * edge_2[0]) / determinant
        b2 = (edge_1[0] * offsets[:, 1] - edge_1[1] * offsets[:, 0]) / determinant
        b0 = 1.0 - b1 - b2
        inside = np.logical_and.reduce(
            (b0 >= -tolerance, b1 >= -tolerance, b2 >= -tolerance)
        )
        inherited_rows[candidates[inside]] = True
    inherited = np.zeros(target_dg0.node_count, dtype=bool)
    inherited[target_cell_dofs[inherited_rows]] = True
    return inherited


def refine_time_grid(ts: np.ndarray, marked_slabs: set[int]) -> np.ndarray:
    """Bisect every selected time slab once."""
    if not marked_slabs:
        return np.asarray(ts, dtype=float)
    refined = [float(ts[0])]
    for slab in range(1, len(ts)):
        if slab in marked_slabs:
            refined.append(0.5 * (float(ts[slab - 1]) + float(ts[slab])))
        refined.append(float(ts[slab]))
    return np.asarray(refined, dtype=float)


__all__ = [
    "inherit_refinement_marks",
    "refine_marked_interval_mesh",
    "refine_marked_mesh",
    "refine_marked_periodic_interval_mesh",
    "refine_time_grid",
]
