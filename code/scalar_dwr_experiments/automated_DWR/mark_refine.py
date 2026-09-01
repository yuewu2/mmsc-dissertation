"""Global Dörfler marking and independent space/time refinement.

The marking decision is made in the product index set ``K x I_n``.  It is
therefore driven by the fully localised estimator, not by a heuristic based on
separate spatial and temporal error lists.
"""

from __future__ import annotations

import numpy as np
from firedrake import Function, FunctionSpace, IntervalMesh, Mesh, PeriodicIntervalMesh


def mark_by_bulk(scores: np.ndarray, fraction: float) -> np.ndarray:
    r"""Mark a smallest descending prefix with bulk ``>= fraction*sum(scores)``.

    This is Dörfler marking for the non-negative array
    ``scores[K,n]=abs(eta[K,n])``:

    .. math::

       \sum_{(K,n)\in M}|\eta_{K,n}|\geq\theta\sum_{K,n}|\eta_{K,n}|.
    """
    values = np.maximum(np.asarray(scores, dtype=float), 0.0)
    marked = np.zeros(values.shape, dtype=bool)
    total = float(values.sum())
    if total <= 0.0:
        return marked
    subtotal = 0.0
    target = float(fraction) * total
    for index in np.argsort(values)[::-1]:
        marked[index] = True
        subtotal += float(values[index])
        if subtotal >= target:
            break
    return marked


def mark_spacetime_cells(
    eta_cell_slab_signed: list[np.ndarray | None], theta: float
) -> list[np.ndarray | None]:
    r"""Apply one global Dörfler operation to all ``abs(eta[K,n])`` values.

    Different slabwise meshes may contain different numbers of cells.  The
    product-space Dörfler criterion is unchanged; unflattening simply uses
    every slab's own vector length.
    """
    slab_values = [np.asarray(values, dtype=float) for values in eta_cell_slab_signed[1:]]
    if not slab_values:
        return [None]
    flat_marks = mark_by_bulk(np.concatenate([np.abs(values) for values in slab_values]), theta)
    marked: list[np.ndarray | None] = [None]
    start = 0
    for values in slab_values:
        stop = start + values.size
        marked.append(flat_marks[start:stop].copy())
        start = stop
    return marked


def vertex_patch_contributions(
    eta_cell_slab_signed: list[np.ndarray | None],
    meshes: list[Mesh | None],
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    r"""Aggregate signed DG0 cell contributions on CG1 vertex patches.

    A cell contribution is distributed equally over its topological vertices.
    For a simplex cell ``K`` with ``d + 1`` vertices this defines

    .. math::

       \eta_v = \sum_{K\ni v}\frac{\eta_K}{d+1}.

    Consequently ``sum_v eta_v == sum_K eta_K`` (up to roundoff).  This is a
    marking-only, partition-of-unity-inspired aggregation: it preserves the
    existing global DWR estimator and localisation, but allows grid-scale
    opposite signs to cancel on one vertex patch before an absolute value is
    taken.  It is not a replacement for assembling the full PU residual
    ``rho(u_h)((z^+ - I_h z^+) chi_v)``.

    The second returned list stores, for every slab, the CG1 vertex dofs of
    each cell row.  It is used to map selected patches back to DG0 cells.
    """
    if len(eta_cell_slab_signed) != len(meshes):
        raise ValueError("Signed slab indicators and meshes must have equal length.")

    patch_signed: list[np.ndarray | None] = [None]
    cell_vertices: list[np.ndarray | None] = [None]
    for slab, (values, mesh) in enumerate(
        zip(eta_cell_slab_signed[1:], meshes[1:]), start=1
    ):
        if values is None or mesh is None:
            raise ValueError(f"Slab {slab} is missing indicators or a mesh.")
        values = np.asarray(values, dtype=float)
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_dofs = np.asarray(DG0.cell_node_map().values[:, 0], dtype=int)
        if values.size != DG0.node_count:
            raise ValueError(
                f"Slab {slab} indicator size {values.size} does not match "
                f"DG0 size {DG0.node_count}."
            )

        # Scalar CG1 has one dof per topological vertex even when the physical
        # coordinate mapping is curved (e.g. geometry degree two).
        P1 = FunctionSpace(mesh, "CG", 1)
        vertices = np.asarray(P1.cell_node_map().values, dtype=int)
        if vertices.shape[0] != cell_dofs.size:
            raise RuntimeError(
                f"Slab {slab} DG0 and CG1 cell maps use different row counts."
            )
        vertices_per_cell = int(vertices.shape[1])
        if vertices_per_cell != int(mesh.topological_dimension) + 1:
            raise NotImplementedError(
                "Vertex-patch marking currently requires simplex cells."
            )

        aggregated = np.zeros(P1.node_count, dtype=float)
        cell_values = values[cell_dofs] / float(vertices_per_cell)
        for local_vertex in range(vertices_per_cell):
            np.add.at(aggregated, vertices[:, local_vertex], cell_values)
        if not np.isclose(
            aggregated.sum(), values.sum(), rtol=1.0e-12, atol=1.0e-14
        ):
            raise RuntimeError(
                f"Slab {slab} vertex-patch aggregation lost signed closure."
            )
        patch_signed.append(aggregated)
        cell_vertices.append(vertices)
    return patch_signed, cell_vertices


def mark_spacetime_vertex_patches(
    eta_cell_slab_signed: list[np.ndarray | None],
    meshes: list[Mesh | None],
    theta: float,
) -> tuple[
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray | None],
]:
    r"""Dörfler-mark signed vertex-patch aggregates and expand to cells.

    The global bulk selection is applied to ``abs(eta_vn)`` over all
    space-time vertex patches.  Every cell incident to a selected vertex is
    then marked.  The return values are ``(cell_marks, patch_signed,
    patch_marks)`` so production runs can record both the original cellwise
    activity and the cancellation-aware patch activity.
    """
    patch_signed, cell_vertices = vertex_patch_contributions(
        eta_cell_slab_signed, meshes
    )
    slab_values = [
        np.asarray(values, dtype=float) for values in patch_signed[1:]
    ]
    if not slab_values:
        return [None], patch_signed, [None]
    flat_marks = mark_by_bulk(
        np.concatenate([np.abs(values) for values in slab_values]), theta
    )

    patch_marks: list[np.ndarray | None] = [None]
    cell_marks: list[np.ndarray | None] = [None]
    start = 0
    for slab, values in enumerate(slab_values, start=1):
        stop = start + values.size
        selected = flat_marks[start:stop].copy()
        patch_marks.append(selected)
        vertices = np.asarray(cell_vertices[slab], dtype=int)
        cell_rows = np.any(selected[vertices], axis=1)
        mesh = meshes[slab]
        DG0 = FunctionSpace(mesh, "DG", 0)
        cell_dofs = np.asarray(DG0.cell_node_map().values[:, 0], dtype=int)
        marked = np.zeros(DG0.node_count, dtype=bool)
        marked[cell_dofs] = cell_rows
        cell_marks.append(marked)
        start = stop
    return cell_marks, patch_signed, patch_marks


def refinement_from_spacetime_marks(
    marked_by_slab: list[np.ndarray | None], DG0, time_slab_marked_fraction: float
) -> tuple[Function, set[int], list[float]]:
    """Map product-space marks to spatial union and time-slab bisections.

    A spatial cell is refined if it was selected in any time slab.  A slab
    ``I_n`` is bisected only when ``#M_n/#T >= time_slab_marked_fraction``;
    this prevents one isolated space--time indicator from refining time.
    """
    markers = Function(DG0, name="space_markers")
    if len(marked_by_slab) <= 1:
        markers.assign(0.0)
        return markers, set(), [0.0]
    slab_masks = [np.asarray(mask, dtype=bool) for mask in marked_by_slab[1:]]
    spatial_union = np.logical_or.reduce(slab_masks)
    markers.dat.data[:] = spatial_union.astype(markers.dat.data.dtype)
    fractions = [0.0]
    time_marked: set[int] = set()
    for n, mask in enumerate(slab_masks, start=1):
        marked_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        fractions.append(marked_fraction)
        if np.any(mask) and marked_fraction >= float(time_slab_marked_fraction):
            time_marked.add(n)
    return markers, time_marked, fractions


def refine_marked_mesh(mesh: Mesh, markers: Function) -> Mesh:
    """Request Firedrake/Netgen h-refinement of cells where ``markers=1``."""
    if not hasattr(mesh, "refine_marked_elements"):
        raise RuntimeError("The current Firedrake mesh does not support marked h-refinement.")
    try:
        return mesh.refine_marked_elements(markers)
    except (ImportError, ValueError) as exception:
        raise RuntimeError(
            "Marked spatial refinement needs a Netgen mesh. "
            "For a generic unit-square input, use create_adaptive_unit_square_mesh "
            "instead of UnitSquareMesh."
        ) from exception


def refine_marked_interval_mesh(mesh: Mesh, markers: Function) -> Mesh:
    r"""Bisect precisely the marked cells of a serial one-dimensional interval.

    Netgen's marked-refinement interface used by the two-dimensional examples
    does not cover a plain 1D interval.  Here the exact analogue is simple:
    insert one midpoint in every marked cell and rebuild the interval with the
    resulting nonuniform coordinate sequence.  This gives true local h
    refinement on each independently adapted time slab.
    """
    if mesh.topological_dimension != 1:
        raise ValueError("Marked interval refinement requires a one-dimensional mesh.")
    if mesh.comm.size != 1:
        raise NotImplementedError("Marked interval refinement is currently serial.")
    DG0 = FunctionSpace(mesh, "DG", 0)
    if markers.function_space() != DG0:
        raise ValueError("markers must belong to DG(0) on the mesh being refined.")

    coordinate_dofs = mesh.coordinates.function_space().cell_node_map().values
    cell_dg0_dofs = DG0.cell_node_map().values[:, 0]
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
        if marker_values[cell_dg0_dofs[cell]] > 0.5:
            vertices.append(0.5 * (a + b))
        vertices.append(b)
    vertices = np.asarray(vertices, dtype=float)
    if np.any(np.diff(vertices) <= 0.0):
        raise RuntimeError("Marked interval refinement did not produce a strictly ordered grid.")

    refined = IntervalMesh(len(vertices) - 1, float(vertices[0]), float(vertices[-1]), reorder=False)
    # IntervalMesh recreates the endpoint labels; replace only its uniform geometry.
    refined.coordinates.dat.data.reshape(-1)[:] = vertices
    return refined


def refine_marked_periodic_interval_mesh(mesh: Mesh, markers: Function, length: float) -> Mesh:
    """Locally bisect marked cells while retaining periodic DMPlex topology.

    A periodic Firedrake interval stores a discontinuous coordinate field on
    a cyclic topology.  We rebuild that topology with the new number of cells
    and replace its cell endpoint coordinates by the nonuniform marked grid.
    Thus the seam remains an internal periodic connection rather than a pair
    of Dirichlet boundaries.
    """
    if mesh.topological_dimension != 1 or mesh.comm.size != 1:
        raise NotImplementedError("Periodic marked interval refinement currently requires serial 1D.")
    DG0 = FunctionSpace(mesh, "DG", 0)
    coordinate_dofs = mesh.coordinates.function_space().cell_node_map().values
    cell_dg0_dofs = DG0.cell_node_map().values[:, 0]
    coordinates = np.asarray(mesh.coordinates.dat.data_ro, dtype=float).reshape(-1)
    endpoints = coordinates[coordinate_dofs]
    left, right = endpoints.min(axis=1), endpoints.max(axis=1)
    order = np.argsort(left)
    marker_values = np.asarray(markers.dat.data_ro, dtype=float)
    vertices = [float(left[order[0]])]
    for cell in order:
        a, b = float(left[cell]), float(right[cell])
        if marker_values[cell_dg0_dofs[cell]] > 0.5:
            vertices.append(0.5 * (a + b))
        vertices.append(b)
    vertices = np.asarray(vertices, dtype=float)
    if abs(vertices[0]) > 1.0e-11 or abs(vertices[-1] - float(length)) > 1.0e-11:
        raise RuntimeError("Periodic refinement lost the physical interval endpoints.")

    refined = PeriodicIntervalMesh(len(vertices) - 1, float(length), reorder=False)
    target_dofs = refined.coordinates.function_space().cell_node_map().values
    target_coordinates = refined.coordinates.dat.data.reshape(-1)
    target_endpoints = target_coordinates[target_dofs]
    target_order = np.argsort(target_endpoints.min(axis=1))
    for i, cell in enumerate(target_order):
        dofs = target_dofs[cell]
        old = target_coordinates[dofs].copy()
        lower, upper = float(old.min()), float(old.max())
        for dof, value in zip(dofs, old):
            target_coordinates[dof] = vertices[i] if abs(value - lower) <= abs(value - upper) else vertices[i + 1]
    return refined


def inherit_refinement_marks(
    source_mesh: Mesh,
    source_marks: np.ndarray,
    target_mesh: Mesh,
    *,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    r"""Map marked parent regions from one slab to a later nested slab mesh.

    Let ``M_n`` be the marked cells of ``T_n``.  A causal mesh sequence needs

    .. math::

       C_{n+1}=M_{n+1}\cup\operatorname{inherit}_{n\to n+1}(C_n),

    so that every later mesh receives at least the refinements requested on
    earlier slabs.  Firedrake's Netgen interface does not expose a persistent
    parent-cell ID after separate refinements.  For two-dimensional triangular
    meshes we therefore transfer the *physical marked region*: every target
    cell whose barycentre lies in a marked source triangle is marked.

    This is exact when ``target_mesh`` is a conforming refinement of
    ``source_mesh``--the invariant maintained by the causal strategy.  It is
    intentionally rejected for non-triangular coordinate cells rather than
    silently guessing a correspondence.
    """
    source_marks = np.asarray(source_marks, dtype=bool)
    source_dg0 = FunctionSpace(source_mesh, "DG", 0)
    target_dg0 = FunctionSpace(target_mesh, "DG", 0)
    source_cell_dofs = source_dg0.cell_node_map().values[:, 0]
    target_cell_dofs = target_dg0.cell_node_map().values[:, 0]
    if source_marks.size != source_dg0.node_count:
        raise ValueError("source_marks must be stored in source DG(0) dof order.")
    if source_mesh is target_mesh:
        return source_marks.copy()

    source_coordinate_nodes = source_mesh.coordinates.function_space().cell_node_map().values
    target_coordinate_nodes = target_mesh.coordinates.function_space().cell_node_map().values
    if source_coordinate_nodes.shape[1] != 3 or target_coordinate_nodes.shape[1] != 3:
        raise NotImplementedError("Causal inheritance currently supports triangular two-dimensional meshes.")
    source_triangles = source_mesh.coordinates.dat.data_ro[source_coordinate_nodes]
    target_triangles = target_mesh.coordinates.dat.data_ro[target_coordinate_nodes]
    source_marked_rows = np.flatnonzero(source_marks[source_cell_dofs])
    inherited_rows = np.zeros(target_triangles.shape[0], dtype=bool)
    if source_marked_rows.size == 0:
        inherited = np.zeros(target_dg0.node_count, dtype=bool)
        return inherited

    # For a target child triangle K', its barycentre lies in its unique parent
    # K.  Barycentric containment gives a geometry-based parent lookup without
    # assuming that Netgen preserves cell ordering across the two meshes.
    target_centres = target_triangles.mean(axis=1)
    for row in source_marked_rows:
        triangle = source_triangles[row]
        lower = triangle.min(axis=0) - tolerance
        upper = triangle.max(axis=0) + tolerance
        candidates = np.flatnonzero(
            np.logical_and(target_centres >= lower, target_centres <= upper).all(axis=1)
        )
        if candidates.size == 0:
            continue
        origin = triangle[0]
        edge_1 = triangle[1] - origin
        edge_2 = triangle[2] - origin
        determinant = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if abs(float(determinant)) <= tolerance:
            raise RuntimeError("Encountered a degenerate source triangle during causal inheritance.")
        offsets = target_centres[candidates] - origin
        barycentric_1 = (offsets[:, 0] * edge_2[1] - offsets[:, 1] * edge_2[0]) / determinant
        barycentric_2 = (edge_1[0] * offsets[:, 1] - edge_1[1] * offsets[:, 0]) / determinant
        barycentric_0 = 1.0 - barycentric_1 - barycentric_2
        inside = np.logical_and.reduce(
            (
                barycentric_0 >= -tolerance,
                barycentric_1 >= -tolerance,
                barycentric_2 >= -tolerance,
            )
        )
        inherited_rows[candidates[inside]] = True

    inherited = np.zeros(target_dg0.node_count, dtype=bool)
    inherited[target_cell_dofs[inherited_rows]] = True
    return inherited


def refine_time_grid(ts: np.ndarray, marked_slabs: set[int]) -> np.ndarray:
    """Bisect each marked interval ``I_n=[t_{n-1},t_n]`` exactly once."""
    if not marked_slabs:
        return np.asarray(ts, dtype=float)
    refined = [float(ts[0])]
    for n in range(1, len(ts)):
        if n in marked_slabs:
            refined.append(0.5 * (float(ts[n - 1]) + float(ts[n])))
        refined.append(float(ts[n]))
    return np.asarray(refined, dtype=float)
