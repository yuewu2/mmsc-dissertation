from __future__ import annotations

"""Mesh construction, marking, and h-refinement shared by both estimators."""

import numpy as np

from firedrake import Function, Mesh, UnitSquareMesh
from firedrake.petsc import PETSc

from heat_dwr_irksome import A_BOX, B_BOX


def create_box_fitted_mesh(nx: int = 8, ny: int = 8) -> Mesh:
    """Create a triangular mesh fitted to the observation-box coordinates."""
    try:
        from netgen.geom2d import SplineGeometry

        geo = SplineGeometry()
        geo.AddRectangle(
            p1=(0.0, 0.0),
            p2=(1.0, 1.0),
            bc="boundary",
            leftdomain=1,
            rightdomain=0,
        )
        geo.AddRectangle(
            p1=(A_BOX, A_BOX),
            p2=(B_BOX, B_BOX),
            bc="box_interface",
            leftdomain=2,
            rightdomain=1,
        )
        geo.SetMaterial(1, "outer")
        geo.SetMaterial(2, "inner")
        return Mesh(geo.GenerateMesh(maxh=1.0 / max(nx, ny)))
    except ImportError:
        PETSc.Sys.Print(
            "WARNING: Netgen/ngsPETSc is unavailable. The estimator can run, "
            "but marked spatial refinement requires `pip install ngsPETSc`."
        )
        if nx % 4 or ny % 4:
            PETSc.Sys.Print(
                "WARNING: choose nx and ny divisible by 4 for an exactly "
                "box-fitted fallback mesh."
            )
        return UnitSquareMesh(int(nx), int(ny), diagonal="left")


def mark_by_bulk(values: np.ndarray, fraction: float) -> np.ndarray:
    """Mark the largest nonnegative values until a bulk fraction is reached."""
    scores = np.maximum(np.asarray(values, dtype=float), 0.0)
    marked = np.zeros(scores.shape, dtype=bool)
    total = float(scores.sum())
    if total <= 0.0:
        return marked

    order = np.argsort(scores)[::-1]
    target = float(fraction) * total
    subtotal = 0.0
    for index in order:
        marked[index] = True
        subtotal += float(scores[index])
        if subtotal >= target:
            break
    return marked


def mark_spacetime_cells(
    eta_cell_slab_signed: list[np.ndarray | None],
    theta: float,
) -> list[np.ndarray | None]:
    r"""One global Doerfler marking operation on ``|eta[K,n]|``.

    The slabs are deliberately allowed to contain different numbers of
    spatial cells.  This is required by a genuine slabwise space--time mesh:
    once ``T_n`` is refined independently of ``T_m``, the vectors
    ``eta[:, n]`` and ``eta[:, m]`` no longer have equal lengths.  The bulk
    criterion is nevertheless unchanged,

    .. math::

       \sum_{(K,n)\in\mathcal M}|\eta_{K,n}|
       \geq \theta\sum_{K,n}|\eta_{K,n}|.
    """
    slab_arrays = [np.asarray(v, dtype=float) for v in eta_cell_slab_signed[1:]]
    if not slab_arrays:
        return [None]

    scores = np.concatenate([np.abs(values) for values in slab_arrays])
    flat_marked = mark_by_bulk(scores, theta)
    marked: list[np.ndarray | None] = [None]
    start = 0
    for values in slab_arrays:
        stop = start + values.size
        marked.append(flat_marked[start:stop].copy())
        start = stop
    return marked


def refinement_from_spacetime_marks(
    marked_by_slab: list[np.ndarray | None],
    DG0,
    time_slab_marked_fraction: float,
) -> tuple[Function, set[int], list[float]]:
    """Union spatial marks and apply the marked-cell-fraction time rule."""
    if len(marked_by_slab) <= 1:
        markers = Function(DG0, name="space_markers")
        markers.assign(0.0)
        return markers, set(), [0.0]

    slab_masks = [np.asarray(mask, dtype=bool) for mask in marked_by_slab[1:]]
    spatial_union = np.logical_or.reduce(slab_masks)
    markers = Function(DG0, name="space_markers")
    markers.dat.data[:] = spatial_union.astype(markers.dat.data.dtype)

    time_marked: set[int] = set()
    marked_fractions = [0.0]
    for n, mask in enumerate(slab_masks, start=1):
        fraction = float(np.count_nonzero(mask)) / float(mask.size)
        marked_fractions.append(fraction)
        if np.any(mask) and fraction >= float(time_slab_marked_fraction):
            time_marked.add(n)
    return markers, time_marked, marked_fractions


def refine_marked_mesh(mesh: Mesh, markers: Function) -> Mesh:
    """Apply Netgen/Firedrake marked-cell refinement."""
    if hasattr(mesh, "refine_marked_elements"):
        try:
            return mesh.refine_marked_elements(markers)
        except ImportError as exc:
            raise RuntimeError(
                "Marked spatial refinement needs Firedrake's Netgen support. "
                "Activate /home/yue/venv-firedrake and run `pip install "
                "ngsPETSc`, then rerun this script."
            ) from exc
    raise RuntimeError("The current mesh does not support refine_marked_elements.")


def refine_time_grid(ts: np.ndarray, marked_slabs: set[int]) -> np.ndarray:
    """Bisect each selected time slab exactly once."""
    if not marked_slabs:
        return np.asarray(ts, dtype=float)
    new_ts = [float(ts[0])]
    for n in range(1, len(ts)):
        if n in marked_slabs:
            new_ts.append(0.5 * (float(ts[n - 1]) + float(ts[n])))
        new_ts.append(float(ts[n]))
    return np.asarray(new_ts, dtype=float)
