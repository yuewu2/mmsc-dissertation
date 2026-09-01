"""Signed space--time indicators and product-space marking."""

from __future__ import annotations

import numpy as np


def mark_by_bulk(scores: np.ndarray, fraction: float) -> np.ndarray:
    r"""Return the smallest descending prefix containing the requested bulk."""
    values = np.maximum(np.asarray(scores, dtype=float), 0.0)
    marked = np.zeros(values.shape, dtype=bool)
    total = float(values.sum())
    if total <= 0.0:
        return marked
    target = float(fraction) * total
    subtotal = 0.0
    for index in np.argsort(values)[::-1]:
        marked[index] = True
        subtotal += float(values[index])
        if subtotal >= target:
            break
    return marked


def mark_by_fixed_rate(scores: np.ndarray, fraction: float) -> np.ndarray:
    r"""Mark the largest ``ceil(fraction*N)`` positive entries by count."""
    values = np.maximum(np.asarray(scores, dtype=float), 0.0)
    marked = np.zeros(values.shape, dtype=bool)
    positive = np.flatnonzero(values > 0.0)
    if positive.size == 0 or fraction <= 0.0:
        return marked
    count = min(positive.size, max(1, int(np.ceil(float(fraction) * values.size))))
    order = positive[np.argsort(values[positive])[::-1]]
    marked[order[:count]] = True
    return marked


def mark_spacetime_cells(
    eta_cell_slab_signed: list[np.ndarray | None], theta: float
) -> list[np.ndarray | None]:
    r"""Apply one Dörfler operation to all ``abs(eta[K,n])`` values."""
    slab_values = [
        np.asarray(values, dtype=float) for values in eta_cell_slab_signed[1:]
    ]
    if not slab_values:
        return [None]
    flat = np.concatenate([np.abs(values) for values in slab_values])
    flat_marks = mark_by_bulk(flat, theta)
    marked: list[np.ndarray | None] = [None]
    start = 0
    for values in slab_values:
        stop = start + values.size
        marked.append(flat_marks[start:stop].copy())
        start = stop
    return marked


def marked_slab_fractions(
    marked_by_slab: list[np.ndarray | None], threshold: float
) -> tuple[set[int], list[float]]:
    """Select time slabs using the current marked-cell-fraction policy."""
    selected: set[int] = set()
    fractions = [0.0]
    for slab, mask in enumerate(marked_by_slab[1:], start=1):
        values = np.asarray(mask, dtype=bool)
        fraction = float(np.count_nonzero(values)) / float(values.size)
        fractions.append(fraction)
        if np.any(values) and fraction >= float(threshold):
            selected.add(slab)
    return selected, fractions


def marked_slab_global_shares(
    marked_by_slab: list[np.ndarray | None], threshold: float
) -> tuple[set[int], list[float]]:
    """Select slabs by their share of all marked space--time cells.

    Unlike :func:`marked_slab_fractions`, the denominator is the number of
    marked cells over the whole current space--time mesh, not the number of
    spatial cells in the individual slab.  This is a concentration trigger;
    it deliberately does not claim to separate spatial and temporal error.
    """
    counts = np.asarray(
        [np.count_nonzero(np.asarray(mask, dtype=bool)) for mask in marked_by_slab[1:]],
        dtype=float,
    )
    total = float(counts.sum())
    shares = np.zeros_like(counts) if total <= 0.0 else counts / total
    selected = {
        int(index + 1)
        for index in np.flatnonzero((counts > 0.0) & (shares >= float(threshold)))
    }
    return selected, [0.0] + shares.tolist()


def mark_each_slab(
    eta_cell_slab_signed: list[np.ndarray | None],
    fraction: float,
    *,
    fixed_rate: bool,
) -> list[np.ndarray | None]:
    """Mark spatial cells independently on every slab."""
    marker = mark_by_fixed_rate if fixed_rate else mark_by_bulk
    return [None] + [
        marker(np.abs(np.asarray(values, dtype=float)), fraction)
        for values in eta_cell_slab_signed[1:]
    ]


def mark_slabs_by_activity(
    slab_activity: list[float], fraction: float, *, fixed_rate: bool
) -> set[int]:
    """Select one-based slab numbers from aggregated absolute activity."""
    values = np.asarray(slab_activity[1:], dtype=float)
    marker = mark_by_fixed_rate if fixed_rate else mark_by_bulk
    mask = marker(values, fraction)
    return {int(index + 1) for index in np.flatnonzero(mask)}


__all__ = [
    "mark_by_bulk",
    "mark_by_fixed_rate",
    "mark_each_slab",
    "mark_slabs_by_activity",
    "mark_spacetime_cells",
    "marked_slab_global_shares",
    "marked_slab_fractions",
]
