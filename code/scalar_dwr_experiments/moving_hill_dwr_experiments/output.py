"""ParaView and CSV output callbacks for a completed adaptive iteration."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

import numpy as np
from firedrake import CellDiameter, Function, FunctionSpace, SpatialCoordinate, VTKFile
from firedrake.petsc import PETSc

from .time import evaluate_slab

if TYPE_CHECKING:
    from .estimate import BubbleEstimate


class AdaptiveOutput:
    """Write visual fields without coupling file-system concerns to assembly.

    With slabwise meshes, a single spatial indicator field would be
    misleading: different time slabs have different cells.  The ordinary VTK
    file therefore contains only the terminal state.  Separate per-slab VTK
    files contain the actual mesh, primal state, signed/absolute indicator,
    and the available entity components.
    """

    def __init__(
        self,
        prefix: str,
        enabled: bool = True,
        mode: str = "all",
        snapshot_times: tuple[float, ...] | None = None,
    ):
        self.prefix = str(prefix)
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.snapshot_times = (
            None
            if snapshot_times is None
            else tuple(float(value) for value in snapshot_times)
        )
        self.vtk_failed = False
        self.slab_history: list[dict[str, object]] = []

    def write_vtk(
        self,
        iteration: int,
        terminal_state: Function,
        final_time: float,
    ) -> None:
        """Write the terminal state on the final slab's actual spatial mesh."""
        if not self.enabled or self.vtk_failed or self.mode == "spacetime_only":
            return
        Path(self.prefix).parent.mkdir(parents=True, exist_ok=True)
        state_out = Function(terminal_state.function_space(), name="u_h_T")
        state_out.assign(terminal_state)
        try:
            VTKFile(f"{self.prefix}_{iteration}.pvd").write(
                state_out, time=float(final_time)
            )
        except ModuleNotFoundError as exception:
            self.vtk_failed = True
            PETSc.Sys.Print(
                "[ParaView output skipped] VTK is unavailable; CSV output remains complete. "
                f"Details: {exception}"
            )

    def write_slabwise_vtk(
        self,
        iteration: int,
        primal: dict,
        estimate: "BubbleEstimate",
        ts: np.ndarray,
    ) -> None:
        """Write actual slab meshes at real physical snapshot times.

        A slabwise-adaptive 2D problem is a nonconforming 3D space--time
        complex.  Separate files faithfully show the mesh carrying each
        ``eta[K,n]`` instead of falsely extruding one mesh through time.  With
        no ``snapshot_times`` this preserves the legacy one-file-per-slab
        behaviour.  Otherwise exactly one containing/nearest slab is selected
        for each requested time.
        """
        if not self.enabled or self.vtk_failed or self.mode == "spacetime_only":
            return
        try:
            selections = self._slab_output_selections(ts)
            snapshot_writer = (
                VTKFile(
                    f"{self.prefix}_iter_{iteration}_snapshots.pvd",
                    adaptive=True,
                )
                if self.snapshot_times is not None
                else None
            )
            for _requested_time, physical_time, n in selections:
                slab = primal["slabs"][n]
                state = Function(
                    primal["nodes"][n].function_space(), name="u_h"
                )
                if self.snapshot_times is None:
                    state.assign(primal["nodes"][n])
                    path = Path(f"{self.prefix}_iter_{iteration}_slab_{n}.pvd")
                    writer = VTKFile(str(path))
                else:
                    step = float(ts[n] - ts[n - 1])
                    reference_time = (physical_time - float(ts[n - 1])) / step
                    state.interpolate(evaluate_slab(slab, reference_time))
                    assert snapshot_writer is not None
                    writer = snapshot_writer
                fields = self._indicator_fields(primal, estimate, n)
                writer.write(state, *fields, time=float(physical_time))
        except ModuleNotFoundError as exception:
            self.vtk_failed = True
            PETSc.Sys.Print(f"[Slabwise ParaView output skipped] {exception}")

    @staticmethod
    def containing_or_nearest_slab(ts: np.ndarray, target: float) -> int:
        """Return the 1-based slab containing ``target``, clamped at the ends.

        At an interior time node the left slab is selected, so the output is
        the already-computed right trace at that physical time.  Targets
        outside the time grid select the nearest end slab.
        """
        values = np.asarray(ts, dtype=float)
        if (
            values.ndim != 1
            or values.size < 2
            or np.any(~np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError("ts must be finite and strictly increasing.")
        target = float(target)
        if not np.isfinite(target):
            raise ValueError("target must be finite.")
        return int(
            np.clip(
                np.searchsorted(values, target, side="left"),
                1,
                values.size - 1,
            )
        )

    def _slab_output_selections(
        self, ts: np.ndarray
    ) -> list[tuple[float, float, int]]:
        """Return ``(requested, physical, slab)`` VTK selections."""
        if self.snapshot_times is None:
            return [
                (float(ts[n]), float(ts[n]), n)
                for n in range(1, len(ts))
            ]
        selections = []
        for requested in self.snapshot_times:
            physical = float(requested)
            selections.append(
                (requested, physical, self.containing_or_nearest_slab(ts, physical))
            )
        return selections

    @staticmethod
    def _cell_field(mesh, name: str, values: np.ndarray) -> Function:
        field = Function(FunctionSpace(mesh, "DG", 0), name=name)
        field.dat.data[:] = np.asarray(values, dtype=field.dat.data.dtype)
        return field

    def _indicator_fields(
        self, primal: dict, estimate: "BubbleEstimate", n: int
    ) -> list[Function]:
        """Create all cell/entity fields used in moving-hill comparisons."""
        mesh = primal["slabs"][n]["mesh"]
        signed = np.asarray(estimate.eta_cell_slab_signed[n], dtype=float)
        arrays = [
            ("eta_K_signed", signed),
            ("eta_K_abs", np.abs(signed)),
            ("eta_component_abs_sum", estimate.eta_component_abs_cell[n]),
            ("eta_marking_score", estimate.eta_cell_slab_marking[n]),
            ("eta_volume", estimate.eta_volume_cell[n]),
            ("eta_spatial_trace", estimate.eta_spatial_facet_cell[n]),
            ("eta_temporal_trace", estimate.eta_temporal_facet_cell[n]),
            ("eta_mixed_ridge", estimate.eta_mixed_ridge_cell[n]),
        ]
        optional = [
            ("eta_primal_residual", estimate.eta_primal_cell),
            ("eta_adjoint_residual", estimate.eta_adjoint_cell),
            ("eta_galerkin_correction", estimate.eta_correction_cell),
        ]
        for name, by_slab in optional:
            if by_slab is not None and by_slab[n] is not None:
                arrays.append((name, by_slab[n]))
        return [
            self._cell_field(mesh, name, np.asarray(values, dtype=float))
            for name, values in arrays
            if values is not None
        ]

    def write_slab_metadata(
        self,
        iteration: int,
        ts: np.ndarray,
        estimate: "BubbleEstimate",
        marks: list[np.ndarray | None],
        time_marked: set[int],
    ) -> None:
        """Save complete per-slab metadata for one adaptive iteration."""
        snapshot_by_slab: dict[int, list[float]] = {}
        if self.snapshot_times is not None:
            for requested, _, slab in self._slab_output_selections(ts):
                snapshot_by_slab.setdefault(slab, []).append(requested)
        rows = []
        for n in range(1, len(ts)):
            values = np.asarray(estimate.eta_cell_slab_signed[n], dtype=float)
            slab_marks = np.asarray(marks[n], dtype=bool)
            rows.append({
                "iteration": int(iteration),
                "slab_index": int(n),
                "t_left": float(ts[n - 1]),
                "t_right": float(ts[n]),
                "n_cells": int(values.size),
                "n_marked": int(np.count_nonzero(slab_marks)),
                "time_marked": int(n in time_marked),
                "eta_signed": float(values.sum()),
                "eta_abs": float(np.abs(values).sum()),
                "eta_component_abs": float(
                    np.asarray(estimate.eta_component_abs_cell[n]).sum()
                ),
                "eta_marking_activity": float(
                    np.abs(np.asarray(estimate.eta_cell_slab_marking[n])).sum()
                ),
                "indicator_semantics": estimate.indicator_semantics,
                "eta_weak_partition_signed": float(
                    np.asarray(estimate.eta_weak_cell_signed[n]).sum()
                ),
                "eta_volume_signed": float(
                    np.asarray(estimate.eta_volume_cell[n]).sum()
                ),
                "eta_spatial_trace_signed": float(
                    np.asarray(estimate.eta_spatial_facet_cell[n]).sum()
                ),
                "eta_temporal_trace_signed": float(
                    np.asarray(estimate.eta_temporal_facet_cell[n]).sum()
                ),
                "eta_mixed_ridge_signed": float(
                    np.asarray(estimate.eta_mixed_ridge_cell[n]).sum()
                ),
                "snapshot_times": " ".join(
                    f"{value:.16g}" for value in snapshot_by_slab.get(n, [])
                ),
            })
        path = Path(f"{self.prefix}_iter_{iteration}_slabs.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.slab_history.extend(rows)
        cumulative_path = Path(f"{self.prefix}_slab_metadata.csv")
        with cumulative_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(self.slab_history[0])
            )
            writer.writeheader()
            writer.writerows(self.slab_history)

    @staticmethod
    def _write_ascii_array(
        stream,
        name: str,
        values: np.ndarray,
        *,
        components: int = 1,
        vtk_type: str = "Float64",
    ) -> None:
        """Write a small, portable ASCII VTK array without an extra dependency."""
        if vtk_type == "Float64":
            flattened = np.asarray(values, dtype=float).reshape(-1)
            payload = " ".join(f"{value:.16e}" for value in flattened)
        else:
            flattened = np.asarray(values, dtype=int).reshape(-1)
            payload = " ".join(str(value) for value in flattened)
        stream.write(
            f'        <DataArray type="{vtk_type}" Name="{escape(name)}" '
            f'NumberOfComponents="{components}" format="ascii">{payload}</DataArray>\n'
        )

    def write_spacetime_vtk(
        self,
        iteration: int,
        primal: dict,
        estimate: "BubbleEstimate",
        ts: np.ndarray,
    ) -> None:
        r"""Write one genuine two-dimensional ``(x,t)`` VTK grid per iteration.

        Each physical cell ``K`` on a slab ``I_n`` becomes a quadrilateral
        ``K x I_n``.  This is deliberately a *space--time indicator* plot,
        not a fictitious extrusion of a final mesh: every slab contributes its
        own actual mesh.  The cell data keep the signed/local DWR components
        available in ParaView, while ``u_h_right`` records the primal trace at
        the right end of each slab.
        """
        if not self.enabled or self.vtk_failed:
            return
        if len(primal["slabs"]) <= 1:
            return
        first_mesh = primal["slabs"][1]["mesh"]
        if first_mesh.topological_dimension != 1:
            return
        try:
            points: list[tuple[float, float, float]] = []
            connectivity: list[tuple[int, int, int, int]] = []
            fields: dict[str, list[float]] = {
                "u_h_right": [], "eta_Kn": [], "eta_K_abs": [],
                "eta_component_abs_sum": [], "eta_marking_score": [],
                "eta_volume": [], "eta_spatial_trace": [],
                "eta_temporal_trace": [], "eta_mixed_ridge": [],
                "eta_primal_residual": [], "eta_adjoint_residual": [],
                "eta_galerkin_correction": [], "h_K": [], "slab_index": [],
            }
            for n in range(1, len(primal["slabs"])):
                slab = primal["slabs"][n]
                mesh = slab["mesh"]
                DG0 = FunctionSpace(mesh, "DG", 0)
                x, = SpatialCoordinate(mesh)
                x_cell = Function(DG0, name="x_cell_centre")
                x_cell.interpolate(x)
                h_cell = Function(DG0, name="h_K")
                h_cell.interpolate(CellDiameter(mesh))
                order = np.argsort(np.asarray(x_cell.dat.data_ro, dtype=float))
                x_values = np.asarray(x_cell.dat.data_ro, dtype=float)[order]
                h_values = np.asarray(h_cell.dat.data_ro, dtype=float)[order]
                state = Function(DG0, name="u_h_right_DG0")
                state.interpolate(primal["nodes"][n])
                field_values = {
                    "u_h_right": np.asarray(state.dat.data_ro, dtype=float)[order],
                    "eta_Kn": np.asarray(estimate.eta_cell_slab_signed[n], dtype=float)[order],
                    "eta_K_abs": np.abs(np.asarray(estimate.eta_cell_slab_signed[n], dtype=float))[order],
                    "eta_component_abs_sum": np.asarray(estimate.eta_component_abs_cell[n], dtype=float)[order],
                    "eta_marking_score": np.asarray(estimate.eta_cell_slab_marking[n], dtype=float)[order],
                    "eta_volume": np.asarray(estimate.eta_volume_cell[n], dtype=float)[order],
                    "eta_spatial_trace": np.asarray(estimate.eta_spatial_facet_cell[n], dtype=float)[order],
                    "eta_temporal_trace": np.asarray(estimate.eta_temporal_facet_cell[n], dtype=float)[order],
                    "eta_mixed_ridge": np.asarray(estimate.eta_mixed_ridge_cell[n], dtype=float)[order],
                    "eta_primal_residual": np.asarray(estimate.eta_primal_cell[n], dtype=float)[order],
                    "eta_adjoint_residual": np.asarray(estimate.eta_adjoint_cell[n], dtype=float)[order],
                    "eta_galerkin_correction": np.asarray(estimate.eta_correction_cell[n], dtype=float)[order],
                    "h_K": h_values,
                    "slab_index": np.full(x_values.size, float(n)),
                }
                for index, (x_mid, h_value) in enumerate(zip(x_values, h_values)):
                    x_left, x_right = float(x_mid - 0.5 * h_value), float(x_mid + 0.5 * h_value)
                    point_offset = len(points)
                    points.extend([
                        (x_left, float(ts[n - 1]), 0.0), (x_right, float(ts[n - 1]), 0.0),
                        (x_right, float(ts[n]), 0.0), (x_left, float(ts[n]), 0.0),
                    ])
                    connectivity.append((point_offset, point_offset + 1, point_offset + 2, point_offset + 3))
                    for name, values in field_values.items():
                        fields[name].append(float(values[index]))

            Path(self.prefix).parent.mkdir(parents=True, exist_ok=True)
            path = Path(f"{self.prefix}_spacetime_iter_{iteration}.vtu")
            with path.open("w", encoding="utf-8") as stream:
                stream.write('<?xml version="1.0"?>\n')
                stream.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
                stream.write("  <UnstructuredGrid>\n")
                stream.write(
                    f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(connectivity)}">\n'
                )
                stream.write("      <Points>\n")
                self._write_ascii_array(stream, "Points", np.asarray(points), components=3)
                stream.write("      </Points>\n      <Cells>\n")
                self._write_ascii_array(
                    stream, "connectivity", np.asarray(connectivity, dtype=int), vtk_type="Int32"
                )
                offsets = 4 * np.arange(1, len(connectivity) + 1, dtype=int)
                self._write_ascii_array(stream, "offsets", offsets, vtk_type="Int32")
                self._write_ascii_array(stream, "types", np.full(len(connectivity), 9, dtype=int), vtk_type="UInt8")
                stream.write("      </Cells>\n      <CellData>\n")
                for name, values in fields.items():
                    self._write_ascii_array(stream, name, np.asarray(values))
                stream.write("      </CellData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")
        except (ImportError, ModuleNotFoundError) as exception:
            self.vtk_failed = True
            PETSc.Sys.Print(f"[Space--time VTK output skipped] {exception}")

    def write_history(self, history: list[dict]) -> None:
        """Write scalar convergence data, one row per adaptive iteration."""
        if not history:
            return
        history_path = Path(f"{self.prefix}_history.csv")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    def write_iteration_collection(
        self, niterations: int, physical_time: float
    ) -> None:
        """Collect terminal meshes at their common real physical time.

        The adaptive iteration is stored in ``group`` rather than being
        misrepresented as a physical PVD time coordinate.
        """
        if not self.enabled or self.vtk_failed:
            return
        prefix_path = Path(self.prefix)
        output_dir = prefix_path.parent
        datasets: list[tuple[int, Path]] = []
        for iteration in range(niterations):
            iteration_dir = output_dir / f"{prefix_path.name}_{iteration}"
            datasets.extend((iteration, path) for path in sorted(iteration_dir.glob("*.vtu")))
        if not datasets:
            return
        collection_path = output_dir / f"{prefix_path.name}_iterations.pvd"
        with collection_path.open("w", encoding="utf-8") as stream:
            stream.write('<?xml version="1.0"?>\n')
            stream.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            stream.write("  <Collection>\n")
            for iteration, dataset in datasets:
                relative_path = dataset.relative_to(output_dir)
                stream.write(
                    f'    <DataSet timestep="{float(physical_time):.16g}" '
                    f'group="adaptation_iteration_{iteration}" part="0" '
                    f'file="{escape(str(relative_path))}"/>\n'
                )
            stream.write("  </Collection>\n</VTKFile>\n")

    def write_spacetime_collection(
        self, niterations: int, physical_time: float
    ) -> None:
        """Collect whole space--time grids without faking iteration as time."""
        if not self.enabled or self.vtk_failed:
            return
        prefix_path = Path(self.prefix)
        output_dir = prefix_path.parent
        paths = [
            output_dir / f"{prefix_path.name}_spacetime_iter_{iteration}.vtu"
            for iteration in range(niterations)
        ]
        paths = [path for path in paths if path.exists()]
        if not paths:
            return
        collection_path = output_dir / f"{prefix_path.name}_spacetime_iterations.pvd"
        with collection_path.open("w", encoding="utf-8") as stream:
            stream.write('<?xml version="1.0"?>\n')
            stream.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            stream.write("  <Collection>\n")
            for iteration, path in enumerate(paths):
                stream.write(
                    f'    <DataSet timestep="{float(physical_time):.16g}" '
                    f'group="adaptation_iteration_{iteration}" part="0" '
                    f'file="{escape(path.name)}"/>\n'
                )
            stream.write("  </Collection>\n</VTKFile>\n")
