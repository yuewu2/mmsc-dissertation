r"""Export a slabwise cylinder grid checkpoint for inspection in ParaView.

No primal or adjoint problem is solved.  The saved Netgen refinement lineage is
replayed, and one VTK data set is written for every time slab.  The top-level
``*_slabwise_meshes.pvd`` file opens the complete causal mesh sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from firedrake import CellSize, Function, FunctionSpace, VTKFile, project
from mpi4py import MPI

from .checkpoint import CylinderCheckpointStore


def _constant_cell_field(space, value: float, name: str) -> Function:
    field = Function(space, name=name)
    field.dat.data[:] = float(value)
    return field


def _cell_field(space, values, name: str) -> Function:
    field = Function(space, name=name)
    data = np.asarray(values, dtype=float)
    if data.size != field.dat.data.size:
        raise ValueError(
            f"Field {name} has {data.size} values, expected {field.dat.data.size}."
        )
    field.dat.data[:] = data
    return field


def _dataset_path(pvd_path: Path) -> str:
    tree = ET.parse(pvd_path)
    dataset = tree.find("./Collection/DataSet")
    if dataset is None or "file" not in dataset.attrib:
        raise ValueError(f"No VTK DataSet entry found in {pvd_path}.")
    return str((pvd_path.parent / dataset.attrib["file"]).relative_to(pvd_path.parent))


def _write_collection(path: Path, entries: list[tuple[float, int, str]]) -> None:
    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for time, slab, filename in entries:
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time:.17g}",
            group=f"slab_{slab:04d}",
            part="0",
            file=filename,
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--indicators",
        default=None,
        help="Optional production indicator NPZ on the checkpoint grid.",
    )
    args = parser.parse_args(argv)

    loaded = CylinderCheckpointStore.load(args.checkpoint)
    manifest = CylinderCheckpointStore.read_manifest(args.checkpoint)
    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    slab_mesh_indices = manifest["slab_mesh_indices"]
    collection_entries: list[tuple[float, int, str]] = []
    indicators = None
    if args.indicators is not None:
        indicators = np.load(args.indicators, allow_pickle=False)
        if not np.array_equal(
            np.asarray(indicators["times"], dtype=float), loaded.times
        ):
            indicators.close()
            raise ValueError("Indicator and checkpoint time grids do not match.")

    for slab in range(1, len(loaded.times)):
        mesh = loaded.meshes[slab]
        dg0 = FunctionSpace(mesh, "DG", 0)
        t_left = float(loaded.times[slab - 1])
        t_right = float(loaded.times[slab])
        mesh_id = int(slab_mesh_indices[slab])
        fields = [
            _constant_cell_field(dg0, slab, "slab_index"),
            _constant_cell_field(dg0, mesh_id, "mesh_id"),
            _constant_cell_field(dg0, t_left, "time_left"),
            _constant_cell_field(dg0, t_right, "time_right"),
            _constant_cell_field(dg0, t_right - t_left, "delta_t"),
        ]
        # UFL's CellSize lowering is only implemented for affine P1/Q1
        # geometry.  Keep curved Netgen geometry intact and omit this purely
        # diagnostic field instead of flattening a degree-2 cylinder mesh.
        coordinate_degree = mesh.coordinates.ufl_element().degree()
        if coordinate_degree == 1:
            cell_size = project(CellSize(mesh), dg0)
            cell_size.rename("cell_size_h")
            fields.insert(0, cell_size)
        if indicators is not None:
            indicator_fields = {
                "eta_signed": f"eta_signed_slab_{slab}",
                "eta_abs": f"eta_signed_slab_{slab}",
                "eta_primal": f"eta_primal_slab_{slab}",
                "eta_volume": f"eta_primal_volume_slab_{slab}",
                "eta_spatial_facet": f"eta_primal_spatial_slab_{slab}",
                "eta_temporal_jump": f"eta_primal_temporal_slab_{slab}",
                "eta_mixed_ridge": f"eta_primal_ridge_slab_{slab}",
                "marked_cell": f"marked_slab_{slab}",
            }
            for name, key in indicator_fields.items():
                if key not in indicators:
                    indicators.close()
                    raise KeyError(f"Indicator archive is missing {key}.")
                values = np.asarray(indicators[key], dtype=float)
                if name == "eta_abs":
                    values = np.abs(values)
                fields.append(_cell_field(dg0, values, name))
        slab_pvd = prefix.parent / f"{prefix.name}_slab_{slab:04d}.pvd"
        VTKFile(str(slab_pvd)).write(*fields, time=t_right)
        if MPI.COMM_WORLD.rank == 0:
            collection_entries.append(
                (t_right, slab, _dataset_path(slab_pvd))
            )

    if indicators is not None:
        indicators.close()

    if MPI.COMM_WORLD.rank == 0:
        collection = prefix.parent / f"{prefix.name}_slabwise_meshes.pvd"
        _write_collection(collection, collection_entries)
        summary = prefix.parent / f"{prefix.name}_mesh_summary.csv"
        lines = ["slab,time_left,time_right,delta_t,mesh_id,num_cells"]
        for slab in range(1, len(loaded.times)):
            lines.append(
                ",".join(
                    (
                        str(slab),
                        f"{loaded.times[slab - 1]:.17g}",
                        f"{loaded.times[slab]:.17g}",
                        f"{loaded.times[slab] - loaded.times[slab - 1]:.17g}",
                        str(int(slab_mesh_indices[slab])),
                        str(int(loaded.meshes[slab].num_cells())),
                    )
                )
            )
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(collection)
        print(summary)


if __name__ == "__main__":
    main()
