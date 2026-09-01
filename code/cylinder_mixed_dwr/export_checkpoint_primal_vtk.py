r"""Replay a cylinder checkpoint primal and export slabwise ParaView fields."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from firedrake import (
    CellSize,
    Function,
    FunctionSpace,
    VTKFile,
    curl,
    div,
    inner,
    project,
    sqrt,
)
from mpi4py import MPI

from .checkpoint import CylinderCheckpointStore
from .export_checkpoint_meshes import (
    _cell_field,
    _constant_cell_field,
    _dataset_path,
    _write_collection,
)
from .slabwise import build_slab_transfers, solve_slabwise_primal


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--indicators", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--lean",
        action="store_true",
        help="Write only Velocity and Speed for a compact ParaView animation.",
    )
    args = parser.parse_args(argv)

    loaded = CylinderCheckpointStore.load(args.checkpoint)
    config = loaded.saved_config
    indicators = np.load(args.indicators, allow_pickle=False)
    if not np.array_equal(np.asarray(indicators["times"]), loaded.times):
        indicators.close()
        raise ValueError("Indicator and checkpoint time grids do not match.")

    transfers = build_slab_transfers(
        loaded.meshes,
        loaded.labels,
        mode=str(config["interface_transfer_mode"]),
        low_family=str(config["primal_space_family"]),
        enriched_velocity_degree=int(config["enriched_velocity_degree"]),
    )
    primal = solve_slabwise_primal(
        loaded.meshes,
        loaded.times,
        transfers,
        loaded.labels,
        viscosity=float(config["viscosity"]),
        time_degree=int(config["primal_time_degree"]),
        report_every=int(args.report_every),
    )

    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    summary_rows = []
    for slab_number in range(1, len(loaded.times)):
        slab = primal["slabs"][slab_number]
        mesh = slab["mesh"]
        state = slab["right_trace"]
        velocity = Function(
            state.subfunctions[0].function_space(), name="Velocity"
        ).assign(state.subfunctions[0])
        cg1 = FunctionSpace(mesh, "CG", 1)
        speed = project(sqrt(inner(velocity, velocity)), cg1, name="Speed")
        eta = np.asarray(indicators[f"eta_signed_slab_{slab_number}"], dtype=float)
        marked = np.asarray(indicators[f"marked_slab_{slab_number}"], dtype=float)
        t_left = float(loaded.times[slab_number - 1])
        t_right = float(loaded.times[slab_number])
        if args.lean:
            fields = [velocity, speed]
        else:
            pressure = Function(
                state.subfunctions[1].function_space(), name="Pressure"
            ).assign(state.subfunctions[1])
            dg0 = FunctionSpace(mesh, "DG", 0)
            vorticity = project(curl(velocity), cg1, name="Vorticity")
            divergence = project(div(velocity), dg0, name="Divergence")
            cell_size = project(CellSize(mesh), dg0, name="CellSize")
            fields = [
                velocity,
                speed,
                pressure,
                vorticity,
                divergence,
                cell_size,
                _cell_field(dg0, eta, "BubbleConeEtaSigned"),
                _cell_field(dg0, np.abs(eta), "BubbleConeEtaAbsolute"),
                _cell_field(dg0, marked, "MarkedCell"),
                _constant_cell_field(dg0, slab_number, "SlabIndex"),
                _constant_cell_field(dg0, t_right - t_left, "DeltaT"),
            ]
        slab_pvd = prefix.parent / f"{prefix.name}_slab_{slab_number:04d}.pvd"
        VTKFile(str(slab_pvd)).write(*fields, time=t_right)
        if MPI.COMM_WORLD.rank == 0:
            entries.append((t_right, slab_number, _dataset_path(slab_pvd)))
            summary_rows.append(
                {
                    "slab": slab_number,
                    "time_left": t_left,
                    "time_right": t_right,
                    "delta_t": t_right - t_left,
                    "num_cells": int(mesh.num_cells()),
                    "eta_signed_sum": float(eta.sum()),
                    "eta_absolute_sum": float(np.abs(eta).sum()),
                    "marked_cells": int(np.count_nonzero(marked)),
                }
            )

    indicators.close()
    if MPI.COMM_WORLD.rank == 0:
        collection = prefix.parent / f"{prefix.name}_fields.pvd"
        _write_collection(collection, entries)
        summary = prefix.parent / f"{prefix.name}_summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(collection)
        print(summary)


if __name__ == "__main__":
    main()
