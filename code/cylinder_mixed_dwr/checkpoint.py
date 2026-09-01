r"""Durable, replayable grid checkpoints for the mixed cylinder DWR loop.

Firedrake's HDF5 checkpoint restores a DMPlex mesh, but it does not restore
the originating Netgen object required by ``refine_marked_elements``.  Each
checkpoint therefore contains both an exact HDF5 audit copy of every unique
slab mesh and the complete marked-refinement lineage from the initial Netgen
grid.  Resume uses the lineage, so local refinement remains available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from firedrake import CheckpointFile
from mpi4py import MPI
from navier_stokes_cylinder_irksome_static_primal import _make_hierarchy

from .adaptive import (
    CylinderAdaptiveConfig,
    refine_causal_slab_grid,
    refine_common_slab_grid,
    refine_independent_slab_grid,
)


SCHEMA_VERSION = 1


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class LoadedCylinderGrid:
    """A reconstructed, still-Netgen-refinable checkpoint grid."""

    path: Path
    saved_config: dict[str, Any]
    grid_iteration: int
    times: np.ndarray
    meshes: list[Any | None]
    hierarchy: Any
    labels: dict[str, tuple[int, ...]]
    history: list[dict[str, Any]]
    refinement_lineage: list[dict[str, Any]]


class CylinderCheckpointStore:
    """Write immutable outer-grid checkpoints below one experiment root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, grid_iteration: int) -> Path:
        return self.root / f"iter_{int(grid_iteration):04d}"

    def save_grid(self, solver, grid_iteration: int) -> Path:
        """Save the grid that will be solved at ``grid_iteration``."""
        comm = MPI.COMM_WORLD
        target = self.path_for(grid_iteration)
        if comm.rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(
                    f"Checkpoint {target} already exists; checkpoints are immutable."
                )
            temporary = self.root / f".{target.name}.tmp-{uuid4().hex}"
            temporary.mkdir()
            temporary_text = str(temporary)
        else:
            temporary_text = None
        temporary = Path(comm.bcast(temporary_text, root=0))
        comm.barrier()

        unique_meshes: list[Any] = []
        identity_to_index: dict[int, int] = {}
        slab_mesh_indices: list[int | None] = [None]
        for mesh in solver.meshes[1:]:
            identity = id(mesh)
            if identity not in identity_to_index:
                identity_to_index[identity] = len(unique_meshes)
                unique_meshes.append(mesh)
            slab_mesh_indices.append(identity_to_index[identity])

        mesh_metadata = []
        for index, mesh in enumerate(unique_meshes):
            filename = f"mesh_{index:04d}.h5"
            path = temporary / filename
            with CheckpointFile(str(path), "w") as checkpoint:
                checkpoint.save_mesh(mesh)
            comm.barrier()
            metadata = {
                "index": index,
                "file": filename,
                "mesh_name": mesh.name,
                "num_cells_local": int(mesh.num_cells()),
            }
            if comm.rank == 0:
                metadata["sha256"] = _file_sha256(path)
            metadata = comm.bcast(metadata, root=0)
            mesh_metadata.append(metadata)

        lineage_metadata: list[dict[str, Any]] = []
        lineage_arrays: dict[str, np.ndarray] = {}
        for step, operation in enumerate(solver.refinement_lineage):
            marks = operation["marked"]
            mark_keys: list[str | None] = [None]
            for slab in range(1, len(marks)):
                key = f"step_{step:04d}_slab_{slab:04d}"
                lineage_arrays[key] = np.asarray(marks[slab], dtype=np.uint8)
                mark_keys.append(key)
            lineage_metadata.append(
                {
                    "source_iteration": int(operation["source_iteration"]),
                    "times_before": np.asarray(
                        operation["times_before"], dtype=float
                    ).tolist(),
                    "mark_keys": mark_keys,
                    "time_marked_fraction": float(
                        operation["time_marked_fraction"]
                    ),
                    "time_marking_strategy": operation.get(
                        "time_marking_strategy", "cell_fraction"
                    ),
                    "time_fixed_rate": float(
                        operation.get("time_fixed_rate", 0.20)
                    ),
                    "time_marked_override": (
                        None
                        if operation.get("time_marked_override") is None
                        else sorted(operation["time_marked_override"])
                    ),
                    "enable_space_refinement": bool(
                        operation["enable_space_refinement"]
                    ),
                    "enable_time_refinement": bool(
                        operation["enable_time_refinement"]
                    ),
                    "space_refinement_mode": operation.get(
                        "space_refinement_mode", "causal"
                    ),
                }
            )

        if comm.rank == 0:
            np.savez_compressed(temporary / "lineage_marks.npz", **lineage_arrays)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "grid_iteration": int(grid_iteration),
                "config": asdict(solver.config),
                "times": np.asarray(solver.times, dtype=float).tolist(),
                "slab_mesh_indices": slab_mesh_indices,
                "meshes": mesh_metadata,
                "labels": solver.labels,
                "history": solver.history,
                "refinement_lineage": lineage_metadata,
                "resume_mode": "replay_netgen_refinement_lineage",
                "hdf5_role": "exact_grid_audit_copy",
                "causal_mesh_identity_policy": (
                    "reuse_same_parent_and_identical_cumulative_marks"
                ),
            }
            (temporary / "grid.json").write_text(
                json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        comm.barrier()
        return target

    @staticmethod
    def read_manifest(path: str | Path) -> dict[str, Any]:
        checkpoint = Path(path)
        manifest = json.loads((checkpoint / "grid.json").read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cylinder checkpoint schema in {checkpoint}."
            )
        return manifest

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        replay_space_mode: str | None = None,
    ) -> LoadedCylinderGrid:
        """Reconstruct a checkpoint by replaying its complete Netgen lineage."""
        if replay_space_mode not in {None, "independent", "causal", "common"}:
            raise ValueError(
                "replay_space_mode must be None, 'independent', 'causal', or 'common'."
            )
        checkpoint = Path(path).resolve()
        manifest = cls.read_manifest(checkpoint)
        saved_config = dict(manifest["config"])
        hierarchy, inlet, wall, cylinder, outlet = _make_hierarchy(
            int(saved_config["hierarchy_levels"]),
            geometry_degree=int(saved_config.get("geometry_degree", 1)),
        )
        labels = {
            "inlet": tuple(inlet),
            "wall": tuple(wall),
            "cylinder": tuple(cylinder),
            "outlet": tuple(outlet),
        }
        times = np.linspace(
            0.0,
            float(saved_config["final_time"]),
            int(saved_config["initial_time_slabs"]) + 1,
        )
        meshes: list[Any | None] = [None] + [
            hierarchy[-1] for _ in range(int(saved_config["initial_time_slabs"]))
        ]
        lineage: list[dict[str, Any]] = []
        arrays_path = checkpoint / "lineage_marks.npz"
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for metadata in manifest["refinement_lineage"]:
                expected_times = np.asarray(metadata["times_before"], dtype=float)
                if not np.array_equal(times, expected_times):
                    raise ValueError("Checkpoint refinement lineage has inconsistent times.")
                marks: list[np.ndarray | None] = [None]
                for key in metadata["mark_keys"][1:]:
                    marks.append(np.asarray(arrays[key], dtype=bool).copy())
                operation = {
                    "source_iteration": int(metadata["source_iteration"]),
                    "times_before": expected_times.copy(),
                    "marked": marks,
                    "time_marked_fraction": float(
                        metadata["time_marked_fraction"]
                    ),
                    "time_marking_strategy": metadata.get(
                        "time_marking_strategy", "cell_fraction"
                    ),
                    "time_fixed_rate": float(
                        metadata.get("time_fixed_rate", 0.20)
                    ),
                    "time_marked_override": (
                        None
                        if metadata.get("time_marked_override") is None
                        else {
                            int(value)
                            for value in metadata["time_marked_override"]
                        }
                    ),
                    "enable_space_refinement": bool(
                        metadata["enable_space_refinement"]
                    ),
                    "enable_time_refinement": bool(
                        metadata["enable_time_refinement"]
                    ),
                    "space_refinement_mode": (
                        replay_space_mode
                        if replay_space_mode is not None
                        else metadata.get("space_refinement_mode", "causal")
                    ),
                }
                refinement_function = {
                    "common": refine_common_slab_grid,
                    "causal": refine_causal_slab_grid,
                    "independent": refine_independent_slab_grid,
                }[operation["space_refinement_mode"]]
                refinement = refinement_function(
                    meshes,
                    times,
                    marks,
                    time_marked_fraction=operation["time_marked_fraction"],
                    time_marking_strategy=operation["time_marking_strategy"],
                    time_fixed_rate=operation["time_fixed_rate"],
                    time_marked_override=operation["time_marked_override"],
                    enable_space_refinement=operation["enable_space_refinement"],
                    enable_time_refinement=operation["enable_time_refinement"],
                )
                times = refinement.times
                meshes = refinement.meshes
                lineage.append(operation)

        target_times = np.asarray(manifest["times"], dtype=float)
        if not np.array_equal(times, target_times):
            raise ValueError("Replayed checkpoint time grid does not match its manifest.")
        if replay_space_mode is None:
            slab_indices = manifest["slab_mesh_indices"]
            expected_cells = {
                int(item["index"]): int(item["num_cells_local"])
                for item in manifest["meshes"]
            }
            for slab in range(1, len(meshes)):
                if int(meshes[slab].num_cells()) != expected_cells[int(slab_indices[slab])]:
                    raise ValueError(
                        f"Replayed checkpoint mesh mismatch on slab {slab}."
                    )
        return LoadedCylinderGrid(
            path=checkpoint,
            saved_config=saved_config,
            grid_iteration=int(manifest["grid_iteration"]),
            times=times,
            meshes=meshes,
            hierarchy=hierarchy,
            labels=labels,
            history=[dict(row) for row in manifest["history"]],
            refinement_lineage=lineage,
        )


__all__ = [
    "CylinderCheckpointStore",
    "LoadedCylinderGrid",
    "SCHEMA_VERSION",
]
