"""Restartable adaptive-grid checkpoints for the slabwise DWR solver."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from firedrake import CheckpointFile


_FORMAT_VERSION = 1
_OPTION_EXCLUSIONS = {
    "max_it",
    "checkpoint_prefix",
    "restart_from",
    "checkpoint_every",
    "verbose",
}


def _jsonable(value: Any) -> Any:
    """Convert scalar configuration/history data to JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _option_signature(options) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in asdict(options).items()
        if key not in _OPTION_EXCLUSIONS
    }


def _problem_signature(problem) -> dict[str, Any]:
    return {
        "class": f"{type(problem).__module__}.{type(problem).__qualname__}",
        "parameters": _jsonable(vars(problem)),
    }


class AdaptiveCheckpoint:
    """Save and restore the adaptive grid at the start of an iteration.

    The checkpoint intentionally stores meshes and adaptive metadata, rather
    than transient primal/adjoint stage vectors.  Every adaptive iteration
    already resolves the complete forward and reverse problems on its current
    grid, so a restart remains deterministic without serialising solver state.
    """

    def __init__(self, prefix: str):
        if not prefix:
            raise ValueError("A non-empty checkpoint prefix is required.")
        self.prefix = Path(prefix)

    @property
    def pointer_path(self) -> Path:
        return Path(f"{self.prefix}_latest.json")

    def _paths(self, next_iteration: int) -> tuple[Path, Path]:
        stem = Path(f"{self.prefix}_iter_{int(next_iteration):05d}")
        return stem.with_suffix(".h5"), stem.with_suffix(".json")

    def save(self, *, next_iteration: int, ts, meshes, history, options, problem) -> Path:
        """Atomically publish a checkpoint for ``next_iteration``."""
        h5_path, metadata_path = self._paths(next_iteration)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        h5_tmp = h5_path.with_suffix(".tmp.h5")
        metadata_tmp = metadata_path.with_suffix(".tmp.json")
        pointer_tmp = self.pointer_path.with_suffix(".tmp.json")

        unique_meshes: list[Any] = []
        name_by_identity: dict[int, str] = {}
        slab_mesh_names: list[str] = []
        for mesh in meshes[1:]:
            identity = id(mesh)
            if identity not in name_by_identity:
                name_by_identity[identity] = f"checkpoint_mesh_{len(unique_meshes):05d}"
                unique_meshes.append(mesh)
            slab_mesh_names.append(name_by_identity[identity])

        original_names = [mesh.name for mesh in unique_meshes]
        try:
            with CheckpointFile(str(h5_tmp), "w") as checkpoint:
                for mesh, name in zip(unique_meshes, name_by_identity.values()):
                    mesh.name = name
                    checkpoint.save_mesh(mesh)
        finally:
            for mesh, name in zip(unique_meshes, original_names):
                mesh.name = name

        metadata = {
            "format_version": _FORMAT_VERSION,
            "next_iteration": int(next_iteration),
            "mesh_file": h5_path.name,
            "time_grid": _jsonable(np.asarray(ts, dtype=float)),
            "slab_mesh_names": slab_mesh_names,
            "history": _jsonable(history),
            "option_signature": _option_signature(options),
            "problem_signature": _problem_signature(problem),
        }
        metadata_tmp.write_text(json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8")
        os.replace(h5_tmp, h5_path)
        os.replace(metadata_tmp, metadata_path)
        pointer_tmp.write_text(
            json.dumps({"metadata_file": metadata_path.name}, indent=2), encoding="utf-8"
        )
        os.replace(pointer_tmp, self.pointer_path)

        # The pointer now names a complete new checkpoint.  Older generations
        # are no longer needed, but retaining one previous generation makes a
        # manually selected rollback possible after an interrupted solve.
        generations = sorted(self.prefix.parent.glob(f"{self.prefix.name}_iter_*.json"))
        for old_metadata in generations[:-2]:
            try:
                old = json.loads(old_metadata.read_text(encoding="utf-8"))
                old_h5 = old_metadata.parent / old.get("mesh_file", "")
                old_metadata.unlink(missing_ok=True)
                if old_h5.is_file():
                    old_h5.unlink()
            except (OSError, ValueError, json.JSONDecodeError):
                # A stale generation is harmless; never risk the new pointer
                # merely to tidy an older checkpoint.
                pass
        return metadata_path

    def _resolve_metadata_path(self) -> Path:
        candidate = Path(self.prefix)
        if candidate.is_file() and candidate.suffix == ".json":
            pointer_or_metadata = candidate
        else:
            pointer_or_metadata = self.pointer_path
        if not pointer_or_metadata.is_file():
            raise FileNotFoundError(f"Checkpoint metadata not found: {pointer_or_metadata}")
        data = json.loads(pointer_or_metadata.read_text(encoding="utf-8"))
        if "metadata_file" in data:
            return pointer_or_metadata.parent / data["metadata_file"]
        return pointer_or_metadata

    def load(self, *, options, problem) -> dict[str, Any]:
        """Load and validate the latest complete checkpoint generation."""
        metadata_path = self._resolve_metadata_path()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format {metadata.get('format_version')}; "
                f"expected {_FORMAT_VERSION}."
            )
        expected_options = _option_signature(options)
        if metadata.get("option_signature") != expected_options:
            raise ValueError(
                "Restart options differ from the checkpoint configuration. "
                "Only max_it, verbosity, and checkpoint controls may change."
            )
        expected_problem = _problem_signature(problem)
        if metadata.get("problem_signature") != expected_problem:
            raise ValueError("Restart problem parameters differ from the checkpoint.")

        h5_path = metadata_path.parent / metadata["mesh_file"]
        if not h5_path.is_file():
            raise FileNotFoundError(f"Checkpoint mesh file not found: {h5_path}")
        loaded_by_name: dict[str, Any] = {}
        with CheckpointFile(str(h5_path), "r") as checkpoint:
            for name in dict.fromkeys(metadata["slab_mesh_names"]):
                loaded_by_name[name] = checkpoint.load_mesh(name)
        meshes = [None] + [loaded_by_name[name] for name in metadata["slab_mesh_names"]]
        ts = np.asarray(metadata["time_grid"], dtype=float)
        if len(meshes) != len(ts):
            raise ValueError(
                "Checkpoint is inconsistent: one spatial mesh is required for each time slab."
            )
        return {
            "next_iteration": int(metadata["next_iteration"]),
            "ts": ts,
            "meshes": meshes,
            "history": list(metadata.get("history", [])),
            "metadata_path": metadata_path,
        }
