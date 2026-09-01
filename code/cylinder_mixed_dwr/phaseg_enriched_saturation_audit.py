"""Frozen-grid enriched-primal saturation audit without DWR or adaptation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from firedrake import FunctionSpace
from firedrake.petsc import PETSc

from navier_stokes_cylinder_irksome_static_primal import StaticCylinderParameters

from .adaptive import refine_independent_slab_grid
from .benchmark import drag_history_diagnostics, mean_drag_from_history
from .checkpoint import CylinderCheckpointStore
from .slabwise import build_slab_transfers, solve_slabwise_enriched_primal


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _history_row(path: Path, iteration: int) -> dict:
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if int(row["iteration"]) == int(iteration):
            return row
    raise ValueError(f"Iteration {iteration} is absent from {path}.")


def _uniform_time_split(meshes, times, factor: int):
    factor = int(factor)
    if factor < 2:
        raise ValueError("The uniform temporal refinement factor must be at least two.")
    new_times = [float(times[0])]
    new_meshes = [None]
    for slab in range(1, len(times)):
        left = float(times[slab - 1])
        right = float(times[slab])
        for child in range(1, factor + 1):
            new_times.append(left + (right - left) * child / factor)
            new_meshes.append(meshes[slab])
    return new_meshes, np.asarray(new_times, dtype=float)


def _uniform_space_refine(meshes, times):
    marks = [None]
    for slab in range(1, len(times)):
        count = FunctionSpace(meshes[slab], "DG", 0).node_count
        marks.append(np.ones(count, dtype=bool))
    refined = refine_independent_slab_grid(
        meshes,
        times,
        marks,
        time_marked_fraction=1.0,
        time_marking_strategy="cell_fraction",
        time_marked_override=set(),
        enable_space_refinement=True,
        enable_time_refinement=False,
    )
    return refined.meshes, refined.times


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--variant",
        # "plain" keeps the checkpoint grid unchanged: it solves the
        # enriched primal in exactly the enrichment used by the production
        # estimator, which is the pairing needed to test the one-term
        # identity eta ~= J(u_H) - J(u_h) and its quadratic remainder.
        choices=("plain", "uniform_time", "uniform_space", "p4p3"),
        required=True,
    )
    parser.add_argument("--quadrature-points", type=int, default=7)
    parser.add_argument(
        "--time-refinement-factor",
        type=int,
        default=2,
        help="Uniform enriched-primal child slabs per checkpoint slab.",
    )
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument(
        "--boundary-time-degree",
        type=int,
        default=None,
        help=(
            "Temporal degree of the slabwise L2 inlet lifting.  Default "
            "(None) keeps the enriched time degree, i.e. a saturation test "
            "against the continuous problem.  Pass the primal time degree "
            "(1) to share the low solve's lifting, which is the pairing "
            "required by the discrete DWR identity "
            "J(u_H) - J(u_h) = rho(u_h)(z_H) + remainder."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    started = perf_counter()
    loaded = CylinderCheckpointStore.load(args.checkpoint)
    config = loaded.saved_config
    baseline = _history_row(args.history, args.iteration)
    meshes = loaded.meshes
    times = loaded.times
    velocity_degree = int(config["enriched_velocity_degree"])
    pressure_degree = int(config["enriched_pressure_degree"])
    time_degree = int(config["enriched_time_degree"])
    boundary_time_degree = (
        time_degree
        if args.boundary_time_degree is None
        else int(args.boundary_time_degree)
    )

    if args.variant == "uniform_time":
        meshes, times = _uniform_time_split(
            meshes, times, int(args.time_refinement_factor)
        )
    elif args.variant == "uniform_space":
        meshes, times = _uniform_space_refine(meshes, times)
    elif args.variant == "p4p3":
        velocity_degree, pressure_degree = 4, 3

    result = {
        "experiment": "frozen_grid_enriched_primal_saturation",
        "status": "running",
        "stage": "grid_ready",
        "variant": args.variant,
        "time_refinement_factor": (
            int(args.time_refinement_factor)
            if args.variant == "uniform_time"
            else 1
        ),
        "checkpoint": str(loaded.path),
        "grid_iteration": int(loaded.grid_iteration),
        "time_slabs": len(times) - 1,
        "minimum_dt": float(np.diff(times).min()),
        "maximum_dt": float(np.diff(times).max()),
        "enriched_velocity_degree": velocity_degree,
        "enriched_pressure_degree": pressure_degree,
        "enriched_time_degree": time_degree,
        "enriched_boundary_time_degree": boundary_time_degree,
        "baseline_mean_drag": float(baseline["mean_drag"]),
        "baseline_enriched_mean_drag": float(baseline["enriched_mean_drag"]),
        "reference_mean_drag": float(config["reference_mean_drag"]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "low_primal_solved": False,
        "adjoint_solved": False,
        "localisation_executed": False,
        "marking_executed": False,
    }
    _write(args.output, result)

    try:
        transfers = build_slab_transfers(
            meshes,
            loaded.labels,
            mode=str(config["interface_transfer_mode"]),
            low_family=str(config["primal_space_family"]),
            enriched_velocity_degree=velocity_degree,
            enriched_pressure_degree=pressure_degree,
        )
        context = {
            "parameters": StaticCylinderParameters(
                hierarchy_levels=0,
                time_steps=len(times) - 1,
                final_time=float(times[-1]),
                time_degree=int(config["primal_time_degree"]),
                viscosity=float(config["viscosity"]),
                store_stages=True,
                report_every=int(args.report_every),
                slabwise_steppers=True,
            ),
            "meshes": meshes,
            "times": times,
            "labels": loaded.labels,
            "viscosity": float(config["viscosity"]),
        }
        rich = solve_slabwise_enriched_primal(
            context,
            transfers,
            time_degree=time_degree,
            # Default: saturation test against the continuous R5 problem,
            # letting the rich dG trajectory approximate the inlet in its
            # own temporal space.  With --boundary-time-degree 1 the rich
            # solve shares the low solve's dG(1) lifting instead, which is
            # the admissible pairing for the discrete DWR identity.
            boundary_time_degree=boundary_time_degree,
            report_every=int(args.report_every),
        )
        goal = float(
            mean_drag_from_history(
                rich, quadrature_points=int(args.quadrature_points)
            )
        )
        reference = float(config["reference_mean_drag"])
        baseline_goal = float(baseline["mean_drag"])
        baseline_error = reference - baseline_goal
        enriched_error = reference - goal
        enriched_goal_difference = goal - baseline_goal
        baseline_eta = float(baseline["eta_global"])
        total_cells = int(sum(mesh.num_cells() for mesh in meshes[1:]))
        spacetime_dofs = int(sum(
            transfers.rich_spaces[n][2].dim() * (time_degree + 1)
            for n in range(1, len(times))
        ))
        result.update(
            {
                "status": "complete",
                "stage": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": perf_counter() - started,
                "total_cells": total_cells,
                "enriched_spacetime_dofs": spacetime_dofs,
                "enriched_mean_drag": goal,
                "enriched_mean_drag_change": goal - baseline_goal,
                "baseline_true_error": baseline_error,
                "enriched_true_error": enriched_error,
                "saturation_ratio": abs(enriched_error)
                / max(abs(baseline_error), 1.0e-30),
                "baseline_eta_global": baseline_eta,
                "eta_over_enriched_goal_difference": baseline_eta
                / max(abs(enriched_goal_difference), 1.0e-30)
                * (1.0 if enriched_goal_difference >= 0.0 else -1.0),
                "drag_diagnostics": drag_history_diagnostics(
                    rich, quadrature_points=int(args.quadrature_points)
                ),
            }
        )
        _write(args.output, result)
        PETSc.Sys.Print(
            "[CYLINDER ENRICHED SATURATION AUDIT] "
            + json.dumps(result, sort_keys=True)
        )
        del rich, transfers
        gc.collect()
        return result
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "wall_seconds": perf_counter() - started,
            }
        )
        _write(args.output, result)
        raise


if __name__ == "__main__":
    main()
