"""Compare uniform low/rich drag trajectories with FeatFlow level 5/6 data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from firedrake import FacetNormal, as_vector, assemble, dot, ds, grad, split
from firedrake.petsc import PETSc

from navier_stokes_cylinder_irksome_static_primal import (
    _lagrange_values,
)

from .adaptive import CylinderAdaptiveConfig, CylinderAdaptiveSolver
from .benchmark import drag_coefficient_form
from .checkpoint import CylinderCheckpointStore
from .slabwise import (
    build_slab_transfers,
    solve_slabwise_enriched_primal,
    solve_slabwise_primal,
)


def _load_featflow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Unexpected FeatFlow drag file format: {path}")
    times = np.asarray(data[:, 1], dtype=float)
    drag = np.asarray(data[:, 3], dtype=float)
    keep = times > 0.0  # The published mean excludes the artificial t=0 row.
    times = times[keep]
    drag = drag[keep]
    if len(times) == 0 or not np.all(np.diff(times) > 0.0):
        raise ValueError(f"FeatFlow sample times are not strictly increasing: {path}")
    return times, drag


def _drag_coefficients(primal: dict, *, drag_scale: float = 20.0) -> dict[str, list[np.ndarray | None]]:
    """Assemble drag components per dG coefficient; reconstruction is exact."""
    values = {
        name: [None] * len(primal["slabs"])
        for name in ("total", "pressure", "viscous")
    }
    for slab_number in range(1, len(primal["slabs"])):
        slab = primal["slabs"][slab_number]
        by_component = {name: [] for name in values}
        for coefficient in slab["coeffs"]:
            velocity, pressure = split(coefficient)
            normal = FacetNormal(coefficient.function_space().mesh())
            pressure_form = (
                float(drag_scale)
                * pressure
                * normal[0]
                * ds(primal["labels"]["cylinder"])
            )
            viscous_form = (
                -float(drag_scale)
                * primal["viscosity"]
                * dot(dot(grad(velocity), normal), as_vector((1.0, 0.0)))
                * ds(primal["labels"]["cylinder"])
            )
            pressure_value = float(assemble(pressure_form))
            viscous_value = float(assemble(viscous_form))
            total_value = float(
                assemble(
                    drag_coefficient_form(
                        coefficient,
                        primal["viscosity"],
                        primal["labels"]["cylinder"],
                        drag_scale=drag_scale,
                    )
                )
            )
            by_component["pressure"].append(pressure_value)
            by_component["viscous"].append(viscous_value)
            by_component["total"].append(total_value)
        for name in values:
            values[name][slab_number] = np.asarray(by_component[name], dtype=float)
    return values


def _sample_drag(primal: dict, sample_times: np.ndarray) -> dict[str, np.ndarray]:
    times = np.asarray(primal["times"], dtype=float)
    if sample_times[0] < times[0] or sample_times[-1] > times[-1]:
        raise ValueError("Requested drag samples lie outside the primal time interval.")
    coefficients = _drag_coefficients(primal)
    sampled = {
        name: np.empty_like(sample_times, dtype=float) for name in coefficients
    }
    for index, physical_time in enumerate(sample_times):
        slab_number = int(np.searchsorted(times, physical_time, side="right"))
        slab_number = min(max(slab_number, 1), len(times) - 1)
        step = float(times[slab_number] - times[slab_number - 1])
        point = (float(physical_time) - float(times[slab_number - 1])) / step
        weights = _lagrange_values(
            int(primal["slabs"][slab_number]["degree"]), point
        )
        for name in sampled:
            sampled[name][index] = float(
                np.dot(coefficients[name][slab_number], weights)
            )
    return sampled


def _sample_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _cumulative_global_mean(error: np.ndarray, sample_step: float, horizon: float) -> np.ndarray:
    return np.cumsum(error) * float(sample_step) / float(horizon)


def _window_rows(
    times: np.ndarray,
    reference: np.ndarray,
    low: np.ndarray,
    rich: np.ndarray,
    *,
    horizon: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for start in np.arange(0.0, horizon, 2.0):
        end = min(float(start + 2.0), float(horizon))
        mask = (times >= start) & (times < end)
        width = end - float(start)
        low_local = float(np.mean(reference[mask] - low[mask]))
        rich_local = float(np.mean(reference[mask] - rich[mask]))
        rows.append(
            {
                "start": float(start),
                "end": end,
                "samples": int(np.count_nonzero(mask)),
                "low_local_mean_error": low_local,
                "rich_local_mean_error": rich_local,
                "low_global_mean_contribution": low_local * width / horizon,
                "rich_global_mean_contribution": rich_local * width / horizon,
            }
        )
    return rows


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))


def _plot(path: Path, columns: dict[str, np.ndarray], windows: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = columns["time"]
    figure, axes = plt.subplots(4, 1, figsize=(11, 14), constrained_layout=True)
    axes[0].plot(time, columns["reference_level6"], color="black", lw=1.6, label="FeatFlow L6")
    axes[0].plot(time, columns["reference_level5"], color="0.55", lw=1.0, label="FeatFlow L5")
    axes[0].plot(time, columns["low_drag"], color="#1f77b4", label="P2/P1-dG(1)")
    axes[0].plot(time, columns["rich_drag"], color="#d62728", label="P3/P2-dG(1)")
    axes[0].set_ylabel(r"$C_D(t)$")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, columns["low_error_vs_level6"], color="#1f77b4", label="L6 - low")
    axes[1].plot(time, columns["rich_error_vs_level6"], color="#d62728", label="L6 - rich")
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel("pointwise error")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(time, columns["low_cumulative_mean_error"], color="#1f77b4", label="low")
    axes[2].plot(time, columns["rich_cumulative_mean_error"], color="#d62728", label="rich")
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_ylabel("cumulative contribution")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    centres = np.asarray([(row["start"] + row["end"]) / 2.0 for row in windows])
    width = 0.7
    axes[3].bar(
        centres - width / 4.0,
        [row["low_global_mean_contribution"] for row in windows],
        width / 2.0,
        color="#1f77b4",
        label="low",
    )
    axes[3].bar(
        centres + width / 4.0,
        [row["rich_global_mean_contribution"] for row in windows],
        width / 2.0,
        color="#d62728",
        label="rich",
    )
    axes[3].axhline(0.0, color="black", lw=0.7)
    axes[3].set_xticks(centres, [f"[{row['start']:.0f},{row['end']:.0f})" for row in windows])
    axes[3].set_xlabel("time window")
    axes[3].set_ylabel("contribution to full mean error")
    axes[3].legend()
    axes[3].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_components(path: Path, columns: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = columns["time"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(time, columns["low_pressure_drag"], color="#1f77b4", label="low pressure")
    axes[0].plot(time, columns["rich_pressure_drag"], color="#d62728", label="rich pressure")
    axes[0].set_ylabel("pressure drag")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, columns["low_viscous_drag"], color="#1f77b4", label="low viscous")
    axes[1].plot(time, columns["rich_viscous_drag"], color="#d62728", label="rich viscous")
    axes[1].set_ylabel("viscous drag")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        time,
        columns["rich_pressure_drag"] - columns["low_pressure_drag"],
        color="#9467bd",
        label="rich - low pressure",
    )
    axes[2].plot(
        time,
        columns["rich_viscous_drag"] - columns["low_viscous_drag"],
        color="#2ca02c",
        label="rich - low viscous",
    )
    axes[2].plot(
        time,
        columns["enrichment_correction"],
        color="black",
        lw=1.0,
        label="rich - low total",
    )
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("enrichment change")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    reference_time, reference_level6 = _load_featflow(args.featflow_level6)
    level5_time, reference_level5 = _load_featflow(args.featflow_level5)
    if not np.allclose(reference_time, level5_time, rtol=0.0, atol=1.0e-13):
        raise ValueError("FeatFlow level 5 and 6 sampling times differ.")

    if args.resume_from is None:
        config = CylinderAdaptiveConfig(
            hierarchy_levels=int(args.levels),
            geometry_degree=int(args.geometry_degree),
            initial_time_slabs=int(args.nt),
            final_time=float(args.T),
            viscosity=float(args.nu),
            primal_time_degree=1,
            enriched_time_degree=int(args.enriched_time_degree),
            max_iterations=1,
            enable_space_refinement=False,
            enable_time_refinement=False,
            enriched_velocity_degree=3,
            enriched_pressure_degree=2,
            report_every=10,
        )
        solver = CylinderAdaptiveSolver(config)
        meshes, times, labels = solver.meshes, solver.times, solver.labels
    else:
        loaded = CylinderCheckpointStore.load(args.resume_from)
        config = CylinderAdaptiveConfig(**loaded.saved_config)
        meshes, times, labels = loaded.meshes, loaded.times, loaded.labels
    transfers = build_slab_transfers(
        meshes,
        labels,
        mode=config.interface_transfer_mode,
        enriched_velocity_degree=3,
        enriched_pressure_degree=2,
    )
    started = perf_counter()
    low = solve_slabwise_primal(
        meshes,
        times,
        transfers,
        labels,
        viscosity=config.viscosity,
        time_degree=1,
        report_every=config.report_every,
    )
    rich = solve_slabwise_enriched_primal(
        low,
        transfers,
        time_degree=int(config.enriched_time_degree),
        report_every=config.report_every,
    )
    solve_seconds = perf_counter() - started

    low_components = _sample_drag(low, reference_time)
    rich_components = _sample_drag(rich, reference_time)
    low_drag = low_components["total"]
    rich_drag = rich_components["total"]
    low_error = reference_level6 - low_drag
    rich_error = reference_level6 - rich_drag
    sample_step = float(np.median(np.diff(reference_time)))
    columns = {
        "time": reference_time,
        "reference_level5": reference_level5,
        "reference_level6": reference_level6,
        "low_drag": low_drag,
        "rich_drag": rich_drag,
        "low_pressure_drag": low_components["pressure"],
        "low_viscous_drag": low_components["viscous"],
        "rich_pressure_drag": rich_components["pressure"],
        "rich_viscous_drag": rich_components["viscous"],
        "low_error_vs_level6": low_error,
        "rich_error_vs_level6": rich_error,
        "enrichment_correction": rich_drag - low_drag,
        "low_cumulative_mean_error": _cumulative_global_mean(low_error, sample_step, args.T),
        "rich_cumulative_mean_error": _cumulative_global_mean(rich_error, sample_step, args.T),
    }
    windows = _window_rows(
        reference_time,
        reference_level6,
        low_drag,
        rich_drag,
        horizon=float(args.T),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "drag_timeseries.csv"
    plot_path = args.output_dir / "drag_timeseries.png"
    component_plot_path = args.output_dir / "drag_components.png"
    _write_csv(csv_path, columns)
    _plot(plot_path, columns, windows)
    _plot_components(component_plot_path, columns)

    summary = {
        "configuration": {
            "levels": int(args.levels),
            "geometry_degree": int(args.geometry_degree),
            "time_slabs": int(len(times) - 1),
            "final_time": float(args.T),
            "low_space": "P2/P1-dG(1)",
            "rich_space": f"P3/P2-dG({int(config.enriched_time_degree)})",
        },
        "sample_count": int(len(reference_time)),
        "sample_step": sample_step,
        "solve_seconds": solve_seconds,
        "means_same_sample_convention": {
            "featflow_level5": _sample_mean(reference_level5),
            "featflow_level6": _sample_mean(reference_level6),
            "low": _sample_mean(low_drag),
            "rich": _sample_mean(rich_drag),
        },
        "component_means": {
            "low_pressure": _sample_mean(low_components["pressure"]),
            "low_viscous": _sample_mean(low_components["viscous"]),
            "rich_pressure": _sample_mean(rich_components["pressure"]),
            "rich_viscous": _sample_mean(rich_components["viscous"]),
            "enrichment_pressure_change": _sample_mean(
                rich_components["pressure"] - low_components["pressure"]
            ),
            "enrichment_viscous_change": _sample_mean(
                rich_components["viscous"] - low_components["viscous"]
            ),
        },
        "errors_vs_level6": {
            "low_mean_error": _sample_mean(low_error),
            "rich_mean_error": _sample_mean(rich_error),
            "low_rmse": float(np.sqrt(np.mean(low_error**2))),
            "rich_rmse": float(np.sqrt(np.mean(rich_error**2))),
            "low_max_abs": float(np.max(np.abs(low_error))),
            "rich_max_abs": float(np.max(np.abs(rich_error))),
            "low_max_abs_time": float(reference_time[np.argmax(np.abs(low_error))]),
            "rich_max_abs_time": float(reference_time[np.argmax(np.abs(rich_error))]),
        },
        "windows": windows,
        "files": {
            "csv": str(csv_path),
            "plot": str(plot_path),
            "component_plot": str(component_plot_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    PETSc.Sys.Print("[FEATFLOW DRAG COMPARISON] " + json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--geometry-degree", type=int, default=2)
    parser.add_argument("--nt", type=int, default=20)
    parser.add_argument("--T", type=float, default=8.0)
    parser.add_argument("--nu", type=float, default=1.0e-3)
    parser.add_argument("--enriched-time-degree", type=int, default=1)
    parser.add_argument("--featflow-level5", type=Path, required=True)
    parser.add_argument("--featflow-level6", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Use an adaptive grid checkpoint instead of a fresh uniform grid.",
    )
    args = parser.parse_args()
    for action in parser._actions:
        for flag in action.option_strings:
            if flag.startswith("--"):
                PETSc.Options().delValue(flag)
    run(args)


if __name__ == "__main__":
    main()
