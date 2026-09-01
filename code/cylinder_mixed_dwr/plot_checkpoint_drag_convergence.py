"""Compare checkpoint drag trajectories with stored FeatFlow reference data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .checkpoint import CylinderCheckpointStore
from .featflow_drag_timeseries import _sample_drag
from .slabwise import build_slab_transfers, solve_slabwise_primal


def reference_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in ("time", "reference_level5", "reference_level6")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("LABEL", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-every", type=int, default=40)
    args = parser.parse_args()

    reference = reference_columns(args.reference_csv)
    sample_times = reference["time"]
    curves: dict[str, np.ndarray] = {}
    summaries = []

    for label, checkpoint_text in args.case:
        checkpoint = Path(checkpoint_text)
        loaded = CylinderCheckpointStore.load(checkpoint)
        config = loaded.saved_config
        transfers = build_slab_transfers(
            loaded.meshes,
            loaded.labels,
            mode=str(config["interface_transfer_mode"]),
            low_family=str(config["primal_space_family"]),
            enriched_velocity_degree=int(config["enriched_velocity_degree"]),
        )
        started = perf_counter()
        primal = solve_slabwise_primal(
            loaded.meshes,
            loaded.times,
            transfers,
            loaded.labels,
            viscosity=float(config["viscosity"]),
            time_degree=int(config["primal_time_degree"]),
            report_every=int(args.report_every),
        )
        drag = _sample_drag(primal, sample_times)["total"]
        curves[label] = drag
        pointwise_error = reference["reference_level6"] - drag
        summaries.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "time_slabs": int(len(loaded.times) - 1),
                "mean_drag_on_reference_samples": float(np.mean(drag)),
                "mean_goal_error_vs_level6": float(np.mean(pointwise_error)),
                "rmse_vs_level6": float(np.sqrt(np.mean(pointwise_error**2))),
                "maximum_absolute_error": float(np.max(np.abs(pointwise_error))),
                "maximum_absolute_error_time": float(
                    sample_times[np.argmax(np.abs(pointwise_error))]
                ),
                "solve_seconds": float(perf_counter() - started),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "drag_trajectory_comparison.csv"
    names = ["time", "reference_level5", "reference_level6", *curves]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(
            zip(
                reference["time"],
                reference["reference_level5"],
                reference["reference_level6"],
                *(curves[name] for name in curves),
            )
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#d95f02", "#1b9e77", "#275dad", "#7b3294"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    axes[0].plot(
        sample_times,
        reference["reference_level6"],
        color="black",
        linewidth=1.5,
        label="FeatFlow level 6",
        zorder=5,
    )
    axes[0].plot(
        sample_times,
        reference["reference_level5"],
        color="0.55",
        linewidth=0.9,
        linestyle="--",
        label="FeatFlow level 5",
    )
    for colour, (label, drag) in zip(colours, curves.items()):
        axes[0].plot(sample_times, drag, color=colour, linewidth=1.0, label=label)
        axes[1].plot(
            sample_times,
            reference["reference_level6"] - drag,
            color=colour,
            linewidth=1.0,
            label=label,
        )

    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$C_D(t)$")
    axes[0].set_xlim(0.0, 8.0)
    axes[0].grid(color="0.75", linestyle="--", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].set_title("(a) Drag coefficient")

    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$C_D^{\mathrm{ref}}(t)-C_D^h(t)$")
    axes[1].set_xlim(0.0, 8.0)
    axes[1].grid(color="0.75", linestyle="--", linewidth=0.7, alpha=0.7)
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].set_title("(b) Pointwise drag error")

    figure_path = args.output_dir / "drag_trajectory_and_error.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = args.output_dir / "drag_trajectory_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "reference_level5_mean": float(
                    np.mean(reference["reference_level5"])
                ),
                "reference_level6_mean": float(
                    np.mean(reference["reference_level6"])
                ),
                "cases": summaries,
                "files": {"csv": str(csv_path), "figure": str(figure_path)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(figure_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
