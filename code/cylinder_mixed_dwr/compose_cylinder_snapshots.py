"""Compose three ParaView cylinder snapshots into a thesis-ready figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = json.loads(
        (args.snapshot_dir / "velocity_snapshots.json").read_text(encoding="utf-8")
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.8))
    labels = ["(a)", "(b)", "(c)"]
    for ax, label, row in zip(axes, labels, rows):
        image = plt.imread(row["file"])
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.5,
            -0.045,
            rf"{label} $t={float(row['actual_time']):.5f}\,\mathrm{{s}}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.99, bottom=0.18, hspace=0.25)
    cax = fig.add_axes([0.31, 0.045, 0.38, 0.022])
    scalar = plt.cm.ScalarMappable(norm=Normalize(0.0, 2.0), cmap="viridis")
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
    colorbar.ax.set_title("velocity magnitude", fontsize=10, pad=8)
    colorbar.outline.set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
