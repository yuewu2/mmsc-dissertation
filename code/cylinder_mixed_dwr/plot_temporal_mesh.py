"""Draw the time-slab endpoints stored in a cylinder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="Checkpoint grid.json file.")
    parser.add_argument("output", type=Path, help="Output PNG or PDF.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    times = [
        float(value)
        for value in json.loads(args.grid.read_text(encoding="utf-8"))["times"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0, 1.35))
    ax.vlines(times, 0.34, 0.76, color="black", linewidth=0.72)
    ax.hlines(0.55, times[0], times[-1], color="black", linewidth=0.55)
    ax.text(times[0], 0.18, f"{times[0]:g}", ha="center", va="top")
    ax.text(times[-1], 0.18, f"{times[-1]:g}", ha="center", va="top")
    ax.set_xlim(times[0] - 0.12, times[-1] + 0.12)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.tight_layout(pad=0.10)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output.resolve())
    print(f"time_slabs={len(times) - 1}; min_dt={min(b-a for a, b in zip(times, times[1:])):.8g}; max_dt={max(b-a for a, b in zip(times, times[1:])):.8g}")


if __name__ == "__main__":
    main()
