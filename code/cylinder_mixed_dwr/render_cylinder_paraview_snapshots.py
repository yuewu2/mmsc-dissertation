"""Render consistent velocity-magnitude snapshots from a cylinder PVD file.

Run with ParaView's ``pvpython`` rather than the Firedrake interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paraview.simple import (  # type: ignore[import-not-found]
    ColorBy,
    CreateView,
    GetColorTransferFunction,
    OpenDataFile,
    Render,
    SaveScreenshot,
    Show,
    UpdatePipeline,
    _DisableFirstRenderCameraReset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[3.0, 5.0, 6.0])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _DisableFirstRenderCameraReset()
    source = OpenDataFile(str(args.pvd.resolve()))
    source.UpdatePipelineInformation()
    available = [float(value) for value in source.TimestepValues]
    selected = [min(available, key=lambda value: abs(value - target)) for target in args.times]

    view = CreateView("RenderView")
    view.ViewSize = [1800, 410]
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.CameraParallelProjection = 1
    view.CameraPosition = [1.1, 0.205, 5.0]
    view.CameraFocalPoint = [1.1, 0.205, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = 0.255

    display = Show(source, view)
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.92, 0.92, 0.92]
    display.LineWidth = 0.45
    ColorBy(display, ("POINTS", "Speed"))
    lut = GetColorTransferFunction("Speed")
    lut.ApplyPreset("Viridis (matplotlib)", True)
    lut.RescaleTransferFunction(0.0, 2.0)
    display.SetScalarBarVisibility(view, False)

    rows = []
    for index, (requested, actual) in enumerate(zip(args.times, selected), start=1):
        UpdatePipeline(time=actual, proxy=source)
        view.ViewTime = actual
        Render(view)
        target = args.output_dir / f"velocity_snapshot_{index}.png"
        SaveScreenshot(
            str(target),
            view,
            ImageResolution=[1800, 410],
            TransparentBackground=0,
            CompressionLevel="3",
        )
        rows.append({"requested_time": requested, "actual_time": actual, "file": str(target)})

    metadata = args.output_dir / "velocity_snapshots.json"
    metadata.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(metadata.resolve())


if __name__ == "__main__":
    main()
