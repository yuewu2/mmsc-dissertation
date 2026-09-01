"""Serial follow-up for combined-indicator 75% bootstrap experiments.

The runner waits for the user's theta=0.30 job.  If its iteration-2 result
has a decreasing goal error, correct estimator sign, negligible localisation
gap, and 0.75 <= I_eff <= 1.25, it resumes that grid with the thesis 1% cell
fraction trigger.  Otherwise it first repeats the two 75% bootstrap steps at
theta=0.40 and resumes that grid.  Trigger runs use no directional split and
stop after iteration 6, which reaches about 203 slabs when the cap of 20 is
active on every loop.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path("/Users/wuyue/Desktop/dissertation")
OUT = ROOT / "output/cylinder_mixed_dwr"
PYTHON = Path("/Users/wuyue/src/firedrake-install/venv-firedrake/bin/python")
STATUS = OUT / "combined_followup_status.log"
SUMMARY_JSON = OUT / "combined_followup_summary.json"
SUMMARY_MD = OUT / "combined_followup_summary.md"

CURRENT = OUT / "r5_theta30_time75_combined"
FALLBACK = OUT / "r5_theta40_time75_combined_auto"


def record(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp} {message}\n")


def manifest_path(prefix: Path) -> Path:
    return Path(f"{prefix}_manifest.json")


def history_path(prefix: Path) -> Path:
    return Path(f"{prefix}_history.csv")


def manifest_status(prefix: Path) -> str | None:
    path = manifest_path(prefix)
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["status"])
    except (KeyError, json.JSONDecodeError):
        return "running"


def wait_for_manifest(prefix: Path, *, timeout_hours: float = 12.0) -> str:
    started = time.monotonic()
    last_status = None
    while True:
        status = manifest_status(prefix)
        if status != last_status:
            record(f"WAIT prefix={prefix.name} status={status}")
            last_status = status
        if status in {"complete", "failed"}:
            return status
        if time.monotonic() - started > timeout_hours * 3600.0:
            record(f"TIMEOUT prefix={prefix.name}")
            return "timeout"
        time.sleep(20.0)


def run(name: str, arguments: list[str]) -> int:
    log = OUT / f"{name}.log"
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    record(f"START {name}")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(PYTHON), *arguments],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record(f"END {name} exit={result.returncode}")
    return int(result.returncode)


def rows(prefix: Path) -> list[dict[str, str]]:
    path = history_path(prefix)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def assess_bootstrap(prefix: Path) -> dict[str, Any]:
    history = rows(prefix)
    result: dict[str, Any] = {
        "prefix": str(prefix),
        "iterations": len(history),
        "acceptable": False,
    }
    if len(history) < 3:
        result["reason"] = "fewer than three completed iterations"
        return result
    first = history[0]
    last = history[-1]
    ieff = number(last, "effectivity_global")
    first_error = abs(number(first, "true_goal_error"))
    last_error = abs(number(last, "true_goal_error"))
    eta = number(last, "eta_global")
    signed_error = number(last, "true_goal_error")
    gap = abs(number(last, "localisation_gap_relative"))
    checks = {
        "iteration_at_least_2": int(last["iteration"]) >= 2,
        "time_grid_at_least_120": int(last["n_time_slabs"]) >= 120,
        "goal_error_decreased": last_error < first_error,
        "correct_estimator_sign": eta * signed_error > 0.0,
        "effectivity_close_to_one": math.isfinite(ieff)
        and 0.75 <= ieff <= 1.25,
        "localisation_closed": math.isfinite(gap) and gap <= 1.0e-6,
    }
    result.update(
        {
            "last_iteration": int(last["iteration"]),
            "last_time_slabs": int(last["n_time_slabs"]),
            "first_goal_error": first_error,
            "last_goal_error": last_error,
            "last_eta": eta,
            "last_effectivity": ieff,
            "last_localisation_gap_relative": gap,
            "checks": checks,
            "acceptable": all(checks.values()),
        }
    )
    result["reason"] = (
        "all acceptance checks passed"
        if result["acceptable"]
        else "failed: " + ", ".join(k for k, value in checks.items() if not value)
    )
    return result


def common_fresh(*, theta: float, prefix: Path) -> list[str]:
    return [
        "-m", "cylinder_mixed_dwr.production",
        "--levels", "1",
        "--geometry-degree", "2",
        "--nt", "40",
        "--T", "8",
        "--nu", "1e-3",
        "--primal-time-degree", "1",
        "--enriched-time-degree", "2",
        "--enriched-velocity-degree", "3",
        "--enriched-pressure-degree", "2",
        "--quadrature-points", "7",
        "--max-it", "3",
        "--theta", str(theta),
        "--space-marking-strategy", "cellwise",
        "--space-mode", "independent",
        "--time-marking-strategy", "fixed_rate",
        "--time-score-source", "combined_indicator",
        "--time-fixed-rate", "0.75",
        "--interface-transfer", "stokes_l2",
        "--dual-weight-mode", "enriched_minus_interpolant",
        "--dual-base-strategy", "interpolated_enriched",
        "--estimator-strategy", "primal_only",
        "--no-directional-split-diagnostic",
        "--reference-drag", "1.6031368",
        "--report-every", "16",
        "--output-prefix", str(prefix),
    ]


def trigger_arguments(
    *, checkpoint: Path, theta: float, prefix: Path
) -> list[str]:
    return [
        "-m", "cylinder_mixed_dwr.production",
        "--resume-from", str(checkpoint),
        "--max-it", "7",
        "--theta", str(theta),
        "--space-marking-strategy", "cellwise",
        "--space-mode", "independent",
        "--time-marking-strategy", "cell_fraction_capped",
        "--time-score-source", "marked_fraction",
        "--time-fraction", "0.01",
        "--time-max-fraction", "0.30",
        "--time-max-count", "20",
        "--quadrature-points", "7",
        "--interface-transfer", "stokes_l2",
        "--dual-weight-mode", "enriched_minus_interpolant",
        "--dual-base-strategy", "interpolated_enriched",
        "--estimator-strategy", "primal_only",
        "--no-directional-split-diagnostic",
        "--reference-drag", "1.6031368",
        "--report-every", "16",
        "--output-prefix", str(prefix),
    ]


def summarize_trigger(prefix: Path) -> dict[str, Any]:
    history = rows(prefix)
    if not history:
        return {"prefix": str(prefix), "status": "missing"}
    last = history[-1]
    return {
        "prefix": str(prefix),
        "status": manifest_status(prefix),
        "iterations": len(history),
        "last_iteration": int(last["iteration"]),
        "last_time_slabs": int(last["n_time_slabs"]),
        "last_primal_dofs": int(last["primal_spacetime_dofs"]),
        "last_adjoint_dofs": int(last["adjoint_spacetime_dofs"]),
        "last_goal_error": number(last, "true_goal_error"),
        "last_eta": number(last, "eta_global"),
        "last_effectivity": number(last, "effectivity_global"),
        "effectivity_sequence": [
            number(row, "effectivity_global") for row in history
        ],
        "time_slab_sequence": [int(row["n_time_slabs"]) for row in history],
    }


def write_summary(data: dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    bootstrap = data["bootstrap_assessment"]
    trigger = data.get("trigger_result", {})
    lines = [
        "# Combined-indicator automatic follow-up",
        "",
        f"Selected branch: **{data.get('selected_branch')}**",
        f"Bootstrap decision: {bootstrap.get('reason')}",
        "",
        "## Trigger result",
        "",
        f"- Status: {trigger.get('status')}",
        f"- Time slabs: {trigger.get('time_slab_sequence')}",
        f"- Effectivity: {trigger.get('effectivity_sequence')}",
        f"- Final goal error: {trigger.get('last_goal_error')}",
        f"- Final estimator: {trigger.get('last_eta')}",
        f"- Final effectivity: {trigger.get('last_effectivity')}",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    record("WATCHER_START")
    current_status = wait_for_manifest(CURRENT)
    current_assessment = assess_bootstrap(CURRENT)
    record(
        f"ASSESS theta30 status={current_status} "
        f"acceptable={current_assessment.get('acceptable')} "
        f"reason={current_assessment.get('reason')}"
    )

    selected = CURRENT
    selected_theta = 0.30
    selected_branch = "theta30"
    selected_assessment = current_assessment

    if current_status != "complete" or not current_assessment["acceptable"]:
        fallback_status = manifest_status(FALLBACK)
        if fallback_status is None:
            code = run(
                "r5_theta40_time75_combined_auto_run",
                common_fresh(theta=0.40, prefix=FALLBACK),
            )
            fallback_status = "complete" if code == 0 else "failed"
        elif fallback_status == "running":
            fallback_status = wait_for_manifest(FALLBACK)
        fallback_assessment = assess_bootstrap(FALLBACK)
        record(
            f"ASSESS theta40 status={fallback_status} "
            f"acceptable={fallback_assessment.get('acceptable')} "
            f"reason={fallback_assessment.get('reason')}"
        )
        if fallback_status != "complete":
            data = {
                "status": "failed_before_trigger",
                "selected_branch": "theta40",
                "bootstrap_assessment": fallback_assessment,
                "theta30_assessment": current_assessment,
            }
            write_summary(data)
            record("WATCHER_STOP fallback failed")
            return
        selected = FALLBACK
        selected_theta = 0.40
        selected_branch = "theta40"
        selected_assessment = fallback_assessment

    checkpoint = Path(f"{selected}_checkpoints/iter_0002")
    trigger = OUT / (
        f"r5_{selected_branch}_time75_combined_trigger1_cap20_auto"
    )
    trigger_status = manifest_status(trigger)
    if trigger_status is None:
        code = run(
            f"{trigger.name}_run",
            trigger_arguments(
                checkpoint=checkpoint,
                theta=selected_theta,
                prefix=trigger,
            ),
        )
        trigger_status = "complete" if code == 0 else "failed"
    elif trigger_status == "running":
        trigger_status = wait_for_manifest(trigger)

    data = {
        "status": trigger_status,
        "selected_branch": selected_branch,
        "selected_theta": selected_theta,
        "bootstrap_assessment": selected_assessment,
        "theta30_assessment": current_assessment,
        "trigger_configuration": {
            "time_fraction": 0.01,
            "time_max_fraction": 0.30,
            "time_max_count": 20,
            "max_iteration": 6,
            "directional_split": False,
            "write_vtk": False,
        },
        "trigger_result": summarize_trigger(trigger),
    }
    write_summary(data)
    record(f"ALL_DONE status={trigger_status}")


if __name__ == "__main__":
    main()
