"""Serial, bounded search for stable combined-indicator effectivities."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

from . import unattended_combined_followup as base


STATUS = base.OUT / "combined_search_status.log"
SUMMARY_JSON = base.OUT / "combined_search_summary.json"
SUMMARY_MD = base.OUT / "combined_search_summary.md"
# The host currently has about 5.9 GiB free.  Keep a hard reserve instead of
# deleting any of the user's historical checkpoints or ParaView output.
MIN_FREE_GIB = 4.5


def record(message: str) -> None:
    original = base.STATUS
    base.STATUS = STATUS
    try:
        base.record(message)
    finally:
        base.STATUS = original


def run(name: str, arguments: list[str]) -> int:
    original = base.STATUS
    base.STATUS = STATUS
    try:
        return base.run(name, arguments)
    finally:
        base.STATUS = original


def wait(prefix: Path) -> str:
    original = base.STATUS
    base.STATUS = STATUS
    try:
        return base.wait_for_manifest(prefix)
    finally:
        base.STATUS = original


def free_gib() -> float:
    return float(shutil.disk_usage(base.ROOT).free) / 1024.0**3


def finite(row: dict[str, str], key: str) -> float:
    return base.number(row, key)


def assess_stability(prefix: Path) -> dict[str, Any]:
    history = base.rows(prefix)
    result: dict[str, Any] = {
        "prefix": str(prefix),
        "status": base.manifest_status(prefix),
        "iterations": len(history),
        "stable": False,
    }
    if len(history) < 3:
        result["reason"] = "fewer than three completed iterations"
        return result
    tail = history[-3:]
    ieffs = [finite(row, "effectivity_global") for row in tail]
    errors = [abs(finite(row, "true_goal_error")) for row in tail]
    signed_errors = [finite(row, "true_goal_error") for row in tail]
    etas = [finite(row, "eta_global") for row in tail]
    gaps = [abs(finite(row, "localisation_gap_relative")) for row in tail]
    checks = {
        "finite_effectivities": all(math.isfinite(value) for value in ieffs),
        "effectivities_in_0.8_1.2": all(0.8 <= value <= 1.2 for value in ieffs),
        "effectivity_spread_at_most_0.2": max(ieffs) - min(ieffs) <= 0.2,
        "correct_sign": all(
            eta * error > 0.0 for eta, error in zip(etas, signed_errors)
        ),
        "localisation_closed": all(
            math.isfinite(gap) and gap <= 1.0e-6 for gap in gaps
        ),
        "tail_error_not_deteriorating": all(
            later <= 1.05 * earlier
            for earlier, later in zip(errors, errors[1:])
        ),
        "overall_error_reduced": errors[-1]
        < abs(finite(history[0], "true_goal_error")),
    }
    last = history[-1]
    result.update(
        {
            "last_iteration": int(last["iteration"]),
            "last_time_slabs": int(last["n_time_slabs"]),
            "last_primal_dofs": int(last["primal_spacetime_dofs"]),
            "last_adjoint_dofs": int(last["adjoint_spacetime_dofs"]),
            "last_goal_error": finite(last, "true_goal_error"),
            "last_eta": finite(last, "eta_global"),
            "last_effectivity": finite(last, "effectivity_global"),
            "tail_effectivities": ieffs,
            "tail_goal_errors": errors,
            "time_slab_sequence": [
                int(row["n_time_slabs"]) for row in history
            ],
            "checks": checks,
            "stable": all(checks.values()),
        }
    )
    result["reason"] = (
        "stable stopping criterion passed"
        if result["stable"]
        else "failed: " + ", ".join(k for k, value in checks.items() if not value)
    )
    return result


def common_resume(
    *, checkpoint: Path, theta: float, max_it: int, prefix: Path
) -> list[str]:
    return [
        "-m", "cylinder_mixed_dwr.production",
        "--resume-from", str(checkpoint),
        "--max-it", str(max_it),
        "--theta", str(theta),
        "--space-marking-strategy", "cellwise",
        "--space-mode", "independent",
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


def continued_75_arguments(*, checkpoint: Path, prefix: Path) -> list[str]:
    return common_resume(
        checkpoint=checkpoint, theta=0.40, max_it=4, prefix=prefix
    ) + [
        "--time-marking-strategy", "fixed_rate",
        "--time-score-source", "combined_indicator",
        "--time-fixed-rate", "0.75",
    ]


def temporal_dorfler_arguments(*, checkpoint: Path, prefix: Path) -> list[str]:
    return common_resume(
        checkpoint=checkpoint, theta=0.40, max_it=7, prefix=prefix
    ) + [
        "--time-marking-strategy", "slab_bulk_capped",
        "--time-score-source", "combined_indicator",
        "--time-bulk-fraction", "0.50",
        "--time-max-fraction", "0.30",
        "--time-max-count", "30",
    ]


def direct_nt160_arguments(prefix: Path) -> list[str]:
    return [
        "-m", "cylinder_mixed_dwr.production",
        "--levels", "1",
        "--geometry-degree", "2",
        "--nt", "160",
        "--T", "8",
        "--nu", "1e-3",
        "--primal-time-degree", "1",
        "--enriched-time-degree", "2",
        "--enriched-velocity-degree", "3",
        "--enriched-pressure-degree", "2",
        "--quadrature-points", "7",
        "--max-it", "3",
        "--theta", "0.40",
        "--space-marking-strategy", "cellwise",
        "--space-mode", "independent",
        "--time-marking-strategy", "cell_fraction_capped",
        "--time-score-source", "marked_fraction",
        "--time-fraction", "0.01",
        "--time-max-fraction", "0.30",
        "--time-max-count", "20",
        "--interface-transfer", "stokes_l2",
        "--dual-weight-mode", "enriched_minus_interpolant",
        "--dual-base-strategy", "interpolated_enriched",
        "--estimator-strategy", "primal_only",
        "--no-directional-split-diagnostic",
        "--reference-drag", "1.6031368",
        "--report-every", "16",
        "--output-prefix", str(prefix),
    ]


def ensure_run(name: str, prefix: Path, arguments: list[str]) -> str:
    status = base.manifest_status(prefix)
    if status == "complete":
        record(f"REUSE {name} status=complete")
        return status
    if status == "running":
        return wait(prefix)
    if status == "failed":
        record(f"SKIP {name} existing failed output")
        return status
    if free_gib() < MIN_FREE_GIB:
        record(
            f"STOP_LOW_DISK before={name} free_gib={free_gib():.2f} "
            f"required={MIN_FREE_GIB:.2f}"
        )
        return "low_disk"
    code = run(name, arguments)
    return "complete" if code == 0 else "failed"


def rank_key(result: dict[str, Any]) -> tuple[float, float, float]:
    values = result.get("tail_effectivities", [])
    if not values:
        return (float("inf"), float("inf"), float("inf"))
    distance = sum(abs(value - 1.0) for value in values) / len(values)
    spread = max(values) - min(values)
    error = abs(float(result.get("last_goal_error", float("inf"))))
    return distance, spread, error


def write_report(results: list[dict[str, Any]], status: str) -> None:
    complete = [item for item in results if item.get("iterations", 0) >= 1]
    ranked = sorted(complete, key=rank_key)
    data = {
        "status": status,
        "stopping_rule": (
            "last three I_eff in [0.8,1.2], spread <= 0.2, correct sign, "
            "closed localisation, and non-deteriorating goal error"
        ),
        "best_experiment": ranked[0].get("name") if ranked else None,
        "free_disk_gib": free_gib(),
        "experiments": results,
    }
    SUMMARY_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "# Combined-indicator stability search",
        "",
        f"Status: **{status}**",
        f"Best current experiment: **{data['best_experiment']}**",
        "",
        "| Experiment | Nt sequence | Last three Ieff | Goal error | Stable |",
        "|---|---|---|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item.get('name')} | {item.get('time_slab_sequence')} | "
            f"{item.get('tail_effectivities')} | "
            f"{item.get('last_goal_error')} | {item.get('stable')} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_result(
    results: list[dict[str, Any]], name: str, prefix: Path
) -> bool:
    assessment = assess_stability(prefix)
    assessment["name"] = name
    results.append(assessment)
    record(
        f"ASSESS {name} stable={assessment.get('stable')} "
        f"reason={assessment.get('reason')}"
    )
    write_report(results, "running")
    return bool(assessment.get("stable"))


def main() -> None:
    base.STATUS = STATUS
    record("SEARCH_START")
    results: list[dict[str, Any]] = []

    current_status = wait(base.CURRENT)
    current_bootstrap = base.assess_bootstrap(base.CURRENT)
    record(
        f"BOOTSTRAP theta30 status={current_status} "
        f"acceptable={current_bootstrap.get('acceptable')} "
        f"reason={current_bootstrap.get('reason')}"
    )

    if current_status == "complete" and current_bootstrap.get("acceptable"):
        prefix = base.OUT / "r5_theta30_time75_combined_trigger1_cap20_auto"
        status = ensure_run(
            "r5_theta30_time75_combined_trigger1_cap20_auto_run",
            prefix,
            base.trigger_arguments(
                checkpoint=Path(f"{base.CURRENT}_checkpoints/iter_0002"),
                theta=0.30,
                prefix=prefix,
            ),
        )
        if status == "complete" and save_result(results, "theta30_trigger", prefix):
            write_report(results, "stable_solution_found")
            record("SEARCH_SUCCESS theta30_trigger")
            return

    fallback_status = ensure_run(
        "r5_theta40_time75_combined_auto_run",
        base.FALLBACK,
        base.common_fresh(theta=0.40, prefix=base.FALLBACK),
    )
    if fallback_status != "complete":
        write_report(results, f"stopped_{fallback_status}")
        return

    theta40_checkpoint = Path(f"{base.FALLBACK}_checkpoints/iter_0002")
    theta40_trigger = base.OUT / "r5_theta40_time75_combined_trigger1_cap20_auto"
    status = ensure_run(
        "r5_theta40_time75_combined_trigger1_cap20_auto_run",
        theta40_trigger,
        base.trigger_arguments(
            checkpoint=theta40_checkpoint,
            theta=0.40,
            prefix=theta40_trigger,
        ),
    )
    if status == "complete" and save_result(
        results, "theta40_trigger", theta40_trigger
    ):
        write_report(results, "stable_solution_found")
        record("SEARCH_SUCCESS theta40_trigger")
        return

    continued = base.OUT / "r5_theta40_combined_time75_to216_auto"
    status = ensure_run(
        "r5_theta40_combined_time75_to216_auto_run",
        continued,
        continued_75_arguments(
            checkpoint=theta40_checkpoint, prefix=continued
        ),
    )
    if status == "complete" and save_result(
        results, "theta40_time75_to216", continued
    ):
        write_report(results, "stable_solution_found")
        record("SEARCH_SUCCESS theta40_time75_to216")
        return

    dorfler = base.OUT / "r5_theta40_combined_timebulk50_cap30_auto"
    status = ensure_run(
        "r5_theta40_combined_timebulk50_cap30_auto_run",
        dorfler,
        temporal_dorfler_arguments(
            checkpoint=theta40_checkpoint, prefix=dorfler
        ),
    )
    if status == "complete" and save_result(
        results, "theta40_timebulk50", dorfler
    ):
        write_report(results, "stable_solution_found")
        record("SEARCH_SUCCESS theta40_timebulk50")
        return

    direct = base.OUT / "r5_direct_nt160_theta40_trigger1_cap20_auto"
    status = ensure_run(
        "r5_direct_nt160_theta40_trigger1_cap20_auto_run",
        direct,
        direct_nt160_arguments(direct),
    )
    if status == "complete":
        save_result(results, "direct_nt160_theta40", direct)

    final_status = (
        "stable_solution_found"
        if any(item.get("stable") for item in results)
        else "queue_exhausted_without_strict_stability"
    )
    write_report(results, final_status)
    record(f"SEARCH_DONE status={final_status}")


if __name__ == "__main__":
    main()
