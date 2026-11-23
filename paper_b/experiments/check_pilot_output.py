"""Scientific validation gate for the Paper B matched-seed pilot."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


FROZEN_ZERO_KEYS = (
    "lambda_first_audit_to_terminal_tv",
    "lambda_mean_period_turnover",
    "lambda_cumulative_turnover",
    "lambda_top_source_changed_share",
    "lambda_top_source_switches_per_citizen",
)


def _read_shards(run_root: Path) -> list[dict]:
    payloads = []
    for path in sorted((run_root / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    return payloads


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def validate_pilot(
    run_root: str | Path,
    *,
    tolerance: float = 1e-12,
    write_report: bool = False,
) -> dict:
    run_root = Path(run_root)
    manifest = json.loads(
        (run_root / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    payloads = _read_shards(run_root)

    expected_blocks = int(manifest["expected_blocks"])
    expected_runs = int(manifest["expected_runs"])
    expected_ids = set(manifest["block_ids"])
    observed_ids = [p.get("block_id") for p in payloads]
    observed_set = set(observed_ids)

    unique_blocks = len(observed_ids) == len(observed_set)
    complete_block_set = observed_set == expected_ids

    all_runs = []
    quartet_ok = True
    fingerprint_ok = True
    finite_ok = True
    horizon_ok = True
    frozen_ok = True
    hold_ok = True
    refresh_ok = True
    block_errors = []

    expected_conditions = set(manifest.get("matched_conditions") or (
        "adaptive__J1", "adaptive__J0", "frozen__J1", "frozen__J0"
    ))
    sender_grid = manifest.get("analysis_mode") == "sender_grid"
    config_ok = True

    for payload in payloads:
        block_id = payload.get("block_id")
        runs = payload.get("runs", [])
        all_runs.extend(runs)

        conditions = {row.get("condition") for row in runs}
        if (
            len(runs) != len(expected_conditions)
            or conditions != expected_conditions
            or len(conditions) != len(runs)
        ):
            quartet_ok = False
            block_errors.append(
                f"{block_id}: incomplete or duplicated matched conditions"
            )

        structural = {row.get("structural_fingerprint") for row in runs}
        initial = {row.get("initial_state_fingerprint") for row in runs}
        if len(structural) != 1 or len(initial) != 1:
            fingerprint_ok = False
            block_errors.append(f"{block_id}: fingerprint mismatch")

        for row in runs:
            if sender_grid:
                expected = {
                    "peer_evidence_mode": manifest["peer_evidence_mode"],
                    "frozen_ranking_mode": manifest["frozen_ranking_mode"],
                    "initial_audit_mode": manifest["initial_audit_mode"],
                }
                for key, value in expected.items():
                    if str(row.get(key)) != str(value):
                        config_ok = False
                        block_errors.append(
                            f"{block_id}/{row.get('condition')}: {key} mismatch"
                        )
                tau = float(row.get("tau_social", math.nan))
                if not math.isclose(
                    tau, float(manifest["tau_social"]),
                    rel_tol=0.0, abs_tol=tolerance
                ):
                    config_ok = False
                    block_errors.append(
                        f"{block_id}/{row.get('condition')}: tau_social mismatch"
                    )
                if "jammer_credibility_penalty" in manifest:
                    penalty = float(
                        row.get("jammer_credibility_penalty", math.nan)
                    )
                    if not math.isclose(
                        penalty,
                        float(manifest["jammer_credibility_penalty"]),
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        config_ok = False
                        block_errors.append(
                            f"{block_id}/{row.get('condition')}: "
                            "jammer_credibility_penalty mismatch"
                        )
                expected_name = (
                    f"{row['reliance_mode']}__{row['jammer_regime']}"
                )
                if row.get("condition") != expected_name:
                    config_ok = False
                    block_errors.append(
                        f"{block_id}: mislabeled sender condition"
                    )
            if int(row.get("n_nonfinite", 1)) != 0:
                finite_ok = False
                block_errors.append(
                    f"{block_id}/{row.get('condition')}: non-finite state"
                )
            if int(row.get("steps_run", -1)) != int(
                row.get("terminal_horizon_T", -2)
            ):
                horizon_ok = False
                block_errors.append(
                    f"{block_id}/{row.get('condition')}: horizon mismatch"
                )

            if row.get("reliance_mode") == "frozen":
                for key in FROZEN_ZERO_KEYS:
                    value = float(row.get(key, math.nan))
                    if not math.isfinite(value) or abs(value) > tolerance:
                        frozen_ok = False
                        block_errors.append(
                            f"{block_id}/{row.get('condition')}: "
                            f"{key}={value}"
                        )
                        break

        jammer_rows = payload.get("jammer_strategy", [])
        interval = int(manifest["surveillance_interval"])

        # Centroid initialization produces one non-disruptive audit strategy
        # at period 0 (including null/fixed sender slots).  Strategic
        # surveillance starts at period 1 and then repeats every interval.
        # Archived legacy runs retain the historical 0, interval, ... cycle.
        for row in jammer_rows:
            period = int(row["period"])
            mode = str(row.get("initial_audit_mode", "legacy"))
            phase = str(row.get("strategy_phase", "adaptive_disruption"))
            if phase == "initial_audit":
                expected_refresh = period == 0 and mode == "centroid"
                if not math.isclose(
                    float(row["message_mean"]),
                    float(row["segment_mean"]),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    refresh_ok = False
                    block_errors.append(
                        f"{block_id}: initial audit mean differs from centroid"
                    )
            else:
                expected_refresh = (
                    period >= 1 and (period - 1) % interval == 0
                    if mode == "centroid"
                    else period % interval == 0
                )
            if _truthy(row.get("refresh")) != expected_refresh:
                refresh_ok = False
                block_errors.append(
                    f"{block_id}: bad refresh flag at period {period}"
                )

        # Compare message means only within a strategic refresh window.
        # Initial audit is a distinct calibration phase, never part of
        # the period-1 adversarial window.
        grouped = defaultdict(list)
        for row in jammer_rows:
            period = int(row["period"])
            mode = str(row.get("initial_audit_mode", "legacy"))
            phase = str(row.get("strategy_phase", "adaptive_disruption"))
            if phase == "initial_audit":
                window = ("initial_audit", 0)
            else:
                offset = 1 if mode == "centroid" else 0
                window = ("adaptive_disruption", (period - offset) // interval)
            key = (
                row.get("reliance_mode"),
                row.get("jammer_regime", row.get("jammer_active")),
                window,
                int(row["cluster"]),
            )
            grouped[key].append(float(row["message_mean"]))

        for key, means in grouped.items():
            if means and max(means) - min(means) > tolerance:
                hold_ok = False
                block_errors.append(
                    f"{block_id}: Jammer message changed within window {key}"
                )

    observed_runs = len(all_runs)
    counts_ok = (
        unique_blocks
        and complete_block_set
        and len(payloads) == expected_blocks
        and observed_runs == expected_runs
    )

    report = {
        "pass": all(
            (
                counts_ok,
                quartet_ok,
                fingerprint_ok,
                finite_ok,
                horizon_ok,
                frozen_ok,
                hold_ok,
                refresh_ok,
                config_ok,
            )
        ),
        "design_id": manifest["design_id"],
        "expected_blocks": expected_blocks,
        "observed_blocks": len(payloads),
        "expected_runs": expected_runs,
        "observed_runs": observed_runs,
        "unique_block_ids": unique_blocks,
        "complete_block_set": complete_block_set,
        "complete_matched_quartets": quartet_ok,
        "complete_matched_blocks": quartet_ok,
        "all_matched_fingerprints": fingerprint_ok,
        "configuration_verified": config_ok,
        "all_runs_finite": finite_ok,
        "all_runs_fixed_horizon": horizon_ok,
        "frozen_lambda_invariant": frozen_ok,
        "jammer_hold_rule": hold_ok,
        "jammer_refresh_schedule": refresh_ok,
        "errors": block_errors[:200],
        "n_errors": len(block_errors),
    }

    if write_report:
        (run_root / "pilot_gate.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_dir")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    result = validate_pilot(
        args.pilot_dir,
        tolerance=args.tolerance,
        write_report=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)
