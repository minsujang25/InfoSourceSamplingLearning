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

    expected_conditions = {
        "adaptive__J1",
        "adaptive__J0",
        "frozen__J1",
        "frozen__J0",
    }

    for payload in payloads:
        block_id = payload.get("block_id")
        runs = payload.get("runs", [])
        all_runs.extend(runs)

        conditions = {row.get("condition") for row in runs}
        if len(runs) != 4 or conditions != expected_conditions:
            quartet_ok = False
            block_errors.append(f"{block_id}: incomplete matched quartet")

        structural = {row.get("structural_fingerprint") for row in runs}
        initial = {row.get("initial_state_fingerprint") for row in runs}
        if len(structural) != 1 or len(initial) != 1:
            fingerprint_ok = False
            block_errors.append(f"{block_id}: fingerprint mismatch")

        for row in runs:
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

        # Only active-Jammer runs emit strategy rows.  At every logged period,
        # refresh should occur exactly on scheduled surveillance periods.
        for row in jammer_rows:
            period = int(row["period"])
            expected_refresh = period % interval == 0
            if _truthy(row.get("refresh")) != expected_refresh:
                refresh_ok = False
                block_errors.append(
                    f"{block_id}: bad refresh flag at period {period}"
                )

        # Within each surveillance window and condition, the optimized message
        # mean must remain fixed for each cluster.
        grouped = defaultdict(list)
        for row in jammer_rows:
            period = int(row["period"])
            window = period // interval
            key = (
                row.get("reliance_mode"),
                bool(row.get("jammer_active")),
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
        "all_matched_fingerprints": fingerprint_ok,
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
