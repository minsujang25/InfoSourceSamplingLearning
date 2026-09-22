"""Compare Mesa 2.4 and Mesa 3.5 matched-seed backtest snapshots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STRICT_MAX_ABS = 1e-8
STRICT_RMSE = 1e-10
REVIEW_MAX_ABS = 1e-6
REVIEW_SUMMARY_ABS = 1e-6


def flatten(values):
    return [x for row in values for x in row]


def rmse(a, b):
    if len(a) != len(b):
        return math.inf
    if not a:
        return 0.0
    diffs = [paired_abs_diff(x, y) for x, y in zip(a, b)]
    if any(math.isinf(d) for d in diffs):
        return math.inf
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def paired_abs_diff(x, y):
    if math.isnan(x) and math.isnan(y):
        return 0.0
    if math.isnan(x) or math.isnan(y):
        return math.inf
    if math.isinf(x) or math.isinf(y):
        return 0.0 if x == y else math.inf
    return abs(x - y)


def max_abs(a, b):
    if len(a) != len(b):
        return math.inf
    if not a:
        return 0.0
    return max(paired_abs_diff(x, y) for x, y in zip(a, b))


def contains_nonfinite(values):
    return any(not math.isfinite(x) for x in values)


def keyed(payload):
    return {(r["scenario_id"], r["seed"]): r for r in payload["runs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    legacy = json.loads(args.legacy.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    left, right = keyed(legacy), keyed(current)

    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise SystemExit(
            f"Run-key mismatch. Missing legacy={missing_left}; missing current={missing_right}"
        )

    results = []
    strict_failures = []
    review_cases = []

    for key in sorted(left):
        a, b = left[key], right[key]
        structural = {
            "steps_equal": a["steps"] == b["steps"],
            "edges_equal": a["network_edges"] == b["network_edges"],
            "clusters_equal": a["jammer_clusters"] == b["jammer_clusters"],
            "trajectory_shape_equal": (
                len(a["trajectory"]) == len(b["trajectory"])
                and all(
                    len(x) == len(y)
                    for x, y in zip(a["trajectory"], b["trajectory"])
                )
            ),
        }

        ta, tb = flatten(a["trajectory"]), flatten(b["trajectory"])
        path_max = max_abs(ta, tb)
        path_rmse = rmse(ta, tb)

        fa = [row["mu_theta"] for row in a["final_by_pos"]]
        fb = [row["mu_theta"] for row in b["final_by_pos"]]
        final_max = max_abs(fa, fb)

        summary_abs = {
            name: paired_abs_diff(a["summary"][name], b["summary"][name])
            for name in ("mean_final", "sd_final", "mae_truth")
        }
        nonfinite_present = (
            contains_nonfinite(ta)
            or contains_nonfinite(tb)
            or any(not math.isfinite(v) for v in a["summary"].values())
            or any(not math.isfinite(v) for v in b["summary"].values())
        )

        strict = (
            all(structural.values())
            and path_max <= STRICT_MAX_ABS
            and path_rmse <= STRICT_RMSE
            and final_max <= STRICT_MAX_ABS
        )
        review = (
            all(structural.values())
            and path_max <= REVIEW_MAX_ABS
            and max(summary_abs.values()) <= REVIEW_SUMMARY_ABS
        )

        row = {
            "scenario_id": key[0],
            "seed": key[1],
            **structural,
            "trajectory_max_abs": path_max,
            "trajectory_rmse": path_rmse,
            "final_max_abs": final_max,
            "summary_abs": summary_abs,
            "nonfinite_present": nonfinite_present,
            "strict_pass": strict,
            "review_band_pass": review,
        }
        results.append(row)

        if not strict:
            strict_failures.append(key)
            if review:
                review_cases.append(key)

    report = {
        "thresholds": {
            "strict_max_abs": STRICT_MAX_ABS,
            "strict_rmse": STRICT_RMSE,
            "review_max_abs": REVIEW_MAX_ABS,
            "review_summary_abs": REVIEW_SUMMARY_ABS,
        },
        "n_runs": len(results),
        "strict_passes": sum(r["strict_pass"] for r in results),
        "strict_failures": [list(k) for k in strict_failures],
        "review_band_cases": [list(k) for k in review_cases],
        "nonfinite_runs": [
            [r["scenario_id"], r["seed"]]
            for r in results
            if r["nonfinite_present"]
        ],
        "results": results,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Strict Mesa migration backtest: "
        f"{report['strict_passes']}/{report['n_runs']} passed."
    )
    if strict_failures:
        print("Strict failures:")
        for scenario, seed in strict_failures:
            print(f"  - {scenario}, seed={seed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
