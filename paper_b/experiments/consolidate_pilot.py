"""Consolidate Paper B matched-pilot shards into tidy analysis files."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from paper_b.experiments.run_local_diagnostic import matched_contrasts


def _read_shard(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def _horizon_sensitivity_rows(
    belief_checkpoints: list[dict],
    *,
    short_horizon: int = 200,
    long_horizon: int = 400,
) -> tuple[list[dict], list[dict]]:
    """Build matched disruption contrasts at two fixed horizons.

    Checkpoint rows report `horizon_step` (1-indexed executed periods).  The
    first table compares D_g(T) at T=short_horizon and T=long_horizon for each
    seed/regime/network/reliance condition.  The second compares the
    adaptive-minus-frozen disruption contrast across those horizons.
    """
    by_key = {}
    for row in belief_checkpoints:
        step = row.get("horizon_step")
        if step is None:
            # Backwards-compatible fallback for older checkpoint files.
            step = int(row["period"]) + 1
        step = int(step)
        if step not in {short_horizon, long_horizon}:
            continue
        key = (
            int(row["seed"]),
            row["initial_regime"],
            row["network_environment"],
            row["reliance_mode"],
            bool(row["jammer_active"]),
            int(row["K"]),
            step,
        )
        by_key[key] = row

    disruption_rows = []
    keys = {
        (
            seed,
            regime,
            env,
            reliance,
            k,
        )
        for (seed, regime, env, reliance, jammer, k, step) in by_key
    }

    for seed, regime, env, reliance, k in sorted(keys):
        values = {}
        complete = True
        for step in (short_horizon, long_horizon):
            j1 = by_key.get((seed, regime, env, reliance, True, k, step))
            j0 = by_key.get((seed, regime, env, reliance, False, k, step))
            if j1 is None or j0 is None:
                complete = False
                break
            values[step] = {
                "d_mse": float(j1["mse_truth"]) - float(j0["mse_truth"]),
                "d_rmse": float(j1["rmse_truth"]) - float(j0["rmse_truth"]),
                "d_mae": float(j1["mae_truth"]) - float(j0["mae_truth"]),
                "j1_mse": float(j1["mse_truth"]),
                "j0_mse": float(j0["mse_truth"]),
            }
        if not complete:
            continue

        short = values[short_horizon]
        long = values[long_horizon]
        disruption_rows.append(
            {
                "seed": seed,
                "initial_regime": regime,
                "network_environment": env,
                "reliance_mode": reliance,
                "K": k,
                "short_horizon": short_horizon,
                "long_horizon": long_horizon,
                "delta_mse_short": short["d_mse"],
                "delta_mse_long": long["d_mse"],
                "delta_mse_change": long["d_mse"] - short["d_mse"],
                "abs_delta_mse_change": abs(long["d_mse"] - short["d_mse"]),
                "delta_rmse_short": short["d_rmse"],
                "delta_rmse_long": long["d_rmse"],
                "delta_rmse_change": long["d_rmse"] - short["d_rmse"],
                "delta_mae_short": short["d_mae"],
                "delta_mae_long": long["d_mae"],
                "delta_mae_change": long["d_mae"] - short["d_mae"],
                "j1_mse_short": short["j1_mse"],
                "j1_mse_long": long["j1_mse"],
                "j0_mse_short": short["j0_mse"],
                "j0_mse_long": long["j0_mse"],
            }
        )

    index = {
        (
            row["seed"],
            row["initial_regime"],
            row["network_environment"],
            row["reliance_mode"],
            row["K"],
        ): row
        for row in disruption_rows
    }

    adaptive_frozen_rows = []
    af_keys = {
        (
            row["seed"],
            row["initial_regime"],
            row["network_environment"],
            row["K"],
        )
        for row in disruption_rows
    }
    for seed, regime, env, k in sorted(af_keys):
        adaptive = index.get((seed, regime, env, "adaptive", k))
        frozen = index.get((seed, regime, env, "frozen", k))
        if adaptive is None or frozen is None:
            continue
        short_af = (
            adaptive["delta_mse_short"] - frozen["delta_mse_short"]
        )
        long_af = (
            adaptive["delta_mse_long"] - frozen["delta_mse_long"]
        )
        adaptive_frozen_rows.append(
            {
                "seed": seed,
                "initial_regime": regime,
                "network_environment": env,
                "K": k,
                "short_horizon": short_horizon,
                "long_horizon": long_horizon,
                "adaptive_minus_frozen_short": short_af,
                "adaptive_minus_frozen_long": long_af,
                "adaptive_minus_frozen_change": long_af - short_af,
                "abs_adaptive_minus_frozen_change": abs(long_af - short_af),
            }
        )

    return disruption_rows, adaptive_frozen_rows


def consolidate(run_root: str | Path) -> dict:
    run_root = Path(run_root)
    manifest = json.loads(
        (run_root / "pilot_manifest.json").read_text(encoding="utf-8")
    )

    shard_paths = sorted((run_root / "shards").glob("*.json.gz"))
    payloads = [_read_shard(path) for path in shard_paths]

    runs = []
    beliefs = []
    lambdas = []
    belief_checkpoints = []
    jammer = []
    edges = []
    blocks = []

    for payload in payloads:
        runs.extend(payload.get("runs", []))
        beliefs.extend(payload.get("terminal_beliefs", []))
        lambdas.extend(payload.get("lambda_checkpoints", []))
        belief_checkpoints.extend(payload.get("belief_checkpoints", []))
        jammer.extend(payload.get("jammer_strategy", []))
        edges.extend(payload.get("reliance_edges", []))
        blocks.append(
            {
                "design_id": payload.get("design_id"),
                "block_id": payload.get("block_id"),
                "seed": payload.get("seed"),
                "initial_regime": payload.get("initial_regime"),
                "network_environment": payload.get("network_environment"),
                "initial_state_fingerprint": payload.get(
                    "initial_state_fingerprint"
                ),
                "structural_fingerprint": payload.get("structural_fingerprint"),
                "complete": payload.get("complete"),
            }
        )

    jammer_contrasts, adaptive_frozen = matched_contrasts(runs)
    horizon_sensitivity, horizon_af_sensitivity = _horizon_sensitivity_rows(
        belief_checkpoints
    )

    _write_csv(run_root / "runs.csv", runs)
    _write_csv(run_root / "jammer_contrasts.csv", jammer_contrasts)
    _write_csv(run_root / "adaptive_frozen_contrasts.csv", adaptive_frozen)
    _write_gzip_csv(run_root / "terminal_beliefs.csv.gz", beliefs)
    _write_csv(run_root / "lambda_checkpoints.csv", lambdas)
    _write_csv(run_root / "belief_checkpoints.csv", belief_checkpoints)
    if horizon_sensitivity:
        _write_csv(
            run_root / "horizon_sensitivity.csv",
            horizon_sensitivity,
        )
    if horizon_af_sensitivity:
        _write_csv(
            run_root / "adaptive_frozen_horizon_sensitivity.csv",
            horizon_af_sensitivity,
        )
    _write_gzip_csv(run_root / "jammer_strategy_trajectory.csv.gz", jammer)
    if edges:
        _write_gzip_csv(run_root / "reliance_checkpoints.csv.gz", edges)
    _write_csv(run_root / "block_manifest.csv", blocks)

    mse = _finite_values(runs, "mse_truth")
    delta_mse = _finite_values(jammer_contrasts, "delta_mse")
    af_delta = _finite_values(
        adaptive_frozen, "adaptive_minus_frozen_delta_mse"
    )

    by_env = defaultdict(list)
    for row in jammer_contrasts:
        try:
            by_env[row["network_environment"]].append(float(row["delta_mse"]))
        except (TypeError, ValueError):
            pass

    summary = {
        "design_id": manifest["design_id"],
        "expected_blocks": manifest["expected_blocks"],
        "observed_blocks": len(payloads),
        "expected_runs": manifest["expected_runs"],
        "observed_runs": len(runs),
        "n_terminal_belief_rows": len(beliefs),
        "n_lambda_checkpoint_rows": len(lambdas),
        "n_belief_checkpoint_rows": len(belief_checkpoints),
        "n_horizon_sensitivity_rows": len(horizon_sensitivity),
        "n_adaptive_frozen_horizon_sensitivity_rows": len(
            horizon_af_sensitivity
        ),
        "n_jammer_strategy_rows": len(jammer),
        "n_edge_checkpoint_rows": len(edges),
        "n_jammer_contrasts": len(jammer_contrasts),
        "n_adaptive_frozen_contrasts": len(adaptive_frozen),
        "mse_min": min(mse) if mse else None,
        "mse_median": (
            sorted(mse)[len(mse) // 2] if mse else None
        ),
        "mse_max": max(mse) if mse else None,
        "delta_mse_min": min(delta_mse) if delta_mse else None,
        "delta_mse_max": max(delta_mse) if delta_mse else None,
        "adaptive_minus_frozen_delta_mse_min": min(af_delta) if af_delta else None,
        "adaptive_minus_frozen_delta_mse_max": max(af_delta) if af_delta else None,
        "mean_delta_mse_by_environment": {
            env: (sum(values) / len(values) if values else None)
            for env, values in sorted(by_env.items())
        },
        "files": [
            "pilot_manifest.json",
            "block_manifest.csv",
            "runs.csv",
            "jammer_contrasts.csv",
            "adaptive_frozen_contrasts.csv",
            "terminal_beliefs.csv.gz",
            "lambda_checkpoints.csv",
            "belief_checkpoints.csv",
        ] + (
            ["horizon_sensitivity.csv"]
            if horizon_sensitivity
            else []
        ) + (
            ["adaptive_frozen_horizon_sensitivity.csv"]
            if horizon_af_sensitivity
            else []
        ) + [
            "jammer_strategy_trajectory.csv.gz",
        ] + (["reliance_checkpoints.csv.gz"] if edges else []),
    }

    (run_root / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_dir")
    args = parser.parse_args()
    result = consolidate(args.pilot_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
