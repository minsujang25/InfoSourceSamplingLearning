"""Passive W-network measurement rerun for canonical Paper B production.

Behavioral design is identical to canonical production c9d09daad143.
The only scientific addition is passive logging of cumulative evidence precision.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import model.InfoSourceSamplingLearning as model_module
from paper_b.experiments.analyze_measurement_phase_a import _structural_baseline
from paper_b.experiments.run_canonical_production import (
    CANONICAL_CREDIT,
    CANONICAL_EPSILON,
    CANONICAL_HORIZON,
    CANONICAL_K,
    CANONICAL_N_CITIZENS,
    CANONICAL_SEEDS,
    CANONICAL_SURVEILLANCE_INTERVAL,
    _run_exp3,
    _run_exp4,
    expected_conditions,
    resolve_experiments,
    resolve_seeds,
)
from paper_b.experiments.run_matched_pilot import canonical_hash, software_versions


REFERENCE_DESIGN_ID = "c9d09daad143"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="500")
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--expert-access-share", type=float, default=0.10)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument("--numerical-min-sd", type=float, default=1e-8)
    parser.add_argument("--experiments", default="III,IV")
    parser.add_argument(
        "--reference-root",
        default=(
            "production_results/paper_b_canonical/"
            "production_c9d09daad143"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_measurement_audit",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-reference-gate", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_csv(path: Path, rows: list[dict], *, gzip_output: bool = False) -> None:
    if not rows:
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    handle = (
        gzip.open(path, "wt", newline="", encoding="utf-8")
        if gzip_output
        else open(path, "w", newline="", encoding="utf-8")
    )
    with handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _labels(row: dict) -> tuple:
    return (
        row["experiment"],
        int(float(row["seed"])),
        row["production_block"],
        row.get("homophily_level", ""),
        row.get("segregation_level", ""),
        row.get("redundancy_level", ""),
        row["reliance_mode"],
        row["sender_regime"],
    )


def _run_block(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2

    experiment = str(task["experiment"])
    seed = int(task["seed"])
    results = _run_exp4(seed, task) if experiment == "IV" else _run_exp3(seed, task)

    payload = {
        "complete": True,
        "design_id": task["design_id"],
        "reference_design_id": REFERENCE_DESIGN_ID,
        "experiment": experiment,
        "seed": seed,
        "runs": [r["run"] for r in results],
        "terminal_beliefs": [x for r in results for x in r["beliefs"]],
        "lambda_checkpoints": [x for r in results for x in r["lambda_checkpoints"]],
        "belief_checkpoints": [x for r in results for x in r["belief_checkpoints"]],
        "evidence_checkpoints": [
            x for r in results for x in r["evidence_checkpoints"]
        ],
        "jammer_strategy": [x for r in results for x in r["jammer_strategy"]],
    }
    path = (
        Path(task["run_root"])
        / "shards"
        / f"{experiment}__s{seed}.json.gz"
    )
    _atomic_write(path, payload)
    return {"experiment": experiment, "seed": seed, "runs": len(payload["runs"])}


def _valid_shard(path: Path, design_id: str, expected_runs: int) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            p = json.load(handle)
        return (
            p.get("complete") is True
            and p.get("design_id") == design_id
            and len(p.get("runs", [])) == expected_runs
        )
    except Exception:
        return False


def _max_numeric_diff(
    left_rows: list[dict],
    right_rows: list[dict],
    *,
    key_fn,
    columns: tuple[str, ...],
) -> tuple[float, int]:
    left = {key_fn(row): row for row in left_rows}
    right = {key_fn(row): row for row in right_rows}
    if set(left) != set(right):
        return math.inf, len(set(left) ^ set(right))

    max_diff = 0.0
    comparisons = 0
    for key in left:
        a = left[key]
        b = right[key]
        for col in columns:
            if col not in a or col not in b:
                continue
            try:
                av = float(a[col])
                bv = float(b[col])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(av) and math.isfinite(bv)):
                if math.isnan(av) and math.isnan(bv):
                    continue
                return math.inf, comparisons
            max_diff = max(max_diff, abs(av - bv))
            comparisons += 1
    return max_diff, comparisons


def _identity_gate(
    *,
    reference_root: Path,
    runs: list[dict],
    beliefs: list[dict],
    belief_checkpoints: list[dict],
    lambda_checkpoints: list[dict],
) -> dict:
    required = {
        "runs": reference_root / "runs.csv",
        "beliefs": reference_root / "terminal_beliefs.csv.gz",
        "belief_checkpoints": reference_root / "belief_checkpoints.csv.gz",
        "lambda_checkpoints": reference_root / "lambda_checkpoints.csv.gz",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)

    ref_runs = _read_csv(required["runs"])
    ref_beliefs = _read_csv(required["beliefs"])
    ref_belief_cp = _read_csv(required["belief_checkpoints"])
    ref_lambda_cp = _read_csv(required["lambda_checkpoints"])

    run_diff, run_n = _max_numeric_diff(
        ref_runs,
        runs,
        key_fn=_labels,
        columns=(
            "mse_truth",
            "rmse_truth",
            "mae_truth",
            "mean_belief",
            "belief_variance",
            "squared_displacement",
            "effective_homophily",
            "dominant_same_group_cycle_share",
            "effective_incoming_hhi",
            "gateway_incoming_reliance_share",
        ),
    )

    belief_key = lambda r: (*_labels(r), int(float(r["citizen_pos"])))
    belief_diff, belief_n = _max_numeric_diff(
        ref_beliefs,
        beliefs,
        key_fn=belief_key,
        columns=("terminal_mu_theta", "terminal_sd_theta"),
    )

    cp_key = lambda r: (*_labels(r), int(float(r["period"])))
    # The measurement rerun adds exact T=100 (period 99), so compare only keys
    # present in the frozen reference.
    ref_belief_keys = {cp_key(r) for r in ref_belief_cp}
    new_belief_common = [r for r in belief_checkpoints if cp_key(r) in ref_belief_keys]
    belief_cp_diff, belief_cp_n = _max_numeric_diff(
        ref_belief_cp,
        new_belief_common,
        key_fn=cp_key,
        columns=(
            "mse_truth",
            "rmse_truth",
            "mae_truth",
            "belief_variance",
            "squared_displacement",
        ),
    )

    ref_lambda_keys = {cp_key(r) for r in ref_lambda_cp}
    new_lambda_common = [r for r in lambda_checkpoints if cp_key(r) in ref_lambda_keys]
    lambda_cp_diff, lambda_cp_n = _max_numeric_diff(
        ref_lambda_cp,
        new_lambda_common,
        key_fn=cp_key,
        columns=(
            "effective_homophily",
            "peer_reliance_mass",
            "dominant_expert_reach_share",
            "dominant_jammer_reach_share",
            "dominant_citizen_cycle_share",
            "dominant_same_group_cycle_share",
            "effective_incoming_hhi",
            "gateway_incoming_reliance_share",
            "posterior_sd_median",
        ),
    )

    fingerprints_match = True
    ref_idx = {_labels(r): r for r in ref_runs}
    for row in runs:
        ref = ref_idx.get(_labels(row))
        if ref is None:
            fingerprints_match = False
            break
        if (
            row.get("structural_fingerprint") != ref.get("structural_fingerprint")
            or row.get("initial_state_fingerprint") != ref.get("initial_state_fingerprint")
        ):
            fingerprints_match = False
            break

    tolerance = 1e-12
    passed = bool(
        fingerprints_match
        and run_diff <= tolerance
        and belief_diff <= tolerance
        and belief_cp_diff <= tolerance
        and lambda_cp_diff <= tolerance
    )
    return {
        "pass": passed,
        "tolerance": tolerance,
        "structural_and_initial_fingerprints_match": fingerprints_match,
        "max_abs_run_metric_diff": run_diff,
        "run_numeric_comparisons": run_n,
        "max_abs_terminal_belief_diff": belief_diff,
        "terminal_belief_numeric_comparisons": belief_n,
        "max_abs_belief_checkpoint_diff": belief_cp_diff,
        "belief_checkpoint_numeric_comparisons": belief_cp_n,
        "max_abs_lambda_checkpoint_diff": lambda_cp_diff,
        "lambda_checkpoint_numeric_comparisons": lambda_cp_n,
    }


def _a_lambda_w_seed_rows(
    runs: list[dict],
    *,
    n_citizens: int,
) -> list[dict]:
    cache = {}
    out = []
    for row in runs:
        seed = int(row["seed"])
        key = (
            row["experiment"],
            seed,
            row.get("homophily_level", ""),
            row.get("redundancy_level", ""),
        )
        if key not in cache:
            cache[key] = _structural_baseline(
                seed,
                row,
                n_citizens=int(n_citizens),
            )
        a = cache[key]

        result = {
            "experiment": row["experiment"],
            "seed": seed,
            "production_block": row["production_block"],
            "homophily_level": row.get("homophily_level", ""),
            "segregation_level": row.get("segregation_level", ""),
            "redundancy_level": row.get("redundancy_level", ""),
            "reliance_mode": row["reliance_mode"],
            "sender_regime": row["sender_regime"],
            **a,
            "Lambda_effective_homophily": float(row.get("effective_homophily", math.nan)),
            "Lambda_incoming_hhi": float(row.get("effective_incoming_hhi", math.nan)),
            "Lambda_gateway_total_share": float(
                row.get("gateway_incoming_reliance_share", math.nan)
            ),
            "Lambda_gateway_peer_conditional_share": float(
                row.get("Lambda_gateway_peer_conditional_share", math.nan)
            ),
            "Lambda_peer_incoming_hhi": float(
                row.get("Lambda_peer_incoming_hhi", math.nan)
            ),
            "W_effective_homophily": float(
                row.get("W_effective_homophily", math.nan)
            ),
            "W_incoming_hhi": float(row.get("W_incoming_hhi", math.nan)),
            "W_gateway_total_share": float(
                row.get("W_gateway_incoming_share", math.nan)
            ),
            "W_gateway_peer_conditional_share": float(
                row.get("W_gateway_peer_conditional_share", math.nan)
            ),
            "W_peer_incoming_hhi": float(
                row.get("W_peer_incoming_hhi", math.nan)
            ),
            "W_dominant_same_group_cycle_share": float(
                row.get("W_dominant_same_group_cycle_share", math.nan)
            ),
            "Lambda_dominant_same_group_cycle_share": float(
                row.get("dominant_same_group_cycle_share", math.nan)
            ),
            "null_last_share": float(row.get("null_last_share", math.nan)),
        }

        h_a = float(a.get("A_uniform_structural_homophily", math.nan))
        if math.isfinite(h_a):
            result["Lambda_minus_A_homophily"] = (
                result["Lambda_effective_homophily"] - h_a
            )
            result["W_minus_A_homophily"] = (
                result["W_effective_homophily"] - h_a
            )

        hhi_a = float(a.get("A_uniform_incoming_hhi", math.nan))
        if math.isfinite(hhi_a):
            result["Lambda_minus_A_incoming_hhi"] = (
                result["Lambda_incoming_hhi"] - hhi_a
            )
            result["W_minus_A_incoming_hhi"] = (
                result["W_incoming_hhi"] - hhi_a
            )

        gateway_a = float(a.get("A_uniform_gateway_total_share", math.nan))
        if math.isfinite(gateway_a):
            result["Lambda_minus_A_gateway_total_share"] = (
                result["Lambda_gateway_total_share"] - gateway_a
            )
            result["W_minus_A_gateway_total_share"] = (
                result["W_gateway_total_share"] - gateway_a
            )

        gateway_peer_a = float(
            a.get("A_uniform_gateway_peer_conditional_share", math.nan)
        )
        if math.isfinite(gateway_peer_a):
            result["Lambda_minus_A_gateway_peer_conditional_share"] = (
                result["Lambda_gateway_peer_conditional_share"]
                - gateway_peer_a
            )
            result["W_minus_A_gateway_peer_conditional_share"] = (
                result["W_gateway_peer_conditional_share"]
                - gateway_peer_a
            )

        peer_hhi_a = float(a.get("A_uniform_peer_incoming_hhi", math.nan))
        if math.isfinite(peer_hhi_a):
            result["Lambda_minus_A_peer_incoming_hhi"] = (
                result["Lambda_peer_incoming_hhi"] - peer_hhi_a
            )
            result["W_minus_A_peer_incoming_hhi"] = (
                result["W_peer_incoming_hhi"] - peer_hhi_a
            )

        out.append(result)
    return out


def _cell_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["experiment"],
            row["production_block"],
            row["homophily_level"],
            row["segregation_level"],
            row["redundancy_level"],
            row["reliance_mode"],
            row["sender_regime"],
        )
        groups[key].append(row)

    metrics = [
        key
        for key in rows[0]
        if key.startswith(("A_", "Lambda_", "W_", "null_last"))
    ] if rows else []

    out = []
    for key, members in sorted(groups.items()):
        result = {
            "experiment": key[0],
            "production_block": key[1],
            "homophily_level": key[2],
            "segregation_level": key[3],
            "redundancy_level": key[4],
            "reliance_mode": key[5],
            "sender_regime": key[6],
            "n": len(members),
        }
        for metric in metrics:
            values = []
            for member in members:
                try:
                    value = float(member.get(metric, math.nan))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            result[f"{metric}_mean"] = (
                float(np.mean(values)) if values else math.nan
            )
            result[f"{metric}_median"] = (
                float(np.median(values)) if values else math.nan
            )
        out.append(result)
    return out


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    experiments = resolve_experiments(args.experiments)

    canonical_match = bool(
        tuple(seeds) == CANONICAL_SEEDS
        and tuple(experiments) == ("III", "IV")
        and int(args.n_citizens) == CANONICAL_N_CITIZENS
        and int(args.horizon) == CANONICAL_HORIZON
        and int(args.k) == CANONICAL_K
        and math.isclose(float(args.epsilon), CANONICAL_EPSILON)
        and int(args.credit) == CANONICAL_CREDIT
        and int(args.surveillance_interval) == CANONICAL_SURVEILLANCE_INTERVAL
        and math.isclose(float(args.numerical_min_sd), 1e-8)
        and math.isclose(float(args.expert_access_share), 0.10)
        and math.isclose(float(args.low_homophily), 0.50)
        and math.isclose(float(args.high_homophily), 0.90)
        and math.isclose(float(args.high_group_shift), 3.0)
        and math.isclose(float(args.prior_residual_sd), 1.0)
    )

    plan_path = Path(__file__).resolve().parents[1] / "MEASUREMENT_AUDIT_PLAN.md"
    code_paths = (
        Path(__file__),
        Path(__file__).resolve().parents[2] / "model" / "InfoSourceSamplingLearning.py",
        Path(__file__).resolve().parents[1] / "measurement.py",
        plan_path,
    )
    code_fp = canonical_hash([
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in code_paths
    ])

    design = {
        "purpose": "Paper B passive A-Lambda-W measurement audit",
        "reference_design_id": REFERENCE_DESIGN_ID,
        "seeds": seeds,
        "experiments": list(experiments),
        "n_citizens": int(args.n_citizens),
        "horizon_T": int(args.horizon),
        "K": int(args.k),
        "epsilon": float(args.epsilon),
        "credit": int(args.credit),
        "surveillance_interval": int(args.surveillance_interval),
        "peer_degree": int(args.peer_degree),
        "expert_access_share": float(args.expert_access_share),
        "low_homophily": float(args.low_homophily),
        "high_homophily": float(args.high_homophily),
        "high_group_shift": float(args.high_group_shift),
        "prior_residual_sd": float(args.prior_residual_sd),
        "numerical_min_sd": float(args.numerical_min_sd),
        "software_versions": software_versions(),
        "measurement_code_fingerprint": code_fp,
        "canonical_design_match": canonical_match,
    }
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id

    root = Path(args.output_dir) / f"measurement_{design_id}"
    shards = root / "shards"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    shards.mkdir(parents=True, exist_ok=True)
    (root / "measurement_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    for experiment in experiments:
        expected_n = len(expected_conditions(experiment))
        for seed in seeds:
            path = shards / f"{experiment}__s{seed}.json.gz"
            if args.resume and _valid_shard(path, design_id, expected_n):
                continue
            tasks.append({
                **design,
                "experiment": experiment,
                "seed": seed,
                "run_root": str(root),
            })

    if tasks:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=context,
        ) as pool:
            futures = [pool.submit(_run_block, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if i % max(int(args.progress_every), 1) == 0:
                    print(
                        f"[{i}/{len(futures)}] {result['experiment']} "
                        f"seed={result['seed']} runs={result['runs']}",
                        flush=True,
                    )

    payloads = []
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            p = json.load(handle)
        if p.get("design_id") == design_id:
            payloads.append(p)

    expected_blocks = len(seeds) * len(experiments)
    if len(payloads) != expected_blocks:
        raise RuntimeError(
            f"Expected {expected_blocks} completed blocks, got {len(payloads)}."
        )

    runs = [x for p in payloads for x in p["runs"]]
    beliefs = [x for p in payloads for x in p["terminal_beliefs"]]
    lambda_cp = [x for p in payloads for x in p["lambda_checkpoints"]]
    belief_cp = [x for p in payloads for x in p["belief_checkpoints"]]
    evidence_cp = [x for p in payloads for x in p["evidence_checkpoints"]]

    expected_runs = len(seeds) * sum(len(expected_conditions(e)) for e in experiments)
    if len(runs) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs, got {len(runs)}.")

    reference_gate = None
    if not args.skip_reference_gate:
        reference_gate = _identity_gate(
            reference_root=Path(args.reference_root),
            runs=runs,
            beliefs=beliefs,
            belief_checkpoints=belief_cp,
            lambda_checkpoints=lambda_cp,
        )

    null_rows = [r for r in runs if r["sender_regime"] == "null"]
    null_last_min = min(
        float(r["null_last_share"])
        for r in null_rows
        if math.isfinite(float(r["null_last_share"]))
    ) if null_rows else math.nan

    a_lambda_w = _a_lambda_w_seed_rows(
        runs,
        n_citizens=int(args.n_citizens),
    )
    a_lambda_w_summary = _cell_summary(a_lambda_w)

    gate = {
        "pass": bool(
            canonical_match
            and (reference_gate is None or reference_gate["pass"])
            and null_last_min == 1.0
        ),
        "design_id": design_id,
        "reference_design_id": REFERENCE_DESIGN_ID,
        "canonical_design_match": canonical_match,
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "null_last_share_minimum": null_last_min,
        "reference_gate_skipped": bool(args.skip_reference_gate),
        "behavioral_identity": reference_gate,
    }
    (root / "measurement_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "terminal_beliefs.csv.gz", beliefs, gzip_output=True)
    _write_csv(root / "lambda_checkpoints.csv.gz", lambda_cp, gzip_output=True)
    _write_csv(root / "belief_checkpoints.csv.gz", belief_cp, gzip_output=True)
    _write_csv(root / "evidence_checkpoints.csv.gz", evidence_cp, gzip_output=True)
    _write_csv(root / "A_lambda_W_seed_metrics.csv", a_lambda_w)
    _write_csv(root / "A_lambda_W_cell_summary.csv", a_lambda_w_summary)

    review = root / f"paper_b_measurement_review_{design_id}.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "measurement_manifest.json",
            "measurement_gate.json",
            "runs.csv",
            "A_lambda_W_seed_metrics.csv",
            "A_lambda_W_cell_summary.csv",
        ):
            zf.write(root / name, arcname=name)
        # Checkpoint evidence is compact enough to include for review.
        zf.write(
            root / "evidence_checkpoints.csv.gz",
            arcname="evidence_checkpoints.csv.gz",
        )
        zf.write(
            root / "lambda_checkpoints.csv.gz",
            arcname="lambda_checkpoints.csv.gz",
        )
        zf.write(
            root / "belief_checkpoints.csv.gz",
            arcname="belief_checkpoints.csv.gz",
        )
        zf.write(plan_path, arcname="MEASUREMENT_AUDIT_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=True))
    print(f"Review bundle: {review}")


if __name__ == "__main__":
    main()
