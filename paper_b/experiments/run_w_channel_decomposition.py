"""Targeted W-channel decomposition for Experiment 1.

Design:
    high segregation x frozen reliance x null sender
    H in {low, high}
    epsilon in {.05, .10, .20}
    d=2
    seeds 6001--6500

The logger is passive and decomposes cumulative evidence precision into:
Expert, same-group peers, other-group peers, and the null sender slot.
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
from paper_b.experiments.run_canonical_production import (
    _base_config,
)
from paper_b.experiments.run_matched_pilot import (
    canonical_hash,
    run_condition,
    software_versions,
)
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


EPSILONS = (0.05, 0.10, 0.20)
H_LEVELS = ("low", "high")
REFERENCE_CANONICAL_ID = "c9d09daad143"
REFERENCE_MECHANISM_ID = "9b5c35506618"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument("--numerical-min-sd", type=float, default=1e-8)
    parser.add_argument(
        "--canonical-root",
        default=(
            "production_results/paper_b_canonical/"
            "production_c9d09daad143"
        ),
    )
    parser.add_argument(
        "--mechanism-root",
        default=(
            "production_results/paper_b_mechanism_robustness/"
            "mechanism_9b5c35506618"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_w_channel_decomposition",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-reference-gate", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
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


def _scientific_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/measurement.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_w_channel_decomposition.py",
        "paper_b/W_CHANNEL_DECOMPOSITION_PLAN.md",
        "environment.yml",
    )
    payload = []
    for name in names:
        path = root / name
        payload.append(
            {
                "path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return canonical_hash(payload)


def _run_seed_epsilon(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2

    seed = int(task["seed"])
    epsilon = float(task["epsilon"])
    n = int(task["n_citizens"])

    groups = balanced_fixed_group_ids(seed=seed, n_citizens=n)
    blueprint = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n,
        group_ids=groups,
        peer_degree=int(task["peer_degree"]),
        low_homophily=float(task["low_homophily"]),
        high_homophily=float(task["high_homophily"]),
    )
    initial = exp4_initial_beliefs(
        seed=seed,
        n_citizens=n,
        group_ids=groups,
        segregation="high",
        high_group_shift=float(task["high_group_shift"]),
        residual_sd=float(task["prior_residual_sd"]),
    )

    results = []
    for h in H_LEVELS:
        base_task = {
            "n_citizens": n,
            "horizon_T": int(task["horizon"]),
            "K": int(task["k"]),
            "epsilon": epsilon,
            "credit": int(task["credit"]),
            "surveillance_interval": int(task["surveillance_interval"]),
        }
        cfg = _base_config(seed, base_task)
        cfg.update(
            {
                "mu_theta": initial,
                "initial_theta_type": "channel_exp1_high",
                "fixed_group_ids": groups,
                "structural_source_map": blueprint[h],
            }
        )

        result = run_condition(
            base_config=cfg,
            seed=seed,
            regime="segregation_high",
            environment=f"channel_exp1_h_{h}",
            reliance_mode="frozen",
            jammer_active=False,
            jammer_regime="null",
            peer_evidence_mode="source_posterior",
            frozen_ranking_mode="pre_disruption",
            k=int(task["k"]),
            design_id=str(task["design_id"]),
            block_id=f"channel_e{epsilon:.2f}__s{seed}",
            save_edge_log=False,
            record_channel_decomposition=True,
        )

        labels = {
            "experiment": "I",
            "legacy_experiment": "IV",
            "decomposition_block": "highS_frozen_null",
            "epsilon": epsilon,
            "peer_degree": int(task["peer_degree"]),
            "homophily_level": h,
            "segregation_level": "high",
            "reliance_mode": "frozen",
            "sender_regime": "null",
        }
        result["run"].update(labels)
        for key in (
            "beliefs",
            "lambda_checkpoints",
            "belief_checkpoints",
            "evidence_checkpoints",
            "channel_checkpoints",
            "channel_citizens",
        ):
            for row in result[key]:
                row.update(labels)
        results.append(result)

    payload = {
        "complete": True,
        "design_id": str(task["design_id"]),
        "seed": seed,
        "epsilon": epsilon,
        "runs": [r["run"] for r in results],
        "channel_checkpoints": [
            x for r in results for x in r["channel_checkpoints"]
        ],
        "channel_citizens": [
            x for r in results for x in r["channel_citizens"]
        ],
    }
    path = (
        Path(task["run_root"])
        / "shards"
        / f"epsilon_{epsilon:.2f}__s{seed}.json.gz"
    )
    _atomic_write(path, payload)
    return {
        "seed": seed,
        "epsilon": epsilon,
        "runs": len(payload["runs"]),
    }


def _valid_shard(path: Path, design_id: str) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            payload.get("complete") is True
            and payload.get("design_id") == design_id
            and len(payload.get("runs", [])) == 2
        )
    except Exception:
        return False


def _target_key(row: dict) -> tuple:
    return (
        int(float(row["seed"])),
        float(row["epsilon"]),
        row["homophily_level"],
    )


def _reference_key_canonical(row: dict) -> tuple | None:
    if row.get("experiment") != "IV":
        return None
    if row.get("production_block") != "null_primary":
        return None
    if row.get("segregation_level") != "high":
        return None
    if row.get("reliance_mode") != "frozen":
        return None
    if row.get("sender_regime") != "null":
        return None
    return (
        int(float(row["seed"])),
        0.05,
        row["homophily_level"],
    )


def _reference_key_mechanism(row: dict) -> tuple | None:
    if row.get("segregation_level") != "high":
        return None
    if row.get("reliance_mode") != "frozen":
        return None
    if row.get("sender_regime") != "null":
        return None
    label = row.get("spec_label", "")
    if label not in {"epsilon_0.10_d2", "epsilon_0.20_d2"}:
        return None
    return (
        int(float(row["seed"])),
        float(row["epsilon"]),
        row["homophily_level"],
    )


def _reference_gate(
    *,
    target_rows: list[dict],
    canonical_root: Path,
    mechanism_root: Path,
) -> dict:
    canonical_path = canonical_root / "runs.csv"
    mechanism_path = mechanism_root / "runs.csv"
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    if not mechanism_path.exists():
        raise FileNotFoundError(mechanism_path)

    reference = {}
    for row in _read_csv(canonical_path):
        key = _reference_key_canonical(row)
        if key is not None:
            reference[key] = row
    for row in _read_csv(mechanism_path):
        key = _reference_key_mechanism(row)
        if key is not None:
            reference[key] = row

    target = {_target_key(row): row for row in target_rows}
    missing = sorted(set(target) ^ set(reference))
    if missing:
        return {
            "pass": False,
            "key_mismatch_count": len(missing),
            "max_abs_diff": math.inf,
            "fingerprints_match": False,
        }

    numeric_columns = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "mean_belief",
        "belief_variance",
        "squared_displacement",
        "effective_homophily",
        "gateway_incoming_reliance_share",
    )
    max_diff = 0.0
    comparisons = 0
    fingerprints_match = True

    for key in sorted(target):
        a = target[key]
        b = reference[key]
        if (
            a.get("structural_fingerprint") != b.get("structural_fingerprint")
            or a.get("initial_state_fingerprint")
            != b.get("initial_state_fingerprint")
        ):
            fingerprints_match = False

        for column in numeric_columns:
            if column not in a or column not in b:
                continue
            av = float(a[column])
            bv = float(b[column])
            if math.isnan(av) and math.isnan(bv):
                continue
            if not (math.isfinite(av) and math.isfinite(bv)):
                max_diff = math.inf
                continue
            max_diff = max(max_diff, abs(av - bv))
            comparisons += 1

    tolerance = 1e-12
    return {
        "pass": bool(
            fingerprints_match
            and max_diff <= tolerance
        ),
        "tolerance": tolerance,
        "key_mismatch_count": 0,
        "fingerprints_match": fingerprints_match,
        "max_abs_diff": max_diff,
        "numeric_comparisons": comparisons,
    }


def _cell_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["epsilon"]),
                row["homophily_level"],
            )
        ].append(row)

    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "W_expert_channel",
        "W_same_peer_channel",
        "W_other_peer_channel",
        "W_corrective_crosscut_channel",
        "I_expert_channel",
        "I_same_peer_channel",
        "I_other_peer_channel",
        "Q_expert_channel",
        "Q_same_peer_channel",
        "Q_other_peer_channel",
        "Q_corrective_crosscut_channel",
    )

    out = []
    for (epsilon, h), members in sorted(grouped.items()):
        row = {
            "epsilon": epsilon,
            "homophily_level": h,
            "n": len(members),
        }
        for metric in metrics:
            values = np.asarray(
                [float(member[metric]) for member in members],
                dtype=float,
            )
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
        out.append(row)
    return out


def _checkpoint_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["epsilon"]),
                row["homophily_level"],
                int(float(row["horizon_step"])),
            )
        ].append(row)

    metrics = (
        "W_expert_channel",
        "W_same_peer_channel",
        "W_other_peer_channel",
        "W_corrective_crosscut_channel",
        "I_expert_channel",
        "I_same_peer_channel",
        "I_other_peer_channel",
        "Q_expert_channel",
        "Q_same_peer_channel",
        "Q_other_peer_channel",
    )
    out = []
    for (epsilon, h, horizon), members in sorted(grouped.items()):
        row = {
            "epsilon": epsilon,
            "homophily_level": h,
            "horizon_step": horizon,
            "n": len(members),
        }
        for metric in metrics:
            values = [float(member[metric]) for member in members]
            row[f"{metric}_mean"] = float(np.mean(values))
        out.append(row)
    return out


def _mechanism_collapse_rows(runs: list[dict]) -> list[dict]:
    out = []
    for row in runs:
        out.append(
            {
                "seed": int(row["seed"]),
                "epsilon": float(row["epsilon"]),
                "homophily_level": row["homophily_level"],
                "mse_truth": float(row["mse_truth"]),
                "rmse_truth": float(row["rmse_truth"]),
                "W_expert": float(row["W_expert_channel"]),
                "W_same_peer": float(row["W_same_peer_channel"]),
                "W_other_peer": float(row["W_other_peer_channel"]),
                "W_expert_plus_other": float(
                    row["W_corrective_crosscut_channel"]
                ),
                "I_expert": float(row["I_expert_channel"]),
                "I_same_peer": float(row["I_same_peer_channel"]),
                "I_other_peer": float(row["I_other_peer_channel"]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    seeds = list(
        range(int(args.seed_start), int(args.seed_start) + int(args.seeds))
    )

    canonical_match = bool(
        int(args.seeds) == 500
        and int(args.seed_start) == 6001
        and int(args.n_citizens) == 100
        and int(args.horizon) == 400
        and int(args.credit) == 20
        and int(args.k) == 1
        and int(args.surveillance_interval) == 5
        and int(args.peer_degree) == 2
        and math.isclose(float(args.low_homophily), 0.50)
        and math.isclose(float(args.high_homophily), 0.90)
        and math.isclose(float(args.high_group_shift), 3.0)
        and math.isclose(float(args.prior_residual_sd), 1.0)
        and math.isclose(float(args.numerical_min_sd), 1e-8)
    )

    design = {
        "purpose": "Paper B exact W-channel decomposition",
        "reference_canonical_design": REFERENCE_CANONICAL_ID,
        "reference_mechanism_design": REFERENCE_MECHANISM_ID,
        "seeds": seeds,
        "epsilons": list(EPSILONS),
        "homophily_levels": list(H_LEVELS),
        "segregation_level": "high",
        "reliance_mode": "frozen",
        "sender_regime": "null",
        "peer_degree": int(args.peer_degree),
        "n_citizens": int(args.n_citizens),
        "horizon_T": int(args.horizon),
        "credit": int(args.credit),
        "K": int(args.k),
        "surveillance_interval": int(args.surveillance_interval),
        "low_homophily": float(args.low_homophily),
        "high_homophily": float(args.high_homophily),
        "high_group_shift": float(args.high_group_shift),
        "prior_residual_sd": float(args.prior_residual_sd),
        "tau_social": 1.0,
        "peer_evidence_mode": "source_posterior",
        "frozen_ranking_mode": "pre_disruption",
        "canonical_base_match": canonical_match,
        "software_versions": software_versions(),
        "scientific_code_fingerprint": _scientific_fingerprint(),
    }
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id

    root = Path(args.output_dir) / f"channel_{design_id}"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / "channel_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    for epsilon in EPSILONS:
        for seed in seeds:
            path = (
                root
                / "shards"
                / f"epsilon_{epsilon:.2f}__s{seed}.json.gz"
            )
            if args.resume and _valid_shard(path, design_id):
                continue
            tasks.append(
                {
                    "design_id": design_id,
                    "run_root": str(root),
                    "seed": seed,
                    "epsilon": epsilon,
                    "n_citizens": int(args.n_citizens),
                    "horizon": int(args.horizon),
                    "credit": int(args.credit),
                    "k": int(args.k),
                    "surveillance_interval": int(
                        args.surveillance_interval
                    ),
                    "peer_degree": int(args.peer_degree),
                    "low_homophily": float(args.low_homophily),
                    "high_homophily": float(args.high_homophily),
                    "high_group_shift": float(args.high_group_shift),
                    "prior_residual_sd": float(args.prior_residual_sd),
                    "numerical_min_sd": float(args.numerical_min_sd),
                }
            )

    if tasks:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=context,
        ) as pool:
            futures = [pool.submit(_run_seed_epsilon, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if i % max(int(args.progress_every), 1) == 0:
                    print(
                        f"[{i}/{len(futures)}] "
                        f"epsilon={result['epsilon']:.2f} "
                        f"seed={result['seed']}",
                        flush=True,
                    )

    payloads = []
    for path in sorted((root / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("design_id") == design_id:
            payloads.append(payload)

    expected_shards = len(EPSILONS) * len(seeds)
    expected_runs = 2 * expected_shards
    if len(payloads) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} shards, got {len(payloads)}."
        )

    runs = [x for p in payloads for x in p["runs"]]
    checkpoints = [
        x for p in payloads for x in p["channel_checkpoints"]
    ]
    citizens = [x for p in payloads for x in p["channel_citizens"]]

    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs, got {len(runs)}."
        )

    finite = all(
        math.isfinite(float(row["mse_truth"]))
        for row in runs
    )
    fixed_horizon = all(
        int(row["terminal_horizon_T"]) == int(args.horizon)
        for row in runs
    )
    null_last = min(float(row["null_last_share"]) for row in runs)

    citizen_sum_deviation = 0.0
    null_precision_max = 0.0
    for row in citizens:
        w_sum = (
            float(row["W_expert"])
            + float(row["W_same_peer"])
            + float(row["W_other_peer"])
            + float(row["W_jammer"])
        )
        citizen_sum_deviation = max(
            citizen_sum_deviation,
            abs(w_sum - 1.0),
        )
        null_precision_max = max(
            null_precision_max,
            abs(float(row["Q_jammer"])),
        )

    reference_gate = None
    if not args.skip_reference_gate:
        reference_gate = _reference_gate(
            target_rows=runs,
            canonical_root=Path(args.canonical_root),
            mechanism_root=Path(args.mechanism_root),
        )

    gate = {
        "pass": bool(
            canonical_match
            and finite
            and fixed_horizon
            and null_last == 1.0
            and citizen_sum_deviation <= 1e-12
            and null_precision_max <= 1e-12
            and (
                reference_gate is None
                or reference_gate["pass"]
            )
        ),
        "design_id": design_id,
        "canonical_base_match": canonical_match,
        "observed_shards": len(payloads),
        "expected_shards": expected_shards,
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "all_terminal_mse_finite": finite,
        "all_fixed_horizon": fixed_horizon,
        "null_last_share_minimum": null_last,
        "max_citizen_W_sum_deviation": citizen_sum_deviation,
        "max_null_precision": null_precision_max,
        "reference_gate_skipped": bool(args.skip_reference_gate),
        "behavioral_identity": reference_gate,
    }
    (root / "channel_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )

    cell_summary = _cell_summary(runs)
    checkpoint_summary = _checkpoint_summary(checkpoints)
    collapse_rows = _mechanism_collapse_rows(runs)

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "cell_summary.csv", cell_summary)
    _write_csv(root / "channel_checkpoint_summary.csv", checkpoint_summary)
    _write_csv(root / "mechanism_collapse.csv", collapse_rows)
    _write_csv(
        root / "channel_checkpoints.csv.gz",
        checkpoints,
        gzip_output=True,
    )
    _write_csv(
        root / "channel_citizens.csv.gz",
        citizens,
        gzip_output=True,
    )

    plan = (
        Path(__file__).resolve().parents[1]
        / "W_CHANNEL_DECOMPOSITION_PLAN.md"
    )
    review = root / f"paper_b_w_channel_review_{design_id}.zip"
    with zipfile.ZipFile(
        review, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for name in (
            "channel_manifest.json",
            "channel_gate.json",
            "runs.csv",
            "cell_summary.csv",
            "channel_checkpoint_summary.csv",
            "mechanism_collapse.csv",
            "channel_checkpoints.csv.gz",
        ):
            zf.write(root / name, arcname=name)
        zf.write(plan, arcname="W_CHANNEL_DECOMPOSITION_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=True))
    print(f"Review bundle: {review}")


if __name__ == "__main__":
    main()
