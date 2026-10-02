"""Targeted Experiment-IV robustness for the Expert-rank mechanism.

Precommitted blocks:
1. epsilon robustness: d=2, epsilon in {.10, .20}, full H x S x reliance null grid.
2. degree robustness: d=4, epsilon=.05, full H x S x reliance null grid.

Canonical epsilon=.05,d=2 results are not re-simulated.
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
import statistics
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import model.InfoSourceSamplingLearning as model_module
from paper_b.experiments.run_canonical_production import (
    CANONICAL_CREDIT,
    CANONICAL_HORIZON,
    CANONICAL_K,
    CANONICAL_N_CITIZENS,
    CANONICAL_SEEDS,
    CANONICAL_SURVEILLANCE_INTERVAL,
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


H_LEVELS = ("low", "high")
S_LEVELS = ("low", "high")
RELIANCE_MODES = ("adaptive", "frozen")
EPSILON_ROBUSTNESS = (0.10, 0.20)
DEGREE_ROBUSTNESS = (4,)
REFERENCE_DESIGN_ID = "c9d09daad143"


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
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument("--numerical-min-sd", type=float, default=1e-8)
    parser.add_argument(
        "--blocks",
        default="epsilon,degree",
        help="Comma-separated subset of epsilon,degree.",
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_mechanism_robustness",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _resolve_blocks(value: str) -> tuple[str, ...]:
    blocks = tuple(x.strip().lower() for x in value.split(",") if x.strip())
    invalid = sorted(set(blocks) - {"epsilon", "degree"})
    if invalid:
        raise ValueError(f"Unknown blocks: {invalid}")
    if not blocks:
        raise ValueError("At least one block is required.")
    return blocks


def _specs(blocks: tuple[str, ...]) -> list[dict]:
    out = []
    if "epsilon" in blocks:
        for epsilon in EPSILON_ROBUSTNESS:
            out.append(
                {
                    "robustness_block": "epsilon",
                    "epsilon": float(epsilon),
                    "peer_degree": 2,
                    "spec_label": f"epsilon_{epsilon:.2f}_d2",
                }
            )
    if "degree" in blocks:
        for degree in DEGREE_ROBUSTNESS:
            out.append(
                {
                    "robustness_block": "degree",
                    "epsilon": 0.05,
                    "peer_degree": int(degree),
                    "spec_label": f"epsilon_0.05_d{degree}",
                }
            )
    return out


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


def _scientific_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/measurement.py",
        "paper_b/metrics.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_mechanism_robustness.py",
        "paper_b/EXPERT_RANK_ROBUSTNESS_PLAN.md",
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


def _nested_degree_check(
    *,
    seed: int,
    n_citizens: int,
    groups: dict[int, int],
    low_homophily: float,
    high_homophily: float,
) -> bool:
    d2 = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n_citizens,
        group_ids=groups,
        peer_degree=2,
        low_homophily=low_homophily,
        high_homophily=high_homophily,
    )
    d4 = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n_citizens,
        group_ids=groups,
        peer_degree=4,
        low_homophily=low_homophily,
        high_homophily=high_homophily,
    )
    for h in H_LEVELS:
        for ego in d2[h]:
            peers2 = [x for x in d2[h][ego] if int(x) >= 2]
            peers4 = [x for x in d4[h][ego] if int(x) >= 2]
            if peers4[:2] != peers2:
                return False
    return True


def _run_spec_seed(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2

    seed = int(task["seed"])
    n = int(task["n_citizens"])
    spec = dict(task["spec"])

    groups = balanced_fixed_group_ids(seed=seed, n_citizens=n)
    blueprint = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n,
        group_ids=groups,
        peer_degree=int(spec["peer_degree"]),
        low_homophily=float(task["low_homophily"]),
        high_homophily=float(task["high_homophily"]),
    )
    initial = {
        s: exp4_initial_beliefs(
            seed=seed,
            n_citizens=n,
            group_ids=groups,
            segregation=s,
            high_group_shift=float(task["high_group_shift"]),
            residual_sd=float(task["prior_residual_sd"]),
        )
        for s in S_LEVELS
    }

    results = []
    for h in H_LEVELS:
        for s in S_LEVELS:
            for reliance in RELIANCE_MODES:
                base_task = {
                    "n_citizens": n,
                    "horizon_T": int(task["horizon"]),
                    "K": int(task["k"]),
                    "epsilon": float(spec["epsilon"]),
                    "credit": int(task["credit"]),
                    "surveillance_interval": int(task["surveillance_interval"]),
                }
                cfg = _base_config(seed, base_task)
                cfg.update(
                    {
                        "mu_theta": initial[s],
                        "initial_theta_type": f"robustness_exp4_{s}",
                        "fixed_group_ids": groups,
                        "structural_source_map": blueprint[h],
                    }
                )

                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime=f"segregation_{s}",
                    environment=(
                        f"mechanism_exp4_{spec['spec_label']}_h_{h}"
                    ),
                    reliance_mode=reliance,
                    jammer_active=False,
                    jammer_regime="null",
                    peer_evidence_mode="source_posterior",
                    frozen_ranking_mode="pre_disruption",
                    k=int(task["k"]),
                    design_id=str(task["design_id"]),
                    block_id=f"{spec['spec_label']}__s{seed}",
                    save_edge_log=False,
                    record_expert_rank_checkpoints=True,
                )

                labels = {
                    "experiment": "IV",
                    "robustness_block": spec["robustness_block"],
                    "spec_label": spec["spec_label"],
                    "epsilon": float(spec["epsilon"]),
                    "peer_degree": int(spec["peer_degree"]),
                    "homophily_level": h,
                    "segregation_level": s,
                    "reliance_mode": reliance,
                    "sender_regime": "null",
                }
                result["run"].update(labels)
                for key in (
                    "beliefs",
                    "lambda_checkpoints",
                    "belief_checkpoints",
                    "evidence_checkpoints",
                    "expert_rank_checkpoints",
                ):
                    for row in result[key]:
                        row.update(labels)
                results.append(result)

    payload = {
        "complete": True,
        "design_id": task["design_id"],
        "seed": seed,
        "spec": spec,
        "runs": [r["run"] for r in results],
        "lambda_checkpoints": [
            x for r in results for x in r["lambda_checkpoints"]
        ],
        "belief_checkpoints": [
            x for r in results for x in r["belief_checkpoints"]
        ],
        "evidence_checkpoints": [
            x for r in results for x in r["evidence_checkpoints"]
        ],
        "expert_rank_checkpoints": [
            x for r in results for x in r["expert_rank_checkpoints"]
        ],
    }
    path = (
        Path(task["run_root"])
        / "shards"
        / f"{spec['spec_label']}__s{seed}.json.gz"
    )
    _atomic_write(path, payload)
    return {
        "seed": seed,
        "spec_label": spec["spec_label"],
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
            and len(payload.get("runs", [])) == 8
        )
    except Exception:
        return False


def _mean(values) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.fmean(vals)) if vals else math.nan


def _cell_summary(rows: list[dict]) -> list[dict]:
    keys = (
        "robustness_block",
        "spec_label",
        "epsilon",
        "peer_degree",
        "homophily_level",
        "segregation_level",
        "reliance_mode",
    )
    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "belief_variance",
        "squared_displacement",
        "effective_homophily",
        "dominant_same_group_cycle_share",
        "W_effective_homophily",
        "W_dominant_same_group_cycle_share",
        "W_dominant_expert_reach_share",
        "expert_reliance",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)

    out = []
    for key, members in sorted(grouped.items()):
        item = {name: value for name, value in zip(keys, key)}
        item["n"] = len(members)
        for metric in metrics:
            values = []
            for member in members:
                try:
                    values.append(float(member.get(metric, math.nan)))
                except (TypeError, ValueError):
                    pass
            item[f"{metric}_mean"] = _mean(values)
        out.append(item)
    return out


def _rank_checkpoint_summary(rows: list[dict]) -> list[dict]:
    keys = (
        "robustness_block",
        "spec_label",
        "epsilon",
        "peer_degree",
        "homophily_level",
        "segregation_level",
        "reliance_mode",
        "period",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    out = []
    for key, members in sorted(grouped.items()):
        item = {name: value for name, value in zip(keys, key)}
        item["n"] = len(members)
        metric_names = set()
        for member in members:
            metric_names.update(
                name
                for name in member
                if name.startswith("expert_rank_")
                or name in {
                    "mean_expert_rank",
                    "mean_expert_acquisition_probability",
                }
            )
        for metric in sorted(metric_names):
            values = []
            for member in members:
                try:
                    values.append(float(member.get(metric, math.nan)))
                except (TypeError, ValueError):
                    pass
            item[f"{metric}_mean"] = _mean(values)
        out.append(item)
    return out


def _seed_contrasts(rows: list[dict]) -> list[dict]:
    index = {
        (
            row["spec_label"],
            int(row["seed"]),
            row["homophily_level"],
            row["segregation_level"],
            row["reliance_mode"],
        ): row
        for row in rows
    }
    specs = sorted(set(row["spec_label"] for row in rows))
    seeds = sorted(set(int(row["seed"]) for row in rows))
    out = []

    for spec in specs:
        for seed in seeds:
            for reliance in RELIANCE_MODES:
                def get(h, s):
                    return index[(spec, seed, h, s, reliance)]

                hh = float(get("high", "high")["mse_truth"])
                lh = float(get("low", "high")["mse_truth"])
                hl = float(get("high", "low")["mse_truth"])
                ll = float(get("low", "low")["mse_truth"])
                out.append(
                    {
                        "spec_label": spec,
                        "seed": seed,
                        "reliance_mode": reliance,
                        "contrast": "H_x_S_MSE",
                        "value": (hh - lh) - (hl - ll),
                    }
                )
    return out


def _contrast_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["spec_label"],
                row["reliance_mode"],
                row["contrast"],
            )
        ].append(float(row["value"]))
    out = []
    for (spec, reliance, contrast), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        mcse = float(arr.std(ddof=1) / math.sqrt(arr.size))
        out.append(
            {
                "spec_label": spec,
                "reliance_mode": reliance,
                "contrast": contrast,
                "n_seeds": int(arr.size),
                "mean": mean,
                "mcse": mcse,
                "mc95_low": mean - 1.96 * mcse,
                "mc95_high": mean + 1.96 * mcse,
                "median": float(np.median(arr)),
                "positive_share": float(np.mean(arr > 0.0)),
                "negative_share": float(np.mean(arr < 0.0)),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    blocks = _resolve_blocks(args.blocks)
    specs = _specs(blocks)
    seeds = list(
        range(int(args.seed_start), int(args.seed_start) + int(args.seeds))
    )

    canonical_match = bool(
        int(args.seeds) == 500
        and int(args.seed_start) == 6001
        and int(args.n_citizens) == CANONICAL_N_CITIZENS
        and int(args.horizon) == CANONICAL_HORIZON
        and int(args.credit) == CANONICAL_CREDIT
        and int(args.k) == CANONICAL_K
        and int(args.surveillance_interval)
        == CANONICAL_SURVEILLANCE_INTERVAL
        and math.isclose(float(args.numerical_min_sd), 1e-8)
        and math.isclose(float(args.low_homophily), 0.50)
        and math.isclose(float(args.high_homophily), 0.90)
        and math.isclose(float(args.high_group_shift), 3.0)
        and math.isclose(float(args.prior_residual_sd), 1.0)
    )

    # Verify the d=4 design is a nested extension of d=2 before any production.
    nested_pass = True
    if "degree" in blocks:
        for seed in seeds[: min(10, len(seeds))]:
            groups = balanced_fixed_group_ids(
                seed=seed,
                n_citizens=int(args.n_citizens),
            )
            if not _nested_degree_check(
                seed=seed,
                n_citizens=int(args.n_citizens),
                groups=groups,
                low_homophily=float(args.low_homophily),
                high_homophily=float(args.high_homophily),
            ):
                nested_pass = False
                break
        if not nested_pass:
            raise RuntimeError(
                "d=4 source maps do not preserve the first two d=2 peers."
            )

    design = {
        "purpose": "Paper B Expert-rank mechanism robustness",
        "reference_design_id": REFERENCE_DESIGN_ID,
        "seeds": seeds,
        "specs": specs,
        "n_citizens": int(args.n_citizens),
        "horizon_T": int(args.horizon),
        "credit": int(args.credit),
        "K": int(args.k),
        "surveillance_interval": int(args.surveillance_interval),
        "low_homophily": float(args.low_homophily),
        "high_homophily": float(args.high_homophily),
        "high_group_shift": float(args.high_group_shift),
        "prior_residual_sd": float(args.prior_residual_sd),
        "numerical_min_sd": float(args.numerical_min_sd),
        "peer_evidence_mode": "source_posterior",
        "tau_social": 1.0,
        "frozen_ranking_mode": "pre_disruption",
        "sender_regime": "null",
        "canonical_base_match": canonical_match,
        "degree_nested_extension_pass": nested_pass,
        "software_versions": software_versions(),
        "scientific_code_fingerprint": _scientific_fingerprint(),
    }
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id

    root = Path(args.output_dir) / f"mechanism_{design_id}"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / "mechanism_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    for spec in specs:
        for seed in seeds:
            path = root / "shards" / f"{spec['spec_label']}__s{seed}.json.gz"
            if args.resume and _valid_shard(path, design_id):
                continue
            tasks.append(
                {
                    "design_id": design_id,
                    "run_root": str(root),
                    "seed": int(seed),
                    "spec": spec,
                    "n_citizens": int(args.n_citizens),
                    "horizon": int(args.horizon),
                    "credit": int(args.credit),
                    "k": int(args.k),
                    "surveillance_interval": int(args.surveillance_interval),
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
            futures = [pool.submit(_run_spec_seed, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if i % max(int(args.progress_every), 1) == 0:
                    print(
                        f"[{i}/{len(futures)}] "
                        f"{result['spec_label']} seed={result['seed']}",
                        flush=True,
                    )

    payloads = []
    for path in sorted((root / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("design_id") == design_id:
            payloads.append(payload)

    expected_blocks = len(specs) * len(seeds)
    expected_runs = expected_blocks * 8
    if len(payloads) != expected_blocks:
        raise RuntimeError(
            f"Expected {expected_blocks} shards, got {len(payloads)}."
        )

    runs = [x for p in payloads for x in p["runs"]]
    lambda_cp = [x for p in payloads for x in p["lambda_checkpoints"]]
    belief_cp = [x for p in payloads for x in p["belief_checkpoints"]]
    evidence_cp = [x for p in payloads for x in p["evidence_checkpoints"]]
    rank_cp = [x for p in payloads for x in p["expert_rank_checkpoints"]]

    if len(runs) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs, got {len(runs)}.")

    finite = all(math.isfinite(float(row["mse_truth"])) for row in runs)
    fixed_horizon = all(
        int(row["terminal_horizon_T"]) == int(args.horizon)
        for row in runs
    )
    null_last = min(
        float(row["null_last_share"]) for row in runs
    )

    gate = {
        "pass": bool(
            canonical_match
            and nested_pass
            and finite
            and fixed_horizon
            and null_last == 1.0
        ),
        "design_id": design_id,
        "reference_design_id": REFERENCE_DESIGN_ID,
        "canonical_base_match": canonical_match,
        "degree_nested_extension_pass": nested_pass,
        "observed_blocks": len(payloads),
        "expected_blocks": expected_blocks,
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "all_terminal_mse_finite": finite,
        "all_fixed_horizon": fixed_horizon,
        "null_last_share_minimum": null_last,
    }
    (root / "mechanism_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cell_summary = _cell_summary(runs)
    rank_summary = _rank_checkpoint_summary(rank_cp)
    seed_contrasts = _seed_contrasts(runs)
    contrast_summary = _contrast_summary(seed_contrasts)

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "cell_summary.csv", cell_summary)
    _write_csv(root / "expert_rank_checkpoint_summary.csv", rank_summary)
    _write_csv(root / "contrast_summary.csv", contrast_summary)
    _write_csv(root / "seed_contrasts.csv.gz", seed_contrasts, gzip_output=True)
    _write_csv(
        root / "lambda_checkpoints.csv.gz",
        lambda_cp,
        gzip_output=True,
    )
    _write_csv(
        root / "belief_checkpoints.csv.gz",
        belief_cp,
        gzip_output=True,
    )
    _write_csv(
        root / "evidence_checkpoints.csv.gz",
        evidence_cp,
        gzip_output=True,
    )

    review = root / f"paper_b_mechanism_review_{design_id}.zip"
    plan = Path(__file__).resolve().parents[1] / "EXPERT_RANK_ROBUSTNESS_PLAN.md"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "mechanism_manifest.json",
            "mechanism_gate.json",
            "runs.csv",
            "cell_summary.csv",
            "expert_rank_checkpoint_summary.csv",
            "contrast_summary.csv",
            "lambda_checkpoints.csv.gz",
            "belief_checkpoints.csv.gz",
            "evidence_checkpoints.csv.gz",
        ):
            zf.write(root / name, arcname=name)
        zf.write(plan, arcname="EXPERT_RANK_ROBUSTNESS_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"Review bundle: {review}")


if __name__ == "__main__":
    main()
