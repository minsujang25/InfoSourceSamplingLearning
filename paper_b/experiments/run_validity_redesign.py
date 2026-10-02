"""Paper B validity-redesign diagnostic runner.

Primary diagnostic:
    24 matched seeds
    Exp IV: 2 H x 2 S x 2 reliance x 3 sender regimes = 24 runs/seed
    Exp III: 2 redundancy x 2 reliance x 3 sender regimes = 12 runs/seed

Total primary runs = 36 per seed = 864 at 24 seeds.

A six-seed truth-clone compatibility block can be included without changing
the primary redesigned contrasts.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import model.InfoSourceSamplingLearning as model_module
from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.experiments.run_matched_pilot import (
    canonical_hash,
    run_condition,
    software_versions,
)
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp3_redundancy_source_maps,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


PRIMARY_SENDERS = ("null", "adaptive", "fixed_biased")
RELIANCE_MODES = ("adaptive", "frozen")
HOMOPHILY_LEVELS = ("low", "high")
SEGREGATION_LEVELS = ("low", "high")
REDUNDANCY_LEVELS = ("low", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="24")
    parser.add_argument("--seed-start", type=int, default=5001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
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
    parser.add_argument("--truth-clone-seeds", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_validity_redesign",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def resolve_seeds(value: str, seed_start: int) -> list[int]:
    if "," in value:
        seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        n = int(value)
        if n <= 0:
            raise ValueError("--seeds must be positive.")
        seeds = list(range(seed_start, seed_start + n))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique.")
    return seeds


def scientific_code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/metrics.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_local_diagnostic.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_validity_redesign.py",
        "paper_b/VALIDITY_REDESIGN_PLAN.md",
        "environment.yml",
    )
    payload = []
    for name in names:
        path = root / name
        payload.append(
            {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return canonical_hash(payload)


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
    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    opener = gzip.open if gzip_output else open
    kwargs = {"mode": "wt", "newline": "", "encoding": "utf-8"} if gzip_output else {
        "mode": "w", "newline": "", "encoding": "utf-8"
    }
    with opener(path, **kwargs) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _run_exp4(seed: int, task: dict, sender_regimes: tuple[str, ...]) -> list[dict]:
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
    initial = {
        segregation: exp4_initial_beliefs(
            seed=seed,
            n_citizens=n,
            group_ids=groups,
            segregation=segregation,
            high_group_shift=float(task["high_group_shift"]),
            residual_sd=float(task["prior_residual_sd"]),
        )
        for segregation in SEGREGATION_LEVELS
    }

    results = []
    for homophily in HOMOPHILY_LEVELS:
        for segregation in SEGREGATION_LEVELS:
            base = base_model_config(
                regime="flat",
                seed=seed,
                n_citizens=n,
                max_steps=int(task["horizon_T"]),
                k=int(task["k"]),
                epsilon=float(task["epsilon"]),
                credit=int(task["credit"]),
                comparison_rule="delta_comparison",
                surveillance_interval=int(task["surveillance_interval"]),
            )
            base.update(
                {
                    "mu_theta": initial[segregation],
                    "initial_theta_type": f"validity_exp4_{segregation}",
                    "fixed_group_ids": groups,
                    "structural_source_map": blueprint[homophily],
                }
            )
            for reliance in RELIANCE_MODES:
                for sender in sender_regimes:
                    result = run_condition(
                        base_config=base,
                        seed=seed,
                        regime=f"segregation_{segregation}",
                        environment=f"validity_exp4_h_{homophily}",
                        reliance_mode=reliance,
                        jammer_active=(sender == "adaptive"),
                        jammer_regime=sender,
                        peer_evidence_mode="source_posterior",
                        frozen_ranking_mode="pre_disruption",
                        k=int(task["k"]),
                        design_id=str(task["design_id"]),
                        block_id=f"exp4__s{seed}",
                        save_edge_log=False,
                    )
                    result["run"].update(
                        {
                            "experiment": "IV",
                            "homophily_level": homophily,
                            "segregation_level": segregation,
                            "sender_regime": sender,
                        }
                    )
                    for key in (
                        "beliefs",
                        "lambda_checkpoints",
                        "belief_checkpoints",
                        "jammer_strategy",
                    ):
                        for row in result[key]:
                            row.update(
                                {
                                    "experiment": "IV",
                                    "homophily_level": homophily,
                                    "segregation_level": segregation,
                                    "sender_regime": sender,
                                }
                            )
                    results.append(result)
    return results


def _run_exp3(seed: int, task: dict, sender_regimes: tuple[str, ...]) -> list[dict]:
    n = int(task["n_citizens"])
    blueprint = exp3_redundancy_source_maps(
        seed=seed,
        n_citizens=n,
        peer_degree=int(task["peer_degree"]),
        expert_access_share=float(task["expert_access_share"]),
    )
    gateways = set(int(x) for x in blueprint["expert_gateways"])
    base = base_model_config(
        regime="flat",
        seed=seed,
        n_citizens=n,
        max_steps=int(task["horizon_T"]),
        k=int(task["k"]),
        epsilon=float(task["epsilon"]),
        credit=int(task["credit"]),
        comparison_rule="delta_comparison",
        surveillance_interval=int(task["surveillance_interval"]),
    )

    results = []
    for redundancy in REDUNDANCY_LEVELS:
        for reliance in RELIANCE_MODES:
            for sender in sender_regimes:
                cfg = dict(base)
                cfg["structural_source_map"] = blueprint[redundancy]
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime="flat",
                    environment=f"validity_exp3_r_{redundancy}",
                    reliance_mode=reliance,
                    jammer_active=(sender == "adaptive"),
                    jammer_regime=sender,
                    peer_evidence_mode="source_posterior",
                    frozen_ranking_mode="pre_disruption",
                    gateway_positions=gateways,
                    k=int(task["k"]),
                    design_id=str(task["design_id"]),
                    block_id=f"exp3__s{seed}",
                    save_edge_log=False,
                )
                result["run"].update(
                    {
                        "experiment": "III",
                        "redundancy_level": redundancy,
                        "sender_regime": sender,
                    }
                )
                for key in (
                    "beliefs",
                    "lambda_checkpoints",
                    "belief_checkpoints",
                    "jammer_strategy",
                ):
                    for row in result[key]:
                        row.update(
                            {
                                "experiment": "III",
                                "redundancy_level": redundancy,
                                "sender_regime": sender,
                            }
                        )
                results.append(result)
    return results


def _run_block(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2
    experiment = str(task["experiment"])
    seed = int(task["seed"])
    senders = tuple(task["sender_regimes"])
    results = (
        _run_exp4(seed, task, senders)
        if experiment == "IV"
        else _run_exp3(seed, task, senders)
    )
    payload = {
        "complete": True,
        "design_id": task["design_id"],
        "experiment": experiment,
        "seed": seed,
        "runs": [r["run"] for r in results],
        "terminal_beliefs": [x for r in results for x in r["beliefs"]],
        "lambda_checkpoints": [
            x for r in results for x in r["lambda_checkpoints"]
        ],
        "belief_checkpoints": [
            x for r in results for x in r["belief_checkpoints"]
        ],
        "jammer_strategy": [x for r in results for x in r["jammer_strategy"]],
    }
    path = Path(task["run_root"]) / "shards" / f"{experiment}__s{seed}.json.gz"
    _atomic_write(path, payload)
    return {
        "experiment": experiment,
        "seed": seed,
        "path": str(path),
        "runs": len(payload["runs"]),
    }


def _valid_shard(path: Path, design_id: str) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        return value.get("complete") is True and value.get("design_id") == design_id
    except Exception:
        return False


def _matched_movement(beliefs: list[dict]) -> list[dict]:
    index = {}
    for row in beliefs:
        key = (
            row["experiment"],
            int(row["seed"]),
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            row["reliance_mode"],
            int(row["citizen_pos"]),
            row["sender_regime"],
        )
        index[key] = row

    out = []
    for key, row in index.items():
        if key[-1] == "null":
            continue
        base_key = (*key[:-1], "null")
        null = index.get(base_key)
        if null is None:
            continue
        sender_mu = float(row["terminal_mu_theta"])
        null_mu = float(null["terminal_mu_theta"])
        truth = 0.0
        out.append(
            {
                "experiment": key[0],
                "seed": key[1],
                "homophily_level": key[2],
                "segregation_level": key[3],
                "redundancy_level": key[4],
                "reliance_mode": key[5],
                "citizen_pos": key[6],
                "citizen_group": row["citizen_group"],
                "sender_regime": key[7],
                "signed_movement_vs_null": sender_mu - null_mu,
                "absolute_movement_vs_null": abs(sender_mu - null_mu),
                "change_absolute_truth_error": (
                    abs(sender_mu - truth) - abs(null_mu - truth)
                ),
                "change_squared_truth_error": (
                    (sender_mu - truth) ** 2 - (null_mu - truth) ** 2
                ),
            }
        )
    return out


def _run_contrasts(runs: list[dict]) -> list[dict]:
    index = {}
    for row in runs:
        key = (
            row["experiment"],
            int(row["seed"]),
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            row["reliance_mode"],
            row["sender_regime"],
        )
        index[key] = row

    out = []
    for key, row in index.items():
        if key[-1] == "null":
            continue
        null = index.get((*key[:-1], "null"))
        if null is None:
            continue
        out.append(
            {
                "experiment": key[0],
                "seed": key[1],
                "homophily_level": key[2],
                "segregation_level": key[3],
                "redundancy_level": key[4],
                "reliance_mode": key[5],
                "sender_regime": key[6],
                "delta_mse_vs_null": float(row["mse_truth"]) - float(null["mse_truth"]),
                "delta_rmse_vs_null": float(row["rmse_truth"]) - float(null["rmse_truth"]),
                "delta_mae_vs_null": float(row["mae_truth"]) - float(null["mae_truth"]),
                "delta_belief_variance_vs_null": (
                    float(row["belief_variance"]) - float(null["belief_variance"])
                ),
                "delta_squared_displacement_vs_null": (
                    float(row["squared_displacement"])
                    - float(null["squared_displacement"])
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if not 0.0 < args.numerical_min_sd < 1.0:
        raise ValueError("--numerical-min-sd must lie in (0,1).")

    design = {
        "design_version": 1,
        "purpose": "Paper B receiver-side validity redesign diagnostic",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "software_versions": software_versions(),
        "seeds": seeds,
        "primary_sender_regimes": list(PRIMARY_SENDERS),
        "truth_clone_compatibility_seeds": min(
            max(int(args.truth_clone_seeds), 0), len(seeds)
        ),
        "peer_evidence_mode": "source_posterior",
        "frozen_ranking_mode": "pre_disruption",
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
    }
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id
    run_root = Path(args.output_dir) / f"validity_{design_id}"
    shards = run_root / "shards"

    if run_root.exists() and not args.resume:
        raise FileExistsError(
            f"{run_root} exists; use --resume or choose a different output directory."
        )
    shards.mkdir(parents=True, exist_ok=True)
    (run_root / "validity_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    compat_n = design["truth_clone_compatibility_seeds"]
    compat = set(seeds[:compat_n])
    for experiment in ("III", "IV"):
        for seed in seeds:
            senders = list(PRIMARY_SENDERS)
            if seed in compat:
                senders.append("truth_clone")
            task = {
                **design,
                "experiment": experiment,
                "seed": seed,
                "sender_regimes": senders,
                "run_root": str(run_root),
            }
            path = shards / f"{experiment}__s{seed}.json.gz"
            if args.resume and _valid_shard(path, design_id):
                continue
            tasks.append(task)

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
                        f"[{i}/{len(futures)}] "
                        f"{result['experiment']} seed={result['seed']} "
                        f"runs={result['runs']}",
                        flush=True,
                    )

    payloads = []
    expected = {(experiment, seed) for experiment in ("III", "IV") for seed in seeds}
    observed = set()
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("design_id") != design_id:
            continue
        payloads.append(value)
        observed.add((value["experiment"], int(value["seed"])))
    if observed != expected:
        missing = sorted(expected - observed)
        raise RuntimeError(f"Missing completed validity blocks: {missing[:10]}")

    runs = [x for p in payloads for x in p["runs"]]
    beliefs = [x for p in payloads for x in p["terminal_beliefs"]]
    lambdas = [x for p in payloads for x in p["lambda_checkpoints"]]
    belief_checkpoints = [x for p in payloads for x in p["belief_checkpoints"]]
    jammer = [x for p in payloads for x in p["jammer_strategy"]]
    movements = _matched_movement(beliefs)
    contrasts = _run_contrasts(runs)

    expected_primary = len(seeds) * (24 + 12)
    expected_compat = compat_n * (8 + 4)
    expected_runs = expected_primary + expected_compat
    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Run-count mismatch: expected {expected_runs}, observed {len(runs)}."
        )
    if any(not math.isfinite(float(row["mse_truth"])) for row in runs):
        raise RuntimeError("Non-finite run-level MSE in validity diagnostic.")

    gate = {
        "pass": True,
        "design_id": design_id,
        "observed_blocks": len(observed),
        "expected_blocks": len(expected),
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "peer_evidence_mode": "source_posterior",
        "frozen_ranking_mode": "pre_disruption",
        "sender_regimes_present": sorted({r["sender_regime"] for r in runs}),
        "all_run_mse_finite": True,
    }
    (run_root / "validity_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _write_csv(run_root / "runs.csv", runs)
    _write_csv(run_root / "terminal_beliefs.csv.gz", beliefs, gzip_output=True)
    _write_csv(run_root / "lambda_checkpoints.csv", lambdas)
    _write_csv(run_root / "belief_checkpoints.csv", belief_checkpoints)
    _write_csv(run_root / "jammer_strategy.csv.gz", jammer, gzip_output=True)
    _write_csv(run_root / "matched_movement.csv.gz", movements, gzip_output=True)
    _write_csv(run_root / "sender_vs_null_contrasts.csv", contrasts)

    bundle = run_root / f"paper_b_validity_{design_id}_shareable.zip"
    files = (
        "validity_manifest.json",
        "validity_gate.json",
        "runs.csv",
        "terminal_beliefs.csv.gz",
        "lambda_checkpoints.csv",
        "belief_checkpoints.csv",
        "jammer_strategy.csv.gz",
        "matched_movement.csv.gz",
        "sender_vs_null_contrasts.csv",
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = run_root / name
            if path.exists():
                archive.write(path, arcname=name)
        plan = Path(__file__).resolve().parents[1] / "VALIDITY_REDESIGN_PLAN.md"
        archive.write(plan, arcname="VALIDITY_REDESIGN_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
