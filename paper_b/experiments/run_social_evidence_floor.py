"""Narrow social-evidence-floor diagnostic for Paper B validity redesign.

Frozen design:
    seeds 5101-5108
    tau_social in {0, 1}
    adaptive reliance only
    source_posterior peer evidence
    pre_disruption ranking initialization

Per seed/tau:
    Exp IV baseline network: 2 H x 2 S x null sender = 4
    Exp IV sender stress:    2 H x high S x {adaptive,fixed_biased} = 4
    Exp III:                 2 R x {null,adaptive,fixed_biased} = 6
    total = 14

Eight seeds x two tau values x 14 = 224 simulations.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


TAU_VALUES = (0.0, 1.0)
HOMOPHILY_LEVELS = ("low", "high")
SEGREGATION_LEVELS = ("low", "high")
REDUNDANCY_LEVELS = ("low", "high")
STRESS_SENDERS = ("adaptive", "fixed_biased")
EXP3_SENDERS = ("null", "adaptive", "fixed_biased")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="8")
    parser.add_argument("--seed-start", type=int, default=5101)
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
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_social_evidence_floor",
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
        "paper_b/experiments/run_social_evidence_floor.py",
        "paper_b/SOCIAL_EVIDENCE_FLOOR_PLAN.md",
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
    if gzip_output:
        handle = gzip.open(path, "wt", newline="", encoding="utf-8")
    else:
        handle = open(path, "w", newline="", encoding="utf-8")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _base_config(seed: int, task: dict) -> dict:
    return base_model_config(
        regime="flat",
        seed=seed,
        n_citizens=int(task["n_citizens"]),
        max_steps=int(task["horizon_T"]),
        k=int(task["K"]),
        epsilon=float(task["epsilon"]),
        credit=int(task["credit"]),
        comparison_rule="delta_comparison",
        surveillance_interval=int(task["surveillance_interval"]),
    )


def _decorate(result: dict, **labels) -> None:
    result["run"].update(labels)
    for key in (
        "beliefs",
        "lambda_checkpoints",
        "belief_checkpoints",
        "jammer_strategy",
    ):
        for row in result[key]:
            row.update(labels)


def _run_exp4(seed: int, tau: float, task: dict) -> list[dict]:
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
    # Baseline-network block: 2H x 2S x null.
    for homophily in HOMOPHILY_LEVELS:
        for segregation in SEGREGATION_LEVELS:
            cfg = _base_config(seed, task)
            cfg.update(
                {
                    "mu_theta": initial[segregation],
                    "initial_theta_type": f"tau_exp4_{segregation}",
                    "fixed_group_ids": groups,
                    "structural_source_map": blueprint[homophily],
                    "tau_social": float(tau),
                }
            )
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime=f"segregation_{segregation}",
                environment=f"tau_exp4_h_{homophily}",
                reliance_mode="adaptive",
                jammer_active=False,
                jammer_regime="null",
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"exp4_null__s{seed}__tau{tau:g}",
                save_edge_log=False,
            )
            _decorate(
                result,
                experiment="IV",
                diagnostic_block="baseline_network",
                homophily_level=homophily,
                segregation_level=segregation,
                redundancy_level="",
                sender_regime="null",
                tau_social=float(tau),
            )
            results.append(result)

    # Sender-stress block: 2H x high-S x two sender regimes.
    for homophily in HOMOPHILY_LEVELS:
        segregation = "high"
        for sender in STRESS_SENDERS:
            cfg = _base_config(seed, task)
            cfg.update(
                {
                    "mu_theta": initial[segregation],
                    "initial_theta_type": "tau_exp4_high",
                    "fixed_group_ids": groups,
                    "structural_source_map": blueprint[homophily],
                    "tau_social": float(tau),
                }
            )
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime="segregation_high",
                environment=f"tau_exp4_h_{homophily}",
                reliance_mode="adaptive",
                jammer_active=(sender == "adaptive"),
                jammer_regime=sender,
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"exp4_stress__s{seed}__tau{tau:g}",
                save_edge_log=False,
            )
            _decorate(
                result,
                experiment="IV",
                diagnostic_block="sender_stress",
                homophily_level=homophily,
                segregation_level="high",
                redundancy_level="",
                sender_regime=sender,
                tau_social=float(tau),
            )
            results.append(result)
    return results


def _run_exp3(seed: int, tau: float, task: dict) -> list[dict]:
    n = int(task["n_citizens"])
    blueprint = exp3_redundancy_source_maps(
        seed=seed,
        n_citizens=n,
        peer_degree=int(task["peer_degree"]),
        expert_access_share=float(task["expert_access_share"]),
    )
    gateways = set(int(x) for x in blueprint["expert_gateways"])
    results = []

    for redundancy in REDUNDANCY_LEVELS:
        for sender in EXP3_SENDERS:
            cfg = _base_config(seed, task)
            cfg.update(
                {
                    "structural_source_map": blueprint[redundancy],
                    "tau_social": float(tau),
                }
            )
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime="flat",
                environment=f"tau_exp3_r_{redundancy}",
                reliance_mode="adaptive",
                jammer_active=(sender == "adaptive"),
                jammer_regime=sender,
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                gateway_positions=gateways,
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"exp3__s{seed}__tau{tau:g}",
                save_edge_log=False,
            )
            _decorate(
                result,
                experiment="III",
                diagnostic_block="redundancy_sender",
                homophily_level="",
                segregation_level="",
                redundancy_level=redundancy,
                sender_regime=sender,
                tau_social=float(tau),
            )
            results.append(result)
    return results


def _run_block(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2
    seed = int(task["seed"])
    tau = float(task["tau_social"])
    results = _run_exp4(seed, tau, task) + _run_exp3(seed, tau, task)

    payload = {
        "complete": True,
        "design_id": task["design_id"],
        "seed": seed,
        "tau_social": tau,
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
    path = (
        Path(task["run_root"])
        / "shards"
        / f"s{seed}__tau{tau:g}.json.gz"
    )
    _atomic_write(path, payload)
    return {"seed": seed, "tau_social": tau, "runs": len(payload["runs"])}


def _valid_shard(path: Path, design_id: str) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        return (
            value.get("complete") is True
            and value.get("design_id") == design_id
            and len(value.get("runs", [])) == 14
        )
    except Exception:
        return False


def _cell_key(row: dict, include_tau: bool) -> tuple:
    key = (
        row["experiment"],
        int(row["seed"]),
        row["diagnostic_block"],
        row.get("homophily_level", ""),
        row.get("segregation_level", ""),
        row.get("redundancy_level", ""),
        row["sender_regime"],
    )
    if include_tau:
        return (*key, float(row["tau_social"]))
    return key


def tau_contrasts(runs: list[dict]) -> list[dict]:
    index = {_cell_key(row, True): row for row in runs}
    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "belief_variance",
        "squared_displacement",
        "dominant_expert_reach_share",
        "dominant_jammer_reach_share",
        "dominant_citizen_cycle_share",
        "dominant_same_group_cycle_share",
        "effective_incoming_hhi",
        "effective_incoming_top5_share",
        "gateway_incoming_reliance_share",
        "effective_homophily",
        "peer_reliance_mass",
    )
    out = []
    bases = sorted({_cell_key(row, False) for row in runs})
    for base in bases:
        zero = index.get((*base, 0.0))
        one = index.get((*base, 1.0))
        if zero is None or one is None:
            continue
        row = {
            "experiment": base[0],
            "seed": base[1],
            "diagnostic_block": base[2],
            "homophily_level": base[3],
            "segregation_level": base[4],
            "redundancy_level": base[5],
            "sender_regime": base[6],
        }
        for metric in metrics:
            left = float(zero.get(metric, math.nan))
            right = float(one.get(metric, math.nan))
            row[f"tau1_minus_tau0_{metric}"] = (
                right - left
                if math.isfinite(left) and math.isfinite(right)
                else math.nan
            )
        out.append(row)
    return out


def sender_vs_null_contrasts(runs: list[dict]) -> list[dict]:
    index = {_cell_key(row, True): row for row in runs}
    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "belief_variance",
        "squared_displacement",
        "dominant_expert_reach_share",
        "dominant_citizen_cycle_share",
        "dominant_same_group_cycle_share",
        "effective_incoming_hhi",
        "gateway_incoming_reliance_share",
    )
    out = []
    for row in runs:
        if row["sender_regime"] == "null":
            continue
        null_key = (
            row["experiment"],
            int(row["seed"]),
            # sender-stress IV has a matching null in baseline-network.
            "baseline_network" if row["experiment"] == "IV" else row["diagnostic_block"],
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            "null",
            float(row["tau_social"]),
        )
        null = index.get(null_key)
        if null is None:
            continue
        contrast = {
            "experiment": row["experiment"],
            "seed": int(row["seed"]),
            "homophily_level": row.get("homophily_level", ""),
            "segregation_level": row.get("segregation_level", ""),
            "redundancy_level": row.get("redundancy_level", ""),
            "sender_regime": row["sender_regime"],
            "tau_social": float(row["tau_social"]),
        }
        for metric in metrics:
            active = float(row.get(metric, math.nan))
            base = float(null.get(metric, math.nan))
            contrast[f"sender_minus_null_{metric}"] = (
                active - base
                if math.isfinite(active) and math.isfinite(base)
                else math.nan
            )
        out.append(contrast)
    return out


def _median(values: list[float]) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    return float(statistics.median(finite)) if finite else math.nan


def _floor_screen(lambda_rows: list[dict], period: int) -> float:
    values = [
        float(row["posterior_sd_floor_share"])
        for row in lambda_rows
        if float(row["tau_social"]) == 1.0
        and row["sender_regime"] == "null"
        and int(row["period"]) == int(period)
    ]
    return _median(values)


def _mean_cell_difference(
    runs: list[dict],
    *,
    experiment: str,
    tau: float,
    sender: str,
    metric: str,
    high_filter,
    low_filter,
) -> float:
    high = {}
    low = {}
    for row in runs:
        if (
            row["experiment"] != experiment
            or float(row["tau_social"]) != float(tau)
            or row["sender_regime"] != sender
        ):
            continue
        seed = int(row["seed"])
        if high_filter(row):
            high[seed] = float(row[metric])
        if low_filter(row):
            low[seed] = float(row[metric])
    matched = sorted(set(high) & set(low))
    values = [high[s] - low[s] for s in matched]
    return float(statistics.fmean(values)) if values else math.nan


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if not 0.0 < args.numerical_min_sd < 1.0:
        raise ValueError("--numerical-min-sd must lie in (0,1).")

    plan_path = Path(__file__).resolve().parents[1] / "SOCIAL_EVIDENCE_FLOOR_PLAN.md"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    design = {
        "design_version": 1,
        "purpose": "Paper B social-evidence floor validity diagnostic",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "plan_sha256": plan_sha,
        "software_versions": software_versions(),
        "seeds": seeds,
        "tau_social_values": list(TAU_VALUES),
        "peer_evidence_mode": "source_posterior",
        "frozen_ranking_mode": "pre_disruption",
        "reliance_mode": "adaptive",
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

    run_root = Path(args.output_dir) / f"social_floor_{design_id}"
    shards = run_root / "shards"
    if run_root.exists() and not args.resume:
        raise FileExistsError(
            f"{run_root} exists; use --resume or choose another output directory."
        )
    shards.mkdir(parents=True, exist_ok=True)
    (run_root / "social_floor_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    for seed in seeds:
        for tau in TAU_VALUES:
            path = shards / f"s{seed}__tau{tau:g}.json.gz"
            if args.resume and _valid_shard(path, design_id):
                continue
            tasks.append(
                {
                    **design,
                    "seed": int(seed),
                    "tau_social": float(tau),
                    "run_root": str(run_root),
                }
            )

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
                        f"[{i}/{len(futures)}] seed={result['seed']} "
                        f"tau={result['tau_social']:g} runs={result['runs']}",
                        flush=True,
                    )

    payloads = []
    expected = {(seed, tau) for seed in seeds for tau in TAU_VALUES}
    observed = set()
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("design_id") != design_id:
            continue
        payloads.append(value)
        observed.add((int(value["seed"]), float(value["tau_social"])))
    if observed != expected:
        raise RuntimeError(f"Missing completed blocks: {sorted(expected - observed)}")

    runs = [x for p in payloads for x in p["runs"]]
    beliefs = [x for p in payloads for x in p["terminal_beliefs"]]
    lambdas = [x for p in payloads for x in p["lambda_checkpoints"]]
    belief_checkpoints = [x for p in payloads for x in p["belief_checkpoints"]]
    jammer = [x for p in payloads for x in p["jammer_strategy"]]

    expected_runs = len(seeds) * len(TAU_VALUES) * 14
    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Run-count mismatch: expected {expected_runs}, observed {len(runs)}."
        )

    all_finite = all(math.isfinite(float(row["mse_truth"])) for row in runs)
    fixed_horizon = all(int(row["steps_run"]) == int(args.horizon) for row in runs)
    max_mse = max(float(row["mse_truth"]) for row in runs)
    max_abs_belief = max(
        abs(float(row["terminal_mu_theta"])) for row in beliefs
    )

    floor25 = _floor_screen(lambdas, 25) if args.horizon > 25 else math.nan
    floor50 = _floor_screen(lambdas, 50) if args.horizon > 50 else math.nan
    early_floor_pass = (
        bool(floor25 < 0.05 and floor50 < 0.25)
        if math.isfinite(floor25) and math.isfinite(floor50)
        else None
    )

    iv_same_group_direction = _mean_cell_difference(
        runs,
        experiment="IV",
        tau=1.0,
        sender="null",
        metric="dominant_same_group_cycle_share",
        high_filter=lambda r: (
            r["homophily_level"] == "high"
            and r["segregation_level"] == "high"
        ),
        low_filter=lambda r: (
            r["homophily_level"] == "low"
            and r["segregation_level"] == "high"
        ),
    )
    exp3_gateway_direction = _mean_cell_difference(
        runs,
        experiment="III",
        tau=1.0,
        sender="null",
        metric="gateway_incoming_reliance_share",
        high_filter=lambda r: r["redundancy_level"] == "high",
        low_filter=lambda r: r["redundancy_level"] == "low",
    )

    numerical_pass = bool(
        all_finite
        and fixed_horizon
        and max_mse < 1_000_000
        and max_abs_belief < 10_000
    )
    production_eligibility_screen = (
        bool(numerical_pass and early_floor_pass)
        if early_floor_pass is not None
        else None
    )

    gate = {
        "pass": numerical_pass,
        "design_id": design_id,
        "observed_blocks": len(observed),
        "expected_blocks": len(expected),
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "all_run_mse_finite": all_finite,
        "all_fixed_horizon": fixed_horizon,
        "max_terminal_mse": max_mse,
        "max_abs_terminal_belief": max_abs_belief,
        "tau1_null_median_floor_share_t25": floor25,
        "tau1_null_median_floor_share_t50": floor50,
        "precommitted_early_floor_screen": early_floor_pass,
        "precommitted_production_eligibility_screen": production_eligibility_screen,
        "tau1_iv_highH_minus_lowH_same_group_cycle_share_highS_null": (
            iv_same_group_direction
        ),
        "tau1_exp3_highR_minus_lowR_gateway_incoming_share_null": (
            exp3_gateway_direction
        ),
        "network_claim_iv_direction_positive": (
            bool(iv_same_group_direction > 0.0)
            if math.isfinite(iv_same_group_direction)
            else None
        ),
        "network_claim_exp3_direction_positive": (
            bool(exp3_gateway_direction > 0.0)
            if math.isfinite(exp3_gateway_direction)
            else None
        ),
    }
    (run_root / "social_floor_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tau_rows = tau_contrasts(runs)
    sender_rows = sender_vs_null_contrasts(runs)

    _write_csv(run_root / "runs.csv", runs)
    _write_csv(run_root / "terminal_beliefs.csv.gz", beliefs, gzip_output=True)
    _write_csv(run_root / "lambda_checkpoints.csv", lambdas)
    _write_csv(run_root / "belief_checkpoints.csv", belief_checkpoints)
    _write_csv(run_root / "jammer_strategy.csv.gz", jammer, gzip_output=True)
    _write_csv(run_root / "tau_contrasts.csv", tau_rows)
    _write_csv(run_root / "sender_vs_null_contrasts.csv", sender_rows)

    bundle = run_root / f"paper_b_social_floor_{design_id}_shareable.zip"
    names = (
        "social_floor_manifest.json",
        "social_floor_gate.json",
        "runs.csv",
        "terminal_beliefs.csv.gz",
        "lambda_checkpoints.csv",
        "belief_checkpoints.csv",
        "jammer_strategy.csv.gz",
        "tau_contrasts.csv",
        "sender_vs_null_contrasts.csv",
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(run_root / name, arcname=name)
        archive.write(plan_path, arcname="SOCIAL_EVIDENCE_FLOOR_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
