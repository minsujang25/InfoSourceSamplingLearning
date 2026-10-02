"""Final narrow calibration for Paper B.

No model redesign occurs here. Frozen receiver specification:
    tau_social = 1
    peer_evidence_mode = source_posterior
    frozen_ranking_mode = pre_disruption

Each run executes to T=400. Period 199 is the T=200 checkpoint and period 399
is the T=400 checkpoint on the same stochastic trajectory.

Per seed:
    Exp IV null:         2 H x 2 S x 2 reliance = 8
    Exp IV fixed high-S: 2 H x 1 S x 2 reliance = 4
    Exp III:             2 R x 2 reliance x 2 sender = 8
    total = 20

Default eight seeds therefore produce 160 simulations.
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


TAU_SOCIAL = 1.0
RELIANCE_MODES = ("adaptive", "frozen")
HOMOPHILY_LEVELS = ("low", "high")
SEGREGATION_LEVELS = ("low", "high")
REDUNDANCY_LEVELS = ("low", "high")
CALIBRATION_SENDERS = ("null", "fixed_biased")
PRIMARY_PERIODS = (199, 399)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="8")
    parser.add_argument("--seed-start", type=int, default=5101)
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
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_final_calibration",
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


def expected_conditions_per_seed() -> list[dict]:
    """Return the frozen 20-condition calibration design for one seed."""
    out = []
    for h in HOMOPHILY_LEVELS:
        for s in SEGREGATION_LEVELS:
            for reliance in RELIANCE_MODES:
                out.append(
                    {
                        "experiment": "IV",
                        "diagnostic_block": "null_primary",
                        "homophily_level": h,
                        "segregation_level": s,
                        "redundancy_level": "",
                        "reliance_mode": reliance,
                        "sender_regime": "null",
                    }
                )
    for h in HOMOPHILY_LEVELS:
        for reliance in RELIANCE_MODES:
            out.append(
                {
                    "experiment": "IV",
                    "diagnostic_block": "fixed_biased_highS",
                    "homophily_level": h,
                    "segregation_level": "high",
                    "redundancy_level": "",
                    "reliance_mode": reliance,
                    "sender_regime": "fixed_biased",
                }
            )
    for redundancy in REDUNDANCY_LEVELS:
        for reliance in RELIANCE_MODES:
            for sender in CALIBRATION_SENDERS:
                out.append(
                    {
                        "experiment": "III",
                        "diagnostic_block": "redundancy",
                        "homophily_level": "",
                        "segregation_level": "",
                        "redundancy_level": redundancy,
                        "reliance_mode": reliance,
                        "sender_regime": sender,
                    }
                )
    return out


def scientific_code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/metrics.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_local_diagnostic.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_final_calibration.py",
        "paper_b/FINAL_CALIBRATION_PLAN.md",
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
    config = base_model_config(
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
    config.update(
        {
            "tau_social": TAU_SOCIAL,
            "peer_evidence_mode": "source_posterior",
            "frozen_ranking_mode": "pre_disruption",
        }
    )
    return config


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


def _run_exp4(seed: int, task: dict) -> list[dict]:
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

    # Full H x S x adaptive/frozen under the null sender.
    for h in HOMOPHILY_LEVELS:
        for s in SEGREGATION_LEVELS:
            for reliance in RELIANCE_MODES:
                cfg = _base_config(seed, task)
                cfg.update(
                    {
                        "mu_theta": initial[s],
                        "initial_theta_type": f"final_calibration_exp4_{s}",
                        "fixed_group_ids": groups,
                        "structural_source_map": blueprint[h],
                    }
                )
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime=f"segregation_{s}",
                    environment=f"final_exp4_h_{h}",
                    reliance_mode=reliance,
                    jammer_active=False,
                    jammer_regime="null",
                    peer_evidence_mode="source_posterior",
                    frozen_ranking_mode="pre_disruption",
                    k=int(task["K"]),
                    design_id=str(task["design_id"]),
                    block_id=f"exp4_null__s{seed}",
                    save_edge_log=False,
                )
                _decorate(
                    result,
                    experiment="IV",
                    diagnostic_block="null_primary",
                    homophily_level=h,
                    segregation_level=s,
                    redundancy_level="",
                    sender_regime="null",
                    tau_social=TAU_SOCIAL,
                )
                results.append(result)

    # High-S fixed-biased stress, paired to the corresponding null conditions.
    for h in HOMOPHILY_LEVELS:
        for reliance in RELIANCE_MODES:
            cfg = _base_config(seed, task)
            cfg.update(
                {
                    "mu_theta": initial["high"],
                    "initial_theta_type": "final_calibration_exp4_high",
                    "fixed_group_ids": groups,
                    "structural_source_map": blueprint[h],
                }
            )
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime="segregation_high",
                environment=f"final_exp4_h_{h}",
                reliance_mode=reliance,
                jammer_active=False,
                jammer_regime="fixed_biased",
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"exp4_fixed__s{seed}",
                save_edge_log=False,
            )
            _decorate(
                result,
                experiment="IV",
                diagnostic_block="fixed_biased_highS",
                homophily_level=h,
                segregation_level="high",
                redundancy_level="",
                sender_regime="fixed_biased",
                tau_social=TAU_SOCIAL,
            )
            results.append(result)
    return results


def _run_exp3(seed: int, task: dict) -> list[dict]:
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
        for reliance in RELIANCE_MODES:
            for sender in CALIBRATION_SENDERS:
                cfg = _base_config(seed, task)
                cfg["structural_source_map"] = blueprint[redundancy]
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime="flat",
                    environment=f"final_exp3_r_{redundancy}",
                    reliance_mode=reliance,
                    jammer_active=False,
                    jammer_regime=sender,
                    peer_evidence_mode="source_posterior",
                    frozen_ranking_mode="pre_disruption",
                    gateway_positions=gateways,
                    k=int(task["K"]),
                    design_id=str(task["design_id"]),
                    block_id=f"exp3__s{seed}",
                    save_edge_log=False,
                )
                _decorate(
                    result,
                    experiment="III",
                    diagnostic_block="redundancy",
                    homophily_level="",
                    segregation_level="",
                    redundancy_level=redundancy,
                    sender_regime=sender,
                    tau_social=TAU_SOCIAL,
                )
                results.append(result)
    return results


def _run_block(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2
    seed = int(task["seed"])
    results = _run_exp4(seed, task) + _run_exp3(seed, task)

    payload = {
        "complete": True,
        "design_id": task["design_id"],
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
    path = Path(task["run_root"]) / "shards" / f"s{seed}.json.gz"
    _atomic_write(path, payload)
    return {"seed": seed, "runs": len(payload["runs"])}


def _valid_shard(path: Path, design_id: str) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        return (
            value.get("complete") is True
            and value.get("design_id") == design_id
            and len(value.get("runs", [])) == 20
        )
    except Exception:
        return False


def _labels(row: dict) -> tuple:
    return (
        row["experiment"],
        int(row["seed"]),
        row["diagnostic_block"],
        row.get("homophily_level", ""),
        row.get("segregation_level", ""),
        row.get("redundancy_level", ""),
        row["reliance_mode"],
        row["sender_regime"],
    )


def _merge_checkpoints(
    belief_rows: list[dict],
    lambda_rows: list[dict],
) -> list[dict]:
    lambda_index = {
        (*_labels(row), int(row["period"])): row
        for row in lambda_rows
    }
    out = []
    for belief in belief_rows:
        key = (*_labels(belief), int(belief["period"]))
        lam = lambda_index.get(key)
        if lam is None:
            continue
        merged = dict(belief)
        for name, value in lam.items():
            if name not in merged:
                merged[name] = value
        out.append(merged)
    return out


def _horizon_pairs(checkpoints: list[dict]) -> list[dict]:
    by_run = defaultdict(dict)
    for row in checkpoints:
        period = int(row["period"])
        if period in PRIMARY_PERIODS:
            by_run[_labels(row)][period] = row

    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "belief_variance",
        "squared_displacement",
        "posterior_sd_median",
        "posterior_sd_floor_share",
        "effective_homophily",
        "peer_reliance_mass",
        "dominant_expert_reach_share",
        "dominant_jammer_reach_share",
        "dominant_citizen_cycle_share",
        "dominant_same_group_cycle_share",
        "effective_incoming_hhi",
        "effective_incoming_top5_share",
        "gateway_incoming_reliance_share",
    )

    out = []
    for key, periods in sorted(by_run.items()):
        if not all(period in periods for period in PRIMARY_PERIODS):
            continue
        p200 = periods[199]
        p400 = periods[399]
        mse200 = float(p200["mse_truth"])
        mse400 = float(p400["mse_truth"])
        abs_change = abs(mse400 - mse200)
        rel_change = abs_change / max(abs(mse200), 1e-6)
        row = {
            "experiment": key[0],
            "seed": key[1],
            "diagnostic_block": key[2],
            "homophily_level": key[3],
            "segregation_level": key[4],
            "redundancy_level": key[5],
            "reliance_mode": key[6],
            "sender_regime": key[7],
            "mse_absolute_change_400_minus_200": abs_change,
            "mse_relative_change_400_vs_200": rel_change,
            "practically_stable_mse": bool(
                rel_change < 0.10 or abs_change < 0.005
            ),
        }
        for metric in metrics:
            row[f"{metric}_t200"] = p200.get(metric, math.nan)
            row[f"{metric}_t400"] = p400.get(metric, math.nan)
            left = float(p200.get(metric, math.nan))
            right = float(p400.get(metric, math.nan))
            row[f"{metric}_change_400_minus_200"] = (
                right - left
                if math.isfinite(left) and math.isfinite(right)
                else math.nan
            )
        out.append(row)
    return out


def _checkpoint_index(checkpoints: list[dict]) -> dict:
    return {
        (*_labels(row), int(row["period"])): row
        for row in checkpoints
    }


def _iv_contrast_rows(checkpoints: list[dict]) -> list[dict]:
    idx = _checkpoint_index(checkpoints)
    out = []
    for seed in sorted({int(r["seed"]) for r in checkpoints}):
        for reliance in RELIANCE_MODES:
            for period in PRIMARY_PERIODS:
                def get(h, s):
                    key = (
                        "IV", seed, "null_primary", h, s, "",
                        reliance, "null", period,
                    )
                    return idx.get(key)

                hh_hs = get("high", "high")
                lh_hs = get("low", "high")
                hh_ls = get("high", "low")
                lh_ls = get("low", "low")
                if not all((hh_hs, lh_hs, hh_ls, lh_ls)):
                    continue

                hs_mse = float(hh_hs["mse_truth"]) - float(lh_hs["mse_truth"])
                ls_mse = float(hh_ls["mse_truth"]) - float(lh_ls["mse_truth"])
                interaction = hs_mse - ls_mse

                same_group_hs = (
                    float(hh_hs["dominant_same_group_cycle_share"])
                    - float(lh_hs["dominant_same_group_cycle_share"])
                )
                effective_h_hs = (
                    float(hh_hs["effective_homophily"])
                    - float(lh_hs["effective_homophily"])
                )
                out.extend(
                    [
                        {
                            "experiment": "IV",
                            "seed": seed,
                            "period": period,
                            "horizon_T": period + 1,
                            "reliance_mode": reliance,
                            "contrast_name": "H_by_S_interaction_mse",
                            "contrast_value": interaction,
                        },
                        {
                            "experiment": "IV",
                            "seed": seed,
                            "period": period,
                            "horizon_T": period + 1,
                            "reliance_mode": reliance,
                            "contrast_name": "highS_highH_minus_lowH_same_group_cycle",
                            "contrast_value": same_group_hs,
                        },
                        {
                            "experiment": "IV",
                            "seed": seed,
                            "period": period,
                            "horizon_T": period + 1,
                            "reliance_mode": reliance,
                            "contrast_name": "highS_highH_minus_lowH_effective_homophily",
                            "contrast_value": effective_h_hs,
                        },
                    ]
                )
    return out


def _iii_contrast_rows(checkpoints: list[dict]) -> list[dict]:
    idx = _checkpoint_index(checkpoints)
    out = []
    for seed in sorted({int(r["seed"]) for r in checkpoints}):
        for reliance in RELIANCE_MODES:
            for period in PRIMARY_PERIODS:
                def get(redundancy, sender):
                    key = (
                        "III", seed, "redundancy", "", "",
                        redundancy, reliance, sender, period,
                    )
                    return idx.get(key)

                low_null = get("low", "null")
                high_null = get("high", "null")
                low_fixed = get("low", "fixed_biased")
                high_fixed = get("high", "fixed_biased")
                if not all((low_null, high_null, low_fixed, high_fixed)):
                    continue

                null_mse = (
                    float(high_null["mse_truth"]) - float(low_null["mse_truth"])
                )
                gateway = (
                    float(high_null["gateway_incoming_reliance_share"])
                    - float(low_null["gateway_incoming_reliance_share"])
                )
                low_damage = (
                    float(low_fixed["mse_truth"]) - float(low_null["mse_truth"])
                )
                high_damage = (
                    float(high_fixed["mse_truth"]) - float(high_null["mse_truth"])
                )
                damage_redundancy = high_damage - low_damage

                out.extend(
                    [
                        {
                            "experiment": "III",
                            "seed": seed,
                            "period": period,
                            "horizon_T": period + 1,
                            "reliance_mode": reliance,
                            "contrast_name": "highR_minus_lowR_null_mse",
                            "contrast_value": null_mse,
                        },
                        {
                            "experiment": "III",
                            "seed": seed,
                            "period": period,
                            "horizon_T": period + 1,
                            "reliance_mode": reliance,
                            "contrast_name": "highR_minus_lowR_gateway_incoming_share",
                            "contrast_value": gateway,
                        },
                        {
                            "experiment": "III",
                            "seed": seed,
                            "period": period,
                            "horizon_T": period + 1,
                            "reliance_mode": reliance,
                            "contrast_name": "highR_minus_lowR_fixed_biased_damage",
                            "contrast_value": damage_redundancy,
                        },
                    ]
                )
    return out


def _sender_damage_rows(checkpoints: list[dict]) -> list[dict]:
    idx = _checkpoint_index(checkpoints)
    out = []
    for row in checkpoints:
        if row["sender_regime"] != "fixed_biased":
            continue
        period = int(row["period"])
        if period not in PRIMARY_PERIODS:
            continue
        null_block = (
            "null_primary"
            if row["experiment"] == "IV"
            else row["diagnostic_block"]
        )
        null_key = (
            row["experiment"],
            int(row["seed"]),
            null_block,
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            row["reliance_mode"],
            "null",
            period,
        )
        null = idx.get(null_key)
        if null is None:
            continue
        out.append(
            {
                "experiment": row["experiment"],
                "seed": int(row["seed"]),
                "period": period,
                "horizon_T": period + 1,
                "homophily_level": row.get("homophily_level", ""),
                "segregation_level": row.get("segregation_level", ""),
                "redundancy_level": row.get("redundancy_level", ""),
                "reliance_mode": row["reliance_mode"],
                "sender_regime": "fixed_biased",
                "delta_mse_vs_null": (
                    float(row["mse_truth"]) - float(null["mse_truth"])
                ),
                "delta_rmse_vs_null": (
                    float(row["rmse_truth"]) - float(null["rmse_truth"])
                ),
                "delta_mae_vs_null": (
                    float(row["mae_truth"]) - float(null["mae_truth"])
                ),
            }
        )
    return out


def _adaptive_frozen_rows(contrast_rows: list[dict]) -> list[dict]:
    index = {}
    for row in contrast_rows:
        key = (
            row["experiment"],
            int(row["seed"]),
            int(row["period"]),
            row["contrast_name"],
            row["reliance_mode"],
        )
        index[key] = float(row["contrast_value"])

    out = []
    bases = {
        (r["experiment"], int(r["seed"]), int(r["period"]), r["contrast_name"])
        for r in contrast_rows
    }
    for experiment, seed, period, name in sorted(bases):
        adaptive = index.get((experiment, seed, period, name, "adaptive"))
        frozen = index.get((experiment, seed, period, name, "frozen"))
        if adaptive is None or frozen is None:
            continue
        out.append(
            {
                "experiment": experiment,
                "seed": seed,
                "period": period,
                "horizon_T": period + 1,
                "contrast_name": name,
                "adaptive_value": adaptive,
                "frozen_value": frozen,
                "adaptive_minus_frozen": adaptive - frozen,
            }
        )
    return out


def _sign(value: float, zero_tol: float = 0.005) -> int:
    if abs(float(value)) <= zero_tol:
        return 0
    return 1 if value > 0 else -1


def _contrast_stability(contrast_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in contrast_rows:
        key = (
            row["experiment"],
            int(row["seed"]),
            row["reliance_mode"],
            row["contrast_name"],
        )
        grouped[key][int(row["period"])] = float(row["contrast_value"])

    out = []
    for key, periods in sorted(grouped.items()):
        if not all(period in periods for period in PRIMARY_PERIODS):
            continue
        c200 = periods[199]
        c400 = periods[399]
        abs_change = abs(c400 - c200)
        rel_change = abs_change / max(abs(c200), 1e-6)
        same_sign = _sign(c200) == _sign(c400)
        stable = bool(same_sign and (rel_change < 0.25 or abs_change < 0.005))
        out.append(
            {
                "experiment": key[0],
                "seed": key[1],
                "reliance_mode": key[2],
                "contrast_name": key[3],
                "contrast_t200": c200,
                "contrast_t400": c400,
                "absolute_change": abs_change,
                "relative_change": rel_change,
                "same_sign": same_sign,
                "precommitted_stable": stable,
            }
        )
    return out


def _primary_cell_stability(horizon_pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in horizon_pairs:
        if row["sender_regime"] != "null":
            continue
        key = (
            row["experiment"],
            row["homophily_level"],
            row["segregation_level"],
            row["redundancy_level"],
            row["reliance_mode"],
        )
        groups[key].append(bool(row["practically_stable_mse"]))

    out = []
    for key, values in sorted(groups.items()):
        share = sum(values) / len(values)
        out.append(
            {
                "experiment": key[0],
                "homophily_level": key[1],
                "segregation_level": key[2],
                "redundancy_level": key[3],
                "reliance_mode": key[4],
                "n_seeds": len(values),
                "stable_seed_share": share,
                "passes_75pct_rule": bool(share >= 0.75),
            }
        )
    return out


def _median(values) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(finite)) if finite else math.nan


def _network_sign_reversal_gate(contrast_rows: list[dict]) -> tuple[bool | None, list[dict]]:
    targets = {
        ("IV", "highS_highH_minus_lowH_same_group_cycle"),
        ("III", "highR_minus_lowR_gateway_incoming_share"),
    }
    summaries = []
    all_ok = True
    any_evaluable = False
    for experiment, contrast_name in sorted(targets):
        for reliance in RELIANCE_MODES:
            values = {
                period: [
                    float(r["contrast_value"])
                    for r in contrast_rows
                    if r["experiment"] == experiment
                    and r["contrast_name"] == contrast_name
                    and r["reliance_mode"] == reliance
                    and int(r["period"]) == period
                ]
                for period in PRIMARY_PERIODS
            }
            if not all(values[p] for p in PRIMARY_PERIODS):
                continue
            any_evaluable = True
            med200 = _median(values[199])
            med400 = _median(values[399])
            same_sign = _sign(med200) == _sign(med400)
            all_ok = all_ok and same_sign
            summaries.append(
                {
                    "experiment": experiment,
                    "reliance_mode": reliance,
                    "contrast_name": contrast_name,
                    "median_t200": med200,
                    "median_t400": med400,
                    "same_sign": same_sign,
                }
            )
    return (all_ok if any_evaluable else None), summaries


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if not 0.0 < args.numerical_min_sd < 1.0:
        raise ValueError("--numerical-min-sd must lie in (0,1).")

    plan_path = Path(__file__).resolve().parents[1] / "FINAL_CALIBRATION_PLAN.md"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    design = {
        "design_version": 1,
        "purpose": "Paper B final horizon/adaptive-reliance calibration",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "plan_sha256": plan_sha,
        "software_versions": software_versions(),
        "seeds": seeds,
        "tau_social": TAU_SOCIAL,
        "peer_evidence_mode": "source_posterior",
        "frozen_ranking_mode": "pre_disruption",
        "reliance_modes": list(RELIANCE_MODES),
        "sender_regimes": list(CALIBRATION_SENDERS),
        "adaptive_jammer_excluded": True,
        "conditions_per_seed": len(expected_conditions_per_seed()),
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

    run_root = Path(args.output_dir) / f"final_calibration_{design_id}"
    shards = run_root / "shards"
    if run_root.exists() and not args.resume:
        raise FileExistsError(
            f"{run_root} exists; use --resume or choose another output directory."
        )
    shards.mkdir(parents=True, exist_ok=True)
    (run_root / "final_calibration_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    for seed in seeds:
        path = shards / f"s{seed}.json.gz"
        if args.resume and _valid_shard(path, design_id):
            continue
        tasks.append({**design, "seed": int(seed), "run_root": str(run_root)})

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
                        f"runs={result['runs']}",
                        flush=True,
                    )

    payloads = []
    observed = set()
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("design_id") != design_id:
            continue
        payloads.append(value)
        observed.add(int(value["seed"]))
    if observed != set(seeds):
        raise RuntimeError(f"Missing completed seeds: {sorted(set(seeds)-observed)}")

    runs = [x for p in payloads for x in p["runs"]]
    beliefs = [x for p in payloads for x in p["terminal_beliefs"]]
    lambdas = [x for p in payloads for x in p["lambda_checkpoints"]]
    belief_checkpoints = [x for p in payloads for x in p["belief_checkpoints"]]
    jammer = [x for p in payloads for x in p["jammer_strategy"]]

    expected_runs = len(seeds) * len(expected_conditions_per_seed())
    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Run-count mismatch: expected {expected_runs}, observed {len(runs)}."
        )

    merged_checkpoints = _merge_checkpoints(belief_checkpoints, lambdas)
    horizon_evaluable = int(args.horizon) >= 400
    horizon_pairs = _horizon_pairs(merged_checkpoints) if horizon_evaluable else []

    contrast_rows = []
    if horizon_evaluable:
        contrast_rows.extend(_iv_contrast_rows(merged_checkpoints))
        contrast_rows.extend(_iii_contrast_rows(merged_checkpoints))
    sender_rows = (
        _sender_damage_rows(merged_checkpoints) if horizon_evaluable else []
    )
    adaptive_frozen = (
        _adaptive_frozen_rows(contrast_rows) if horizon_evaluable else []
    )
    contrast_stability = (
        _contrast_stability(contrast_rows) if horizon_evaluable else []
    )
    cell_stability = (
        _primary_cell_stability(horizon_pairs) if horizon_evaluable else []
    )

    all_finite = all(math.isfinite(float(row["mse_truth"])) for row in runs)
    fixed_horizon = all(int(row["steps_run"]) == int(args.horizon) for row in runs)
    max_mse = max(float(row["mse_truth"]) for row in runs)
    max_abs_belief = max(abs(float(row["terminal_mu_theta"])) for row in beliefs)
    numerical_pass = bool(
        all_finite
        and fixed_horizon
        and max_mse < 1_000_000
        and max_abs_belief < 10_000
    )

    if horizon_evaluable:
        all_cells_75 = bool(cell_stability) and all(
            bool(row["passes_75pct_rule"]) for row in cell_stability
        )
        null_lambdas = [
            row for row in lambdas
            if row["sender_regime"] == "null"
            and int(row["period"]) in PRIMARY_PERIODS
        ]
        floor199 = _median(
            row["posterior_sd_floor_share"]
            for row in null_lambdas
            if int(row["period"]) == 199
        )
        floor399 = _median(
            row["posterior_sd_floor_share"]
            for row in null_lambdas
            if int(row["period"]) == 399
        )
        floor_gate = bool(floor199 == 0.0 and floor399 == 0.0)
        network_sign_gate, network_summaries = _network_sign_reversal_gate(
            contrast_rows
        )
        retain_t200 = bool(
            numerical_pass
            and all_cells_75
            and floor_gate
            and network_sign_gate is True
        )
        recommended_horizon = 200 if retain_t200 else 400
    else:
        all_cells_75 = None
        floor199 = math.nan
        floor399 = math.nan
        floor_gate = None
        network_sign_gate = None
        network_summaries = []
        retain_t200 = None
        recommended_horizon = None

    gate = {
        "pass": numerical_pass,
        "design_id": design_id,
        "observed_seeds": len(observed),
        "expected_seeds": len(seeds),
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "tau_social": TAU_SOCIAL,
        "adaptive_jammer_excluded": True,
        "horizon_evaluable": horizon_evaluable,
        "all_run_mse_finite": all_finite,
        "all_fixed_horizon": fixed_horizon,
        "max_terminal_mse": max_mse,
        "max_abs_terminal_belief": max_abs_belief,
        "all_primary_null_cells_pass_75pct_mse_stability": all_cells_75,
        "null_median_posterior_floor_share_t200": floor199,
        "null_median_posterior_floor_share_t400": floor399,
        "posterior_floor_gate": floor_gate,
        "key_network_contrast_no_sign_reversal": network_sign_gate,
        "precommitted_retain_T200": retain_t200,
        "precommitted_recommended_production_horizon": recommended_horizon,
        "network_sign_summaries": network_summaries,
    }
    (run_root / "final_calibration_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )

    _write_csv(run_root / "runs.csv", runs)
    _write_csv(run_root / "terminal_beliefs.csv.gz", beliefs, gzip_output=True)
    _write_csv(run_root / "lambda_checkpoints.csv", lambdas)
    _write_csv(run_root / "belief_checkpoints.csv", belief_checkpoints)
    _write_csv(run_root / "merged_checkpoints.csv", merged_checkpoints)
    _write_csv(run_root / "horizon_pairs.csv", horizon_pairs)
    _write_csv(run_root / "mechanism_contrasts.csv", contrast_rows)
    _write_csv(run_root / "adaptive_frozen_contrasts.csv", adaptive_frozen)
    _write_csv(run_root / "fixed_biased_vs_null.csv", sender_rows)
    _write_csv(run_root / "contrast_stability.csv", contrast_stability)
    _write_csv(run_root / "primary_cell_horizon_stability.csv", cell_stability)
    _write_csv(run_root / "jammer_strategy.csv.gz", jammer, gzip_output=True)

    bundle = run_root / f"paper_b_final_calibration_{design_id}_shareable.zip"
    names = (
        "final_calibration_manifest.json",
        "final_calibration_gate.json",
        "runs.csv",
        "terminal_beliefs.csv.gz",
        "lambda_checkpoints.csv",
        "belief_checkpoints.csv",
        "merged_checkpoints.csv",
        "horizon_pairs.csv",
        "mechanism_contrasts.csv",
        "adaptive_frozen_contrasts.csv",
        "fixed_biased_vs_null.csv",
        "contrast_stability.csv",
        "primary_cell_horizon_stability.csv",
        "jammer_strategy.csv.gz",
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = run_root / name
            if path.exists():
                archive.write(path, arcname=name)
        archive.write(plan_path, arcname="FINAL_CALIBRATION_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=True))
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
