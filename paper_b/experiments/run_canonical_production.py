"""Canonical 500-seed production runner for Social Networks Paper B.

Frozen scientific specification:
    N=100, T=400, epsilon=.05, credit=20, K=1
    peer_evidence_mode=source_posterior
    tau_social=1
    frozen_ranking_mode=pre_disruption

Default seeds: 6001-6500.

Per seed:
    Experiment IV: 14 runs
    Experiment III: 10 runs
    Total: 24 runs

Across 500 seeds: 12,000 simulations.
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
PRIMARY_SENDERS = ("null", "fixed_biased")
TERMINAL_PERIOD = 399
HORIZON_SENSITIVITY_PERIOD = 199


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
    parser.add_argument(
        "--experiments",
        default="III,IV",
        help="Comma-separated subset of III,IV.",
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_canonical",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
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


def resolve_experiments(value: str) -> tuple[str, ...]:
    experiments = tuple(x.strip().upper() for x in value.split(",") if x.strip())
    if not experiments:
        raise ValueError("At least one experiment is required.")
    invalid = sorted(set(experiments) - {"III", "IV"})
    if invalid:
        raise ValueError(f"Unknown experiment(s): {invalid}.")
    if len(set(experiments)) != len(experiments):
        raise ValueError("Experiments must be unique.")
    return experiments


def expected_conditions(experiment: str) -> list[dict]:
    out = []
    if experiment == "IV":
        for h in HOMOPHILY_LEVELS:
            for s in SEGREGATION_LEVELS:
                for reliance in RELIANCE_MODES:
                    out.append(
                        {
                            "experiment": "IV",
                            "production_block": "null_primary",
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
                        "production_block": "fixed_biased_highS",
                        "homophily_level": h,
                        "segregation_level": "high",
                        "redundancy_level": "",
                        "reliance_mode": reliance,
                        "sender_regime": "fixed_biased",
                    }
                )
        for h in HOMOPHILY_LEVELS:
            out.append(
                {
                    "experiment": "IV",
                    "production_block": "adaptive_jammer_secondary",
                    "homophily_level": h,
                    "segregation_level": "high",
                    "redundancy_level": "",
                    "reliance_mode": "adaptive",
                    "sender_regime": "adaptive",
                }
            )
    elif experiment == "III":
        for redundancy in REDUNDANCY_LEVELS:
            for reliance in RELIANCE_MODES:
                for sender in PRIMARY_SENDERS:
                    out.append(
                        {
                            "experiment": "III",
                            "production_block": "redundancy_primary",
                            "homophily_level": "",
                            "segregation_level": "",
                            "redundancy_level": redundancy,
                            "reliance_mode": reliance,
                            "sender_regime": sender,
                        }
                    )
        for redundancy in REDUNDANCY_LEVELS:
            out.append(
                {
                    "experiment": "III",
                    "production_block": "adaptive_jammer_secondary",
                    "homophily_level": "",
                    "segregation_level": "",
                    "redundancy_level": redundancy,
                    "reliance_mode": "adaptive",
                    "sender_regime": "adaptive",
                }
            )
    else:
        raise ValueError(experiment)
    return out


def scientific_code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/metrics.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_local_diagnostic.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_canonical_production.py",
        "paper_b/CANONICAL_PRODUCTION_PLAN.md",
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
        s: exp4_initial_beliefs(
            seed=seed,
            n_citizens=n,
            group_ids=groups,
            segregation=s,
            high_group_shift=float(task["high_group_shift"]),
            residual_sd=float(task["prior_residual_sd"]),
        )
        for s in SEGREGATION_LEVELS
    }
    results = []

    for h in HOMOPHILY_LEVELS:
        for s in SEGREGATION_LEVELS:
            for reliance in RELIANCE_MODES:
                cfg = _base_config(seed, task)
                cfg.update(
                    {
                        "mu_theta": initial[s],
                        "initial_theta_type": f"production_exp4_{s}",
                        "fixed_group_ids": groups,
                        "structural_source_map": blueprint[h],
                    }
                )
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime=f"segregation_{s}",
                    environment=f"production_exp4_h_{h}",
                    reliance_mode=reliance,
                    jammer_active=False,
                    jammer_regime="null",
                    peer_evidence_mode="source_posterior",
                    frozen_ranking_mode="pre_disruption",
                    k=int(task["K"]),
                    design_id=str(task["design_id"]),
                    block_id=f"IV_null__s{seed}",
                    save_edge_log=False,
                )
                _decorate(
                    result,
                    experiment="IV",
                    production_block="null_primary",
                    homophily_level=h,
                    segregation_level=s,
                    redundancy_level="",
                    sender_regime="null",
                    tau_social=TAU_SOCIAL,
                )
                results.append(result)

    for h in HOMOPHILY_LEVELS:
        for reliance in RELIANCE_MODES:
            cfg = _base_config(seed, task)
            cfg.update(
                {
                    "mu_theta": initial["high"],
                    "initial_theta_type": "production_exp4_high",
                    "fixed_group_ids": groups,
                    "structural_source_map": blueprint[h],
                }
            )
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime="segregation_high",
                environment=f"production_exp4_h_{h}",
                reliance_mode=reliance,
                jammer_active=False,
                jammer_regime="fixed_biased",
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"IV_fixed__s{seed}",
                save_edge_log=False,
            )
            _decorate(
                result,
                experiment="IV",
                production_block="fixed_biased_highS",
                homophily_level=h,
                segregation_level="high",
                redundancy_level="",
                sender_regime="fixed_biased",
                tau_social=TAU_SOCIAL,
            )
            results.append(result)

    for h in HOMOPHILY_LEVELS:
        cfg = _base_config(seed, task)
        cfg.update(
            {
                "mu_theta": initial["high"],
                "initial_theta_type": "production_exp4_high",
                "fixed_group_ids": groups,
                "structural_source_map": blueprint[h],
            }
        )
        result = run_condition(
            base_config=cfg,
            seed=seed,
            regime="segregation_high",
            environment=f"production_exp4_h_{h}",
            reliance_mode="adaptive",
            jammer_active=True,
            jammer_regime="adaptive",
            peer_evidence_mode="source_posterior",
            frozen_ranking_mode="pre_disruption",
            k=int(task["K"]),
            design_id=str(task["design_id"]),
            block_id=f"IV_adaptive_jammer__s{seed}",
            save_edge_log=False,
        )
        _decorate(
            result,
            experiment="IV",
            production_block="adaptive_jammer_secondary",
            homophily_level=h,
            segregation_level="high",
            redundancy_level="",
            sender_regime="adaptive",
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
            for sender in PRIMARY_SENDERS:
                cfg = _base_config(seed, task)
                cfg["structural_source_map"] = blueprint[redundancy]
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime="flat",
                    environment=f"production_exp3_r_{redundancy}",
                    reliance_mode=reliance,
                    jammer_active=False,
                    jammer_regime=sender,
                    peer_evidence_mode="source_posterior",
                    frozen_ranking_mode="pre_disruption",
                    gateway_positions=gateways,
                    k=int(task["K"]),
                    design_id=str(task["design_id"]),
                    block_id=f"III_primary__s{seed}",
                    save_edge_log=False,
                )
                _decorate(
                    result,
                    experiment="III",
                    production_block="redundancy_primary",
                    homophily_level="",
                    segregation_level="",
                    redundancy_level=redundancy,
                    sender_regime=sender,
                    tau_social=TAU_SOCIAL,
                )
                results.append(result)

    for redundancy in REDUNDANCY_LEVELS:
        cfg = _base_config(seed, task)
        cfg["structural_source_map"] = blueprint[redundancy]
        result = run_condition(
            base_config=cfg,
            seed=seed,
            regime="flat",
            environment=f"production_exp3_r_{redundancy}",
            reliance_mode="adaptive",
            jammer_active=True,
            jammer_regime="adaptive",
            peer_evidence_mode="source_posterior",
            frozen_ranking_mode="pre_disruption",
            gateway_positions=gateways,
            k=int(task["K"]),
            design_id=str(task["design_id"]),
            block_id=f"III_adaptive_jammer__s{seed}",
            save_edge_log=False,
        )
        _decorate(
            result,
            experiment="III",
            production_block="adaptive_jammer_secondary",
            homophily_level="",
            segregation_level="",
            redundancy_level=redundancy,
            sender_regime="adaptive",
            tau_social=TAU_SOCIAL,
        )
        results.append(result)

    return results


def _run_block(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2
    seed = int(task["seed"])
    experiment = str(task["experiment"])
    results = _run_exp4(seed, task) if experiment == "IV" else _run_exp3(seed, task)

    payload = {
        "complete": True,
        "design_id": task["design_id"],
        "experiment": experiment,
        "seed": seed,
        "runs": [r["run"] for r in results],
        "terminal_beliefs": [x for r in results for x in r["beliefs"]],
        "lambda_checkpoints": [x for r in results for x in r["lambda_checkpoints"]],
        "belief_checkpoints": [x for r in results for x in r["belief_checkpoints"]],
        "jammer_strategy": [x for r in results for x in r["jammer_strategy"]],
    }
    path = (
        Path(task["run_root"])
        / "shards"
        / f"{experiment}__s{seed}.json.gz"
    )
    _atomic_write(path, payload)
    return {
        "experiment": experiment,
        "seed": seed,
        "runs": len(payload["runs"]),
    }


def _valid_shard(path: Path, design_id: str, expected_runs: int) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        return (
            value.get("complete") is True
            and value.get("design_id") == design_id
            and len(value.get("runs", [])) == expected_runs
        )
    except Exception:
        return False


def _labels(row: dict) -> tuple:
    return (
        row["experiment"],
        int(row["seed"]),
        row["production_block"],
        row.get("homophily_level", ""),
        row.get("segregation_level", ""),
        row.get("redundancy_level", ""),
        row["reliance_mode"],
        row["sender_regime"],
    )


def _merge_checkpoints(belief_rows: list[dict], lambda_rows: list[dict]) -> list[dict]:
    idx = {(*_labels(r), int(r["period"])): r for r in lambda_rows}
    out = []
    for belief in belief_rows:
        key = (*_labels(belief), int(belief["period"]))
        lam = idx.get(key)
        if lam is None:
            continue
        row = dict(belief)
        for name, value in lam.items():
            if name not in row:
                row[name] = value
        out.append(row)
    return out


def _run_index(rows: list[dict]) -> dict:
    return {_labels(row): row for row in rows}


def _checkpoint_index(rows: list[dict]) -> dict:
    return {(*_labels(row), int(row["period"])): row for row in rows}


def _contrast_row(
    *,
    seed: int,
    experiment: str,
    contrast_name: str,
    value: float,
    reliance_mode: str = "",
    sender_regime: str = "",
    period: int = TERMINAL_PERIOD,
) -> dict:
    return {
        "experiment": experiment,
        "seed": int(seed),
        "period": int(period),
        "horizon_T": int(period) + 1,
        "reliance_mode": reliance_mode,
        "sender_regime": sender_regime,
        "contrast_name": contrast_name,
        "contrast_value": float(value),
    }


def _compute_contrasts_from_index(idx: dict, *, checkpoint: bool) -> list[dict]:
    periods = (HORIZON_SENSITIVITY_PERIOD, TERMINAL_PERIOD) if checkpoint else (TERMINAL_PERIOD,)
    seeds = sorted({int(key[1]) for key in idx})
    out = []

    def get_iv(seed, block, h, s, reliance, sender, period):
        key = (
            "IV", seed, block, h, s, "", reliance, sender
        )
        return idx.get((*key, period)) if checkpoint else idx.get(key)

    def get_iii(seed, block, r, reliance, sender, period):
        key = (
            "III", seed, block, "", "", r, reliance, sender
        )
        return idx.get((*key, period)) if checkpoint else idx.get(key)

    for seed in seeds:
        for period in periods:
            # Experiment IV primary contrasts.
            for reliance in RELIANCE_MODES:
                hh_hs = get_iv(seed, "null_primary", "high", "high", reliance, "null", period)
                lh_hs = get_iv(seed, "null_primary", "low", "high", reliance, "null", period)
                hh_ls = get_iv(seed, "null_primary", "high", "low", reliance, "null", period)
                lh_ls = get_iv(seed, "null_primary", "low", "low", reliance, "null", period)
                if all((hh_hs, lh_hs, hh_ls, lh_ls)):
                    h_hs_mse = float(hh_hs["mse_truth"]) - float(lh_hs["mse_truth"])
                    h_ls_mse = float(hh_ls["mse_truth"]) - float(lh_ls["mse_truth"])
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="IV",
                        reliance_mode=reliance,
                        contrast_name=f"IV_HxS_MSE__{reliance}",
                        value=h_hs_mse - h_ls_mse,
                        period=period,
                    ))
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="IV",
                        reliance_mode=reliance,
                        contrast_name=f"IV_highS_H_effect_same_group_cycle__{reliance}",
                        value=(
                            float(hh_hs["dominant_same_group_cycle_share"])
                            - float(lh_hs["dominant_same_group_cycle_share"])
                        ),
                        period=period,
                    ))
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="IV",
                        reliance_mode=reliance,
                        contrast_name=f"IV_highS_H_effect_effective_homophily__{reliance}",
                        value=(
                            float(hh_hs["effective_homophily"])
                            - float(lh_hs["effective_homophily"])
                        ),
                        period=period,
                    ))

                fixed_h = get_iv(seed, "fixed_biased_highS", "high", "high", reliance, "fixed_biased", period)
                fixed_l = get_iv(seed, "fixed_biased_highS", "low", "high", reliance, "fixed_biased", period)
                null_h = get_iv(seed, "null_primary", "high", "high", reliance, "null", period)
                null_l = get_iv(seed, "null_primary", "low", "high", reliance, "null", period)
                if all((fixed_h, fixed_l, null_h, null_l)):
                    damage_h = float(fixed_h["mse_truth"]) - float(null_h["mse_truth"])
                    damage_l = float(fixed_l["mse_truth"]) - float(null_l["mse_truth"])
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="IV",
                        reliance_mode=reliance,
                        sender_regime="fixed_biased",
                        contrast_name=f"IV_highS_H_effect_fixed_biased_damage__{reliance}",
                        value=damage_h - damage_l,
                        period=period,
                    ))

            jam_h = get_iv(seed, "adaptive_jammer_secondary", "high", "high", "adaptive", "adaptive", period)
            jam_l = get_iv(seed, "adaptive_jammer_secondary", "low", "high", "adaptive", "adaptive", period)
            null_h = get_iv(seed, "null_primary", "high", "high", "adaptive", "null", period)
            null_l = get_iv(seed, "null_primary", "low", "high", "adaptive", "null", period)
            if all((jam_h, jam_l, null_h, null_l)):
                out.append(_contrast_row(
                    seed=seed,
                    experiment="IV",
                    reliance_mode="adaptive",
                    sender_regime="adaptive",
                    contrast_name="IV_highS_H_effect_adaptive_jammer_damage__adaptive",
                    value=(
                        (float(jam_h["mse_truth"]) - float(null_h["mse_truth"]))
                        - (float(jam_l["mse_truth"]) - float(null_l["mse_truth"]))
                    ),
                    period=period,
                ))

            # Experiment III primary contrasts.
            for reliance in RELIANCE_MODES:
                low_null = get_iii(seed, "redundancy_primary", "low", reliance, "null", period)
                high_null = get_iii(seed, "redundancy_primary", "high", reliance, "null", period)
                low_fixed = get_iii(seed, "redundancy_primary", "low", reliance, "fixed_biased", period)
                high_fixed = get_iii(seed, "redundancy_primary", "high", reliance, "fixed_biased", period)
                if all((low_null, high_null)):
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="III",
                        reliance_mode=reliance,
                        contrast_name=f"III_highR_minus_lowR_null_MSE__{reliance}",
                        value=float(high_null["mse_truth"]) - float(low_null["mse_truth"]),
                        period=period,
                    ))
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="III",
                        reliance_mode=reliance,
                        contrast_name=f"III_highR_minus_lowR_gateway_share__{reliance}",
                        value=(
                            float(high_null["gateway_incoming_reliance_share"])
                            - float(low_null["gateway_incoming_reliance_share"])
                        ),
                        period=period,
                    ))
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="III",
                        reliance_mode=reliance,
                        contrast_name=f"III_highR_minus_lowR_incoming_HHI__{reliance}",
                        value=(
                            float(high_null["effective_incoming_hhi"])
                            - float(low_null["effective_incoming_hhi"])
                        ),
                        period=period,
                    ))
                if all((low_null, high_null, low_fixed, high_fixed)):
                    out.append(_contrast_row(
                        seed=seed,
                        experiment="III",
                        reliance_mode=reliance,
                        sender_regime="fixed_biased",
                        contrast_name=f"III_highR_minus_lowR_fixed_biased_damage__{reliance}",
                        value=(
                            (float(high_fixed["mse_truth"]) - float(high_null["mse_truth"]))
                            - (float(low_fixed["mse_truth"]) - float(low_null["mse_truth"]))
                        ),
                        period=period,
                    ))

            low_jam = get_iii(seed, "adaptive_jammer_secondary", "low", "adaptive", "adaptive", period)
            high_jam = get_iii(seed, "adaptive_jammer_secondary", "high", "adaptive", "adaptive", period)
            low_null = get_iii(seed, "redundancy_primary", "low", "adaptive", "null", period)
            high_null = get_iii(seed, "redundancy_primary", "high", "adaptive", "null", period)
            if all((low_jam, high_jam, low_null, high_null)):
                out.append(_contrast_row(
                    seed=seed,
                    experiment="III",
                    reliance_mode="adaptive",
                    sender_regime="adaptive",
                    contrast_name="III_highR_minus_lowR_adaptive_jammer_damage__adaptive",
                    value=(
                        (float(high_jam["mse_truth"]) - float(high_null["mse_truth"]))
                        - (float(low_jam["mse_truth"]) - float(low_null["mse_truth"]))
                    ),
                    period=period,
                ))

    return out


def _adaptive_frozen_contrasts(contrasts: list[dict]) -> list[dict]:
    mapping = {
        "IV_HxS_MSE": ("IV_HxS_MSE__adaptive", "IV_HxS_MSE__frozen"),
        "IV_highS_H_effect_same_group_cycle": (
            "IV_highS_H_effect_same_group_cycle__adaptive",
            "IV_highS_H_effect_same_group_cycle__frozen",
        ),
        "IV_highS_H_effect_effective_homophily": (
            "IV_highS_H_effect_effective_homophily__adaptive",
            "IV_highS_H_effect_effective_homophily__frozen",
        ),
        "IV_highS_H_effect_fixed_biased_damage": (
            "IV_highS_H_effect_fixed_biased_damage__adaptive",
            "IV_highS_H_effect_fixed_biased_damage__frozen",
        ),
        "III_highR_minus_lowR_null_MSE": (
            "III_highR_minus_lowR_null_MSE__adaptive",
            "III_highR_minus_lowR_null_MSE__frozen",
        ),
        "III_highR_minus_lowR_gateway_share": (
            "III_highR_minus_lowR_gateway_share__adaptive",
            "III_highR_minus_lowR_gateway_share__frozen",
        ),
        "III_highR_minus_lowR_incoming_HHI": (
            "III_highR_minus_lowR_incoming_HHI__adaptive",
            "III_highR_minus_lowR_incoming_HHI__frozen",
        ),
        "III_highR_minus_lowR_fixed_biased_damage": (
            "III_highR_minus_lowR_fixed_biased_damage__adaptive",
            "III_highR_minus_lowR_fixed_biased_damage__frozen",
        ),
    }
    idx = {
        (int(r["seed"]), int(r["period"]), r["contrast_name"]): float(r["contrast_value"])
        for r in contrasts
    }
    out = []
    seeds = sorted({int(r["seed"]) for r in contrasts})
    periods = sorted({int(r["period"]) for r in contrasts})
    for seed in seeds:
        for period in periods:
            for base, (a_name, f_name) in mapping.items():
                a = idx.get((seed, period, a_name))
                f = idx.get((seed, period, f_name))
                if a is None or f is None:
                    continue
                out.append(
                    {
                        "seed": seed,
                        "period": period,
                        "horizon_T": period + 1,
                        "contrast_name": f"{base}__adaptive_minus_frozen",
                        "adaptive_value": a,
                        "frozen_value": f,
                        "contrast_value": a - f,
                    }
                )
    return out


def _summary(values: list[float]) -> dict:
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "mcse": math.nan,
            "mc95_low": math.nan,
            "mc95_high": math.nan,
            "median": math.nan,
            "trimmed_mean_10pct": math.nan,
            "negative_share": math.nan,
            "positive_share": math.nan,
            "loo_mean_min": math.nan,
            "loo_mean_max": math.nan,
        }
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    mcse = sd / math.sqrt(n) if n > 0 else math.nan
    ordered = np.sort(arr)
    trim = int(math.floor(0.10 * n))
    trimmed = ordered[trim:n-trim] if n - 2 * trim > 0 else ordered
    if n > 1:
        total = float(arr.sum())
        loo = (total - arr) / (n - 1)
        loo_min = float(loo.min())
        loo_max = float(loo.max())
    else:
        loo_min = loo_max = mean
    return {
        "n": n,
        "mean": mean,
        "mcse": float(mcse),
        "mc95_low": float(mean - 1.96 * mcse),
        "mc95_high": float(mean + 1.96 * mcse),
        "median": float(np.median(arr)),
        "trimmed_mean_10pct": float(trimmed.mean()),
        "negative_share": float(np.mean(arr < 0.0)),
        "positive_share": float(np.mean(arr > 0.0)),
        "loo_mean_min": loo_min,
        "loo_mean_max": loo_max,
    }


def _contrast_summaries(contrasts: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in contrasts:
        key = (
            row.get("experiment", ""),
            int(row.get("period", TERMINAL_PERIOD)),
            row["contrast_name"],
        )
        groups[key].append(float(row["contrast_value"]))
    out = []
    for key, values in sorted(groups.items()):
        out.append(
            {
                "experiment": key[0],
                "period": key[1],
                "horizon_T": key[1] + 1,
                "contrast_name": key[2],
                **_summary(values),
            }
        )
    return out


def _cell_summaries(runs: list[dict]) -> list[dict]:
    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "belief_variance",
        "squared_displacement",
        "effective_homophily",
        "peer_reliance_mass",
        "dominant_expert_reach_share",
        "dominant_jammer_reach_share",
        "dominant_citizen_cycle_share",
        "dominant_same_group_cycle_share",
        "effective_incoming_hhi",
        "effective_incoming_top5_share",
        "gateway_incoming_reliance_share",
        "expert_reliance",
        "jammer_reliance",
        "peer_reliance",
    )
    groups = defaultdict(list)
    for row in runs:
        key = (
            row["experiment"],
            row["production_block"],
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            row["reliance_mode"],
            row["sender_regime"],
        )
        groups[key].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        result = {
            "experiment": key[0],
            "production_block": key[1],
            "homophily_level": key[2],
            "segregation_level": key[3],
            "redundancy_level": key[4],
            "reliance_mode": key[5],
            "sender_regime": key[6],
            "n_runs": len(rows),
        }
        for metric in metrics:
            vals = [
                float(r.get(metric, math.nan))
                for r in rows
                if math.isfinite(float(r.get(metric, math.nan)))
            ]
            result[f"{metric}_mean"] = float(np.mean(vals)) if vals else math.nan
            result[f"{metric}_median"] = float(np.median(vals)) if vals else math.nan
        out.append(result)
    return out


def _checkpoint_cell_summaries(checkpoints: list[dict]) -> list[dict]:
    metrics = (
        "mse_truth",
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
        "gateway_incoming_reliance_share",
    )
    groups = defaultdict(list)
    for row in checkpoints:
        if int(row["period"]) not in {
            0, 5, 10, 25, 50, 100, 150, 199, 299, 399
        }:
            continue
        key = (
            row["experiment"],
            row["production_block"],
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            row["reliance_mode"],
            row["sender_regime"],
            int(row["period"]),
        )
        groups[key].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        result = {
            "experiment": key[0],
            "production_block": key[1],
            "homophily_level": key[2],
            "segregation_level": key[3],
            "redundancy_level": key[4],
            "reliance_mode": key[5],
            "sender_regime": key[6],
            "period": key[7],
            "horizon_T": key[7] + 1,
            "n_runs": len(rows),
        }
        for metric in metrics:
            vals = [
                float(r.get(metric, math.nan))
                for r in rows
                if math.isfinite(float(r.get(metric, math.nan)))
            ]
            result[f"{metric}_mean"] = float(np.mean(vals)) if vals else math.nan
            result[f"{metric}_median"] = float(np.median(vals)) if vals else math.nan
        out.append(result)
    return out


def _sender_damage_rows(runs: list[dict]) -> list[dict]:
    idx = _run_index(runs)
    out = []
    for row in runs:
        sender = row["sender_regime"]
        if sender not in {"fixed_biased", "adaptive"}:
            continue
        if row["experiment"] == "IV":
            null_key = (
                "IV",
                int(row["seed"]),
                "null_primary",
                row["homophily_level"],
                row["segregation_level"],
                "",
                row["reliance_mode"],
                "null",
            )
        else:
            null_key = (
                "III",
                int(row["seed"]),
                "redundancy_primary",
                "",
                "",
                row["redundancy_level"],
                row["reliance_mode"],
                "null",
            )
        null = idx.get(null_key)
        if null is None:
            continue
        out.append(
            {
                "experiment": row["experiment"],
                "seed": int(row["seed"]),
                "homophily_level": row.get("homophily_level", ""),
                "segregation_level": row.get("segregation_level", ""),
                "redundancy_level": row.get("redundancy_level", ""),
                "reliance_mode": row["reliance_mode"],
                "sender_regime": sender,
                "delta_mse_vs_null": float(row["mse_truth"]) - float(null["mse_truth"]),
                "delta_rmse_vs_null": float(row["rmse_truth"]) - float(null["rmse_truth"]),
                "delta_mae_vs_null": float(row["mae_truth"]) - float(null["mae_truth"]),
                "delta_belief_variance_vs_null": (
                    float(row["belief_variance"]) - float(null["belief_variance"])
                ),
                "delta_squared_displacement_vs_null": (
                    float(row["squared_displacement"]) - float(null["squared_displacement"])
                ),
            }
        )
    return out


def _group_sender_damage_rows(beliefs: list[dict]) -> list[dict]:
    group_mse = {}
    grouped = defaultdict(list)
    for row in beliefs:
        if row["experiment"] != "IV" or row["segregation_level"] != "high":
            continue
        key = (
            int(row["seed"]),
            row["production_block"],
            row["homophily_level"],
            row["reliance_mode"],
            row["sender_regime"],
            int(row["citizen_group"]),
        )
        grouped[key].append(float(row["terminal_mu_theta"]) ** 2)
    for key, values in grouped.items():
        group_mse[key] = float(np.mean(values))

    out = []
    for key, mse in group_mse.items():
        seed, block, h, reliance, sender, group = key
        if sender not in {"fixed_biased", "adaptive"}:
            continue
        null_key = (
            seed, "null_primary", h, reliance, "null", group
        )
        null = group_mse.get(null_key)
        if null is None:
            continue
        out.append(
            {
                "experiment": "IV",
                "seed": seed,
                "homophily_level": h,
                "segregation_level": "high",
                "reliance_mode": reliance,
                "sender_regime": sender,
                "citizen_group": group,
                "group_mse_sender": mse,
                "group_mse_null": null,
                "group_delta_mse_vs_null": mse - null,
            }
        )
    return out


def _validate_condition_counts(runs: list[dict], seeds: list[int], experiments: tuple[str, ...]) -> None:
    observed = defaultdict(list)
    for row in runs:
        observed[(row["experiment"], int(row["seed"]))].append(row)
    for experiment in experiments:
        expected = expected_conditions(experiment)
        expected_keys = {
            (
                r["production_block"],
                r["homophily_level"],
                r["segregation_level"],
                r["redundancy_level"],
                r["reliance_mode"],
                r["sender_regime"],
            )
            for r in expected
        }
        for seed in seeds:
            rows = observed.get((experiment, seed), [])
            keys = {
                (
                    r["production_block"],
                    r.get("homophily_level", ""),
                    r.get("segregation_level", ""),
                    r.get("redundancy_level", ""),
                    r["reliance_mode"],
                    r["sender_regime"],
                )
                for r in rows
            }
            if keys != expected_keys:
                raise RuntimeError(
                    f"Incomplete/incorrect {experiment} seed {seed}: "
                    f"expected {len(expected_keys)} conditions, observed {len(keys)}."
                )


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    experiments = resolve_experiments(args.experiments)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if not 0.0 < args.numerical_min_sd < 1.0:
        raise ValueError("--numerical-min-sd must lie in (0,1).")

    plan_path = Path(__file__).resolve().parents[1] / "CANONICAL_PRODUCTION_PLAN.md"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    design = {
        "design_version": 1,
        "purpose": "Paper B canonical 500-seed production",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "plan_sha256": plan_sha,
        "software_versions": software_versions(),
        "seeds": seeds,
        "experiments": list(experiments),
        "tau_social": TAU_SOCIAL,
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
        "expected_runs_per_seed": {
            exp: len(expected_conditions(exp)) for exp in experiments
        },
    }
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id

    run_root = Path(args.output_dir) / f"production_{design_id}"
    shards = run_root / "shards"
    if run_root.exists() and not args.resume:
        raise FileExistsError(
            f"{run_root} exists; use --resume or choose another output directory."
        )
    shards.mkdir(parents=True, exist_ok=True)
    (run_root / "production_manifest.json").write_text(
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
            tasks.append(
                {
                    **design,
                    "experiment": experiment,
                    "seed": int(seed),
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
                        f"[{i}/{len(futures)}] "
                        f"{result['experiment']} seed={result['seed']} "
                        f"runs={result['runs']}",
                        flush=True,
                    )

    payloads = []
    observed_blocks = set()
    for path in sorted(shards.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("design_id") != design_id:
            continue
        if value.get("experiment") not in experiments:
            continue
        payloads.append(value)
        observed_blocks.add((value["experiment"], int(value["seed"])))

    expected_blocks = {(exp, seed) for exp in experiments for seed in seeds}
    if observed_blocks != expected_blocks:
        missing = sorted(expected_blocks - observed_blocks)
        raise RuntimeError(f"Missing completed production blocks: {missing[:20]}")

    runs = [x for p in payloads for x in p["runs"]]
    beliefs = [x for p in payloads for x in p["terminal_beliefs"]]
    lambda_rows = [x for p in payloads for x in p["lambda_checkpoints"]]
    belief_rows = [x for p in payloads for x in p["belief_checkpoints"]]
    jammer_rows = [x for p in payloads for x in p["jammer_strategy"]]

    expected_runs = len(seeds) * sum(
        len(expected_conditions(exp)) for exp in experiments
    )
    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Run-count mismatch: expected {expected_runs}, observed {len(runs)}."
        )
    _validate_condition_counts(runs, seeds, experiments)

    all_finite = all(math.isfinite(float(r["mse_truth"])) for r in runs)
    fixed_horizon = all(int(r["steps_run"]) == int(args.horizon) for r in runs)
    max_mse = max(float(r["mse_truth"]) for r in runs)
    max_abs_belief = max(abs(float(r["terminal_mu_theta"])) for r in beliefs)
    bad_tau = [r for r in runs if float(r.get("tau_social", math.nan)) != TAU_SOCIAL]
    bad_peer = [r for r in runs if r.get("peer_evidence_mode") != "source_posterior"]
    bad_frozen = [r for r in runs if r.get("frozen_ranking_mode") != "pre_disruption"]
    bad_sender_cross = [
        r for r in runs
        if r["sender_regime"] == "adaptive" and r["reliance_mode"] == "frozen"
    ]

    numerical_pass = bool(
        all_finite
        and fixed_horizon
        and max_mse < 1_000_000
        and max_abs_belief < 10_000
        and not bad_tau
        and not bad_peer
        and not bad_frozen
        and not bad_sender_cross
    )

    merged = _merge_checkpoints(belief_rows, lambda_rows)
    run_contrasts = _compute_contrasts_from_index(_run_index(runs), checkpoint=False)
    checkpoint_idx = _checkpoint_index(merged)
    checkpoint_contrasts = _compute_contrasts_from_index(checkpoint_idx, checkpoint=True)
    adaptive_frozen = _adaptive_frozen_contrasts(checkpoint_contrasts + run_contrasts)
    contrast_summary = _contrast_summaries(run_contrasts + adaptive_frozen)
    horizon_contrast_summary = _contrast_summaries(checkpoint_contrasts)
    cells = _cell_summaries(runs)
    checkpoint_cells = _checkpoint_cell_summaries(merged)
    sender_damage = _sender_damage_rows(runs)
    group_sender_damage = _group_sender_damage_rows(beliefs)

    terminal_floor_values = [
        float(r["posterior_sd_floor_share"])
        for r in merged
        if int(r["period"]) == TERMINAL_PERIOD
        and math.isfinite(float(r.get("posterior_sd_floor_share", math.nan)))
    ]
    median_terminal_floor = (
        float(np.median(terminal_floor_values))
        if terminal_floor_values else math.nan
    )

    gate = {
        "pass": numerical_pass,
        "design_id": design_id,
        "observed_blocks": len(observed_blocks),
        "expected_blocks": len(expected_blocks),
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "observed_unique_seeds": len({int(r["seed"]) for r in runs}),
        "expected_unique_seeds": len(seeds),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "all_run_mse_finite": all_finite,
        "all_fixed_horizon": fixed_horizon,
        "max_terminal_mse": max_mse,
        "max_abs_terminal_belief": max_abs_belief,
        "bad_tau_rows": len(bad_tau),
        "bad_peer_evidence_rows": len(bad_peer),
        "bad_frozen_ranking_rows": len(bad_frozen),
        "forbidden_frozen_adaptive_jammer_rows": len(bad_sender_cross),
        "median_terminal_posterior_floor_share": median_terminal_floor,
    }
    (run_root / "production_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )

    _write_csv(run_root / "runs.csv", runs)
    _write_csv(run_root / "terminal_beliefs.csv.gz", beliefs, gzip_output=True)
    _write_csv(run_root / "lambda_checkpoints.csv.gz", lambda_rows, gzip_output=True)
    _write_csv(run_root / "belief_checkpoints.csv.gz", belief_rows, gzip_output=True)
    _write_csv(run_root / "jammer_strategy.csv.gz", jammer_rows, gzip_output=True)
    _write_csv(run_root / "cell_summary.csv", cells)
    _write_csv(run_root / "checkpoint_cell_summary.csv", checkpoint_cells)
    _write_csv(run_root / "seed_contrasts.csv", run_contrasts)
    _write_csv(run_root / "contrast_summary.csv", contrast_summary)
    _write_csv(run_root / "checkpoint_seed_contrasts.csv", checkpoint_contrasts)
    _write_csv(run_root / "horizon_contrast_summary.csv", horizon_contrast_summary)
    _write_csv(run_root / "adaptive_frozen_contrasts.csv", adaptive_frozen)
    _write_csv(run_root / "sender_damage.csv", sender_damage)
    _write_csv(run_root / "group_sender_damage.csv", group_sender_damage)

    review_bundle = run_root / f"paper_b_production_review_{design_id}.zip"
    review_names = (
        "production_manifest.json",
        "production_gate.json",
        "runs.csv",
        "cell_summary.csv",
        "checkpoint_cell_summary.csv",
        "seed_contrasts.csv",
        "contrast_summary.csv",
        "checkpoint_seed_contrasts.csv",
        "horizon_contrast_summary.csv",
        "adaptive_frozen_contrasts.csv",
        "sender_damage.csv",
        "group_sender_damage.csv",
    )
    with zipfile.ZipFile(
        review_bundle, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in review_names:
            path = run_root / name
            if path.exists():
                archive.write(path, arcname=name)
        archive.write(plan_path, arcname="CANONICAL_PRODUCTION_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=True))
    print(f"Review bundle: {review_bundle}")
    print(f"Full raw outputs remain in: {run_root}")


if __name__ == "__main__":
    main()
