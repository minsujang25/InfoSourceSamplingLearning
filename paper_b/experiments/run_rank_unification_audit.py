"""Targeted audit unifying rank competition across Experiment 2 and fixed-biased stress.

Blocks:
A. Experiment 2 null: low/high corrective-route multiplicity x adaptive/frozen.
B. Experiment 1 fixed biased: low/high H x adaptive/frozen, high segregation.

500 matched seeds each. Total = 4,000 runs.
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
from model.InfoSourceSamplingLearning import recursive_rank_probabilities
from paper_b.experiments.run_canonical_production import _base_config
from paper_b.experiments.run_matched_pilot import canonical_hash, run_condition, software_versions
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp3_redundancy_source_maps,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)
from paper_b.experiments.run_local_diagnostic import initial_beliefs


REFERENCE_MEASUREMENT_ID = "087afed31ccf"
RELIANCE_MODES = ("adaptive", "frozen")
REDUNDANCY_LEVELS = ("low", "high")
H_LEVELS = ("low", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--expert-access-share", type=float, default=0.10)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument("--numerical-min-sd", type=float, default=1e-8)
    parser.add_argument(
        "--measurement-root",
        default=(
            "production_results/paper_b_measurement_audit/"
            "measurement_087afed31ccf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_rank_unification",
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
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scientific_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/measurement.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_rank_unification_audit.py",
        "paper_b/RANK_COMPETITION_UNIFICATION_PLAN.md",
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


def _decorate(result: dict, **labels) -> None:
    result["run"].update(labels)
    for key in (
        "beliefs",
        "lambda_checkpoints",
        "belief_checkpoints",
        "evidence_checkpoints",
        "channel_checkpoints",
        "channel_group_checkpoints",
        "channel_citizens",
        "rank_unification_checkpoints",
    ):
        for row in result[key]:
            row.update(labels)


def _run_seed(task: dict) -> dict:
    model_module.MIN_SD = float(task["numerical_min_sd"])
    model_module.MIN_VAR = float(task["numerical_min_sd"]) ** 2

    seed = int(task["seed"])
    n = int(task["n_citizens"])
    results = []

    # Experiment 2 / historical III: null sender.
    exp2 = exp3_redundancy_source_maps(
        seed=seed,
        n_citizens=n,
        peer_degree=int(task["peer_degree"]),
        expert_access_share=float(task["expert_access_share"]),
    )
    gateways = set(int(x) for x in exp2["expert_gateways"])

    for multiplicity in REDUNDANCY_LEVELS:
        for reliance in RELIANCE_MODES:
            cfg = _base_config(seed, task)
            cfg["structural_source_map"] = exp2[multiplicity]
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime="flat",
                environment=f"rank_exp2_{multiplicity}",
                reliance_mode=reliance,
                jammer_active=False,
                jammer_regime="null",
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                gateway_positions=gateways,
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"rank_exp2__s{seed}",
                save_edge_log=False,
                record_rank_unification_checkpoints=True,
            )
            _decorate(
                result,
                audit_block="exp2_gateway_rank",
                experiment="III",
                production_block="redundancy_primary",
                redundancy_level=multiplicity,
                homophily_level="",
                segregation_level="",
                sender_regime="null",
            )
            results.append(result)

    # Experiment 1 fixed-biased stress / historical IV.
    groups = balanced_fixed_group_ids(seed=seed, n_citizens=n)
    exp1 = exp4_homophily_source_maps(
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

    for h in H_LEVELS:
        for reliance in RELIANCE_MODES:
            cfg = _base_config(seed, task)
            cfg.update(
                {
                    "mu_theta": initial,
                    "initial_theta_type": "production_exp4_high",
                    "fixed_group_ids": groups,
                    "structural_source_map": exp1[h],
                }
            )
            result = run_condition(
                base_config=cfg,
                seed=seed,
                regime="segregation_high",
                environment=f"rank_fixed_h_{h}",
                reliance_mode=reliance,
                jammer_active=False,
                jammer_regime="fixed_biased",
                peer_evidence_mode="source_posterior",
                frozen_ranking_mode="pre_disruption",
                k=int(task["K"]),
                design_id=str(task["design_id"]),
                block_id=f"rank_fixed__s{seed}",
                save_edge_log=False,
                record_channel_decomposition=True,
                record_rank_unification_checkpoints=True,
            )
            _decorate(
                result,
                audit_block="fixed_biased_crowding_in",
                experiment="IV",
                production_block="fixed_biased_highS",
                redundancy_level="",
                homophily_level=h,
                segregation_level="high",
                sender_regime="fixed_biased",
            )
            results.append(result)

    payload = {
        "complete": True,
        "design_id": str(task["design_id"]),
        "seed": seed,
        "runs": [r["run"] for r in results],
        "beliefs": [x for r in results for x in r["beliefs"]],
        "rank_unification_checkpoints": [
            x for r in results for x in r["rank_unification_checkpoints"]
        ],
        "channel_group_checkpoints": [
            x for r in results for x in r["channel_group_checkpoints"]
        ],
        "channel_citizens": [
            x for r in results for x in r["channel_citizens"]
        ],
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
            and len(value.get("runs", [])) == 8
        )
    except Exception:
        return False


def _target_key(row: dict) -> tuple:
    return (
        row["experiment"],
        int(float(row["seed"])),
        row["production_block"],
        row.get("redundancy_level", ""),
        row.get("homophily_level", ""),
        row["reliance_mode"],
        row["sender_regime"],
    )


def _reference_rows(root: Path) -> dict:
    path = root / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    out = {}
    for row in _read_csv(path):
        keep = False
        if (
            row.get("experiment") == "III"
            and row.get("production_block") == "redundancy_primary"
            and row.get("sender_regime") == "null"
        ):
            keep = True
        if (
            row.get("experiment") == "IV"
            and row.get("production_block") == "fixed_biased_highS"
            and row.get("sender_regime") == "fixed_biased"
        ):
            keep = True
        if keep:
            out[_target_key(row)] = row
    return out


def _identity_gate(target_rows: list[dict], measurement_root: Path) -> dict:
    reference = _reference_rows(measurement_root)
    target = {_target_key(row): row for row in target_rows}
    missing = set(reference) ^ set(target)
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
        "expert_reliance",
        "jammer_reliance",
        "peer_reliance",
        "effective_homophily",
        "W_expert_precision_share",
        "W_jammer_precision_share",
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

    return {
        "pass": bool(fingerprints_match and max_diff <= 1e-12),
        "tolerance": 1e-12,
        "key_mismatch_count": 0,
        "fingerprints_match": fingerprints_match,
        "max_abs_diff": max_diff,
        "numeric_comparisons": comparisons,
    }


def _deterministic_reconstruction(args, seeds: list[int]) -> list[dict]:
    rows = []
    probs3 = recursive_rank_probabilities(3, float(args.epsilon))

    for seed in seeds:
        n = int(args.n_citizens)

        exp2 = exp3_redundancy_source_maps(
            seed=seed,
            n_citizens=n,
            peer_degree=int(args.peer_degree),
            expert_access_share=float(args.expert_access_share),
        )
        gateways = set(int(x) for x in exp2["expert_gateways"])
        initial = initial_beliefs(
            regime="flat",
            seed=seed,
            n_citizens=n,
        )

        for multiplicity in REDUNDANCY_LEVELS:
            for ego, sources in exp2[multiplicity].items():
                if int(ego) in gateways:
                    continue
                peer_sources = [
                    int(source)
                    for source in sources
                    if int(source) >= 2
                ]
                ranked = sorted(
                    peer_sources,
                    key=lambda source: (
                        abs(float(initial[source]) - float(initial[int(ego)])),
                        source,
                    ),
                )
                gateway_ranks = [
                    idx + 1
                    for idx, source in enumerate(ranked)
                    if source in gateways
                ]
                mass = float(sum(probs3[rank - 1] for rank in gateway_ranks))
                rows.append(
                    {
                        "audit_block": "exp2_gateway_rank",
                        "seed": seed,
                        "condition": multiplicity,
                        "citizen_group": "",
                        "top_target_share": int(ranked[0] in gateways),
                        "best_target_rank": min(gateway_ranks),
                        "target_acquisition_mass": mass,
                        "target_inclusion_probability": float(
                            1.0 - (1.0 - mass) ** int(args.credit)
                        ),
                    }
                )

        groups = balanced_fixed_group_ids(seed=seed, n_citizens=n)
        maps = exp4_homophily_source_maps(
            seed=seed,
            n_citizens=n,
            group_ids=groups,
            peer_degree=int(args.peer_degree),
            low_homophily=float(args.low_homophily),
            high_homophily=float(args.high_homophily),
        )
        initial4 = exp4_initial_beliefs(
            seed=seed,
            n_citizens=n,
            group_ids=groups,
            segregation="high",
            high_group_shift=float(args.high_group_shift),
            residual_sd=float(args.prior_residual_sd),
        )

        for h in H_LEVELS:
            for ego, sources in maps[h].items():
                ranking = sorted(
                    [int(source) for source in sources],
                    key=lambda source: (
                        abs(float(initial4[source]) - float(initial4[int(ego)])),
                        source,
                    ),
                )
                jammer_rank = ranking.index(1) + 1
                probs = recursive_rank_probabilities(
                    len(ranking),
                    float(args.epsilon),
                )
                rows.append(
                    {
                        "audit_block": "fixed_biased_crowding_in",
                        "seed": seed,
                        "condition": h,
                        "citizen_group": int(groups[int(ego)]),
                        "top_target_share": int(jammer_rank == 1),
                        "best_target_rank": jammer_rank,
                        "target_acquisition_mass": float(
                            probs[jammer_rank - 1]
                        ),
                        "target_inclusion_probability": float(
                            1.0
                            - (1.0 - probs[jammer_rank - 1])
                            ** int(args.credit)
                        ),
                    }
                )
    return rows


def _summary(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    out = []
    for key, members in sorted(grouped.items(), key=lambda x: tuple(map(str, x[0]))):
        item = {name: value for name, value in zip(keys, key)}
        item["n"] = len(members)
        for metric in metrics:
            vals = []
            for member in members:
                value = member.get(metric, math.nan)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    vals.append(value)
            item[f"{metric}_mean"] = (
                float(np.mean(vals)) if vals else math.nan
            )
        out.append(item)
    return out


def _group_terminal_loss(beliefs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in beliefs:
        if row.get("audit_block") != "fixed_biased_crowding_in":
            continue
        grouped[
            (
                row["homophily_level"],
                row["reliance_mode"],
                int(row["citizen_group"]),
            )
        ].append(float(row["terminal_mu_theta"]))

    out = []
    for (h, reliance, group), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        out.append(
            {
                "homophily_level": h,
                "reliance_mode": reliance,
                "citizen_group": group,
                "n_citizen_seed": int(arr.size),
                "terminal_mse": float(np.mean(arr ** 2)),
                "terminal_mean_belief": float(np.mean(arr)),
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
        and math.isclose(float(args.epsilon), 0.05)
        and int(args.k) == 1
        and int(args.surveillance_interval) == 5
        and int(args.peer_degree) == 2
        and math.isclose(float(args.expert_access_share), 0.10)
        and math.isclose(float(args.low_homophily), 0.50)
        and math.isclose(float(args.high_homophily), 0.90)
        and math.isclose(float(args.high_group_shift), 3.0)
        and math.isclose(float(args.prior_residual_sd), 1.0)
        and math.isclose(float(args.numerical_min_sd), 1e-8)
    )

    design = {
        "purpose": "Paper B rank-competition unification audit",
        "reference_measurement_id": REFERENCE_MEASUREMENT_ID,
        "seeds": seeds,
        "n_citizens": int(args.n_citizens),
        "horizon_T": int(args.horizon),
        "epsilon": float(args.epsilon),
        "credit": int(args.credit),
        "K": int(args.k),
        "surveillance_interval": int(args.surveillance_interval),
        "peer_degree": int(args.peer_degree),
        "expert_access_share": float(args.expert_access_share),
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

    root = Path(args.output_dir) / f"rank_{design_id}"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / "rank_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    deterministic = _deterministic_reconstruction(args, seeds)
    _write_csv(root / "initial_rank_reconstruction.csv", deterministic)

    tasks = []
    for seed in seeds:
        path = root / "shards" / f"s{seed}.json.gz"
        if args.resume and _valid_shard(path, design_id):
            continue
        tasks.append(
            {
                "design_id": design_id,
                "run_root": str(root),
                "seed": seed,
                "n_citizens": int(args.n_citizens),
                "horizon_T": int(args.horizon),
                "epsilon": float(args.epsilon),
                "credit": int(args.credit),
                "K": int(args.k),
                "surveillance_interval": int(args.surveillance_interval),
                "peer_degree": int(args.peer_degree),
                "expert_access_share": float(args.expert_access_share),
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
            futures = [pool.submit(_run_seed, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if i % max(int(args.progress_every), 1) == 0:
                    print(
                        f"[{i}/{len(futures)}] seed={result['seed']}",
                        flush=True,
                    )

    payloads = []
    for path in sorted((root / "shards").glob("s*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("design_id") == design_id:
            payloads.append(payload)

    expected_shards = len(seeds)
    expected_runs = 8 * len(seeds)
    if len(payloads) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} shards, got {len(payloads)}."
        )

    runs = [x for p in payloads for x in p["runs"]]
    beliefs = [x for p in payloads for x in p["beliefs"]]
    ranks = [x for p in payloads for x in p["rank_unification_checkpoints"]]
    channel_groups = [
        x for p in payloads for x in p["channel_group_checkpoints"]
    ]
    channel_citizens = [
        x for p in payloads for x in p["channel_citizens"]
    ]

    identity = None
    if not args.skip_reference_gate:
        identity = _identity_gate(
            runs,
            Path(args.measurement_root),
        )

    finite = all(math.isfinite(float(row["mse_truth"])) for row in runs)
    gate = {
        "pass": bool(
            canonical_match
            and len(runs) == expected_runs
            and finite
            and (identity is None or identity["pass"])
        ),
        "design_id": design_id,
        "canonical_base_match": canonical_match,
        "observed_shards": len(payloads),
        "expected_shards": expected_shards,
        "observed_runs": len(runs),
        "expected_runs": expected_runs,
        "all_terminal_mse_finite": finite,
        "reference_gate_skipped": bool(args.skip_reference_gate),
        "behavioral_identity": identity,
    }
    (root / "rank_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )

    initial_summary = _summary(
        deterministic,
        ("audit_block", "condition", "citizen_group"),
        (
            "top_target_share",
            "best_target_rank",
            "target_acquisition_mass",
            "target_inclusion_probability",
        ),
    )
    rank_summary = _summary(
        ranks,
        (
            "audit_block",
            "redundancy_level",
            "homophily_level",
            "reliance_mode",
            "diagnostic_type",
            "citizen_group",
            "period",
        ),
        (
            "top_peer_gateway_share",
            "mean_best_gateway_rank",
            "mean_gateway_acquisition_mass",
            "mean_gateway_inclusion_probability",
            "mean_jammer_rank",
            "mean_jammer_acquisition_probability",
            "jammer_rank_1_share",
            "jammer_rank_2_share",
            "jammer_rank_3_share",
            "jammer_rank_4_share",
        ),
    )
    group_channel_summary = _summary(
        channel_groups,
        (
            "homophily_level",
            "reliance_mode",
            "citizen_group",
            "horizon_step",
        ),
        (
            "W_jammer",
            "I_jammer",
            "Q_jammer",
            "W_expert",
            "I_expert",
            "Q_expert",
            "W_same_peer",
            "W_other_peer",
        ),
    )
    group_loss = _group_terminal_loss(beliefs)

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "initial_rank_summary.csv", initial_summary)
    _write_csv(root / "rank_checkpoint_summary.csv", rank_summary)
    _write_csv(root / "fixed_group_channel_summary.csv", group_channel_summary)
    _write_csv(root / "fixed_group_terminal_loss.csv", group_loss)
    _write_csv(
        root / "rank_unification_checkpoints.csv.gz",
        ranks,
        gzip_output=True,
    )
    _write_csv(
        root / "channel_group_checkpoints.csv.gz",
        channel_groups,
        gzip_output=True,
    )

    plan = Path(__file__).resolve().parents[1] / "RANK_COMPETITION_UNIFICATION_PLAN.md"
    review = root / f"paper_b_rank_unification_review_{design_id}.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "rank_manifest.json",
            "rank_gate.json",
            "initial_rank_summary.csv",
            "rank_checkpoint_summary.csv",
            "fixed_group_channel_summary.csv",
            "fixed_group_terminal_loss.csv",
        ):
            zf.write(root / name, arcname=name)
        zf.write(plan, arcname="RANK_COMPETITION_UNIFICATION_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=True))
    print(f"Review bundle: {review}")


if __name__ == "__main__":
    main()
