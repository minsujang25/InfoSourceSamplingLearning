"""Experiment IIIb: shared bottleneck versus independent corrective paths.

This is a pre-committed diagnostic mechanism experiment. It does not replace
Experiment IIIa. The design holds each focal citizen's immediate relay
neighborhood fixed and rewires only relay->gateway assignments so that the two
corrective paths either share one gateway bottleneck or reach two distinct
gateways. Gateway indegree from relays is identical across conditions.

Default diagnostic:
    50 matched seeds x flat prior
    x 2 path structures (shared, independent)
    x 2 reliance modes (adaptive, frozen)
    x 2 Jammer states
    = 400 simulations at fixed T=200.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import model.InfoSourceSamplingLearning as model_module

from paper_b.experiments.run_local_diagnostic import (
    base_model_config,
    matched_contrasts,
)
from paper_b.experiments.run_matched_pilot import (
    canonical_hash,
    run_condition,
    software_versions,
)
from paper_b.structural_designs import (
    EXPERT_POS,
    JAMMER_POS,
    exp3b_focal_corrective_connectivity,
    exp3b_path_independence_source_maps,
)


PATH_STRUCTURES = ("shared", "independent")
RELIANCE_MODES = ("adaptive", "frozen")
JAMMER_STATES = (True, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="50")
    parser.add_argument("--seed-start", type=int, default=4001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--n-gateways", type=int, default=10)
    parser.add_argument("--n-relays", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument("--numerical-min-sd", type=float, default=1e-8)
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_exp3b_path_independence",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
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
    files = (
        "model/InfoSourceSamplingLearning.py",
        "paper_b/metrics.py",
        "paper_b/structural_designs.py",
        "paper_b/experiments/run_local_diagnostic.py",
        "paper_b/experiments/run_matched_pilot.py",
        "paper_b/experiments/run_exp3b_path_independence.py",
        "environment.yml",
    )
    payload = []
    for name in files:
        path = root / name
        payload.append(
            {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return canonical_hash(payload)


def _elite_signature(source_map: dict[int, list[int]]) -> str:
    payload = {
        int(ego): [
            int(source)
            for source in sources
            if int(source) in {EXPERT_POS, JAMMER_POS}
        ]
        for ego, sources in source_map.items()
    }
    return canonical_hash(payload)


def _source_degree(source_map: dict[int, list[int]]) -> dict[int, int]:
    return {int(ego): len(sources) for ego, sources in source_map.items()}


def _gateway_relay_indegree(
    relay_gateway: dict[int, int],
    gateways: tuple[int, ...],
) -> dict[int, int]:
    out = {int(g): 0 for g in gateways}
    for gateway in relay_gateway.values():
        out[int(gateway)] += 1
    return out


def _subgroup_metrics(beliefs: list[dict], positions: set[int]) -> dict:
    values = np.asarray(
        [
            float(row["terminal_mu_theta"])
            for row in beliefs
            if int(row["citizen_pos"]) in positions
        ],
        dtype=float,
    )
    if values.size == 0:
        raise RuntimeError("Empty Exp IIIb subgroup.")
    errors = values  # Paper B truth is fixed at zero.
    mse = float(np.mean(errors**2))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=0)),
    }


def shard_path(root: Path, block_id: str) -> Path:
    return root / "shards" / f"{block_id}.json.gz"


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def valid_shard(path: Path, design_id: str) -> bool:
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


def run_block(task: dict) -> dict:
    seed = int(task["seed"])
    block_id = task["block_id"]
    design_id = task["design_id"]
    root = Path(task["run_root"])

    numerical_min_sd = float(task["numerical_min_sd"])
    if not 0.0 < numerical_min_sd < 1.0:
        raise ValueError("numerical_min_sd must lie in (0,1).")
    model_module.MIN_SD = numerical_min_sd
    model_module.MIN_VAR = numerical_min_sd**2

    blueprint = exp3b_path_independence_source_maps(
        seed=seed,
        n_citizens=int(task["n_citizens"]),
        n_gateways=int(task["n_gateways"]),
        n_relays=int(task["n_relays"]),
    )
    shared = blueprint["shared"]
    independent = blueprint["independent"]

    # Exact matched-counterfactual gates.
    if _elite_signature(shared) != _elite_signature(independent):
        raise RuntimeError(f"{block_id}: elite access changed across IIIb.")
    if _source_degree(shared) != _source_degree(independent):
        raise RuntimeError(f"{block_id}: per-node source degree changed across IIIb.")

    focals = set(int(x) for x in blueprint["focals"])
    relays = set(int(x) for x in blueprint["relays"])
    gateways = tuple(int(x) for x in blueprint["gateways"])

    for focal in focals:
        if shared[focal] != independent[focal]:
            raise RuntimeError(
                f"{block_id}: focal immediate neighborhood changed for {focal}."
            )

    shared_indegree = _gateway_relay_indegree(
        blueprint["shared_relay_gateway"], gateways
    )
    independent_indegree = _gateway_relay_indegree(
        blueprint["independent_relay_gateway"], gateways
    )
    if shared_indegree != independent_indegree:
        raise RuntimeError(f"{block_id}: gateway relay indegree changed across IIIb.")

    shared_conn = exp3b_focal_corrective_connectivity(
        shared,
        focals=focals,
        relays=relays,
        gateways=gateways,
    )
    independent_conn = exp3b_focal_corrective_connectivity(
        independent,
        focals=focals,
        relays=relays,
        gateways=gateways,
    )
    if any(v != 1 for v in shared_conn.values()):
        raise RuntimeError(f"{block_id}: shared-bottleneck connectivity is not one.")
    if any(v != 2 for v in independent_conn.values()):
        raise RuntimeError(f"{block_id}: independent-path connectivity is not two.")

    base = base_model_config(
        regime="flat",
        seed=seed,
        n_citizens=int(task["n_citizens"]),
        max_steps=int(task["horizon"]),
        k=int(task["k"]),
        epsilon=float(task["epsilon"]),
        credit=int(task["credit"]),
        comparison_rule="delta_comparison",
        surveillance_interval=int(task["surveillance_interval"]),
    )

    results = []
    for structure in PATH_STRUCTURES:
        for reliance in RELIANCE_MODES:
            for jammer_active in JAMMER_STATES:
                cfg = dict(base)
                cfg["structural_source_map"] = blueprint[structure]
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime="flat",
                    environment=f"exp3b_path_{structure}",
                    reliance_mode=reliance,
                    jammer_active=jammer_active,
                    k=int(task["k"]),
                    design_id=design_id,
                    block_id=block_id,
                    save_edge_log=False,
                )
                run = result["run"]
                run["path_structure"] = structure
                run["n_gateways"] = len(gateways)
                run["n_relays"] = len(relays)
                run["n_focals"] = len(focals)

                role_positions = {
                    "focal": focals,
                    "relay": relays,
                    "gateway": set(gateways),
                }
                for role, positions in role_positions.items():
                    metrics = _subgroup_metrics(result["beliefs"], positions)
                    for key, value in metrics.items():
                        run[f"{role}_{key}_truth"] = value

                for row in result["beliefs"]:
                    row["exp3b_role"] = blueprint["roles"][
                        int(row["citizen_pos"])
                    ]
                results.append(result)

    # Same initial state across all eight runs.
    if len({r["run"]["initial_state_fingerprint"] for r in results}) != 1:
        raise RuntimeError(f"{block_id}: initial state changed within IIIb block.")

    # Within a path condition J/reliance must share the exact same structure.
    for structure in PATH_STRUCTURES:
        fps = {
            r["run"]["structural_fingerprint"]
            for r in results
            if r["run"]["path_structure"] == structure
        }
        if len(fps) != 1:
            raise RuntimeError(
                f"{block_id}: structure changed within {structure} condition."
            )

    payload = {
        "complete": True,
        "design_id": design_id,
        "block_id": block_id,
        "seed": seed,
        "roles": blueprint["roles"],
        "gateways": list(gateways),
        "relays": sorted(relays),
        "focals": sorted(focals),
        "focal_to_relays": {
            str(k): list(v) for k, v in blueprint["focal_to_relays"].items()
        },
        "shared_relay_gateway": blueprint["shared_relay_gateway"],
        "independent_relay_gateway": blueprint["independent_relay_gateway"],
        "gateway_relay_indegree": shared_indegree,
        "mean_focal_connectivity_shared": float(
            statistics.fmean(shared_conn.values())
        ),
        "mean_focal_connectivity_independent": float(
            statistics.fmean(independent_conn.values())
        ),
        "runs": [r["run"] for r in results],
        "terminal_beliefs": [x for r in results for x in r["beliefs"]],
        "belief_checkpoints": [
            x for r in results for x in r["belief_checkpoints"]
        ],
        "lambda_checkpoints": [
            x for r in results for x in r["lambda_checkpoints"]
        ],
        "jammer_strategy": [
            x for r in results for x in r["jammer_strategy"]
        ],
    }
    atomic_write(shard_path(root, block_id), payload)
    return {
        "block_id": block_id,
        "max_mse": max(float(r["run"]["mse_truth"]) for r in results),
    }


def _read_shards(root: Path) -> list[dict]:
    out = []
    for path in sorted((root / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def _csv_columns(rows: list[dict]) -> list[str]:
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float:
    return float(statistics.fmean(float(r[key]) for r in rows))


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(float(r[key]) for r in rows))


def _mcse(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows]
    if len(values) < 2:
        return float("nan")
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def _trimmed_mean(rows: list[dict], key: str, proportion: float = 0.05) -> float:
    values = sorted(float(r[key]) for r in rows)
    if not values:
        return float("nan")
    trim = int(math.floor(len(values) * proportion))
    kept = values[trim : len(values) - trim] if trim else values
    return float(statistics.fmean(kept))


def _share_negative(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows]
    if not values:
        return float("nan")
    return float(sum(value < 0.0 for value in values) / len(values))


def _leave_one_out_range(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(r[key]) for r in rows]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    total = sum(values)
    means = [(total - value) / (len(values) - 1) for value in values]
    return float(min(means)), float(max(means))


def consolidate(root: Path, manifest: dict) -> dict:
    shards = _read_shards(root)
    runs = [row for shard in shards for row in shard["runs"]]
    terminal_beliefs = [
        row for shard in shards for row in shard.get("terminal_beliefs", [])
    ]
    belief_checkpoints = [
        row for shard in shards for row in shard.get("belief_checkpoints", [])
    ]
    lambda_checkpoints = [
        row for shard in shards for row in shard.get("lambda_checkpoints", [])
    ]
    jammer_strategy = [
        row for shard in shards for row in shard.get("jammer_strategy", [])
    ]

    jammer_rows, _ = matched_contrasts(runs)
    jammer_index = {
        (
            int(r["seed"]),
            r["network_environment"],
            r["reliance_mode"],
        ): r
        for r in jammer_rows
    }
    run_index = {
        (
            int(r["seed"]),
            r["path_structure"],
            r["reliance_mode"],
            bool(r["jammer_active"]),
        ): r
        for r in runs
    }

    contrasts = []
    for seed in manifest["seeds"]:
        for reliance in RELIANCE_MODES:
            shared = jammer_index[
                (seed, "exp3b_path_shared", reliance)
            ]
            independent = jammer_index[
                (seed, "exp3b_path_independent", reliance)
            ]
            s0 = run_index[(seed, "shared", reliance, False)]
            s1 = run_index[(seed, "shared", reliance, True)]
            i0 = run_index[(seed, "independent", reliance, False)]
            i1 = run_index[(seed, "independent", reliance, True)]

            row = {
                "seed": seed,
                "reliance_mode": reliance,
                "delta_mse_shared": shared["delta_mse"],
                "delta_mse_independent": independent["delta_mse"],
                "independent_minus_shared_delta_mse": (
                    independent["delta_mse"] - shared["delta_mse"]
                ),
                "independent_minus_shared_delta_rmse": (
                    independent["delta_rmse"] - shared["delta_rmse"]
                ),
                "independent_minus_shared_delta_mae": (
                    independent["delta_mae"] - shared["delta_mae"]
                ),
                "independent_minus_shared_mse_J0": (
                    float(i0["mse_truth"]) - float(s0["mse_truth"])
                ),
                "independent_minus_shared_mse_J1": (
                    float(i1["mse_truth"]) - float(s1["mse_truth"])
                ),
                "independent_minus_shared_focal_mse_J0": (
                    float(i0["focal_mse_truth"]) - float(s0["focal_mse_truth"])
                ),
                "independent_minus_shared_focal_mse_J1": (
                    float(i1["focal_mse_truth"]) - float(s1["focal_mse_truth"])
                ),
                "independent_minus_shared_focal_delta_mse": (
                    (float(i1["focal_mse_truth"]) - float(i0["focal_mse_truth"]))
                    - (float(s1["focal_mse_truth"]) - float(s0["focal_mse_truth"]))
                ),
            }
            contrasts.append(row)

    contrast_index = {
        (int(r["seed"]), r["reliance_mode"]): r for r in contrasts
    }
    activation = []
    for seed in manifest["seeds"]:
        adaptive = contrast_index[(seed, "adaptive")]
        frozen = contrast_index[(seed, "frozen")]
        activation.append(
            {
                "seed": seed,
                "adaptive_path_effect": adaptive[
                    "independent_minus_shared_delta_mse"
                ],
                "frozen_path_effect": frozen[
                    "independent_minus_shared_delta_mse"
                ],
                "adaptive_minus_frozen_path_effect": (
                    adaptive["independent_minus_shared_delta_mse"]
                    - frozen["independent_minus_shared_delta_mse"]
                ),
                "adaptive_focal_path_effect": adaptive[
                    "independent_minus_shared_focal_delta_mse"
                ],
                "frozen_focal_path_effect": frozen[
                    "independent_minus_shared_focal_delta_mse"
                ],
            }
        )

    audits = [
        {
            "seed": shard["seed"],
            "n_gateways": len(shard["gateways"]),
            "n_relays": len(shard["relays"]),
            "n_focals": len(shard["focals"]),
            "mean_focal_connectivity_shared": shard[
                "mean_focal_connectivity_shared"
            ],
            "mean_focal_connectivity_independent": shard[
                "mean_focal_connectivity_independent"
            ],
            "gateway_relay_indegree_min": min(
                int(v) for v in shard["gateway_relay_indegree"].values()
            ),
            "gateway_relay_indegree_max": max(
                int(v) for v in shard["gateway_relay_indegree"].values()
            ),
        }
        for shard in shards
    ]

    adaptive = [r for r in contrasts if r["reliance_mode"] == "adaptive"]
    frozen = [r for r in contrasts if r["reliance_mode"] == "frozen"]
    adaptive_loo = _leave_one_out_range(
        adaptive, "independent_minus_shared_delta_mse"
    )
    focal_loo = _leave_one_out_range(
        adaptive, "independent_minus_shared_focal_delta_mse"
    )
    summary = {
        "n_seeds": len(manifest["seeds"]),
        "adaptive_population_path_effect_mean": _mean(
            adaptive, "independent_minus_shared_delta_mse"
        ),
        "adaptive_population_path_effect_median": _median(
            adaptive, "independent_minus_shared_delta_mse"
        ),
        "adaptive_population_path_effect_mcse": _mcse(
            adaptive, "independent_minus_shared_delta_mse"
        ),
        "adaptive_population_path_effect_trimmed_mean_5pct": _trimmed_mean(
            adaptive, "independent_minus_shared_delta_mse", 0.05
        ),
        "adaptive_population_path_effect_share_negative": _share_negative(
            adaptive, "independent_minus_shared_delta_mse"
        ),
        "adaptive_population_path_effect_loo_min": adaptive_loo[0],
        "adaptive_population_path_effect_loo_max": adaptive_loo[1],
        "adaptive_population_J1_difference_mean": _mean(
            adaptive, "independent_minus_shared_mse_J1"
        ),
        "adaptive_population_J0_difference_mean": _mean(
            adaptive, "independent_minus_shared_mse_J0"
        ),
        "adaptive_focal_path_effect_mean": _mean(
            adaptive, "independent_minus_shared_focal_delta_mse"
        ),
        "adaptive_focal_path_effect_median": _median(
            adaptive, "independent_minus_shared_focal_delta_mse"
        ),
        "adaptive_focal_path_effect_mcse": _mcse(
            adaptive, "independent_minus_shared_focal_delta_mse"
        ),
        "adaptive_focal_path_effect_trimmed_mean_5pct": _trimmed_mean(
            adaptive, "independent_minus_shared_focal_delta_mse", 0.05
        ),
        "adaptive_focal_path_effect_share_negative": _share_negative(
            adaptive, "independent_minus_shared_focal_delta_mse"
        ),
        "adaptive_focal_path_effect_loo_min": focal_loo[0],
        "adaptive_focal_path_effect_loo_max": focal_loo[1],
        "adaptive_focal_J1_difference_mean": _mean(
            adaptive, "independent_minus_shared_focal_mse_J1"
        ),
        "adaptive_focal_J0_difference_mean": _mean(
            adaptive, "independent_minus_shared_focal_mse_J0"
        ),
        "frozen_population_path_effect_mean": _mean(
            frozen, "independent_minus_shared_delta_mse"
        ),
        "activation_interaction_mean": _mean(
            activation, "adaptive_minus_frozen_path_effect"
        ),
        "activation_interaction_mcse": _mcse(
            activation, "adaptive_minus_frozen_path_effect"
        ),
    }

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "jammer_contrasts.csv", jammer_rows)
    _write_csv(root / "exp3b_path_contrasts.csv", contrasts)
    _write_csv(root / "exp3b_activation_interaction.csv", activation)
    _write_csv(root / "exp3b_design_audit.csv", audits)
    _write_csv_gz(root / "terminal_beliefs.csv.gz", terminal_beliefs)
    _write_csv(root / "belief_checkpoints.csv", belief_checkpoints)
    _write_csv(root / "lambda_checkpoints.csv", lambda_checkpoints)
    _write_csv_gz(root / "jammer_strategy.csv.gz", jammer_strategy)
    (root / "exp3b_decision_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    expected_blocks = int(manifest["expected_blocks"])
    expected_runs = int(manifest["expected_runs"])
    finite = all(int(r.get("n_nonfinite", 1)) == 0 for r in runs)
    horizon = all(
        int(r["steps_run"]) == int(r["terminal_horizon_T"])
        for r in runs
    )
    frozen_gate = all(
        abs(float(r["lambda_cumulative_turnover"])) <= 1e-12
        for r in runs
        if r["reliance_mode"] == "frozen"
    )
    connectivity_gate = all(
        abs(float(r["mean_focal_connectivity_shared"]) - 1.0) <= 1e-12
        and abs(float(r["mean_focal_connectivity_independent"]) - 2.0) <= 1e-12
        for r in audits
    )
    gateway_balance_gate = all(
        int(r["gateway_relay_indegree_min"])
        == int(r["gateway_relay_indegree_max"])
        for r in audits
    )

    max_mse = max((float(r["mse_truth"]) for r in runs), default=float("nan"))
    max_abs_terminal_belief = max(
        (abs(float(r["terminal_mu_theta"])) for r in terminal_beliefs),
        default=float("nan"),
    )
    min_terminal_sd = min(
        (float(r["terminal_sd_theta"]) for r in terminal_beliefs),
        default=float("nan"),
    )
    numerical_min_sd = float(manifest["numerical_min_sd"])
    terminal_sd_floor_count = sum(
        float(r["terminal_sd_theta"]) <= numerical_min_sd * (1.0 + 1e-12)
        for r in terminal_beliefs
    )
    terminal_sd_floor_share = (
        terminal_sd_floor_count / len(terminal_beliefs)
        if terminal_beliefs else float("nan")
    )
    max_abs_jammer_message = max(
        (abs(float(r["message_mean"])) for r in jammer_strategy),
        default=float("nan"),
    )
    max_individual_gain = max(
        (float(r["response_gain_max"]) for r in jammer_strategy),
        default=float("nan"),
    )
    min_objective_denom = min(
        (float(r["objective_denominator"]) for r in jammer_strategy),
        default=float("nan"),
    )

    report = {
        "pass": (
            len(shards) == expected_blocks
            and len(runs) == expected_runs
            and finite
            and horizon
            and frozen_gate
            and connectivity_gate
            and gateway_balance_gate
        ),
        "expected_blocks": expected_blocks,
        "observed_blocks": len(shards),
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "all_runs_finite": finite,
        "all_runs_fixed_horizon": horizon,
        "frozen_lambda_invariant": frozen_gate,
        "focal_connectivity_gate": connectivity_gate,
        "gateway_indegree_balance_gate": gateway_balance_gate,
        "max_terminal_mse": max_mse,
        "max_abs_terminal_belief": max_abs_terminal_belief,
        "min_terminal_sd_theta": min_terminal_sd,
        "terminal_sd_floor_count": terminal_sd_floor_count,
        "terminal_sd_floor_share": terminal_sd_floor_share,
        "max_abs_jammer_message_mean": max_abs_jammer_message,
        "max_individual_response_gain": max_individual_gain,
        "min_jammer_objective_denominator": min_objective_denom,
    }
    (root / "exp3b_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def package(root: Path, design_id: str) -> Path:
    import zipfile

    names = (
        "exp3b_manifest.json",
        "runs.csv",
        "jammer_contrasts.csv",
        "exp3b_path_contrasts.csv",
        "exp3b_activation_interaction.csv",
        "exp3b_design_audit.csv",
        "exp3b_decision_metrics.json",
        "terminal_beliefs.csv.gz",
        "belief_checkpoints.csv",
        "lambda_checkpoints.csv",
        "jammer_strategy.csv.gz",
        "exp3b_gate.json",
    )
    path = root.parent / f"paper_b_exp3b_{design_id}_shareable.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            p = root / name
            if p.exists():
                archive.write(p, arcname=name)
    return path


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    n_focals = (
        int(args.n_citizens) - int(args.n_gateways) - int(args.n_relays)
    )
    if n_focals <= 0:
        raise ValueError("Exp IIIb requires at least one focal citizen.")

    design = {
        "design_version": 1,
        "experiment": "IIIb_path_independence_diagnostic",
        "status": "diagnostic_not_precommitted_for_manuscript_inclusion",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "software_versions": software_versions(),
        "seeds": seeds,
        "initial_regime": "flat",
        "n_citizens": args.n_citizens,
        "n_gateways": args.n_gateways,
        "n_relays": args.n_relays,
        "n_focals": n_focals,
        "horizon_T": args.horizon,
        "K": args.k,
        "epsilon": args.epsilon,
        "credit": args.credit,
        "surveillance_interval": args.surveillance_interval,
        "numerical_min_sd": args.numerical_min_sd,
        "path_structures": list(PATH_STRUCTURES),
        "reliance_modes": list(RELIANCE_MODES),
        "jammer_states": [True, False],
        "manipulation": (
            "same focal->relay edges and same gateway relay-indegree; "
            "relay->gateway rewiring changes shared bottleneck (1) to "
            "two internally vertex-disjoint corrective routes (2)"
        ),
    }
    design_id = canonical_hash(design)[:12]
    blocks = [
        {"seed": seed, "block_id": f"exp3b__s{seed}"}
        for seed in seeds
    ]
    design.update(
        {
            "design_id": design_id,
            "expected_blocks": len(blocks),
            "expected_runs": len(blocks) * 8,
            "block_ids": [b["block_id"] for b in blocks],
        }
    )

    root = Path(args.output_dir) / f"exp3b_{design_id}"
    manifest_path = root / "exp3b_manifest.json"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design_id") != design_id:
            raise RuntimeError("Existing Exp IIIb manifest does not match.")
    else:
        manifest_path.write_text(
            json.dumps(design, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    pending = []
    skipped = 0
    for block in blocks:
        path = shard_path(root, block["block_id"])
        if args.resume and valid_shard(path, design_id):
            skipped += 1
            continue
        pending.append(
            {
                **block,
                "design_id": design_id,
                "run_root": str(root),
                "n_citizens": args.n_citizens,
                "n_gateways": args.n_gateways,
                "n_relays": args.n_relays,
                "horizon": args.horizon,
                "k": args.k,
                "epsilon": args.epsilon,
                "credit": args.credit,
                "surveillance_interval": args.surveillance_interval,
                "numerical_min_sd": args.numerical_min_sd,
            }
        )

    print("Paper B Experiment IIIb: path independence diagnostic")
    print(f"  design_id      : {design_id}")
    print(f"  blocks         : {len(blocks)}")
    print(f"  runs           : {len(blocks) * 8}")
    print(f"  pending        : {len(pending)}")
    print(f"  skipped        : {skipped}")
    print(f"  workers        : {args.workers}")

    if pending:
        ctx = mp.get_context("spawn")
        completed = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
        ) as pool:
            futures = {pool.submit(run_block, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if completed % max(args.progress_every, 1) == 0:
                    print(
                        f"[{completed}/{len(pending)}] "
                        f"{result['block_id']} | "
                        f"max MSE={result['max_mse']:.4f}"
                    )

    report = consolidate(root, design)
    print(f"Exp IIIb gate: {'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(1)
    bundle = package(root, design_id)
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
