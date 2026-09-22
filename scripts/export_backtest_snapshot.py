"""Export a compact matched-seed snapshot for Mesa migration backtesting.

The harness is deliberately version-agnostic: point --model-root at either the
legacy Mesa 2.4 checkout or the migrated Mesa 3.5 checkout. The same scenario
definitions and seeds are used in both environments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCENARIOS = [
    {
        "id": "baseline_flat_k1_n0_delta",
        "mode": "baseline",
        "initial": "flat",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 0,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_flat_k1_n2_delta",
        "mode": "random",
        "initial": "flat",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_flat_k4_n5_delta",
        "mode": "random",
        "initial": "flat",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 5,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "group_flat_k1_n2_delta",
        "mode": "group_id_matching",
        "initial": "flat",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "group_polarized_k4_n5_delta",
        "mode": "group_id_matching",
        "initial": "polarized",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 5,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_consensus_k4_n2_delta",
        "mode": "random",
        "initial": "consensus",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_polarized_k1_n2_z",
        "mode": "random",
        "initial": "polarized",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "z_stat_comparison",
    },
    {
        "id": "group_consensus_k4_n2_z",
        "mode": "group_id_matching",
        "initial": "consensus",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "z_stat_comparison",
    },
]

DEFAULT_SEEDS = [101, 202, 303]
N = 42
MAX_STEPS = 40


def load_model_module(model_root: Path):
    model_file = model_root / "model" / "InfoSourceSamplingLearning.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    spec = importlib.util.spec_from_file_location("issl_backtest_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initial_beliefs(kind: str, seed: int) -> list[float]:
    rng = np.random.default_rng(seed + 10_000)
    n_citizens = N - 2
    if kind == "flat":
        citizens = rng.uniform(-5.0, 5.0, n_citizens)
    elif kind == "polarized":
        half = n_citizens // 2
        citizens = np.concatenate(
            [rng.normal(-3.0, 1.0, half), rng.normal(3.0, 1.0, n_citizens - half)]
        )
    elif kind == "consensus":
        citizens = rng.normal(0.0, 1.0, n_citizens)
    else:
        raise ValueError(f"Unknown initial condition: {kind}")
    return [0.0, 4.0] + citizens.tolist()


def build_config(module, scenario: dict, seed: int) -> dict:
    mu_theta = initial_beliefs(scenario["initial"], seed)
    return {
        "state_of_the_world": 0.0,
        "num_nodes": N,
        "comparison_rule": scenario["comparison_rule"],
        "epsilon": 0.05,
        "credit": 20,
        "mu_delta": [[0.0] * 8 for _ in range(N)],
        "sd_delta": [[5.0] * 8 for _ in range(N)],
        "mu_theta": mu_theta,
        "sd_theta": [1.0, 1.0] + [5.0] * (N - 2),
        "initial_theta_type": scenario["initial"],
        "seq_meaningful": True,
        "type_of_agent": [module.InfoProvider, module.DisruptiveJammer]
        + [module.Citizen] * (N - 2),
        "max_steps": MAX_STEPS,
        "network_type": "fully_connected",
        "mode": scenario["mode"],
        "network_structure": None,
        "learn_method": "cautious",
        "counterpart_pick_mechanism": "equal",
        "surveil_ability": scenario["surveil_ability"],
        "num_max_citizen_neighbor": scenario["num_max_citizen_neighbor"],
    }


def agents_of(model):
    if hasattr(model, "schedule"):
        return list(model.schedule.agents)
    return list(model.agents)


def model_steps(model) -> int:
    if hasattr(model, "schedule"):
        return int(model.schedule.steps)
    return int(model.steps)


def canonical_clusters(model) -> list[list[int]]:
    agents = agents_of(model)
    jammers = [a for a in agents if a.type_of_agent == "disruptivejammer"]
    if not jammers:
        return []
    clusters = getattr(jammers[0], "citizen_per_cl", {})
    canonical = []
    for citizens in clusters.values():
        canonical.append(sorted(int(c.pos) for c in citizens))
    return sorted(canonical)


def citizen_final_by_pos(model) -> list[dict]:
    citizens = sorted(
        [a for a in agents_of(model) if a.type_of_agent == "citizen"],
        key=lambda a: int(a.pos),
    )
    return [
        {
            "pos": int(a.pos),
            "mu_theta": float(a.mu_theta_beliefs[-1]),
            "sd_theta": float(a.sd_theta_beliefs[-1]),
        }
        for a in citizens
    ]


def run_one(module, scenario: dict, seed: int) -> dict:
    np.random.seed(seed)
    config = build_config(module, scenario, seed)

    try:
        model = module.InfoSampleModel(model_attribute=config, rng=seed)
    except TypeError:
        model = module.InfoSampleModel(model_attribute=config)

    model.run_model(print_time=False)

    trajectory = [
        [float(x) for x in row]
        for row in getattr(model, "agent_mu_theta_list", [])
    ]
    edges = sorted(
        [[int(u), int(v)] for u, v in model.network.edges()],
        key=lambda e: (e[0], e[1]),
    )
    final = citizen_final_by_pos(model)
    final_mu = np.array([row["mu_theta"] for row in final], dtype=float)

    return {
        "scenario_id": scenario["id"],
        "seed": seed,
        "steps": model_steps(model),
        "network_edges": edges,
        "jammer_clusters": canonical_clusters(model),
        "trajectory": trajectory,
        "final_by_pos": final,
        "summary": {
            "mean_final": float(np.mean(final_mu)),
            "sd_final": float(np.std(final_mu)),
            "mae_truth": float(np.mean(np.abs(final_mu))),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    module = load_model_module(args.model_root.resolve())
    runs = []
    for scenario in SCENARIOS:
        for seed in args.seeds:
            runs.append(run_one(module, scenario, seed))

    payload = {
        "schema_version": 1,
        "n": N,
        "max_steps": MAX_STEPS,
        "seeds": args.seeds,
        "scenarios": SCENARIOS,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs)} matched backtest runs to {args.output}")


if __name__ == "__main__":
    main()
