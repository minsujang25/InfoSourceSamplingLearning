"""Local matched diagnostic runner for Social Networks Paper B.

Default design:
    4 network environments
    x 2 jammer states (J=1/J=0)
    x 2 reliance modes (adaptive/frozen)
    x 5 matched seeds
    x 1 initial-belief regime (flat)

The runner writes compact CSV/JSON outputs and then bundles them into a ZIP
that can be uploaded back to ChatGPT for analysis.

Run from the repository root, for example:

    python -m paper_b.experiments.run_local_diagnostic

A larger diagnostic can be requested with:

    python -m paper_b.experiments.run_local_diagnostic \
        --seeds 20 \
        --initial-regimes flat,consensus,polarized \
        --n-citizens 100 \
        --max-steps 200 \
        --save-edge-log
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.InfoSourceSamplingLearning import (
    Citizen,
    DisruptiveJammer,
    InfoProvider,
    InfoSampleModel,
)
from paper_b.metrics import (
    realized_flow_composition,
    reliance_hhi,
    theory_metrics,
)
from paper_b.validation import assert_finite_state


NETWORK_ENVIRONMENTS = ("elite_only", "random_2", "group_id", "extended")
RELIANCE_MODES = ("adaptive", "frozen")
JAMMER_STATES = (True, False)
REGIME_OFFSETS = {
    "flat": 10_000,
    "consensus": 20_000,
    "polarized": 30_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched local diagnostic grid for Paper B."
    )
    parser.add_argument(
        "--seeds",
        default="5",
        help=(
            "Either a positive integer N (uses seeds 1001..1000+N) or a "
            "comma-separated explicit seed list, e.g. 1001,1002,1003."
        ),
    )
    parser.add_argument(
        "--initial-regimes",
        default="flat",
        help="Comma-separated subset of flat,consensus,polarized.",
    )
    parser.add_argument(
        "--n-citizens",
        type=int,
        default=100,
        help="Number of citizen agents; two elite agents are added automatically.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum simulation periods per run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Jammer belief-surveillance resolution K.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Rank-based exploration parameter in [0, 0.5].",
    )
    parser.add_argument(
        "--credit",
        type=int,
        default=20,
        help="Per-period information-acquisition budget.",
    )
    parser.add_argument(
        "--comparison-rule",
        choices=("delta_comparison", "z_stat_comparison"),
        default="delta_comparison",
    )
    parser.add_argument(
        "--surveillance-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_diagnostic",
        help="Parent directory for timestamped diagnostic outputs.",
    )
    parser.add_argument(
        "--save-edge-log",
        action="store_true",
        help=(
            "Save Lambda/X edge records at periods 0,1,5,10 and final. "
            "Useful for mechanism inspection; off by default for speed/size."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed runs.",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    value = value.strip()
    if "," in value:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        n = int(value)
        if n <= 0:
            raise ValueError("--seeds must be positive.")
        seeds = list(range(1001, 1001 + n))
    if not seeds:
        raise ValueError("No seeds resolved.")
    return seeds


def parse_regimes(value: str) -> list[str]:
    regimes = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = [r for r in regimes if r not in REGIME_OFFSETS]
    if invalid:
        raise ValueError(f"Unknown initial regime(s): {invalid}")
    if not regimes:
        raise ValueError("At least one initial regime is required.")
    return regimes


def initial_beliefs(
    *,
    regime: str,
    seed: int,
    n_citizens: int,
) -> list[float]:
    rng = np.random.default_rng(seed + REGIME_OFFSETS[regime])

    if regime == "flat":
        citizens = rng.uniform(-5.0, 5.0, n_citizens)
    elif regime == "consensus":
        citizens = rng.normal(0.0, 1.0, n_citizens)
    elif regime == "polarized":
        n_left = n_citizens // 2
        n_right = n_citizens - n_left
        citizens = np.concatenate(
            (
                rng.normal(-3.0, 1.0, n_left),
                rng.normal(3.0, 1.0, n_right),
            )
        )
    else:  # pragma: no cover - guarded by parse_regimes
        raise ValueError(regime)

    # Node 0 = Expert centered on truth; node 1 = Jammer underlying position.
    return [0.0, 4.0] + [float(x) for x in citizens]


def base_model_config(
    *,
    regime: str,
    seed: int,
    n_citizens: int,
    max_steps: int,
    k: int,
    epsilon: float,
    credit: int,
    comparison_rule: str,
    surveillance_interval: int,
) -> dict:
    n = n_citizens + 2
    mu_theta = initial_beliefs(
        regime=regime,
        seed=seed,
        n_citizens=n_citizens,
    )

    return {
        "state_of_the_world": 0.0,
        "num_nodes": n,
        "comparison_rule": comparison_rule,
        "epsilon": epsilon,
        "credit": credit,
        "mu_delta": [[0.0] * 4 for _ in range(n)],
        "sd_delta": [[5.0] * 4 for _ in range(n)],
        "mu_theta": mu_theta,
        "sd_theta": [1.0, 1.0] + [5.0] * n_citizens,
        "initial_theta_type": regime,
        "seq_meaningful": True,
        "type_of_agent": [InfoProvider, DisruptiveJammer] + [Citizen] * n_citizens,
        "max_steps": max_steps,
        "network_type": "fully_connected",
        "mode": "baseline",
        "learn_method": "cautious",
        "counterpart_pick_mechanism": "equal",
        "surveil_ability": k,
        "surveillance_interval": surveillance_interval,
        "local_degree": 2,
        "peer_degree": 2,
        "same_group_probability": 0.9,
        "num_max_citizen_neighbor": 2,
    }


def topology_summary(model: InfoSampleModel) -> dict:
    degrees = []
    elite_counts = []
    peer_counts = []
    same = 0
    peer_edges = 0

    for citizen in model.citizens:
        sources = citizen.info_source
        degrees.append(len(sources))
        elite_counts.append(
            sum(
                source.type_of_agent in {"infoprovider", "disruptivejammer"}
                for source in sources
            )
        )
        peers = [
            source for source in sources
            if source.type_of_agent == "citizen"
        ]
        peer_counts.append(len(peers))
        for source in peers:
            same += int(source.group_id == citizen.group_id)
            peer_edges += 1

    return {
        "mean_structural_degree": float(np.mean(degrees)),
        "min_structural_degree": int(min(degrees)),
        "max_structural_degree": int(max(degrees)),
        "mean_elite_sources": float(np.mean(elite_counts)),
        "mean_peer_sources": float(np.mean(peer_counts)),
        "share_citizens_with_expert": float(
            np.mean(
                [
                    any(s.type_of_agent == "infoprovider" for s in c.info_source)
                    for c in model.citizens
                ]
            )
        ),
        "share_citizens_with_jammer": float(
            np.mean(
                [
                    any(s.type_of_agent == "disruptivejammer" for s in c.info_source)
                    for c in model.citizens
                ]
            )
        ),
        "structural_peer_same_group_share": (
            float(same / peer_edges) if peer_edges else math.nan
        ),
    }


def git_metadata() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def matched_contrasts(run_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key = {}
    for row in run_rows:
        key = (
            row["seed"],
            row["initial_regime"],
            row["network_environment"],
            row["reliance_mode"],
            row["jammer_active"],
            row["K"],
        )
        by_key[key] = row

    jammer_contrasts = []
    for seed, regime, env, reliance, k_active, k in list(by_key):
        if not k_active:
            continue
        j1 = by_key[(seed, regime, env, reliance, True, k)]
        j0 = by_key.get((seed, regime, env, reliance, False, k))
        if j0 is None:
            continue
        jammer_contrasts.append(
            {
                "seed": seed,
                "initial_regime": regime,
                "network_environment": env,
                "reliance_mode": reliance,
                "K": k,
                "delta_mse": j1["mse_truth"] - j0["mse_truth"],
                "delta_rmse": j1["rmse_truth"] - j0["rmse_truth"],
                "delta_mae": j1["mae_truth"] - j0["mae_truth"],
                "delta_squared_displacement": (
                    j1["squared_displacement"] - j0["squared_displacement"]
                ),
                "delta_belief_variance": (
                    j1["belief_variance"] - j0["belief_variance"]
                ),
            }
        )

    jammer_index = {
        (
            r["seed"],
            r["initial_regime"],
            r["network_environment"],
            r["reliance_mode"],
            r["K"],
        ): r
        for r in jammer_contrasts
    }

    adaptive_frozen = []
    for seed, regime, env, k in {
        (
            r["seed"],
            r["initial_regime"],
            r["network_environment"],
            r["K"],
        )
        for r in jammer_contrasts
    }:
        adaptive = jammer_index.get((seed, regime, env, "adaptive", k))
        frozen = jammer_index.get((seed, regime, env, "frozen", k))
        if adaptive is None or frozen is None:
            continue
        adaptive_frozen.append(
            {
                "seed": seed,
                "initial_regime": regime,
                "network_environment": env,
                "K": k,
                "adaptive_minus_frozen_delta_mse": (
                    adaptive["delta_mse"] - frozen["delta_mse"]
                ),
                "adaptive_minus_frozen_delta_rmse": (
                    adaptive["delta_rmse"] - frozen["delta_rmse"]
                ),
                "adaptive_minus_frozen_delta_mae": (
                    adaptive["delta_mae"] - frozen["delta_mae"]
                ),
            }
        )

    return jammer_contrasts, adaptive_frozen


def checkpoint_periods(model: InfoSampleModel) -> set[int]:
    if not model.reliance_history:
        return set()
    last = len(model.reliance_history) - 1
    return {p for p in (0, 1, 5, 10, last) if 0 <= p <= last}


def run_one(
    *,
    config: dict,
    seed: int,
    regime: str,
    environment: str,
    reliance_mode: str,
    jammer_active: bool,
    k: int,
    save_edge_log: bool,
) -> tuple[dict, list[dict], list[dict]]:
    cfg = dict(config)
    cfg.update(
        {
            "network_environment": environment,
            "reliance_mode": reliance_mode,
            "jammer_active": jammer_active,
            "surveil_ability": k,
        }
    )

    model = InfoSampleModel(model_attribute=cfg, rng=seed)
    topo = topology_summary(model)

    while model.running:
        model.step()
        assert_finite_state(
            model,
            context=(
                f"{environment}/{regime}/{reliance_mode}/"
                f"J{int(jammer_active)}/seed={seed}"
            ),
        )
        if model.steps >= model.max_steps:
            model.running = False
        elif (
            model.steps >= 3
            and math.isfinite(model.avg_agent_mu_theta_diff_rate)
            and model.avg_agent_mu_theta_diff_rate < model.convergence_tolerance
        ):
            model.running = False

    metrics = theory_metrics(model)
    metrics.update(realized_flow_composition(model))
    metrics["reliance_hhi"] = reliance_hhi(model)

    row = {
        "seed": seed,
        "initial_regime": regime,
        "network_environment": environment,
        "reliance_mode": reliance_mode,
        "jammer_active": jammer_active,
        "K": k,
        "steps_run": int(model.steps),
        "converged_before_max": bool(model.steps < model.max_steps),
        **topo,
        **metrics,
    }

    belief_rows = [
        {
            "seed": seed,
            "initial_regime": regime,
            "network_environment": environment,
            "reliance_mode": reliance_mode,
            "jammer_active": jammer_active,
            "K": k,
            "citizen_pos": int(citizen.pos),
            "citizen_group": int(citizen.group_id),
            "terminal_mu_theta": float(citizen.mu_theta_beliefs[-1]),
            "terminal_sd_theta": float(citizen.sd_theta_beliefs[-1]),
        }
        for citizen in model.citizens
    ]

    edge_rows = []
    if save_edge_log:
        for period in sorted(checkpoint_periods(model)):
            for record in model.reliance_history[period]:
                edge_rows.append(
                    {
                        "seed": seed,
                        "initial_regime": regime,
                        "network_environment": environment,
                        "reliance_mode": reliance_mode,
                        "jammer_active": jammer_active,
                        "K": k,
                        **record,
                    }
                )

    return row, belief_rows, edge_rows


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    regimes = parse_regimes(args.initial_regimes)

    if args.n_citizens < 4:
        raise ValueError("--n-citizens must be at least 4.")
    if not 0.0 <= args.epsilon <= 0.5:
        raise ValueError("--epsilon must lie in [0, 0.5].")
    if args.credit <= 0 or args.max_steps <= 0 or args.k <= 0:
        raise ValueError("credit, max-steps, and K must be positive.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    run_dir = output_root / f"diagnostic_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_at_local": stamp,
        "seeds": seeds,
        "initial_regimes": regimes,
        "network_environments": list(NETWORK_ENVIRONMENTS),
        "reliance_modes": list(RELIANCE_MODES),
        "jammer_states": [True, False],
        "n_citizens": args.n_citizens,
        "num_nodes_total": args.n_citizens + 2,
        "max_steps": args.max_steps,
        "K": args.k,
        "epsilon": args.epsilon,
        "credit": args.credit,
        "comparison_rule": args.comparison_rule,
        "surveillance_interval": args.surveillance_interval,
        "save_edge_log": args.save_edge_log,
        "design": {
            "elite_only": "Expert + Jammer only",
            "random_2": "exactly two sources from full non-ego citizen+elite pool",
            "group_id": (
                "exactly two sources from full pool with 0.9 same-group / "
                "0.1 cross-group preference"
            ),
            "extended": "Expert + Jammer + exactly two random citizen peers",
        },
        **git_metadata(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    total_runs = (
        len(seeds)
        * len(regimes)
        * len(NETWORK_ENVIRONMENTS)
        * len(RELIANCE_MODES)
        * len(JAMMER_STATES)
    )
    print(f"Paper B local diagnostic: {total_runs} runs")
    print(f"Output directory: {run_dir}")

    run_rows: list[dict] = []
    belief_rows: list[dict] = []
    edge_rows: list[dict] = []

    completed = 0
    for regime in regimes:
        for seed in seeds:
            base = base_model_config(
                regime=regime,
                seed=seed,
                n_citizens=args.n_citizens,
                max_steps=args.max_steps,
                k=args.k,
                epsilon=args.epsilon,
                credit=args.credit,
                comparison_rule=args.comparison_rule,
                surveillance_interval=args.surveillance_interval,
            )

            # Important pairing property:
            # the exact same initial-belief vector and model seed are reused
            # across J=1/J=0 and adaptive/frozen within each environment.
            for environment in NETWORK_ENVIRONMENTS:
                for reliance_mode in RELIANCE_MODES:
                    for jammer_active in JAMMER_STATES:
                        row, beliefs, edges = run_one(
                            config=base,
                            seed=seed,
                            regime=regime,
                            environment=environment,
                            reliance_mode=reliance_mode,
                            jammer_active=jammer_active,
                            k=args.k,
                            save_edge_log=args.save_edge_log,
                        )
                        run_rows.append(row)
                        belief_rows.extend(beliefs)
                        edge_rows.extend(edges)
                        completed += 1

                        if completed % max(args.progress_every, 1) == 0:
                            print(
                                f"[{completed:>4}/{total_runs}] "
                                f"{environment:>10} | {reliance_mode:>8} | "
                                f"J={int(jammer_active)} | seed={seed} | "
                                f"MSE={row['mse_truth']:.4f}"
                            )

    jammer_rows, adaptive_frozen_rows = matched_contrasts(run_rows)

    write_csv(run_dir / "runs.csv", run_rows)
    write_csv(run_dir / "jammer_contrasts.csv", jammer_rows)
    write_csv(run_dir / "adaptive_frozen_contrasts.csv", adaptive_frozen_rows)
    write_gzip_csv(run_dir / "terminal_beliefs.csv.gz", belief_rows)
    if args.save_edge_log:
        write_gzip_csv(run_dir / "reliance_checkpoints.csv.gz", edge_rows)

    summary = {
        "n_runs": len(run_rows),
        "n_jammer_contrasts": len(jammer_rows),
        "n_adaptive_frozen_contrasts": len(adaptive_frozen_rows),
        "all_runs_finite": all(r["n_nonfinite"] == 0 for r in run_rows),
        "output_files": [
            "manifest.json",
            "runs.csv",
            "jammer_contrasts.csv",
            "adaptive_frozen_contrasts.csv",
            "terminal_beliefs.csv.gz",
        ] + (["reliance_checkpoints.csv.gz"] if args.save_edge_log else []),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    zip_path = output_root / f"paper_b_diagnostic_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(run_dir.iterdir()):
            if file.is_file():
                archive.write(file, arcname=file.name)

    print()
    print("Diagnostic complete.")
    print(f"Runs: {len(run_rows)}")
    print(f"Finite-state check: {'PASS' if summary['all_runs_finite'] else 'FAIL'}")
    print(f"Upload this file back to ChatGPT:")
    print(f"  {zip_path}")


if __name__ == "__main__":
    main()
