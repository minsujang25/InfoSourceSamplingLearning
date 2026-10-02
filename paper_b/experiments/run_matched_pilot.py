"""Matched-block pilot runner for Social Networks Paper B.

The pilot is the calibration gate between the reconstruction diagnostics and the
full production grid.  Work is parallelized by matched block:

    (seed, initial regime, network environment)

Each worker runs the four counterfactual conditions sequentially:

    adaptive/J=1, adaptive/J=0, frozen/J=1, frozen/J=0

so structural opportunity and initial-state fingerprints can be verified inside
one block before the shard is committed.

Default pilot:
    20 seeds x 3 prior regimes x 4 network environments = 240 blocks
    240 blocks x 4 matched conditions = 960 individual simulations
    fixed horizon T = 200
    K = 1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from importlib import metadata as importlib_metadata
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from model.InfoSourceSamplingLearning import InfoSampleModel
from paper_b.experiments.run_local_diagnostic import (
    NETWORK_ENVIRONMENTS,
    base_model_config,
    checkpoint_periods,
    parse_regimes,
    topology_summary,
)
from paper_b.metrics import (
    dominant_reliance_skeleton_metrics,
    posterior_precision_checkpoint,
    realized_flow_composition,
    reliance_checkpoint_metrics,
    reliance_hhi,
    theory_metrics,
)
from paper_b.validation import assert_finite_state


CONDITION_ORDER = (
    ("adaptive", True),
    ("adaptive", False),
    ("frozen", True),
    ("frozen", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Paper B larger matched-seed calibration pilot."
    )
    parser.add_argument(
        "--seeds",
        default="20",
        help=(
            "Positive integer N or comma-separated explicit seeds. With integer N, "
            "--seed-start determines the first seed."
        ),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1001,
        help="First seed when --seeds is an integer count.",
    )
    parser.add_argument(
        "--initial-regimes",
        default="flat,consensus,polarized",
        help="Comma-separated subset of flat,consensus,polarized.",
    )
    parser.add_argument(
        "--network-environments",
        default=",".join(NETWORK_ENVIRONMENTS),
        help=(
            "Comma-separated subset of "
            + ",".join(NETWORK_ENVIRONMENTS)
            + "."
        ),
    )
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument(
        "--horizon",
        "--max-steps",
        dest="horizon",
        type=int,
        default=200,
        help="Exact terminal horizon T for every matched condition.",
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument(
        "--comparison-rule",
        choices=("delta_comparison", "z_stat_comparison"),
        default="delta_comparison",
    )
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of matched blocks to run concurrently. Set this explicitly "
            "to the CPU allocation on UCloud to avoid oversubscription."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="ucloud_results/paper_b_pilot",
        help="Parent directory. A deterministic design-id subdirectory is created.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip valid completed block shards and continue an interrupted pilot.",
    )
    parser.add_argument(
        "--save-edge-log",
        dest="save_edge_log",
        action="store_true",
        default=True,
        help="Save selected-period edge-level Lambda/X records (default on).",
    )
    parser.add_argument(
        "--no-edge-log",
        dest="save_edge_log",
        action="store_false",
        help="Skip edge-level checkpoint records.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed matched blocks.",
    )
    return parser.parse_args()


def resolve_seeds(value: str, seed_start: int) -> list[int]:
    value = value.strip()
    if "," in value:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        n = int(value)
        if n <= 0:
            raise ValueError("--seeds must be positive.")
        seeds = list(range(int(seed_start), int(seed_start) + n))
    if not seeds:
        raise ValueError("No seeds resolved.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def parse_network_environments(value: str) -> list[str]:
    environments = [
        part.strip().lower()
        for part in value.split(",")
        if part.strip()
    ]
    invalid = [
        env for env in environments
        if env not in NETWORK_ENVIRONMENTS
    ]
    if invalid:
        raise ValueError(
            "Unknown network environment(s): "
            f"{invalid}. Valid values: {list(NETWORK_ENVIRONMENTS)}"
        )
    if not environments:
        raise ValueError("At least one network environment is required.")
    if len(set(environments)) != len(environments):
        raise ValueError("Network environments must be unique.")
    return environments


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_fingerprint() -> str:
    """Hash the scientific code that defines a pilot shard.

    The design ID includes this fingerprint so --resume can never silently mix
    shards produced by different reconstruction code under the same parameter
    grid.
    """
    repo_root = Path(__file__).resolve().parents[2]
    paths = (
        repo_root / "model" / "InfoSourceSamplingLearning.py",
        repo_root / "paper_b" / "metrics.py",
        repo_root / "paper_b" / "experiments" / "run_local_diagnostic.py",
        repo_root / "paper_b" / "experiments" / "run_matched_pilot.py",
        repo_root / "environment.yml",
    )
    payload = []
    for path in paths:
        payload.append(
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return canonical_hash(payload)


def software_versions() -> dict:
    packages = ("mesa", "numpy", "scipy", "scikit-learn", "networkx")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def initial_state_fingerprint(config: dict) -> str:
    payload = {
        "state_of_the_world": float(config["state_of_the_world"]),
        "mu_theta": [float(x) for x in config["mu_theta"]],
        "sd_theta": [float(x) for x in config["sd_theta"]],
    }
    return canonical_hash(payload)


def structural_fingerprint(model: InfoSampleModel) -> str:
    edges = []
    for citizen in model.citizens:
        for source in citizen.info_source:
            edges.append(
                (
                    int(citizen.pos),
                    int(source.pos),
                    str(source.type_of_agent),
                    int(citizen.group_id),
                    int(source.group_id),
                )
            )
    edges.sort()
    return canonical_hash(edges)


def pilot_checkpoint_periods(model: InfoSampleModel) -> list[int]:
    if not model.reliance_history:
        return []
    last = len(model.reliance_history) - 1
    # Keep the reconstruction checkpoints and add explicit horizon-calibration
    # points. Because history is zero-indexed, period 199 is the state after
    # 200 executed periods and period 399 is the state after 400 periods.
    values = set(checkpoint_periods(model))
    values.update(
        p for p in (2, 100, 150, 199, 299, 399)
        if p <= last
    )
    return sorted(values)


def belief_checkpoint_metrics(model: InfoSampleModel, period: int) -> dict:
    """Truth-loss summary from the stored post-period citizen belief vector."""
    values = np.asarray(model.agent_mu_theta_list[period], dtype=float)
    truth = float(model.state_of_the_world)
    errors = values - truth
    mean = float(values.mean())
    variance = float(values.var(ddof=0))
    mse = float(np.mean(errors**2))
    return {
        "period": int(period),
        "horizon_step": int(period) + 1,
        "mean_belief": mean,
        "belief_variance": variance,
        "belief_sd": float(math.sqrt(variance)),
        "squared_displacement": float((mean - truth) ** 2),
        "mse_truth": mse,
        "rmse_truth": float(math.sqrt(mse)),
        "mae_truth": float(np.mean(np.abs(errors))),
    }


def _condition_label(reliance_mode: str, jammer_active: bool) -> str:
    return f"{reliance_mode}__J{int(jammer_active)}"


def run_condition(
    *,
    base_config: dict,
    seed: int,
    regime: str,
    environment: str,
    reliance_mode: str,
    jammer_active: bool,
    k: int,
    design_id: str,
    block_id: str,
    save_edge_log: bool,
    jammer_regime: str | None = None,
    peer_evidence_mode: str = "legacy_batch",
    frozen_ranking_mode: str = "first_audit",
    gateway_positions: set[int] | None = None,
) -> dict:
    config = dict(base_config)
    config.update(
        {
            "network_environment": environment,
            "reliance_mode": reliance_mode,
            "jammer_active": bool(jammer_active),
            "surveil_ability": int(k),
            "stop_on_convergence": False,
            "peer_evidence_mode": peer_evidence_mode,
            "frozen_ranking_mode": frozen_ranking_mode,
        }
    )

    if jammer_regime is not None:
        config["jammer_regime"] = str(jammer_regime)

    model = InfoSampleModel(model_attribute=config, rng=seed)
    structure_fp = structural_fingerprint(model)
    initial_fp = initial_state_fingerprint(config)
    topology = topology_summary(model)

    for _ in range(model.max_steps):
        model.step()
        assert_finite_state(
            model,
            context=(
                f"pilot/{block_id}/{reliance_mode}/"
                f"J{int(jammer_active)}"
            ),
        )
    model.running = False

    if model.steps != model.max_steps:
        raise RuntimeError(
            f"{block_id}: fixed-horizon violation: "
            f"expected {model.max_steps}, observed {model.steps}."
        )

    metrics = theory_metrics(model)
    metrics.update(realized_flow_composition(model))
    metrics["reliance_hhi"] = reliance_hhi(model)
    metrics.update(
        dominant_reliance_skeleton_metrics(
            model,
            gateway_positions=gateway_positions,
        )
    )
    audit_counts = [
        sum(int(value == 1) for value in citizen.theta_or_delta_history[1:])
        for citizen in model.citizens
    ]
    metrics["audit_count_mean"] = (
        float(np.mean(audit_counts)) if audit_counts else math.nan
    )
    metrics["audit_count_median"] = (
        float(np.median(audit_counts)) if audit_counts else math.nan
    )

    common = {
        "design_id": design_id,
        "block_id": block_id,
        "seed": int(seed),
        "initial_regime": regime,
        "network_environment": environment,
        "reliance_mode": reliance_mode,
        "jammer_active": bool(jammer_active),
        "jammer_regime": str(model.jammer_regime),
        "peer_evidence_mode": str(model.peer_evidence_mode),
        "tau_social": float(model.tau_social),
        "frozen_ranking_mode": str(model.frozen_ranking_mode),
        "K": int(k),
        "initial_state_fingerprint": initial_fp,
        "structural_fingerprint": structure_fp,
    }

    run_row = {
        **common,
        "condition": (
            _condition_label(reliance_mode, jammer_active)
            if jammer_regime is None
            else f"{reliance_mode}__{model.jammer_regime}"
        ),
        "steps_run": int(model.steps),
        "terminal_horizon_T": int(model.max_steps),
        "first_convergence_period": model.first_convergence_period,
        "converged_by_T": model.first_convergence_period is not None,
        "final_relative_theta_change": float(model.avg_agent_mu_theta_diff_rate),
        **topology,
        **metrics,
    }

    belief_rows = [
        {
            **common,
            "citizen_pos": int(citizen.pos),
            "citizen_group": int(citizen.group_id),
            "terminal_mu_theta": float(citizen.mu_theta_beliefs[-1]),
            "terminal_sd_theta": float(citizen.sd_theta_beliefs[-1]),
        }
        for citizen in model.citizens
    ]

    lambda_rows = []
    belief_checkpoint_rows = []
    edge_rows = []
    for period in pilot_checkpoint_periods(model):
        lambda_rows.append(
            {
                **common,
                **reliance_checkpoint_metrics(model, period),
                **posterior_precision_checkpoint(model, period),
                **dominant_reliance_skeleton_metrics(
                    model,
                    period=period,
                    gateway_positions=gateway_positions,
                ),
            }
        )
        belief_checkpoint_rows.append(
            {
                **common,
                **belief_checkpoint_metrics(model, period),
            }
        )
        if save_edge_log:
            for record in model.reliance_history[period]:
                edge_rows.append({**common, **record})

    jammer_rows = [
        {**common, **record}
        for record in model.jammer_strategy_history
    ]

    return {
        "run": run_row,
        "beliefs": belief_rows,
        "lambda_checkpoints": lambda_rows,
        "belief_checkpoints": belief_checkpoint_rows,
        "jammer_strategy": jammer_rows,
        "edges": edge_rows,
    }


def shard_path(root: Path, block_id: str) -> Path:
    return root / "shards" / f"{block_id}.json.gz"


def valid_existing_shard(path: Path, *, design_id: str, block_id: str) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            payload.get("design_id") == design_id
            and payload.get("block_id") == block_id
            and payload.get("complete") is True
            and len(payload.get("runs", [])) == 4
        )
    except Exception:
        return False


def atomic_write_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_block(task: dict) -> dict:
    seed = int(task["seed"])
    regime = str(task["regime"])
    environment = str(task["environment"])
    design_id = str(task["design_id"])
    block_id = str(task["block_id"])
    root = Path(task["run_root"])

    base = base_model_config(
        regime=regime,
        seed=seed,
        n_citizens=int(task["n_citizens"]),
        max_steps=int(task["horizon"]),
        k=int(task["k"]),
        epsilon=float(task["epsilon"]),
        credit=int(task["credit"]),
        comparison_rule=str(task["comparison_rule"]),
        surveillance_interval=int(task["surveillance_interval"]),
    )

    results = []
    for reliance_mode, jammer_active in CONDITION_ORDER:
        results.append(
            run_condition(
                base_config=base,
                seed=seed,
                regime=regime,
                environment=environment,
                reliance_mode=reliance_mode,
                jammer_active=jammer_active,
                k=int(task["k"]),
                design_id=design_id,
                block_id=block_id,
                save_edge_log=bool(task["save_edge_log"]),
            )
        )

    structural_fps = {r["run"]["structural_fingerprint"] for r in results}
    initial_fps = {r["run"]["initial_state_fingerprint"] for r in results}
    if len(structural_fps) != 1:
        raise RuntimeError(
            f"{block_id}: structural fingerprints differ across matched quartet."
        )
    if len(initial_fps) != 1:
        raise RuntimeError(
            f"{block_id}: initial-state fingerprints differ across matched quartet."
        )

    payload = {
        "complete": True,
        "design_id": design_id,
        "block_id": block_id,
        "seed": seed,
        "initial_regime": regime,
        "network_environment": environment,
        "initial_state_fingerprint": next(iter(initial_fps)),
        "structural_fingerprint": next(iter(structural_fps)),
        "runs": [r["run"] for r in results],
        "terminal_beliefs": [
            row for result in results for row in result["beliefs"]
        ],
        "lambda_checkpoints": [
            row for result in results for row in result["lambda_checkpoints"]
        ],
        "belief_checkpoints": [
            row for result in results for row in result["belief_checkpoints"]
        ],
        "jammer_strategy": [
            row for result in results for row in result["jammer_strategy"]
        ],
        "reliance_edges": [
            row for result in results for row in result["edges"]
        ],
    }

    path = shard_path(root, block_id)
    atomic_write_gzip_json(path, payload)
    return {
        "block_id": block_id,
        "shard": str(path),
        "max_mse": max(float(r["run"]["mse_truth"]) for r in results),
    }


def build_design(
    args: argparse.Namespace,
    seeds: list[int],
    regimes: list[str],
    environments: list[str],
) -> dict:
    return {
        "design_version": 2,
        "purpose": "Paper B matched-seed production-calibration pilot",
        "scientific_code_fingerprint": code_fingerprint(),
        "software_versions": software_versions(),
        "seeds": seeds,
        "initial_regimes": regimes,
        "network_environments": environments,
        "matched_conditions": [
            _condition_label(mode, jammer) for mode, jammer in CONDITION_ORDER
        ],
        "n_citizens": int(args.n_citizens),
        "horizon_T": int(args.horizon),
        "K": int(args.k),
        "epsilon": float(args.epsilon),
        "credit": int(args.credit),
        "comparison_rule": args.comparison_rule,
        "surveillance_interval": int(args.surveillance_interval),
        "save_edge_log": bool(args.save_edge_log),
    }


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    regimes = parse_regimes(args.initial_regimes)
    environments = parse_network_environments(args.network_environments)

    if args.n_citizens < 4:
        raise ValueError("--n-citizens must be at least 4.")
    if args.horizon <= 0 or args.k <= 0 or args.credit <= 0:
        raise ValueError("horizon, K, and credit must be positive.")
    if not 0.0 <= args.epsilon <= 0.5:
        raise ValueError("--epsilon must lie in [0, 0.5].")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    design = build_design(args, seeds, regimes, environments)
    design_id = canonical_hash(design)[:12]
    blocks = [
        {
            "seed": seed,
            "regime": regime,
            "environment": environment,
            "block_id": f"{environment}__{regime}__s{seed}",
        }
        for regime in regimes
        for seed in seeds
        for environment in environments
    ]
    design["design_id"] = design_id
    design["expected_blocks"] = len(blocks)
    design["expected_runs"] = len(blocks) * len(CONDITION_ORDER)
    design["block_ids"] = [b["block_id"] for b in blocks]

    run_root = Path(args.output_dir) / f"pilot_{design_id}"
    manifest_path = run_root / "pilot_manifest.json"

    if run_root.exists() and not args.resume:
        raise FileExistsError(
            f"{run_root} already exists. Use --resume to continue this exact "
            "design, or choose a different --output-dir/design."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "shards").mkdir(exist_ok=True)

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design_id") != design_id:
            raise RuntimeError("Existing pilot manifest does not match this design.")
    else:
        manifest_path.write_text(
            json.dumps(design, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    pending = []
    skipped = 0
    for block in blocks:
        path = shard_path(run_root, block["block_id"])
        if args.resume and valid_existing_shard(
            path,
            design_id=design_id,
            block_id=block["block_id"],
        ):
            skipped += 1
            continue

        pending.append(
            {
                **block,
                "design_id": design_id,
                "run_root": str(run_root),
                "n_citizens": args.n_citizens,
                "horizon": args.horizon,
                "k": args.k,
                "epsilon": args.epsilon,
                "credit": args.credit,
                "comparison_rule": args.comparison_rule,
                "surveillance_interval": args.surveillance_interval,
                "save_edge_log": args.save_edge_log,
            }
        )

    print("Paper B matched pilot")
    print(f"  design_id       : {design_id}")
    print(f"  output           : {run_root}")
    print(f"  networks         : {','.join(environments)}")
    print(f"  matched blocks   : {len(blocks)}")
    print(f"  individual runs  : {len(blocks) * 4}")
    print(f"  pending blocks   : {len(pending)}")
    print(f"  resumed/skipped  : {skipped}")
    print(f"  workers          : {args.workers}")
    print(f"  fixed horizon T  : {args.horizon}")

    completed = 0
    if pending:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
        ) as pool:
            futures = {pool.submit(run_block, task): task for task in pending}
            try:
                for future in as_completed(futures):
                    task = futures[future]
                    result = future.result()
                    completed += 1
                    if completed % max(args.progress_every, 1) == 0:
                        print(
                            f"[{completed:>4}/{len(pending)}] "
                            f"{result['block_id']} | "
                            f"max MSE={result['max_mse']:.4f}"
                        )
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    # Consolidation and validation are part of the pilot gate.
    from paper_b.experiments.consolidate_pilot import consolidate
    from paper_b.experiments.check_pilot_output import validate_pilot
    from paper_b.experiments.package_pilot import package_pilot

    consolidate(run_root)
    report = validate_pilot(run_root, write_report=True)

    print()
    print(f"Pilot gate: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"  completed blocks: {report['observed_blocks']}/{report['expected_blocks']}")
    print(f"  individual runs : {report['observed_runs']}/{report['expected_runs']}")
    print(f"  finite states   : {report['all_runs_finite']}")
    print(f"  fixed horizon   : {report['all_runs_fixed_horizon']}")
    print(f"  fingerprints    : {report['all_matched_fingerprints']}")
    print(f"  frozen Lambda   : {report['frozen_lambda_invariant']}")
    print(f"  Jammer hold     : {report['jammer_hold_rule']}")

    if not report["pass"]:
        raise SystemExit("Pilot validation gate failed.")

    bundle = package_pilot(run_root)
    print(f"Pilot outputs ready at: {run_root}")
    print(f"Shareable result bundle: {bundle}")


if __name__ == "__main__":
    main()
