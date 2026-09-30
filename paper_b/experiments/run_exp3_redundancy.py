"""Experiment III: corrective-pathway redundancy under matched elite access.

Primary production design:
    500 matched seeds x 1 prior regime (flat)
    x 2 redundancy levels
    x 2 reliance modes
    x 2 Jammer states
    = 4,000 simulations at fixed T=200.

The matched block is (seed, initial regime).  Within a block, low/high
redundancy share the exact same direct Expert gateway set, universal Jammer
access, initial beliefs, and peer degree.  Only the number of distinct
length-2 Expert routes changes for non-gateway citizens.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from paper_b.experiments.run_local_diagnostic import (
    base_model_config,
    matched_contrasts,
    parse_regimes,
)
from paper_b.experiments.run_matched_pilot import (
    canonical_hash,
    run_condition,
    software_versions,
)
from paper_b.structural_designs import (
    EXPERT_POS,
    JAMMER_POS,
    exp3_redundancy_source_maps,
    two_step_expert_route_count,
)


REDUNDANCY_LEVELS = ("low", "high")
RELIANCE_MODES = ("adaptive", "frozen")
JAMMER_STATES = (True, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="500")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--initial-regimes", default="flat")
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--expert-access-share", type=float, default=0.10)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_exp3_redundancy",
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
        "paper_b/experiments/run_exp3_redundancy.py",
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


def _peer_degree(source_map: dict[int, list[int]]) -> dict[int, int]:
    return {
        int(ego): sum(int(source) >= 2 for source in sources)
        for ego, sources in source_map.items()
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
    regime = task["initial_regime"]
    block_id = task["block_id"]
    design_id = task["design_id"]
    root = Path(task["run_root"])

    blueprint = exp3_redundancy_source_maps(
        seed=seed,
        n_citizens=int(task["n_citizens"]),
        peer_degree=int(task["peer_degree"]),
        expert_access_share=float(task["expert_access_share"]),
    )

    if _elite_signature(blueprint["low"]) != _elite_signature(blueprint["high"]):
        raise RuntimeError(f"{block_id}: elite access differs across redundancy.")
    if _peer_degree(blueprint["low"]) != _peer_degree(blueprint["high"]):
        raise RuntimeError(f"{block_id}: peer degree differs across redundancy.")

    low_routes = two_step_expert_route_count(
        blueprint["low"],
        expert_gateways=blueprint["expert_gateways"],
    )
    high_routes = two_step_expert_route_count(
        blueprint["high"],
        expert_gateways=blueprint["expert_gateways"],
    )
    gateways = set(blueprint["expert_gateways"])
    nongateways = [ego for ego in low_routes if ego not in gateways]
    if any(low_routes[ego] != 1 for ego in nongateways):
        raise RuntimeError(f"{block_id}: LOW redundancy route count is not one.")
    if any(high_routes[ego] != 2 for ego in nongateways):
        raise RuntimeError(f"{block_id}: HIGH redundancy route count is not two.")

    base = base_model_config(
        regime=regime,
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
    for redundancy in REDUNDANCY_LEVELS:
        for reliance in RELIANCE_MODES:
            for jammer_active in JAMMER_STATES:
                cfg = dict(base)
                cfg["structural_source_map"] = blueprint[redundancy]
                result = run_condition(
                    base_config=cfg,
                    seed=seed,
                    regime=regime,
                    environment=f"exp3_redundancy_{redundancy}",
                    reliance_mode=reliance,
                    jammer_active=jammer_active,
                    k=int(task["k"]),
                    design_id=design_id,
                    block_id=block_id,
                    save_edge_log=False,
                )
                result["run"]["redundancy"] = redundancy
                result["run"]["expert_gateway_count"] = len(
                    blueprint["expert_gateways"]
                )
                result["run"]["expert_access_share"] = float(
                    blueprint["expert_access_share_realized"]
                )
                results.append(result)

    # Initial states must be identical across all eight conditions.
    initial_fps = {r["run"]["initial_state_fingerprint"] for r in results}
    if len(initial_fps) != 1:
        raise RuntimeError(f"{block_id}: initial-state mismatch.")

    # Within a redundancy level, J/reliance conditions use the same structure.
    for redundancy in REDUNDANCY_LEVELS:
        fps = {
            r["run"]["structural_fingerprint"]
            for r in results
            if r["run"]["redundancy"] == redundancy
        }
        if len(fps) != 1:
            raise RuntimeError(
                f"{block_id}: structural mismatch within {redundancy}."
            )

    payload = {
        "complete": True,
        "design_id": design_id,
        "block_id": block_id,
        "seed": seed,
        "initial_regime": regime,
        "expert_gateways": list(blueprint["expert_gateways"]),
        "elite_signature": _elite_signature(blueprint["low"]),
        "mean_two_step_routes_low": sum(low_routes[e] for e in nongateways)
        / len(nongateways),
        "mean_two_step_routes_high": sum(high_routes[e] for e in nongateways)
        / len(nongateways),
        "runs": [r["run"] for r in results],
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

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


def consolidate(root: Path, manifest: dict) -> dict:
    shards = _read_shards(root)
    runs = [row for shard in shards for row in shard["runs"]]
    jammer_rows, _ = matched_contrasts(runs)

    index = {
        (
            row["seed"],
            row["initial_regime"],
            row["network_environment"],
            row["reliance_mode"],
        ): row
        for row in jammer_rows
    }

    contrasts = []
    activation = []
    keys = {
        (row["seed"], row["initial_regime"], row["reliance_mode"])
        for row in jammer_rows
    }
    for seed, regime, reliance in sorted(keys):
        low = index.get(
            (seed, regime, "exp3_redundancy_low", reliance)
        )
        high = index.get(
            (seed, regime, "exp3_redundancy_high", reliance)
        )
        if low is None or high is None:
            continue
        contrasts.append(
            {
                "seed": seed,
                "initial_regime": regime,
                "reliance_mode": reliance,
                "delta_mse_low_redundancy": low["delta_mse"],
                "delta_mse_high_redundancy": high["delta_mse"],
                "high_minus_low_delta_mse": (
                    high["delta_mse"] - low["delta_mse"]
                ),
            }
        )

    cindex = {
        (r["seed"], r["initial_regime"], r["reliance_mode"]): r
        for r in contrasts
    }
    for seed, regime in sorted(
        {(r["seed"], r["initial_regime"]) for r in contrasts}
    ):
        adaptive = cindex.get((seed, regime, "adaptive"))
        frozen = cindex.get((seed, regime, "frozen"))
        if adaptive is None or frozen is None:
            continue
        activation.append(
            {
                "seed": seed,
                "initial_regime": regime,
                "adaptive_redundancy_effect": adaptive[
                    "high_minus_low_delta_mse"
                ],
                "frozen_redundancy_effect": frozen[
                    "high_minus_low_delta_mse"
                ],
                "adaptive_minus_frozen_redundancy_effect": (
                    adaptive["high_minus_low_delta_mse"]
                    - frozen["high_minus_low_delta_mse"]
                ),
            }
        )

    audit_rows = [
        {
            "block_id": shard["block_id"],
            "seed": shard["seed"],
            "initial_regime": shard["initial_regime"],
            "expert_gateway_count": len(shard["expert_gateways"]),
            "mean_two_step_routes_low": shard["mean_two_step_routes_low"],
            "mean_two_step_routes_high": shard["mean_two_step_routes_high"],
        }
        for shard in shards
    ]

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "jammer_contrasts.csv", jammer_rows)
    _write_csv(root / "exp3_redundancy_contrasts.csv", contrasts)
    _write_csv(root / "exp3_activation_interaction.csv", activation)
    _write_csv(root / "exp3_design_audit.csv", audit_rows)

    expected_blocks = int(manifest["expected_blocks"])
    expected_runs = int(manifest["expected_runs"])
    finite = all(int(r.get("n_nonfinite", 1)) == 0 for r in runs)
    horizon = all(
        int(r["steps_run"]) == int(r["terminal_horizon_T"])
        for r in runs
    )
    frozen = all(
        abs(float(r["lambda_cumulative_turnover"])) <= 1e-12
        for r in runs
        if r["reliance_mode"] == "frozen"
    )
    route_gate = all(
        abs(float(r["mean_two_step_routes_low"]) - 1.0) <= 1e-12
        and abs(float(r["mean_two_step_routes_high"]) - 2.0) <= 1e-12
        for r in audit_rows
    )

    report = {
        "pass": (
            len(shards) == expected_blocks
            and len(runs) == expected_runs
            and finite
            and horizon
            and frozen
            and route_gate
        ),
        "expected_blocks": expected_blocks,
        "observed_blocks": len(shards),
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "all_runs_finite": finite,
        "all_runs_fixed_horizon": horizon,
        "frozen_lambda_invariant": frozen,
        "redundancy_route_gate": route_gate,
    }
    (root / "exp3_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def package(root: Path, design_id: str) -> Path:
    import zipfile

    names = (
        "exp3_manifest.json",
        "runs.csv",
        "jammer_contrasts.csv",
        "exp3_redundancy_contrasts.csv",
        "exp3_activation_interaction.csv",
        "exp3_design_audit.csv",
        "exp3_gate.json",
    )
    path = root.parent / f"paper_b_exp3_{design_id}_shareable.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            p = root / name
            if p.exists():
                archive.write(p, arcname=name)
    return path


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args.seeds, args.seed_start)
    regimes = parse_regimes(args.initial_regimes)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    design = {
        "design_version": 1,
        "experiment": "III_corrective_redundancy",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "software_versions": software_versions(),
        "seeds": seeds,
        "initial_regimes": regimes,
        "n_citizens": args.n_citizens,
        "horizon_T": args.horizon,
        "K": args.k,
        "epsilon": args.epsilon,
        "credit": args.credit,
        "expert_access_share": args.expert_access_share,
        "peer_degree": args.peer_degree,
        "surveillance_interval": args.surveillance_interval,
        "redundancy_levels": list(REDUNDANCY_LEVELS),
        "reliance_modes": list(RELIANCE_MODES),
        "jammer_states": [True, False],
        "jammer_access": "universal structural slot; neutralized under J=0",
    }
    design_id = canonical_hash(design)[:12]
    blocks = [
        {
            "seed": seed,
            "initial_regime": regime,
            "block_id": f"exp3__{regime}__s{seed}",
        }
        for regime in regimes
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

    root = Path(args.output_dir) / f"exp3_{design_id}"
    manifest_path = root / "exp3_manifest.json"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design_id") != design_id:
            raise RuntimeError("Existing Exp III manifest does not match.")
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
                "horizon": args.horizon,
                "k": args.k,
                "epsilon": args.epsilon,
                "credit": args.credit,
                "expert_access_share": args.expert_access_share,
                "peer_degree": args.peer_degree,
                "surveillance_interval": args.surveillance_interval,
            }
        )

    print("Paper B Experiment III: corrective redundancy")
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
    print(f"Exp III gate: {'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(1)
    bundle = package(root, design_id)
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
