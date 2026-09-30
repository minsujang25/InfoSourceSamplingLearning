"""Experiment IV: 2x2 homophily x prior segregation.

Primary production design:
    500 matched seeds
    x 2 structural homophily levels
    x 2 prior-segregation levels
    x 2 Jammer states
    = 4,000 adaptive-reliance simulations at fixed T=200.

Within each matched seed:
  * citizen group labels are fixed before beliefs are generated;
  * direct Expert gateway access is fixed across all four H x S cells;
  * the Jammer structural slot is universal and fixed;
  * peer degree is exactly two;
  * low/high homophily differ only in the citizen-peer mixing rule;
  * low/high segregation use the same individual residual draws and differ only
    in the group-mean shift.

This implements the frozen Theory v1.0 requirement that group labels and Expert
access remain fixed while H^A and S0 vary independently.
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

from paper_b.experiments.run_local_diagnostic import base_model_config
from paper_b.experiments.run_matched_pilot import (
    canonical_hash,
    run_condition,
    software_versions,
)
from paper_b.structural_designs import (
    EXPERT_POS,
    JAMMER_POS,
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
    realized_prior_segregation,
    structural_peer_homophily,
)


HOMOPHILY_LEVELS = ("low", "high")
SEGREGATION_LEVELS = ("low", "high")
JAMMER_STATES = (True, False)
RELIANCE_MODE = "adaptive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="500")
    parser.add_argument("--seed-start", type=int, default=3001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_exp4_homophily_segregation",
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
        "paper_b/experiments/run_exp4_homophily_segregation.py",
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
    n_citizens = int(task["n_citizens"])

    group_ids = balanced_fixed_group_ids(
        seed=seed,
        n_citizens=n_citizens,
    )
    blueprint = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n_citizens,
        group_ids=group_ids,
        peer_degree=int(task["peer_degree"]),
        low_homophily=float(task["low_homophily"]),
        high_homophily=float(task["high_homophily"]),
    )

    if _elite_signature(blueprint["low"]) != _elite_signature(blueprint["high"]):
        raise RuntimeError(f"{block_id}: elite access changed with homophily.")

    for homophily in HOMOPHILY_LEVELS:
        for ego, sources in blueprint[homophily].items():
            elites = sorted(int(source) for source in sources if int(source) < 2)
            peers = [int(source) for source in sources if int(source) >= 2]
            if elites != [EXPERT_POS, JAMMER_POS]:
                raise RuntimeError(
                    f"{block_id}: asymmetric elite access for ego {ego}: {elites}."
                )
            if len(peers) != int(task["peer_degree"]):
                raise RuntimeError(
                    f"{block_id}: peer-degree mismatch for ego {ego}: "
                    f"{len(peers)}."
                )

    h_low = structural_peer_homophily(
        blueprint["low"],
        group_ids=group_ids,
    )
    h_high = structural_peer_homophily(
        blueprint["high"],
        group_ids=group_ids,
    )
    if not h_high > h_low + 0.20:
        raise RuntimeError(
            f"{block_id}: homophily manipulation too weak "
            f"({h_low:.3f} vs {h_high:.3f})."
        )

    initial = {
        segregation: exp4_initial_beliefs(
            seed=seed,
            n_citizens=n_citizens,
            group_ids=group_ids,
            segregation=segregation,
            high_group_shift=float(task["high_group_shift"]),
            residual_sd=float(task["prior_residual_sd"]),
        )
        for segregation in SEGREGATION_LEVELS
    }
    s_low = realized_prior_segregation(initial["low"], group_ids=group_ids)
    s_high = realized_prior_segregation(initial["high"], group_ids=group_ids)
    if not s_high > s_low + 3.0:
        raise RuntimeError(
            f"{block_id}: segregation manipulation too weak "
            f"({s_low:.3f} vs {s_high:.3f})."
        )

    results = []
    for homophily in HOMOPHILY_LEVELS:
        for segregation in SEGREGATION_LEVELS:
            base = base_model_config(
                regime="flat",
                seed=seed,
                n_citizens=n_citizens,
                max_steps=int(task["horizon"]),
                k=int(task["k"]),
                epsilon=float(task["epsilon"]),
                credit=int(task["credit"]),
                comparison_rule="delta_comparison",
                surveillance_interval=int(task["surveillance_interval"]),
            )
            base["mu_theta"] = initial[segregation]
            base["initial_theta_type"] = f"exp4_segregation_{segregation}"
            base["fixed_group_ids"] = group_ids
            base["structural_source_map"] = blueprint[homophily]

            for jammer_active in JAMMER_STATES:
                result = run_condition(
                    base_config=base,
                    seed=seed,
                    regime=f"segregation_{segregation}",
                    environment=f"exp4_homophily_{homophily}",
                    reliance_mode=RELIANCE_MODE,
                    jammer_active=jammer_active,
                    k=int(task["k"]),
                    design_id=design_id,
                    block_id=block_id,
                    save_edge_log=False,
                )
                result["run"]["homophily_level"] = homophily
                result["run"]["segregation_level"] = segregation
                result["run"]["structural_peer_homophily_target"] = float(
                    task[f"{homophily}_homophily"]
                )
                result["run"]["structural_peer_homophily_realized"] = (
                    h_low if homophily == "low" else h_high
                )
                result["run"]["prior_segregation_realized"] = (
                    s_low if segregation == "low" else s_high
                )
                result["run"]["expert_gateway_count"] = len(
                    blueprint["expert_gateways"]
                )
                results.append(result)

    # Same H must have same structural graph across both segregation levels.
    for homophily in HOMOPHILY_LEVELS:
        fps = {
            r["run"]["structural_fingerprint"]
            for r in results
            if r["run"]["homophily_level"] == homophily
        }
        if len(fps) != 1:
            raise RuntimeError(
                f"{block_id}: structure changed across segregation at H={homophily}."
            )

    # Same S must have same initial state across both homophily levels.
    for segregation in SEGREGATION_LEVELS:
        fps = {
            r["run"]["initial_state_fingerprint"]
            for r in results
            if r["run"]["segregation_level"] == segregation
        }
        if len(fps) != 1:
            raise RuntimeError(
                f"{block_id}: initial state changed across homophily at S={segregation}."
            )

    payload = {
        "complete": True,
        "design_id": design_id,
        "block_id": block_id,
        "seed": seed,
        "fixed_group_ids": group_ids,
        "expert_gateways": list(blueprint["expert_gateways"]),
        "elite_signature": _elite_signature(blueprint["low"]),
        "structural_homophily_low": h_low,
        "structural_homophily_high": h_high,
        "prior_segregation_low": s_low,
        "prior_segregation_high": s_high,
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
        "h_low": h_low,
        "h_high": h_high,
        "s_low": s_low,
        "s_high": s_high,
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
    belief_checkpoints = [
        row for shard in shards for row in shard.get("belief_checkpoints", [])
    ]
    lambda_checkpoints = [
        row for shard in shards for row in shard.get("lambda_checkpoints", [])
    ]
    jammer_strategy = [
        row for shard in shards for row in shard.get("jammer_strategy", [])
    ]

    # Build D = MSE(J1)-MSE(J0) within each H x S cell.
    by_cell = {}
    for row in runs:
        key = (
            int(row["seed"]),
            row["homophily_level"],
            row["segregation_level"],
            bool(row["jammer_active"]),
        )
        by_cell[key] = row

    cells = []
    for seed in sorted({int(row["seed"]) for row in runs}):
        for h in HOMOPHILY_LEVELS:
            for s in SEGREGATION_LEVELS:
                j1 = by_cell.get((seed, h, s, True))
                j0 = by_cell.get((seed, h, s, False))
                if j1 is None or j0 is None:
                    continue
                cells.append(
                    {
                        "seed": seed,
                        "homophily_level": h,
                        "segregation_level": s,
                        "structural_peer_homophily_realized": j1[
                            "structural_peer_homophily_realized"
                        ],
                        "prior_segregation_realized": j1[
                            "prior_segregation_realized"
                        ],
                        "delta_mse": float(j1["mse_truth"])
                        - float(j0["mse_truth"]),
                        "delta_rmse": float(j1["rmse_truth"])
                        - float(j0["rmse_truth"]),
                        "delta_mae": float(j1["mae_truth"])
                        - float(j0["mae_truth"]),
                    }
                )

    cindex = {
        (r["seed"], r["homophily_level"], r["segregation_level"]): r
        for r in cells
    }
    interactions = []
    for seed in sorted({r["seed"] for r in cells}):
        ll = cindex.get((seed, "low", "low"))
        hl = cindex.get((seed, "high", "low"))
        lh = cindex.get((seed, "low", "high"))
        hh = cindex.get((seed, "high", "high"))
        if any(x is None for x in (ll, hl, lh, hh)):
            continue

        homophily_effect_low_s = hl["delta_mse"] - ll["delta_mse"]
        homophily_effect_high_s = hh["delta_mse"] - lh["delta_mse"]
        segregation_effect_low_h = lh["delta_mse"] - ll["delta_mse"]
        segregation_effect_high_h = hh["delta_mse"] - hl["delta_mse"]

        interactions.append(
            {
                "seed": seed,
                "D_lowH_lowS": ll["delta_mse"],
                "D_highH_lowS": hl["delta_mse"],
                "D_lowH_highS": lh["delta_mse"],
                "D_highH_highS": hh["delta_mse"],
                "homophily_effect_low_segregation": homophily_effect_low_s,
                "homophily_effect_high_segregation": homophily_effect_high_s,
                "segregation_effect_low_homophily": segregation_effect_low_h,
                "segregation_effect_high_homophily": segregation_effect_high_h,
                "homophily_x_segregation_interaction": (
                    homophily_effect_high_s - homophily_effect_low_s
                ),
            }
        )

    audits = [
        {
            "block_id": shard["block_id"],
            "seed": shard["seed"],
            "expert_gateway_count": len(shard["expert_gateways"]),
            "structural_homophily_low": shard["structural_homophily_low"],
            "structural_homophily_high": shard["structural_homophily_high"],
            "prior_segregation_low": shard["prior_segregation_low"],
            "prior_segregation_high": shard["prior_segregation_high"],
        }
        for shard in shards
    ]

    _write_csv(root / "runs.csv", runs)
    _write_csv(root / "exp4_cell_disruption.csv", cells)
    _write_csv(root / "exp4_homophily_segregation_interaction.csv", interactions)
    _write_csv(root / "exp4_design_audit.csv", audits)
    _write_csv(root / "belief_checkpoints.csv", belief_checkpoints)
    _write_csv(root / "lambda_checkpoints.csv", lambda_checkpoints)
    _write_csv(root / "jammer_strategy.csv", jammer_strategy)

    expected_blocks = int(manifest["expected_blocks"])
    expected_runs = int(manifest["expected_runs"])
    finite = all(int(r.get("n_nonfinite", 1)) == 0 for r in runs)
    horizon = all(
        int(r["steps_run"]) == int(r["terminal_horizon_T"])
        for r in runs
    )
    h_gate = all(
        float(a["structural_homophily_high"])
        > float(a["structural_homophily_low"]) + 0.20
        for a in audits
    )
    s_gate = all(
        float(a["prior_segregation_high"])
        > float(a["prior_segregation_low"]) + 3.0
        for a in audits
    )

    max_mse = max((float(r["mse_truth"]) for r in runs), default=float("nan"))
    max_abs_jammer_message = max(
        (abs(float(r["message_mean"])) for r in jammer_strategy),
        default=float("nan"),
    )
    max_jammer_response_gain = max(
        (float(r["response_gain"]) for r in jammer_strategy),
        default=float("nan"),
    )

    report = {
        "pass": (
            len(shards) == expected_blocks
            and len(runs) == expected_runs
            and len(cells) == expected_blocks * 4
            and len(interactions) == expected_blocks
            and finite
            and horizon
            and h_gate
            and s_gate
        ),
        "expected_blocks": expected_blocks,
        "observed_blocks": len(shards),
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "cell_contrasts": len(cells),
        "interaction_rows": len(interactions),
        "all_runs_finite": finite,
        "all_runs_fixed_horizon": horizon,
        "homophily_manipulation_gate": h_gate,
        "segregation_manipulation_gate": s_gate,
        "max_terminal_mse": max_mse,
        "max_abs_jammer_message_mean": max_abs_jammer_message,
        "max_jammer_response_gain": max_jammer_response_gain,
    }
    (root / "exp4_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def package(root: Path, design_id: str) -> Path:
    import zipfile

    names = (
        "exp4_manifest.json",
        "runs.csv",
        "exp4_cell_disruption.csv",
        "exp4_homophily_segregation_interaction.csv",
        "exp4_design_audit.csv",
        "belief_checkpoints.csv",
        "lambda_checkpoints.csv",
        "jammer_strategy.csv",
        "exp4_gate.json",
    )
    path = root.parent / f"paper_b_exp4_{design_id}_shareable.zip"
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

    design = {
        "design_version": 2,
        "experiment": "IV_homophily_x_prior_segregation",
        "scientific_code_fingerprint": scientific_code_fingerprint(),
        "software_versions": software_versions(),
        "seeds": seeds,
        "n_citizens": args.n_citizens,
        "horizon_T": args.horizon,
        "K": args.k,
        "epsilon": args.epsilon,
        "credit": args.credit,
        "elite_access_mode": "universal_expert_and_jammer",
        "peer_degree": args.peer_degree,
        "low_homophily": args.low_homophily,
        "high_homophily": args.high_homophily,
        "low_prior_group_shift": 0.0,
        "high_prior_group_shift": args.high_group_shift,
        "prior_residual_sd": args.prior_residual_sd,
        "surveillance_interval": args.surveillance_interval,
        "reliance_mode": RELIANCE_MODE,
        "jammer_states": [True, False],
        "jammer_access": "universal structural slot; neutralized under J=0",
        "fixed_group_labels": True,
    }
    design_id = canonical_hash(design)[:12]
    blocks = [
        {"seed": seed, "block_id": f"exp4__s{seed}"}
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

    root = Path(args.output_dir) / f"exp4_{design_id}"
    manifest_path = root / "exp4_manifest.json"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design_id") != design_id:
            raise RuntimeError("Existing Exp IV manifest does not match.")
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
                "peer_degree": args.peer_degree,
                "low_homophily": args.low_homophily,
                "high_homophily": args.high_homophily,
                "high_group_shift": args.high_group_shift,
                "prior_residual_sd": args.prior_residual_sd,
                "surveillance_interval": args.surveillance_interval,
            }
        )

    print("Paper B Experiment IV: homophily x prior segregation")
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
                        f"[{completed}/{len(pending)}] {result['block_id']} | "
                        f"H={result['h_low']:.3f}/{result['h_high']:.3f} | "
                        f"S={result['s_low']:.3f}/{result['s_high']:.3f} | "
                        f"max MSE={result['max_mse']:.4f}"
                    )

    report = consolidate(root, design)
    print(f"Exp IV gate: {'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(1)
    bundle = package(root, design_id)
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
