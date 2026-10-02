"""Full epsilon-by-degree scope-condition phase diagram for Experiment 1.

Reuses four existing specifications:
- d2/e=.05 canonical production
- d2/e=.10 mechanism robustness
- d2/e=.20 mechanism robustness
- d4/e=.05 mechanism robustness

Runs only the remaining eleven specifications across the full
H x S x adaptive/frozen null factorial for 500 matched seeds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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

from model.InfoSourceSamplingLearning import recursive_rank_probabilities
from paper_b.experiments.run_mechanism_robustness import (
    _run_spec_seed,
    _valid_shard,
)
from paper_b.experiments.run_matched_pilot import canonical_hash
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


EPSILONS = (0.02, 0.05, 0.10, 0.20, 0.30)
DEGREES = (2, 3, 4)
H_LEVELS = ("low", "high")
S_LEVELS = ("low", "high")
RELIANCE_MODES = ("adaptive", "frozen")

EXISTING_SPECS = {
    (0.05, 2): "canonical",
    (0.10, 2): "mechanism",
    (0.20, 2): "mechanism",
    (0.05, 4): "mechanism",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--surveillance-interval", type=int, default=5)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument("--numerical-min-sd", type=float, default=1e-8)
    parser.add_argument(
        "--canonical-root",
        default=(
            "production_results/paper_b_canonical/"
            "production_c9d09daad143"
        ),
    )
    parser.add_argument(
        "--mechanism-root",
        default=(
            "production_results/paper_b_mechanism_robustness/"
            "mechanism_9b5c35506618"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_epsilon_degree_phase",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _degree_nesting_check(
    *,
    seeds: list[int],
    n_citizens: int,
    low_homophily: float,
    high_homophily: float,
) -> bool:
    for seed in seeds:
        groups = balanced_fixed_group_ids(
            seed=seed,
            n_citizens=n_citizens,
        )
        maps = {}
        for degree in DEGREES:
            maps[degree] = exp4_homophily_source_maps(
                seed=seed,
                n_citizens=n_citizens,
                group_ids=groups,
                peer_degree=degree,
                low_homophily=low_homophily,
                high_homophily=high_homophily,
            )
        for h in H_LEVELS:
            for ego in maps[2][h]:
                peers2 = {
                    int(x)
                    for x in maps[2][h][ego]
                    if int(x) >= 2
                }
                for degree in (3, 4):
                    peers = {
                        int(x)
                        for x in maps[degree][h][ego]
                        if int(x) >= 2
                    }
                    if not peers2.issubset(peers):
                        return False
    return True


def _initial_rank_surface(
    *,
    seeds: list[int],
    n_citizens: int,
    credit: int,
    low_homophily: float,
    high_homophily: float,
    high_group_shift: float,
    prior_residual_sd: float,
) -> list[dict]:
    raw = defaultdict(list)

    for seed in seeds:
        groups = balanced_fixed_group_ids(
            seed=seed,
            n_citizens=n_citizens,
        )
        initial_by_s = {
            s: exp4_initial_beliefs(
                seed=seed,
                n_citizens=n_citizens,
                group_ids=groups,
                segregation=s,
                high_group_shift=high_group_shift,
                residual_sd=prior_residual_sd,
            )
            for s in S_LEVELS
        }

        for degree in DEGREES:
            blueprint = exp4_homophily_source_maps(
                seed=seed,
                n_citizens=n_citizens,
                group_ids=groups,
                peer_degree=degree,
                low_homophily=low_homophily,
                high_homophily=high_homophily,
            )
            for h in H_LEVELS:
                for s in S_LEVELS:
                    initial = initial_by_s[s]
                    for ego, sources in blueprint[h].items():
                        prior = float(initial[int(ego)])
                        non_null = [
                            int(source)
                            for source in sources
                            if int(source) != 1
                        ]
                        scored = []
                        for source in non_null:
                            source_mu = float(initial[source])
                            scored.append(
                                (
                                    abs(source_mu - prior),
                                    int(source),
                                )
                            )
                        scored.sort()
                        ranking = [source for _, source in scored]
                        expert_rank = ranking.index(0) + 1
                        raw[(degree, h, s)].append(expert_rank)

    out = []
    for degree in DEGREES:
        num_sources = degree + 2
        for epsilon in EPSILONS:
            probs = recursive_rank_probabilities(
                num_sources,
                epsilon,
            )
            for h in H_LEVELS:
                for s in S_LEVELS:
                    ranks = raw[(degree, h, s)]
                    row = {
                        "epsilon": epsilon,
                        "peer_degree": degree,
                        "homophily_level": h,
                        "segregation_level": s,
                        "n_citizen_seed": len(ranks),
                        "mean_expert_rank": float(np.mean(ranks)),
                        "mean_expert_acquisition_probability": float(
                            np.mean([probs[rank - 1] for rank in ranks])
                        ),
                        "mean_expert_inclusion_probability": float(
                            np.mean(
                                [
                                    1.0
                                    - (1.0 - probs[rank - 1]) ** credit
                                    for rank in ranks
                                ]
                            )
                        ),
                    }
                    for rank in range(1, num_sources):
                        row[f"expert_rank_{rank}_share"] = float(
                            np.mean([value == rank for value in ranks])
                        )
                    out.append(row)
    return out


def _new_specs() -> list[dict]:
    specs = []
    for degree in DEGREES:
        for epsilon in EPSILONS:
            if (float(epsilon), int(degree)) in EXISTING_SPECS:
                continue
            specs.append(
                {
                    "robustness_block": "phase_diagram",
                    "epsilon": float(epsilon),
                    "peer_degree": int(degree),
                    "spec_label": f"epsilon_{epsilon:.2f}_d{degree}",
                }
            )
    assert len(specs) == 11
    return specs


def _all_specs() -> list[tuple[float, int]]:
    return [
        (float(epsilon), int(degree))
        for degree in DEGREES
        for epsilon in EPSILONS
    ]


def _canonical_reference_rows(root: Path) -> list[dict]:
    path = root / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out = []
    for raw in _read_csv(path):
        if raw.get("experiment") != "IV":
            continue
        if raw.get("production_block") != "null_primary":
            continue
        row = dict(raw)
        row["epsilon"] = 0.05
        row["peer_degree"] = 2
        row["spec_label"] = "epsilon_0.05_d2"
        row["phase_source"] = "canonical"
        out.append(row)
    return out


def _mechanism_reference_rows(root: Path) -> list[dict]:
    path = root / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    allowed = {
        "epsilon_0.10_d2",
        "epsilon_0.20_d2",
        "epsilon_0.05_d4",
    }
    out = []
    for raw in _read_csv(path):
        if raw.get("spec_label") not in allowed:
            continue
        row = dict(raw)
        row["phase_source"] = "mechanism"
        out.append(row)
    return out


def _normalize_new_rows(payloads: list[dict]) -> list[dict]:
    out = []
    for payload in payloads:
        for raw in payload["runs"]:
            row = dict(raw)
            row["phase_source"] = "new"
            out.append(row)
    return out


def _key(row: dict) -> tuple:
    return (
        int(float(row["seed"])),
        float(row["epsilon"]),
        int(float(row["peer_degree"])),
        row["homophily_level"],
        row["segregation_level"],
        row["reliance_mode"],
    )


def _validate_combined(rows: list[dict], seeds: list[int]) -> dict:
    expected = set()
    for seed in seeds:
        for epsilon, degree in _all_specs():
            for h in H_LEVELS:
                for s in S_LEVELS:
                    for reliance in RELIANCE_MODES:
                        expected.add(
                            (
                                seed,
                                epsilon,
                                degree,
                                h,
                                s,
                                reliance,
                            )
                        )

    observed = {_key(row) for row in rows}
    duplicates = len(rows) - len(observed)
    missing = expected - observed
    extra = observed - expected

    return {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "unique_keys": len(observed),
        "duplicate_count": duplicates,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "pass": bool(
            not duplicates
            and not missing
            and not extra
        ),
    }


def _cell_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    keys = (
        "epsilon",
        "peer_degree",
        "homophily_level",
        "segregation_level",
        "reliance_mode",
    )
    metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "belief_variance",
        "squared_displacement",
        "expert_reliance",
        "W_expert_precision_share",
        "W_effective_homophily",
        "W_dominant_expert_reach_share",
        "W_dominant_same_group_cycle_share",
    )
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)

    out = []
    for key, members in sorted(grouped.items(), key=lambda x: tuple(map(str,x[0]))):
        item = {
            name: value
            for name, value in zip(keys, key)
        }
        item["n"] = len(members)
        for metric in metrics:
            values = []
            for member in members:
                if metric not in member:
                    continue
                try:
                    value = float(member[metric])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            item[f"{metric}_mean"] = (
                float(np.mean(values)) if values else math.nan
            )
            item[f"{metric}_median"] = (
                float(np.median(values)) if values else math.nan
            )
        out.append(item)
    return out


def _seed_contrasts(rows: list[dict]) -> list[dict]:
    index = {_key(row): row for row in rows}
    seeds = sorted({int(float(row["seed"])) for row in rows})
    out = []

    for seed in seeds:
        for epsilon, degree in _all_specs():
            for reliance in RELIANCE_MODES:
                def get(h, s):
                    return index[
                        (
                            seed,
                            epsilon,
                            degree,
                            h,
                            s,
                            reliance,
                        )
                    ]

                hh = float(get("high", "high")["mse_truth"])
                lh = float(get("low", "high")["mse_truth"])
                hl = float(get("high", "low")["mse_truth"])
                ll = float(get("low", "low")["mse_truth"])

                out.append(
                    {
                        "seed": seed,
                        "epsilon": epsilon,
                        "peer_degree": degree,
                        "reliance_mode": reliance,
                        "contrast": "H_x_S_MSE",
                        "value": (hh - lh) - (hl - ll),
                    }
                )
                out.append(
                    {
                        "seed": seed,
                        "epsilon": epsilon,
                        "peer_degree": degree,
                        "reliance_mode": reliance,
                        "contrast": "highS_H_penalty_MSE",
                        "value": hh - lh,
                    }
                )
    return out


def _contrast_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                float(row["epsilon"]),
                int(row["peer_degree"]),
                row["reliance_mode"],
                row["contrast"],
            )
        ].append(float(row["value"]))

    out = []
    for (epsilon, degree, reliance, contrast), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        mcse = float(arr.std(ddof=1) / math.sqrt(arr.size))
        out.append(
            {
                "epsilon": epsilon,
                "peer_degree": degree,
                "reliance_mode": reliance,
                "contrast": contrast,
                "n_seeds": int(arr.size),
                "mean": mean,
                "mcse": mcse,
                "mc95_low": mean - 1.96 * mcse,
                "mc95_high": mean + 1.96 * mcse,
                "median": float(np.median(arr)),
                "positive_share": float(np.mean(arr > 0.0)),
                "negative_share": float(np.mean(arr < 0.0)),
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
        and int(args.k) == 1
        and int(args.surveillance_interval) == 5
        and math.isclose(float(args.low_homophily), 0.50)
        and math.isclose(float(args.high_homophily), 0.90)
        and math.isclose(float(args.high_group_shift), 3.0)
        and math.isclose(float(args.prior_residual_sd), 1.0)
        and math.isclose(float(args.numerical_min_sd), 1e-8)
    )

    nesting_pass = _degree_nesting_check(
        seeds=seeds,
        n_citizens=int(args.n_citizens),
        low_homophily=float(args.low_homophily),
        high_homophily=float(args.high_homophily),
    )
    if not nesting_pass:
        raise RuntimeError(
            "d=3/d=4 opportunity sets do not preserve canonical d=2 peers."
        )

    initial_rank_surface = _initial_rank_surface(
        seeds=seeds,
        n_citizens=int(args.n_citizens),
        credit=int(args.credit),
        low_homophily=float(args.low_homophily),
        high_homophily=float(args.high_homophily),
        high_group_shift=float(args.high_group_shift),
        prior_residual_sd=float(args.prior_residual_sd),
    )

    specs = _new_specs()
    design = {
        "purpose": "Paper B epsilon-by-degree phase diagram",
        "seeds": seeds,
        "epsilons": list(EPSILONS),
        "peer_degrees": list(DEGREES),
        "new_specs": specs,
        "existing_specs": [
            {
                "epsilon": epsilon,
                "peer_degree": degree,
                "source": source,
            }
            for (epsilon, degree), source in EXISTING_SPECS.items()
        ],
        "n_citizens": int(args.n_citizens),
        "horizon_T": int(args.horizon),
        "credit": int(args.credit),
        "K": int(args.k),
        "surveillance_interval": int(args.surveillance_interval),
        "low_homophily": float(args.low_homophily),
        "high_homophily": float(args.high_homophily),
        "high_group_shift": float(args.high_group_shift),
        "prior_residual_sd": float(args.prior_residual_sd),
        "numerical_min_sd": float(args.numerical_min_sd),
        "tau_social": 1.0,
        "peer_evidence_mode": "source_posterior",
        "frozen_ranking_mode": "pre_disruption",
        "sender_regime": "null",
        "canonical_base_match": canonical_match,
        "degree_nesting_pass": nesting_pass,
    }
    design_id = canonical_hash(design)[:12]
    design["design_id"] = design_id

    root = Path(args.output_dir) / f"phase_{design_id}"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; use --resume.")
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / "phase_manifest.json").write_text(
        json.dumps(design, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tasks = []
    for spec in specs:
        for seed in seeds:
            path = (
                root / "shards"
                / f"{spec['spec_label']}__s{seed}.json.gz"
            )
            if args.resume and _valid_shard(path, design_id):
                continue
            tasks.append(
                {
                    "design_id": design_id,
                    "run_root": str(root),
                    "seed": seed,
                    "spec": spec,
                    "n_citizens": int(args.n_citizens),
                    "horizon": int(args.horizon),
                    "credit": int(args.credit),
                    "k": int(args.k),
                    "surveillance_interval": int(
                        args.surveillance_interval
                    ),
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
            futures = [pool.submit(_run_spec_seed, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if i % max(int(args.progress_every), 1) == 0:
                    print(
                        f"[{i}/{len(futures)}] "
                        f"{result['spec_label']} seed={result['seed']}",
                        flush=True,
                    )

    payloads = []
    for path in sorted((root / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("design_id") == design_id:
            payloads.append(payload)

    expected_shards = len(specs) * len(seeds)
    expected_new_runs = expected_shards * 8
    if len(payloads) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} shards, got {len(payloads)}."
        )

    new_rows = _normalize_new_rows(payloads)
    canonical_rows = _canonical_reference_rows(Path(args.canonical_root))
    mechanism_rows = _mechanism_reference_rows(Path(args.mechanism_root))
    combined = canonical_rows + mechanism_rows + new_rows

    validation = _validate_combined(combined, seeds)
    finite = all(
        math.isfinite(float(row["mse_truth"]))
        for row in combined
    )

    gate = {
        "pass": bool(
            canonical_match
            and nesting_pass
            and validation["pass"]
            and finite
            and len(new_rows) == expected_new_runs
        ),
        "design_id": design_id,
        "canonical_base_match": canonical_match,
        "degree_nesting_pass": nesting_pass,
        "new_spec_count": len(specs),
        "observed_new_shards": len(payloads),
        "expected_new_shards": expected_shards,
        "observed_new_runs": len(new_rows),
        "expected_new_runs": expected_new_runs,
        "combined_validation": validation,
        "all_combined_terminal_mse_finite": finite,
    }
    (root / "phase_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cells = _cell_summary(combined)
    seed_contrasts = _seed_contrasts(combined)
    contrasts = _contrast_summary(seed_contrasts)

    _write_csv(root / "combined_runs.csv.gz", combined, gzip_output=True)
    _write_csv(root / "cell_summary.csv", cells)
    _write_csv(root / "initial_expert_rank_surface.csv", initial_rank_surface)
    _write_csv(root / "seed_contrasts.csv.gz", seed_contrasts, gzip_output=True)
    _write_csv(root / "contrast_summary.csv", contrasts)

    plan = Path(__file__).resolve().parents[1] / "EPSILON_DEGREE_PHASE_PLAN.md"
    review = root / f"paper_b_phase_review_{design_id}.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "phase_manifest.json",
            "phase_gate.json",
            "cell_summary.csv",
            "contrast_summary.csv",
            "initial_expert_rank_surface.csv",
        ):
            zf.write(root / name, arcname=name)
        zf.write(plan, arcname="EPSILON_DEGREE_PHASE_PLAN.md")

    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"Review bundle: {review}")


if __name__ == "__main__":
    main()
