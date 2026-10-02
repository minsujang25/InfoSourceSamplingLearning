"""Reconstruct the Experiment-IV Expert-rank mechanism without learning runs.

The frozen pre-disruption ranking is deterministic conditional on:
    structural source map + initial beliefs + null-last rule.

This script reconstructs that ranking exactly for canonical matched seeds and
optionally joins the existing passive-measurement ZIP to report the adaptive
Expert-acquisition path.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.InfoSourceSamplingLearning import recursive_rank_probabilities
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


EXPERT_POS = 0
JAMMER_POS = 1
CITIZEN_START = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--credit", type=int, default=20)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument(
        "--measurement-review",
        default="",
        help="Optional paper_b_measurement_review_*.zip for adaptive path join.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_expert_rank_audit",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else math.nan


def _source_mu(
    source_pos: int,
    initial_beliefs: list[float],
) -> float:
    return float(initial_beliefs[int(source_pos)])


def _expert_rank_record(
    *,
    seed: int,
    homophily_level: str,
    ego: int,
    sources: list[int],
    group_ids: dict[int, int],
    initial_beliefs: list[float],
    epsilon: float,
    credit: int,
) -> dict:
    prior_mu = float(initial_beliefs[int(ego)])
    peers = [int(source) for source in sources if int(source) >= CITIZEN_START]

    # Canonical null semantics: Jammer is present structurally but always moved
    # to the final behavioral rank.
    non_null = [int(source) for source in sources if int(source) != JAMMER_POS]
    scored = [
        (
            abs(_source_mu(source, initial_beliefs) - prior_mu),
            int(source),
        )
        for source in non_null
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    ordered = [source for _, source in scored] + [JAMMER_POS]
    expert_rank = ordered.index(EXPERT_POS) + 1

    probs = recursive_rank_probabilities(len(ordered), float(epsilon))
    expert_prob = float(probs[expert_rank - 1])
    inclusion_prob = float(1.0 - (1.0 - expert_prob) ** int(credit))

    expert_distance = abs(prior_mu)
    same_group_peers = sum(
        int(group_ids[int(peer)] == group_ids[int(ego)])
        for peer in peers
    )
    peers_outrank_expert = sum(
        int(
            abs(float(initial_beliefs[int(peer)]) - prior_mu)
            < expert_distance
        )
        for peer in peers
    )

    edge_same_outrank = []
    edge_other_outrank = []
    for peer in peers:
        outranks = int(
            abs(float(initial_beliefs[int(peer)]) - prior_mu)
            < expert_distance
        )
        if group_ids[int(peer)] == group_ids[int(ego)]:
            edge_same_outrank.append(outranks)
        else:
            edge_other_outrank.append(outranks)

    return {
        "seed": int(seed),
        "homophily_level": homophily_level,
        "ego": int(ego),
        "ego_group": int(group_ids[int(ego)]),
        "prior_mu": prior_mu,
        "same_group_peer_count": int(same_group_peers),
        "peer_outrank_expert_count": int(peers_outrank_expert),
        "expert_rank": int(expert_rank),
        "expert_acquisition_probability": expert_prob,
        "expert_expected_requests": float(int(credit) * expert_prob),
        "expert_inclusion_probability": inclusion_prob,
        "same_peer_outrank_sum": int(sum(edge_same_outrank)),
        "same_peer_edge_count": int(len(edge_same_outrank)),
        "other_peer_outrank_sum": int(sum(edge_other_outrank)),
        "other_peer_edge_count": int(len(edge_other_outrank)),
    }


def reconstruct(args: argparse.Namespace) -> list[dict]:
    rows = []
    seeds = range(int(args.seed_start), int(args.seed_start) + int(args.seeds))
    for seed in seeds:
        groups = balanced_fixed_group_ids(
            seed=seed,
            n_citizens=int(args.n_citizens),
        )
        blueprint = exp4_homophily_source_maps(
            seed=seed,
            n_citizens=int(args.n_citizens),
            group_ids=groups,
            peer_degree=int(args.peer_degree),
            low_homophily=float(args.low_homophily),
            high_homophily=float(args.high_homophily),
        )
        initial = exp4_initial_beliefs(
            seed=seed,
            n_citizens=int(args.n_citizens),
            group_ids=groups,
            segregation="high",
            high_group_shift=float(args.high_group_shift),
            residual_sd=float(args.prior_residual_sd),
        )

        for h in ("low", "high"):
            for ego, sources in blueprint[h].items():
                rows.append(
                    _expert_rank_record(
                        seed=seed,
                        homophily_level=h,
                        ego=int(ego),
                        sources=[int(x) for x in sources],
                        group_ids=groups,
                        initial_beliefs=initial,
                        epsilon=float(args.epsilon),
                        credit=int(args.credit),
                    )
                )
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    cells = []
    conditional = []
    group_rows = []

    for h in ("low", "high"):
        subset = [row for row in rows if row["homophily_level"] == h]
        n = len(subset)
        same_edges = sum(row["same_peer_edge_count"] for row in subset)
        other_edges = sum(row["other_peer_edge_count"] for row in subset)
        cell = {
            "homophily_level": h,
            "n_citizens_x_seeds": n,
            "mean_same_group_peer_count": _mean(
                [r["same_group_peer_count"] for r in subset]
            ),
            "mean_peer_outrank_expert_count": _mean(
                [r["peer_outrank_expert_count"] for r in subset]
            ),
            "mean_expert_rank": _mean([r["expert_rank"] for r in subset]),
            "mean_expert_acquisition_probability": _mean(
                [r["expert_acquisition_probability"] for r in subset]
            ),
            "mean_expert_expected_requests": _mean(
                [r["expert_expected_requests"] for r in subset]
            ),
            "mean_expert_inclusion_probability": _mean(
                [r["expert_inclusion_probability"] for r in subset]
            ),
            "same_group_peer_outrank_rate": (
                sum(r["same_peer_outrank_sum"] for r in subset) / same_edges
                if same_edges else math.nan
            ),
            "other_group_peer_outrank_rate": (
                sum(r["other_peer_outrank_sum"] for r in subset) / other_edges
                if other_edges else math.nan
            ),
        }
        for rank in range(1, int(max(r["expert_rank"] for r in subset)) + 1):
            cell[f"expert_rank_{rank}_share"] = float(
                np.mean([r["expert_rank"] == rank for r in subset])
            )
        cells.append(cell)

        for same_count in range(0, 3):
            ss = [
                row for row in subset
                if row["same_group_peer_count"] == same_count
            ]
            if not ss:
                continue
            item = {
                "homophily_level": h,
                "same_group_peer_count": same_count,
                "n": len(ss),
                "mean_expert_rank": _mean([r["expert_rank"] for r in ss]),
            }
            for rank in range(1, 4):
                item[f"expert_rank_{rank}_share"] = float(
                    np.mean([r["expert_rank"] == rank for r in ss])
                )
            conditional.append(item)

        for group in (-1, 1):
            ss = [row for row in subset if row["ego_group"] == group]
            item = {
                "homophily_level": h,
                "ego_group": group,
                "n": len(ss),
                "mean_expert_rank": _mean([r["expert_rank"] for r in ss]),
                "mean_expert_acquisition_probability": _mean(
                    [r["expert_acquisition_probability"] for r in ss]
                ),
            }
            for rank in range(1, 4):
                item[f"expert_rank_{rank}_share"] = float(
                    np.mean([r["expert_rank"] == rank for r in ss])
                )
            group_rows.append(item)

    return cells, conditional, group_rows


def _measurement_path(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path) as zf:
        raw = zf.read("lambda_checkpoints.csv.gz")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8")
        reader = csv.DictReader(text)
        accum: dict[tuple, list[float]] = defaultdict(list)
        for row in reader:
            if row.get("experiment") != "IV":
                continue
            if row.get("production_block") != "null_primary":
                continue
            if row.get("segregation_level") != "high":
                continue
            key = (
                row.get("homophily_level"),
                row.get("reliance_mode"),
                int(float(row["period"])),
            )
            accum[key].append(float(row["expert_reliance"]))

    out = []
    for (h, reliance, period), values in sorted(accum.items()):
        out.append(
            {
                "homophily_level": h,
                "reliance_mode": reliance,
                "period": int(period),
                "horizon_step": int(period) + 1,
                "mean_expert_acquisition_probability": _mean(values),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    rows = reconstruct(args)
    cells, conditional, group_rows = summarize(rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "expert_rank_micro.csv", rows)
    _write_csv(output / "expert_rank_cell_summary.csv", cells)
    _write_csv(
        output / "expert_rank_by_same_group_peer_count.csv",
        conditional,
    )
    _write_csv(output / "expert_rank_by_group.csv", group_rows)

    path_rows = []
    comparison = {}
    if args.measurement_review:
        path_rows = _measurement_path(Path(args.measurement_review))
        _write_csv(output / "canonical_expert_acquisition_path.csv", path_rows)

        period0 = {
            (row["homophily_level"], row["reliance_mode"]):
            row["mean_expert_acquisition_probability"]
            for row in path_rows
            if row["period"] == 0
        }
        reconstructed = {
            row["homophily_level"]:
            row["mean_expert_acquisition_probability"]
            for row in cells
        }
        diffs = []
        for h in ("low", "high"):
            for reliance in ("adaptive", "frozen"):
                observed = float(period0[(h, reliance)])
                expected = float(reconstructed[h])
                diffs.append(abs(observed - expected))
        comparison = {
            "period0_reconstruction_max_abs_diff": max(diffs),
            "period0_reconstruction_pass": max(diffs) <= 1e-12,
        }

    probabilities = recursive_rank_probabilities(
        int(args.peer_degree) + 2,
        float(args.epsilon),
    )
    gate = {
        "canonical_seed_match": bool(
            int(args.seeds) == 500 and int(args.seed_start) == 6001
        ),
        "n_rows": len(rows),
        "epsilon": float(args.epsilon),
        "credit": int(args.credit),
        "peer_degree": int(args.peer_degree),
        "rank_probabilities": [float(x) for x in probabilities],
        "expected_requests_by_rank": [
            float(int(args.credit) * x) for x in probabilities
        ],
        "inclusion_probabilities_by_rank": [
            float(1.0 - (1.0 - x) ** int(args.credit))
            for x in probabilities
        ],
        **comparison,
    }
    (output / "expert_rank_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    bundle = output / "paper_b_expert_rank_audit.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "expert_rank_cell_summary.csv",
            "expert_rank_by_same_group_peer_count.csv",
            "expert_rank_by_group.csv",
            "canonical_expert_acquisition_path.csv",
            "expert_rank_gate.json",
        ):
            path = output / name
            if path.exists():
                zf.write(path, arcname=name)

    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"Audit bundle: {bundle}")


if __name__ == "__main__":
    main()
