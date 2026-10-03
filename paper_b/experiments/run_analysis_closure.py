"""Final passive analysis-closure audit for Social Networks Paper B.

No model simulation is run. The audit reuses terminal citizen beliefs from the
canonical passive measurement rerun and reconstructs the deterministic
Experiment-1 structural opportunity networks from the frozen source-map
constructor.

Outputs:
1. terminal between-group belief-gap diagnostics for Experiment 1 null cells;
2. basic SNA descriptives for the citizen-to-citizen opportunity network A.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    citizen_positions,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
    realized_prior_segregation,
)


H_LEVELS = ("low", "high")
RELIANCE_MODES = ("adaptive", "frozen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--n-citizens", type=int, default=100)
    parser.add_argument("--peer-degree", type=int, default=2)
    parser.add_argument("--low-homophily", type=float, default=0.50)
    parser.add_argument("--high-homophily", type=float, default=0.90)
    parser.add_argument("--high-group-shift", type=float, default=3.0)
    parser.add_argument("--prior-residual-sd", type=float, default=1.0)
    parser.add_argument(
        "--measurement-root",
        default=(
            "production_results/paper_b_measurement_audit/"
            "measurement_087afed31ccf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_analysis_closure",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
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


def _mean_mcse(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, math.nan
    return mean, float(arr.std(ddof=1) / math.sqrt(arr.size))


def _group_gap_seed_rows(
    *,
    measurement_root: Path,
    seeds: list[int],
    n_citizens: int,
    high_group_shift: float,
    prior_residual_sd: float,
) -> list[dict]:
    runs_path = measurement_root / "runs.csv"
    beliefs_path = measurement_root / "terminal_beliefs.csv.gz"
    for path in (runs_path, beliefs_path):
        if not path.exists():
            raise FileNotFoundError(path)

    runs = _read_csv(runs_path)
    beliefs = _read_csv(beliefs_path)

    wanted = {}
    for row in runs:
        if row.get("experiment") != "IV":
            continue
        if row.get("production_block") != "null_primary":
            continue
        if row.get("sender_regime") != "null":
            continue
        key = (
            int(float(row["seed"])),
            row["homophily_level"],
            row["segregation_level"],
            row["reliance_mode"],
        )
        wanted[key] = row

    by_run_group: dict[tuple, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in beliefs:
        if row.get("experiment") != "IV":
            continue
        if row.get("production_block") != "null_primary":
            continue
        if row.get("sender_regime") != "null":
            continue
        key = (
            int(float(row["seed"])),
            row["homophily_level"],
            row["segregation_level"],
            row["reliance_mode"],
        )
        if key not in wanted:
            continue
        by_run_group[key][int(float(row["citizen_group"]))].append(
            float(row["terminal_mu_theta"])
        )

    out = []
    seed_set = set(seeds)
    initial_cache = {}
    for key in sorted(wanted):
        seed, h, s, reliance = key
        if seed not in seed_set:
            continue
        groups = by_run_group.get(key, {})
        if -1 not in groups or 1 not in groups:
            raise RuntimeError(f"Missing group terminal beliefs for {key}.")
        minus = np.asarray(groups[-1], dtype=float)
        plus = np.asarray(groups[1], dtype=float)
        if minus.size + plus.size != n_citizens:
            raise RuntimeError(
                f"Expected {n_citizens} citizens for {key}, "
                f"got {minus.size + plus.size}."
            )

        init_key = (seed, s)
        if init_key not in initial_cache:
            group_ids = balanced_fixed_group_ids(
                seed=seed,
                n_citizens=n_citizens,
            )
            initial = exp4_initial_beliefs(
                seed=seed,
                n_citizens=n_citizens,
                group_ids=group_ids,
                segregation=s,
                high_group_shift=high_group_shift,
                residual_sd=prior_residual_sd,
            )
            initial_cache[init_key] = realized_prior_segregation(
                initial,
                group_ids=group_ids,
            )

        terminal_minus = float(minus.mean())
        terminal_plus = float(plus.mean())
        terminal_gap = float(abs(terminal_plus - terminal_minus))
        initial_gap = float(initial_cache[init_key])
        out.append(
            {
                "seed": seed,
                "homophily_level": h,
                "segregation_level": s,
                "reliance_mode": reliance,
                "n_group_minus": int(minus.size),
                "n_group_plus": int(plus.size),
                "terminal_group_minus_mean": terminal_minus,
                "terminal_group_plus_mean": terminal_plus,
                "terminal_group_mean_gap": terminal_gap,
                "initial_group_mean_gap": initial_gap,
                "terminal_gap_fraction_of_initial": (
                    terminal_gap / initial_gap
                    if initial_gap > 0.0
                    else math.nan
                ),
            }
        )
    return out


def _group_gap_cell_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    keys = ("homophily_level", "segregation_level", "reliance_mode")
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)

    out = []
    metrics = (
        "terminal_group_minus_mean",
        "terminal_group_plus_mean",
        "terminal_group_mean_gap",
        "initial_group_mean_gap",
        "terminal_gap_fraction_of_initial",
    )
    for key, members in sorted(grouped.items()):
        item = {name: value for name, value in zip(keys, key)}
        item["n_seeds"] = len(members)
        for metric in metrics:
            vals = [
                float(member[metric])
                for member in members
                if math.isfinite(float(member[metric]))
            ]
            if vals:
                mean, mcse = _mean_mcse(vals)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_mcse"] = mcse
                item[f"{metric}_median"] = float(np.median(vals))
            else:
                item[f"{metric}_mean"] = math.nan
                item[f"{metric}_mcse"] = math.nan
                item[f"{metric}_median"] = math.nan
        out.append(item)
    return out


def _group_gap_contrasts(rows: list[dict]) -> list[dict]:
    index = {
        (
            int(row["seed"]),
            row["homophily_level"],
            row["segregation_level"],
            row["reliance_mode"],
        ): float(row["terminal_group_mean_gap"])
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    raw = []
    for seed in seeds:
        for reliance in RELIANCE_MODES:
            def get(h: str, s: str) -> float:
                return index[(seed, h, s, reliance)]

            raw.append(
                {
                    "seed": seed,
                    "reliance_mode": reliance,
                    "contrast": "H_x_S_terminal_group_gap",
                    "value": (
                        (get("high", "high") - get("low", "high"))
                        - (get("high", "low") - get("low", "low"))
                    ),
                }
            )
            raw.append(
                {
                    "seed": seed,
                    "reliance_mode": reliance,
                    "contrast": "highS_H_penalty_terminal_group_gap",
                    "value": get("high", "high") - get("low", "high"),
                }
            )

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["reliance_mode"], row["contrast"])].append(
            float(row["value"])
        )
    out = []
    for (reliance, contrast), values in sorted(grouped.items()):
        mean, mcse = _mean_mcse(values)
        out.append(
            {
                "reliance_mode": reliance,
                "contrast": contrast,
                "n_seeds": len(values),
                "mean": mean,
                "mcse": mcse,
                "mc95_low": mean - 1.96 * mcse,
                "mc95_high": mean + 1.96 * mcse,
                "median": float(np.median(values)),
                "positive_share": float(np.mean(np.asarray(values) > 0.0)),
                "negative_share": float(np.mean(np.asarray(values) < 0.0)),
            }
        )
    return out


def _a_peer_sna_seed_rows(
    *,
    seeds: list[int],
    n_citizens: int,
    peer_degree: int,
    low_homophily: float,
    high_homophily: float,
) -> list[dict]:
    out = []
    positions = citizen_positions(n_citizens)
    node_set = set(positions)

    for seed in seeds:
        groups = balanced_fixed_group_ids(
            seed=seed,
            n_citizens=n_citizens,
        )
        maps = exp4_homophily_source_maps(
            seed=seed,
            n_citizens=n_citizens,
            group_ids=groups,
            peer_degree=peer_degree,
            low_homophily=low_homophily,
            high_homophily=high_homophily,
        )
        for h in H_LEVELS:
            graph = nx.DiGraph()
            graph.add_nodes_from(positions)
            internal = 0
            external = 0
            for ego, sources in maps[h].items():
                for source in sources:
                    source = int(source)
                    if source not in node_set:
                        continue
                    graph.add_edge(int(ego), source)
                    if groups[int(ego)] == groups[source]:
                        internal += 1
                    else:
                        external += 1

            outdegrees = np.asarray(
                [graph.out_degree(node) for node in positions],
                dtype=float,
            )
            indegrees = np.asarray(
                [graph.in_degree(node) for node in positions],
                dtype=float,
            )
            total = internal + external
            reciprocity = (
                nx.reciprocity(graph)
                if graph.number_of_edges()
                else math.nan
            )
            undirected = graph.to_undirected()
            out.append(
                {
                    "seed": seed,
                    "homophily_level": h,
                    "n_citizens": n_citizens,
                    "n_peer_edges": graph.number_of_edges(),
                    "peer_outdegree_mean": float(outdegrees.mean()),
                    "peer_outdegree_min": int(outdegrees.min()),
                    "peer_outdegree_max": int(outdegrees.max()),
                    "peer_indegree_mean": float(indegrees.mean()),
                    "peer_indegree_sd": float(indegrees.std(ddof=0)),
                    "peer_reciprocity": float(reciprocity),
                    "peer_undirected_transitivity": float(
                        nx.transitivity(undirected)
                    ),
                    "peer_same_group_share": (
                        float(internal / total) if total else math.nan
                    ),
                    "peer_ei_index": (
                        float((external - internal) / total)
                        if total
                        else math.nan
                    ),
                }
            )
    return out


def _a_peer_sna_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["homophily_level"]].append(row)
    metrics = (
        "n_peer_edges",
        "peer_outdegree_mean",
        "peer_indegree_mean",
        "peer_indegree_sd",
        "peer_reciprocity",
        "peer_undirected_transitivity",
        "peer_same_group_share",
        "peer_ei_index",
    )
    out = []
    for h, members in sorted(grouped.items()):
        item = {"homophily_level": h, "n_seeds": len(members)}
        for metric in metrics:
            vals = [float(row[metric]) for row in members]
            mean, mcse = _mean_mcse(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_mcse"] = mcse
            item[f"{metric}_median"] = float(np.median(vals))
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    seeds = list(
        range(int(args.seed_start), int(args.seed_start) + int(args.seeds))
    )
    measurement_root = Path(args.measurement_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_rows = _group_gap_seed_rows(
        measurement_root=measurement_root,
        seeds=seeds,
        n_citizens=int(args.n_citizens),
        high_group_shift=float(args.high_group_shift),
        prior_residual_sd=float(args.prior_residual_sd),
    )
    group_summary = _group_gap_cell_summary(group_rows)
    group_contrasts = _group_gap_contrasts(group_rows)
    sna_rows = _a_peer_sna_seed_rows(
        seeds=seeds,
        n_citizens=int(args.n_citizens),
        peer_degree=int(args.peer_degree),
        low_homophily=float(args.low_homophily),
        high_homophily=float(args.high_homophily),
    )
    sna_summary = _a_peer_sna_summary(sna_rows)

    expected_group_rows = int(args.seeds) * 8
    expected_sna_rows = int(args.seeds) * 2
    canonical_match = bool(
        int(args.seeds) == 500
        and int(args.seed_start) == 6001
        and int(args.n_citizens) == 100
        and int(args.peer_degree) == 2
        and math.isclose(float(args.low_homophily), 0.50)
        and math.isclose(float(args.high_homophily), 0.90)
        and math.isclose(float(args.high_group_shift), 3.0)
        and math.isclose(float(args.prior_residual_sd), 1.0)
    )
    outdegree_ok = all(
        int(row["peer_outdegree_min"]) == int(args.peer_degree)
        and int(row["peer_outdegree_max"]) == int(args.peer_degree)
        for row in sna_rows
    )
    finite_group = all(
        math.isfinite(float(row["terminal_group_mean_gap"]))
        for row in group_rows
    )
    finite_sna = all(
        math.isfinite(float(row[metric]))
        for row in sna_rows
        for metric in (
            "peer_indegree_sd",
            "peer_reciprocity",
            "peer_undirected_transitivity",
            "peer_same_group_share",
            "peer_ei_index",
        )
    )
    gate = {
        "pass": bool(
            canonical_match
            and len(group_rows) == expected_group_rows
            and len(sna_rows) == expected_sna_rows
            and outdegree_ok
            and finite_group
            and finite_sna
        ),
        "canonical_design_match": canonical_match,
        "expected_group_gap_rows": expected_group_rows,
        "observed_group_gap_rows": len(group_rows),
        "expected_A_sna_rows": expected_sna_rows,
        "observed_A_sna_rows": len(sna_rows),
        "peer_outdegree_exact": outdegree_ok,
        "all_group_gaps_finite": finite_group,
        "all_A_sna_metrics_finite": finite_sna,
        "measurement_root": str(measurement_root),
        "note": "Passive post-processing only; no model simulation is run.",
    }

    _write_csv(output_dir / "group_gap_seed.csv", group_rows)
    _write_csv(output_dir / "group_gap_cell_summary.csv", group_summary)
    _write_csv(output_dir / "group_gap_contrast_summary.csv", group_contrasts)
    _write_csv(output_dir / "A_peer_sna_seed.csv", sna_rows)
    _write_csv(output_dir / "A_peer_sna_summary.csv", sna_summary)
    (output_dir / "analysis_closure_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    bundle = output_dir / "paper_b_analysis_closure_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "analysis_closure_gate.json",
            "group_gap_seed.csv",
            "group_gap_cell_summary.csv",
            "group_gap_contrast_summary.csv",
            "A_peer_sna_seed.csv",
            "A_peer_sna_summary.csv",
        ):
            zf.write(output_dir / name, arcname=name)

    if not gate["pass"]:
        raise RuntimeError(
            "Analysis-closure gate failed; inspect analysis_closure_gate.json."
        )
    print("Analysis-closure gate: PASS")
    print(f"Review bundle: {bundle}")


if __name__ == "__main__":
    main()
