"""Phase-A measurement audit using frozen canonical production outputs.

No simulation is run. This script reconstructs structural uniform-use baselines
from the frozen source-map generators and joins them to canonical run outputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_b.experiments.run_local_diagnostic import initial_beliefs
from paper_b.measurement import structural_uniform_metrics
from paper_b.structural_designs import (
    balanced_fixed_group_ids,
    exp3_redundancy_source_maps,
    exp4_homophily_source_maps,
    exp4_initial_beliefs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production-root",
        required=True,
        help="Canonical production directory containing runs.csv and raw outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_measurement_audit_phase_a",
    )
    return parser.parse_args()


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, "r", newline="", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with _open_csv(path) as handle:
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


def _f(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _i(value) -> int:
    return int(float(value))


def _run_key(row: dict) -> tuple:
    return (
        row["experiment"],
        _i(row["seed"]),
        row["production_block"],
        row.get("homophily_level", ""),
        row.get("segregation_level", ""),
        row.get("redundancy_level", ""),
        row["reliance_mode"],
        row["sender_regime"],
    )


def _structural_baseline(seed: int, row: dict, n_citizens: int = 100) -> dict:
    if row["experiment"] == "III":
        design = exp3_redundancy_source_maps(
            seed=seed,
            n_citizens=n_citizens,
            peer_degree=2,
            expert_access_share=0.10,
        )
        redundancy = row["redundancy_level"]
        return structural_uniform_metrics(
            design[redundancy],
            gateway_positions=set(int(x) for x in design["expert_gateways"]),
        )

    groups = balanced_fixed_group_ids(
        seed=seed,
        n_citizens=n_citizens,
    )
    design = exp4_homophily_source_maps(
        seed=seed,
        n_citizens=n_citizens,
        group_ids=groups,
        peer_degree=2,
        low_homophily=0.50,
        high_homophily=0.90,
    )
    return structural_uniform_metrics(
        design[row["homophily_level"]],
        group_ids=groups,
        gateway_positions=set(int(x) for x in design["expert_gateways"]),
    )


def _initial_mse(seed: int, row: dict, n_citizens: int = 100) -> float:
    if row["experiment"] == "III":
        values = np.asarray(
            initial_beliefs(
                regime="flat",
                seed=seed,
                n_citizens=n_citizens,
            )[2:],
            dtype=float,
        )
        return float(np.mean(values**2))

    groups = balanced_fixed_group_ids(
        seed=seed,
        n_citizens=n_citizens,
    )
    values = np.asarray(
        exp4_initial_beliefs(
            seed=seed,
            n_citizens=n_citizens,
            group_ids=groups,
            segregation=row["segregation_level"],
            high_group_shift=3.0,
            residual_sd=1.0,
        )[2:],
        dtype=float,
    )
    return float(np.mean(values**2))


def _mean(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.fmean(values)) if values else math.nan


def _median(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(values)) if values else math.nan


def _group_terminal_metrics(belief_path: Path) -> dict[tuple, dict]:
    values = defaultdict(lambda: defaultdict(list))
    with _open_csv(belief_path) as handle:
        for row in csv.DictReader(handle):
            key = _run_key(row)
            group = _i(row["citizen_group"])
            values[key][group].append(_f(row["terminal_mu_theta"]))

    out = {}
    for key, by_group in values.items():
        means = {
            group: _mean(group_values)
            for group, group_values in by_group.items()
        }
        row = {
            "terminal_group_minus_mean": means.get(-1, math.nan),
            "terminal_group_plus_mean": means.get(1, math.nan),
        }
        if -1 in means and 1 in means:
            row["terminal_group_mean_gap"] = abs(means[1] - means[-1])
        else:
            row["terminal_group_mean_gap"] = math.nan
        out[key] = row
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.production_root)
    runs_path = root / "runs.csv"
    beliefs_path = root / "terminal_beliefs.csv.gz"
    checkpoint_path = root / "checkpoint_cell_summary.csv"

    for required in (runs_path, beliefs_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest_path = root / "production_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_citizens = int(manifest["n_citizens"])

    runs = _read_csv(runs_path)
    group_metrics = _group_terminal_metrics(beliefs_path)

    augmented = []
    structural_cache = {}
    for row in runs:
        seed = _i(row["seed"])
        structure_key = (
            row["experiment"],
            seed,
            row.get("homophily_level", ""),
            row.get("redundancy_level", ""),
        )
        if structure_key not in structural_cache:
            structural_cache[structure_key] = _structural_baseline(
                seed,
                row,
                n_citizens=n_citizens,
            )
        baseline = structural_cache[structure_key]

        out = dict(row)
        out.update(baseline)

        initial_mse = _initial_mse(
            seed,
            row,
            n_citizens=n_citizens,
        )
        out["initial_mse"] = initial_mse
        terminal_mse = _f(row["mse_truth"])
        out["terminal_mse_fraction_of_initial"] = (
            terminal_mse / initial_mse if initial_mse > 0.0 else math.nan
        )

        h_lambda = _f(row.get("effective_homophily"))
        h_a = _f(baseline.get("A_uniform_structural_homophily"))
        out["Lambda_minus_A_homophily"] = (
            h_lambda - h_a
            if math.isfinite(h_lambda) and math.isfinite(h_a)
            else math.nan
        )

        lambda_hhi = _f(row.get("effective_incoming_hhi"))
        a_hhi = _f(baseline.get("A_uniform_incoming_hhi"))
        out["Lambda_minus_A_incoming_hhi"] = (
            lambda_hhi - a_hhi
            if math.isfinite(lambda_hhi) and math.isfinite(a_hhi)
            else math.nan
        )

        lambda_top5 = _f(row.get("effective_incoming_top5_share"))
        a_top5 = _f(baseline.get("A_uniform_incoming_top5_share"))
        out["Lambda_minus_A_incoming_top5_share"] = (
            lambda_top5 - a_top5
            if math.isfinite(lambda_top5) and math.isfinite(a_top5)
            else math.nan
        )

        lambda_gateway = _f(row.get("gateway_incoming_reliance_share"))
        a_gateway = _f(baseline.get("A_uniform_gateway_total_share"))
        out["Lambda_minus_A_gateway_total_share"] = (
            lambda_gateway - a_gateway
            if math.isfinite(lambda_gateway) and math.isfinite(a_gateway)
            else math.nan
        )

        out.update(group_metrics.get(_run_key(row), {}))
        augmented.append(out)

    # Cell summaries.
    groups = defaultdict(list)
    for row in augmented:
        key = (
            row["experiment"],
            row["production_block"],
            row.get("homophily_level", ""),
            row.get("segregation_level", ""),
            row.get("redundancy_level", ""),
            row["reliance_mode"],
            row["sender_regime"],
        )
        groups[key].append(row)

    summary_metrics = (
        "mse_truth",
        "rmse_truth",
        "mae_truth",
        "initial_mse",
        "terminal_mse_fraction_of_initial",
        "A_uniform_structural_homophily",
        "effective_homophily",
        "Lambda_minus_A_homophily",
        "A_uniform_incoming_hhi",
        "effective_incoming_hhi",
        "Lambda_minus_A_incoming_hhi",
        "A_uniform_incoming_top5_share",
        "effective_incoming_top5_share",
        "Lambda_minus_A_incoming_top5_share",
        "A_uniform_gateway_total_share",
        "A_uniform_gateway_peer_conditional_share",
        "gateway_incoming_reliance_share",
        "Lambda_minus_A_gateway_total_share",
        "terminal_group_minus_mean",
        "terminal_group_plus_mean",
        "terminal_group_mean_gap",
    )
    cell_summary = []
    for key, rows in sorted(groups.items()):
        result = {
            "experiment": key[0],
            "production_block": key[1],
            "homophily_level": key[2],
            "segregation_level": key[3],
            "redundancy_level": key[4],
            "reliance_mode": key[5],
            "sender_regime": key[6],
            "n": len(rows),
        }
        for metric in summary_metrics:
            vals = [_f(r.get(metric)) for r in rows]
            result[f"{metric}_mean"] = _mean(vals)
            result[f"{metric}_median"] = _median(vals)
        cell_summary.append(result)

    # Total-loss 2x2 cells: null vs fixed-biased x adaptive vs frozen.
    total_loss = [
        row for row in cell_summary
        if row["sender_regime"] in {"null", "fixed_biased"}
    ]

    # Existing canonical trajectory checkpoints. Period 100 is T=101 in the
    # frozen production; exact T=100 is added in the passive measurement rerun.
    checkpoints = _read_csv(checkpoint_path)
    trajectory = []
    for row in checkpoints:
        period = _i(row["period"])
        if period not in {100, 199, 299, 399}:
            continue
        if row["sender_regime"] != "null":
            continue
        if row["experiment"] == "IV":
            if not (
                row["homophily_level"] == "high"
                and row["segregation_level"] == "high"
            ):
                continue
        elif row["experiment"] == "III":
            if row["redundancy_level"] not in {"low", "high"}:
                continue
        trajectory.append(dict(row))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    _write_csv(out_root / "A_lambda_seed_metrics.csv", augmented)
    _write_csv(out_root / "A_lambda_cell_summary.csv", cell_summary)
    _write_csv(out_root / "total_loss_2x2.csv", total_loss)
    _write_csv(out_root / "canonical_horizon_trajectory.csv", trajectory)

    notes = {
        "source_production_root": str(root),
        "n_runs": len(runs),
        "structural_uniform_definition": "U^A_ij=A_ij/outdegree_i",
        "gateway_total_denominator": "all source mass",
        "gateway_peer_conditional_denominator": (
            "peer opportunities renormalized within ego, then averaged"
        ),
        "dominant_skeleton_baseline": (
            "not assigned: uniform A creates tied top-1 edges"
        ),
        "canonical_checkpoint_caveat": (
            "existing production stores period 100 (=T101), not exact T100"
        ),
    }
    (out_root / "phase_a_notes.json").write_text(
        json.dumps(notes, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    bundle = out_root / "paper_b_measurement_phase_a.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "A_lambda_seed_metrics.csv",
            "A_lambda_cell_summary.csv",
            "total_loss_2x2.csv",
            "canonical_horizon_trajectory.csv",
            "phase_a_notes.json",
        ):
            path = out_root / name
            if path.exists():
                zf.write(path, arcname=name)

    print(f"Phase-A bundle: {bundle}")


if __name__ == "__main__":
    main()
