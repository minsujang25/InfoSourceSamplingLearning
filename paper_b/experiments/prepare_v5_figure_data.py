"""Prepare frozen data tables for Paper B v5 main figures.

This is a passive aggregation step only. It reads already-completed production,
phase-diagram, rank-unification, and analysis-closure outputs and writes compact
CSV tables for manuscript figures. It never instantiates the model or runs a
simulation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-root",
        default=(
            "production_results/paper_b_measurement_audit/"
            "measurement_087afed31ccf"
        ),
    )
    parser.add_argument(
        "--phase-root",
        default=(
            "production_results/paper_b_epsilon_degree_phase/"
            "phase_3f9c496208d5"
        ),
    )
    parser.add_argument(
        "--rank-root",
        default=(
            "production_results/paper_b_rank_unification/"
            "rank_da7a0299e0f9"
        ),
    )
    parser.add_argument(
        "--closure-root",
        default="production_results/paper_b_analysis_closure",
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_v5_figure_data",
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


def _summarize(
    rows: list[dict],
    keys: tuple[str, ...],
    metric: str,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(float(row[metric]))
    out = []
    for key, values in sorted(grouped.items(), key=lambda x: tuple(map(str, x[0]))):
        mean, mcse = _mean_mcse(values)
        item = {name: value for name, value in zip(keys, key)}
        item.update(
            {
                "n": len(values),
                f"{metric}_mean": mean,
                f"{metric}_mcse": mcse,
                f"{metric}_median": float(np.median(values)),
            }
        )
        out.append(item)
    return out


def _figure2(measurement_root: Path, closure_root: Path) -> tuple[list[dict], list[dict]]:
    runs = _read_csv(measurement_root / "runs.csv")
    null_iv = [
        row for row in runs
        if row.get("experiment") == "IV"
        and row.get("production_block") == "null_primary"
        and row.get("sender_regime") == "null"
    ]
    cells = _summarize(
        null_iv,
        ("homophily_level", "segregation_level", "reliance_mode"),
        "mse_truth",
    )

    gap_path = closure_root / "group_gap_cell_summary.csv"
    if gap_path.exists():
        gaps = {
            (
                row["homophily_level"],
                row["segregation_level"],
                row["reliance_mode"],
            ): row
            for row in _read_csv(gap_path)
        }
        for row in cells:
            key = (
                row["homophily_level"],
                row["segregation_level"],
                row["reliance_mode"],
            )
            gap = gaps.get(key)
            if gap:
                row["terminal_group_mean_gap"] = float(
                    gap["terminal_group_mean_gap_mean"]
                )

    checkpoints = _read_csv(measurement_root / "belief_checkpoints.csv.gz")
    target = [
        row for row in checkpoints
        if row.get("experiment") == "IV"
        and row.get("production_block") == "null_primary"
        and row.get("sender_regime") == "null"
        and row.get("homophily_level") == "high"
        and row.get("segregation_level") == "high"
        and int(float(row.get("horizon_step", 0))) in {100, 200, 300, 400}
    ]
    trajectory = _summarize(
        target,
        ("reliance_mode", "horizon_step"),
        "mse_truth",
    )
    return cells, trajectory


def _figure3(phase_root: Path) -> list[dict]:
    rows = _read_csv(phase_root / "contrast_summary.csv")
    out = []
    for row in rows:
        if row.get("contrast") != "H_x_S_MSE":
            continue
        out.append(
            {
                "epsilon": float(row["epsilon"]),
                "peer_degree": int(float(row["peer_degree"])),
                "reliance_mode": row["reliance_mode"],
                "interaction_mean": float(row["mean"]),
                "interaction_mcse": float(row["mcse"]),
                "mc95_low": float(row["mc95_low"]),
                "mc95_high": float(row["mc95_high"]),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["reliance_mode"],
            r["peer_degree"],
            r["epsilon"],
        ),
    )


def _figure4(phase_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rank_rows = _read_csv(phase_root / "initial_expert_rank_surface.csv")
    access = []
    for row in rank_rows:
        if row.get("homophily_level") != "high":
            continue
        if row.get("segregation_level") != "high":
            continue
        access.append(
            {
                "epsilon": float(row["epsilon"]),
                "peer_degree": int(float(row["peer_degree"])),
                "mean_expert_rank": float(row["mean_expert_rank"]),
                "mean_expert_acquisition_probability": float(
                    row["mean_expert_acquisition_probability"]
                ),
                "mean_expert_inclusion_probability": float(
                    row["mean_expert_inclusion_probability"]
                ),
            }
        )
    access = sorted(access, key=lambda r: (r["peer_degree"], r["epsilon"]))

    rank_by_degree = []
    for degree in sorted({r["peer_degree"] for r in access}):
        subset = [r for r in access if r["peer_degree"] == degree]
        ranks = [r["mean_expert_rank"] for r in subset]
        if max(ranks) - min(ranks) > 1e-12:
            raise RuntimeError("Initial Expert rank unexpectedly changes with epsilon.")
        rank_by_degree.append(
            {
                "peer_degree": degree,
                "mean_expert_rank": float(np.mean(ranks)),
            }
        )

    cells = _read_csv(phase_root / "cell_summary.csv")
    mse_index = {}
    for row in cells:
        if row.get("homophily_level") != "high":
            continue
        if row.get("segregation_level") != "high":
            continue
        key = (
            float(row["epsilon"]),
            int(float(row["peer_degree"])),
            row["reliance_mode"],
        )
        mse_index[key] = float(row["mse_truth_mean"])

    access_index = {
        (row["epsilon"], row["peer_degree"]): row
        for row in access
    }
    merged = []
    for (epsilon, degree, reliance), mse in sorted(
        mse_index.items(),
        key=lambda x: (x[0][2], x[0][1], x[0][0]),
    ):
        a = access_index[(epsilon, degree)]
        merged.append(
            {
                "epsilon": epsilon,
                "peer_degree": degree,
                "reliance_mode": reliance,
                "mean_expert_rank": a["mean_expert_rank"],
                "mean_expert_inclusion_probability": (
                    a["mean_expert_inclusion_probability"]
                ),
                "terminal_mse": mse,
                "log10_terminal_mse": math.log10(mse),
            }
        )
    return rank_by_degree, access, merged


def _figure5(
    measurement_root: Path,
    rank_root: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    initial = _read_csv(rank_root / "initial_rank_summary.csv")
    gateway = []
    for row in initial:
        if row.get("audit_block") != "exp2_gateway_rank":
            continue
        gateway.append(
            {
                "multiplicity": row["condition"],
                "top_gateway_share": float(row["top_target_share_mean"]),
                "best_gateway_rank": float(row["best_target_rank_mean"]),
                "gateway_acquisition_mass": float(
                    row["target_acquisition_mass_mean"]
                ),
                "gateway_inclusion_probability": float(
                    row["target_inclusion_probability_mean"]
                ),
            }
        )
    gateway = sorted(gateway, key=lambda r: r["multiplicity"])

    runs = _read_csv(measurement_root / "runs.csv")
    exp2 = [
        row for row in runs
        if row.get("experiment") == "III"
        and row.get("production_block") == "redundancy_primary"
        and row.get("sender_regime") in {"null", "fixed_biased"}
    ]
    loss = _summarize(
        exp2,
        ("redundancy_level", "reliance_mode", "sender_regime"),
        "mse_truth",
    )

    idx = {
        (
            int(float(row["seed"])),
            row["redundancy_level"],
            row["reliance_mode"],
            row["sender_regime"],
        ): float(row["mse_truth"])
        for row in exp2
    }
    seeds = sorted({int(float(row["seed"])) for row in exp2})
    damage = []
    for multiplicity in ("low", "high"):
        for reliance in ("adaptive", "frozen"):
            diffs = []
            null_values = []
            fixed_values = []
            for seed in seeds:
                null_key = (seed, multiplicity, reliance, "null")
                fixed_key = (seed, multiplicity, reliance, "fixed_biased")
                if null_key not in idx or fixed_key not in idx:
                    continue
                null_value = idx[null_key]
                fixed_value = idx[fixed_key]
                null_values.append(null_value)
                fixed_values.append(fixed_value)
                diffs.append(fixed_value - null_value)

            damage_mean, damage_mcse = _mean_mcse(diffs)
            damage.append(
                {
                    "multiplicity": multiplicity,
                    "reliance_mode": reliance,
                    "n_seeds": len(diffs),
                    "null_mse": float(np.mean(null_values)),
                    "fixed_biased_mse": float(np.mean(fixed_values)),
                    "fixed_biased_damage": damage_mean,
                    "fixed_biased_damage_mcse": damage_mcse,
                    "fixed_biased_damage_median": float(np.median(diffs)),
                }
            )
    return gateway, loss, damage


def main() -> None:
    args = parse_args()
    measurement_root = Path(args.measurement_root)
    phase_root = Path(args.phase_root)
    rank_root = Path(args.rank_root)
    closure_root = Path(args.closure_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = (
        measurement_root / "runs.csv",
        measurement_root / "belief_checkpoints.csv.gz",
        phase_root / "contrast_summary.csv",
        phase_root / "cell_summary.csv",
        phase_root / "initial_expert_rank_surface.csv",
        rank_root / "initial_rank_summary.csv",
        closure_root / "group_gap_cell_summary.csv",
        closure_root / "A_peer_sna_summary.csv",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    fig2_cells, fig2_trajectory = _figure2(measurement_root, closure_root)
    fig3_surface = _figure3(phase_root)
    fig4_rank, fig4_access, fig4_link = _figure4(phase_root)
    fig5_gateway, fig5_loss, fig5_damage = _figure5(
        measurement_root,
        rank_root,
    )

    outputs = {
        "fig2_exp1_terminal_cells.csv": fig2_cells,
        "fig2_exp1_highH_highS_trajectory.csv": fig2_trajectory,
        "fig3_phase_surface.csv": fig3_surface,
        "fig4_expert_rank_by_degree.csv": fig4_rank,
        "fig4_expert_access_surface.csv": fig4_access,
        "fig4_access_vs_terminal_mse.csv": fig4_link,
        "fig5_gateway_rank_protection.csv": fig5_gateway,
        "fig5_exp2_loss_cells.csv": fig5_loss,
        "fig5_exp2_fixed_damage.csv": fig5_damage,
    }
    for name, rows in outputs.items():
        _write_csv(out / name, rows)

    # Copy frozen supplement-facing closure/stress summaries for the rewrite.
    copy_specs = (
        (closure_root / "group_gap_cell_summary.csv", "supp_group_gap.csv"),
        (closure_root / "group_gap_contrast_summary.csv", "supp_group_gap_contrasts.csv"),
        (closure_root / "A_peer_sna_summary.csv", "supp_A_peer_sna.csv"),
        (rank_root / "initial_rank_summary.csv", "supp_rank_unification_initial.csv"),
        (rank_root / "fixed_group_channel_summary.csv", "supp_fixed_group_channels.csv"),
        (rank_root / "fixed_group_terminal_loss.csv", "supp_fixed_group_terminal_loss.csv"),
    )
    for src, name in copy_specs:
        if src.exists():
            shutil.copy2(src, out / name)

    gate = {
        "pass": bool(
            len(fig2_cells) == 8
            and len(fig2_trajectory) == 8
            and len(fig3_surface) == 30
            and len(fig4_rank) == 3
            and len(fig4_access) == 15
            and len(fig4_link) == 30
            and len(fig5_gateway) == 2
            and len(fig5_loss) == 8
            and len(fig5_damage) == 4
        ),
        "fig2_terminal_cell_rows": len(fig2_cells),
        "fig2_trajectory_rows": len(fig2_trajectory),
        "fig3_phase_rows": len(fig3_surface),
        "fig4_rank_rows": len(fig4_rank),
        "fig4_access_rows": len(fig4_access),
        "fig4_link_rows": len(fig4_link),
        "fig5_gateway_rows": len(fig5_gateway),
        "fig5_loss_rows": len(fig5_loss),
        "fig5_damage_rows": len(fig5_damage),
        "note": "Passive aggregation only; no model simulation is run.",
    }
    (out / "v5_figure_data_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    bundle = out / "paper_b_v5_figure_data_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.iterdir()):
            if path == bundle or not path.is_file():
                continue
            zf.write(path, arcname=path.name)

    if not gate["pass"]:
        raise RuntimeError(
            "v5 figure-data gate failed; inspect v5_figure_data_gate.json."
        )
    print("Paper B v5 figure-data gate: PASS")
    print(f"Review bundle: {bundle}")


if __name__ == "__main__":
    main()
