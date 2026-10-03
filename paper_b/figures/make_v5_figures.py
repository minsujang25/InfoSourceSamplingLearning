"""Generate draft publication panels for Paper B v5.

Reads only the frozen figure-data CSVs produced by
prepare_v5_figure_data.py. No model simulation is run.

Each manuscript panel is written as its own PDF and PNG so panel composition
can be reviewed separately and assembled later in LaTeX.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="production_results/paper_b_v5_figure_data",
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_v5_figures",
    )
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    # Preserve literal strings such as sender_regime == "null".
    return pd.read_csv(path, keep_default_na=False)


def _save(fig, root: Path, stem: str) -> None:
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _check_gate(data_dir: Path) -> None:
    gate_path = data_dir / "v5_figure_data_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("pass"):
        raise RuntimeError("Frozen v5 figure-data gate is not PASS.")


def figure2a(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig2_exp1_terminal_cells.csv")
    order = [
        ("low", "low"),
        ("high", "low"),
        ("low", "high"),
        ("high", "high"),
    ]
    labels = ["Low H\nLow S", "High H\nLow S", "Low H\nHigh S", "High H\nHigh S"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(order), dtype=float)
    for offset, mode, marker in (
        (-0.07, "adaptive", "o"),
        (0.07, "frozen", "s"),
    ):
        ys, es = [], []
        for h, s in order:
            row = df[
                (df["homophily_level"] == h)
                & (df["segregation_level"] == s)
                & (df["reliance_mode"] == mode)
            ].iloc[0]
            ys.append(float(row["mse_truth_mean"]))
            es.append(float(row["mse_truth_mcse"]))
        ax.errorbar(
            x + offset,
            ys,
            yerr=es,
            marker=marker,
            linestyle="none",
            capsize=3,
            label=mode.capitalize(),
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Terminal MSE")
    ax.set_xlabel("Structural homophily and prior segregation")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig2a_exp1_terminal_mse")


def figure2b(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig2_exp1_highH_highS_trajectory.csv")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for mode, marker in (("adaptive", "o"), ("frozen", "s")):
        d = df[df["reliance_mode"] == mode].sort_values("horizon_step")
        ax.errorbar(
            d["horizon_step"],
            d["mse_truth_mean"],
            yerr=d["mse_truth_mcse"],
            marker=marker,
            linewidth=1.5,
            capsize=3,
            label=mode.capitalize(),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MSE (high H, high S)")
    ax.set_xticks([100, 200, 300, 400])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig2b_exp1_highH_highS_trajectory")


def _phase_matrix(df: pd.DataFrame, mode: str, value: str) -> tuple[np.ndarray, list[float], list[int]]:
    d = df[df["reliance_mode"] == mode].copy()
    eps = sorted(float(x) for x in d["epsilon"].unique())
    degrees = sorted(int(x) for x in d["peer_degree"].unique())
    matrix = np.empty((len(degrees), len(eps)), dtype=float)
    for i, degree in enumerate(degrees):
        for j, epsilon in enumerate(eps):
            row = d[
                (d["peer_degree"].astype(int) == degree)
                & np.isclose(d["epsilon"].astype(float), epsilon)
            ].iloc[0]
            matrix[i, j] = float(row[value])
    return matrix, eps, degrees


def figure3(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig3_phase_surface.csv")
    all_log = np.log10(df["interaction_mean"].astype(float).to_numpy())
    vmin, vmax = float(all_log.min()), float(all_log.max())

    for mode in ("adaptive", "frozen"):
        raw, eps, degrees = _phase_matrix(df, mode, "interaction_mean")
        logm = np.log10(raw)

        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        im = ax.imshow(logm, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(eps)))
        ax.set_xticklabels([f"{e:.2f}" for e in eps])
        ax.set_yticks(np.arange(len(degrees)))
        ax.set_yticklabels([str(d) for d in degrees])
        ax.set_xlabel(r"Exploration parameter $\epsilon$")
        ax.set_ylabel(r"Peer degree $d$")
        ax.set_title(mode.capitalize())

        midpoint = (vmin + vmax) / 2.0
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                value = raw[i, j]
                txt = f"{value:.3g}"
                # Use default black/white text only for legibility.
                text_color = "white" if logm[i, j] < midpoint else "black"
                ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\log_{10}$ H$\times$S terminal-MSE interaction")
        fig.tight_layout()
        _save(fig, out, f"fig3_{mode}_phase_surface")


def figure4a(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig4_expert_rank_by_degree.csv").sort_values("peer_degree")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(df["peer_degree"], df["mean_expert_rank"], marker="o")
    ax.set_xticks(df["peer_degree"])
    ax.set_xlabel(r"Peer degree $d$")
    ax.set_ylabel("Mean initial Expert rank (high H, high S)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig4a_expert_rank_by_degree")


def figure4b(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig4_expert_access_surface.csv").copy()
    eps = sorted(float(x) for x in df["epsilon"].unique())
    degrees = sorted(int(x) for x in df["peer_degree"].unique())
    matrix = np.empty((len(degrees), len(eps)), dtype=float)
    for i, degree in enumerate(degrees):
        for j, epsilon in enumerate(eps):
            row = df[
                (df["peer_degree"].astype(int) == degree)
                & np.isclose(df["epsilon"].astype(float), epsilon)
            ].iloc[0]
            matrix[i, j] = float(row["mean_expert_inclusion_probability"])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(eps)))
    ax.set_xticklabels([f"{e:.2f}" for e in eps])
    ax.set_yticks(np.arange(len(degrees)))
    ax.set_yticklabels([str(d) for d in degrees])
    ax.set_xlabel(r"Exploration parameter $\epsilon$")
    ax.set_ylabel(r"Peer degree $d$")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < 0.45 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Initial Expert inclusion probability")
    fig.tight_layout()
    _save(fig, out, "fig4b_expert_inclusion_surface")


def figure4c(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig4_access_vs_terminal_mse.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for mode, marker in (("adaptive", "o"), ("frozen", "s")):
        d = df[df["reliance_mode"] == mode]
        ax.scatter(
            d["mean_expert_inclusion_probability"],
            d["terminal_mse"],
            marker=marker,
            label=mode.capitalize(),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Initial Expert inclusion probability")
    ax.set_ylabel("Terminal MSE (high H, high S)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig4c_expert_access_vs_terminal_mse")


def figure5a(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig5_gateway_rank_protection.csv")
    df = df.set_index("multiplicity").loc[["low", "high"]].reset_index()
    metrics = [
        ("top_gateway_share", "Top-ranked gateway share"),
        ("gateway_acquisition_mass", "Gateway acquisition mass"),
        ("gateway_inclusion_probability", "Gateway inclusion probability"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    y = np.arange(len(metrics), dtype=float)
    for offset, multiplicity, marker in (
        (-0.08, "low", "o"),
        (0.08, "high", "s"),
    ):
        row = df[df["multiplicity"] == multiplicity].iloc[0]
        values = [float(row[m]) for m, _ in metrics]
        ax.scatter(
            values,
            y + offset,
            marker=marker,
            label=f"{multiplicity.capitalize()} multiplicity",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accessibility measure (0–1)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig5a_gateway_accessibility")


def figure5b(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig5_gateway_rank_protection.csv")
    df = df.set_index("multiplicity").loc[["low", "high"]].reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot(
        [0, 1],
        df["best_gateway_rank"].astype(float),
        marker="o",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low multiplicity", "High multiplicity"])
    ax.set_ylabel("Mean best gateway rank")
    ax.set_ylim(0.9, 1.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig5b_best_gateway_rank")


def figure5c(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig5_exp2_loss_cells.csv")
    d = df[df["sender_regime"] == "null"].copy()

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.array([0.0, 1.0])
    for mode, marker in (("adaptive", "o"), ("frozen", "s")):
        m = d[d["reliance_mode"] == mode].set_index("redundancy_level").loc[["low", "high"]]
        ax.errorbar(
            x,
            m["mse_truth_mean"].astype(float),
            yerr=m["mse_truth_mcse"].astype(float),
            marker=marker,
            linewidth=1.5,
            capsize=3,
            label=mode.capitalize(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low multiplicity", "High multiplicity"])
    ax.set_ylabel("Terminal MSE, null sender")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig5c_exp2_null_mse")


def figure5d(data_dir: Path, out: Path) -> None:
    df = _read(data_dir / "fig5_exp2_fixed_damage.csv")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.array([0.0, 1.0])
    for mode, marker in (("adaptive", "o"), ("frozen", "s")):
        m = df[df["reliance_mode"] == mode].set_index("multiplicity").loc[["low", "high"]]
        ax.errorbar(
            x,
            m["fixed_biased_damage"].astype(float),
            yerr=m["fixed_biased_damage_mcse"].astype(float),
            marker=marker,
            linewidth=1.5,
            capsize=3,
            label=mode.capitalize(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low multiplicity", "High multiplicity"])
    ax.set_ylabel("Excess MSE (fixed-biased − null)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig5d_exp2_fixed_biased_damage")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _check_gate(data_dir)

    figure2a(data_dir, out)
    figure2b(data_dir, out)
    figure3(data_dir, out)
    figure4a(data_dir, out)
    figure4b(data_dir, out)
    figure4c(data_dir, out)
    figure5a(data_dir, out)
    figure5b(data_dir, out)
    figure5c(data_dir, out)
    figure5d(data_dir, out)

    expected = (
        "fig2a_exp1_terminal_mse",
        "fig2b_exp1_highH_highS_trajectory",
        "fig3_adaptive_phase_surface",
        "fig3_frozen_phase_surface",
        "fig4a_expert_rank_by_degree",
        "fig4b_expert_inclusion_surface",
        "fig4c_expert_access_vs_terminal_mse",
        "fig5a_gateway_accessibility",
        "fig5b_best_gateway_rank",
        "fig5c_exp2_null_mse",
        "fig5d_exp2_fixed_biased_damage",
    )
    missing = [
        stem
        for stem in expected
        if not (out / f"{stem}.pdf").exists()
        or not (out / f"{stem}.png").exists()
    ]
    gate = {
        "pass": not missing,
        "panel_count": len(expected),
        "missing_panels": missing,
        "note": "Passive plotting only; no model simulation is run.",
    }
    (out / "v5_figure_plot_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if missing:
        raise RuntimeError(f"Missing draft figure panels: {missing}")

    bundle = out / "paper_b_v5_draft_figures_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.iterdir()):
            if path == bundle or not path.is_file():
                continue
            zf.write(path, arcname=path.name)

    print("Paper B v5 draft figure gate: PASS")
    print(f"Panels written to: {out}")
    print(f"Review bundle: {bundle}")


if __name__ == "__main__":
    main()
