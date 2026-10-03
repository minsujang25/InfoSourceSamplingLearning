"""Prepare the final passive/theoretical closure package before Theory v5.

Inputs are frozen or behaviorally identical outputs:
1. W-channel decomposition rerun with T=10 passive logging;
2. full epsilon-by-degree phase surface.

The analytical q(m) curve is computed by deterministic quadrature.

No new behavioral treatment or parameter condition is introduced here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


EPSILONS = (0.02, 0.05, 0.10, 0.20, 0.30)
DEGREES = (2, 3, 4)
EARLY_STEPS = (10, 25)
Q_ANCHORS = {
    0.0: 0.3524163823495668,
    1.0: 0.5484596872836032,
    2.0: 0.7924032310064489,
    3.0: 0.9087963048585311,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-channel-root", required=True)
    parser.add_argument(
        "--phase-root",
        default=(
            "production_results/paper_b_epsilon_degree_phase/"
            "phase_3f9c496208d5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="production_results/paper_b_theory_closure",
    )
    parser.add_argument("--q-max", type=float, default=4.0)
    parser.add_argument("--q-step", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_mcse(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, math.nan
    return mean, float(arr.std(ddof=1) / math.sqrt(arr.size))


def early_channel_rows(w_root: Path) -> tuple[list[dict], list[dict]]:
    summary_path = w_root / "channel_checkpoint_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = read_csv(summary_path)

    keep_metrics = (
        "W_expert_channel_mean",
        "W_same_peer_channel_mean",
        "W_other_peer_channel_mean",
        "W_corrective_crosscut_channel_mean",
        "I_expert_channel_mean",
        "I_same_peer_channel_mean",
        "I_other_peer_channel_mean",
        "Q_expert_channel_mean",
        "Q_same_peer_channel_mean",
        "Q_other_peer_channel_mean",
    )

    out = []
    for row in rows:
        h = int(float(row["horizon_step"]))
        if h not in {10, 25, 400}:
            continue
        item = {
            "epsilon": float(row["epsilon"]),
            "homophily_level": row["homophily_level"],
            "horizon_step": h,
            "n": int(float(row["n"])),
        }
        for metric in keep_metrics:
            item[metric] = float(row[metric])
        out.append(item)

    # Primary high-H epsilon .05 -> .20 contrasts at early horizons.
    idx = {
        (
            float(row["epsilon"]),
            row["homophily_level"],
            int(row["horizon_step"]),
        ): row
        for row in out
    }
    contrasts = []
    metric_stems = (
        "W_expert_channel_mean",
        "W_same_peer_channel_mean",
        "W_other_peer_channel_mean",
        "I_expert_channel_mean",
        "I_same_peer_channel_mean",
        "I_other_peer_channel_mean",
        "Q_expert_channel_mean",
        "Q_same_peer_channel_mean",
        "Q_other_peer_channel_mean",
    )
    for horizon in (10, 25, 400):
        low = idx[(0.05, "high", horizon)]
        high = idx[(0.20, "high", horizon)]
        for metric in metric_stems:
            a = float(low[metric])
            b = float(high[metric])
            contrasts.append(
                {
                    "horizon_step": horizon,
                    "metric": metric,
                    "epsilon_low": 0.05,
                    "epsilon_high": 0.20,
                    "value_low": a,
                    "value_high": b,
                    "absolute_change": b - a,
                    "ratio_high_to_low": (b / a if a != 0.0 else math.nan),
                    "percent_change": (
                        100.0 * (b - a) / abs(a)
                        if a != 0.0
                        else math.nan
                    ),
                }
            )
    return sorted(
        out,
        key=lambda r: (
            r["homophily_level"],
            r["epsilon"],
            r["horizon_step"],
        ),
    ), contrasts


def attenuation_surface(phase_root: Path) -> list[dict]:
    path = phase_root / "contrast_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [
        row
        for row in read_csv(path)
        if row.get("contrast") == "H_x_S_MSE"
    ]
    idx = {
        (
            float(row["epsilon"]),
            int(float(row["peer_degree"])),
            row["reliance_mode"],
        ): row
        for row in rows
    }
    out = []
    for degree in DEGREES:
        for epsilon in EPSILONS:
            a = float(idx[(epsilon, degree, "adaptive")]["mean"])
            f = float(idx[(epsilon, degree, "frozen")]["mean"])
            if f <= 0.0:
                raise RuntimeError(
                    f"Non-positive frozen interaction at epsilon={epsilon}, d={degree}."
                )
            out.append(
                {
                    "epsilon": epsilon,
                    "top_rank_request_share": 1.0 - epsilon,
                    "peer_degree": degree,
                    "adaptive_interaction": a,
                    "frozen_interaction": f,
                    "absolute_reduction": f - a,
                    "attenuation_fraction": 1.0 - a / f,
                    "attenuation_percent": 100.0 * (1.0 - a / f),
                    "frozen_over_adaptive": (
                        f / a if a > 0.0 else math.inf
                    ),
                }
            )
    return out


def q_of_m(m: float) -> float:
    """Pr(same-group peer closer than truthful Expert) under N(0,1) residuals."""

    def integrand(e: float) -> float:
        radius = abs(m + e)
        conditional = norm.cdf(e + radius) - norm.cdf(e - radius)
        return float(conditional * norm.pdf(e))

    value, _ = quad(
        integrand,
        -10.0,
        10.0,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=300,
    )
    return float(value)


def q_curve(q_max: float, q_step: float) -> tuple[list[dict], list[dict]]:
    grid = np.arange(0.0, q_max + q_step / 2.0, q_step)
    rows = [
        {
            "group_center_separation_m": float(m),
            "same_group_peer_beats_expert_probability_q": q_of_m(float(m)),
        }
        for m in grid
    ]
    anchors = [
        {
            "group_center_separation_m": m,
            "q_numeric": q_of_m(m),
            "q_target": target,
            "absolute_error": abs(q_of_m(m) - target),
        }
        for m, target in Q_ANCHORS.items()
    ]
    return rows, anchors


def plot_early_channels(rows: list[dict], out: Path) -> None:
    high = [r for r in rows if r["homophily_level"] == "high"]
    for stem, ylabel in (
        ("Q_expert_channel_mean", "Cumulative Expert precision Q"),
        ("W_expert_channel_mean", "Expert precision share W"),
        ("I_expert_channel_mean", "Expert inclusion frequency I"),
    ):
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        for horizon, marker in ((10, "o"), (25, "s"), (400, "^")):
            d = sorted(
                [r for r in high if r["horizon_step"] == horizon],
                key=lambda r: r["epsilon"],
            )
            ax.plot(
                [r["epsilon"] for r in d],
                [r[stem] for r in d],
                marker=marker,
                label=f"T={horizon}",
            )
        ax.set_xlabel(r"Exploration parameter $\epsilon$")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"early_{stem}.pdf", bbox_inches="tight")
        fig.savefig(out / f"early_{stem}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_attenuation(rows: list[dict], out: Path) -> None:
    matrix = np.empty((len(DEGREES), len(EPSILONS)), dtype=float)
    idx = {
        (int(r["peer_degree"]), float(r["epsilon"])): r
        for r in rows
    }
    for i, degree in enumerate(DEGREES):
        for j, epsilon in enumerate(EPSILONS):
            matrix[i, j] = float(
                idx[(degree, epsilon)]["attenuation_percent"]
            )

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=100.0)
    ax.set_xticks(np.arange(len(EPSILONS)))
    ax.set_xticklabels([f"{e:.2f}" for e in EPSILONS])
    ax.set_yticks(np.arange(len(DEGREES)))
    ax.set_yticklabels([str(d) for d in DEGREES])
    ax.set_xlabel(r"Exploration parameter $\epsilon$")
    ax.set_ylabel(r"Peer degree $d$")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value < 45 else "black"
            ax.text(
                j,
                i,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adaptive attenuation of frozen H×S penalty (%)")
    fig.tight_layout()
    fig.savefig(out / "adaptive_attenuation_surface.pdf", bbox_inches="tight")
    fig.savefig(
        out / "adaptive_attenuation_surface.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_q_curve(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = [r["group_center_separation_m"] for r in rows]
    y = [r["same_group_peer_beats_expert_probability_q"] for r in rows]
    ax.plot(x, y)
    anchors_x = list(Q_ANCHORS)
    anchors_y = [q_of_m(m) for m in anchors_x]
    ax.scatter(anchors_x, anchors_y)
    ax.set_xlabel(r"Group-center separation from truth $m$")
    ax.set_ylabel(r"$q(m)=Pr(\mathrm{peer\ closer\ than\ Expert})$")
    ax.set_ylim(0.0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "analytical_q_curve.pdf", bbox_inches="tight")
    fig.savefig(out / "analytical_q_curve.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    w_root = Path(args.w_channel_root)
    phase_root = Path(args.phase_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    w_gate_path = w_root / "channel_gate.json"
    if not w_gate_path.exists():
        raise FileNotFoundError(w_gate_path)
    w_gate = json.loads(w_gate_path.read_text(encoding="utf-8"))
    if not w_gate.get("pass"):
        raise RuntimeError("W-channel behavioral-identity gate is not PASS.")

    early, early_contrasts = early_channel_rows(w_root)
    attenuation = attenuation_surface(phase_root)
    q_rows, q_anchors = q_curve(float(args.q_max), float(args.q_step))

    write_csv(out / "early_channel_summary.csv", early)
    write_csv(out / "early_channel_epsilon_contrasts.csv", early_contrasts)
    write_csv(out / "adaptive_attenuation_surface.csv", attenuation)
    write_csv(out / "analytical_q_curve.csv", q_rows)
    write_csv(out / "analytical_q_anchors.csv", q_anchors)

    plot_early_channels(early, out)
    plot_attenuation(attenuation, out)
    plot_q_curve(q_rows, out)

    early_primary = [
        r for r in early
        if r["homophily_level"] == "high"
        and r["horizon_step"] in EARLY_STEPS
        and float(r["epsilon"]) in {0.05, 0.10, 0.20}
    ]
    expected_early = 2 * 3
    early_complete = len(early_primary) == expected_early

    attenuation_complete = len(attenuation) == 15
    denominators_positive = all(
        float(r["frozen_interaction"]) > 0.0
        for r in attenuation
    )
    q_anchor_error = max(float(r["absolute_error"]) for r in q_anchors)

    gate = {
        "pass": bool(
            w_gate.get("pass")
            and early_complete
            and attenuation_complete
            and denominators_positive
            and q_anchor_error <= 5e-4
        ),
        "w_channel_gate_pass": bool(w_gate.get("pass")),
        "w_channel_design_id": w_gate.get("design_id"),
        "early_primary_rows_expected": expected_early,
        "early_primary_rows_observed": len(early_primary),
        "attenuation_rows_expected": 15,
        "attenuation_rows_observed": len(attenuation),
        "all_frozen_denominators_positive": denominators_positive,
        "max_q_anchor_absolute_error": q_anchor_error,
        "note": (
            "Task 1 is a behaviorally identical passive measurement rerun; "
            "Tasks 2-3 are post-processing/analytical only."
        ),
    }
    (out / "theory_closure_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    review = out / "paper_b_theory_closure_review.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.iterdir()):
            if path == review or not path.is_file():
                continue
            zf.write(path, arcname=path.name)
        plan = (
            Path(__file__).resolve().parents[1]
            / "THEORY_CLOSURE_PLAN_2026-10-03.md"
        )
        zf.write(plan, arcname=plan.name)

    if not gate["pass"]:
        raise RuntimeError(
            "Theory-closure gate failed; inspect theory_closure_gate.json."
        )
    print("Paper B theory-closure gate: PASS")
    print(f"Review bundle: {review}")


if __name__ == "__main__":
    main()
