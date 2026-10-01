"""Combine frozen Exp IIIb base-50 and extension-50 bundles.

This module never reruns simulations. It verifies that the two shareable
bundles are scientifically compatible, keeps the original and extension cohorts
separate, and writes a cumulative N=100 precision summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import zipfile
from pathlib import Path


FIXED_FIELDS = (
    "initial_regime",
    "n_citizens",
    "n_gateways",
    "n_relays",
    "n_focals",
    "horizon_T",
    "K",
    "epsilon",
    "credit",
    "surveillance_interval",
    "numerical_min_sd",
    "path_structures",
    "reliance_modes",
    "jammer_states",
    "scientific_code_fingerprint",
    "decision_rule_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-source",
        default="local_results/paper_b_exp3b_path_independence",
        help="Base-50 ZIP or directory containing exactly one Exp IIIb ZIP.",
    )
    parser.add_argument(
        "--extension-source",
        default="local_results/paper_b_exp3b_extension50",
        help="Extension-50 ZIP or directory containing exactly one Exp IIIb ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_results/paper_b_exp3b_cumulative100",
    )
    parser.add_argument("--expected-base-seeds", default="4001-4050")
    parser.add_argument("--expected-extension-seeds", default="4051-4100")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seed_spec(value: str) -> list[int]:
    value = value.strip()
    if not value:
        raise ValueError("Seed specification cannot be empty.")
    if "," not in value and "-" in value:
        left, right = value.split("-", 1)
        start, stop = int(left), int(right)
        if stop < start:
            raise ValueError("Seed range must be increasing.")
        return list(range(start, stop + 1))
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seed specification must be nonempty and unique.")
    return seeds


def resolve_bundle(source: str) -> Path:
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ValueError(f"Expected ZIP bundle, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    matches = sorted(path.glob("paper_b_exp3b_*_shareable.zip"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Exp IIIb ZIP under {path}, found {len(matches)}."
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(bundle: Path, name: str) -> dict:
    with zipfile.ZipFile(bundle) as archive:
        with archive.open(name) as handle:
            return json.loads(handle.read().decode("utf-8"))


def read_csv(bundle: Path, name: str) -> list[dict]:
    with zipfile.ZipFile(bundle) as archive:
        with archive.open(name) as handle:
            text = handle.read().decode("utf-8").splitlines()
    return list(csv.DictReader(text))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns: list[str] = []
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


def _values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _mean(rows: list[dict], key: str) -> float:
    return float(statistics.fmean(_values(rows, key)))


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(_values(rows, key)))


def _mcse(rows: list[dict], key: str) -> float:
    vals = _values(rows, key)
    if len(vals) < 2:
        return float("nan")
    return float(statistics.stdev(vals) / math.sqrt(len(vals)))


def _mc95(rows: list[dict], key: str) -> tuple[float, float]:
    mean = _mean(rows, key)
    se = _mcse(rows, key)
    if not math.isfinite(se):
        return float("nan"), float("nan")
    return float(mean - 1.96 * se), float(mean + 1.96 * se)


def _trimmed_mean(rows: list[dict], key: str, proportion: float = 0.05) -> float:
    vals = sorted(_values(rows, key))
    if not vals:
        return float("nan")
    trim = int(math.floor(len(vals) * proportion))
    kept = vals[trim : len(vals) - trim] if trim else vals
    return float(statistics.fmean(kept))


def _share_negative(rows: list[dict], key: str) -> float:
    vals = _values(rows, key)
    return float(sum(v < 0.0 for v in vals) / len(vals))


def cohort_summary(path_rows: list[dict], activation_rows: list[dict]) -> dict:
    adaptive = [r for r in path_rows if r["reliance_mode"] == "adaptive"]
    frozen = [r for r in path_rows if r["reliance_mode"] == "frozen"]
    if not adaptive or not frozen:
        raise RuntimeError("Both adaptive and frozen path contrasts are required.")

    key = "independent_minus_shared_delta_mse"
    focal_key = "independent_minus_shared_focal_delta_mse"
    low, high = _mc95(adaptive, key)
    focal_low, focal_high = _mc95(adaptive, focal_key)

    j0 = _mean(adaptive, "independent_minus_shared_mse_J0")
    j1 = _mean(adaptive, "independent_minus_shared_mse_J1")
    path_mean = _mean(adaptive, key)
    baseline_to_active_ratio = (
        abs(j0) / abs(j1) if abs(j1) > 0.0 else float("inf")
    )

    return {
        "n_seeds": len(adaptive),
        "adaptive_population_path_effect_mean": path_mean,
        "adaptive_population_path_effect_median": _median(adaptive, key),
        "adaptive_population_path_effect_mcse": _mcse(adaptive, key),
        "adaptive_population_path_effect_mc95_low": low,
        "adaptive_population_path_effect_mc95_high": high,
        "adaptive_population_path_effect_trimmed_mean_5pct": _trimmed_mean(
            adaptive, key
        ),
        "adaptive_population_path_effect_share_negative": _share_negative(
            adaptive, key
        ),
        "adaptive_population_J1_difference_mean": j1,
        "adaptive_population_J0_difference_mean": j0,
        "baseline_to_active_abs_ratio": baseline_to_active_ratio,
        "adaptive_focal_path_effect_mean": _mean(adaptive, focal_key),
        "adaptive_focal_path_effect_median": _median(adaptive, focal_key),
        "adaptive_focal_path_effect_mcse": _mcse(adaptive, focal_key),
        "adaptive_focal_path_effect_mc95_low": focal_low,
        "adaptive_focal_path_effect_mc95_high": focal_high,
        "adaptive_focal_path_effect_share_negative": _share_negative(
            adaptive, focal_key
        ),
        "frozen_population_path_effect_mean": _mean(frozen, key),
        "activation_interaction_mean": _mean(
            activation_rows, "adaptive_minus_frozen_path_effect"
        ),
        "activation_interaction_mcse": _mcse(
            activation_rows, "adaptive_minus_frozen_path_effect"
        ),
        "direction_negative": bool(path_mean < 0.0),
        "mc95_entirely_below_zero": bool(math.isfinite(high) and high < 0.0),
        "j1_absolute_loss_lower": bool(j1 < 0.0),
    }


def fixed_signature(manifest: dict) -> dict:
    return {field: manifest.get(field) for field in FIXED_FIELDS}


def cohort_label(rows: list[dict], label: str) -> list[dict]:
    return [{"cohort": label, **row} for row in rows]


def main() -> None:
    args = parse_args()
    base_bundle = resolve_bundle(args.base_source)
    extension_bundle = resolve_bundle(args.extension_source)

    expected_base = parse_seed_spec(args.expected_base_seeds)
    expected_extension = parse_seed_spec(args.expected_extension_seeds)
    if set(expected_base) & set(expected_extension):
        raise ValueError("Expected base and extension seeds overlap.")

    base_manifest = read_json(base_bundle, "exp3b_manifest.json")
    extension_manifest = read_json(extension_bundle, "exp3b_manifest.json")
    base_gate = read_json(base_bundle, "exp3b_gate.json")
    extension_gate = read_json(extension_bundle, "exp3b_gate.json")

    if base_gate.get("pass") is not True:
        raise RuntimeError("Frozen base bundle did not pass its Exp IIIb gate.")
    if extension_gate.get("pass") is not True:
        raise RuntimeError("Extension bundle did not pass its Exp IIIb gate.")

    base_seeds = [int(x) for x in base_manifest["seeds"]]
    extension_seeds = [int(x) for x in extension_manifest["seeds"]]
    if base_seeds != expected_base:
        raise RuntimeError(
            f"Base seeds mismatch: expected {expected_base}, got {base_seeds}."
        )
    if extension_seeds != expected_extension:
        raise RuntimeError(
            "Extension seeds mismatch: "
            f"expected {expected_extension}, got {extension_seeds}."
        )
    if set(base_seeds) & set(extension_seeds):
        raise RuntimeError("Observed base and extension seed sets overlap.")

    base_sig = fixed_signature(base_manifest)
    extension_sig = fixed_signature(extension_manifest)
    if base_sig != extension_sig:
        differences = {
            key: {"base": base_sig[key], "extension": extension_sig[key]}
            for key in FIXED_FIELDS
            if base_sig[key] != extension_sig[key]
        }
        raise RuntimeError(
            "Base and extension scientific specifications differ: "
            + json.dumps(differences, sort_keys=True)
        )

    base_paths = read_csv(base_bundle, "exp3b_path_contrasts.csv")
    extension_paths = read_csv(extension_bundle, "exp3b_path_contrasts.csv")
    base_activation = read_csv(base_bundle, "exp3b_activation_interaction.csv")
    extension_activation = read_csv(
        extension_bundle, "exp3b_activation_interaction.csv"
    )
    base_audit = read_csv(base_bundle, "exp3b_design_audit.csv")
    extension_audit = read_csv(extension_bundle, "exp3b_design_audit.csv")

    combined_paths = cohort_label(base_paths, "original50") + cohort_label(
        extension_paths, "extension50"
    )
    combined_activation = cohort_label(
        base_activation, "original50"
    ) + cohort_label(extension_activation, "extension50")
    combined_audit = cohort_label(base_audit, "original50") + cohort_label(
        extension_audit, "extension50"
    )

    base_summary = cohort_summary(base_paths, base_activation)
    extension_summary = cohort_summary(extension_paths, extension_activation)
    cumulative_summary = cohort_summary(
        base_paths + extension_paths,
        base_activation + extension_activation,
    )

    plan_path = Path(__file__).resolve().parents[1] / "EXP3B_EXTENSION100_PLAN.md"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} exists; pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    summary = {
        "status": "post_caseB_one_time_precision_extension",
        "original50": base_summary,
        "extension50": extension_summary,
        "cumulative100": cumulative_summary,
        "readout": {
            "original_caseB_is_not_reclassified": True,
            "extension_direction_consistent": extension_summary[
                "direction_negative"
            ],
            "cumulative_direction_negative": cumulative_summary[
                "direction_negative"
            ],
            "cumulative_mc95_entirely_below_zero": cumulative_summary[
                "mc95_entirely_below_zero"
            ],
            "cumulative_j1_absolute_loss_lower": cumulative_summary[
                "j1_absolute_loss_lower"
            ],
            "larger_confirmation_may_be_considered": bool(
                extension_summary["direction_negative"]
                and cumulative_summary["mc95_entirely_below_zero"]
                and cumulative_summary["j1_absolute_loss_lower"]
            ),
        },
    }

    manifest = {
        "experiment": "IIIb_cumulative100_precision_extension",
        "extension_plan": "paper_b/EXP3B_EXTENSION100_PLAN.md",
        "extension_plan_sha256": plan_sha,
        "base_bundle": str(base_bundle),
        "base_bundle_sha256": sha256_file(base_bundle),
        "extension_bundle": str(extension_bundle),
        "extension_bundle_sha256": sha256_file(extension_bundle),
        "base_design_id": base_manifest["design_id"],
        "extension_design_id": extension_manifest["design_id"],
        "base_seeds": base_seeds,
        "extension_seeds": extension_seeds,
        "cumulative_seeds": base_seeds + extension_seeds,
        "fixed_scientific_signature": base_sig,
        "original50_gate_pass": True,
        "extension50_gate_pass": True,
        "expected_cumulative_seeds": len(base_seeds) + len(extension_seeds),
    }

    gate = {
        "pass": True,
        "base_gate_pass": True,
        "extension_gate_pass": True,
        "seed_sets_disjoint": True,
        "fixed_scientific_signature_match": True,
        "observed_base_seeds": len(base_seeds),
        "observed_extension_seeds": len(extension_seeds),
        "observed_cumulative_seeds": len(base_seeds) + len(extension_seeds),
    }

    (output_dir / "exp3b_cumulative100_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "exp3b_cumulative100_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "exp3b_cumulative100_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "exp3b_cumulative100_path_contrasts.csv",
        combined_paths,
    )
    write_csv(
        output_dir / "exp3b_cumulative100_activation_interaction.csv",
        combined_activation,
    )
    write_csv(
        output_dir / "exp3b_cumulative100_design_audit.csv",
        combined_audit,
    )

    bundle_path = output_dir / "paper_b_exp3b_cumulative100_shareable.zip"
    names = (
        "exp3b_cumulative100_manifest.json",
        "exp3b_cumulative100_summary.json",
        "exp3b_cumulative100_gate.json",
        "exp3b_cumulative100_path_contrasts.csv",
        "exp3b_cumulative100_activation_interaction.csv",
        "exp3b_cumulative100_design_audit.csv",
    )
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in names:
            archive.write(output_dir / name, arcname=name)
        archive.write(plan_path, arcname="EXP3B_EXTENSION100_PLAN.md")

    print("Experiment IIIb cumulative N=100 precision check")
    print(f"  original bundle : {base_bundle}")
    print(f"  extension bundle: {extension_bundle}")
    print(f"  output bundle   : {bundle_path}")
    print(
        "  cumulative P^A  : "
        f"{cumulative_summary['adaptive_population_path_effect_mean']:.6f}"
    )
    print(
        "  cumulative MC95 : "
        f"[{cumulative_summary['adaptive_population_path_effect_mc95_low']:.6f}, "
        f"{cumulative_summary['adaptive_population_path_effect_mc95_high']:.6f}]"
    )


if __name__ == "__main__":
    main()
