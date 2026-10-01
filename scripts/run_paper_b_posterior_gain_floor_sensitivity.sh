#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
ROOT="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_posterior_gain_floor_sensitivity}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

mkdir -p "${ROOT}"
python -m paper_b.experiments.check_reconstruction

for MIN_SD in 1e-6 1e-8 1e-10; do
  TAG="${MIN_SD//-/m}"
  TAG="${TAG//+/p}"

  echo
  echo "== Exp III floor sensitivity: MIN_SD=${MIN_SD} =="
  python -m paper_b.experiments.run_exp3_redundancy \
    --seeds 10 \
    --seed-start 2001 \
    --initial-regimes flat \
    --n-citizens 100 \
    --horizon 200 \
    --workers "${WORKERS}" \
    --k 1 \
    --epsilon 0.05 \
    --credit 20 \
    --expert-access-share 0.10 \
    --peer-degree 2 \
    --surveillance-interval 5 \
    --numerical-min-sd "${MIN_SD}" \
    --output-dir "${ROOT}/minsd_${TAG}/exp3" \
    --resume

  echo
  echo "== Exp IV floor sensitivity: MIN_SD=${MIN_SD} =="
  python -m paper_b.experiments.run_exp4_homophily_segregation \
    --seeds 10 \
    --seed-start 3001 \
    --n-citizens 100 \
    --horizon 200 \
    --workers "${WORKERS}" \
    --k 1 \
    --epsilon 0.05 \
    --credit 20 \
    --peer-degree 2 \
    --low-homophily 0.50 \
    --high-homophily 0.90 \
    --high-group-shift 3.0 \
    --prior-residual-sd 1.0 \
    --surveillance-interval 5 \
    --numerical-min-sd "${MIN_SD}" \
    --output-dir "${ROOT}/minsd_${TAG}/exp4" \
    --resume
done

python - "${ROOT}" <<'PY'
import csv
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])

def one(pattern):
    hits = list(root.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"Expected one match for {pattern}, got {hits}")
    return hits[0]

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def values(rows, key):
    return [float(r[key]) for r in rows]

def mean(rows, key):
    xs = values(rows, key)
    return statistics.fmean(xs)

def median(rows, key):
    xs = values(rows, key)
    return statistics.median(xs)

def mcse(rows, key):
    xs = values(rows, key)
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")

rows = []
for floor_dir in sorted(root.glob("minsd_*")):
    exp3_gate_path = one(f"{floor_dir.name}/exp3/exp3_*/exp3_gate.json")
    exp4_gate_path = one(f"{floor_dir.name}/exp4/exp4_*/exp4_gate.json")
    exp3_root = exp3_gate_path.parent
    exp4_root = exp4_gate_path.parent

    exp3_gate = json.loads(exp3_gate_path.read_text(encoding="utf-8"))
    exp4_gate = json.loads(exp4_gate_path.read_text(encoding="utf-8"))
    exp3_manifest = json.loads(
        (exp3_root / "exp3_manifest.json").read_text(encoding="utf-8")
    )
    exp4_manifest = json.loads(
        (exp4_root / "exp4_manifest.json").read_text(encoding="utf-8")
    )
    if exp3_manifest["numerical_min_sd"] != exp4_manifest["numerical_min_sd"]:
        raise SystemExit("Exp III/IV floor mismatch")

    exp3 = read_csv(exp3_root / "exp3_redundancy_contrasts.csv")
    activation = read_csv(exp3_root / "exp3_activation_interaction.csv")
    exp4 = read_csv(exp4_root / "exp4_homophily_segregation_interaction.csv")
    adaptive = [r for r in exp3 if r["reliance_mode"] == "adaptive"]
    frozen = [r for r in exp3 if r["reliance_mode"] == "frozen"]

    rows.append({
        "numerical_min_sd": exp3_manifest["numerical_min_sd"],
        "exp3_gate_pass": exp3_gate["pass"],
        "exp4_gate_pass": exp4_gate["pass"],
        "exp3_terminal_floor_share": exp3_gate["terminal_sd_floor_share"],
        "exp4_terminal_floor_share": exp4_gate["terminal_sd_floor_share"],
        "exp3_max_terminal_mse": exp3_gate["max_terminal_mse"],
        "exp4_max_terminal_mse": exp4_gate["max_terminal_mse"],
        "exp3_max_abs_jammer_message_mean": exp3_gate[
            "max_abs_jammer_message_mean"
        ],
        "exp4_max_abs_jammer_message_mean": exp4_gate[
            "max_abs_jammer_message_mean"
        ],
        "exp3_max_individual_response_gain": exp3_gate[
            "max_individual_response_gain"
        ],
        "exp4_max_individual_response_gain": exp4_gate[
            "max_individual_response_gain"
        ],
        "exp3_min_objective_denominator": exp3_gate[
            "min_jammer_objective_denominator"
        ],
        "exp4_min_objective_denominator": exp4_gate[
            "min_jammer_objective_denominator"
        ],
        "exp3_adaptive_redundancy_mean": mean(
            adaptive, "high_minus_low_delta_mse"
        ),
        "exp3_adaptive_redundancy_median": median(
            adaptive, "high_minus_low_delta_mse"
        ),
        "exp3_adaptive_redundancy_mcse": mcse(
            adaptive, "high_minus_low_delta_mse"
        ),
        "exp3_frozen_redundancy_mean": mean(
            frozen, "high_minus_low_delta_mse"
        ),
        "exp3_activation_interaction_mean": mean(
            activation, "adaptive_minus_frozen_redundancy_effect"
        ),
        "exp3_adaptive_high_minus_low_mse_J1_mean": mean(
            adaptive, "high_minus_low_mse_J1"
        ),
        "exp3_adaptive_high_minus_low_mse_J0_mean": mean(
            adaptive, "high_minus_low_mse_J0"
        ),
        "exp4_interaction_mean": mean(
            exp4, "homophily_x_segregation_interaction"
        ),
        "exp4_interaction_median": median(
            exp4, "homophily_x_segregation_interaction"
        ),
        "exp4_interaction_mcse": mcse(
            exp4, "homophily_x_segregation_interaction"
        ),
    })

max_initial_gain = 25.0 / 26.0
min_initial_denom = 1.0 - max_initial_gain**2
for row in rows:
    if not row["exp3_gate_pass"] or not row["exp4_gate_pass"]:
        raise SystemExit("At least one floor-sensitivity gate failed")
    if row["exp3_max_individual_response_gain"] > max_initial_gain + 1e-12:
        raise SystemExit("Exp III response-gain bound failed")
    if row["exp4_max_individual_response_gain"] > max_initial_gain + 1e-12:
        raise SystemExit("Exp IV response-gain bound failed")
    if row["exp3_min_objective_denominator"] < min_initial_denom - 1e-12:
        raise SystemExit("Exp III curvature bound failed")
    if row["exp4_min_objective_denominator"] < min_initial_denom - 1e-12:
        raise SystemExit("Exp IV curvature bound failed")

columns = list(rows[0])
summary_csv = root / "posterior_gain_floor_sensitivity_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

summary_json = root / "posterior_gain_floor_sensitivity_summary.json"
summary_json.write_text(
    json.dumps(rows, indent=2, sort_keys=True),
    encoding="utf-8",
)

print("\nPosterior-gain floor sensitivity")
for row in rows:
    print(json.dumps(row, sort_keys=True))

bundle = root / "paper_b_posterior_gain_floor_sensitivity_shareable.zip"
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(summary_csv, arcname=summary_csv.name)
    archive.write(summary_json, arcname=summary_json.name)
    for nested in sorted(root.glob("**/paper_b_exp*_shareable.zip")):
        archive.write(nested, arcname=str(nested.relative_to(root)))

print(f"\nShareable bundle: {bundle}")
PY
