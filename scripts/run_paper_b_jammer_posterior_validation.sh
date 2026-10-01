#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
ROOT="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_jammer_posterior_validation}"

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

echo
echo "== Targeted regression: former Exp III runaway seed 2173 =="
python -m paper_b.experiments.run_exp3_redundancy \
  --seeds 1 \
  --seed-start 2173 \
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
  --numerical-min-sd 1e-8 \
  --output-dir "${ROOT}/exp3_seed2173" \
  --resume

echo
echo "== Small matched Exp III validation: 20 flat-prior seeds =="
python -m paper_b.experiments.run_exp3_redundancy \
  --seeds 20 \
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
  --numerical-min-sd 1e-8 \
  --output-dir "${ROOT}/exp3_pilot" \
  --resume

echo
echo "== Small matched Exp IV validation: 20 seeds =="
python -m paper_b.experiments.run_exp4_homophily_segregation \
  --seeds 20 \
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
  --numerical-min-sd 1e-8 \
  --output-dir "${ROOT}/exp4_pilot" \
  --resume

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

def mean(xs):
    xs = [float(x) for x in xs]
    return statistics.fmean(xs) if xs else float("nan")

def median(xs):
    xs = [float(x) for x in xs]
    return statistics.median(xs) if xs else float("nan")

def mcse(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs) / math.sqrt(len(xs))

seed2173_gate_path = one("exp3_seed2173/exp3_*/exp3_gate.json")
exp3_gate_path = one("exp3_pilot/exp3_*/exp3_gate.json")
exp4_gate_path = one("exp4_pilot/exp4_*/exp4_gate.json")

seed2173_root = seed2173_gate_path.parent
exp3_root = exp3_gate_path.parent
exp4_root = exp4_gate_path.parent

seed2173_gate = json.loads(seed2173_gate_path.read_text(encoding="utf-8"))
exp3_gate = json.loads(exp3_gate_path.read_text(encoding="utf-8"))
exp4_gate = json.loads(exp4_gate_path.read_text(encoding="utf-8"))

exp3_contrasts = read_csv(exp3_root / "exp3_redundancy_contrasts.csv")
exp3_activation = read_csv(exp3_root / "exp3_activation_interaction.csv")
exp4_interactions = read_csv(exp4_root / "exp4_homophily_segregation_interaction.csv")

adaptive = [r for r in exp3_contrasts if r["reliance_mode"] == "adaptive"]
frozen = [r for r in exp3_contrasts if r["reliance_mode"] == "frozen"]

summary = {
    "seed2173_gate": seed2173_gate,
    "exp3_pilot_gate": exp3_gate,
    "exp4_pilot_gate": exp4_gate,
    "exp3_20seed": {
        "adaptive_redundancy_effect_mean": mean(
            r["high_minus_low_delta_mse"] for r in adaptive
        ),
        "adaptive_redundancy_effect_median": median(
            r["high_minus_low_delta_mse"] for r in adaptive
        ),
        "adaptive_redundancy_effect_mcse": mcse(
            r["high_minus_low_delta_mse"] for r in adaptive
        ),
        "frozen_redundancy_effect_mean": mean(
            r["high_minus_low_delta_mse"] for r in frozen
        ),
        "frozen_redundancy_effect_median": median(
            r["high_minus_low_delta_mse"] for r in frozen
        ),
        "frozen_redundancy_effect_mcse": mcse(
            r["high_minus_low_delta_mse"] for r in frozen
        ),
        "activation_interaction_mean": mean(
            r["adaptive_minus_frozen_redundancy_effect"] for r in exp3_activation
        ),
        "activation_interaction_median": median(
            r["adaptive_minus_frozen_redundancy_effect"] for r in exp3_activation
        ),
        "activation_interaction_mcse": mcse(
            r["adaptive_minus_frozen_redundancy_effect"] for r in exp3_activation
        ),
        "adaptive_high_minus_low_mse_J1_mean": mean(
            r["high_minus_low_mse_J1"] for r in adaptive
        ),
        "adaptive_high_minus_low_mse_J0_mean": mean(
            r["high_minus_low_mse_J0"] for r in adaptive
        ),
    },
    "exp4_20seed": {
        "interaction_mean": mean(
            r["homophily_x_segregation_interaction"] for r in exp4_interactions
        ),
        "interaction_median": median(
            r["homophily_x_segregation_interaction"] for r in exp4_interactions
        ),
        "interaction_mcse": mcse(
            r["homophily_x_segregation_interaction"] for r in exp4_interactions
        ),
    },
}

max_initial_gain = 25.0 / 26.0
min_initial_denom = 1.0 - max_initial_gain**2
for name, gate in (
    ("seed2173", seed2173_gate),
    ("exp3_pilot", exp3_gate),
    ("exp4_pilot", exp4_gate),
):
    if gate.get("pass") is not True:
        raise SystemExit(f"{name} gate failed")
    if gate.get("max_individual_response_gain", 1.0) > max_initial_gain + 1e-12:
        raise SystemExit(f"{name} exceeded the model-implied gain bound")
    if gate.get("min_jammer_objective_denominator", 0.0) < min_initial_denom - 1e-12:
        raise SystemExit(f"{name} violated the model-implied curvature bound")

summary_path = root / "validation_summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)

print("\nPosterior-gain validation summary")
print(json.dumps(summary, indent=2, sort_keys=True))

bundle = root / "paper_b_jammer_posterior_validation_shareable.zip"
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(summary_path, arcname=summary_path.name)
    for nested in sorted(root.glob("**/paper_b_exp*_shareable.zip")):
        archive.write(nested, arcname=str(nested.relative_to(root)))

print(f"\nShareable bundle: {bundle}")
PY
