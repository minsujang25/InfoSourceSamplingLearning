#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
ROOT="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_exp3_floor_sensitivity}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

python -m paper_b.experiments.check_reconstruction

# Same 20 matched flat-prior seeds at three numerical SD floors.
# This is a numerical-sensitivity audit only; it does not redefine the
# substantive model or the primary production estimand.
for MIN_SD in 1e-6 1e-8 1e-10; do
  TAG="${MIN_SD//-/m}"
  TAG="${TAG//+/p}"
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
    --numerical-min-sd "${MIN_SD}" \
    --output-dir "${ROOT}/minsd_${TAG}" \
    --resume
done

python - "${ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
rows = []
for gate_path in sorted(root.glob("minsd_*/exp3_*/exp3_gate.json")):
    run_root = gate_path.parent
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "exp3_manifest.json").read_text(encoding="utf-8"))
    contrasts = pd.read_csv(run_root / "exp3_redundancy_contrasts.csv")
    activation = pd.read_csv(run_root / "exp3_activation_interaction.csv")

    adaptive = contrasts[contrasts["reliance_mode"] == "adaptive"]
    frozen = contrasts[contrasts["reliance_mode"] == "frozen"]
    rows.append(
        {
            "numerical_min_sd": manifest["numerical_min_sd"],
            "design_id": manifest["design_id"],
            "gate_pass": gate["pass"],
            "terminal_sd_floor_share": gate["terminal_sd_floor_share"],
            "max_terminal_mse": gate["max_terminal_mse"],
            "max_abs_jammer_message_mean": gate["max_abs_jammer_message_mean"],
            "max_jammer_response_gain": gate["max_jammer_response_gain"],
            "adaptive_redundancy_effect_mean": adaptive[
                "high_minus_low_delta_mse"
            ].mean(),
            "frozen_redundancy_effect_mean": frozen[
                "high_minus_low_delta_mse"
            ].mean(),
            "activation_interaction_mean": activation[
                "adaptive_minus_frozen_redundancy_effect"
            ].mean(),
            "high_minus_low_mse_J1_adaptive_mean": adaptive[
                "high_minus_low_mse_J1"
            ].mean(),
            "high_minus_low_mse_J0_adaptive_mean": adaptive[
                "high_minus_low_mse_J0"
            ].mean(),
        }
    )

summary = pd.DataFrame(rows).sort_values("numerical_min_sd", ascending=False)
out = root / "floor_sensitivity_summary.csv"
summary.to_csv(out, index=False)
print("\nExperiment III numerical-floor sensitivity")
print(summary.to_string(index=False))
print(f"\nSummary: {out}")

if not summary["gate_pass"].all():
    raise SystemExit("At least one numerical-floor sensitivity run failed.")
PY
