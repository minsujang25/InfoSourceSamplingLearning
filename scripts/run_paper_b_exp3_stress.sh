#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT_DIR="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_exp3_stress}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

python -m paper_b.experiments.check_reconstruction

# Pre-production stress audit for the intentionally asymmetric Exp III
# opportunity structure: universal Jammer access, localized Expert gateways.
#
# 50 matched seeds x 2 prior regimes x 8 conditions = 800 simulations.
# The polarized regime is diagnostic-only and is included to expose runaway
# feedback that may not appear under the flat primary production regime.
python -m paper_b.experiments.run_exp3_redundancy \
  --seeds 50 \
  --seed-start 2001 \
  --initial-regimes flat,polarized \
  --n-citizens 100 \
  --horizon 200 \
  --workers "${WORKERS}" \
  --k 1 \
  --epsilon 0.05 \
  --credit 20 \
  --expert-access-share 0.10 \
  --peer-degree 2 \
  --surveillance-interval 5 \
  --output-dir "${OUTPUT_DIR}" \
  --resume \
  "$@"

python - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
gates = sorted(root.glob("exp3_*/exp3_gate.json"))
if not gates:
    raise SystemExit("No Exp III stress gate found.")

gate = json.loads(gates[-1].read_text(encoding="utf-8"))
print("\nExperiment III stress diagnostics")
for key in (
    "pass",
    "observed_blocks",
    "observed_runs",
    "max_terminal_mse",
    "max_abs_terminal_belief",
    "max_abs_jammer_message_mean",
    "max_jammer_response_gain",
    "max_individual_response_gain",
    "min_jammer_objective_denominator",
    "min_terminal_sd_theta",
    "terminal_sd_floor_count",
    "terminal_sd_floor_share",
):
    print(f"  {key:32s}: {gate.get(key)}")

if gate.get("pass") is not True:
    raise SystemExit("Experiment III stress gate failed.")
PY
