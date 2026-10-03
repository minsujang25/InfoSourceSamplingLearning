#!/usr/bin/env bash
set -euo pipefail

MEASUREMENT_ROOT="${PAPER_B_MEASUREMENT_ROOT:-production_results/paper_b_measurement_audit/measurement_087afed31ccf}"
OUTPUT_DIR="${PAPER_B_CLOSURE_OUTPUT:-production_results/paper_b_analysis_closure}"

python -m paper_b.experiments.check_analysis_closure

if [[ "${PAPER_B_VALIDATE_ONLY:-0}" == "1" ]]; then
  test -f "${MEASUREMENT_ROOT}/runs.csv"
  test -f "${MEASUREMENT_ROOT}/terminal_beliefs.csv.gz"
  echo "Measurement reference: ${MEASUREMENT_ROOT}"
  echo "Wrapper validation: PASS"
  exit 0
fi

python -m paper_b.experiments.run_analysis_closure \
  --seeds 500 \
  --seed-start 6001 \
  --n-citizens 100 \
  --peer-degree 2 \
  --low-homophily 0.50 \
  --high-homophily 0.90 \
  --high-group-shift 3.0 \
  --prior-residual-sd 1.0 \
  --measurement-root "${MEASUREMENT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
