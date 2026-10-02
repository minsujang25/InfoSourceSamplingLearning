#!/usr/bin/env bash
set -euo pipefail

CANONICAL_ROOT="${PAPER_B_CANONICAL_ROOT:-production_results/paper_b_canonical/production_c9d09daad143}"
OUTPUT_DIR="${PAPER_B_PHASE_A_OUTPUT:-local_results/paper_b_measurement_audit_phase_a}"

python -m paper_b.experiments.check_measurement_audit
python -m paper_b.experiments.analyze_measurement_phase_a \
  --production-root "${CANONICAL_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
