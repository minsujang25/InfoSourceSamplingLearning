#!/usr/bin/env bash
set -euo pipefail

BASE_SOURCE="${PAPER_B_EXP3B_BASE_SOURCE:-local_results/paper_b_exp3b_path_independence}"
EXTENSION_SOURCE="${PAPER_B_EXP3B_EXTENSION_SOURCE:-local_results/paper_b_exp3b_extension50}"
OUTPUT_DIR="${PAPER_B_EXP3B_CUMULATIVE_OUTPUT_DIR:-local_results/paper_b_exp3b_cumulative100}"

python -m paper_b.experiments.combine_exp3b_cumulative100 \
  --base-source "${BASE_SOURCE}" \
  --extension-source "${EXTENSION_SOURCE}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-base-seeds 4001-4050 \
  --expected-extension-seeds 4051-4100 \
  --overwrite
