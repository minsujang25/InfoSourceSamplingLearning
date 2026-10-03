#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${PAPER_B_V5_FIGURE_DATA:-production_results/paper_b_v5_figure_data}"
OUTPUT_DIR="${PAPER_B_V5_FIGURE_OUTPUT:-production_results/paper_b_v5_figures}"

if [[ ! -f "${DATA_DIR}/v5_figure_data_gate.json" ]]; then
  echo "ERROR: missing frozen figure-data gate in ${DATA_DIR}" >&2
  exit 2
fi

python -m paper_b.figures.make_v5_figures   --data-dir "${DATA_DIR}"   --output-dir "${OUTPUT_DIR}"
