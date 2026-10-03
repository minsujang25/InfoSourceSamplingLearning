#!/usr/bin/env bash
set -euo pipefail

MEASUREMENT_ROOT="${PAPER_B_MEASUREMENT_ROOT:-production_results/paper_b_measurement_audit/measurement_087afed31ccf}"
PHASE_ROOT="${PAPER_B_PHASE_ROOT:-production_results/paper_b_epsilon_degree_phase/phase_3f9c496208d5}"
RANK_ROOT="${PAPER_B_RANK_ROOT:-production_results/paper_b_rank_unification/rank_da7a0299e0f9}"
CLOSURE_ROOT="${PAPER_B_CLOSURE_ROOT:-production_results/paper_b_analysis_closure}"
OUTPUT_DIR="${PAPER_B_V5_FIGURE_DATA_OUTPUT:-production_results/paper_b_v5_figure_data}"

for path in   "${MEASUREMENT_ROOT}/runs.csv"   "${MEASUREMENT_ROOT}/belief_checkpoints.csv.gz"   "${PHASE_ROOT}/contrast_summary.csv"   "${PHASE_ROOT}/cell_summary.csv"   "${PHASE_ROOT}/initial_expert_rank_surface.csv"   "${RANK_ROOT}/initial_rank_summary.csv"   "${CLOSURE_ROOT}/group_gap_cell_summary.csv"   "${CLOSURE_ROOT}/A_peer_sna_summary.csv"
do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing required frozen result: ${path}" >&2
    exit 2
  fi
done

if [[ "${PAPER_B_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Frozen v5 figure-data inputs: PASS"
  exit 0
fi

python -m paper_b.experiments.prepare_v5_figure_data   --measurement-root "${MEASUREMENT_ROOT}"   --phase-root "${PHASE_ROOT}"   --rank-root "${RANK_ROOT}"   --closure-root "${CLOSURE_ROOT}"   --output-dir "${OUTPUT_DIR}"
