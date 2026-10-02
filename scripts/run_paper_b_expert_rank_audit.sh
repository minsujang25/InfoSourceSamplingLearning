#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="\${PAPER_B_EXPERT_RANK_OUTPUT:-local_results/paper_b_expert_rank_audit}"
MEASUREMENT_REVIEW="\${PAPER_B_MEASUREMENT_REVIEW:-}"

python -m paper_b.experiments.check_mechanism_robustness

ARGS=(
  --seeds 500
  --seed-start 6001
  --n-citizens 100
  --epsilon 0.05
  --credit 20
  --peer-degree 2
  --low-homophily 0.50
  --high-homophily 0.90
  --high-group-shift 3.0
  --prior-residual-sd 1.0
  --output-dir "\${OUTPUT_DIR}"
)

if [[ -n "\${MEASUREMENT_REVIEW}" ]]; then
  ARGS+=(--measurement-review "\${MEASUREMENT_REVIEW}")
fi

python -m paper_b.experiments.analyze_expert_rank_mechanism "\${ARGS[@]}"
