#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT_DIR="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_exp3b_path_independence}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

python -m paper_b.experiments.check_reconstruction

python -m paper_b.experiments.run_exp3b_path_independence \
  --seeds 50 \
  --seed-start 4001 \
  --n-citizens 100 \
  --n-gateways 10 \
  --n-relays 40 \
  --horizon 200 \
  --workers "${WORKERS}" \
  --k 1 \
  --epsilon 0.05 \
  --credit 20 \
  --surveillance-interval 5 \
  --numerical-min-sd 1e-8 \
  --output-dir "${OUTPUT_DIR}" \
  --resume

# This wrapper intentionally freezes the first-stage IIIb design at 50 matched
# seeds. Use the Python module directly only for CI/smoke tests; do not escalate
# to a 500-seed production until the 50-seed bundle has been classified under
# the precommitted rule in paper_b/EXP3B_DIAGNOSTIC_PLAN.md.
