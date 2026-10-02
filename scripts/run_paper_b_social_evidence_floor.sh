#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT_DIR="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_social_evidence_floor}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

python -m paper_b.experiments.check_reconstruction
python -m paper_b.experiments.check_validity_redesign

python -m paper_b.experiments.run_social_evidence_floor \
  --seeds 8 \
  --seed-start 5101 \
  --n-citizens 100 \
  --horizon 200 \
  --workers "${WORKERS}" \
  --k 1 \
  --epsilon 0.05 \
  --credit 20 \
  --surveillance-interval 5 \
  --peer-degree 2 \
  --expert-access-share 0.10 \
  --low-homophily 0.50 \
  --high-homophily 0.90 \
  --high-group-shift 3.0 \
  --prior-residual-sd 1.0 \
  --numerical-min-sd 1e-8 \
  --output-dir "${OUTPUT_DIR}" \
  --resume
