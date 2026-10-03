#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
MEASUREMENT_ROOT="${PAPER_B_MEASUREMENT_ROOT:-production_results/paper_b_measurement_audit/measurement_087afed31ccf}"
OUTPUT_DIR="${PAPER_B_RANK_OUTPUT:-production_results/paper_b_rank_unification}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

if [[ "${PAPER_B_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "PAPER_B_WORKERS=${WORKERS}"
  echo "Measurement reference: ${MEASUREMENT_ROOT}"
  echo "Wrapper validation: PASS"
  exit 0
fi

python -m paper_b.experiments.check_reconstruction
python -m paper_b.experiments.check_validity_redesign
python -m paper_b.experiments.check_final_calibration
python -m paper_b.experiments.check_canonical_production
python -m paper_b.experiments.check_measurement_audit
python -m paper_b.experiments.check_mechanism_robustness
python -m paper_b.experiments.check_w_channel_decomposition
python -m paper_b.experiments.check_rank_unification

python -m paper_b.experiments.run_rank_unification_audit \
  --seeds 500 \
  --seed-start 6001 \
  --n-citizens 100 \
  --horizon 400 \
  --workers "${WORKERS}" \
  --credit 20 \
  --epsilon 0.05 \
  --k 1 \
  --surveillance-interval 5 \
  --peer-degree 2 \
  --expert-access-share 0.10 \
  --low-homophily 0.50 \
  --high-homophily 0.90 \
  --high-group-shift 3.0 \
  --prior-residual-sd 1.0 \
  --numerical-min-sd 1e-8 \
  --measurement-root "${MEASUREMENT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume
