#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
CANONICAL_ROOT="${PAPER_B_CANONICAL_ROOT:-production_results/paper_b_canonical/production_c9d09daad143}"
MECHANISM_ROOT="${PAPER_B_MECHANISM_ROOT:-production_results/paper_b_mechanism_robustness/mechanism_9b5c35506618}"
OUTPUT_DIR="${PAPER_B_PHASE_OUTPUT:-production_results/paper_b_epsilon_degree_phase}"

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
  echo "Canonical reference: ${CANONICAL_ROOT}"
  echo "Mechanism reference: ${MECHANISM_ROOT}"
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
python -m paper_b.experiments.check_epsilon_degree_phase

python -m paper_b.experiments.run_epsilon_degree_phase \
  --seeds 500 \
  --seed-start 6001 \
  --n-citizens 100 \
  --horizon 400 \
  --workers "${WORKERS}" \
  --credit 20 \
  --k 1 \
  --surveillance-interval 5 \
  --low-homophily 0.50 \
  --high-homophily 0.90 \
  --high-group-shift 3.0 \
  --prior-residual-sd 1.0 \
  --numerical-min-sd 1e-8 \
  --canonical-root "${CANONICAL_ROOT}" \
  --mechanism-root "${MECHANISM_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume
