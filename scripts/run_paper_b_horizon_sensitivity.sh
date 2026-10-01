#!/usr/bin/env bash
set -euo pipefail

# Targeted Paper B horizon sensitivity:
#   20 seeds x 3 priors x 2 sparse/local network environments
#   x 4 matched conditions = 480 individual simulations
#   exact T = 400
#
# The run stores checkpoints at T=200 and T=400 so the consolidation step can
# compare the same trajectory at both horizons.

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT_DIR="${PAPER_B_OUTPUT_DIR:-local_results/paper_b_horizon_sensitivity}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PAPER_B_WORKERS must be a positive integer." >&2
  exit 2
fi

echo "== Paper B reconstruction checks =="
python -m paper_b.experiments.check_reconstruction

echo
echo "== Paper B T=400 targeted horizon sensitivity =="
echo "workers : ${WORKERS}"
echo "output  : ${OUTPUT_DIR}"
echo "networks: random_2,group_id"
echo "seeds   : 20 (1001-1020)"
echo "priors  : flat,consensus,polarized"
echo "horizon : 400"

python -m paper_b.experiments.run_matched_pilot \
  --seeds 20 \
  --seed-start 1001 \
  --initial-regimes flat,consensus,polarized \
  --network-environments random_2,group_id \
  --n-citizens 100 \
  --horizon 400 \
  --k 1 \
  --epsilon 0.05 \
  --credit 20 \
  --comparison-rule delta_comparison \
  --surveillance-interval 5 \
  --workers "${WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  --no-edge-log \
  --resume \
  "$@"
