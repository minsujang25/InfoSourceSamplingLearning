#!/usr/bin/env bash
set -euo pipefail

# Paper B UCloud calibration pilot.
#
# IMPORTANT: set PAPER_B_WORKERS to the number of CPU cores actually allocated
# to this UCloud job.  The runner parallelizes across matched blocks and does
# not start nested worker pools.
#
# Example:
#   PAPER_B_WORKERS=16 bash scripts/run_paper_b_pilot_ucloud.sh
#
# Optional:
#   PAPER_B_OUTPUT_DIR=/work/paper_b_pilot \
#   PAPER_B_WORKERS=32 \
#   bash scripts/run_paper_b_pilot_ucloud.sh

WORKERS="${PAPER_B_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
OUTPUT_DIR="${PAPER_B_OUTPUT_DIR:-ucloud_results/paper_b_pilot}"

# One matched-block process per allocated CPU. Keep numerical libraries
# single-threaded inside each process to avoid nested CPU oversubscription.
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
echo "== Paper B matched calibration pilot =="
echo "workers: ${WORKERS}"
echo "output : ${OUTPUT_DIR}"

python -m paper_b.experiments.run_matched_pilot \
  --seeds 20 \
  --seed-start 1001 \
  --initial-regimes flat,consensus,polarized \
  --n-citizens 100 \
  --horizon 200 \
  --k 1 \
  --epsilon 0.05 \
  --credit 20 \
  --comparison-rule delta_comparison \
  --surveillance-interval 5 \
  --workers "${WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume \
  "$@"
