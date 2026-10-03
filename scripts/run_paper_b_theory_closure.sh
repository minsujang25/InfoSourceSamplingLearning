#!/usr/bin/env bash
set -euo pipefail

WORKERS="${PAPER_B_WORKERS:-6}"

W_PARENT="${PAPER_B_THEORY_W_PARENT:-production_results/paper_b_theory_closure_wchannel}"
PHASE_ROOT="${PAPER_B_PHASE_ROOT:-production_results/paper_b_epsilon_degree_phase/phase_3f9c496208d5}"
CLOSURE_OUT="${PAPER_B_THEORY_CLOSURE_OUT:-production_results/paper_b_theory_closure}"

CANONICAL_ROOT="${PAPER_B_CANONICAL_ROOT:-production_results/paper_b_canonical/production_c9d09daad143}"
MECHANISM_ROOT="${PAPER_B_MECHANISM_ROOT:-production_results/paper_b_mechanism_robustness/mechanism_9b5c35506618}"

for path in   "${CANONICAL_ROOT}/runs.csv"   "${MECHANISM_ROOT}/runs.csv"   "${PHASE_ROOT}/contrast_summary.csv"
do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing required frozen result: ${path}" >&2
    exit 2
  fi
done

if [[ "${PAPER_B_VALIDATE_ONLY:-0}" == "1" ]]; then
  python - <<'PY'
from paper_b.experiments.prepare_theory_closure import q_of_m, Q_ANCHORS
for m, target in Q_ANCHORS.items():
    value = q_of_m(m)
    if abs(value - target) > 5e-4:
        raise SystemExit(f"q(m) anchor failed at m={m}: {value} vs {target}")
print("Theory-closure static inputs and analytical q(m) anchors: PASS")
PY
  exit 0
fi

echo "[1/3] Passive early W-channel measurement rerun"
python -m paper_b.experiments.run_w_channel_decomposition   --workers "${WORKERS}"   --canonical-root "${CANONICAL_ROOT}"   --mechanism-root "${MECHANISM_ROOT}"   --output-dir "${W_PARENT}"   --resume

W_ROOT="$(ls -dt "${W_PARENT}"/channel_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${W_ROOT}" || ! -f "${W_ROOT}/channel_gate.json" ]]; then
  echo "ERROR: could not resolve completed W-channel result under ${W_PARENT}" >&2
  exit 3
fi

echo "[2/3] Adaptive attenuation surface"
echo "[3/3] Analytical q(m) curve and final closure gate"
rm -rf "${CLOSURE_OUT}"
python -m paper_b.experiments.prepare_theory_closure   --w-channel-root "${W_ROOT}"   --phase-root "${PHASE_ROOT}"   --output-dir "${CLOSURE_OUT}"

echo "Theory closure complete."
echo "W-channel root: ${W_ROOT}"
echo "Review bundle: ${CLOSURE_OUT}/paper_b_theory_closure_review.zip"
