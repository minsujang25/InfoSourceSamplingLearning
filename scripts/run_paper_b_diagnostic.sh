#!/usr/bin/env bash
set -euo pipefail

echo "== Paper B reconstruction checks =="
python -m paper_b.experiments.check_reconstruction

echo
echo "== Paper B matched local diagnostic =="
python -m paper_b.experiments.run_local_diagnostic "$@"
