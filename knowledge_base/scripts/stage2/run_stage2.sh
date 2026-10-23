#!/usr/bin/env bash
# Stage 2 runner. Usage:
#   ./run_stage2.sh            self-test, expert samples, fit, input check (stops at review point 1)
#   ./run_stage2.sh validate   leave-one-collection-out validation (review point 2)
#   ./run_stage2.sh tcga       TCGA pilot run and panels
# NSHARD (default 4) parallel shards for the long steps; N_PER (default 100) R2Seg cases per collection; N_VAL (default 20).
set -euo pipefail
cd "$(dirname "$0")"
NSHARD=${NSHARD:-4}; N_PER=${N_PER:-100}; N_VAL=${N_VAL:-20}
shards() {   # run "$@" in NSHARD parallel shards and wait
  for i in $(seq 0 $((NSHARD - 1))); do SHARD=$i/$NSHARD python "$@" > "log_$(basename "$1" .py)_s$i.txt" 2>&1 & done; wait
}
case "${1:-fit}" in
  fit)
    python s2_selftest.py
    shards s2_expert_build.py --n-per "$N_PER"
    python s2_fit.py
    shards s2_check_inputs.py
    python s2_check_inputs.py   # summary over all shards (everything already done)
    echo "Review point 1: knowledge_base/knowledge_base/stage2/stage2_fit_report.txt and stage2/check_inputs/summary.txt"
    ;;
  validate)
    python s2_validate.py --n-val "$N_VAL"
    echo "Review point 2: clinical/PanCIA_outputs/stage2/validation/report.txt"
    ;;
  tcga)
    shards s2_run_tcga.py
    python s2_png.py --n 60
    ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
