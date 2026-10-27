#!/usr/bin/env bash
# Stage 2 runner. Usage:
#   ./run_stage2.sh            self-test, expert samples, fit, input check (stops at review point 1)
#   ./run_stage2.sh validate   leave-one-site-out validation (review point 2)
#   ./run_stage2.sh tcga       TCGA pilot run and panels
# NSHARD (default 4) parallel shards for the long steps (NSHARD_TCGA, default 2, for the TCGA run: memory; NSHARD_VAL, default 3, for the validation folds); N_PER (default 100) R2Seg cases per collection, drawn at random from the usable cases; N_VAL (default 20).
set -euo pipefail
cd "$(dirname "$0")"
NSHARD=${NSHARD:-4}; NSHARD_TCGA=${NSHARD_TCGA:-2}; NSHARD_VAL=${NSHARD_VAL:-3}; N_PER=${N_PER:-100}; N_VAL=${N_VAL:-20}
# logs go to clinical/PanCIA_outputs/stage2/logs (PanCIA keeps code only)
LOGD=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('logs'))")
# one thread per process, low priority (several shards x multi-threaded BLAS overloaded the laptop, 26 Oct)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
shards() {   # run "$@" in NSHARD parallel shards and wait
  for i in $(seq 0 $((NSHARD - 1))); do SHARD=$i/$NSHARD nice -n 15 python "$@" > "$LOGD/log_$(basename "$1" .py)_s$i.txt" 2>&1 & done; wait
  echo "$1 done (logs: $LOGD)"
}
case "${1:-fit}" in
  fit)
    python s2_selftest.py
    shards s2_r2seg_index.py                     # usability of every R2Seg label file (cases are drawn from the usable ones)
    python s2_r2seg_index.py                     # summary table (nothing left to do)
    shards s2_expert_build.py --n-per "$N_PER"
    # shards s2_nu.py --n-per "$N_PER"             # tail shape nu per tumour tissue: on hold (voxel correlation; see README 25 Oct)
    python s2_fit.py
    shards s2_check_inputs.py
    python s2_check_inputs.py   # summary over all shards (everything already done)
    echo "Review point 1: knowledge_base/knowledge_base/stage2/stage2_fit_report.txt and stage2/check_inputs/summary.txt"
    ;;
  validate)
    NSHARD=$NSHARD_VAL shards s2_validate.py --n-val "$N_VAL"   # 9 site folds split over NSHARD_VAL processes
    python s2_validate.py --summary
    echo "Review point 2: clinical/PanCIA_outputs/stage2/validation/report.txt"
    ;;
  tcga)
    NSHARD=$NSHARD_TCGA shards s2_run_tcga.py      # large clusters need several GB each
    python s2_png.py --n 60
    ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
