#!/usr/bin/env bash
# Tail-weight prior (modality x tissue class), from the tail study. Steps:
#   1. breast tumour interiors (no segmented host: tumour only), N_PER random usable scans (default 200; about half of breast tumours are too small for a fit)
#   2. tail-study summary (refreshes components.csv with the breast components)
#   3. split-half check and the prior fit (writes knowledge_base/knowledge_base/stage2/nu_prior.json)
# Load (laptop): NSHARD (default 2) single-threaded, low-priority processes; PAUSE (default 0.5) rests that fraction of each
# scan's compute time. Resumable: rerun after an interruption. Needs knowledge base v0.5.9 (tissue_class).
# Outputs: clinical/PanCIA_outputs/stage2/tail_study/nu_prior/report.txt; logs: clinical/PanCIA_outputs/stage2/logs/.
set -euo pipefail
cd "$(dirname "$0")"
NSHARD=${NSHARD:-2}; PAUSE=${PAUSE:-0.5}; N_PER=${N_PER:-200}; MIN_N=${MIN_N:-100}
PY=python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=$("$PY" -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('logs'))")
echo "$(date '+%F %T') 1/3 breast tumour interiors: $NSHARD shards, $N_PER scans"
pids=()
for i in $(seq 0 $((NSHARD - 1))); do
  SHARD=$i/$NSHARD nice -n 15 "$PY" s2_tail_study.py --tumour-only --n-per "$N_PER" --pause "$PAUSE" > "$LOGD/log_nu_prior_breast_s$i.txt" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "WARNING: a shard exited with an error; see $LOGD/log_nu_prior_breast_s*.txt (rerun to resume)"; done
echo "$(date '+%F %T') 2/3 tail-study summary"
nice -n 15 "$PY" s2_tail_study.py --summary > "$LOGD/log_tail_study_summary.txt" 2>&1
echo "$(date '+%F %T') 3/3 split-half check and prior fit"
nice -n 15 "$PY" s2_nu_prior.py --split-half --min-n "$MIN_N" > "$LOGD/log_nu_prior.txt" 2>&1
echo "$(date '+%F %T') done: clinical/PanCIA_outputs/stage2/tail_study/nu_prior/report.txt and knowledge_base/knowledge_base/stage2/nu_prior.json"
