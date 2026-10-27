#!/usr/bin/env bash
# Tail study (steps 1 + 2; review only, nothing enters stage2_fit.json). Usage:
#   ./run_tail_study.sh              all usable tumour scans + up to 50 tumour-free scans per collection, then the summary
#   ./run_tail_study.sh summary      summary only (tables and report from what is done)
# Load (laptop): NSHARD (default 2) parallel processes, each single-threaded and at low priority (nice); PAUSE (default 0.5)
# rests that fraction of each scan's compute time (0 = full speed; 1 = about half the average load). Faster: NSHARD=4 PAUSE=0.
# N_NEG (default 50) tumour-free scans per collection; MIN_N (default 200) smallest fit
# used in the statistics. Resumable: rerun after an interruption and finished scans are skipped.
# Outputs: clinical/PanCIA_outputs/stage2/tail_study/ (report.txt first); logs: clinical/PanCIA_outputs/stage2/logs/.
set -euo pipefail
cd "$(dirname "$0")"
NSHARD=${NSHARD:-2}; PAUSE=${PAUSE:-0.5}; N_NEG=${N_NEG:-50}; MIN_N=${MIN_N:-200}
PY=python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=$("$PY" -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('logs'))")
OUT=$("$PY" -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('tail_study'))")
if [ "${1:-run}" != "summary" ]; then
  echo "$(date '+%F %T') tail study: $NSHARD shards, pause $PAUSE; logs $LOGD/log_tail_study_s<i>.txt"
  pids=()
  for i in $(seq 0 $((NSHARD - 1))); do
    SHARD=$i/$NSHARD nice -n 15 "$PY" s2_tail_study.py --n-neg "$N_NEG" --pause "$PAUSE" > "$LOGD/log_tail_study_s$i.txt" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || echo "WARNING: a shard exited with an error; see $LOGD/log_tail_study_s*.txt (rerun to resume)"
  n_err=$(cat "$OUT"/notes*.txt 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%F %T') scans done: $(cat "$OUT"/done*.txt 2>/dev/null | wc -l | tr -d ' '); with errors: $n_err (see $OUT/notes*.txt)"
fi
"$PY" s2_tail_study.py --summary --min-n "$MIN_N" > "$LOGD/log_tail_study_summary.txt" 2>&1
echo "$(date '+%F %T') summary written: $OUT/report.txt"
