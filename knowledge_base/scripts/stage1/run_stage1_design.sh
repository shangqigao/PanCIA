#!/usr/bin/env bash
# Stage-1 overnight run on the DESIGN SET only (28 Nov 2026; plan claude/stage1_decision_plan_2026-11-28.md; workflow
# claude/stage1_workflow.md). Never touches jobs_v3.csv: the v3 evaluation waits for the decision gate (written and approved first).
# Steps (each resumable; re-running the script skips finished scans):
#   1 real candidates (tumour scans)        s1_augment_candidates.py --stage real     -> augment/real.csv
#   2 real candidates (tumour-free scans)   s1_augment_candidates.py --stage real_tf  -> augment/real_tf.csv   (evaluation only)
#   3 augmented candidates                  s1_augment_candidates.py --stage aug      -> augment/aug.csv
#   4 realism check                         s1_augment_realism.py                     -> augment/realism.{txt,csv}
#   5 elongation logLR (LOSO)               s1_fit_elong_lr.py                        -> elong_lr/
#   6 decision exploration                  s1_decision_design.py                     -> design/
# Log: PanCIA_outputs/ledger/decision_s1/run_design.log ; summary: decision_s1/overnight_summary.txt
# Usage (from this folder; caffeinate keeps the Mac awake): caffeinate -i bash run_stage1_design.sh
set -o pipefail
cd "$(dirname "$0")"
L=$(python -c "import sys; sys.path.insert(0,'../..'); from pancia_kb.stage2 import paths; print(paths.ledger_dir())") || exit 1
O="$L/decision_s1"; mkdir -p "$O"; LOG="$O/run_design.log"
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
         if [ $rc -ne 0 ]; then echo "=== FAILED ($rc): $*" | tee -a "$LOG"; exit $rc; fi; }
echo "=== start $(date '+%F %T') md5:" | tee -a "$LOG"
md5 -r s1_augment_candidates.py s1_augment_realism.py s1_fit_elong_lr.py s1_decision_design.py 2>/dev/null | tee -a "$LOG" \
  || md5sum s1_augment_candidates.py s1_augment_realism.py s1_fit_elong_lr.py s1_decision_design.py | tee -a "$LOG"
step python s1_augment_candidates.py --stage real
step python s1_augment_candidates.py --stage real_tf
step python s1_augment_candidates.py --stage aug
step python s1_augment_realism.py
step python s1_fit_elong_lr.py
step python s1_decision_design.py
{ echo "Stage-1 design-set overnight run, finished $(date '+%F %T')"
  for f in augment/realism.txt elong_lr/eval.txt design/summary.txt; do echo; echo "########## $f"; cat "$O/$f"; done
  echo; echo "########## notes / errors in augment files"
  python - "$O" <<'PY'
import sys, pandas as pd
for f in ('real', 'real_tf', 'aug'):
    d = pd.read_csv(f'{sys.argv[1]}/augment/{f}.csv', low_memory=False)
    n = d[d.note.notna() & (d.note != '')]
    print(f, 'scans', d[['ds', 'case']].drop_duplicates().shape[0], 'rows', len(d), 'notes', n.note.astype(str).str[:40].value_counts().head(5).to_dict())
PY
} > "$O/overnight_summary.txt" 2>&1
echo "=== done $(date '+%F %T'); summary: $O/overnight_summary.txt" | tee -a "$LOG"
