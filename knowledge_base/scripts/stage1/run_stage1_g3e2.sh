#!/usr/bin/env bash
# Stage-1 design-set run for the elongation (e2) and pelvic (g3) fixes (28 Nov 2026; user-approved plan
# claude/elongation_and_pelvic_fix_plan_2026-11-28.md; workflow claude/stage1_workflow.md). DESIGN SET ONLY (jobs.csv + jobs_v2.csv);
# jobs_v3.csv is never read. No decision rule is frozen here: the full decision is designed after these results are reviewed.
# Steps (each resumable; the run stops at the first failing step):
#   1 step 2g g3 on scans with an unresolved sex-class dispute   s1_anatomy_g.py --g3_contradiction   (g2 copies -> anatomy/r2seg_g2/)
#   2 check: g3 identical to g2 apart from the node             s1_g3_check.py --stage anatomy
#   3 step 4g on the design set                                  s1_graph_v3.py --tag design_g3
#   4 check: undisputed scans identical; before / after per site s1_g3_check.py --stage graph
#   5-7 candidates with e2 elongation (real, real_tf, aug)       s1_augment_candidates.py --outdir augment_e2 --design design_g3
#   8 realism                                                    s1_augment_realism.py --augdir augment_e2
#   9 elongation logLR (logistic, shape only, LOSO)              s1_fit_elong_lr.py  -> elong_lr_e2/
#  10 decision exploration                                       s1_decision_design.py -> design_e2/
# Log: PanCIA_outputs/ledger/decision_s1/run_g3e2.log ; summary: decision_s1/g3e2_summary.txt
# Usage (from this folder): caffeinate -i bash run_stage1_g3e2.sh
set -o pipefail
cd "$(dirname "$0")"
L=$(python -c "import sys; sys.path.insert(0,'../..'); from pancia_kb.stage2 import paths; print(paths.ledger_dir())") || exit 1
O="$L/decision_s1"; mkdir -p "$O"; LOG="$O/run_g3e2.log"
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
         if [ $rc -ne 0 ]; then echo "=== FAILED ($rc): $*" | tee -a "$LOG"; exit $rc; fi; }
F="../../pancia_kb/anatomy_check_g.py s1_anatomy_g.py s1_g3_check.py s1_graph_v3.py s1_graph_v3_eval.py s1_augment_candidates.py s1_augment_realism.py s1_fit_elong_lr.py s1_decision_design.py"
echo "=== start $(date '+%F %T') md5:" | tee -a "$LOG"
md5 -r $F 2>/dev/null | tee -a "$LOG" || md5sum $F | tee -a "$LOG"
J=jobs.csv,jobs_v2.csv
step python s1_anatomy_g.py --jobs $J --g3_contradiction --retry_heavy --budget 1e9 --guard 0
step python s1_g3_check.py --stage anatomy
step python s1_graph_v3.py --jobs $J --tag design_g3
step python s1_g3_check.py --stage graph --old design --new design_g3
A="--outdir augment_e2 --design design_g3"
step python s1_augment_candidates.py --stage real $A
step python s1_augment_candidates.py --stage real_tf $A
step python s1_augment_candidates.py --stage aug $A
step python s1_augment_realism.py --augdir augment_e2
step python s1_fit_elong_lr.py --augdir augment_e2 --outdir elong_lr_e2 --design design_g3
step python s1_decision_design.py --lrdir elong_lr_e2 --design design_g3 --augdir augment_e2 --outdir design_e2
{ echo "Stage-1 g3 + e2 design-set run, finished $(date '+%F %T')"
  for f in "$L/graph_v3/g3_check_anatomy.txt" "$L/graph_v3/g3_check_graph.txt" "$O/augment_e2/realism.txt" "$O/elong_lr_e2/eval.txt" "$O/design_e2/summary.txt"; do
    echo; echo "########## $f"; cat "$f"; done
} > "$O/g3e2_summary.txt" 2>&1
echo "=== done $(date '+%F %T'); summary: $O/g3e2_summary.txt" | tee -a "$LOG"
