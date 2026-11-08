#!/usr/bin/env bash
# Stage 2 runner. Usage:
#   ./run_stage2.sh            self-test, expert samples, fit, input check (stops at review point 1)
#   ./run_stage2.sh validate   leave-one-site-out validation (review point 2)
#   ./run_stage2.sh validate_agents   the same folds with real BiomedParse + VoxTell outlines on R2Seg (N_VAL_AG, default 40;
#                                     N_NEG, default 20 tumour-free scans per site); python s2_validate_agents.py --check first
#   ./run_stage2.sh psi_fit             stage-1 extent prior psi refitted on real agent outlines (run before agent_fit)
#   ./run_stage2.sh agent_fit           spatial prior refitted on real agent outlines (samples, fold fits, calibration)
#   ./run_stage2.sh validate_agents_v2  real-agent validation with those fits (anchor-based tumour appearance)
#   ./run_stage2.sh agent_true_fit      spatial prior refitted on TRUE lesions only: P(tumour | lesion real) (1 Nov)
#   ./run_stage2.sh agent_beyond_fit    true-lesion prior + 'in an outline beyond xi' (fits only; samples of agent_true_fit) (1 Nov)
#   ./run_stage2.sh validate_agents_v3  real-agent validation with the agent_beyond_fit fold fits
#   ./run_stage2.sh validate_agents_v3q as v3, tumour appearance from q instead of the anchors (2 Nov)
#   ./run_stage2.sh tcga       TCGA pilot run and panels
# NSHARD (default 4) parallel shards for the long steps (NSHARD_TCGA, default 2, for the TCGA run: memory; NSHARD_VAL, default 3, for the validations: s2_validate_agents.py splits by (site, case), so up to ~400 shards are useful; s2_validate.py by site); N_PER (default 100) R2Seg cases per collection, drawn at random from the usable cases; N_VAL (default 20).
set -euo pipefail
cd "$(dirname "$0")"
NSHARD=${NSHARD:-4}; NSHARD_TCGA=${NSHARD_TCGA:-2}; NSHARD_VAL=${NSHARD_VAL:-3}; N_PER=${N_PER:-100}; N_VAL=${N_VAL:-20}
# logs go to clinical/PanCIA_outputs/stage2/logs (PanCIA keeps code only)
LOGD=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('logs'))")
# one thread per process, low priority (several shards x multi-threaded BLAS overloaded the laptop, 26 Oct)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Ctrl-C / kill stops every shard too (background jobs of a script ignore Ctrl-C; a restart while old shards still run would solve
# the same scans twice)
trap 'trap - INT TERM; kill 0' INT TERM
shards() {   # run "$@" in NSHARD parallel shards and wait
  for i in $(seq 0 $((NSHARD - 1))); do SHARD=$i/$NSHARD nice -n 15 python "$@" > "$LOGD/log_$(basename "$1" .py)_s$i.txt" 2>&1 & done; wait
  echo "$1 done (logs: $LOGD)"
}
# stage-1 extent prior for the real-agent steps: ledger/v85/extent/fit_psi.json when it exists (./run_stage2.sh psi_fit), else v8.4
PSI_FILE=$(python -c "import sys, os; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json'))")
PSI_ARG=""; [ -f "$PSI_FILE" ] && PSI_ARG="--psi $PSI_FILE"
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
  validate_agents)
    NSHARD=$NSHARD_VAL shards s2_validate_agents.py --n-val "${N_VAL_AG:-40}" --n-neg "${N_NEG:-20}"
    python s2_validate_agents.py --summary
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents/report.txt"
    ;;
  psi_fit)
    # 29 Oct: stage-1 extent prior psi refitted on real agent outlines (v8.5 item brought forward; review fit_psi_report.txt)
    (cd ../stage1 && shards s1_extent_rows.py --n-per "$N_PER" && python s1_fit_psi.py)
    ;;
  agent_fit)
    # 29 Oct: spatial prior on real agent outlines. Samples (BiomedParse + VoxTell on R2Seg), one fold fit per held-out site and the
    # full fit (stage2_fit_agent.json; stage2_fit.json is not replaced), then the held-out calibration of the prior
    shards s2_agent_build.py --n-per "$N_PER" $PSI_ARG
    for D in kidney liver pancreas colon lung cervix urinary_bladder breast prostate; do
      EXCL=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import expert as EX; print(','.join(k for k, v in EX.SITE_ENTITY.items() if v == '$D'))")
      OUTD=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('validation_agents', 'folds_agent'))")
      [ -f "$OUTD/fit_$D.json" ] || python s2_fit.py --samples agent --tumour-from anchor --exclude "$EXCL" --out "$OUTD/fit_$D.json" --max-rows 1.5e6
    done
    python s2_fit.py --samples agent --tumour-from anchor --max-rows 1.5e6 \
      --out "$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.fit_path('stage2_fit_agent.json'))")"
    python s2_prior_calibration.py
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents/prior_calibration.txt and knowledge_base/knowledge_base/stage2/stage2_fit_agent_report.txt"
    ;;
  agent_true_fit)
    # 1 Nov: the spatial prior describes a real tumour only (existence is the stage-1 decision and the stage-2 log BF). Samples of the
    # true lesions (s2_agent_build.py --true-only -> agent_samples_true), fold fits (validation_agents/folds_agent_true), full fit
    # (stage2_fit_agent_true.json), held-out calibration on the true-lesion samples (prior_calibration_true.txt)
    shards s2_agent_build.py --n-per "$N_PER" --true-only $PSI_ARG
    OUTD=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('validation_agents', 'folds_agent_true'))")
    for D in kidney liver pancreas colon lung cervix urinary_bladder breast prostate; do
      EXCL=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import expert as EX; print(','.join(k for k, v in EX.SITE_ENTITY.items() if v == '$D'))")
      [ -f "$OUTD/fit_$D.json" ] || python s2_fit.py --samples agent_true --tumour-from anchor --exclude "$EXCL" --out "$OUTD/fit_$D.json" --max-rows 1.5e6
    done
    python s2_fit.py --samples agent_true --tumour-from anchor --max-rows 1.5e6 \
      --out "$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.fit_path('stage2_fit_agent_true.json'))")"
    python s2_prior_calibration.py --samples agent_samples_true --out prior_calibration_true \
      --fits sim=validation/folds,agent=validation_agents/folds_agent,agent_true=validation_agents/folds_agent_true
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents/prior_calibration_true.txt and knowledge_base/knowledge_base/stage2/stage2_fit_agent_true_report.txt"
    ;;
  agent_beyond_fit)
    # 1 Nov (user: option B): the true-lesion prior plus one input, 1[in any outline] x 1[beyond xi] (prior.NAMES_TPO). Reuses
    # agent_samples_true (no rebuild). Fold fits -> validation_agents/folds_agent_beyond, full fit stage2_fit_agent_beyond.json,
    # calibration on the true-lesion samples with every fit (prior_calibration_beyond.txt)
    OUTD=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.out_dir('validation_agents', 'folds_agent_beyond'))")
    for D in kidney liver pancreas colon lung cervix urinary_bladder breast prostate; do
      EXCL=$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import expert as EX; print(','.join(k for k, v in EX.SITE_ENTITY.items() if v == '$D'))")
      [ -f "$OUTD/fit_$D.json" ] || python s2_fit.py --samples agent_true --beyond --tumour-from anchor --exclude "$EXCL" --out "$OUTD/fit_$D.json" --max-rows 1.5e6
    done
    python s2_fit.py --samples agent_true --beyond --tumour-from anchor --max-rows 1.5e6 \
      --out "$(python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(paths.fit_path('stage2_fit_agent_beyond.json'))")"
    python s2_prior_calibration.py --samples agent_samples_true --out prior_calibration_beyond \
      --fits sim=validation/folds,agent=validation_agents/folds_agent,agent_true=validation_agents/folds_agent_true,agent_beyond=validation_agents/folds_agent_beyond
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents/prior_calibration_beyond.txt and knowledge_base/knowledge_base/stage2/stage2_fit_agent_beyond_report.txt"
    ;;
  validate_agents_v3q)
    # 2 Nov: v3 on all sites with tumour appearance from q-weighted voxels (tumour_from q) instead of the anchors
    NSHARD=$NSHARD_VAL shards s2_validate_agents.py --n-val "${N_VAL_AG:-40}" --n-neg "${N_NEG:-20}" \
      --fit-dir validation_agents/folds_agent_beyond --fit-samples agent_true --fit-beyond --tumour-from anchor --appearance-from q \
      --single --tag validation_agents_v3q $PSI_ARG
    python s2_validate_agents.py --summary --tag validation_agents_v3q
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents_v3q/report.txt (compare with validation_agents_v3)"
    ;;
  kidney_appearance_q)
    # 2 Nov test: v3 on the kidney site only, tumour appearance from q-weighted voxels (tumour_from q) instead of the anchors;
    # same fits and scans as validate_agents_v3 -> validation_agents_v3q (compare kidney with validation_agents_v3)
    NSHARD=$NSHARD_VAL shards s2_validate_agents.py --folds "${FOLDS:-kidney}" --n-val "${N_VAL_AG:-40}" --n-neg "${N_NEG:-20}" \
      --fit-dir validation_agents/folds_agent_beyond --fit-samples agent_true --fit-beyond --tumour-from anchor --appearance-from q \
      --single --tag validation_agents_v3q $PSI_ARG
    python s2_validate_agents.py --summary --tag validation_agents_v3q
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents_v3q/report.txt (kidney; compare with validation_agents_v3)"
    ;;
  validate_agents_v3)
    NSHARD=$NSHARD_VAL shards s2_validate_agents.py --n-val "${N_VAL_AG:-40}" --n-neg "${N_NEG:-20}" \
      --fit-dir validation_agents/folds_agent_beyond --fit-samples agent_true --fit-beyond --tumour-from anchor --single --tag validation_agents_v3 $PSI_ARG
    python s2_validate_agents.py --summary --tag validation_agents_v3
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents_v3/report.txt (compare with validation_agents_v2 and validation_agents)"
    ;;
  validate_agents_v2)
    NSHARD=$NSHARD_VAL shards s2_validate_agents.py --n-val "${N_VAL_AG:-40}" --n-neg "${N_NEG:-20}" \
      --fit-dir validation_agents/folds_agent --fit-samples agent --tumour-from anchor --single --tag validation_agents_v2 $PSI_ARG
    python s2_validate_agents.py --summary --tag validation_agents_v2
    echo "Review: clinical/PanCIA_outputs/stage2/validation_agents_v2/report.txt (compare with validation_agents/report.txt)"
    ;;
  tcga)
    NSHARD=$NSHARD_TCGA shards s2_run_tcga.py      # large clusters need several GB each
    python s2_png.py --n 60
    ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
