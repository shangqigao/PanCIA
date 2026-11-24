#!/bin/bash
# Login-node check before sbatch run_tcga_s2.slurm (nothing heavy is computed)
cd "$(dirname "$0")"; source env_tcga.sh; mkdir -p logs
if [ -n "${CONDA_EXE:-}" ]; then source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"; fi
conda activate "$CONDA_ENV" || echo "WARNING: conda env $CONDA_ENV not activated"
ok=1; O=$PANCIA_OUTPUTS
for v in PANCIA_PANCIA PANCIA_OUTPUTS PANCIA_TCGA_SEG PANCIA_TCGA_NIFTI; do [ -d "${!v}" ] && echo "ok   $v" || { echo "MISSING $v=${!v}"; ok=0; }; done
for f in ledger/v86_tcga_all_bp2/anchors_s0.csv ledger/v86_tcga_all_bp2/series_s0.csv ledger/v85/extent/fit_psi.json ledger/v8/testbed/fit_host6.json ledger/v8/testbed/relational_fit.json \
         ledger/v6/grounding/g0/h5/tables_g1.json ledger/v8/grounding/test_sets.json; do
  [ -f "$O/$f" ] && echo "ok   $f" || { echo "MISSING $O/$f"; ok=0; }; done
grep -q '"xi_rule": *"anchor"' "$O/ledger/v85/extent/fit_psi.json" && echo "ok   fit_psi.json has xi_rule anchor" || { echo "OLD fit_psi.json (no xi_rule anchor)"; ok=0; }
[ -f "$PANCIA_PANCIA/knowledge_base/knowledge_base/stage2/stage2_fit_agent_beyond.json" ] && grep -q '"2026-11-23' "$PANCIA_PANCIA/knowledge_base/knowledge_base/stage2/stage2_fit_agent_beyond.json" \
  && echo "ok   stage2_fit_agent_beyond.json (23 Nov bp2 fit)" || { echo "stage2_fit_agent_beyond.json missing or not the 23 Nov fit (git pull)"; ok=0; }
for a in BiomedParse VoxTell TotalSegmentator Planner; do [ -d "$PANCIA_TCGA_SEG/$a/Radiology" ] && echo "ok   TCGA_Seg/$a" || { echo "MISSING TCGA_Seg/$a/Radiology"; ok=0; }; done
python -c "import numpy, scipy, pandas, nibabel, yaml; print('ok   packages numpy', numpy.__version__)" || ok=0
python -c "import cupy; print('ok   cupy', cupy.__version__)" || echo "note: cupy not importable on the login node (fine if it imports on GPU nodes)"
python -c "
import os, glob, pandas as pd
L=os.path.join('$O','ledger','v86_tcga_all_bp2'); s=pd.concat(pd.read_csv(f) for f in glob.glob(L+'/series_s*.csv'))
print('ok   series with an accepted lesion:', int((s.n_acc > 0).sum()), 'of', len(s))" || ok=0
[ $ok -eq 1 ] && echo "ALL OK. Smoke: see README_tcga.txt step 3; then: sbatch --array=0-7 run_tcga_s2.slurm" || echo "fix the items above"
