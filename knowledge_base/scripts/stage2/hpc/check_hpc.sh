#!/bin/bash
# Run on a login node before submitting: bash check_hpc.sh   (reads env.sh; nothing is computed)
cd "$(dirname "$0")"; source env.sh
# activate the conda env (non-interactive shells skip the conda block of ~/.bashrc, so source conda.sh directly)
if [ -n "${CONDA_EXE:-}" ]; then source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"; fi
conda activate "${CONDA_ENV:-}" 2>/dev/null || echo "WARNING: could not activate conda env '${CONDA_ENV:-}'; using $(which python)"
echo "python: $(which python)"
ok=1
for v in PANCIA_PANCIA PANCIA_CLINICAL PANCIA_R2SEG PANCIA_R2SEG_SEG; do
  [ -d "${!v}" ] && echo "ok   $v=${!v}" || { echo "MISSING $v=${!v}"; ok=0; }
done
O=$PANCIA_CLINICAL/PanCIA_outputs
for f in ledger/v8/testbed/fit_host6.json ledger/v85/extent/fit_psi.json; do
  [ -f "$O/$f" ] && echo "ok   $f" || { echo "MISSING $O/$f"; ok=0; }
done
n=$(ls $O/stage2/expert/index/r2seg_index*.csv 2>/dev/null | wc -l); [ "$n" -gt 0 ] && echo "ok   usability index ($n files)" || { echo "MISSING usability index $O/stage2/expert/index/"; ok=0; }
n=$(ls $O/stage2/validation_agents/folds_agent_beyond/fit_*.json 2>/dev/null | wc -l); [ "$n" -eq 9 ] && echo "ok   9 fold fits (folds_agent_beyond)" || { echo "MISSING fold fits: $n of 9 in $O/stage2/validation_agents/folds_agent_beyond"; ok=0; }
for a in BiomedParse VoxTell TotalSegmentator; do [ -d "$PANCIA_R2SEG_SEG/$a" ] && echo "ok   R2Seg_Seg/$a" || { echo "MISSING $PANCIA_R2SEG_SEG/$a"; ok=0; }; done
python -c "import numpy, scipy, pandas, nibabel, yaml, skimage; print('ok   python packages: numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)" || ok=0
cd ..
python -c "import sys; sys.path.insert(0, '../..'); from pancia_kb.stage2 import agents, prior; assert prior.P_TPO == 33; print('ok   code (agents.py, prior with outline_beyond_xi)')" || { echo "code check failed (see the error above: a missing package, or code out of date -> git pull)"; ok=0; }
echo "scans with both agent masks per site (validate uses the first 40 tumour and 20 tumour-free in a fixed order):"
python s2_validate_agents.py --check || ok=0
[ $ok -eq 1 ] && echo "ALL OK: NSHARD=<n> nohup bash run_server.sh > logs/run_server.out 2>&1 &" || echo "fix the MISSING items above"
