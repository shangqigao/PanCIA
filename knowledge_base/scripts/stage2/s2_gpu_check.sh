#!/bin/bash
# Step S (23 Nov 2026): check the GPU path of stage 2 (pancia_kb/stage2/vb.py, PANCIA_GPU=1) against the CPU path on this machine.
# Needs: cupy for the machine's CUDA (e.g. pip install cupy-cuda12x), the PanCIA data roots (R2Seg, R2Seg_Seg, outputs with
# ledger/v86_r2seg_bp2 incl. extents/, ledger/v85/extent/fit_psi.json, stage2/validation_agents/folds_agent_beyond, the bp2 fold fits).
# 1) selftest on CPU and GPU (printed values must match); 2) the same 2 kidney + 2 liver R2Seg scans (v4 settings) on CPU and GPU:
# per-lesion Dice / precision / recall of every method, log BF, iterations and seconds compared.
# Usage: ./s2_gpu_check.sh            outputs: stage2/_gpucheck_cpu, stage2/_gpucheck_gpu (move them to _to_delete afterwards)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import cupy; p = cupy.cuda.runtime.getDeviceProperties(0); print('GPU:', p['name'].decode() if isinstance(p['name'], bytes) else p['name'])"
PSI=$(python -c "import sys, os; sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths; print(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json'))")
echo "== selftest"; python s2_selftest.py > /tmp/s2_st_cpu.txt 2>&1; PANCIA_GPU=1 PANCIA_GPU_MIN_N=0 python s2_selftest.py > /tmp/s2_st_gpu.txt 2>&1
if diff <(sed 's/([0-9]* s)//' /tmp/s2_st_cpu.txt) <(sed 's/([0-9]* s)//' /tmp/s2_st_gpu.txt); then echo "selftest: identical printed values"; else echo "selftest: differences above"; fi
tail -1 /tmp/s2_st_cpu.txt; tail -1 /tmp/s2_st_gpu.txt
for MODE in cpu gpu; do
  echo "== validate ($MODE)"
  G=0; [ "$MODE" = gpu ] && G=1
  PANCIA_GPU=$G python s2_validate_agents.py --folds kidney,liver --n-val 2 --n-neg 0 --max-false 2 \
    --fit-dir validation_agents/folds_agent_beyond --fit-samples agent_true --fit-beyond --tumour-from anchor \
    --stage1-ledger v86_r2seg_bp2 --psi "$PSI" --tag _gpucheck_$MODE > /tmp/s2_gpucheck_$MODE.log 2>&1 || { tail -20 /tmp/s2_gpucheck_$MODE.log; exit 1; }
done
python - <<'PY'
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, '../..'); from pancia_kb.stage2 import paths
d = {m: pd.read_csv(os.path.join(paths.out_dir('_gpucheck_' + m), 'rows.csv'), low_memory=False) for m in ('cpu', 'gpu')}
k = ['ds', 'case', 'kind', 'lesion', 'method', 'where', 'anchor_ml']
for m in d:
    d[m] = d[m][d[m].kind.isin(['lesion', 'false_anchor'])]
j = d['cpu'].merge(d['gpu'], on=[c for c in k if c in d['cpu']], suffixes=('_cpu', '_gpu'))
print(f"{len(j)} matched rows (cpu {len(d['cpu'])}, gpu {len(d['gpu'])})")
for c in ['dice', 'precision', 'recall', 'log_bf', 'iters', 'vol_Z']:
    if c + '_cpu' in j:
        x = (j[c + '_cpu'] - j[c + '_gpu']).abs()
        print(f"{c:10s} max |cpu - gpu| {x.max():.3g}   mean {x.mean():.3g}")
s = {m: d[m].drop_duplicates(['ds', 'case', 'lesion', 'where'])['sec'].sum() for m in d}
print(f"solve seconds: cpu {s['cpu']:.0f}, gpu {s['gpu']:.0f} (x{s['cpu'] / max(s['gpu'], 1e-9):.1f})")
PY
