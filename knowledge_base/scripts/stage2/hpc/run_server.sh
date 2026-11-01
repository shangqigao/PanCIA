#!/bin/bash
# validate_agents_v3 on a plain multi-core server (no SLURM).
#   NSHARD=32 nohup bash run_server.sh > logs/run_server.out 2>&1 &     (survives logout; or run it inside tmux/screen)
# NSHARD parallel single-thread shards (default: min(cores - 2, free RAM GB / 3)); the summary (report.txt) is written at the end.
# Stop everything: bash stop_server.sh. Resumable: run again (any NSHARD); finished scans are skipped.
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p logs; source env.sh
source ~/.bashrc >/dev/null 2>&1; conda activate "${CONDA_ENV:-}" 2>/dev/null || true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CORES=$(nproc); MEMG=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
AUTO=$(( CORES - 2 < MEMG / 3 ? CORES - 2 : MEMG / 3 )); [ "$AUTO" -lt 1 ] && AUTO=1
NSHARD=${NSHARD:-$AUTO}
echo "$(date) cores $CORES, available RAM ${MEMG} GB -> $NSHARD shards (about 2-3 GB each)"
echo $$ > logs/run_server.pid
trap 'trap - INT TERM; pkill -TERM -P $$; exit 143' INT TERM
cd ..
PSI=$PANCIA_CLINICAL/PanCIA_outputs/ledger/v85/extent/fit_psi.json
for i in $(seq 0 $((NSHARD - 1))); do
  SHARD=$i/$NSHARD nice -n 10 python s2_validate_agents.py --n-val 40 --n-neg 20 \
    --fit-dir validation_agents/folds_agent_beyond --fit-samples agent_true --fit-beyond --tumour-from anchor --single \
    --tag validation_agents_v3 --psi "$PSI" > hpc/logs/v3_s$i.txt 2>&1 &
done
wait
echo "$(date) shards finished; summary"
python s2_validate_agents.py --summary --tag validation_agents_v3 > hpc/logs/v3_summary.txt 2>&1
echo "$(date) done. Report: $PANCIA_CLINICAL/PanCIA_outputs/stage2/validation_agents_v3/report.txt"
grep -h "scans failed" hpc/logs/v3_summary.txt || true
