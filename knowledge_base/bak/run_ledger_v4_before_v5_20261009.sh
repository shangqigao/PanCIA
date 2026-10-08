#!/bin/bash
# Ledger v4 (stage 3 + R2 calibrated EM) on the baseline-study series, CSD3 ampere (CPU-only code; GPU account), as a SLURM array.
#
#   sbatch scripts/run_ledger_v4.sh                      # 1 task, 1 GPU allocation, all 3,291 series in scripts/rel_list_baseline.txt
#
# Resumable: a shard skips series whose <out>/<rel>_lesions.json already exists (rerun the same command after a timeout).
# Outputs: $SEG_ROOT/Ledger_v4/Radiology/<rel>_lesions.{json,npz} and ledger_summary_<i>of<n>.csv per shard.
# The v3 ledger ($SEG_ROOT/Ledger) is not touched.
#
# Set before submitting (or edit the defaults below): SEG_ROOT, IMG_ROOT, CONDA_ENV, and the account in the #SBATCH -A line.
# CPU only. Peak memory per worker ~ 20 bytes x working-box voxels (a 512 x 512 x 600 CT box ~ 3 GB), hence himem nodes
# and 2 CPUs' worth of memory per worker.

#SBATCH -A ECI-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.%A_%a
#SBATCH --array=0                      # one task = one GPU allocation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32            # the CPU share of one A100 on CSD3; the code itself is CPU-only
#SBATCH --time=0-36:00:00
##SBATCH --time=0-00:10:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
##SBATCH --qos=intr

echo "[$(date)] start on $(hostname), task ${SLURM_ARRAY_TASK_ID:-none}, SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}, usable cores=$(nproc --all 2>/dev/null; true) / affinity $(taskset -cp $$ 2>/dev/null | cut -d: -f2)"
# Environment setup runs WITHOUT strict mode: ~/.bashrc and conda's activate scripts often return non-zero
# or reference unset variables, which under set -e/-u would kill the job silently before anything is logged.
source ~/.bashrc || echo "warning: ~/.bashrc returned $?"
conda activate PanCIA || { echo "ERROR: conda activate PanCIA failed"; exit 1; }
echo "python: $(which python)"
python -c "import numpy, scipy, nibabel, yaml, pandas" || { echo "ERROR: missing python packages"; exit 1; }
set -euo pipefail

KB_DIR="${KB_DIR:-/home/sg2162/rds/hpc-work/PanCIA/knowledge_base}"
SEG_ROOT="${SEG_ROOT:-/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/TCGA_Seg}"
IMG_ROOT="${IMG_ROOT:-/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/TCGA_NIFTI}"
REL_LIST="${REL_LIST:-$KB_DIR/scripts/rel_list_baseline.txt}"
NSHARD="${NSHARD:-1}"                    # must equal the number of array tasks
WORKERS="${WORKERS:-32}"
OUT_NAME="${OUT_NAME:-Ledger_v4}"          # 7 Oct rerun: move the first run aside first (mv Ledger_v4 Ledger_v4_run1); finished series are skipped                 # one worker per core; peak ~5 GB each on the largest boxes
# large volumes are processed in full on HPC (the local pilot skipped boxes > 45 M voxels)
DEFAULT_PARAMS='{"em_max_box_vox": 2.5e8}'
PARAMS="${PARAMS:-$DEFAULT_PARAMS}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker process

for f in "$KB_DIR/scripts/run_lesion_ledger.py" "$KB_DIR/pancia_kb/r2_em.py" "$REL_LIST"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
[ -d "$SEG_ROOT" ] && [ -d "$IMG_ROOT" ] || { echo "ERROR: SEG_ROOT or IMG_ROOT not found"; exit 1; }
[ -e "$SEG_ROOT/$OUT_NAME/Radiology/ledger_summary.csv" ] && { echo "ERROR: $SEG_ROOT/$OUT_NAME already holds a finished run; move it aside or set OUT_NAME"; exit 1; }
echo "shard ${SLURM_ARRAY_TASK_ID}/${NSHARD}  KB=$KB_DIR  SEG=$SEG_ROOT  IMG=$IMG_ROOT  list=$REL_LIST"
python "$KB_DIR/scripts/run_lesion_ledger.py" \
  --seg_root "$SEG_ROOT" --img_root "$IMG_ROOT" \
  --rel_list "$REL_LIST" --out_name "$OUT_NAME" \
  --workers "$WORKERS" --params "$PARAMS" \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshard "$NSHARD"
echo "[$(date)] finished"

# After all shards finish, merge the summaries:
#   python -c "import glob,pandas as pd;fs=sorted(glob.glob('$SEG_ROOT/Ledger_v4/Radiology/ledger_summary_*of*.csv'));\
#   pd.concat(map(pd.read_csv,fs)).to_csv('$SEG_ROOT/Ledger_v4/Radiology/ledger_summary.csv',index=False)"
