#!/bin/bash
# Ledger v4 (stage 3 + R2 calibrated EM) on the baseline-study series, CSD3 CPU, as a SLURM array.
#
#   sbatch scripts/run_ledger_v4.sh                      # 16 shards over scripts/rel_list_baseline.txt (3,291 series)
#   NSHARD=32 sbatch --array=0-31 scripts/run_ledger_v4_csd3.sh
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
#SBATCH --array=0-15
#SBATCH --nodes=1
##SBATCH --cpus-per-task=32
#SBATCH --time=0-36:00:00
##SBATCH --time=0-00:10:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
##SBATCH --qos=intr

set -eo pipefail
source ~/.bashrc
conda activate PanCIA
set -u                                  # after conda: its activate scripts reference unset variables

KB_DIR="${KB_DIR:-/home/sg2162/rds/hpc-work/PanCIA/knowledge_base}"
SEG_ROOT="${SEG_ROOT:-/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/TCGA_Seg}"
IMG_ROOT="${IMG_ROOT:-/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/TCGA_NIFTI}"
REL_LIST="${REL_LIST:-$KB_DIR/scripts/rel_list_baseline.txt}"
NSHARD="${NSHARD:-16}"
WORKERS="${WORKERS:-8}"
# large volumes are processed in full on HPC (the local pilot skipped boxes > 45 M voxels)
DEFAULT_PARAMS='{"em_max_box_vox": 2.5e8}'
PARAMS="${PARAMS:-$DEFAULT_PARAMS}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker process

echo "shard ${SLURM_ARRAY_TASK_ID}/${NSHARD}  KB=$KB_DIR  SEG=$SEG_ROOT  IMG=$IMG_ROOT  list=$REL_LIST"
python "$KB_DIR/scripts/run_lesion_ledger.py" \
  --seg_root "$SEG_ROOT" --img_root "$IMG_ROOT" \
  --rel_list "$REL_LIST" --out_name Ledger_v4 \
  --workers "$WORKERS" --params "$PARAMS" \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshard "$NSHARD"

# After all shards finish, merge the summaries:
#   python -c "import glob,pandas as pd;fs=sorted(glob.glob('$SEG_ROOT/Ledger_v4/Radiology/ledger_summary_*of*.csv'));\
#   pd.concat(map(pd.read_csv,fs)).to_csv('$SEG_ROOT/Ledger_v4/Radiology/ledger_summary.csv',index=False)"
