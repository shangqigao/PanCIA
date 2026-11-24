#!/bin/bash
# TCGA stage 2 on the HPC (24 Nov 2026, step H): accepted lesions only, no Bayes factor, VB on the GPU.
# One array task = one A100 running K shard processes; global shards = (#tasks x K), shard id = task * K + j.
# Resumable: finished series are listed in done_s<i>.txt and skipped, so resubmit the same array after a time-out.
# Before: edit env_tcga.sh (paths, conda env), then  bash check_tcga_s2.sh  on a login node.
# Submit:  sbatch --array=0-7 run_tcga_s2.slurm        (8 concurrent jobs; NTASK below must equal the array size)
#SBATCH --job-name=s2_tcga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=logs/s2_tcga_%A_%a.out
# set to your HPC's GPU partition / account, e.g.:
##SBATCH --partition=ampere
##SBATCH --account=ECI-SL2-GPU
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source env_tcga.sh
if [ -n "${CONDA_EXE:-}" ]; then source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"; fi
conda activate "$CONDA_ENV"
NTASK=${NTASK:-8}; K=${K:-8}; N=$((NTASK * K))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PANCIA_GPU=1 PANCIA_FAST_EDT=${PANCIA_FAST_EDT:-1}
cd "$PANCIA_PANCIA/knowledge_base/scripts/stage2"
for j in $(seq 0 $((K - 1))); do
  i=$((SLURM_ARRAY_TASK_ID * K + j))
  SHARD=$i/$N python s2_run_tcga.py --stage1 v86_tcga_all_bp2 --accepted-only --no-bf --tag "${S2_TAG:-tcga_bp2}" \
    > "$SLURM_SUBMIT_DIR/logs/s2_tcga_s$i.txt" 2>&1 &
done
wait
