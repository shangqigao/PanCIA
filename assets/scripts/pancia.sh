#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=32
##SBATCH --time=0-36:00:00
#SBATCH --time=0-00:08:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
##SBATCH --qos=intr

## activate environment
source ~/.bashrc
conda activate biomedparse

img_dir="/home/sg2162/rds/hpc-work/sanity-check/images"
save_dir="/home/sg2162/rds/hpc-work/sanity-check/predictions"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
srun --mpi=pmi2 python analysis/tumor_segmentation/m_tumor_segmentation.py \
            --img_dir $img_dir \
            --save_dir $save_dir


            