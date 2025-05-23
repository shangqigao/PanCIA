#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=8
##SBATCH --time=0-12:00:00
#SBATCH --time=0-00:08:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --qos=intr

## activate environment
source ~/.bashrc
conda activate pancia

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# fit Beta distributions on training data
# img_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast_Tumor/train"
# save_dir="/home/sg2162/rds/hpc-work/PanCIA/analysis/tumor_segmentation"
# srun --mpi=pmi2 python analysis/tumor_segmentation/m_fit_beta_distribution.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir

# # tumor segmentation
img_dir="/home/sg2162/rds/hpc-work/sanity-check/images"
save_dir="/home/sg2162/rds/hpc-work/sanity-check/predictions"
beta_params="/home/sg2162/rds/hpc-work/PanCIA/analysis/tumor_segmentation/Beta_params.json"
srun --mpi=pmi2 python analysis/tumor_segmentation/m_tumor_segmentation.py \
            --img_dir $img_dir \
            --save_dir $save_dir \
            --beta_params $beta_params


            