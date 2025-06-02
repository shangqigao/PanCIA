#!/bin/bash

#SBATCH -A PION-P3-DAWN-GPU
##SBATCH -A PION-P3-CPU
#SBATCH -J pancia
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=8
##SBATCH --time=0-12:00:00
#SBATCH --time=0-00:08:00
#SBATCH --partition=pvc9
#SBATCH --gres=gpu:1
#SBATCH --qos=intr

## load dawn
module purge
module load default-dawn

## load MPI
module av intel-oneapi-mpi

## activate environment
source ~/.bashrc
conda activate pancia

export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_SYMMETRIC=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Force output flushing
export PYTHONUNBUFFERED=1   # if running Python
export SLURM_EXPORT_ENV=ALL
stdbuf -oL -eL echo "Starting job at $(date)"

# fit Beta distributions on training data
# img_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast_Tumor/train"
# save_dir="/home/sg2162/rds/hpc-work/PanCIA/analysis/tumor_segmentation"
# srun --mpi=pmi2 python analysis/tumor_segmentation/m_fit_beta_distribution.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir

# # tumor segmentation
# img_dir="/home/sg2162/rds/hpc-work/sanity-check/images"
# save_dir="/home/sg2162/rds/hpc-work/sanity-check/predictions"
# beta_params="/home/sg2162/rds/hpc-work/PanCIA/analysis/tumor_segmentation/Beta_params.json"
# srun --mpi=pmi2 python analysis/tumor_segmentation/m_tumor_segmentation.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir \
#             --beta_params $beta_params

# extract radiomic features
img_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast_Tumor/train"
save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
meta_info="/home/sg2162/rds/hpc-work/Experiments/clinical/MAMA-MIA_clinical_and_imaging_info.xlsx"
srun --mpi=pmi2 python analysis/feature_extraction/m_radiomics_extraction.py \
            --img_dir $img_dir \
            --save_dir $save_dir \
            --meta_info $meta_info           