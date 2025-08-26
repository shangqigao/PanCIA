#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL3-CPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
##SBATCH --time=0-36:00:00
#SBATCH --time=0-00:10:00
##SBATCH -p cclake
#SBATCH -p cclake-himem
##SBATCH -p ampere
##SBATCH --gres=gpu:1
#SBATCH --qos=intr

## activate environment
source ~/.bashrc
conda activate /home/sg2162/rds/hpc-work/miniconda3/PanCIA

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

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
img_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/images"
lab_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/segmentations"
save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
meta_info="/home/sg2162/rds/hpc-work/Experiments/clinical/MAMA-MIA_clinical_and_imaging_info.xlsx"

srun --mpi=pmi2 python analysis/feature_extraction/m_radiomics_extraction.py \
            --img_dir $img_dir \
            --lab_dir $lab_dir \
            --save_dir $save_dir \
            --meta_info $meta_info           