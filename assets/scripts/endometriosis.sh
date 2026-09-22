#!/bin/bash

#SBATCH -A ECI-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=32
#SBATCH --time=0-36:00:00
##SBATCH --time=0-00:10:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
##SBATCH --qos=intr

## activate environment
source ~/.bashrc
conda activate PanCIA

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

export TMPDIR="/home/sg2162/rds/hpc-work/tmp"
export TORCHINDUCTOR_CACHE_DIR="/home/sg2162/rds/hpc-work/torch_cache"
export TRITON_CACHE_DIR="/home/sg2162/rds/hpc-work/triton_cache"

# Force output flushing
export PYTHONUNBUFFERED=1   # if running Python
export SLURM_EXPORT_ENV=ALL
stdbuf -oL -eL echo "Starting job at $(date)"

# python analysis/utilities/m_prepare_biomedparse_endometriosis_dataset.py
# python analysis/utilities/m_calculate_endometriosis_segmentation_metrics.py
# python analysis/utilities/m_plot_endometriosis_box.py

#----------------Endometriosis--------------------
# radiology exclusion and inclusion
# data_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer"
# save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"

# python analysis/a01_data_preprocessiong/m_inclusion_exclusion.py \
#             --data_dir $data_dir \
#             --dataset CPTAC \
#             --modality radiology \
#             --save_dir $save_dir

# dicom to nifti (save to /parent/to/dataset/dataset_NIFTI)
# series="/home/sg2162/rds/hpc-work/Experiments/radiomics/CPTAC_included_raw_series.json"

# python analysis/a01_data_preprocessiong/m_dicom2nii.py \
#             --series $series \
#             --dataset CPTAC


# Endometrioma segmentation
# radiology="/home/sg2162/rds/hpc-work/EndoMRI_All"
# save_dir="/home/sg2162/rds/hpc-work/EndoMRI_All/segmentations_r3"
# srun python analysis/a02_tumor_segmentation/m_endometrioma_segmentation.py \
#             --radiology $radiology \
#             --dataset EndoMRI_All \
#             --save_dir $save_dir