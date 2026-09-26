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
# data_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik"
# save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
# extract_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/OV04_endometriosis"
# csv_path="/home/sg2162/rds/hpc-work/Experiments/clinical/OV04_endometriosis_has_radiology.csv"

# python analysis/a01_data_preprocessiong/m_inclusion_exclusion.py \
#             --data_dir $data_dir \
#             --dataset OV04 \
#             --csv_path $csv_path \
#             --modality radiology \
#             --save_dir $save_dir \
#             --extract_dir $extract_dir

# dicom to nifti (save to /parent/to/dataset/dataset_NIFTI)
# series="/home/sg2162/rds/hpc-work/Experiments/radiomics/OV04_included_raw_series.json"

# python analysis/a01_data_preprocessiong/m_dicom2nii.py \
#             --series $series \
#             --dataset OV04_endometriosis


# Endometrioma segmentation
radiology="/home/sg2162/rds/hpc-work/Experiments/radiomics/OV04_endometriosis_included_nifti.json"
save_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/OV04_endometriosis_Seg"
srun python analysis/a02_tumor_segmentation/m_endometrioma_segmentation.py \
            --radiology $radiology \
            --dataset OV04_endometriosis \
            --save_dir $save_dir