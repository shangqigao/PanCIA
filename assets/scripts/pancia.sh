#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL2-GPU
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
conda activate /home/sg2162/rds/hpc-work/miniconda3/PanCIA

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Force output flushing
export PYTHONUNBUFFERED=1   # if running Python
export SLURM_EXPORT_ENV=ALL
stdbuf -oL -eL echo "Starting job at $(date)"

#---------------MAMA-MIA----------------
# fit Beta distributions on training data
# img_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast_Tumor/train"
# save_dir="/home/sg2162/rds/hpc-work/PanCIA/analysis/tumor_segmentation"
# srun --mpi=pmi2 python analysis/a02_tumor_segmentation/m_fit_beta_distribution.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir

# tumor segmentation sanity test
# img_dir="/home/sg2162/rds/hpc-work/sanity-check/images"
# save_dir="/home/sg2162/rds/hpc-work/sanity-check/predictions"
# beta_params="/home/sg2162/rds/hpc-work/BCIA/CIA/analysis/tumor_segmentation/Beta_params.json"
# meta_info="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/clinical_and_imaging_info.xlsx"
# srun python analysis/a02_tumor_segmentation/m_tumor_segmentation.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir \
#             --beta_params $beta_params \
#             --meta_info $meta_info

# tumor segmentation MAMA-MIA
# img_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/images"
# save_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/segmentations/BiomedParse"
# beta_params="/home/sg2162/rds/hpc-work/BCIA/CIA/analysis/tumor_segmentation/Beta_params.json"
# meta_info="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/clinical_and_imaging_info.xlsx"
# srun python analysis/a02_tumor_segmentation/m_tumor_segmentation.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir \
#             --beta_params $beta_params \
#             --meta_info $meta_info

# extract radiomic features
# img_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/images"
# lab_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/segmentations"
# save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
# meta_info="/home/sg2162/rds/hpc-work/Experiments/clinical/MAMA-MIA_clinical_and_imaging_info.xlsx"

# python analysis/a03_feature_extraction/m_radiomics_extraction.py \
#             --img_dir $img_dir \
#             --lab_dir $lab_dir \
#             --save_dir $save_dir \
#             --meta_info $meta_info 

#----------------Pan-Cancer--------------------
# radiology exclusion and inclusion
# data_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer"
# save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"

# python analysis/a01_data_preprocessiong/m_inclusion_exclusion.py \
#             --data_dir $data_dir \
#             --dataset TCGA \
#             --modality radiology \
#             --save_dir $save_dir

# dicom to nifit
series="/home/sg2162/rds/hpc-work/Experiments/radiomics/TCGA_included_raw_series.json"

python analysis/a01_data_preprocessiong/m_dicom2nii.py \
            --series $series \
            --dataset TCGA

# pathology exclusion and inclusion
# data_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer"
# save_dir="/home/sg2162/rds/hpc-work/Experiments/pathomics"

# python analysis/a01_data_preprocessiong/m_inclusion_exclusion.py \
#             --data_dir $data_dir \
#             --dataset TCGA \
#             --modality pathology \
#             --save_dir $save_dir

# subject exclusion and inclusion
# included_nifti="/home/sg2162/rds/hpc-work/Experiments/radiomics/TCGA_included_nifti.json"
# included_wsi="/home/sg2162/rds/hpc-work/Experiments/pathomics/TCGA_included_wsi.json"
# meta_data="/home/sg2162/rds/hpc-work/Experiments/clinical/TCGA_pathology_has_radiology.csv"
# save_dir="/home/sg2162/rds/hpc-work/Experiments/clinical"

# python analysis/a01_data_preprocessiong/m_sortout_subjects.py \
#             --included_nifti $included_nifti \
#             --included_wsi $included_wsi \
#             --meta_data $meta_data \
#             --dataset TCGA \
#             --save_dir $save_dir

# pan-cancer segmentation
# radiology="/home/sg2162/rds/hpc-work/Experiments/clinical/TCGA_included_subjects.json"
# save_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_Seg"
# srun python analysis/a02_tumor_segmentation/m_tumor_segmentation.py \
#             --radiology $radiology \
#             --dataset TCGA \
#             --save_dir $save_dir
    

# classification
# img_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/images"
# save_radiomics_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
# save_clinical_dir="/home/sg2162/rds/hpc-work/Experiments/clinical"
# save_model_dir="/home/sg2162/rds/hpc-work/Experiments/outcomes"

# python analysis/a05_outcome_prediction/m_therapy_response.py \
#             --img_dir $img_dir \
#             --save_radiomics_dir $save_radiomics_dir \
#             --save_clinical_dir $save_clinical_dir \
#             --save_model_dir $save_model_dir  


# survival
# img_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/images"
# save_radiomics_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
# save_clinical_dir="/home/sg2162/rds/hpc-work/Experiments/clinical"
# save_model_dir="/home/sg2162/rds/hpc-work/Experiments/outcomes"

# python analysis/a05_outcome_prediction/m_survival_analysis.py \
#             --img_dir $img_dir \
#             --outcome recurrence \
#             --save_radiomics_dir $save_radiomics_dir \
#             --save_clinical_dir $save_clinical_dir \
#             --save_model_dir $save_model_dir     