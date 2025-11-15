#!/bin/bash

#SBATCH -A PION-P3-DAWN-GPU
##SBATCH -A PION-P3-CPU
#SBATCH -J pancia
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
##SBATCH --time=0-00:10:00
#SBATCH --partition=pvc9
#SBATCH --gres=gpu:1
##SBATCH --qos=intr

## load dawn
module purge
module load default-dawn

## load MPI
module av intel-oneapi-mpi

## activate environment
source ~/.bashrc
conda activate /home/sg2162/rds/hpc-work/miniconda3/PanCIA

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Force output flushing
export PYTHONUNBUFFERED=1   # if running Python
export SLURM_EXPORT_ENV=ALL
stdbuf -oL -eL echo "Starting job at $(date)"

# python analysis/utilities/m_prepare_biomedparse_TumorSegmentation_dataset.py
# python analysis/utilities/m_calculate_pancancer_segmentation_metrics.py

# fit Beta distributions on training data
# img_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast_Tumor/train"
# save_dir="/home/sg2162/rds/hpc-work/PanCIA/analysis/tumor_segmentation"
# srun --mpi=pmi2 python analysis/tumor_segmentation/m_fit_beta_distribution.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir

# tumor segmentation sanity test
# img_dir="/home/sg2162/rds/hpc-work/sanity-check/debugs"
# save_dir="/home/sg2162/rds/hpc-work/sanity-check/predictions"
# beta_params="/home/sg2162/rds/hpc-work/BCIA/CIA/analysis/tumor_segmentation/Beta_params.json"
# meta_info="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/clinical_and_imaging_info.xlsx"
# srun python analysis/tumor_segmentation/m_tumor_segmentation.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir \
#             --beta_params $beta_params \
#             --meta_info $meta_info

# tumor segmentation MAMA-MIA
# img_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/images"
# save_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/segmentations/BiomedParse"
# beta_params="/home/sg2162/rds/hpc-work/BCIA/CIA/analysis/tumor_segmentation/Beta_params.json"
# meta_info="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/MAMA-MIA/clinical_and_imaging_info.xlsx"
# srun python analysis/tumor_segmentation/m_tumor_segmentation.py \
#             --img_dir $img_dir \
#             --save_dir $save_dir \
#             --beta_params $beta_params \
#             --meta_info $meta_info

# extract radiomic features
# radiomics_config="/home/sg2162/rds/hpc-work/PanCIA/configs/feature_extraction/radiomics_extraction.yaml"
# python analysis/a03_feature_extraction/m_radiomics_extraction.py --config_files $radiomics_config
        
# extract pathomic features
# pathomics_config="/home/sg2162/rds/hpc-work/PanCIA/configs/feature_extraction/pathomics_extraction.yaml"
# python analysis/a03_feature_extraction/m_pathomics_extraction.py --config_files $pathomics_config


# survival analysis
# survival_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/survival_analysis.yaml"
# python analysis/a05_outcome_prediction/m_survival_analysis.py --config_files $survival_config

# phenotype prediction
# phenotype_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/phenotype_prediction.yaml"
# python analysis/a05_outcome_prediction/m_phenotype_prediction.py --config_files $phenotype_config

# signature prediction
signature_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/signature_prediction.yaml"
python analysis/a05_outcome_prediction/m_signature_prediction.py --config_files $signature_config