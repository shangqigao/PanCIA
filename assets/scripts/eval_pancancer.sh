#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=32
#SBATCH --time=0-02:00:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
##SBATCH --qos=intr

## activate environment
source ~/.bashrc
conda activate biomedparse

data_root="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation/"
export DETECTRON2_DATASETS=$data_root
export DATASET=$data_root
export DATASET2=$data_root
export VLDATASET=$data_root
export PATH=$PATH:$data_root/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:$data_root/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here
srun --mpi=pmi2 python entry.py evaluate \
            --conf_files configs/biomed_seg_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 1 \
            FP16 True \
            WEIGHT True \
            STANDARD_TEXT_FOR_EVAL True \
            RESUME_FROM checkpoints/biomedparse_v1.pt \
            