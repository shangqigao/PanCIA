#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=32
#SBATCH --time=0-36:00:00
##SBATCH --time=0-00:08:00
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
srun --mpi=pmi2 python entry.py train \
            --conf_files configs/pancia_bayes_seg_lang.yaml \
            --overrides \
            FP16 True \
            RANDOM_SEED 2024 \
            BioMed.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            MODEL.ENCODER.BINARY_CLASSES True \
            TEST.BATCH_SIZE_TOTAL 4 \
            TRAIN.BATCH_SIZE_TOTAL 4 \
            TRAIN.BATCH_SIZE_PER_GPU 4 \
            SOLVER.MAX_NUM_EPOCHS 5 \
            SOLVER.BASE_LR 0.00001 \
            SOLVER.FIX_PARAM.decomposition False \
            SOLVER.FIX_PARAM.backbone True \
            SOLVER.FIX_PARAM.lang_encoder True \
            SOLVER.FIX_PARAM.pixel_decoder False \
            MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 1.0 \
            MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 1.0 \
            MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 1.0 \
            MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
            MODEL.DECODER.SPATIAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            LOADER.SAMPLE_PROB prop \
            BioMed.INPUT.RANDOM_ROTATE True \
            BioMed.INPUT.MRI_AUG_ICNB False \
            FIND_UNUSED_PARAMETERS True \
            ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
            MODEL.DECODER.SPATIAL.MAX_ITER 0 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            STROKE_SAMPLER.MAX_CANDIDATE 10 \
            WEIGHT True \
            RESUME_FROM checkpoints/bayes_multiphase_breastcancer.pt