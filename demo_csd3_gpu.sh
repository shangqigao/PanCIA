#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL2-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
##SBATCH --cpus-per-task=8
#SBATCH --time=0-00:08:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --qos=intr

## activate environment
source ~/.bashrc
conda activate biomedparse

## test
srun --mpi=pmi2 python example_prediction.py
