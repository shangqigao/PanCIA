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
conda activate PanCIA

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Force output flushing
export PYTHONUNBUFFERED=1   # if running Python
export SLURM_EXPORT_ENV=ALL
stdbuf -oL -eL echo "Starting job at $(date)"


CONFIG="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/phenotype_prediction.yaml"
OUTCOME="ImmuneSubtype"

# Common lists
RADIOMICS_MODES=("pyradiomics" "FMCIB" "BiomedParse" "LVMMed")
PATHOMICS_MODES=("UNI" "CONCH" "CHIEF")
AGG_MODES=("MEAN" "ABMIL" "SPARRA")

############################################
# 1️⃣ USED_OMICS = radiomics
############################################
USED_OMICS="radiomics"

for R_MODE in "${RADIOMICS_MODES[@]}"; do

  if [[ "$R_MODE" == "pyradiomics" || "$R_MODE" == "FMCIB" ]]; then
    R_AGG="None"

    echo "[radiomics] MODE=$R_MODE AGG=None"

    python analysis/a05_outcome_prediction/m_phenotype_prediction.py \
      --config_files "$CONFIG" \
      --override OUTCOME.VALUE="$OUTCOME" \
      --override PREDICTION.USED_OMICS.VALUE="$USED_OMICS" \
      --override RADIOMICS.MODE.VALUE="$R_MODE" \
      --override RADIOMICS.AGGREGATED_MODE.VALUE="$R_AGG" \
      --override PATHOMICS.MODE.VALUE="UNI" \
      --override PATHOMICS.AGGREGATED_MODE.VALUE="None"

  else
    for R_AGG in "${AGG_MODES[@]}"; do
      echo "[radiomics] MODE=$R_MODE AGG=$R_AGG"

      python analysis/a05_outcome_prediction/m_phenotype_prediction.py \
        --config_files "$CONFIG" \
        --override OUTCOME.VALUE="$OUTCOME" \
        --override PREDICTION.USED_OMICS.VALUE="$USED_OMICS" \
        --override RADIOMICS.MODE.VALUE="$R_MODE" \
        --override RADIOMICS.AGGREGATED_MODE.VALUE="$R_AGG" \
        --override PATHOMICS.MODE.VALUE="UNI" \
        --override PATHOMICS.AGGREGATED_MODE.VALUE="None"
    done
  fi
done


############################################
# 2️⃣ USED_OMICS = pathomics
############################################
USED_OMICS="pathomics"

for P_MODE in "${PATHOMICS_MODES[@]}"; do
  for P_AGG in "${AGG_MODES[@]}"; do

    echo "[pathomics] MODE=$P_MODE AGG=$P_AGG"

    python analysis/a05_outcome_prediction/m_phenotype_prediction.py \
      --config_files "$CONFIG" \
      --override OUTCOME.VALUE="$OUTCOME" \
      --override PREDICTION.USED_OMICS.VALUE="$USED_OMICS" \
      --override PATHOMICS.MODE.VALUE="$P_MODE" \
      --override PATHOMICS.AGGREGATED_MODE.VALUE="$P_AGG" \
      --override RADIOMICS.MODE.VALUE="LVMMed" \
      --override RADIOMICS.AGGREGATED_MODE.VALUE="None"
  done
done


############################################
# 3️⃣ USED_OMICS = radiopathomics
############################################
USED_OMICS="radiopathomics"

for R_MODE in "${RADIOMICS_MODES[@]}"; do

  if [[ "$R_MODE" == "pyradiomics" || "$R_MODE" == "FMCIB" ]]; then
    R_AGG="None"
    P_AGG="MEAN"

    for P_MODE in "${PATHOMICS_MODES[@]}"; do
      echo "[radiopathomics] R=$R_MODE(None) P=$P_MODE(MEAN)"

      python analysis/a05_outcome_prediction/m_phenotype_prediction.py \
        --config_files "$CONFIG" \
        --override OUTCOME.VALUE="$OUTCOME" \
        --override PREDICTION.USED_OMICS.VALUE="$USED_OMICS" \
        --override RADIOMICS.MODE.VALUE="$R_MODE" \
        --override RADIOMICS.AGGREGATED_MODE.VALUE="$R_AGG" \
        --override PATHOMICS.MODE.VALUE="$P_MODE" \
        --override PATHOMICS.AGGREGATED_MODE.VALUE="$P_AGG"
    done

  else
    for AGG in "${AGG_MODES[@]}"; do
      for P_MODE in "${PATHOMICS_MODES[@]}"; do

        echo "[radiopathomics] R=$R_MODE($AGG) P=$P_MODE($AGG)"

        python analysis/a05_outcome_prediction/m_phenotype_prediction.py \
          --config_files "$CONFIG" \
          --override OUTCOME.VALUE="$OUTCOME" \
          --override PREDICTION.USED_OMICS.VALUE="$USED_OMICS" \
          --override RADIOMICS.MODE.VALUE="$R_MODE" \
          --override RADIOMICS.AGGREGATED_MODE.VALUE="$AGG" \
          --override PATHOMICS.MODE.VALUE="$P_MODE" \
          --override PATHOMICS.AGGREGATED_MODE.VALUE="$AGG"
      done
    done
  fi
done
