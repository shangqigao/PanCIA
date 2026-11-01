# Edit these four paths (HPC locations), then: source env.sh
# Layout expected under each root (relative paths as on the Mac):
#   PANCIA_PANCIA     .../PanCIA                     (git clone; code)
#   PANCIA_CLINICAL   .../clinical                   (holds PanCIA_outputs/; unpack hpc_bundle_v3.tgz here)
#   PANCIA_R2SEG      .../R2Seg/<collection>/{images,labels}/
#   PANCIA_R2SEG_SEG  .../R2Seg_Seg/{BiomedParse,VoxTell,TotalSegmentator}/<collection>/
export PANCIA_PANCIA=/home/sg2162/projects/PanCIA
export PANCIA_CLINICAL=/home/sg2162/projects/Experiments/clinical
export PANCIA_R2SEG=/data/hdd1/PanCancer/R2Seg
export PANCIA_R2SEG_SEG=/data/hdd1/PanCancer/R2Seg_Seg
# python environment with numpy, scipy, pandas, nibabel, pyyaml, scikit-image (e.g. conda activate <env>)
CONDA_ENV="PanCIA"
