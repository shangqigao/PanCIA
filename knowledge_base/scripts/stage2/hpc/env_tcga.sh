# Edit for the HPC, then: source env_tcga.sh
#   PANCIA_PANCIA     .../PanCIA (git clone; code + knowledge_base/knowledge_base/stage2/stage2_fit_agent_beyond.json)
#   PANCIA_OUTPUTS    folder holding ledger/ (v86_tcga_all_bp2, v85/extent/fit_psi.json, v8/..., v6/...) and receiving stage2/<tag>/
#   PANCIA_TCGA_SEG   .../TCGA_Seg/{BiomedParse,VoxTell,TotalSegmentator,Planner}/Radiology/...
#   PANCIA_TCGA_NIFTI .../TCGA_NIFTI
export PANCIA_PANCIA=/path/to/PanCIA
export PANCIA_OUTPUTS=/path/to/PanCIA_outputs
export PANCIA_TCGA_SEG=/path/to/TCGA_Seg
export PANCIA_TCGA_NIFTI=/path/to/TCGA_NIFTI
CONDA_ENV="pancia_s2gpu"     # numpy, scipy, pandas, nibabel, pyyaml, edt, cupy (cupy-cuda12x on CUDA 12)
