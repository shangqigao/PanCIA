#!/bin/bash
# Planner for R2Seg (8 Nov 2026): one plan per scan, as for TCGA, for the VoxTell plan call (run on the HPC with
# analysis/a02_tumor_segmentation/m_tumor_segmentation.py --dataset R2Seg --model VoxTell --planner_dir <R2Seg_Seg>/Planner).
#   bash run_r2seg_plan.sh meta       scan list (every R2Seg scan with a TotalSegmentator output; 4,048) -> r2seg_scan_meta.csv
#   bash run_r2seg_plan.sh planner    plans -> $R2SEG_SEG/Planner/<collection>/<stem>_plan.json  (CPU; resumable; OVERWRITE=1 to redo)
# Defaults: the paths on the Mac (override as environment variables elsewhere). Python needs numpy, scipy, nibabel, pyyaml.
set -uo pipefail
PANCIA=${PANCIA:-$(cd "$(dirname "$0")/../../.." && pwd)}    # repo root (holds knowledge_base/), from this script's location
R2SEG_SEG=${R2SEG_SEG:-/Users/sg2162/Datasets/CancerDatasets/R2Seg_Seg}
CLINICAL=${CLINICAL:-/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/clinical}
PY=${PY:-python}
WORKERS=${WORKERS:-6}
META=$CLINICAL/PanCIA_outputs/cohort/r2seg_scan_meta.csv
case "${1:-}" in
  meta)
    PANCIA_PANCIA=$PANCIA PANCIA_CLINICAL=$CLINICAL PANCIA_R2SEG_SEG=$R2SEG_SEG \
      "$PY" "$PANCIA/knowledge_base/scripts/stage1/r2seg_scan_meta.py" --ts_dir "$R2SEG_SEG/TotalSegmentator" --out "$META"
    ;;
  planner)
    mkdir -p "$R2SEG_SEG/Planner"; cd "$PANCIA/knowledge_base"
    "$PY" scripts/run_planner_batch.py --kb knowledge_base --ts_dir "$R2SEG_SEG/TotalSegmentator" --ts_obj organ \
      --tumour_dir "$R2SEG_SEG/VoxTell" --tumour_dir "$R2SEG_SEG/BiomedParse" --tumour_obj tumor \
      --meta_csv "$META" --only_meta --out_dir "$R2SEG_SEG/Planner" --workers "$WORKERS" $([ "${OVERWRITE:-0}" = 1 ] && echo --overwrite) \
      2>&1 | tee "$R2SEG_SEG/Planner/planner_$(date +%Y%m%d_%H%M%S).log"
    ;;
  *) echo "usage: bash $0 {meta|planner}"; exit 1 ;;
esac
