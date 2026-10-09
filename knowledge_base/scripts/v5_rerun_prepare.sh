#!/bin/bash
# Prepare the Ledger_v5 partial rerun on HPC (9 Oct 2026). Moves the outputs of the series in rel_list_v5_rerun.txt
# (22 that crashed + 450 with lesions wrongly recovered through shared claims) and the run-1 summary aside into
# Ledger_v5_replaced/, so the resumable run recomputes exactly these series with the fixed r2_em.py.
#   bash scripts/v5_rerun_prepare.sh            then:   REL_LIST=$PWD/scripts/rel_list_v5_rerun.txt sbatch scripts/run_ledger_v4.sh
set -euo pipefail
SEG_ROOT="${SEG_ROOT:-/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/TCGA_Seg}"
KB_DIR="${KB_DIR:-/home/sg2162/rds/hpc-work/PanCIA/knowledge_base}"
L="$SEG_ROOT/Ledger_v5/Radiology"; A="$SEG_ROOT/Ledger_v5_replaced/Radiology"; LIST="$KB_DIR/scripts/rel_list_v5_rerun.txt"
[ "$(md5sum "$KB_DIR/pancia_kb/r2_em.py" | cut -c1-8)" = "1bfd8725" ] || { echo "ERROR: pancia_kb/r2_em.py is not the fixed version (md5 1bfd8725...)"; exit 1; }
[ "$(md5sum "$KB_DIR/scripts/run_lesion_ledger.py" | cut -c1-8)" = "e309e21a" ] || { echo "ERROR: scripts/run_lesion_ledger.py is not the fixed version (md5 e309e21a...)"; exit 1; }
[ -f "$LIST" ] || { echo "ERROR: $LIST not found"; exit 1; }
mkdir -p "$A"
[ -f "$L/ledger_summary.csv" ] && mv -n "$L/ledger_summary.csv" "$A/ledger_summary_run1.csv"
n=0
while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  for s in _lesions.json _lesions.npz _r2maps.npz; do
    if [ -e "$L/$rel$s" ]; then mkdir -p "$(dirname "$A/$rel")"; mv -n "$L/$rel$s" "$A/$rel$s"; n=$((n+1)); fi
  done
done < "$LIST"
echo "moved $n files for $(grep -c . "$LIST") series to $A"
echo "next: REL_LIST=$LIST sbatch scripts/run_ledger_v4.sh"
