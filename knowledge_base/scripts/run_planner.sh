#!/usr/bin/env bash
# Run the KB planner over every TotalSegmentator output in TCGA_Seg (single-pass plans for the VoxTell anatomy call).
#
#   bash scripts/run_planner.sh              # full cohort (resumable: existing plans are skipped)
#   LIMIT=50 bash scripts/run_planner.sh     # smoke test on the first 50 TS outputs
#   WORKERS=16 bash scripts/run_planner.sh   # more processes
#   OVERWRITE=1 bash scripts/run_planner.sh  # re-plan everything (after a KB change)
#
# Layout expected (as written by m_tumor_segmentation.py):
#   $SEG_ROOT/TotalSegmentator/Radiology/<CT|MR>/<Project>/<series_uid>/<desc>_organ{.nii.gz,_labels.json,_structures.csv}
#   $SEG_ROOT/VoxTell/Radiology/...      /<desc>_tumor.nii.gz      (initial VoxTell tumour, rater 0)
#   $SEG_ROOT/BiomedParse/Radiology/...  /<desc>_tumor.nii.gz      (initial BiomedParse tumour, rater 1)
# Output: $OUT_DIR/<CT|MR>/<Project>/<series_uid>/<desc>_plan.json + $OUT_DIR/planner_batch_summary.csv
set -euo pipefail

KB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEG_ROOT="${SEG_ROOT:-/Users/sg2162/Datasets/CancerDatasets/PanCIA/TCGA-TCIA/TCGA_Seg}"
META_CSV="${META_CSV:-/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/clinical/scan_meta.csv}"
OUT_DIR="${OUT_DIR:-$SEG_ROOT/Planner/Radiology}"
TS_OBJ="${TS_OBJ:-organ}"
TUMOUR_OBJ="${TUMOUR_OBJ:-tumor}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-0}"
PYTHON="${PYTHON:-python3}"

TS_DIR="$SEG_ROOT/TotalSegmentator/Radiology"
VT_DIR="$SEG_ROOT/VoxTell/Radiology"
BP_DIR="$SEG_ROOT/BiomedParse/Radiology"

for d in "$TS_DIR" "$VT_DIR" "$BP_DIR"; do
  [[ -d "$d" ]] || { echo "missing folder: $d" >&2; exit 1; }
done
if [[ ! -f "$META_CSV" ]]; then
  echo "scan_meta.csv not found at $META_CSV — building it from the stage-analysis table" >&2
  "$PYTHON" "$KB_DIR/scripts/make_scan_meta.py" --stage_csv "$(dirname "$META_CSV")/seg_stage_analysis_v3.csv" --out "$META_CSV"
fi
"$PYTHON" -c "import numpy, scipy, nibabel, yaml" 2>/dev/null || { echo "need numpy scipy nibabel pyyaml in $PYTHON" >&2; exit 1; }

mkdir -p "$OUT_DIR"
echo "KB:        $KB_DIR (version $("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$KB_DIR/knowledge_base/kb_meta.yaml'))['version'])"))"
echo "TS:        $TS_DIR"
echo "tumour:    $VT_DIR ; $BP_DIR"
echo "meta:      $META_CSV"
echo "plans ->   $OUT_DIR   (workers=$WORKERS limit=$LIMIT overwrite=${OVERWRITE:-0})"

args=(--kb "$KB_DIR/knowledge_base"
      --ts_dir "$TS_DIR" --ts_obj "$TS_OBJ"
      --tumour_dir "$VT_DIR" --tumour_dir "$BP_DIR" --tumour_obj "$TUMOUR_OBJ"
      --meta_csv "$META_CSV" --out_dir "$OUT_DIR" --workers "$WORKERS")
[[ "$LIMIT" != "0" ]] && args+=(--limit "$LIMIT")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)

cd "$KB_DIR"
"$PYTHON" scripts/run_planner_batch.py "${args[@]}" 2>&1 | tee "$OUT_DIR/planner_run_$(date +%Y%m%d_%H%M%S).log"

echo
echo "quick audit of $OUT_DIR/planner_batch_summary.csv"
"$PYTHON" - "$OUT_DIR/planner_batch_summary.csv" <<'EOF'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
ok = [r for r in rows if r['status'] == 'ok']
print(f"{len(rows)} rows: ok={len(ok)} skipped={sum(r['status']=='skipped' for r in rows)} error={sum(r['status']=='error' for r in rows)}")
def tab(name, key):
    c = collections.Counter(r.get(key) or '' for r in ok)
    print(f"  {name}: " + ', '.join(f"{k or '-'}={v}" for k, v in c.most_common(12)))
tab('frame', 'frame_method'); tab('tumour evidence', 'tumour_evidence'); tab('task', 'task')
sec = collections.Counter()
for r in ok:
    for h in (r.get('hosts') or '').split(';'):
        p = h.split(':')
        if len(p) == 4 and p[1] != 'pri' and p[2] in ('ev1', 'ev2'):
            sec[p[0]] += 1
print("  secondary hosts with evidence: " + (', '.join(f"{k}={v}" for k, v in sec.most_common(10)) or 'none'))
by = collections.defaultdict(list)
for r in ok: by[r.get('cancer_type') or '?'].append(r)
print("  per cohort (n, none-evidence %, median prompts, budget-dropped %):")
for ct, rs in sorted(by.items()):
    none = sum(r['tumour_evidence'] == 'none' for r in rs) / len(rs) * 100
    pr = sorted(int(r['n_prompts']) for r in rs)[len(rs) // 2]
    drop = sum(int(r['n_budget_dropped'] or 0) > 0 for r in rs) / len(rs) * 100
    print(f"    {ct:12s} {len(rs):5d}  none={none:5.1f}%  prompts={pr:3d}  dropped={drop:5.1f}%")
errs = [r for r in rows if r['status'] == 'error']
if errs:
    print("  first errors:"); [print('   ', r['img_name'], '|', r['error'][:120]) for r in errs[:5]]
EOF
