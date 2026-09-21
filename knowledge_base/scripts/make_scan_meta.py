"""Build scan_meta.csv for run_planner_batch.py from the stage-analysis table (series_uid, project, Modality).

  python scripts/make_scan_meta.py --stage_csv <clinical>/seg_stage_analysis_v3.csv --out scan_meta.csv

Columns written: series_uid, modality, cancer_type, site (Radiology/<CT|MR>/<project>). Sex is NOT included by
default (optional metadata, never used for filtering); pass --sex_csv <file> with columns case_id,sex to add it.
"""
import argparse, csv

ap = argparse.ArgumentParser()
ap.add_argument('--stage_csv', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--sex_csv', default=None)
a = ap.parse_args()
sex = {}
if a.sex_csv:
    with open(a.sex_csv, newline='') as f:
        for r in csv.DictReader(f):
            sex[r['case_id']] = r['sex']
rows, seen = [], set()
with open(a.stage_csv, newline='') as f:
    for r in csv.DictReader(f):
        uid = r['series_uid']
        if uid in seen: continue
        seen.add(uid)
        rows.append(dict(series_uid=uid, modality=r['Modality'], cancer_type=r['project'], site=f"Radiology/{r['Modality']}/{r['project']}",
                         sex=sex.get(r.get('Subject ID', ''), '')))
with open(a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['series_uid', 'modality', 'cancer_type', 'site', 'sex']); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} series -> {a.out}')
