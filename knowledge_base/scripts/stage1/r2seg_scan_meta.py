"""R2Seg scan metadata for the stage-1 planner (8 Nov 2026): one row per scan with a TotalSegmentator output.

Columns (as scan_meta.csv for TCGA; run_planner_batch.py --meta_csv): img_name = '<collection>/<stem>' (the R2Seg output name),
modality, cancer_type (the TCGA project whose tumour phrases and KB spread list are used), host (KB anchor of the primary).
The collection label gives the cancer, as the TCGA project does. Collections with mixed subtypes use the most common subtype's
project (kidney KiTS -> TCGA-KIRC, lung NSCLC -> TCGA-LUAD); their phrase lists also hold subtype-neutral phrases.
All R2Seg scans with a TotalSegmentator output (user, 8 Nov: plan every scan): prostate T2 only (ADC has no TS output).
Uterus_Tumor_00 is benign (not a tumour in evaluations); it gets host uterus and no cancer type, so the planner uses its generic
tumour templates instead of an endometrial-cancer phrase. Ovary was removed from R2Seg.
Usage: python r2seg_scan_meta.py [--ts_dir <R2Seg_Seg>/TotalSegmentator] [--out <clinical>/PanCIA_outputs/cohort/r2seg_scan_meta.csv]"""
import os, sys, csv, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

COLL = {  # collection: (modality, TCGA project, KB host)
    'Bladder_Tumor_00': ('MR', 'TCGA-BLCA', 'urinary_bladder'), 'Breast_Tumor_00': ('MR', 'TCGA-BRCA', 'breast'),
    'Cervix_Tumor_00': ('MR', 'TCGA-CESC', 'cervix'), 'Colon_Tumor_00': ('CT', 'TCGA-COAD', 'colon'),
    'Kidney_Tumor_00': ('CT', 'TCGA-KIRC', 'kidney'), 'Liver_Tumor_00': ('CT', 'TCGA-LIHC', 'liver'),
    'Lung_Tumor_00': ('CT', 'TCGA-LUAD', 'lung'), 'Lung_Tumor_01': ('CT', 'TCGA-LUAD', 'lung'),
    'Pancreas_Tumor_00': ('CT', 'TCGA-PAAD', 'pancreas'), 'Prostate_Tumor_00': ('MR', 'TCGA-PRAD', 'prostate'),
    'Uterus_Tumor_00': ('MR', '', 'uterus')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts_dir', default=os.path.join(paths.root('R2Seg_Seg'), 'TotalSegmentator'))
    ap.add_argument('--out', default=os.path.join(paths.root('clinical'), 'PanCIA_outputs', 'cohort', 'r2seg_scan_meta.csv'))
    a = ap.parse_args()
    rows = []
    for c, (mod, ct, host) in COLL.items():
        for f in sorted(glob.glob(os.path.join(a.ts_dir, c, '*_organ_structures.csv'))):
            stem = os.path.basename(f)[:-len('_organ_structures.csv')]
            rows.append(dict(img_name=f'{c}/{stem}', modality=mod, cancer_type=ct, host=host, sex=''))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['img_name', 'modality', 'cancer_type', 'host', 'sex']); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} scans -> {a.out}')
    for c in COLL:
        print(f'  {c}: {sum(r["img_name"].startswith(c + "/") for r in rows)}')


if __name__ == '__main__':
    main()
