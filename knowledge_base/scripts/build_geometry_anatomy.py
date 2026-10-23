"""Anatomy geometry statistics from the TotalSegmentator CT (v3.0.0) and MR (v2.0.0) datasets (expert-reviewed labels,
public, Zenodo). Never uses sex/gender. Normal-organ filter (CT meta): an organ is kept if pathology == no_pathology, or
the pathology location does not include the organ's region and is not 'unclear'. MR has no pathology field (kept, flagged).
Organs touching the image border (truncated by the field of view) are skipped.
Output: clinical/PanCIA_outputs/geometry_priors/anatomy_cases.csv (one row per case x organ).
Resumable, parallel: python3 build_geometry_anatomy.py [budget_s] [n_ct] [n_mr]"""
import os, sys, time, numpy as np, pandas as pd, nibabel as nib
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pancia_kb import geometry as G
H = os.path.expanduser('~'); TS = f'{H}/mnt/Totalsegmentator'
O = f'{H}/mnt/clinical/PanCIA_outputs/geometry_priors/anatomy_cases.csv'
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150
NCT = int(sys.argv[2]) if len(sys.argv) > 2 else 300; NMR = int(sys.argv[3]) if len(sys.argv) > 3 else 250
ORG = {  # organ -> (TS files, region, hollow)
    'liver': (['liver'], 'abdomen', 0), 'spleen': (['spleen'], 'abdomen', 0), 'pancreas': (['pancreas'], 'abdomen', 0),
    'kidney_left': (['kidney_left'], 'abdomen', 0), 'kidney_right': (['kidney_right'], 'abdomen', 0),
    'gallbladder': (['gallbladder'], 'abdomen', 1), 'stomach': (['stomach'], 'abdomen', 1), 'duodenum': (['duodenum'], 'abdomen', 1),
    'colon': (['colon'], 'abdomen,pelvis', 1), 'small_bowel': (['small_bowel'], 'abdomen,pelvis', 1),
    'esophagus': (['esophagus'], 'thorax,throat', 1), 'urinary_bladder': (['urinary_bladder'], 'pelvis', 1),
    'prostate': (['prostate'], 'pelvis', 0), 'thyroid_gland': (['thyroid_gland'], 'throat', 0),
    'lung_left': (['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_left'], 'thorax', 0),
    'lung_right': (['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'lung_right'], 'thorax', 0)}
COLS = ['source', 'case', 'organ', 'hollow', 'normal_flag', 'max_sp', 'vol_ml', 'main_ml', 'r_eq_mm', 'sphericity', 'solidity',
        'elong', 'flat', 'n_comp', 'frac_main', 'irr_rms', 'irr_p95', 'irr_max', 'wall_med_mm', 'wall_p95_mm', 'wall_rel_med', 'lumen_frac', 'note']


def eligible(meta_row, region):
    p = str(meta_row.get('pathology', 'nan')); loc = str(meta_row.get('pathology_location', 'nan'))
    if p == 'no_pathology': return 'normal'
    if p in ('nan', ''): return 'unknown'
    if 'unclear' in loc or any(r in loc for r in region.split(',')): return None
    return 'other_region'


def work(task):
    src, case, meta = task; d = f"{TS}/{'Totalsegmentator_dataset_v300' if src == 'CT' else 'TotalsegmentatorMRI_dataset_v200'}/{case}"
    rows, img = [], None
    try:
        for organ, (files, region, hollow) in ORG.items():
            fl = [f'{d}/segmentations/{f}.nii.gz' for f in files if os.path.exists(f'{d}/segmentations/{f}.nii.gz')]
            if not fl: continue
            flag = eligible(meta, region) if src == 'CT' else 'mr_unknown'
            if flag is None: continue
            ims = [nib.load(f) for f in fl]; sp = np.array(ims[0].header.get_zooms()[:3], float)
            m = np.zeros(ims[0].shape[:3], bool)
            for im in ims: m |= np.asarray(im.dataobj) > 0
            if m.sum() == 0: continue
            idx = np.argwhere(m)
            if (idx.min(0) == 0).any() or (idx.max(0) == np.array(m.shape) - 1).any():
                rows.append(dict(source=src, case=case, organ=organ, note='truncated')); continue
            r = dict(source=src, case=case, organ=organ, hollow=hollow, normal_flag=flag, max_sp=float(sp.max()))
            s = G.shape_descriptors(m, sp, min_comp_ml=1.0)
            if s is None: continue
            r.update(s)
            if hollow:
                if img is None: img = np.asarray(nib.load(f"{d}/{'ct' if src == 'CT' else 'mri'}.nii.gz").dataobj, np.float32)
                w = G.wall_descriptors(m, img, sp)
                if w: r.update(w)
            rows.append(r)
    except Exception as e:
        rows.append(dict(source=src, case=case, note=repr(e)[:100]))
    return (src, case), rows


if __name__ == '__main__':
    done = set(pd.read_csv(O, usecols=['case', 'source']).apply(lambda x: f'{x.source}/{x.case}', axis=1)) if os.path.exists(O) else set()
    rng = np.random.default_rng(0); tasks = []
    for src, sub, n in (('CT', 'Totalsegmentator_dataset_v300', NCT), ('MR', 'TotalsegmentatorMRI_dataset_v200', NMR)):
        meta = pd.read_csv(f'{TS}/{sub}/meta.csv', sep=';', encoding='utf-8-sig').drop(columns=['gender'], errors='ignore')
        meta = meta[[os.path.isdir(f'{TS}/{sub}/{c}') for c in meta.image_id]]
        pick = meta.iloc[rng.permutation(len(meta))[:n]]
        tasks += [(src, r.image_id, r.to_dict()) for _, r in pick.iterrows() if f'{src}/{r.image_id}' not in done]
    t0 = time.time(); k = 0
    with Pool(4) as pool:
        for (src_, case), rows in pool.imap_unordered(work, tasks):
            if rows: pd.DataFrame(rows).reindex(columns=COLS).to_csv(O, mode='a', header=not os.path.exists(O), index=False)
            else: pd.DataFrame([dict(source=src_, case=case, note='no_organs')]).reindex(columns=COLS).to_csv(O, mode='a', header=not os.path.exists(O), index=False)
            k += 1
            if time.time() - t0 > budget: pool.terminate(); break
    print('cases this call', k, 'done', len(done) + k, 'of', len(tasks) + len(done))
