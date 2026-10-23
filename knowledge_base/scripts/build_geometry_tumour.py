"""Tumour geometry statistics from R2Seg expert annotations (public datasets only; Ovary_Tumor_00 / OV04 is
confidential and excluded). Radiology segments masses, so benign masses (e.g. UMD fibroids) count as tumours.
Per case: multiplicity (components >= 0.1 ml, satellites within 20 mm, distant), shape of the main component, and,
where the dataset has expert host-organ labels, tumour-host relations of the main component.
Architecture (host): hollow = bladder, colon, cervix/uterus; solid = breast, kidney, liver, lung, pancreas, prostate.
Output: clinical/PanCIA_outputs/geometry_priors/tumour_cases.csv. Resumable, parallel:
python3 build_geometry_tumour.py [budget_s] [n_per_dataset]"""
import os, sys, json, glob, time, numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pancia_kb import geometry as G
H = os.path.expanduser('~'); R = f'{H}/mnt/R2Seg'; O = f'{H}/mnt/clinical/PanCIA_outputs/geometry_priors/tumour_cases.csv'
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150; NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 120
# dataset -> (architecture, host, tumour label, host-organ labels, lumen labels, excluded labels, file filter)
CFG = {'Bladder_Tumor_00': ('hollow', 'urinary_bladder', 1, [], [], [], None),
       'Breast_Tumor_00': ('solid', 'breast', 1, [], [], [], None),
       'Cervix_Tumor_00': ('hollow', 'uterus_cervix', 2, [1], [], [], None),
       'Colon_Tumor_00': ('hollow', 'colon', 14, [], [], [], None),
       'Kidney_Tumor_00': ('solid', 'kidney', 2, [1], [], [3], None),
       'Liver_Tumor_00': ('solid', 'liver', 14, [1], [], [], None),
       'Lung_Tumor_00': ('solid', 'lung', 1, [], [], [], None),
       'Lung_Tumor_01': ('solid', 'lung', 5, [1], [], [], None),
       'Pancreas_Tumor_00': ('solid', 'pancreas', 14, [4], [], [], None),
       'Prostate_Tumor_00': ('solid', 'prostate', 3, [], [], [], '_adc_seg'),
       'Uterus_Tumor_00': ('hollow', 'uterus', 3, [1], [2], [4], None)}
COLS = ['dataset', 'case', 'architecture', 'host', 'modality', 'max_sp', 'tumour_ml', 'n_comp', 'n_satellite', 'n_distant',
        'main_ml', 'r_eq_mm', 'sphericity', 'solidity', 'elong', 'flat', 'frac_main', 'irr_rms', 'irr_p95', 'irr_max',
        'host_ml', 'vol_ratio', 'contact_frac', 'lumen_contact_frac', 'outside_hull_frac', 'depth_rel',
        'host_irr_near_p95', 'host_irr_far_p95', 'bulge_ratio', 'note']


def work(task):
    ds, f, mod = task; arch, host, tl, nl, ll, xl, _ = CFG[ds]
    r = dict(dataset=ds, case=os.path.basename(f), architecture=arch, host=host, modality=mod)
    try:
        im = nib.load(f); a = np.asanyarray(im.dataobj); sp = np.array(im.header.get_zooms()[:3], float)
        c, _ = G.crop(a > 0, pad=10)  # memory: work only inside the labelled region
        if c is None: r['note'] = 'empty'; return r
        a = np.array(a[c], dtype=np.int16)
        T = a == tl; r['max_sp'] = float(sp.max())
        if T.sum() == 0: r['note'] = 'no_tumour'; return r
        r['tumour_ml'] = round(float(T.sum() * np.prod(sp) / 1000), 3)
        r.update(G.components(T, sp))
        lab, n = ndimage.label(T); sz = np.bincount(lab.ravel())[1:]; main = lab == (np.argmax(sz) + 1)
        s = G.shape_descriptors(main, sp, min_comp_ml=0.0)
        if s: r.update({k: v for k, v in s.items() if k not in ('vol_ml', 'n_comp')})
        if nl:
            N = np.isin(a, nl); L = np.isin(a, ll) if ll else None
            if N.sum() > 0:
                # host organ = the connected host part touching / nearest the main tumour (paired organs, e.g. kidneys)
                labN, nN = ndimage.label(N | main | (L if L is not None else False))
                k = np.bincount(labN[main].ravel()).argmax(); hostc = labN == k
                r['host_ml'] = round(float((hostc).sum() * np.prod(sp) / 1000), 2)
                rel = G.relation_descriptors(main, N & hostc, sp, (L & hostc) if L is not None else None)
                if rel: r.update(rel)
    except Exception as e:
        r['note'] = repr(e)[:100]
    return r


if __name__ == '__main__':
    done = set(pd.read_csv(O, usecols=['case']).case) if os.path.exists(O) else set()
    rng = np.random.default_rng(0); tasks = []
    for ds, cfg in CFG.items():
        js = json.load(open(f'{R}/{ds}/dataset.json')); mod = list(js['modality'].values())[0]
        fs = sorted(glob.glob(f'{R}/{ds}/labels/*.nii.gz'))
        if cfg[6]: fs = [f for f in fs if cfg[6] in f]
        fs = [fs[i] for i in rng.permutation(len(fs))[:NPER]]
        tasks += [(ds, f, mod) for f in fs if os.path.basename(f) not in done]
    t0 = time.time(); k = 0
    with Pool(3, maxtasksperchild=20) as pool:
        for r in pool.imap_unordered(work, tasks):
            pd.DataFrame([r]).reindex(columns=COLS).to_csv(O, mode='a', header=not os.path.exists(O), index=False); k += 1
            if time.time() - t0 > budget: pool.terminate(); break
    print('cases this call', k, 'done', len(done) + k, 'of', len(tasks) + len(done))
