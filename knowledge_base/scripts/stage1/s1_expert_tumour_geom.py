"""Expert-tumour geometry for the pan-cancer explanation thresholds (26 Nov 2026; claude/explanation_v1_gate_2026-11-26.md).
R2Seg expert tumour components >= 0.5 ml from scans NOT in the evidence-screen jobs (up to 30 scans per site, one image per patient;
cohort/r2seg_tumour_free.csv non-tumour scans excluded). Per component and per outline variant (orig; core = deepest 50% of voxels by
the distance transform): thickness (2 x max EDT, mm) and elongation sqrt(lmax/lmin) of the voxel coordinates (mm).
Output: ledger/evidence_screen/expert_tumour_geom.csv. Usage: python s1_expert_tumour_geom.py [budget_s] [workers]"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402
import s1_evidence_screen as SC                     # noqa: E402
O = os.path.join(SC.OUT, 'expert_tumour_geom.csv')
COLS = ['ds', 'case', 'site', 'comp', 'variant', 'ml', 'thick_mm', 'elong', 'note']


def geom(m, sp):
    th = float(2 * ndimage.distance_transform_edt(np.pad(m, 1), sampling=sp).max())
    xyz = np.argwhere(m) * np.asarray(sp)
    if len(xyz) < 4: return th, np.nan
    ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T))); return th, float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))


def one(job):
    ds, case = job; site = EX.SITE_ENTITY[ds]
    try:
        lf = os.path.join(SC.R, ds, 'labels', case + '_seg.nii.gz')
        T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour']); sp = tuple(float(x) for x in nib.load(lf).header.get_zooms()[:3])
        lb, n = ndimage.label(T, structure=EX.STRUCT26); vml = np.prod(sp) / 1000; rows = []
        for j, ob in enumerate(ndimage.find_objects(lb), start=1):
            pad = tuple(slice(max(0, s.start - 2), min(sh, s.stop + 2)) for s, sh in zip(ob, T.shape)); c = lb[pad] == j
            if c.sum() * vml < EX.MIN_ML: continue
            th, el = geom(c, sp); rows.append(dict(ds=ds, case=case, site=site, comp=j, variant='orig', ml=float(c.sum() * vml), thick_mm=th, elong=el))
            dt = ndimage.distance_transform_edt(np.pad(c, 1), sampling=sp)[1:-1, 1:-1, 1:-1]
            core = c & (dt >= np.median(dt[c])); cl, cn = ndimage.label(core, structure=EX.STRUCT26)
            if cn:
                core = cl == (np.bincount(cl.ravel())[1:].argmax() + 1); th, el = geom(core, sp)
                rows.append(dict(ds=ds, case=case, site=site, comp=j, variant='core50', ml=float(core.sum() * vml), thick_mm=th, elong=el))
        return rows or [dict(ds=ds, case=case, site=site, note='no tumour >= MIN_ML')]
    except Exception as e:
        return [dict(ds=ds, case=case, site=site, note='error ' + repr(e)[:120])]


def jobs(n_per=30, seed=1):
    J = pd.read_csv(os.path.join(SC.OUT, 'jobs.csv')); J['case'] = J.case.astype(str)
    used = {(d, SC.patient(d, c)) for d, c in zip(J.ds, J.case)}
    C = pd.read_csv(os.path.join(os.path.dirname(paths.ledger_dir()), 'cohort', 'r2seg_tumour_free.csv'))[['ds', 'case']]; C['case'] = C.case.astype(str)
    excl = set(zip(C.ds, C.case))
    t = pd.read_csv(os.path.join(paths.ledger_dir(), 'v86_r2seg_bp2', 'diag', 'anchor_truth_lite.csv'), usecols=['ds', 'case'], low_memory=False).drop_duplicates()
    t = t[t.ds != 'Uterus_Tumor_00']; t['case'] = t.case.astype(str)
    t = t[[(d, c) not in excl and (d, SC.patient(d, c)) not in used for d, c in zip(t.ds, t.case)]]
    t['site'] = t.ds.map(EX.SITE_ENTITY); t['pat'] = [SC.patient(d, c) for d, c in zip(t.ds, t.case)]
    out = [g.sample(frac=1, random_state=seed).drop_duplicates('pat').head(n_per) for _, g in t.groupby('site')]
    return [(r.ds, r.case) for r in pd.concat(out).itertuples()]


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150; W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    todo = jobs(); done = set()
    if os.path.exists(O):
        d = pd.read_csv(O, usecols=['ds', 'case']); done = set(zip(d.ds, d.case.astype(str)))
    todo = [j for j in todo if j not in done]; print('todo', len(todo), flush=True); t0 = time.time(); n = 0
    with Pool(W) as p:
        for rows in p.imap_unordered(one, todo):
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(O, mode='a', index=False, header=not os.path.exists(O)); n += 1
            if time.time() - t0 > budget: p.terminate(); break
    print('processed', n, 'in', round(time.time() - t0), 's', flush=True)
