"""Lite anchor truth for any R2Seg stage-1 ledger (26 Nov 2026): per anchor, does it touch an expert tumour (any voxel, 26-connected
components; also share inside, and whether it touches a component >= EX.MIN_ML). No extents. Expert labels are used for evaluation only.
Output: ledger/<LEDGER>/diag/anchor_truth_lite.csv (ds, case, site, id, kind, ml, true, true_big, in_frac). Resumable; time budget.
Usage: LEDGER=v86_r2seg_bp2 python s1_anchor_truth_lite.py [budget_s] [workers]"""
import os, sys, json, glob, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402
COLS = ['ds', 'case', 'site', 'id', 'kind', 'ml', 'true', 'true_big', 'in_frac', 'tumour_free', 'note']
LEDGER = os.environ.get('LEDGER', 'v86_r2seg_bp2')
D = os.path.join(paths.ledger_dir(), LEDGER); O = os.path.join(D, 'diag', 'anchor_truth_lite.csv')


def unpack(z, key, box):
    shp = [b[1] - b[0] for b in box]
    return np.unpackbits(z[key])[:int(np.prod(shp))].reshape(shp).astype(bool)


def one(fz):
    ds = fz.split('/')[-2]; stem = os.path.basename(fz)[:-4]
    try:
        lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
        if not os.path.exists(lf) or ds not in EX.SITE_ENTITY:
            return [dict(ds=ds, case=stem, note='no label')]
        T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour'])
        vml = float(np.prod(nib.load(lf).header.get_zooms()[:3])) / 1000.0
        labT, nT = ndimage.label(T, structure=EX.STRUCT26)
        tsz = np.bincount(labT.ravel(), minlength=nT + 1)
        z = np.load(fz); meta = json.loads(str(z['meta']))
        if tuple(meta['shape']) != T.shape:
            return [dict(ds=ds, case=stem, note='grid mismatch')]
        rows = []
        for i, an in meta['anchors'].items():
            box = an['box']; sl = tuple(slice(b[0], b[1]) for b in box); a = unpack(z, f'a{i}', box)
            ids = set(np.unique(labT[sl][a]).tolist()) - {0}
            rows.append(dict(ds=ds, case=stem, site=EX.SITE_ENTITY[ds], id=int(i), kind=an.get('kind'), ml=float(a.sum() * vml),
                             true=bool(ids), true_big=any(tsz[j] * vml >= EX.MIN_ML for j in ids),
                             in_frac=int((a & T[sl]).sum()) / max(int(a.sum()), 1), tumour_free=bool(nT == 0)))
        return rows or [dict(ds=ds, case=stem, note='no anchors', tumour_free=bool(nT == 0))]
    except Exception as e:
        return [dict(ds=ds, case=stem, note='error ' + repr(e)[:150])]


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 160; W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(os.path.dirname(O), exist_ok=True)
    files = sorted(glob.glob(os.path.join(D, 'extents', '*', '*.npz')))
    done = set()
    if os.path.exists(O):
        d = pd.read_csv(O, usecols=['ds', 'case']); done = set(zip(d.ds, d.case.astype(str)))
    todo = [f for f in files if (f.split('/')[-2], os.path.basename(f)[:-4]) not in done]
    print('total', len(files), 'todo', len(todo), flush=True)
    t0 = time.time(); n = 0
    with Pool(W) as p:
        for rows in p.imap_unordered(one, todo, chunksize=2):
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(O, mode='a', index=False, header=not os.path.exists(O)); n += 1
            if time.time() - t0 > budget:
                p.terminate(); break
    print('processed', n, 'in', round(time.time() - t0), 's', flush=True)
