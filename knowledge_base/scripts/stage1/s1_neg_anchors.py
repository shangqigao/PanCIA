"""What do the agents agree on in tumour-free scans? (13 Nov 2026). Per accepted stage-1 v8.4 anchor of one collection, its anatomy,
intensity and shape, for three groups: 'neg' (any accepted anchor on a tumour-free scan, i.e. no expert tumour voxel),
'pos_true' / 'pos_false' (accepted anchors on tumour scans that touch / miss the expert tumour). Expert labels: evaluation only.
Per anchor: rel, id, group, decision, kind, ml, host, host_cls; expert organ with the largest share of the anchor (FLARE organ labels
where the collection has them) and that share; TotalSegmentator organ with the largest share; HU (median, p10, p90) inside the anchor
and in a 2-5 mm shell around it (same expert organ when the anchor sits in one), contrast = median - shell median; depth (median
distance of the anchor voxels to the organ surface, mm); elongation (sqrt of the largest / smallest coordinate-covariance
eigenvalue, mm); slices spanned; scan aorta HU (median inside the expert aorta label; contrast phase).
Usage: python s1_neg_anchors.py --ds Liver_Tumor_00 [--max-pos 40] [--out DIR]   (resumable; writes DIR/neg_anchors_<ds>.csv)"""
import os, sys, json, glob, time, argparse, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402

ACC = ('accepted', 'accepted_override')


def unpack(z, key, box):
    shp = [b[1] - b[0] for b in box]
    return np.unpackbits(z[key])[:int(np.prod(shp))].reshape(shp).astype(bool)


def ts_map(ds, case, shape):
    f, j = EX.ts_paths(ds, case)
    if not os.path.exists(f):
        return None, {}
    lab = np.asarray(nib.load(f).dataobj).astype(np.int16)
    if lab.shape != shape:
        return None, {}
    return lab, {int(k): v for k, v in json.load(open(j))['labels'].items()}


def one(ds, stem, dec):
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
    L = EX._read_labels(lf).astype(np.int16)
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); vml = float(np.prod(sp)) / 1000.0
    T = np.isin(L, EX.APPEARANCE[ds]['tumour'])
    fz = os.path.join(paths.ledger_dir(), 'v84_r2seg', 'extents', ds, stem + '.npz')
    z = np.load(fz); meta = json.loads(str(z['meta']))
    if tuple(meta['shape']) != T.shape:
        return []
    I = np.asarray(img.dataobj, dtype=np.float32)
    names = {v: k for k, vs in EX.APPEARANCE[ds]['organs'].items() for v in vs}
    TS, tsn = ts_map(ds, stem, T.shape)
    aor = [v for v, n in names.items() if n == 'aorta']
    aorta_hu = float(np.median(I[np.isin(L, aor)])) if aor and np.isin(L, aor).any() else np.nan
    group_scan = 'neg' if not T.any() else 'pos'
    rows = []
    for i, an in meta['anchors'].items():
        i = int(i)
        if i not in dec.index or dec.loc[i, 'decision'] not in ACC:
            continue
        box = an['box']; sl = tuple(slice(b[0], b[1]) for b in box)
        a = np.zeros(T.shape, bool); a[sl] = unpack(z, f'a{i}', box)
        # working window: anchor box + 10 mm
        pad = np.ceil(10.0 / sp).astype(int)
        W = tuple(slice(max(0, b[0] - p), min(n, b[1] + p)) for b, p, n in zip(box, pad, T.shape))
        aw, Lw, Iw, Tw = a[W], L[W], I[W], T[W]
        true = bool((aw & Tw).any())
        lab_in = Lw[aw & ~Tw]
        org = 'none'; org_frac = 0.0
        if lab_in.size:
            v, c = np.unique(lab_in, return_counts=True)
            k = int(np.argmax(c)); org = names.get(int(v[k]), 'none') if v[k] else 'none'; org_frac = float(c[k]) / aw.sum()
        ts_org = 'n/a'; ts_frac = np.nan
        if TS is not None:
            v, c = np.unique(TS[W][aw], return_counts=True); k = int(np.argmax(c))
            ts_org = tsn.get(int(v[k]), 'background') if v[k] else 'background'; ts_frac = float(c[k]) / aw.sum()
        d_out = ndimage.distance_transform_edt(~aw, sampling=sp)
        shell = (d_out > 2) & (d_out <= 5) & ~Tw
        ov = [v for v, n in names.items() if n == org]
        if ov and np.isin(Lw, ov)[shell].sum() >= 20:
            shell &= np.isin(Lw, ov)
        hu = Iw[aw]; sh = Iw[shell]
        depth = np.nan
        if ov:
            O = np.isin(Lw, ov) | (Tw & aw)
            depth = float(np.median(ndimage.distance_transform_edt(O, sampling=sp)[aw]))
        xyz = np.argwhere(aw) * sp
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T))) if len(xyz) > 3 else np.array([1, 1, 1.0])
        rows.append(dict(rel=f'CT/{ds}/{stem}', id=i, group=group_scan if group_scan == 'neg' else ('pos_true' if true else 'pos_false'),
                         decision=dec.loc[i, 'decision'], kind=dec.loc[i, 'kind'], ml=float(aw.sum() * vml), host=dec.loc[i, 'host'],
                         host_cls=dec.loc[i, 'host_cls'], organ=org, organ_frac=round(org_frac, 3), ts_organ=ts_org,
                         ts_frac=round(ts_frac, 3) if ts_frac == ts_frac else np.nan,
                         hu_med=float(np.median(hu)), hu_p10=float(np.percentile(hu, 10)), hu_p90=float(np.percentile(hu, 90)),
                         shell_hu=float(np.median(sh)) if sh.size else np.nan,
                         contrast=float(np.median(hu) - np.median(sh)) if sh.size else np.nan, depth_mm=depth,
                         elong=float(np.sqrt(ev[-1] / max(ev[0], 1e-6))), n_slices=int(len(np.unique(np.argwhere(aw)[:, 2]))),
                         aorta_hu=aorta_hu))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ds', default='Liver_Tumor_00'); ap.add_argument('--max-pos', type=int, default=40)
    ap.add_argument('--out', default=None); a = ap.parse_args()
    O = a.out or os.path.join(paths.ledger_dir(), 'v84_r2seg', 'diag'); os.makedirs(O, exist_ok=True)
    f_out = os.path.join(O, f'neg_anchors_{a.ds}.csv'); f_done = os.path.join(O, f'neg_anchors_{a.ds}_done.txt')
    fs = sorted(glob.glob(os.path.join(paths.ledger_dir(), 'v84_r2seg', 'anchors*.csv')))
    d = pd.concat([pd.read_csv(f, usecols=['rel', 'id', 'kind', 'decision', 'host', 'host_cls']) for f in fs], ignore_index=True)
    d = d[d.rel.str.split('/').str[1] == a.ds].drop_duplicates(['rel', 'id'], keep='last')
    ev = pd.concat([pd.read_csv(f, usecols=['ds', 'case', 'method', 'T_ml'], dtype={'case': str})
                    for f in glob.glob(os.path.join(paths.ledger_dir(), 'v84_r2seg', 'eval', 'scans*.csv'))])
    ev = ev[(ev.method == 'vt') & (ev.ds == a.ds)].drop_duplicates('case')
    neg = sorted(ev[ev.T_ml == 0].case); pos = sorted(ev[ev.T_ml > 0].case)
    pos = list(np.random.default_rng(0).permutation(pos))[:a.max_pos]
    done = set(open(f_done).read().split()) if os.path.exists(f_done) else set()
    t0 = time.time()
    for stem in neg + pos:
        if stem in done:
            continue
        g = d[d.rel.str.endswith('/' + stem)].set_index('id')
        try:
            rows = one(a.ds, stem, g)
        except Exception:
            print('error', stem, traceback.format_exc()[-300:], flush=True); rows = []
        if rows:
            pd.DataFrame(rows).to_csv(f_out, mode='a', index=False, header=not os.path.exists(f_out))
        open(f_done, 'a').write(stem + '\n')
        print(stem, len(rows), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
