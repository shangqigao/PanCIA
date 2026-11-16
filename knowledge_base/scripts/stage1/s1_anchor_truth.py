"""Diagnosis of the v8.4 decisions on R2Seg (13 Nov 2026): every stage-1 anchor and candidate extent matched to the expert tumour.
Expert labels are used for evaluation only; nothing is fitted here. Uterus_Tumor_00 (benign) is not evaluated.

Per anchor (anchor_truth[_s<i>].csv): ds, case, site, id, ml, true (touches expert tumour), in_frac (share of the anchor inside expert
tumour), tumours (ids of the expert components >= 0.5 ml it touches; id = label in the 26-connected component map), G_ml (their
volume), xi_new / xi_v84 (index of the chosen extent with the refitted / v8.4 psi).
Per candidate extent of a TRUE anchor (extent_truth[_s<i>].csv): j, name, ml, logratio, flags, dice / precision / recall against G
(the expert components the anchor touches).
Per expert tumour >= 0.5 ml (tumour_truth[_s<i>].csv): ds, case, site, tumour, ml, anchors touching it (ids).
Decisions, levels, tests and signals come from ledger/v84_r2seg/anchors*.csv; the analysis joins them (s1_diagnose.py).
Outputs: clinical/PanCIA_outputs/ledger/v84_r2seg/diag/. Usage: [SHARD=i/n] python s1_anchor_truth.py  |  python s1_anchor_truth.py --shards 4
--tumours-only (fix 13 Nov): rebuild the expert-tumour rows for scans that have anchor rows but no tumour rows (an earlier run of this
script wrote anchor/extent rows without tumour rows, and resumed runs skipped those scans); writes tumour_truth_fix[_s<i>].csv."""
import os, sys, csv, json, glob, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402


def out_dir():
    d = os.path.join(paths.ledger_dir(), 'v84_r2seg', 'diag'); os.makedirs(d, exist_ok=True); return d


def psi_pair(site):
    pj = json.load(open(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json')))
    old = G2.Psi(json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))['psi'])
    new = G2.Psi(dict(edges=pj['psi']['edges'], val=pj['folds'][site][pj['method']])) if (pj.get('adopted') and site in (pj.get('folds') or {})) else G2.Psi(pj['psi'])
    return old, new


def unpack(z, key, box):
    shp = [b[1] - b[0] for b in box]
    return np.unpackbits(z[key])[:int(np.prod(shp))].reshape(shp).astype(bool)


def one(ds, stem, psis):
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
    fz = os.path.join(paths.ledger_dir(), 'v84_r2seg', 'extents', ds, stem + '.npz')
    site = EX.SITE_ENTITY[ds]
    T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour'])
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    labT, nT = ndimage.label(T, structure=EX.STRUCT26)
    tsz = np.bincount(labT.ravel(), minlength=nT + 1)
    big = {j for j in range(1, nT + 1) if tsz[j] * vml >= EX.MIN_ML}
    A, E, U = [], [], []
    touch = {j: [] for j in big}
    if os.path.exists(fz):
        z = np.load(fz); meta = json.loads(str(z['meta']))
        if tuple(meta['shape']) != T.shape:
            return [dict(ds=ds, case=stem, site=site, note='grid mismatch')], [], []
        old, new = psis
        for i, an in meta['anchors'].items():
            box = an['box']; sl = tuple(slice(b[0], b[1]) for b in box)
            a = unpack(z, f'a{i}', box)
            ids = sorted(set(np.unique(labT[sl][a]).tolist()) - {0})
            n_in = int((a & T[sl]).sum())
            ex = an['extents']
            cands = [dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ex]
            jn = int(np.argmax(G2.extent_prior(cands, new, {}))) if ex else None
            jo = int(np.argmax(G2.extent_prior(cands, old, {}))) if ex else None
            G_ml = float(sum(tsz[j] for j in ids) * vml)
            A.append(dict(ds=ds, case=stem, site=site, id=int(i), kind=an.get('kind'), ml=float(a.sum() * vml), true=bool(ids),
                          in_frac=n_in / max(int(a.sum()), 1), tumours=json.dumps([int(j) for j in ids if j in big]), G_ml=G_ml,
                          xi_new=jn, xi_v84=jo, n_ext=len(ex)))
            for j in ids:
                if j in big:
                    touch[j].append(int(i))
            if ids:
                G = np.isin(labT, ids)
                for jj, e in enumerate(ex):
                    eb = e['box']; esl = tuple(slice(b[0], b[1]) for b in eb)
                    m = unpack(z, f'a{i}_e{jj}', eb)
                    tp = float((m & G[esl]).sum()); gs = float(G.sum()); ms = float(m.sum())
                    E.append(dict(ds=ds, case=stem, site=site, id=int(i), j=jj, name=e['name'], ml=e['ml'], logratio=e['logratio'],
                                  flags=json.dumps(e.get('flags', [])), dice=2 * tp / (ms + gs) if (ms + gs) else np.nan,
                                  precision=tp / ms if ms else np.nan, recall=tp / gs if gs else np.nan,
                                  is_new=jj == jn, is_v84=jj == jo))
    for j in sorted(big):
        U.append(dict(ds=ds, case=stem, site=site, tumour=int(j), ml=float(tsz[j] * vml), anchors=json.dumps(touch[j])))
    if not A:
        A = [dict(ds=ds, case=stem, site=site, note='no anchors')]
    return A, E, U


def tumours_only(ds, stem, A):
    """expert components >= MIN_ML of one scan, with the anchors touching them taken from its anchor rows"""
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
    T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour'])
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    labT, nT = ndimage.label(T, structure=EX.STRUCT26)
    tsz = np.bincount(labT.ravel(), minlength=nT + 1)
    touch = {}
    for r in A.itertuples(index=False):
        for j in (json.loads(r.tumours) if isinstance(r.tumours, str) and r.tumours.startswith('[') else []):
            touch.setdefault(int(j), []).append(int(float(r.id)))
    return [dict(ds=ds, case=stem, site=EX.SITE_ENTITY[ds], tumour=int(j), ml=float(tsz[j] * vml), anchors=json.dumps(touch.get(j, [])))
            for j in range(1, nT + 1) if tsz[j] * vml >= EX.MIN_ML]


def append(f, rows):
    if rows:
        pd.DataFrame(rows).to_csv(f, mode='a', index=False, header=not os.path.exists(f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shards', type=int, default=0); ap.add_argument('--only', default=None)
    ap.add_argument('--tumours-only', action='store_true')
    a = ap.parse_args(); O = out_dir()
    if a.shards:
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args, env=dict(os.environ, SHARD=f'{i}/{a.shards}'),
                               stdout=open(os.path.join(O, f'log_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        try:
            for p in ps:
                p.wait()
        except KeyboardInterrupt:
            for p in ps:
                p.terminate()
        print('done:', O); return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    meta = list(csv.DictReader(open(os.path.join(paths.root('outputs'), 'cohort', 'r2seg_scan_meta.csv'))))
    meta = [r for r in meta if r['img_name'].split('/')[0] in EX.SITE_ENTITY]
    if a.only:
        meta = [r for r in meta if any(s_ in r['img_name'] for s_ in a.only.split(','))]
    if a.tumours_only:
        At = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(O, 'anchor_truth*.csv'))], ignore_index=True)
        have = set()
        for f in glob.glob(os.path.join(O, 'tumour_truth*.csv')) + glob.glob(os.path.join(O, 'no_expert_tumour*.csv')):
            d0 = pd.read_csv(f, usecols=['ds', 'case'], dtype={'case': str}); have |= set(zip(d0.ds, d0.case))
        todo = sorted(set(zip(At.ds, At.case)) - have); t0 = time.time()
        for k, (ds, stem) in enumerate(todo):
            if k % SN != SI:
                continue
            try:
                U = tumours_only(ds, stem, At[(At.ds == ds) & (At.case == stem)])
            except Exception:
                print('error', ds, stem, traceback.format_exc()[-300:], flush=True); continue
            append(os.path.join(O, f'tumour_truth_fix{SUF}.csv'), U)
            if not U:                                              # scan without an expert tumour >= MIN_ML: recorded so reruns skip it
                append(os.path.join(O, f'no_expert_tumour{SUF}.csv'), [dict(ds=ds, case=stem, site=EX.SITE_ENTITY[ds])])
            print(f'{k} {ds}/{stem} {len(U)} tumours {time.time() - t0:.0f}s', flush=True)
        return
    done = set()
    for f in glob.glob(os.path.join(O, 'anchor_truth*.csv')):
        d0 = pd.read_csv(f, usecols=['ds', 'case']); done |= set(zip(d0.ds, d0.case.astype(str)))
    psis = {}; t0 = time.time(); n = 0
    for k, r in enumerate(meta):
        if k % SN != SI:
            continue
        ds, stem = r['img_name'].split('/', 1)
        if (ds, stem) in done:
            continue
        site = EX.SITE_ENTITY[ds]
        psis.setdefault(site, psi_pair(site))
        try:
            A, E, U = one(ds, stem, psis[site])
        except Exception:
            A, E, U = [dict(ds=ds, case=stem, site=site, note='error: ' + traceback.format_exc()[-300:].replace('\n', ' | '))], [], []
        append(os.path.join(O, f'extent_truth{SUF}.csv'), E); append(os.path.join(O, f'tumour_truth{SUF}.csv'), U)
        append(os.path.join(O, f'anchor_truth{SUF}.csv'), A)
        n += 1
        print(f'{n} {ds}/{stem} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
