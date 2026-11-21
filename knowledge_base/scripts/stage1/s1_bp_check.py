"""Acceptance check of retrained BiomedParse masks on R2Seg (19 Nov 2026; claude/kidney_agent_cyst_check_2026-11-16.md).
Per scan, the agent's tumour mask on the image grid (stage2.agents) against the expert labels, with the definitions of
s1_eval_r2seg.py so the numbers compare with the old 'bp' rows of ledger/v86_r2seg/eval: per expert tumour >= 0.5 ml (26-connected)
detected = any overlap, recall; per scan: false-positive components (agent components touching no expert tumour), FP ml, Dice.
Components on a label named 'cyst' (kidney, uterus nabothian) are counted separately. Tumour-free scans: cohort/r2seg_tumour_free.csv
status 'negative'. Uterus (benign) left out. Expert labels: evaluation only.
Outputs: ledger/v86_r2seg/diag/bp_check_<tag>/scans[_s<i>].csv, tumours[_s<i>].csv; --summary writes report.txt (new vs old, sites equal).
Usage: [SHARD=i/n] python s1_bp_check.py [--agent BiomedParse] [--tag bp2] [--collections Kidney_Tumor_00,...]   (resumable)
       python s1_bp_check.py --shards 4      |   python s1_bp_check.py --summary"""
import os, sys, glob, json, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX, agents as AG   # noqa: E402
S26 = EX.STRUCT26


def append(f, rows):
    if rows:
        pd.DataFrame(rows).to_csv(f, mode='a', index=False, header=not os.path.exists(f))


def one(ds, stem, agent):
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
    L = EX._read_labels(lf); img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    labs = json.load(open(os.path.join(paths.root('R2Seg'), ds, 'dataset.json')))['labels']
    cyst = [int(k) for k, v in labs.items() if 'cyst' in v.lower()]
    T = np.isin(L, EX.APPEARANCE[ds]['tumour']); C = np.isin(L, cyst) if cyst else None
    A, _ = AG._on_grid(AG.mask_path(agent, ds, stem), img)
    A = np.zeros(T.shape, bool) if A is None else A.astype(bool)
    site = EX.SITE_ENTITY[ds]
    la, na = ndimage.label(A, structure=S26)
    fp = fp_ml = cy = 0
    for k, sl in enumerate(ndimage.find_objects(la), 1):
        c = la[sl] == k
        if not T[sl][c].any():
            fp += 1; fp_ml += c.sum() * vml
            if C is not None and C[sl][c].any():
                cy += 1
    tp = (A & T).sum(); dice = 2 * tp / (A.sum() + T.sum()) if (A.sum() + T.sum()) else np.nan
    srow = dict(ds=ds, case=stem, site=site, T_ml=T.sum() * vml, pred_ml=A.sum() * vml, dice=dice, fp_comps=fp, fp_ml=fp_ml,
                cyst_comps=cy, scan_has_cyst=bool(C is not None and C.any()))
    trows = []
    lt, nt = ndimage.label(T, structure=S26)
    for k, sl in enumerate(ndimage.find_objects(lt), 1):
        c = lt[sl] == k; v = c.sum() * vml
        if v < 0.5:
            continue
        hit = A[sl][c]
        trows.append(dict(ds=ds, case=stem, site=site, tumour=k, ml=v, detected=bool(hit.any()), recall=float(hit.mean())))
    return [srow], trows


def summary(O):
    rd = lambda d, p: pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(d, p))], ignore_index=True)
    s, t = rd(O, 'scans*.csv').drop_duplicates(['ds', 'case']), rd(O, 'tumours*.csv').drop_duplicates(['ds', 'case', 'tumour'])
    E = os.path.join(paths.ledger_dir(), 'v86_r2seg', 'eval')
    so = rd(E, 'scans*.csv'); so = so[so.method == 'bp'].drop_duplicates(['ds', 'case'])
    to = rd(E, 'tumours*.csv'); to = to[to.method == 'bp'].drop_duplicates(['ds', 'case', 'tumour'])
    tf = pd.read_csv(os.path.join(paths.root('outputs'), 'cohort', 'r2seg_tumour_free.csv'), dtype={'case': str})
    negk = set(zip(tf.ds[tf.status == 'negative'], tf.case[tf.status == 'negative']))
    isneg = lambda d: np.array([(a, c) in negk for a, c in zip(d.ds, d.case)])
    keys = set(zip(s.ds, s.case)); so = so[[k in keys for k in zip(so.ds, so.case)]]; to = to[[k in keys for k in zip(to.ds, to.case)]]
    so = so[pd.to_numeric(so.T_ml, errors='coerce').notna()]
    to['detected'] = to.detected.astype(str).isin(['True', '1', '1.0'])
    for d in (so, to):
        for c in ('dice', 'fp_comps', 'recall', 'T_ml'):
            if c in d:
                d[c] = pd.to_numeric(d[c], errors='coerce')
    R = pd.DataFrame({'tumours': t.groupby('site').size(),
                      'det_old': to.groupby('site').detected.mean(), 'det_new': t.groupby('site').detected.mean(),
                      'recall_old': to.groupby('site').recall.mean(), 'recall_new': t.groupby('site').recall.mean(),
                      'scanDice_old': so[(so.T_ml > 0)].groupby('site').dice.mean(), 'scanDice_new': s[s.T_ml > 0].groupby('site').dice.mean(),
                      'FP_old': so[so.T_ml > 0].groupby('site').fp_comps.mean(), 'FP_new': s[s.T_ml > 0].groupby('site').fp_comps.mean()})
    sn, son = s[isneg(s)], so[isneg(so)]
    N = pd.DataFrame({'neg_scans': sn.groupby('site').size(), 'neg_anyFP_old': son.groupby('site').fp_comps.apply(lambda x: (x > 0).mean()),
                      'neg_anyFP_new': sn.groupby('site').fp_comps.apply(lambda x: (x > 0).mean()),
                      'neg_FP_old': son.groupby('site').fp_comps.mean(), 'neg_FP_new': sn.groupby('site').fp_comps.mean()})
    k = s[s.scan_has_cyst]
    rep = [f'Retrained agent check ({os.path.basename(O)}) vs the masks used in v86_r2seg/eval (bp). {s.case.nunique()} scans. Sites equal.', '',
           '== tumour and scan level (scans with tumour) ==', R.round(3).to_string(),
           '(sites equal) ' + ', '.join(f'{c} {v:.3f}' for c, v in R.drop(columns='tumours').mean().items()), '',
           '== tumour-free scans (140 negatives) ==', N.round(3).to_string(), '']
    if len(k):
        rep += ['== scans with an expert cyst ==',
                f'share with a cyst-only false component: {(k.cyst_comps > 0).mean():.3f} (old masks: 0.955 on kidney); scans {len(k)}', '']
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--agent', default='BiomedParse'); ap.add_argument('--tag', default='bp2')
    ap.add_argument('--collections', default=None); ap.add_argument('--shards', type=int, default=0); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args(); O = os.path.join(paths.ledger_dir(), 'v86_r2seg', 'diag', f'bp_check_{a.tag}'); os.makedirs(O, exist_ok=True)
    if a.summary:
        summary(O); return
    if a.shards:
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args, env=dict(os.environ, SHARD=f'{i}/{a.shards}'),
                               stdout=open(os.path.join(O, f'log_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        for p in ps:
            p.wait()
        summary(O); return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    cols = a.collections.split(',') if a.collections else sorted(d for d in os.listdir(paths.root('R2Seg')) if d in EX.SITE_ENTITY and d != 'Uterus_Tumor_00')
    todo = [(ds, os.path.basename(f)[:-len('_seg.nii.gz')]) for ds in cols for f in sorted(glob.glob(os.path.join(paths.root('R2Seg'), ds, 'labels', '*_seg.nii.gz')))]
    done = set()
    for f in glob.glob(os.path.join(O, 'scans*.csv')):
        d0 = pd.read_csv(f, usecols=['ds', 'case'], dtype={'case': str}); done |= set(zip(d0.ds, d0.case))
    t0 = time.time(); n = 0
    for k, (ds, stem) in enumerate(todo):
        if k % SN != SI or (ds, stem) in done:
            continue
        try:
            srows, trows = one(ds, stem, a.agent)
        except Exception:
            print('error', ds, stem, traceback.format_exc()[-300:], flush=True); continue
        append(os.path.join(O, f'tumours{SUF}.csv'), trows); append(os.path.join(O, f'scans{SUF}.csv'), srows); n += 1
        print(n, ds, stem, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
