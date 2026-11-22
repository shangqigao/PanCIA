"""How much would a lower two-agent agreement floor gain, and cost? (22 Nov 2026; claude/prostate_diagnosis_2026-11-22.md)
Stage 1 forms a two-agent anchor only from a BP ∩ VT component >= min_ml 0.5 ml (hand-set; audit A9). This check measures, without
changing stage 1, every BP ∩ VT component (26-connected) on R2Seg: VT = initial tumour mask OR every plan-call tumour mask
(<stem>_plan/*_tumor.nii.gz), BP = its tumour mask, both on the image grid. Per component: ml and whether it touches an expert tumour.
Per expert tumour (>= 0.5 ml, numbering as s1_eval_r2seg): the largest agreement component touching it.
Summary per site for candidate floors 0.1 / 0.2 / 0.3 / 0.4 ml, counting only components in [floor, 0.5):
  tumours added  = tumours not accepted in v86_r2seg_bp2 (diag/tumour_tiers.csv) whose largest agreement component is in the range;
  false comps    = components touching no tumour, per tumour scan;
  negatives      = tumour-free scans (cohort/r2seg_tumour_free.csv) with >= 1 such component, among those not already flagged
                   (no accepted false component in v86_r2seg_bp2/eval).
Upper bound: an added two-agent anchor can still be demoted by signals (flat, prompt_mismatch, no_site_agreement) or the host-far cap.
Uterus left out; prostate ADC series have no BP mask (skipped). Expert labels: evaluation only.
Outputs: ledger/v86_r2seg_bp2/diag/agree_floor/{comps,tumours}[_s<i>].csv, report.txt.
Usage: python s1_agree_floor_check.py --shards 4   (resumable)   |   python s1_agree_floor_check.py --summary"""
import os, sys, glob, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX, agents as AG   # noqa: E402
S26 = EX.STRUCT26; LEDGER = 'v86_r2seg_bp2'; FLOORS = (0.1, 0.2, 0.3, 0.4); MIN_ML = 0.5


def out_dir():
    d = os.path.join(paths.ledger_dir(), LEDGER, 'diag', 'agree_floor'); os.makedirs(d, exist_ok=True); return d


def append(f, rows):
    if rows:
        pd.DataFrame(rows).to_csv(f, mode='a', index=False, header=not os.path.exists(f))


def vt_union(ds, stem, img):
    V, _ = AG._on_grid(AG.mask_path('VoxTell', ds, stem), img)
    V = np.zeros(img.shape[:3], bool) if V is None else V.astype(bool)
    for f in glob.glob(os.path.join(paths.root('R2Seg_Seg'), 'VoxTell', ds, stem + '_plan', '*_tumor.nii.gz')):
        P, _ = AG._on_grid(f, img)
        if P is not None:
            V |= P.astype(bool)
    return V


def one(ds, stem):
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    B, _ = AG._on_grid(AG.mask_path('BiomedParse', ds, stem), img)
    if B is None:
        return None, None
    T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour'])
    A = B.astype(bool) & vt_union(ds, stem, img)
    la, _ = ndimage.label(A, structure=S26); lt, _ = ndimage.label(T, structure=S26)
    site = EX.SITE_ENTITY[ds]; crows = []; best = {}
    for k, sl in enumerate(ndimage.find_objects(la), 1):
        c = la[sl] == k; ml = c.sum() * vml; hit = set(np.unique(lt[sl][c]).tolist()) - {0}
        crows.append(dict(ds=ds, case=stem, site=site, comp=k, ml=ml, true=bool(hit)))
        for h in hit:
            best[h] = max(best.get(h, 0.0), ml)
    trows = []
    for k, sl in enumerate(ndimage.find_objects(lt), 1):
        v = (lt[sl] == k).sum() * vml
        if v >= 0.5:
            trows.append(dict(ds=ds, case=stem, site=site, tumour=k, ml=v, max_agree_ml=best.get(k, 0.0)))
    if not crows:
        crows = [dict(ds=ds, case=stem, site=site, comp=0, ml=0.0, true=False)]
    return crows, trows


def summary(O):
    rd = lambda p: pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(O, p))], ignore_index=True)
    c = rd('comps*.csv').drop_duplicates(['ds', 'case', 'comp']); t = rd('tumours*.csv').drop_duplicates(['ds', 'case', 'tumour'])
    L = os.path.join(paths.ledger_dir(), LEDGER)
    tt = pd.read_csv(os.path.join(L, 'diag', 'tumour_tiers.csv'), dtype={'case': str})[['ds', 'case', 'tumour', 'tier']]
    t = t.merge(tt, on=['ds', 'case', 'tumour'], how='left')
    tf = pd.read_csv(os.path.join(paths.root('outputs'), 'cohort', 'r2seg_tumour_free.csv'), dtype={'case': str})
    negk = set(zip(tf.ds[tf.status == 'negative'], tf.case[tf.status == 'negative']))
    s = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(L, 'eval', 'scans*.csv'))])
    s = s[s.method == 's1_acc']; s['fp_comps'] = pd.to_numeric(s.fp_comps, errors='coerce'); s['T_ml'] = pd.to_numeric(s.T_ml, errors='coerce')
    flagged = set(zip(s.ds[s.fp_comps > 0], s.case[s.fp_comps > 0]))
    tscans = set(zip(s.ds[s.T_ml > 0], s.case[s.T_ml > 0]))
    c['neg'] = [(a, b) in negk for a, b in zip(c.ds, c.case)]; c['tscan'] = [(a, b) in tscans for a, b in zip(c.ds, c.case)]
    n_ts = c[c.tscan].groupby('site').case.nunique()
    rep = [f'Two-agent agreement floor check on {LEDGER} inputs ({c.case.nunique()} scans, {len(t)} tumours >= 0.5 ml). '
           'Counts are upper bounds (signals and the host-far cap can still demote a two-agent anchor). Sites equal where averaged.', '',
           'Current floor 0.5 ml: tumours whose largest agreement component is >= 0.5 ml, by tier:',
           pd.crosstab(t.site, [t.max_agree_ml >= MIN_ML, t.tier]).to_string(), '']
    for F in FLOORS:
        rng = lambda x: (x >= F) & (x < MIN_ML)
        add = t[rng(t.max_agree_ml) & (t.tier != 'acc')]
        fp = c[rng(c.ml) & ~c.true]
        newneg = fp[fp.neg.values & np.array([(a, b) not in flagged for a, b in zip(fp.ds, fp.case)], bool)]
        R = pd.DataFrame({'tumours': t.groupby('site').size(), 'acc_now': t.groupby('site').tier.apply(lambda x: (x == 'acc').mean()),
                          'added': add.groupby('site').size(), 'added_from_unc': add[add.tier == 'unc_only'].groupby('site').size(),
                          'det_acc_new': (t.tier.eq('acc') | (rng(t.max_agree_ml))).groupby(t.site).mean(),
                          'false_per_tscan': fp[fp.tscan].groupby('site').size() / n_ts,
                          'true_share': (c[rng(c.ml)].groupby('site').true.mean())}).fillna(0)
        nn = pd.Series({ds: len({k for k in negk if k[0] == ds and k in set(zip(newneg.ds, newneg.case))}) for ds in sorted({k[0] for k in negk})})
        rep += [f'== floor {F} ml (components in [{F}, {MIN_ML})) ==', R.round(3).to_string(),
                f"(sites equal) accepted detection {R.acc_now.mean():.3f} -> {R.det_acc_new.mean():.3f}; added false components per tumour scan "
                f"{R.false_per_tscan.mean():.2f}; tumours added {int(R.added.sum())}",
                'tumour-free scans newly flagged (of those not flagged now): ' + ', '.join(f'{k} {v}' for k, v in nn.items()), '']
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shards', type=int, default=0); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--collections', default=None); a = ap.parse_args(); O = out_dir()
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
    for f in glob.glob(os.path.join(O, 'comps*.csv')):
        d0 = pd.read_csv(f, usecols=['ds', 'case'], dtype={'case': str}); done |= set(zip(d0.ds, d0.case))
    t0 = time.time(); n = 0
    for k, (ds, stem) in enumerate(todo):
        if k % SN != SI or (ds, stem) in done or stem.endswith('_adc'):
            continue
        try:
            crows, trows = one(ds, stem)
        except Exception:
            print('error', ds, stem, traceback.format_exc()[-300:], flush=True); continue
        if crows is None:
            continue
        append(os.path.join(O, f'tumours{SUF}.csv'), trows); append(os.path.join(O, f'comps{SUF}.csv'), crows); n += 1
        print(n, ds, stem, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
