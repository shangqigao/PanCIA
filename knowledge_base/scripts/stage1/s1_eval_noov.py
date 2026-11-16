"""v8.4 with the primary-not-detected override turned off (user, 13 Nov 2026): re-evaluates s1_acc on R2Seg.
Only scans with an 'accepted_override' anchor change; there the override anchors are dropped from the accepted set and the scan is
re-scored with s1_eval_r2seg.eval_scan (method s1_acc only). Every other scan keeps its s1_acc row from eval/scans*.csv.
Outputs: ledger/v84_r2seg/eval/noov/scans[_s<i>].csv, tumours[_s<i>].csv, report.txt.
Usage: python s1_eval_noov.py --shards 4   |   python s1_eval_noov.py --summary"""
import os, sys, glob, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_eval_r2seg as E   # noqa: E402

E.METHODS = ('s1_acc',)     # eval_scan loops over E.METHODS; the bit table is unchanged
M_ALL = ('vt', 'bp', 'agree', 's1_acc', 's1_acc_noov')


def out_dir():
    d = os.path.join(E.out_dir(), 'noov'); os.makedirs(d, exist_ok=True); return d


def summarise(O):
    B = E.out_dir()
    s = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(B, 'scans*.csv'))], ignore_index=True)
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(B, 'tumours*.csv'))], ignore_index=True)
    sn = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(O, 'scans*.csv'))], ignore_index=True)
    tn = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(O, 'tumours*.csv'))], ignore_index=True)
    for d in (s, t, sn, tn):
        d['case'] = d.case.astype(str)
    sn = sn.drop_duplicates(['ds', 'case', 'method'], keep='last'); tn = tn.drop_duplicates(['ds', 'case', 'tumour', 'method'], keep='last')
    key = set(zip(sn.ds, sn.case))
    keep = lambda d: d[[(a, c) not in key for a, c in zip(d.ds, d.case)]]
    s2 = pd.concat([keep(s[s.method == 's1_acc']), sn[sn.method == 's1_acc']]).assign(method='s1_acc_noov')
    t2 = pd.concat([keep(t[t.method == 's1_acc']), tn[tn.method == 's1_acc']]).assign(method='s1_acc_noov')
    s = pd.concat([s, s2]); t = pd.concat([t, t2])
    sw = lambda df, col: df.groupby(['site', 'method'])[col].mean().unstack('method')[list(M_ALL)]
    # tumour-free scans: cohort/r2seg_tumour_free.csv (14 Nov). FLARE23 scans without a lesion label are unannotated unless their
    # header matches NIH Pancreas-CT (healthy pancreas); unannotated scans are left out of every scan-level metric
    tf = pd.read_csv(os.path.join(E.paths.root('outputs'), 'cohort', 'r2seg_tumour_free.csv'), dtype={'case': str})
    st = {(a, c): x for a, c, x in zip(tf.ds, tf.case, tf.status)}
    s['tf'] = [st.get((a, c), '') for a, c in zip(s.ds, s.case)]
    unl = s.tf == 'unannotated'
    pos, neg = s[s.T_ml > 0], s[s.tf == 'negative']
    rep = [f'v8.4 accepted with the override off (s1_acc_noov): {len(key)} scans re-scored. Sites equal. Tumour-free = no expert tumour voxel, '
           f'unannotated FLARE23 scans excluded ({s[unl].case.nunique()}); negatives from cohort/r2seg_tumour_free.csv ({neg.case.nunique()}).', '']
    for title, df, col, r in (('per tumour: detection', t, 'detected', 3), ('per tumour: Dice', t, 'dice', 3),
                              ('scans with tumour: scan-level Dice', pos, 'dice', 3),
                              ('scans with tumour: false-positive components per scan', pos, 'fp_comps', 2),
                              ('tumour-free scans: false-positive components per scan', neg, 'fp_comps', 2)):
        x = sw(df, col)
        rep += [f'== {title} ==', x.round(r).to_string(), '(sites equal) ' + ', '.join(f'{m} {v:.{r}f}' for m, v in x.mean().items()), '']
    x = neg.assign(any_=neg.fp_comps > 0).groupby(['site', 'method']).any_.mean().unstack('method')[list(M_ALL)]
    rep += ['== tumour-free scans: share with any false positive ==', x.round(2).to_string(), '']
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shards', type=int, default=0); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args(); O = out_dir()
    if a.summary:
        summarise(O); return
    if a.shards:
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)], env=dict(os.environ, SHARD=f'{i}/{a.shards}'),
                               stdout=open(os.path.join(O, f'log_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        try:
            for p in ps:
                p.wait()
        except KeyboardInterrupt:
            for p in ps:
                p.terminate()
        summarise(O); return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    decs = E.load_decisions()
    todo = sorted(rel for rel, g in decs.items() if (g.decision == 'accepted_override').any() and rel.split('/')[1] in E.EX.SITE_ENTITY)
    done = set()
    for f in glob.glob(os.path.join(O, 'scans*.csv')):
        d0 = pd.read_csv(f, usecols=['ds', 'case']); done |= set(zip(d0.ds, d0.case.astype(str)))
    psis = {}; t0 = time.time(); n = 0
    for k, rel in enumerate(todo):
        if k % SN != SI:
            continue
        _, ds, stem = rel.split('/', 2)
        if (ds, stem) in done:
            continue
        site = E.EX.SITE_ENTITY[ds]
        psis.setdefault(site, E.psi_for(site))
        dec = decs[rel].copy(); dec.loc[dec.decision == 'accepted_override', 'decision'] = 'rejected'
        try:
            srows, trows = E.eval_scan(ds, stem, dec, psis[site])
        except Exception:
            srows = [dict(ds=ds, case=stem, site=site, method='error', note=traceback.format_exc()[-400:].replace('\n', ' | '))]; trows = []
        E.append(os.path.join(O, f'tumours{SUF}.csv'), trows); E.append(os.path.join(O, f'scans{SUF}.csv'), srows)
        n += 1
        print(f'{n}/{len(todo) // max(SN, 1)} {rel} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
