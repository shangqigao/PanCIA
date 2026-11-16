"""Evaluate a re-decided stage-1 variant on R2Seg at voxel level (14 Nov 2026), e.g. v8.5 (s1_v85_decide.py).
Reads a decisions file (rel, id, decision_<name>); only scans whose accepted set differs from v8.4 accepted (accepted +
accepted_override) are re-scored with s1_eval_r2seg.eval_scan (method s1_acc only); every other scan reuses its s1_acc row.
The summary sets the variant beside vt, bp, agree, v8.4 accepted, v8.4 with the override off (eval/noov) and itself. Tumour-free
scans: cohort/r2seg_tumour_free.csv status 'negative' (140); unannotated FLARE23 scans are left out. Sites equal.
Outputs: ledger/v84_r2seg/eval/<name>/scans[_s<i>].csv, tumours[_s<i>].csv, report.txt.
Usage: python s1_eval_variant.py --name v85 --shards 4   |   python s1_eval_variant.py --name v85 --summary"""
import os, sys, glob, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_eval_r2seg as E   # noqa: E402

E.METHODS = ('s1_acc',)     # eval_scan loops over E.METHODS; the bit table is unchanged
ACC = ('accepted', 'accepted_override')


def out_dir(name):
    d = os.path.join(E.out_dir(), name); os.makedirs(d, exist_ok=True); return d


def load_variant(name):
    f = os.path.join(E.paths.ledger_dir(), 'v84_r2seg', f'decisions_{name}.csv')
    v = pd.read_csv(f, usecols=['rel', 'id', f'decision_{name}'])
    return {rel: g.set_index('id')[f'decision_{name}'] for rel, g in v.groupby('rel')}


def changed(decs, var):
    todo = []
    for rel, g in decs.items():
        if rel.split('/')[1] not in E.EX.SITE_ENTITY or rel not in var:
            continue
        a0 = set(g.index[g.decision.isin(ACC)]); v = var[rel]; a1 = set(v.index[v == 'accepted'])
        if a0 != a1:
            todo.append(rel)
    return sorted(todo)


def summarise(O, name):
    B = E.out_dir()
    rd = lambda pat: pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(pat)], ignore_index=True)
    s, t = rd(os.path.join(B, 'scans*.csv')), rd(os.path.join(B, 'tumours*.csv'))
    cols = ['vt', 'bp', 'agree', 's1_acc']
    for tag, D in (('s1_acc_noov', os.path.join(B, 'noov')), (f's1_acc_{name}', O)):
        if not glob.glob(os.path.join(D, 'scans*.csv')):
            continue
        sn = rd(os.path.join(D, 'scans*.csv')).drop_duplicates(['ds', 'case', 'method'], keep='last')
        tn = rd(os.path.join(D, 'tumours*.csv')).drop_duplicates(['ds', 'case', 'tumour', 'method'], keep='last')
        key = set(zip(sn.ds, sn.case))
        keep = lambda d: d[[(a, c) not in key for a, c in zip(d.ds, d.case)]]
        s = pd.concat([s, pd.concat([keep(s[s.method == 's1_acc']), sn[sn.method == 's1_acc']]).assign(method=tag)])
        t = pd.concat([t, pd.concat([keep(t[t.method == 's1_acc']), tn[tn.method == 's1_acc']]).assign(method=tag)])
        cols.append(tag)
    tf = pd.read_csv(os.path.join(E.paths.root('outputs'), 'cohort', 'r2seg_tumour_free.csv'), dtype={'case': str})
    negk = set(zip(tf.ds[tf.status == 'negative'], tf.case[tf.status == 'negative']))
    pos = s[s.T_ml > 0]; neg = s[[(a, c) in negk for a, c in zip(s.ds, s.case)]]
    sw = lambda df, col: df.groupby(['site', 'method'])[col].mean().unstack('method')[cols]
    rep = [f'Stage-1 variant {name} on R2Seg (voxel level; sites equal). Tumour-free: {neg.case.nunique()} negatives from '
           'cohort/r2seg_tumour_free.csv; unannotated FLARE23 scans left out.', '']
    for title, df, col, r in (('per tumour: detection', t, 'detected', 3), ('per tumour: Dice', t, 'dice', 3),
                              ('scans with tumour: scan-level Dice', pos, 'dice', 3),
                              ('scans with tumour: false-positive components per scan', pos, 'fp_comps', 2),
                              ('tumour-free scans: false-positive components per scan', neg, 'fp_comps', 2)):
        x = sw(df, col)
        rep += [f'== {title} ==', x.round(r).to_string(), '(sites equal) ' + ', '.join(f'{m} {v:.{r}f}' for m, v in x.mean().items()), '']
    x = neg.assign(any_=neg.fp_comps > 0).groupby(['site', 'method']).any_.mean().unstack('method')[cols]
    rep += ['== tumour-free scans: share with any false positive ==', x.round(2).to_string(), '']
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', default='v85'); ap.add_argument('--shards', type=int, default=0)
    ap.add_argument('--summary', action='store_true'); a = ap.parse_args(); O = out_dir(a.name)
    if a.summary:
        summarise(O, a.name); return
    if a.shards:
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__), '--name', a.name],
                               env=dict(os.environ, SHARD=f'{i}/{a.shards}'),
                               stdout=open(os.path.join(O, f'log_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        try:
            for p in ps:
                p.wait()
        except KeyboardInterrupt:
            for p in ps:
                p.terminate()
        summarise(O, a.name); return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    decs = E.load_decisions(); var = load_variant(a.name); todo = changed(decs, var)
    done = set()
    for f in glob.glob(os.path.join(O, 'scans*.csv')):
        d0 = pd.read_csv(f, usecols=['ds', 'case'], dtype={'case': str}); done |= set(zip(d0.ds, d0.case))
    psis = {}; t0 = time.time(); n = 0
    for k, rel in enumerate(todo):
        if k % SN != SI:
            continue
        _, ds, stem = rel.split('/', 2)
        if (ds, stem) in done:
            continue
        site = E.EX.SITE_ENTITY[ds]
        psis.setdefault(site, E.psi_for(site))
        dec = decs[rel].copy()
        v = var[rel].reindex(dec.index)
        dec['decision'] = v.fillna('rejected').values
        try:
            srows, trows = E.eval_scan(ds, stem, dec, psis[site])
        except Exception:
            srows = [dict(ds=ds, case=stem, site=site, method='error', note=traceback.format_exc()[-400:].replace('\n', ' | '))]; trows = []
        E.append(os.path.join(O, f'tumours{SUF}.csv'), trows); E.append(os.path.join(O, f'scans{SUF}.csv'), srows)
        n += 1
        print(f'{n} ({len(todo)} changed scans in total) {rel} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
