"""Why are some expert tumours reached only by rejected (or only by uncertain) stage-1 lesions? (16 Nov 2026). Expert labels: evaluation only.
Input: ledger/<ledger>/diag/tumour_tiers.csv (per expert tumour >= 0.5 ml: tier acc / unc_only / rej_only / none, from eval/tumours*.csv).
For each unc_only / rej_only tumour, every anchor whose selected extent (refitted psi, as in s1_eval_r2seg) touches it: kind, agents, ml,
decision, reason, MAP host, host_cls, host_far, and the MAP host's support level and demoting signals (ev[host]['sig']).
Output: ledger/<ledger>/diag/tier_anchors.csv. Usage: [SHARD=i/n] python s1_tier_diag.py [--ledger v86_r2seg]   (resumable)
21 Nov: tumour_tiers.csv is built here from eval/tumours*.csv (it was made ad hoc on 16 Nov) when missing or with --build-tiers:
tier acc (touched by s1_acc), unc_only (s1_acc_unc but not s1_acc), rej_only (s1_all only), none; agents = which agent masks touch it."""
import os, sys, json, glob, time, argparse, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s1_eval_r2seg as E   # noqa: E402


def build_tiers(L, D):
    t = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(L, 'eval', 'tumours*.csv'))])
    t = t.drop_duplicates(['ds', 'case', 'tumour', 'method'], keep='last')
    p = t.pivot_table(index=['ds', 'case', 'site', 'tumour', 'ml'], columns='method', values='detected', aggfunc='first').reset_index()
    for c in ('vt', 'bp', 'agree', 's1_all', 's1_acc', 's1_acc_unc'):
        p[c] = p[c].astype(str).isin(['True', '1', '1.0'])
    p['tier'] = 'none'
    p.loc[p.s1_all, 'tier'] = 'rej_only'; p.loc[p.s1_acc_unc, 'tier'] = 'unc_only'; p.loc[p.s1_acc, 'tier'] = 'acc'
    p['agents'] = ['both_agree' if g else 'vt+bp_sep' if (v and b) else 'vt_only' if v else 'bp_only' if b else 'neither'
                   for g, v, b in zip(p.agree, p.vt, p.bp)]
    os.makedirs(D, exist_ok=True); p.to_csv(os.path.join(D, 'tumour_tiers.csv'), index=False)
    print(pd.crosstab(p.tier, p.agents))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ledger', default='v86_r2seg'); ap.add_argument('--build-tiers', action='store_true')
    ap.add_argument('--collections', default=None, help='comma list of R2Seg collections (22 Nov)')
    a = ap.parse_args()
    E.LEDGER[0] = a.ledger
    L = os.path.join(E.paths.ledger_dir(), a.ledger); D = os.path.join(L, 'diag')
    if a.build_tiers or not os.path.exists(os.path.join(D, 'tumour_tiers.csv')):
        build_tiers(L, D)
        if a.build_tiers:
            return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    f_out = os.path.join(D, f'tier_anchors{SUF}.csv'); f_done = os.path.join(D, f'tier_done{SUF}.txt')
    tt = pd.read_csv(os.path.join(D, 'tumour_tiers.csv'), dtype={'case': str}); tt = tt[tt.tier.isin(['unc_only', 'rej_only'])]
    if a.collections:
        tt = tt[tt.ds.isin(a.collections.split(','))]
    an = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(L, 'anchors*.csv')))]).drop_duplicates(['rel', 'id'], keep='last')
    an['ds'] = an.rel.str.split('/').str[1]; an['case'] = an.rel.str.split('/', n=2).str[2]
    A = {k: g.set_index('id') for k, g in an.groupby(['ds', 'case'])}
    done = set(open(f_done).read().split()) if os.path.exists(f_done) else set()
    psis = {}; t0 = time.time()
    for k, ((ds, case), g) in enumerate(tt.groupby(['ds', 'case'])):
        if k % SN != SI or f'{ds}/{case}' in done:
            continue
        rows = []
        try:
            site = E.EX.SITE_ENTITY[ds]; psis.setdefault(site, E.psi_for(site)); new = psis[site][1]
            lf = os.path.join(E.paths.root('R2Seg'), ds, 'labels', case + '_seg.nii.gz')
            T = np.isin(E.EX._read_labels(lf), E.EX.APPEARANCE[ds]['tumour'])
            labT, _ = E.comps(T)
            z = np.load(os.path.join(L, 'extents', ds, case + '.npz')); meta = json.loads(str(z['meta']))
            ga = A.get((ds, case))
            for i, m in meta['anchors'].items():
                ex = m['extents']
                if not ex or ga is None or int(i) not in ga.index:
                    continue
                cands = [dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ex]
                j = int(np.argmax(E.G2.extent_prior(cands, new, {})))
                box = ex[j]['box']; sl = tuple(slice(b[0], b[1]) for b in box)
                x = E.unpack(z, f'a{i}_e{j}', box); hit = set(np.unique(labT[sl][x]).tolist()) - {0}
                tum = set(g.tumour.astype(int)) & hit
                if not tum:
                    continue
                r = ga.loc[int(i)]; ev = json.loads(r['ev']) if isinstance(r['ev'], str) else {}
                h = r['host_map'] if isinstance(r['host_map'], str) else None; e = ev.get(h) or {}
                for tu in sorted(tum):
                    tr = g[g.tumour == tu].iloc[0]
                    rows.append(dict(ds=ds, case=case, site=site, tumour=tu, tumour_ml=tr.ml, tier=tr.tier, tumour_agents=tr.agents, id=int(i),
                                     kind=r['kind'], agents=r['agents'], anchor_ml=r['ml'], decision=r['decision'], reason=r['reason'],
                                     host=h, host_cls=r['host_cls'], host_far=r['host_far'], S=r['S'], level=e.get('level'),
                                     sig='|'.join(e.get('sig') or []), test=(e.get('test') or {}).get('0.05'), tree=e.get('tree')))
        except Exception:
            print('error', ds, case, traceback.format_exc()[-300:], flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(f_out, mode='a', index=False, header=not os.path.exists(f_out))
        open(f_done, 'a').write(f'{ds}/{case}\n')
        print(k, ds, case, len(rows), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
