"""Stage-1 decision, part 1b: elongation likelihood ratio, version e2 (28 Nov 2026; user-approved plan
claude/elongation_and_pelvic_fix_plan_2026-11-28.md, A1-A3 'shape only'; workflow claude/stage1_workflow.md step 6g). DESIGN SET ONLY.
Pan-cancer: no cancer type, no sex, no stage / outcome, no agreement. Replaces the binned estimator of 28 Nov morning (it lost
information across sites; claude/stage1_overnight_design_results_2026-11-28.md).

Evidence: e = log10 elongation, e2 definition (covariance of the voxelised solid, s1_augment_candidates.feats; no cap).
Shape only (user, 28 Nov): candidate SIZE is not evidence (the small-candidate excess of false candidates may be an agent property,
e.g. BiomedParse, not a tumour property). The logLR is conditioned on the stratum = size band (< 1, 1-10, > 10 ml) x slice-spacing
class (thin <= 2.5, mid <= 5, thick mm); a stratum with < 30 training rows on either side falls back to its band, a band to 'all'.
Model: logistic regression, one intercept per stratum + ONE slope on e (monotone by construction), L2 1e-2. Weights: within each
stratum the two classes have equal total weight (so the stratum intercept absorbs the stratum's class balance and the logit is the
log likelihood ratio of e given the stratum), and within a class the sites have equal total weight (sites count equally).
Fit rows: real design-set candidates (in scope, T / F; three modes pooled) + augmented candidates of the sides that passed realism.
logLR clipped to +-4 (as approved 28 Nov). Leave-one-site-out values for every real candidate and for the tumour-free design scans
(evaluation reference, never fitted); full fit saved.
Acceptance (pre-specified in the plan): 1 every LOSO slope < 0; 2 mean AUC within site x stratum >= raw elongation - 0.01 per mode;
3 pooled calibration: empirical log ratio non-decreasing across logLR bins.
Output: ledger/decision_s1/<outdir>[/<tag>]/{candidates.csv, fit_full.json, eval.txt, eval_auc.csv, eval_calib.csv}.
Usage: python s1_fit_elong_lr.py --augdir augment_e2 --outdir elong_lr_e2 --design design_g3 [--tag smoke]"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

L = paths.ledger_dir()
MIN_N, CLIP, LAM = 30, 4.0, 1e-2
BANDS, BL = [0, 1, 10, 1e9], ['<1', '1-10', '>10']


def sp_class(m):
    return 'thin' if m <= 2.5 else ('mid' if m <= 5.0 else 'thick')


def auc(y, s, nmin=1):
    y = np.asarray(y); s = np.asarray(s, float); n1 = y.sum(); n0 = len(y) - n1
    if n1 < nmin or n0 < nmin: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prep(d):
    d = d[d.note.isna() | (d.note == '')].copy(); d['case'] = d.case.astype(str)
    d['lel'] = np.log10(d.elong.astype(float).clip(lower=1))
    d['band'] = pd.cut(d.ml.astype(float), BANDS, labels=BL, right=False).astype(str)
    d['spc'] = d.maxsp.astype(float).map(sp_class)
    d['st'] = d.band + '|' + d.spc
    d['in_scope'] = d.in_scope.astype(str) == 'True'
    return d


def levels(tr):
    """stratum -> level key used (stratum, band or 'all'), from training counts per side"""
    cnt = tr.groupby(['st', 'side']).size().unstack(fill_value=0).reindex(columns=['T', 'F'], fill_value=0)
    bc = tr.groupby(['band', 'side']).size().unstack(fill_value=0).reindex(columns=['T', 'F'], fill_value=0)
    out = {}
    for b in BL:
        for c in ('thin', 'mid', 'thick'):
            k = f'{b}|{c}'
            if k in cnt.index and cnt.loc[k].min() >= MIN_N: out[k] = k
            elif b in bc.index and bc.loc[b].min() >= MIN_N: out[k] = b
            else: out[k] = 'all'
    return out


def weights(tr, lv):
    w = np.zeros(len(tr)); y = tr.side.to_numpy(); s = tr.site.to_numpy(); g = tr.lvl.to_numpy()
    for k in np.unique(g):
        if len(np.unique(y[g == k])) < 2: continue
        for c in ('T', 'F'):
            m = (g == k) & (y == c); ss = np.unique(s[m])
            for q in ss:
                sel = m & (s == q); w[sel] = 1.0 / (sel.sum() * len(ss))
    return w * len(tr) / max(w.sum(), 1e-12)


def design(d, keys):
    X = np.zeros((len(d), len(keys) + 1)); idx = {k: i for i, k in enumerate(keys)}
    for i, k in enumerate(d.lvl): X[i, idx[k]] = 1.0
    X[:, -1] = d.lel.to_numpy(); return X


def fit(tr):
    lv = levels(tr); tr = tr.assign(lvl=tr.st.map(lv)); keys = sorted(tr.lvl.unique())
    X = design(tr, keys); y = (tr.side == 'T').to_numpy(float); w = weights(tr, lv); W = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ W, -30, 30))); H = X.T @ (X * (w * p * (1 - p))[:, None]) + LAM * np.eye(len(W))
        step = np.linalg.solve(H, X.T @ (w * (p - y)) + LAM * W); W -= step
        if np.abs(step).max() < 1e-8: break
    return dict(levels=lv, keys=keys, intercepts={k: float(W[i]) for i, k in enumerate(keys)}, slope=float(W[-1]),
                n={k: dict(T=int(((tr.lvl == k) & (tr.side == 'T')).sum()), F=int(((tr.lvl == k) & (tr.side == 'F')).sum())) for k in keys})


def apply(m, d):
    ic = m['intercepts']; mean_ic = float(np.mean(list(ic.values())))
    def key(st):                                  # level from the training counts; if that level had no training rows: band, all, mean
        k = m['levels'].get(st, 'all')
        for c in (k, st.split('|')[0], 'all'):
            if c in ic: return c
        return None
    lv = d.st.map(key)
    a = np.array([ic[k] if k is not None else mean_ic for k in lv])
    return np.clip(a + m['slope'] * d.lel.to_numpy(), -CLIP, CLIP), lv.fillna('mean').to_numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default=''); ap.add_argument('--augdir', default='augment_e2')
    ap.add_argument('--outdir', default='elong_lr_e2'); ap.add_argument('--design', default='design_g3'); a = ap.parse_args()
    AUGD = os.path.join(L, 'decision_s1', a.augdir)
    ad = os.path.join(AUGD, a.tag) if a.tag else AUGD
    od = os.path.join(L, 'decision_s1', a.outdir, a.tag) if a.tag else os.path.join(L, 'decision_s1', a.outdir); os.makedirs(od, exist_ok=True)
    re_ = prep(pd.read_csv(os.path.join(AUGD, 'real.csv'), low_memory=False))
    ag = prep(pd.read_csv(os.path.join(ad, 'aug.csv'), low_memory=False))
    tfp = os.path.join(ad, 'real_tf.csv')
    tf = prep(pd.read_csv(tfp, low_memory=False)) if os.path.exists(tfp) else re_.iloc[:0].copy()
    rz = pd.read_csv(os.path.join(ad, 'realism.csv'))
    use_aug = {}
    for side, s in (('T', 'tumour'), ('F', 'non-tumour')):
        r = rz[(rz.side == s) & (rz.band != 'all') & rz.passed.notna()]
        use_aug[side] = bool(len(r)) and bool(r.passed.astype(str).eq('True').all())
    if a.tag:                                     # tagged (smoke) runs: real restricted to the augmented scans, like with like
        sc = set(zip(ag.ds, ag.case)); re_ = re_[[(u, v) in sc for u, v in zip(re_.ds, re_.case)]]
    des = pd.read_csv(os.path.join(L, 'graph_v3', a.design + '.csv'), usecols=['ds', 'case', 'site'], low_memory=False).drop_duplicates(['ds', 'case'])
    des['case'] = des.case.astype(str); site = {(u, v): s for u, v, s in des.itertuples(index=False)}
    for x in (re_, ag, tf): x['site'] = [site.get((u, v), '?') for u, v in zip(x.ds, x.case)]
    fr = re_[re_.in_scope & re_.lab.isin(['T', 'F'])].assign(side=lambda x: x.lab)
    fa = ag[ag.lab.isin(['T', 'F'])].assign(side=lambda x: x.lab); fa = fa[[use_aug[s] for s in fa.side]]
    fd = pd.concat([fr, fa], ignore_index=True)
    log = [f'fit rows: real T {(fr.side == "T").sum()} F {(fr.side == "F").sum()} | augmented used: tumour side {use_aug["T"]}, '
           f'non-tumour side {use_aug["F"]} -> aug T {(fa.side == "T").sum()} F {(fa.side == "F").sum()} | sites {sorted(fd.site.unique())}']
    full = fit(fd); json.dump(full, open(os.path.join(od, 'fit_full.json'), 'w'), indent=1)
    log.append(f'full fit: slope {full["slope"]:.3f}; intercepts ' + ', '.join(f'{k} {v:.2f} (T {full["n"][k]["T"]}, F {full["n"][k]["F"]})'
                                                                            for k, v in full['intercepts'].items()))
    cand = pd.concat([re_.assign(tumour_free=False), tf.assign(tumour_free=True)], ignore_index=True)
    cand['llr'] = np.nan; cand['level'] = ''; slopes = {}
    for s in sorted(cand.site.unique()):
        m_ = fit(fd[fd.site != s]); slopes[s] = round(m_['slope'], 3)
        sel = cand.site == s
        if sel.sum():
            v, lv = apply(m_, cand[sel]); cand.loc[sel, 'llr'] = v; cand.loc[sel, 'level'] = lv
    cand['llr_full'] = apply(full, cand)[0]
    cand[['ds', 'case', 'site', 'tumour_free', 'mode', 'cid', 'lab', 'in_scope', 'category', 'ml', 'elong', 'lel', 'band', 'spc', 'llr',
          'llr_full', 'level']].to_csv(os.path.join(od, 'candidates.csv'), index=False)
    ev = cand[~cand.tumour_free & cand.in_scope & cand.lab.isin(['T', 'F'])].copy(); ev['y'] = (ev.lab == 'T').astype(int)
    ev['st'] = ev.band + '|' + ev.spc
    rows = []
    for mode, g in ev.groupby('mode'):
        cell = lambda col: np.nanmean([auc(h.y, h[col], 5) for _, h in g.groupby(['site', 'st'])])
        ev_raw = g.assign(raw=-g.lel)
        rows.append(dict(mode=mode, by='site x stratum (mean)', key='', nT=int(g.y.sum()), nF=int((1 - g.y).sum()),
                         auc_llr=round(cell('llr'), 3), auc_raw=round(np.nanmean([auc(h.y, -h.lel, 5) for _, h in g.groupby(['site', 'st'])]), 3)))
        for kind, keys in (('mode', None), ('site', 'site'), ('band', 'band')):
            parts = [('all', g)] if keys is None else list(g.groupby(keys))
            for k, h in parts:
                rows.append(dict(mode=mode, by=kind, key=k, nT=int(h.y.sum()), nF=int((1 - h.y).sum()),
                                 auc_llr=round(auc(h.y, h.llr), 3), auc_raw=round(auc(h.y, -h.lel), 3)))
    A = pd.DataFrame(rows); A.to_csv(os.path.join(od, 'eval_auc.csv'), index=False)
    cb = [-4.01, -2, -1, -0.5, 0, 0.5, 1, 2, 4.01]; crow = []
    for mode, g in list(ev.groupby('mode')) + [('pooled', ev)]:
        nT, nF = g.y.sum(), (1 - g.y).sum()
        for iv, h in g.groupby(pd.cut(g.llr, cb), observed=True):
            t, f = h.y.sum(), (1 - h.y).sum()
            crow.append(dict(mode=mode, bin=str(iv), n=len(h), nT=int(t), nF=int(f), pred=round(h.llr.mean(), 2),
                             empirical=round(float(np.log(((t + .5) / (nT + .5)) / ((f + .5) / (nF + .5)))), 2)))
    C = pd.DataFrame(crow); C.to_csv(os.path.join(od, 'eval_calib.csv'), index=False)
    cell = A[A.by == 'site x stratum (mean)']
    a1 = all(v < 0 for v in slopes.values())
    ct = cell[cell.auc_llr.notna() & cell.auc_raw.notna()]
    a2 = None if len(ct) < 3 else bool((ct.auc_llr >= ct.auc_raw - 0.01).all())
    pe = C[C['mode'] == 'pooled'].empirical.to_numpy(); a3 = bool(np.all(np.diff(pe) >= 0))
    log += [f'\nLOSO slopes by held-out site: {slopes}',
            f'ACCEPTANCE 1 (every LOSO slope < 0): {"PASS" if a1 else "FAIL"}',
            f'ACCEPTANCE 2 (mean AUC within site x stratum >= raw - 0.01, per mode): {"not testable" if a2 is None else ("PASS" if a2 else "FAIL")}',
            f'ACCEPTANCE 3 (pooled calibration non-decreasing): {"PASS" if a3 else "FAIL"}  empirical {pe.tolist()}',
            '\nAUC (in scope, T vs F; LOSO logLR vs raw elongation):', A[A.by.isin(['site x stratum (mean)', 'mode'])].to_string(index=False),
            '\nby band:', A[A.by == 'band'].to_string(index=False),
            '\nby site:', A[A.by == 'site'].to_string(index=False),
            '\ncalibration (LOSO logLR bins; empirical = log of class-conditional bin frequencies):', C.to_string(index=False),
            f'\nlevels used (real candidates): {cand.level.value_counts().to_dict()}',
            f'tumour-free design candidates scored: {int(cand.tumour_free.sum())} (evaluation reference, never fitted)']
    txt = '\n'.join(log); print(txt); open(os.path.join(od, 'eval.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
