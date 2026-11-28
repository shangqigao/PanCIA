"""Stage-1 decision, part 1b: elongation likelihood ratio (28 Nov 2026; plan claude/stage1_decision_plan_2026-11-28.md; workflow
claude/stage1_workflow.md step 6g). DESIGN SET ONLY. Pan-cancer: no cancer type, no sex, no stage / outcome, no agreement.

Evidence: e = log10 elongation (sqrt of largest / smallest covariance eigenvalue, mm), the feature with the best separation inside the
v3 unresolved buckets (claude/v3_remaining_false_candidates_elongation_2026-11-28.md). Output: logLR(e) = log p(e | T) / p(e | F).
Truth from expert labels (strict rule), feature values from agent outlines and augmented candidates (option D, user-approved).

Fit rows: real design-set tumour-scan candidates (augment/real.csv: in scope, lab T or F; P excluded; three modes pooled) and augmented
candidates (augment/[<tag>/]aug.csv; lab T or F), the augmented rows of a side used only if that side PASSED the realism check
(realism.csv); augmented weights scaled so that, per side, augmented weight total = real count of that side.
Strata: size band (< 1, 1-10, > 10 ml) x slice-spacing class (thin <= 2.5, mid <= 5, thick mm); a stratum with < 30 rows on either side
falls back to its size band, a band with < 30 on either side to all rows.
Estimator per stratum: K quantile bins of e (K = min(T, F rows) // 20 clipped to 3..10), weighted counts per side + 0.5 (additive
smoothing), logLR per bin, pool-adjacent-violators so logLR is non-increasing in elongation (weights = bin weight), clipped to +-4.
Leave-one-site-out: for each site, tables fitted without that site's scans (real and augmented) and applied to that site's real
candidates (all of them, including out-of-scope and P) and to that site's tumour-free design scans (augment/real_tf.csv; evaluation
reference only, never fitted). The full fit (all sites) is saved for later use.
Evaluation (real, in scope, T / F; LOSO values): AUC of logLR vs AUC of raw elongation (lower = more tumour-like) by mode, mode x site,
mode x band; calibration by logLR bin (predicted mean vs empirical log ratio).
Output: ledger/decision_s1/elong_lr[/<tag>]/{candidates.csv, tables_full.json, eval.txt, eval_auc.csv, eval_calib.csv}.
Usage: python s1_fit_elong_lr.py [--tag smoke]"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

AUGD = os.path.join(paths.ledger_dir(), 'decision_s1', 'augment')
OUTD = os.path.join(paths.ledger_dir(), 'decision_s1', 'elong_lr')
DESIGN = os.path.join(paths.ledger_dir(), 'graph_v3', 'design.csv')
MIN_N, ALPHA, CLIP, KMIN, KMAX = 30, 0.5, 4.0, 3, 10
BANDS, BL = [0, 1, 10, 1e9], ['<1', '1-10', '>10']


def sp_class(m):
    return 'thin' if m <= 2.5 else ('mid' if m <= 5.0 else 'thick')


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prep(d):
    d = d[d.note.isna() | (d.note == '')].copy(); d['case'] = d.case.astype(str)
    d['lel'] = np.log10(d.elong.astype(float).fillna(1e4).clip(1, 1e4))
    d['band'] = pd.cut(d.ml.astype(float), BANDS, labels=BL, right=False).astype(str)
    d['spc'] = d.maxsp.astype(float).map(sp_class)
    d['in_scope'] = d.in_scope.astype(str) == 'True'
    return d


def pav_nonincreasing(y, w):
    """weighted isotonic regression, non-increasing"""
    blocks = []                                   # [value, weight, count]
    for v, ww in zip(y, w):
        blocks.append([v, max(ww, 1e-9), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop(); v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[v] * n for v, _, n in blocks])


def fit_table(sub):
    T, F = sub[sub.side == 'T'], sub[sub.side == 'F']
    K = int(np.clip(min(len(T), len(F)) // 20, KMIN, KMAX))
    edges = np.unique(np.quantile(sub.lel, np.linspace(0, 1, K + 1)))
    inner = edges[1:-1]; nb = len(inner) + 1
    b = np.searchsorted(inner, sub.lel.to_numpy(), side='right')
    isT = (sub.side == 'T').to_numpy(); w = sub.w.to_numpy()
    wT = np.bincount(b[isT], w[isT], nb); wF = np.bincount(b[~isT], w[~isT], nb)
    pT = (wT + ALPHA) / (wT.sum() + nb * ALPHA); pF = (wF + ALPHA) / (wF.sum() + nb * ALPHA)
    l = np.clip(pav_nonincreasing(np.log(pT / pF), wT + wF), -CLIP, CLIP)
    return dict(inner=[round(float(x), 5) for x in inner], llr=[round(float(x), 4) for x in l],
                nT=int(len(T)), nF=int(len(F)), wT=[round(float(x), 2) for x in wT], wF=[round(float(x), 2) for x in wF])


def ok(sub):
    return (sub.side == 'T').sum() >= MIN_N and (sub.side == 'F').sum() >= MIN_N


def fit_all(fd):
    """tables per stratum with fallback; returns dict key -> table, and resolver"""
    tabs = {'all': fit_table(fd)}
    for b in BL:
        s = fd[fd.band == b]
        if ok(s): tabs[b] = fit_table(s)
        for c in ('thin', 'mid', 'thick'):
            s2 = s[s.spc == c]
            if ok(s2): tabs[f'{b}|{c}'] = fit_table(s2)
    return tabs


def level(tabs, band, spc):
    k = f'{band}|{spc}'
    return k if k in tabs else (band if band in tabs else 'all')


def apply(tabs, d):
    lv = [level(tabs, b, c) for b, c in zip(d.band, d.spc)]
    out = np.array([tabs[k]['llr'][int(np.searchsorted(tabs[k]['inner'], e, side='right'))] for k, e in zip(lv, d.lel)])
    return out, lv


def weights(fd):
    fd = fd.copy(); fd['w'] = 1.0
    for s in ('T', 'F'):
        nr = ((fd.source == 'real') & (fd.side == s)).sum(); m = (fd.source == 'aug') & (fd.side == s)
        if m.sum(): fd.loc[m, 'w'] = nr / m.sum()
    return fd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default=''); a = ap.parse_args()
    ad = os.path.join(AUGD, a.tag) if a.tag else AUGD; od = os.path.join(OUTD, a.tag) if a.tag else OUTD; os.makedirs(od, exist_ok=True)
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
    des = pd.read_csv(DESIGN, usecols=['ds', 'case', 'site'], low_memory=False).drop_duplicates(['ds', 'case'])
    des['case'] = des.case.astype(str); site = {(u, v): s for u, v, s in des.itertuples(index=False)}
    for x in (re_, ag, tf): x['site'] = [site.get((u, v), '?') for u, v in zip(x.ds, x.case)]
    fr = re_[re_.in_scope & re_.lab.isin(['T', 'F'])].assign(side=lambda x: x.lab)
    fa = ag[ag.lab.isin(['T', 'F'])].assign(side=lambda x: x.lab)
    fa = fa[[use_aug[s] for s in fa.side]]
    fd = pd.concat([fr, fa], ignore_index=True)
    log = [f'fit rows: real T {(fr.side == "T").sum()} F {(fr.side == "F").sum()} | augmented used: tumour side {use_aug["T"]}, '
           f'non-tumour side {use_aug["F"]} -> aug T {(fa.side == "T").sum()} F {(fa.side == "F").sum()} | sites {sorted(fd.site.unique())}']
    full = fit_all(weights(fd))
    json.dump(full, open(os.path.join(od, 'tables_full.json'), 'w'), indent=1)
    log.append('full-fit strata: ' + ', '.join(f'{k} (T {v["nT"]}, F {v["nF"]}, bins {len(v["llr"])})' for k, v in full.items()))
    # leave-one-site-out
    cand = pd.concat([re_.assign(tumour_free=False), tf.assign(tumour_free=True)], ignore_index=True)
    cand['llr'] = np.nan; cand['level'] = ''
    for s in sorted(cand.site.unique()):
        tabs = fit_all(weights(fd[fd.site != s]))
        m = cand.site == s
        if m.sum():
            v, lv = apply(tabs, cand[m]); cand.loc[m, 'llr'] = v; cand.loc[m, 'level'] = lv
    cand['llr_full'] = apply(full, cand)[0]
    cand[['ds', 'case', 'site', 'tumour_free', 'mode', 'cid', 'lab', 'in_scope', 'category', 'ml', 'elong', 'lel', 'band', 'spc', 'llr',
          'llr_full', 'level']].to_csv(os.path.join(od, 'candidates.csv'), index=False)
    # evaluation
    ev = cand[~cand.tumour_free & cand.in_scope & cand.lab.isin(['T', 'F'])].copy(); ev['y'] = (ev.lab == 'T').astype(int)
    rows = []
    for mode, g in ev.groupby('mode'):
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
    log += ['\nAUC (in scope, T vs F; LOSO logLR vs raw elongation):', A[A.by == 'mode'].to_string(index=False),
            '\nby band:', A[A.by == 'band'].to_string(index=False),
            '\nby site (worst 12 by auc_llr):', A[A.by == 'site'].sort_values('auc_llr').head(12).to_string(index=False),
            '\ncalibration (LOSO logLR bins; empirical = log of class-conditional bin frequencies):', C.to_string(index=False),
            f'\nlevels used (real candidates): {cand.level.value_counts().to_dict()}',
            f'tumour-free design candidates scored: {int(cand.tumour_free.sum())} (evaluation reference, never fitted)']
    txt = '\n'.join(log); print(txt); open(os.path.join(od, 'eval.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
