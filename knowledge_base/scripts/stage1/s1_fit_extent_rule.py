"""Extent rule on the ledger's own candidates, step 2: fit and held-out choices (23 Nov 2026; claude/extent_rule_refit_plan_2026-11-23.md).
Input: ledger/<L>/extent_rule/rows*.csv (s1_extent_rule_rows.py): every candidate extent of every accepted / uncertain anchor.
Rules:  R0 current psi (fit_psi.json, held-out site fold of the adopted method; ties -> first candidate, the anchor)
        R1 anchor only
        R2 size only: conditional logit on log-ratio bins (< -0.05 | -0.05..0.05 | then the psi edges 0.05 .. 2.5)
        R3 size + image: R2's bins + s_ext (does the extension look like the surroundings rather than the anchor?)
Target per anchor (true and false): the candidate with the largest net voxels tp - fp (expert labels: fitting / evaluation only).
Weights: sites equal, a site shared by its scans, a scan by its anchors; L2 0.01 on coefficients. Leave one site out: each site's
choices use coefficients fitted without it. Approximate held-out scan Dice (sum of chosen candidates' tp / fp per scan, overlaps
between anchors ignored) for accepted and accepted + uncertain lesions; the exact evaluation is s1_eval_r2seg --extent-choice.
Outputs: extent_rule/fit.json (per-fold coefficients), choices_R{0,1,2,3}.csv (ds, case, anchor, j), fit_report.txt.
Usage: python s1_fit_extent_rule.py [--ledger v86_r2seg_bp2]"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402
EDGES = [-0.05, 0.05, 0.4, 0.8, 1.2, 1.8, 2.5]          # 8 bins
L2 = 0.01


def bins(lr):
    return np.digitize(lr, EDGES)   # 0 .. 7


def design(d, rule):
    B = np.zeros((len(d), len(EDGES) + 1)); B[np.arange(len(d)), bins(d.logratio.values)] = 1
    B = B[:, [k for k in range(B.shape[1]) if k != 1]]   # reference bin: -0.05..0.05 (the anchor's own bin)
    return np.column_stack([B, d.s_ext.values]) if rule == 'R3' else B


def fit(d, rule):
    X = design(d, rule); g = d.gid.values; y = d.best.values.astype(float); w = d.w.values
    order = np.argsort(g, kind='stable'); X, g, y, w = X[order], g[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; wa = w[starts]
    def nll(b):
        s = X @ b; m = np.maximum.reduceat(s, starts); e = np.exp(s - np.repeat(m, np.diff(np.r_[starts, len(s)])))
        z = np.add.reduceat(e, starts); ll = np.add.reduceat(y * s, starts) - m - np.log(z)
        p = e / np.repeat(z, np.diff(np.r_[starts, len(s)]))
        grad = X.T @ ((y - p) * np.repeat(wa, np.diff(np.r_[starts, len(s)])))
        return -(wa * ll).sum() + L2 * b @ b, -grad + 2 * L2 * b
    r = minimize(nll, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B')
    return r.x


def choose(d, rule, coef=None, psi=None):
    if rule == 'R1':
        s = -d.j.values.astype(float)               # the anchor is candidate 0
    elif rule == 'R0':
        from pancia_kb import grounding_g2 as G2   # same psi evaluation as the ledger / s1_eval_r2seg
        P = G2.Psi(psi); s = np.log(np.maximum(np.array([P(x) for x in d.logratio.values], float), 1e-12)) - 1e-9 * d.j.values
    else:
        s = design(d, rule) @ coef - 1e-9 * d.j.values
    d = d.assign(score=s)
    return d.loc[d.groupby('gid').score.idxmax()]


def approx(ch, T):
    out = {}
    for name, dec in (('acc', ('accepted',)), ('acc_unc', ('accepted', 'uncertain'))):
        c = ch[ch.decision.isin(dec)].groupby(['site', 'ds', 'case']).agg(tp=('tp_ml', 'sum'), fp=('fp_ml', 'sum')).reset_index()
        c = T.merge(c, on=['site', 'ds', 'case'], how='left').fillna({'tp': 0, 'fp': 0})
        pos = c[c.T_ml > 0].copy(); pos['tp'] = np.minimum(pos.tp, pos.T_ml)
        pos['dice'] = 2 * pos.tp / (pos.tp + pos.fp + pos.T_ml); pos['fp_rel'] = pos.fp / pos.T_ml
        neg = c[c.neg]
        out[name] = dict(scan_dice=pos.groupby('site').dice.mean().mean(), recall=(pos.tp / pos.T_ml).groupby(pos.site).mean().mean(),
                         fp_rel_med=pos.groupby('site').fp_rel.median().mean(), neg_fp_ml=neg.fp.mean() if len(neg) else np.nan,
                         per_site=pos.groupby('site').dice.mean().round(3).to_dict())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ledger', default='v86_r2seg_bp2'); a = ap.parse_args()
    L = os.path.join(paths.ledger_dir(), a.ledger); O = os.path.join(L, 'extent_rule')
    d = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(O, 'rows*.csv'))], ignore_index=True)
    d = d.drop_duplicates(['ds', 'case', 'anchor', 'j']).sort_values(['ds', 'case', 'anchor', 'j']).reset_index(drop=True)
    d['gid'] = d.groupby(['ds', 'case', 'anchor']).ngroup(); d['net'] = d.tp_ml - d.fp_ml
    d['best'] = (d.groupby('gid').net.transform('max') == d.net) & ~d.duplicated(['gid', 'net'])   # one best per anchor (first on ties)
    na = d.groupby('gid').ds.transform('size')
    n_anchor_scan = d.drop_duplicates('gid').groupby(['ds', 'case']).size(); n_scan_site = d.drop_duplicates(['ds', 'case']).groupby('site').size()
    d['w'] = [1.0 / n_scan_site[s] / n_anchor_scan[(x, c)] for s, x, c in zip(d.site, d.ds, d.case)]
    pj = json.load(open(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json')))
    ev = pd.concat([pd.read_csv(f, usecols=['ds', 'case', 'site', 'T_ml', 'method'], dtype={'case': str}) for f in glob.glob(os.path.join(L, 'eval_psi_bp2', 'scans*.csv'))])
    ev = ev[ev.method == 'vt'].drop_duplicates(['ds', 'case']); ev['T_ml'] = pd.to_numeric(ev.T_ml, errors='coerce')
    neg = d.drop_duplicates(['ds', 'case']).set_index(['ds', 'case']).neg
    T = ev[['site', 'ds', 'case', 'T_ml']].copy(); T['neg'] = [bool(neg.get((x, c), False)) for x, c in zip(T.ds, T.case)]
    T = T[(T.T_ml > 0) | T.neg]
    multi = d[na >= 2]
    coefs = {'R2': {}, 'R3': {}}; choices = {r: [] for r in ('R0', 'R1', 'R2', 'R3')}
    for s in sorted(d.site.unique()):
        tr, te = multi[multi.site != s], d[d.site == s]
        for r in ('R2', 'R3'):
            coefs[r][s] = fit(tr, r).tolist(); choices[r].append(choose(te, r, np.asarray(coefs[r][s])))
        psi = dict(edges=pj['psi']['edges'], val=pj['folds'][s][pj['method']]) if s in pj.get('folds', {}) else pj['psi']
        choices['R0'].append(choose(te, 'R0', psi=psi)); choices['R1'].append(choose(te, 'R1'))
    full = {r: fit(multi, r).tolist() for r in ('R2', 'R3')}
    names = [f'bin{k}' for k in range(len(EDGES) + 1) if k != 1]
    rep = [f'Extent rule refit on {a.ledger} candidates: {d.gid.nunique()} accepted/uncertain anchors ({(na >= 2).groupby(d.gid).first().sum()} with >= 2 candidates), '
           f'{d.drop_duplicates(["ds","case"]).shape[0]} scans ({int(T.neg.sum())} tumour-free with anchors). Leave one site out, sites equal. '
           'Scan metrics are APPROXIMATE (overlaps between anchors ignored); the exact evaluation is s1_eval_r2seg --extent-choice.', '',
           'bins (log ratio): ' + ', '.join(f'{n}' for n in names) + ' (reference: -0.05..0.05); edges ' + str(EDGES),
           'R2 coefficients (all sites): ' + ', '.join(f'{n} {v:.2f}' for n, v in zip(names, full['R2'])),
           'R3 coefficients (all sites): ' + ', '.join(f'{n} {v:.2f}' for n, v in zip(names + ['s_ext'], full['R3'])), '']
    res = {}
    for r in ('R0', 'R1', 'R2', 'R3'):
        ch = pd.concat(choices[r]); ch[['ds', 'case', 'anchor', 'j']].to_csv(os.path.join(O, f'choices_{r}.csv'), index=False)
        res[r] = approx(ch, T)
    for k in ('acc', 'acc_unc'):
        rep += [f'== {k} (held out, approximate) ==']
        rep += [f"{r}: scan Dice {res[r][k]['scan_dice']:.3f}, recall {res[r][k]['recall']:.3f}, median FP/tumour {res[r][k]['fp_rel_med']:.3f}, "
                f"mean false ml per tumour-free scan {res[r][k]['neg_fp_ml']:.2f}" for r in res]
        rep += ['per site scan Dice: ' + json.dumps({r: res[r][k]['per_site'] for r in res}), '']
    json.dump(dict(date='2026-11-23', ledger=a.ledger, edges=EDGES, features={'R2': names, 'R3': names + ['s_ext']}, l2=L2,
                   folds=coefs, full=full), open(os.path.join(O, 'fit.json'), 'w'), indent=1)
    open(os.path.join(O, 'fit_report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


if __name__ == '__main__':
    main()
