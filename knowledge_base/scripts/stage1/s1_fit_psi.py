"""Stage 1 (v8.5 item brought forward, 29 Oct): refit the extent prior psi on real agent outlines (rows of s1_extent_rows.py).

psi is a piecewise-constant function of log(|xi| / |a|) (bin edges as v8.4: 0.05, 0.4, 0.8, 1.2, 1.8, 2.5; design values), applied
to every candidate extent of every anchor alike: w0(xi) ~ psi(bin(xi)), normalised over the anchor's candidates. eta_f (flags) stays
{} as in v8.4: the v8.4 flags (leak, organ, flat, prompt mismatch) are computed inside the TCGA ledger and are not reproduced on R2Seg,
so no flag weight is fitted (disclosed).
Methods (groups = anchors with >= 2 candidates; weights: tumour sites equal, a site's weight shared equally by its cases, a case's
by its anchors, as every other estimate of the framework):
  M1  density ratio (v7/v8.4 method): psi(bin) = weighted share of best candidates in the bin / weighted share of all candidates,
      one pseudo-group of smoothing spread over the bins (as the v7 add-one), normalised to bin 0
  M2  conditional logit over each anchor's candidates on bin indicators (bin 0 reference), L2 0.01 (as v8)
Held out: leave one site out (9 folds). Per held-out site and then sites equal, over ALL true anchors (one-candidate anchors
included): Dice of the chosen extent (argmax w0; ties -> first candidate, as stage 1), best-choice rate, share chosen > 1.5x the
tumour, and Dice of q >= 1/2 (exact, from the voxel patterns), for anchor only, v8.4 psi, M1, M2, U and the oracle.
Gate (pre-specified; anchor-only added 29 Oct): the refit (M2 if it beats M1 by >= 0.01 held-out Dice, else M1) is adopted only if it
beats BOTH v8.4 psi and the anchor alone on held-out sites-equal Dice of the chosen extent; otherwise the anchor is the extent (psi =
indicator of the anchor's bin) if the anchor beats v8.4 psi, else v8.4 psi is kept. fit_psi.json 'method' = M1 / M2 / anchor / v8.4.
Outputs: ledger/v85/extent/fit_psi.json (psi dict as fit_host6.json, eta {}, method, gate, held-out table), fit_psi_report.txt
Usage: python s1_fit_psi.py"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

EDGES = [-1e9, 0.05, 0.4, 0.8, 1.2, 1.8, 2.5, 1e9]; NB = len(EDGES) - 1


def load(O):
    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(O, 'extent_rows*.csv')))], ignore_index=True)
    E['grp'] = E.ds + '|' + E.case.astype(str) + '|' + E.anchor.astype(str)
    E = E.drop_duplicates(['grp', 'cand'])
    E['bin'] = np.clip(np.searchsorted(EDGES, E.logratio, side='right') - 1, 0, NB - 1)
    P = {}
    for f in sorted(glob.glob(os.path.join(O, 'patterns*.jsonl'))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); P[f"{r['ds']}|{r['case']}|{r['anchor']}"] = r
    return E, P


def group_weights(E):
    """framework rule: every tumour site counts equally, each site's weight shared equally by its cases, each case's by its anchors"""
    g = E.drop_duplicates('grp')[['grp', 'site', 'ds', 'case']].copy()
    g['ck'] = g.ds + '|' + g.case.astype(str)
    ns = g.site.nunique()
    ncase = g.groupby('site').ck.transform('nunique'); nanc = g.groupby('ck').grp.transform('size')
    return dict(zip(g.grp, 1.0 / (ns * ncase * nanc)))


def fit_m1(T, gw):
    T = T[T.groupby('grp').cand.transform('size') >= 2]
    w = T.grp.map(gw).to_numpy()
    nc = T.groupby('grp').cand.transform('size').to_numpy()
    best = np.bincount(T.bin[T.best == 1], weights=w[T.best.to_numpy() == 1], minlength=NB)
    allc = np.bincount(T.bin, weights=w / nc, minlength=NB)                 # each group's candidates share its weight
    pseudo = w.mean() if len(w) else 1.0
    best = best + pseudo / NB; allc = allc + pseudo / NB
    v = (best / best.sum()) / (allc / allc.sum())
    return v / v[0]


def fit_m2(T, gw, l2=0.01):
    T = T[T.groupby('grp').cand.transform('size') >= 2]
    g = pd.factorize(T.grp)[0]; ng = g.max() + 1
    wg = np.zeros(ng); wg[g] = T.grp.map(gw).to_numpy(); wg = wg / wg.sum() * ng
    X = np.zeros((len(T), NB - 1)); b = T.bin.to_numpy()
    for j in range(1, NB):
        X[:, j - 1] = b == j
    y = T.best.to_numpy(float)

    def f(th):
        z = X @ th; m = np.full(ng, -np.inf); np.maximum.at(m, g, z)
        lse = m + np.log(np.bincount(g, weights=np.exp(z - m[g]), minlength=ng)); p = np.exp(z - lse[g])
        nll = -((wg[g] * z * y).sum() - (wg * lse).sum()) + l2 * (th ** 2).sum()
        return nll, -(X.T @ (wg[g] * (y - p))) + 2 * l2 * th
    th = minimize(f, np.zeros(NB - 1), jac=True, method='L-BFGS-B').x
    return np.r_[1.0, np.exp(th)]


def q_dice(pat, w0):
    """Dice of {q >= 1/2} with the anchor's expert tumour, exactly, from the voxel patterns"""
    q = np.array([sum(w0[j] for j in range(len(w0)) if (c >> j) & 1) for c in pat['codes']])
    sel = q >= 0.5 - 1e-9; n = np.asarray(pat['n']); nt = np.asarray(pat['nt'])
    tp = nt[sel].sum(); pv = n[sel].sum(); tv = nt.sum() + pat['t_out']
    return 2 * tp / max(pv + tv, 1)


def evaluate(te, P, scorers):
    """per group: Dice of the chosen extent, best choice, over 1.5x, q-Dice; one row per group and method"""
    out = []
    for gname, d in te.groupby('grp', sort=False):
        d = d.sort_values('cand')
        for meth, sc in scorers.items():
            if meth == 'oracle':
                j = int(d.dice.to_numpy().argmax()); w0 = None
            elif meth == 'anchor':
                j = 0; w0 = None
            elif meth == 'U':
                nm = d.name.tolist(); j = nm.index('U') if 'U' in nm else int(d.ml.to_numpy().argmax()); w0 = None
            else:
                w = sc(d); w0 = w / w.sum(); j = int(np.argmax(w0))
            r = d.iloc[j]
            row = dict(grp=gname, site=r.site, method=meth, dice=r.dice, best=int(r.best == 1), over15=float(r.ml > 1.5 * r.G_ml))
            if w0 is not None and gname in P:
                row['q_dice'] = q_dice(P[gname], w0)
            out.append(row)
    return pd.DataFrame(out)


def main():
    O = os.path.join(paths.ledger_dir(), 'v85', 'extent')
    E, P = load(O)
    v84 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))['psi']
    v84v = np.asarray(v84['val'], float)
    gw_all = group_weights(E)
    rep = [f'psi refit on real agent outlines (R2Seg; BiomedParse + VoxTell; two- and single-agent anchors); rows {len(E)}, '
           f'anchors {E.grp.nunique()} (with >= 2 candidates: {(E.groupby("grp").cand.size() >= 2).sum()}), sites {E.site.nunique()}',
           'per site anchors: ' + json.dumps(E.drop_duplicates('grp').site.value_counts().to_dict()),
           'chosen by v8.4 psi (names): ' + json.dumps(E[E.groupby('grp').w0_v84.transform('max') == E.w0_v84].name.value_counts().to_dict()),
           'best (highest Dice) candidate names: ' + json.dumps(E[E.best == 1].name.value_counts().to_dict()), '']
    res = []; fold_psis = {}
    for s in sorted(E.site.unique()):
        tr, te = E[E.site != s], E[E.site == s]
        gw = group_weights(tr)
        m1 = fit_m1(tr, gw); m2 = fit_m2(tr, gw); fold_psis[s] = dict(M1=m1.tolist(), M2=m2.tolist())
        sc = {'anchor': None, 'v8.4': lambda d: v84v[d.bin.to_numpy()], 'M1': lambda d, v=m1: v[d.bin.to_numpy()],
              'M2': lambda d, v=m2: v[d.bin.to_numpy()], 'U': None, 'oracle': None}
        res.append(evaluate(te, P, sc))
    R = pd.concat(res, ignore_index=True)
    R['w'] = R.grp.map(gw_all); R = R.merge(E.drop_duplicates('grp')[['grp', 'kind']], on='grp', how='left')
    wmean = lambda d: pd.Series({c: np.average(d[c].dropna(), weights=d.loc[d[c].notna(), 'w']) if d[c].notna().any() else np.nan
                                 for c in ('dice', 'best', 'over15', 'q_dice')})
    per = R.groupby(['site', 'method']).apply(wmean, include_groups=False)          # cases equal within a site
    byk = R.groupby(['kind', 'site', 'method']).apply(wmean, include_groups=False).groupby(level=['kind', 'method']).mean()
    sw = per.groupby(level='method').mean()
    rep += ['== held out (leave one site out): per site, then sites equal ==',
            per['dice'].unstack('method').round(3).to_string(), '', sw.round(3).to_string(), '',
            'by anchor kind (two-agent / single-agent; sites equal):', byk['dice'].unstack('method').round(3).to_string(), '']
    d1, d2, d0, da = sw.loc['M1', 'dice'], sw.loc['M2', 'dice'], sw.loc['v8.4', 'dice'], sw.loc['anchor', 'dice']
    pick = 'M2' if d2 >= d1 + 0.01 else 'M1'
    gw = group_weights(E)
    psi_full = (fit_m2 if pick == 'M2' else fit_m1)(E, gw)
    # gate (pre-specified, 29 Oct): the refit is adopted only if it beats BOTH v8.4 psi and the anchor alone on held-out sites-equal
    # Dice of the chosen extent. Otherwise: the anchor itself is the extent when the anchor beats v8.4 psi (psi = indicator of the
    # anchor's bin: every candidate larger than 1.05x the anchor gets weight 0; the anchor is listed first, so it is chosen), else v8.4.
    ANCHOR = np.r_[1.0, np.zeros(NB - 1)]
    if sw.loc[pick, 'dice'] > d0 and sw.loc[pick, 'dice'] > da:
        outcome, final = pick, psi_full
    elif da > d0:
        outcome, final = 'anchor', ANCHOR
    else:
        outcome, final = 'v8.4', v84v
    for s_ in fold_psis:
        fold_psis[s_]['anchor'] = ANCHOR.tolist(); fold_psis[s_]['v8.4'] = v84v.tolist()
    adopted = outcome != 'v8.4'
    rep += [f'GATE: refit = {pick} (held-out Dice M1 {d1:.3f}, M2 {d2:.3f}); v8.4 psi {d0:.3f}; anchor only {da:.3f} -> extent prior = '
            + {'anchor': 'ANCHOR ONLY (refit does not beat the anchor; the anchor beats v8.4)', 'v8.4': 'v8.4 psi kept'}.get(outcome, f'{pick} REFIT ADOPTED'),
            f'psi refit (all sites, {pick}): ' + ', '.join(f'{v:.3g}' for v in psi_full), f'v8.4 psi: ' + ', '.join(f'{v:.3g}' for v in v84v),
            'bin edges (log ratio): ' + ', '.join(str(e) for e in EDGES[1:-1]) + '; ratio = exp(edge): ' + ', '.join(f'{np.exp(e):.2f}' for e in EDGES[1:-1]),
            'per-fold psi: see fit_psi.json']
    out = dict(version='v8.5-psi', date=pd.Timestamp.today().date().isoformat(), source='R2Seg real agent outlines (s1_extent_rows.py)',
               method=outcome, refit_method=pick, adopted=adopted,
               psi=dict(edges=EDGES, val=[float(x) for x in final]),
               psi_refit=dict(edges=EDGES, val=[float(x) for x in psi_full]), psi_v84=v84, eta={},
               heldout=sw.round(4).to_dict(), heldout_per_site=per.round(4).reset_index().to_dict('records'), folds=fold_psis,
               weighting='tumour sites equal; a site shared equally by its cases, a case by its anchors')
    json.dump(out, open(os.path.join(O, 'fit_psi.json'), 'w'), indent=1)
    open(os.path.join(O, 'fit_psi_report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


if __name__ == '__main__':
    main()
