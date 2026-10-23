"""v6.2 fixed-sign size-conditioned geometry score (FROZEN_SCORE_v62_fixedsign_20261012.txt), packaged for the pipeline (16 Oct 2026).
Identical maths to scripts/v6_v62_eval.py (build/score_raw/score); the expert reference pool is ledger/v6/ll61/{pos,ts,r2n}.csv
(R2Seg tumours + variants, TotalSegmentator and R2Seg normals; expert data only). Features: elong, flat (geometry_terms), tube, blobby."""
from __future__ import annotations
import os, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

FEAT = ['elong', 'flat', 'tube', 'blobby']; DIR = np.array([1, 1, -1, 1]); CAP = np.log(100); THR = np.log(3)
band = lambda v: pd.cut(pd.Series(v, dtype=float), [0, 3, 20, 1e12], labels=['<3', '3-20', '>20']).astype(str).values


def _weights(g):
    w = np.zeros(len(g)); n = len(g)
    for y in (0, 1):
        m = (g.y == y).values; gs = g.grp[m]
        for k in gs.unique():
            mk = m & (g.grp == k).values; w[mk] = (n / 2) / gs.nunique() / mk.sum()
    return w


def score_raw(df, ref):
    Z = []
    for f, d in zip(FEAT, DIR):
        xs, cw = ref[f]; u = np.interp(df[f].values, xs, cw, left=cw[0] / 2, right=1 - 1e-4); u = np.clip(u, 1e-3, 1 - 1e-3)
        Z.append(d * np.clip(norm.ppf(u), -3, 3))
    return np.mean(Z, axis=0)


def build(ll61_dir):
    pos = pd.read_csv(f'{ll61_dir}/pos.csv'); ts = pd.read_csv(f'{ll61_dir}/ts.csv'); r2n = pd.read_csv(f'{ll61_dir}/r2n.csv')
    pos = pos[pos[FEAT + ['vol_ml']].notna().all(1)].assign(y=1, grp=lambda x: 'T_' + x.variant)
    neg = pd.concat([ts, r2n]); neg = neg[neg[FEAT + ['vol_ml']].notna().all(1)].assign(y=0, grp=lambda x: 'N_' + x.cat)
    D = pd.concat([pos, neg], ignore_index=True); D['band'] = band(D.vol_ml); M = {}
    for bn in ['<3', '3-20', '>20']:
        gb = D[D.band == bn]; w = _weights(gb); ref = {}
        for f in FEAT:
            o = np.argsort(gb[f].values); xs = gb[f].values[o]; cw = np.cumsum(w[o]); cw = cw / cw[-1]; ref[f] = (xs, cw)
        S = score_raw(gb, ref); y = gb.y.values
        fn = lambda p: np.sum(w * (np.logaddexp(0, p[0] + p[1] * S) - y * (p[0] + p[1] * S)))
        p = minimize(fn, [0, 1], method='L-BFGS-B', bounds=[(None, None), (0, None)]).x
        M[bn] = dict(ref=ref, a=float(p[0]), b=float(p[1]))
    return M


def score(df, M):
    """returns (log Lambda capped +-log 100, raw S, class)"""
    out = np.full(len(df), np.nan); S = np.full(len(df), np.nan); bd = band(df.vol_ml.values)
    for bn, m in M.items():
        k = bd == bn
        if k.any():
            S[k] = score_raw(df[k], m['ref']); out[k] = np.clip(m['a'] + m['b'] * S[k], -CAP, CAP)
    cls = np.where(out > THR, 'tumour-like', np.where(out < -THR, 'organ-like', 'uncertain'))
    return out, S, cls
