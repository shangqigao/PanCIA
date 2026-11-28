"""Stage-1 decision, part 1a: realism check of augmented candidates (28 Nov 2026; plan claude/stage1_decision_plan_2026-11-28.md;
criterion pre-specified and user-approved: a classifier separating augmented from real candidates has AUC <= 0.65, per side and
size band; a side that fails uses real candidates only).

Input: ledger/decision_s1/augment/real.csv and [<tag>/]aug.csv (s1_augment_candidates.py; design set, tumour scans only). Real
candidates are restricted to the scans present in aug.csv (so a smoke run compares like with like).
Real candidates: in-scope, lab T or F (P excluded). Augmented: tumour side (category 'tumour', lab T; augmented F from tumour seeds
are counted on the false side) and non-tumour side (lab F).
Classifier: logistic regression (numpy, L2 1e-3), features log10 ml, log10 elongation (clipped 1..1e4), log10 nsl, thickness / largest
spacing, sphericity, log10 largest spacing; 5-fold CV AUC (folds by scan, so a scan's candidates never span train and test).
Bands: < 1, 1-10, > 10 ml. Also single-feature AUCs (real vs augmented) to show which feature differs.
Output: [<tag>/]realism.txt and realism.csv. Usage: python s1_augment_realism.py [--tag smoke]"""
import os, sys, zlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

D = os.path.join(paths.ledger_dir(), 'decision_s1', 'augment')
FEATS = ['lml', 'lel', 'lnsl', 'thk', 'sph', 'lsp']
LIMIT = 0.65


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logreg(X, y, lam=1e-3, it=50):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1]); g = X.T @ (p - y) + lam * w
        w -= np.linalg.solve(H, g)
    return w


def cv_auc(df, y):
    X = df[FEATS].to_numpy(float); mu, sd = X.mean(0), X.std(0) + 1e-9; X = (X - mu) / sd
    fold = np.array([zlib.crc32(f'{a}/{b}'.encode()) % 5 for a, b in zip(df.ds, df.case)])
    s = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(np.unique(y[tr])) < 2: continue
        w = logreg(X[tr], y[tr]); s[te] = np.c_[np.ones(te.sum()), X[te]] @ w
    return auc(y, s)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default=''); a = ap.parse_args()
    od = os.path.join(D, a.tag) if a.tag else D
    ag = pd.read_csv(os.path.join(od, 'aug.csv'), low_memory=False); re_ = pd.read_csv(os.path.join(D, 'real.csv'), low_memory=False)
    for x in (ag, re_): x['case'] = x.case.astype(str)
    sc = set(zip(ag.ds, ag.case)); re_ = re_[[(u, v) in sc for u, v in zip(re_.ds, re_.case)]]
    d = pd.concat([re_, ag], ignore_index=True)
    d = d[d.note.isna() | (d.note == '')].copy()
    d['in_scope'] = d.in_scope.astype(str) == 'True'
    d = d[(d.source == 'aug') | (d.in_scope & d.lab.isin(['T', 'F']))].copy()
    d['lml'] = np.log10(d.ml.clip(lower=1e-3)); d['lel'] = np.log10(d.elong.fillna(1e4).clip(1, 1e4))
    d['lnsl'] = np.log10(d.nsl.clip(lower=1)); d['thk'] = d.thick_mm / d.maxsp; d['lsp'] = np.log10(d.maxsp)
    d['band'] = pd.cut(d.ml, [0, 1, 10, 1e9], labels=['<1', '1-10', '>10'], right=False)
    d['side'] = d.lab.map({'T': 'tumour', 'F': 'non-tumour'})
    rows, out = [], []
    for side, g in d.groupby('side'):
        for band, h in [('all', g)] + [(b, g[g.band == b]) for b in ['<1', '1-10', '>10']]:
            y = (h.source == 'aug').to_numpy(int)
            n_real, n_aug = int((y == 0).sum()), int(y.sum())
            if n_real < 20 or n_aug < 20:
                rows.append(dict(side=side, band=band, n_real=n_real, n_aug=n_aug, cv_auc=np.nan, passed=None)); continue
            a = cv_auc(h, y); a2 = max(a, 1 - a)
            single = {f: round(max(auc(y, h[f]), 1 - auc(y, h[f])), 3) for f in FEATS}
            rows.append(dict(side=side, band=band, n_real=n_real, n_aug=n_aug, cv_auc=round(a2, 3), passed=a2 <= LIMIT, **{f'auc_{k}': v for k, v in single.items()}))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(od, 'realism.csv'), index=False)
    out.append(R.to_string(index=False))
    for side in ['tumour', 'non-tumour']:
        r = R[(R.side == side) & R.passed.notna() & (R.band != 'all')]   # criterion per size band; 'all' reported only
        verdict = 'PASS' if len(r) and r.passed.all() else 'FAIL'
        out.append(f'{side}: {verdict} (criterion CV AUC <= {LIMIT} in every band with >= 20 real and >= 20 augmented)')
    out.append('\nAugmented counts by category / ops (top):')
    a = d[d.source == 'aug']
    out.append(a.groupby(['category', 'lab']).size().to_string())
    out.append(a.ops.value_counts().head(12).to_string())
    txt = '\n'.join(out); print(txt); open(os.path.join(od, 'realism.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
