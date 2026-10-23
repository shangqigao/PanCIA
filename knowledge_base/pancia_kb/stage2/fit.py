"""Fitting of the stage-2 quantities on expert samples (expert.py / s2_expert_build.py). Nothing here reads agent outputs or TCGA.

  r       95th percentile over simulated anchors of the farthest expert-tumour voxel from the anchor (mm)
  eta     weighted logistic regression of t_v on the prior basis (prior.basis), voxels within r of the anchor or in the host
          territory; weights = sampling weights x 1 / (number of anchor-scenario rows of the case) so every case counts equally
  a_w,b_w per case: pseudo-likelihood estimate of a common coupling weight omega on the expert mask with the fitted prior as unary,
          P(t_v = 1 | neighbours) = sigma(logit pi_v - omega (ns_v - 2 n1_v)); Gamma(a_w, b_w) by moments over cases
  nu      median over cases of the Student-t degrees of freedom fitted to the tumour's smoothed intensity (clipped to [2, 50])
  appearance  per modality: tumour (by host class, and pooled), organ normal tissue (by organ, by class, pooled), unlabelled tissue;
          mean = mean of case means, covariance = mean within-case covariance + covariance of case means (between patients)
Design values (disclosed, not fitted): K_max = 4, alpha0 = 1 / K_max, Normal-Wishart strength beta0 = 1, nu0 = D + 2, L2 1e-4."""
import json
import numpy as np
from scipy import optimize, stats
from . import prior as PR, features as F

DESIGN = dict(K_max=4, alpha0=0.25, nw_beta0=1.0, nw_nu0=F.D + 2.0, l2=1e-4)


def fit_r(rrows, q=95.0):
    v = np.asarray([r['far_mm'] for r in rrows], float)
    return float(np.percentile(v, q)) if len(v) else None


def design_matrix(S):
    return PR.basis(S['q'], S['sd'], S['in_h'], S['in_o'], S['kb_h'], S['dep_h'], S['dep_o'])


def fit_eta(X, t, w, l2=DESIGN['l2']):
    """weighted logistic regression, columns standardised (intercept excluded), returns eta on the original scale"""
    X = np.asarray(X, np.float64); t = np.asarray(t, np.float64); w = np.asarray(w, np.float64); w = w / w.sum()
    mu = X.mean(0); sd = X.std(0); mu[0] = 0.0; sd[0] = 1.0; sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd

    def f(b):
        z = Z @ b; lp = -np.logaddexp(0, -z); lq = -np.logaddexp(0, z)
        nll = -(w * (t * lp + (1 - t) * lq)).sum() + l2 * (b[1:] ** 2).sum()
        p = 1.0 / (1.0 + np.exp(-z)); g = Z.T @ (w * (p - t)); g[1:] += 2 * l2 * b[1:]
        return nll, g
    b0 = np.zeros(Z.shape[1]); pt = np.clip((w * t).sum(), 1e-4, 1 - 1e-4); b0[0] = np.log(pt / (1 - pt))
    o = optimize.minimize(f, b0, jac=True, method='L-BFGS-B', options=dict(maxiter=2000))
    b = o.x / sd; b[0] = o.x[0] - (o.x[1:] * mu[1:] / sd[1:]).sum()
    return b, dict(success=bool(o.success), nll=float(o.fun), iters=int(o.nit))


def fit_omega_case(z, ns, n1, t, w):
    """1-D pseudo-likelihood MLE of omega in [0, 50] for one case"""
    t = t.astype(np.float64); w = w / w.sum(); u = (ns - 2 * n1).astype(np.float64)

    def nll(om):
        zz = z - om * u
        return -(w * (t * -np.logaddexp(0, -zz) + (1 - t) * -np.logaddexp(0, zz))).sum()
    o = optimize.minimize_scalar(nll, bounds=(0.0, 50.0), method='bounded')
    return float(o.x)


def gamma_moments(x):
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 3:
        return None, None
    m, v = x.mean(), x.var(ddof=1)
    if v <= 1e-12:
        v = (0.1 * m) ** 2
    return float(m * m / v), float(m / v)


def fit_nu(samples):
    nus = []
    for xs in samples:
        xs = np.asarray(xs, float)
        if len(xs) < 200 or xs.std() <= 0:
            continue
        z = (xs - np.median(xs)) / xs.std()
        try:
            df, _, _ = stats.t.fit(z)
            nus.append(float(np.clip(df, 2.0, 50.0)))
        except Exception:
            pass
    return (float(np.median(nus)) if nus else 5.0), nus


def _pool(items):
    items = [d for d in items if d is not None and d.get('n', 0) >= 20]
    if not items:
        return None
    M = np.array([d['mean'] for d in items], float); C = np.array([d['cov'] for d in items], float)
    within = C.mean(0); between = np.cov(M.T) if len(M) > 1 else np.zeros_like(within)
    return dict(mean=M.mean(0).tolist(), cov=(within + between).tolist(), n_cases=len(items))


def fit_appearance(apps):
    out = {}
    for mod in sorted({a['mod'] for a in apps}):
        A = [a for a in apps if a['mod'] == mod]
        tum = {'pooled': _pool([a['tumour'] for a in A])}
        for cls in ('solid', 'hollow'):
            p = _pool([a['tumour'] for a in A if a['cls'] == cls])
            if p: tum[cls] = p
        org = {}; keys = sorted({k for a in A for k in a['organs']})
        for k in keys:
            p = _pool([a['organs'][k] for a in A if k in a['organs']])
            if p: org[k] = p
        from .expert import organ_cls
        ocls = {'pooled': _pool([a['organs'][k] for a in A for k in a['organs']])}
        for cls in ('solid', 'hollow'):
            p = _pool([a['organs'][k] for a in A for k in a['organs'] if organ_cls(k) == cls])
            if p: ocls[cls] = p
        out[mod] = dict(tumour={k: v for k, v in tum.items() if v}, organ=org, organ_cls={k: v for k, v in ocls.items() if v},
                        generic={'pooled': _pool([a['generic'] for a in A])} if _pool([a['generic'] for a in A]) else {})
    return out


def calibration(p, t, w, bins=10):
    p = np.asarray(p); edges = np.linspace(0, 1, bins + 1); rows = []
    for i in range(bins):
        s = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if s.any():
            rows.append(dict(bin=f'{edges[i]:.1f}-{edges[i + 1]:.1f}', mean_p=float(np.average(p[s], weights=w[s])),
                             freq=float(np.average(t[s], weights=w[s])), weight=float(w[s].sum())))
    return rows
