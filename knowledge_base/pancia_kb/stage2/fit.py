"""Fitting of the stage-2 quantities on expert samples (expert.py / s2_expert_build.py). Nothing here reads agent outputs or TCGA.

  r       95th percentile over simulated anchors of the farthest expert-tumour voxel from the anchor (mm)
  eta     noisy-OR regression of t_v over the scan's simulated lesions, pi_v = 1 - prod_L (1 - sigma(eta . psi_v(L))), matching how
          run.py combines a cluster (one lesion: weighted logistic regression); voxels within r of an anchor or in the host
          territory; weights = sampling weights x 1 / (number of anchor-scenario rows of the case), times the site weight
  a_w,b_w per case: pseudo-likelihood estimate of a common coupling weight omega on the expert mask with the fitted prior as unary,
          P(t_v = 1 | neighbours) = sigma(logit pi_v - omega (ns_v - 2 n1_v)); Gamma(a_w, b_w) by moments over cases
  nu      population tail shape of a tumour TISSUE (user, 25 Oct): per expert tumour, a Student-t mixture (1..K_max components by
          lower bound, nu learned per component; fit_tumour_components), pooled over components on 1/nu with sites, cases,
          tumours and components weighted equally (pooled_nu). Before: one Student-t fitted to the whole tumour, which reads
          between-tissue heterogeneity (e.g. rim + necrosis) as heavy tails.
No appearance prior is fitted: appearance does not transfer across scans (CT phase, MR scale and sequence, scanner; user, 25 Oct),
so the Normal-Wishart priors are set within each scan (vb.class_priors).
Site weighting (user, 25 Oct: pan-cancer, no site bias): every estimate pools over tumour SITES with equal weight, each site's weight
shared equally by its contributing cases (site_weights): eta (case weights), r (weighted quantile), (a_w, b_w) (weighted moments), nu
(weighted median). Weights are computed over the cases that actually contribute.
Design values (disclosed, not fitted): K_max = 4, alpha0 = 1 / K_max, within-scan Normal-Wishart strength beta0 = 1, nu0 = D + 2,
L2 1e-4."""
import json
import numpy as np
from scipy import optimize, stats
from . import prior as PR, features as F

DESIGN = dict(K_max=4, alpha0=0.25, nw_beta0=1.0, nw_nu0=F.D + 2.0, l2=1e-4)   # nw_*: strength of the within-scan priors


def site_weights(sites):
    """equal weight per site, shared equally by its items: w_i = 1 / (n_sites * n_items(site_i)); sums to 1"""
    sites = list(sites)
    if not sites:
        return np.zeros(0)
    cnt = {}
    for x in sites:
        cnt[x] = cnt.get(x, 0) + 1
    return np.array([1.0 / (len(cnt) * cnt[x]) for x in sites], float)


def wquantile(x, w, q):
    """weighted quantile (q in [0, 1]); the smallest x whose cumulative weight reaches q"""
    x = np.asarray(x, float); w = np.asarray(w, float); o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(int(np.searchsorted(c, q - 1e-12)), len(x) - 1)])


def fit_r(rrows, q=95.0, site_of=None):
    """95th percentile of the farthest tumour distance over simulated anchors; each site weighs equally (each case's anchors share
    its weight) when site_of (ds -> site) is given"""
    if not rrows:
        return None
    v = np.asarray([r['far_mm'] for r in rrows], float)
    if site_of is None:
        return float(np.percentile(v, q))
    case = [(r['ds'], r['case']) for r in rrows]
    wc = dict(zip(sorted(set(case)), site_weights([site_of[d] for d, _ in sorted(set(case))])))
    nper = {}
    for k in case:
        nper[k] = nper.get(k, 0) + 1
    w = np.array([wc[k] / nper[k] for k in case])
    return wquantile(v, w, q / 100.0)


def design_matrix(S):
    return PR.basis(S['q'], S['sd'], S['in_h'], S['in_o'], S['kb_h'], S['dep_h'], S['dep_o'])


def fit_eta(X, t, w, l2=DESIGN['l2']):
    """weighted logistic regression, columns standardised (intercept excluded), returns eta on the original scale"""
    X = np.asarray(X, np.float64); t = np.asarray(t, np.float64); w = np.asarray(w, np.float64); w = w / w.sum()
    # standardisation with the fit weights (sites equal), so the L2 penalty's scale does not depend on which site has more rows
    mu = (w[:, None] * X).sum(0); sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0)); mu[0] = 0.0; sd[0] = 1.0; sd[sd < 1e-8] = 1.0
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


def _softplus(z):
    return np.logaddexp(0.0, z)


def noisy_or_S(Xl, grp, n, b):
    """S_r = sum_{j in r} softplus(z_rj), z = Xl b; p_r = 1 - exp(-S_r) = 1 - prod_j (1 - sigma(z_rj))"""
    z = Xl @ b
    return np.bincount(grp, weights=_softplus(z), minlength=n), z


def logit_from_S(S):
    """logit of p = 1 - exp(-S): log(exp(S) - 1), stable for small and large S"""
    S = np.maximum(S, 1e-12)
    return np.where(S > 30.0, S, np.log(np.expm1(np.minimum(S, 30.0))))


def fit_eta_noisy_or(Xl, grp, t, w, l2=DESIGN['l2']):
    """eta for the cluster prior used at inference (run.py): pi_v = 1 - prod_L (1 - sigma(eta . psi_v(L))) over the cluster's lesions.
    Long format: one row of Xl per (voxel r, lesion L), grp[i] = r. t, w per voxel. With one lesion per voxel this is the weighted
    logistic regression of fit_eta. Columns standardised with the fit weights (intercept excluded); eta on the original scale."""
    Z = np.array(Xl, np.float64); grp = np.asarray(grp, np.int64); t = np.asarray(t, np.float64)      # Z: one float64 copy (memory)
    w = np.asarray(w, np.float64); w = w / w.sum(); n = len(t)
    nl = np.bincount(grp, minlength=n).astype(np.float64); wl = (w / np.maximum(nl, 1))[grp]
    mu = wl @ Z; sd = np.sqrt(np.maximum(wl @ (Z * Z) - mu * mu, 0.0)); mu[0] = 0.0; sd[0] = 1.0; sd[sd < 1e-8] = 1.0
    Z -= mu; Z /= sd

    def f(b):
        S, z = noisy_or_S(Z, grp, n, b); S = np.maximum(S, 1e-12)
        lp = np.log(-np.expm1(-S))                                  # log p
        nll = -(w * (t * lp - (1 - t) * S)).sum() + l2 * (b[1:] ** 2).sum()
        dS = w * ((1 - t) - t / np.expm1(S))                        # d nll / d S_r
        g = Z.T @ (dS[grp] / (1.0 + np.exp(-z))); g[1:] += 2 * l2 * b[1:]
        return nll, g
    b0 = np.zeros(Z.shape[1]); pt = np.clip((w * t).sum(), 1e-4, 1 - 1e-4); b0[0] = np.log(pt / (1 - pt))
    o = optimize.minimize(f, b0, jac=True, method='L-BFGS-B', options=dict(maxiter=3000))
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


def gamma_moments(x, w=None):
    x = np.asarray(x, float); w = np.ones(len(x)) if w is None else np.asarray(w, float)
    k = np.isfinite(x) & (x > 0); x, w = x[k], w[k]
    if len(x) < 3:
        return None, None
    w = w / w.sum(); m = float((w * x).sum()); v = float((w * (x - m) ** 2).sum() / max(1.0 - (w ** 2).sum(), 1e-12))
    if v <= 1e-12:
        v = (0.1 * m) ** 2
    return float(m * m / v), float(m / v)


def fit_nu(samples, sites=None):
    """per-case Student-t degrees of freedom of the standardised tumour intensity; weighted median (sites equal) when sites given"""
    nus, keep = [], []
    for i, xs in enumerate(samples):
        xs = np.asarray(xs, float)
        if len(xs) < 200 or xs.std() <= 0:
            continue
        z = (xs - np.median(xs)) / xs.std()
        try:
            df, _, _ = stats.t.fit(z)
            nus.append(float(np.clip(df, 2.0, 50.0))); keep.append(i)
        except Exception:
            pass
    if not nus:
        return 5.0, nus
    if sites is None:
        return float(np.median(nus)), nus
    return wquantile(nus, site_weights([sites[i] for i in keep]), 0.5), nus


def fit_tumour_components(Y, K_max=DESIGN['K_max'], alpha0=DESIGN['alpha0'], nw=(DESIGN['nw_beta0'], DESIGN['nw_nu0']), nu_init=10.0,
                          max_iter=1000):
    """Student-t mixture of ONE expert tumour's voxels (every voxel is tumour), as stage 2 models a tumour: sparse Dirichlet
    components, within-scan Normal-Wishart priors, nu learned per component (vb.nu_update). The number of components K = 1..K_max
    is chosen by the variational lower bound (Bayesian model selection), because with spare components coordinate ascent can split
    one tissue into overlapping copies that a fixed K_max never merges back (25 Oct, synthetic check). Returns dict(K, elbos,
    components=[dict(rho, n, nu)])."""
    from . import vb
    Y = np.asarray(Y, np.float64); n = len(Y)
    tp, bps = vb.class_priors(Y, np.ones(n), np.ones((n, 1)), beta0=nw[0], nu0=nw[1])
    best = None; elbos = []
    for K in range(1, K_max + 1):
        m = vb.Model(Y, np.ones(n), np.zeros(n, int), np.ones((1, 1), bool), tp, bps, K=K, alpha0=alpha0, nu=nu_init, tumour=True,
                     learn_nu=True)
        m.init(np.ones(n)).run(max_iter=max_iter)
        elbos.append(float(m.elbo))
        if best is None or m.elbo > best[0]:
            rho = m.a_rho / m.a_rho.sum()
            best = (m.elbo, K, [dict(rho=float(rho[k]), n=float(m.R[:, k].sum()), nu=float(m.nu_k[k])) for k in range(K)])
    return dict(K=best[1], elbos=[round(e, 2) for e in elbos], components=best[2])


def pooled_nu(rows, site_of, min_rho=0.05, min_n=200.0):
    """population prior of the Student-t tail shape from the expert tumours' components (rows: s2_nu.py output, one per tumour).
    Pooled on the tail index 1/nu (0 = Gaussian; avoids averaging nu near its upper bound). Weights: sites equal, cases within a
    site equal, tumours within a case equal, a tumour's components equal (a one-tissue tumour counts as much as a four-tissue one).
    Components with weight < min_rho or fewer than min_n voxels give no estimate. Returns dict (None without components)."""
    items = []
    for r in rows:
        comps = [c for c in r['components'] if c['rho'] >= min_rho and c['n'] >= min_n]
        for c in comps:
            items.append((site_of[r['ds']], (r['ds'], r['case']), (r['ds'], r['case'], r['j']), len(comps), 1.0 / c['nu']))
    if not items:
        return None
    sites = sorted({i[0] for i in items}); w = []
    for s_, case, tum, nc, _ in items:
        cases = {i[1] for i in items if i[0] == s_}; tums = {i[2] for i in items if i[1] == case}
        w.append(1.0 / (len(sites) * len(cases) * len(tums) * nc))
    w = np.asarray(w); x = np.asarray([i[4] for i in items]); w = w / w.sum()
    lx = np.log(x); mlog = float((w * lx).sum()); sdlog = float(np.sqrt((w * (lx - mlog) ** 2).sum()))
    inv_med = wquantile(x, w, 0.5)
    per_site = {s_: round(1.0 / wquantile(x[[i[0] == s_ for i in items]], w[[i[0] == s_ for i in items]], 0.5), 2) for s_ in sites}
    return dict(nu=1.0 / inv_med, inv_nu_median=inv_med, log_inv_nu_mean=mlog, log_inv_nu_sd=sdlog, n_components=len(items),
                n_tumours=len({i[2] for i in items}), n_sites=len(sites), nu_median_per_site=per_site)


def calibration(p, t, w, bins=10):
    p = np.asarray(p); edges = np.linspace(0, 1, bins + 1); rows = []
    for i in range(bins):
        s = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if s.any():
            rows.append(dict(bin=f'{edges[i]:.1f}-{edges[i + 1]:.1f}', mean_p=float(np.average(p[s], weights=w[s])),
                             freq=float(np.average(t[s], weights=w[s])), weight=float(w[s].sum())))
    return rows
