"""Mean-field variational Bayes for stage 2 (main.tex Eqs. mrf, appearance, stage2_bf).

Labels per voxel: tumour components k = 1..K (sparse Dirichlet weights rho ~ Dir(alpha0)) or background classes c (organ normal
tissue or generic tissue; per-voxel allowed set given by the voxel's type = the organs whose territory contains it; mixing weights
lambda_type ~ Dir(1) over the allowed classes, learned per scan).
Prior on the tumour indicator t_v: pi_v (spatial prior) with a Potts coupling exp(-omega_vw s_vw 1[t_v != t_w]) on 6-neighbours,
omega_vw ~ Gamma(a_w, b_w) latent per edge. The Potts factor is used unnormalised (as a pairwise penalty); its posterior is
q(omega_vw) = Gamma(a_w, b_w + s_vw E[1[t_v != t_w]]), so E[omega] falls where the labels disagree, which the appearance drives.
Appearance: multivariate Student-t per class as a Gaussian scale mixture, y | z, u ~ N(mu, (u Lambda)^-1), u ~ Gamma(nu/2, nu/2),
conjugate Normal-Wishart priors on (mu, Lambda). Appearance is inferred WITHIN the scan only (user, 25 Oct: across scans it is not
transferable - CT phase, MR scale and sequence, scanner): every class gets the same weak prior from the voxels being modelled
(scan_prior: their mean and covariance, strength beta0 = 1 voxel, nu0 = D + 2), so no class is told in advance what tumour or an
organ looks like; tumour and background are told apart by the spatial prior, the territories, the initialisation from q and the
coupling.

Tail weights (26 Oct, main.tex Eq. nu_prior): with nu_prior (a mixture of normals on x = log(1/nu) per class) every class's nu
is a MAP estimate (nu_map), updated once per iteration from E[u] at the previous nu; the log prior density then enters the ELBO
(MAP point estimate). Without it nu is fixed (or learned by maximum likelihood for tumour components with learn_nu).

ELBO: reported for the model WITHOUT the coupling term (independent labels with prior pi_v), evaluated at the fitted q. Any q gives a
lower bound on that model's log evidence, so ELBO(tumour + background) - ELBO(background only) on the same voxels is a consistent
approximation of the log Bayes factor (ratio of lower bounds; not exact)."""
import numpy as np
from scipy.special import digamma, gammaln

LOG2PI = np.log(2 * np.pi)
DT = np.float32       # dtype of the large voxel x class arrays (memory on large TCGA clusters; sums are taken in float64)


class NW:
    """Normal-Wishart prior / posterior on (mu, Lambda): mu | Lambda ~ N(m, (beta Lambda)^-1), Lambda ~ Wishart(W, nu)."""

    def __init__(self, m, beta, W, nu):
        self.m = np.asarray(m, np.float64); self.beta = float(beta); self.W = np.asarray(W, np.float64); self.nu = float(nu)
        self.D = self.m.size

    @classmethod
    def from_moments(cls, mean, cov, beta0=1.0, nu0=None, ridge=1e-6):
        mean = np.asarray(mean, np.float64); D = mean.size; nu0 = D + 2.0 if nu0 is None else float(nu0)
        cov = np.asarray(cov, np.float64) + ridge * np.eye(D) * max(1.0, float(np.trace(cov)) / D)
        return cls(mean, beta0, np.linalg.inv(cov) / nu0, nu0)

    def to_dict(self):
        return dict(m=self.m.tolist(), beta=self.beta, W=self.W.tolist(), nu=self.nu)

    @classmethod
    def from_dict(cls, d):
        return cls(d['m'], d['beta'], d['W'], d['nu'])

    def e_logdet(self):
        D = self.D
        return float(np.sum(digamma((self.nu + 1 - np.arange(1, D + 1)) / 2.0)) + D * np.log(2.0) + np.linalg.slogdet(self.W)[1])

    def log_B(self):
        D = self.D
        return float(-0.5 * self.nu * np.linalg.slogdet(self.W)[1] - 0.5 * self.nu * D * np.log(2.0) - 0.25 * D * (D - 1) * np.log(np.pi)
                     - np.sum(gammaln((self.nu + 1 - np.arange(1, D + 1)) / 2.0)))

    @staticmethod
    def posterior(prior, Y, w, r_sum):
        """posterior given weights w (= r * E[u]) and r_sum = sum r (Wishart dof)."""
        sw = float(w.sum())
        if sw <= 1e-10:
            return NW(prior.m, prior.beta, prior.W, prior.nu + r_sum)
        ybar = (w[:, None] * Y).sum(0) / sw
        Z = Y - ybar
        S = (Z * w[:, None]).T @ Z
        beta = prior.beta + sw
        m = (prior.beta * prior.m + sw * ybar) / beta
        dm = (ybar - prior.m)[:, None]
        Winv = np.linalg.inv(prior.W) + S + (prior.beta * sw / beta) * (dm @ dm.T)
        Winv = 0.5 * (Winv + Winv.T)
        return NW(m, beta, np.linalg.inv(Winv), prior.nu + r_sum)

    def kl_term(self, prior):
        """E_q[log p(mu, Lambda)] - E_q[log q(mu, Lambda)] (<= 0)"""
        D = self.D; ELL = self.e_logdet()
        dm = self.m - prior.m
        Elp = (0.5 * (D * np.log(prior.beta / (2 * np.pi)) + ELL - D * prior.beta / self.beta
                      - prior.beta * self.nu * float(dm @ self.W @ dm))
               + prior.log_B() + 0.5 * (prior.nu - D - 1) * ELL - 0.5 * self.nu * float(np.trace(np.linalg.inv(prior.W) @ self.W)))
        H_L = -self.log_B() - 0.5 * (self.nu - D - 1) * ELL + 0.5 * self.nu * D
        Elq = 0.5 * ELL + 0.5 * D * np.log(self.beta / (2 * np.pi)) - 0.5 * D - H_L
        return float(Elp - Elq)


def scan_prior(Y, w=None, beta0=1.0, nu0=None, min_w=20.0):
    """weak Normal-Wishart prior from voxels of THIS scan: (weighted) mean and covariance of Y. With weights w that sum to less than
    min_w voxels the unweighted moments of all of Y are used."""
    Y = np.asarray(Y, np.float64)
    if w is None or float(np.sum(w)) < min_w:
        w = np.ones(len(Y))
    w = np.asarray(w, np.float64); sw = w.sum(); mu = (w[:, None] * Y).sum(0) / sw
    Z = Y - mu; cov = (Z * w[:, None]).T @ Z / sw
    return NW.from_moments(mu, np.atleast_2d(cov), beta0=beta0, nu0=nu0)


def class_priors(Y, q, member, beta0=1.0, nu0=None):
    """within-scan priors, one per class, from the classes' own voxels on this scan (weak: beta0 voxels, nu0 = D + 2):
    tumour components  voxels weighted by the stage-1 extent marginal q;
    background class c voxels of its territory weighted by (1 - q) (member[:, c]: territory indicator; generic classes: all voxels).
    Returns (tumour prior, list of background priors)."""
    q = np.clip(np.asarray(q, np.float64), 0.0, 1.0)
    tp = scan_prior(Y, q, beta0, nu0)
    bps = [scan_prior(Y, member[:, c] * (1.0 - q), beta0, nu0) for c in range(member.shape[1])]
    return tp, bps


def _dir_e_log(a):
    a = np.asarray(a, np.float64)
    return digamma(a) - digamma(a.sum())


def _dir_kl_term(a_post, a_prior):
    a_post = np.asarray(a_post, np.float64); a_prior = np.asarray(a_prior, np.float64); El = _dir_e_log(a_post)
    logC = lambda a: gammaln(a.sum()) - gammaln(a).sum()
    return float(logC(a_prior) + ((a_prior - 1) * El).sum() - logC(a_post) - ((a_post - 1) * El).sum())


NU_RANGE = (2.0, 100.0)      # bounds of a learned Student-t nu (design values; 100 ~ Gaussian)
NU_TOL = 0.01                # MAP nu counts as settled when its relative change per iteration is below 1% (design value)


def nu_update(r, Eu, Elu, lo=NU_RANGE[0], hi=NU_RANGE[1]):
    """maximiser of the expected complete log likelihood in nu for one component (Student-t mixtures, e.g. Peel & McLachlan 2000):
    log(nu/2) - digamma(nu/2) = -(1 + sum r (E[log u] - E[u]) / sum r), solved on [lo, hi] (hi when the component is ~Gaussian)"""
    from scipy.optimize import brentq
    sr = float(np.sum(r))
    if sr <= 1e-9:
        return None
    c = -(1.0 + float(np.sum(r * (Elu - Eu))) / sr)
    g = lambda v: np.log(v / 2.0) - digamma(v / 2.0) - c
    if g(hi) >= 0:
        return hi
    if g(lo) <= 0:
        return lo
    return float(brentq(g, lo, hi, xtol=1e-4))


def log_mix(x, mix):
    """log of a mixture of normals sum_i w_i N(x; m_i, s_i^2), mix = [(w, m, s), ...]"""
    w = np.array([a[0] for a in mix]); m = np.array([a[1] for a in mix]); s_ = np.array([a[2] for a in mix])
    lp = np.log(np.maximum(w, 1e-300)) - 0.5 * np.log(2 * np.pi * s_ ** 2) - 0.5 * ((x - m) / s_) ** 2
    mx = lp.max()
    return float(mx + np.log(np.exp(lp - mx).sum()))


def nu_map(r, Eu, Elu, mix, lo=NU_RANGE[0], hi=NU_RANGE[1]):
    """MAP of nu under a prior on x = log(1/nu) (main.tex Eq. nu_update): maximises
    sum r [nu/2 log(nu/2) - lgamma(nu/2) + nu/2 (E log u - E u)] + log p(x) over nu in [lo, hi]. Without data (sum r ~ 0) the prior
    mode. Returns nu."""
    from scipy.optimize import minimize_scalar
    sr = float(np.sum(r)); a = float(np.sum(r * (Elu - Eu)))
    def negf(x):
        nu = np.exp(-x)
        return -(sr * (0.5 * nu * np.log(nu / 2.0) - gammaln(nu / 2.0)) + 0.5 * nu * a + log_mix(x, mix))
    res = minimize_scalar(negf, bounds=(np.log(1.0 / hi), np.log(1.0 / lo)), method='bounded', options=dict(xatol=1e-4))
    return float(np.exp(-res.x))


def nu_map_marginal(r, Delta, D, mix, lo=NU_RANGE[0], hi=NU_RANGE[1], nbins=256):
    """MAP of nu from the Student-t marginal (ECME step: u integrated out at the current q(mu, Lambda)), on the responsibility-
    weighted histogram of log Delta (256 bins; Delta = E[(y - mu)' Lambda (y - mu)]):
    max_nu sum r [lgamma((nu + D)/2) - lgamma(nu/2) + nu/2 log nu - (nu + D)/2 log(nu + Delta)] + log p(x), x = log(1/nu).
    Converges in far fewer iterations than the expected-complete-data update (nu_map), whose E[u] lags one step. Not used by
    Model (see Model._ell); kept for the comparison."""
    from scipy.optimize import minimize_scalar
    r = np.asarray(r, np.float64); keep = r > 1e-8
    if keep.sum() == 0:
        sr = 0.0; lc = np.zeros(1); w = np.zeros(1)
    else:
        lD = np.log(np.maximum(np.asarray(Delta, np.float64)[keep], 1e-12)); rk = r[keep]; sr = float(rk.sum())
        if lD.max() - lD.min() < 1e-9:
            lc = np.array([lD[0]]); w = np.array([sr])
        else:
            w, e = np.histogram(lD, bins=nbins, weights=rk); lc = 0.5 * (e[:-1] + e[1:]); nz = w > 0; w = w[nz]; lc = lc[nz]
    Dc = np.exp(lc)
    def negf(x):
        nu = np.exp(-x)
        ll = sr * (gammaln((nu + D) / 2.0) - gammaln(nu / 2.0) + 0.5 * nu * np.log(nu)) - 0.5 * (nu + D) * float((w * np.log(nu + Dc)).sum())
        return -(ll + log_mix(x, mix))
    res = minimize_scalar(negf, bounds=(np.log(1.0 / hi), np.log(1.0 / lo)), method='bounded', options=dict(xatol=1e-4))
    return float(np.exp(-res.x))


def _delta(Y, post):
    """E over q(mu, Lambda) of the Mahalanobis distance of every voxel to the class"""
    D = Y.shape[1]; Z = Y - post.m
    return D / post.beta + post.nu * np.einsum('nd,de,ne->n', Z, post.W, Z)


def student_terms(Y, post, nu, with_elu=False, Delta=None):
    """per-voxel expected log joint of y and u minus log q(u), for one class. Returns (ell, Eu) (and E[log u] when with_elu).
    Delta: precomputed _delta(Y, post) (saves the Mahalanobis step)."""
    D = Y.shape[1]
    Delta = _delta(Y, post) if Delta is None else Delta
    au = (nu + D) / 2.0; bu = (nu + Delta) / 2.0
    Eu = au / bu; Elu = digamma(au) - np.log(bu)
    ell = (0.5 * D * Elu + 0.5 * post.e_logdet() - 0.5 * D * LOG2PI - 0.5 * Eu * Delta
           + 0.5 * nu * np.log(nu / 2.0) - gammaln(nu / 2.0) + (nu / 2.0 - 1.0) * Elu - 0.5 * nu * Eu
           + au - np.log(bu) + gammaln(au) + (1.0 - au) * digamma(au))
    return (ell, Eu, Elu) if with_elu else (ell, Eu)


def parity(M):
    """checkerboard parity (i + j + k) mod 2 of the voxels of mask M, in M order"""
    idx = np.nonzero(np.asarray(M, bool))
    return ((idx[0] + idx[1] + idx[2]) % 2).astype(np.int8)


def build_edges(M, sp):
    """6-neighbour edges between voxels of mask M (flattened in M order). s = min(spacing) / spacing along the edge."""
    M = np.asarray(M, bool); pos = -np.ones(M.shape, np.int64); pos[M] = np.arange(int(M.sum()))
    sp = np.asarray(sp, float); I, J, S = [], [], []
    for a in range(3):
        lo = [slice(None)] * 3; hi = [slice(None)] * 3; lo[a] = slice(0, -1); hi[a] = slice(1, None)
        both = M[tuple(lo)] & M[tuple(hi)]
        I.append(pos[tuple(lo)][both]); J.append(pos[tuple(hi)][both])
        S.append(np.full(int(both.sum()), sp.min() / sp[a], np.float32))
    return (np.concatenate(I).astype(np.int64), np.concatenate(J).astype(np.int64), np.concatenate(S))


def kmeans(Y, k, seed=0, iters=25, max_n=50000):
    """deterministic k-means++ on standardised data; returns centroids in the original units"""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, np.float64)
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    mu, sd = Y.mean(0), Y.std(0) + 1e-6; Z = (Y - mu) / sd
    k = max(1, min(k, len(Z)))
    C = [Z[rng.integers(len(Z))]]
    for _ in range(1, k):
        d = np.min(((Z[:, None, :] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(Z[rng.choice(len(Z), p=d / d.sum())] if d.sum() > 0 else Z[rng.integers(len(Z))])
    C = np.array(C)
    for _ in range(iters):
        a = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (a == j).any(): C[j] = Z[a == j].mean(0)
    return C * sd + mu


def _nearest(Y, C):
    sd = Y.std(0) + 1e-6
    return np.argmin((((Y[:, None, :] - C[None]) / sd) ** 2).sum(-1), 1)


class Model:
    """One fit on a set of voxels.
    Y (n, D) features; pi (n,) spatial prior (ignored when tumour=False); types (n,) int; allowed (T, C) bool;
    tum_prior NW; bg_priors list of C NW; edges (I, J, S) or None (no coupling); init_q (n,) initial tumour probability."""

    def __init__(self, Y, pi, types, allowed, tum_prior, bg_priors, K=4, alpha0=0.25, nu=5.0, a_w=1.0, b_w=1.0, edges=None,
                 tumour=True, learn_nu=False, nu_prior=None):
        self.Y = np.asarray(Y, np.float64); self.n, self.D = self.Y.shape
        self.tumour = bool(tumour); self.K = K if tumour else 0
        self.pi = np.clip(np.asarray(pi, np.float64), 1e-6, 1 - 1e-6) if tumour else np.zeros(self.n)
        self.types = np.asarray(types, np.int64); self.allowed = np.asarray(allowed, bool); self.C = self.allowed.shape[1]
        self.tp = tum_prior; self.bp = list(bg_priors); assert len(self.bp) == self.C
        self.alpha0 = float(alpha0); self.a_w = float(a_w); self.b_w = float(b_w)
        # nu per class (tumour components first); learn_nu: the tumour components' nu are updated in the run (nu_update)
        self.nu_k = np.full(self.K + self.C, float(nu)) if np.isscalar(nu) else np.asarray(nu, np.float64)
        self.nu = float(np.median(self.nu_k)); self.learn_nu = bool(learn_nu)
        # nu_prior: per class (tumour components first, then background classes) a mixture [(w, m, s)] on x = log(1/nu); every
        # class's nu is then a MAP estimate updated in the run (nu_map), started at the prior's mean (main.tex Eq. nu_prior)
        self.nu_prior = None if nu_prior is None else list(nu_prior)
        if self.nu_prior is not None:
            assert len(self.nu_prior) == self.K + self.C, (len(self.nu_prior), self.K, self.C)
            self.nu_k = np.array([float(np.clip(np.exp(-sum(w * m for w, m, _ in mx) / sum(w for w, _, _ in mx)), *NU_RANGE))
                                  for mx in self.nu_prior])
        self.edges = edges if (tumour and edges is not None and len(edges[0])) else None
        self.mask_bg = self.allowed[self.types]                        # (n, C)

    # ------------------------------------------------------------------ initialisation
    def init(self, init_q, seed=0):
        n, K, C = self.n, self.K, self.C
        q = np.clip(np.asarray(init_q, np.float64), 0.01, 0.99) if self.tumour else np.zeros(n)
        R = np.zeros((n, K + C), DT)
        if self.tumour:
            sel = q >= 0.5
            if sel.sum() < 5 * K:
                sel = np.zeros(n, bool); sel[np.argsort(-q)[:max(5 * K, min(n, 50))]] = True
            Ct = kmeans(self.Y[sel], K, seed)
            if len(Ct) < K:
                Ct = np.vstack([Ct, np.repeat(Ct[-1:], K - len(Ct), 0)])
            a = _nearest(self.Y, Ct)
            R[np.arange(n), a] = q
        gen = [c for c in range(C) if self.allowed[:, c].all()]       # generic classes: allowed everywhere
        org = [c for c in range(C) if c not in gen]
        nb = self.mask_bg[:, org].any(1) if org else np.zeros(n, bool)
        if gen:
            sel = (q < 0.5) & ~nb
            if sel.sum() < 5 * len(gen):
                sel = q < 0.5
            if sel.sum() < len(gen):
                sel = np.ones(n, bool)
            Cg = kmeans(self.Y[sel], len(gen), seed + 1)
            if len(Cg) < len(gen):
                Cg = np.vstack([Cg, np.repeat(Cg[-1:], len(gen) - len(Cg), 0)])
            g = np.array(gen)[_nearest(self.Y, Cg)]
        share_org = np.zeros((n, C))
        if org:
            mo = self.mask_bg[:, org].astype(float); cnt = mo.sum(1, keepdims=True)
            share_org[:, org] = np.where(cnt > 0, mo / np.maximum(cnt, 1), 0.0)
        bgw = (1 - q)
        if gen:
            Rg = np.zeros((n, C)); Rg[np.arange(n), g] = 1.0
            in_org = nb[:, None]
            Rbg = np.where(in_org, 0.5 * share_org + 0.5 * Rg, Rg)
        else:
            Rbg = share_org
        R[:, K:] = bgw[:, None] * Rbg
        R /= R.sum(1, keepdims=True)
        self.R = R.astype(DT)
        self.Eu = np.ones((n, K + C), DT)
        return self

    # ------------------------------------------------------------------ updates
    def _m_step(self):
        K, C = self.K, self.C; R = self.R
        self.post = []
        for j in range(K + C):
            prior = self.tp if j < K else self.bp[j - K]
            w = R[:, j] * self.Eu[:, j]
            self.post.append(NW.posterior(prior, self.Y, w, float(R[:, j].sum())))
        if K:
            self.a_rho = self.alpha0 + R[:, :K].sum(0)
            self.Elrho = _dir_e_log(self.a_rho)
        T = self.allowed.shape[0]
        self.a_lam = np.where(self.allowed, 1.0, 0.0) + np.zeros((T, C))
        for t in range(T):
            sel = self.types == t
            if sel.any():
                self.a_lam[t] += np.where(self.allowed[t], R[sel, K:].sum(0), 0.0)
        self.Ellam = np.full((T, C), -np.inf)
        for t in range(T):
            al = self.allowed[t]
            if al.any():
                self.Ellam[t, al] = _dir_e_log(self.a_lam[t, al])

    def _ell(self, update_nu=True):
        L = np.empty((self.n, self.K + self.C), DT); Eu = np.empty_like(L)
        for j, post in enumerate(self.post):
            if self.nu_prior is not None:
                # terms at the current nu; nu is then updated for the next iteration (coordinate ascent on the expected complete
                # log likelihood, as every other update). The ECME step (nu_map_marginal) converges in fewer iterations but jumped to
                # nu ~ 100 before the edge voxels were absorbed and lost the tumour's boundary layer on the phantom (Dice 0.95 vs 0.999).
                L[:, j], Eu[:, j], Elu = student_terms(self.Y, post, self.nu_k[j], with_elu=True)
                if update_nu:
                    self.nu_k[j] = nu_map(self.R[:, j], Eu[:, j], Elu, self.nu_prior[j])
            elif self.learn_nu and j < self.K and update_nu:
                L[:, j], Eu[:, j], Elu = student_terms(self.Y, post, self.nu_k[j], with_elu=True)
                v = nu_update(self.R[:, j], Eu[:, j], Elu)
                if v is not None:
                    self.nu_k[j] = v
            else:
                L[:, j], Eu[:, j] = student_terms(self.Y, post, self.nu_k[j])
        return L, Eu

    def _coupling(self, p):
        if self.edges is None:
            return np.zeros(self.n), np.zeros(self.n), None
        I, J, S = self.edges
        d = p[I] * (1 - p[J]) + p[J] * (1 - p[I])
        Ew = self.a_w / (self.b_w + S * d)
        ws = Ew * S
        c1 = np.bincount(I, ws * (1 - p[J]), self.n) + np.bincount(J, ws * (1 - p[I]), self.n)
        c0 = np.bincount(I, ws * p[J], self.n) + np.bincount(J, ws * p[I], self.n)
        return c1, c0, Ew

    def prior_logits(self):
        K = self.K; lg = np.full((self.n, K + self.C), -np.inf, DT)
        if K:
            lg[:, :K] = np.log(self.pi)[:, None] + self.Elrho[None, :]
        lb = self.Ellam[self.types]
        lg[:, K:] = (np.log1p(-self.pi)[:, None] if self.tumour else 0.0) + lb
        return lg

    def run(self, max_iter=300, tol=1e-3, min_iter=5, verbose=False, parity=None):
        """coordinate ascent; label updates in two half-sweeps (checkerboard parity of the voxel grid) so that coupled neighbours are
        never updated together (parallel mean-field updates of a Potts model oscillate). parity: (n,) 0/1 array (from the grid)."""
        K = self.K
        par = np.zeros(self.n, np.int8) if parity is None else np.asarray(parity, np.int8)
        p_old = self.R[:, :K].sum(1) if K else np.zeros(self.n)
        self.converged = False; self.trace = []; stable = 0
        for it in range(1, max_iter + 1):
            self._m_step()
            L, self.Eu = self._ell()
            base = self.prior_logits() + L
            p_cur = p_old.copy()
            for ph in ((0, 1) if (K and self.edges is not None) else (None,)):
                sel = np.ones(self.n, bool) if ph is None else (par == ph)
                lg = base[sel].copy()
                if K and self.edges is not None:
                    c1, c0, _ = self._coupling(p_cur)
                    lg[:, :K] -= c1[sel, None]; lg[:, K:] -= c0[sel, None]
                lg -= lg.max(1, keepdims=True)
                Rn = np.exp(lg); Rn /= Rn.sum(1, keepdims=True)
                self.R[sel] = Rn
                if K:
                    p_cur[sel] = Rn[:, :K].sum(1)
            p = p_cur if K else np.zeros(self.n)
            dp = float(np.abs(p - p_old).max()) if K else 0.0
            if self.learn_nu and K:
                # learning nu: the components and their nu must also settle (with every voxel tumour, p alone never changes)
                Rk = np.asarray(self.R[:, :K], np.float64)
                if getattr(self, '_Rk_old', None) is not None:
                    dp = max(dp, float(np.abs(Rk - self._Rk_old).max()),
                             float(np.max(np.abs(self.nu_k[:K] - self._nu_old) / self._nu_old)))
                else:
                    dp = max(dp, 1.0)
                self._Rk_old = Rk; self._nu_old = self.nu_k[:K].copy()
            nu_settled = True
            if self.nu_prior is not None:
                # MAP tail weights: their relative change must also settle (each update uses E[u] at the previous nu, so nu moves to
                # its fixed point over several iterations; with every voxel tumour, p alone never changes)
                if getattr(self, '_nuall_old', None) is not None:
                    dnu = float(np.max(np.abs(self.nu_k - self._nuall_old) / self._nuall_old))
                    dp = max(dp, tol * dnu / NU_TOL); nu_settled = dnu < NU_TOL
                else:
                    dp = max(dp, 1.0); nu_settled = False
                self._nuall_old = self.nu_k.copy()
            flips = int(((p >= 0.5) != (p_old >= 0.5)).sum()) if K else 0
            stable = stable + 1 if (flips == 0 and float(np.abs(p - p_old).mean()) < 1e-4 and nu_settled) else 0
            self.trace.append(dp)
            if verbose:
                print(it, round(dp, 5), flips, flush=True)
            p_old = p
            # converged: every voxel moved < tol, or the refined tumour {p >= 1/2} unchanged for 10 iterations with mean change < 1e-4
            # (mixture components can drift slowly without changing the segmentation)
            if it >= min_iter and (dp < tol or (stable >= 10 and not self.learn_nu)):
                self.converged = True; break
        self.iters = it
        if K and self.edges is not None:
            _, _, self.Ew = self._coupling(p_old)
        self._m_step(); L, self.Eu = self._ell(update_nu=False)
        self.elbo = self.elbo_independent(L)
        return self

    # ------------------------------------------------------------------ outputs
    def elbo_independent(self, L=None):
        """ELBO of the model without coupling, at the current q"""
        if L is None:
            L, _ = self._ell(update_nu=False)
        R = self.R; lg = self.prior_logits()
        with np.errstate(invalid='ignore'):
            t = np.where(R > 0, R * (lg + L - np.log(np.maximum(R, 1e-30))), 0.0)
        e = float(t.sum(dtype=np.float64))
        for j, post in enumerate(self.post):
            e += post.kl_term(self.tp if j < self.K else self.bp[j - self.K])
        if self.K:
            e += _dir_kl_term(self.a_rho, np.full(self.K, self.alpha0))
        if self.nu_prior is not None:                                  # MAP nu: the log prior density enters the objective
            e += sum(log_mix(float(np.log(1.0 / v)), mx) for v, mx in zip(self.nu_k, self.nu_prior))
        for t_ in range(self.allowed.shape[0]):
            al = self.allowed[t_]
            if al.any() and (self.types == t_).any():
                e += _dir_kl_term(self.a_lam[t_, al], np.ones(int(al.sum())))
        return e

    def p(self):
        return self.R[:, :self.K].sum(1) if self.K else np.zeros(self.n)

    def atypicality(self):
        """1 / E[u_v] under the voxel's class posterior (main.tex: atypicality map): > 1 where a voxel is unlike the rest of its
        class (the Student-t scale shrinks its weight), about 1 for a typical voxel"""
        Eu = np.asarray(self.Eu, np.float64); R = np.asarray(self.R, np.float64)
        return 1.0 / np.maximum((R * Eu).sum(1), 1e-12)

    def components(self):
        """tumour-component posteriors given tumour, and their weights E[rho]"""
        if not self.K:
            return None, None
        Rt = self.R[:, :self.K]; s = Rt.sum(1, keepdims=True)
        return Rt / np.maximum(s, 1e-12), self.a_rho / self.a_rho.sum()

    def comp_summary(self, min_frac=0.05):
        """effective number of components: E[rho_k] >= min_frac (reported only; min_frac is a display threshold)"""
        _, w = self.components()
        if w is None:
            return dict(k_eff=0)
        return dict(k_eff=int((w >= min_frac).sum()), rho=[round(float(x), 4) for x in w],
                    means=[[round(float(v), 2) for v in self.post[k].m] for k in range(self.K)],
                    nu=[round(float(v), 2) for v in self.nu_k[:self.K]])


def _extended(allowed, bg_priors, tum_prior, K):
    """background with K extra classes allowed everywhere, with the tumour Normal-Wishart prior (free, not placed)"""
    allowed = np.asarray(allowed, bool)
    return np.hstack([allowed, np.ones((allowed.shape[0], K), bool)]), list(bg_priors) + [tum_prior] * K


def log_bayes_factor(Y, pi, types, allowed, tum_prior, bg_priors, init_q, K=4, alpha0=0.25, nu=5.0, a_w=1.0, b_w=1.0, edges=None,
                     parity=None, max_iter=200, nu_prior=None):
    """approximate log BF (tumour + background vs background only) on one set of voxels; returns (log_bf, tumour_model, bg_model).
    Matched flexibility: BOTH models get K extra background classes allowed everywhere, with the tumour Normal-Wishart prior, so
    tumour-like tissue can be explained as background in either; the tumour model adds K tumour classes PLACED by the spatial prior
    pi. The comparison then measures whether distinct tissue sits where the prior puts the tumour.
    nu_prior = (tumour mixtures [K], background mixtures [C], extra-class mixture): the extra classes take the given (normal
    fallback) mixture in both models, so the tail priors are matched too.
    (Phantom: without extra classes in the background model, a false anchor gave log BF > 0. R2Seg smoke: with extra classes in the
    background model only, a true kidney anchor in a large region gave log BF < 0, because the free classes also explained normal-tissue
    heterogeneity far from the anchor.)"""
    al2, bp2 = _extended(allowed, bg_priors, tum_prior, K)
    npt = npb = None
    if nu_prior is not None:
        tmix, bmix, xmix = nu_prior
        npt = list(tmix) + list(bmix) + [xmix] * K; npb = list(bmix) + [xmix] * K
    mt = Model(Y, pi, types, al2, tum_prior, bp2, K=K, alpha0=alpha0, nu=nu, a_w=a_w, b_w=b_w, edges=edges, tumour=True, nu_prior=npt)
    mt.init(init_q).run(max_iter=max_iter, parity=parity)
    mb = Model(Y, np.zeros(len(Y)), types, al2, tum_prior, bp2, K=K, alpha0=alpha0, nu=nu, edges=None, tumour=False, nu_prior=npb)
    mb.init(np.zeros(len(Y)))
    # warm start: background as in the tumour model; its tumour responsibilities go to the K extra (free) classes
    C0 = mt.C - K
    R = np.array(mt.R[:, mt.K:], np.float64)
    R[:, C0:] += mt.R[:, :mt.K]
    mb.R = (R / R.sum(1, keepdims=True)).astype(DT); mb.run(max_iter=max_iter)
    return mt.elbo - mb.elbo, mt, mb
