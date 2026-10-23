"""Mean-field variational Bayes for stage 2 (main.tex Eqs. mrf, appearance, stage2_bf).

Labels per voxel: tumour components k = 1..K (sparse Dirichlet weights rho ~ Dir(alpha0)) or background classes c (organ normal
tissue or generic tissue; per-voxel allowed set given by the voxel's type = the organs whose territory contains it; mixing weights
lambda_type ~ Dir(1) over the allowed classes, learned per scan).
Prior on the tumour indicator t_v: pi_v (spatial prior) with a Potts coupling exp(-omega_vw s_vw 1[t_v != t_w]) on 6-neighbours,
omega_vw ~ Gamma(a_w, b_w) latent per edge. The Potts factor is used unnormalised (as a pairwise penalty); its posterior is
q(omega_vw) = Gamma(a_w, b_w + s_vw E[1[t_v != t_w]]), so E[omega] falls where the labels disagree, which the appearance drives.
Appearance: multivariate Student-t per class as a Gaussian scale mixture, y | z, u ~ N(mu, (u Lambda)^-1), u ~ Gamma(nu/2, nu/2),
conjugate Normal-Wishart priors on (mu, Lambda).

ELBO: reported for the model WITHOUT the coupling term (independent labels with prior pi_v), evaluated at the fitted q. Any q gives a
lower bound on that model's log evidence, so ELBO(tumour + background) - ELBO(background only) on the same voxels is a consistent
approximation of the log Bayes factor (ratio of lower bounds; not exact)."""
import numpy as np
from scipy.special import digamma, gammaln

LOG2PI = np.log(2 * np.pi)


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


def _dir_e_log(a):
    a = np.asarray(a, np.float64)
    return digamma(a) - digamma(a.sum())


def _dir_kl_term(a_post, a_prior):
    a_post = np.asarray(a_post, np.float64); a_prior = np.asarray(a_prior, np.float64); El = _dir_e_log(a_post)
    logC = lambda a: gammaln(a.sum()) - gammaln(a).sum()
    return float(logC(a_prior) + ((a_prior - 1) * El).sum() - logC(a_post) - ((a_post - 1) * El).sum())


def student_terms(Y, post, nu):
    """per-voxel expected log joint of y and u minus log q(u), for one class. Returns (ell, Eu)."""
    D = Y.shape[1]
    Z = Y - post.m
    maha = np.einsum('nd,de,ne->n', Z, post.W, Z)
    Delta = D / post.beta + post.nu * maha
    au = (nu + D) / 2.0; bu = (nu + Delta) / 2.0
    Eu = au / bu; Elu = digamma(au) - np.log(bu)
    ell = (0.5 * D * Elu + 0.5 * post.e_logdet() - 0.5 * D * LOG2PI - 0.5 * Eu * Delta
           + 0.5 * nu * np.log(nu / 2.0) - gammaln(nu / 2.0) + (nu / 2.0 - 1.0) * Elu - 0.5 * nu * Eu
           + au - np.log(bu) + gammaln(au) + (1.0 - au) * digamma(au))
    return ell, Eu


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
                 tumour=True):
        self.Y = np.asarray(Y, np.float64); self.n, self.D = self.Y.shape
        self.tumour = bool(tumour); self.K = K if tumour else 0
        self.pi = np.clip(np.asarray(pi, np.float64), 1e-6, 1 - 1e-6) if tumour else np.zeros(self.n)
        self.types = np.asarray(types, np.int64); self.allowed = np.asarray(allowed, bool); self.C = self.allowed.shape[1]
        self.tp = tum_prior; self.bp = list(bg_priors); assert len(self.bp) == self.C
        self.alpha0 = float(alpha0); self.nu = float(nu); self.a_w = float(a_w); self.b_w = float(b_w)
        self.edges = edges if (tumour and edges is not None and len(edges[0])) else None
        self.mask_bg = self.allowed[self.types]                        # (n, C)

    # ------------------------------------------------------------------ initialisation
    def init(self, init_q, seed=0):
        n, K, C = self.n, self.K, self.C
        q = np.clip(np.asarray(init_q, np.float64), 0.01, 0.99) if self.tumour else np.zeros(n)
        R = np.zeros((n, K + C))
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
        self.R = R
        self.Eu = np.ones((n, K + C))
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

    def _ell(self):
        L = np.empty((self.n, self.K + self.C)); Eu = np.empty_like(L)
        for j, post in enumerate(self.post):
            L[:, j], Eu[:, j] = student_terms(self.Y, post, self.nu)
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
        K = self.K; lg = np.full((self.n, K + self.C), -np.inf)
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
            flips = int(((p >= 0.5) != (p_old >= 0.5)).sum()) if K else 0
            stable = stable + 1 if (flips == 0 and float(np.abs(p - p_old).mean()) < 1e-4) else 0
            self.trace.append(dp)
            if verbose:
                print(it, round(dp, 5), flips, flush=True)
            p_old = p
            # converged: every voxel moved < tol, or the refined tumour {p >= 1/2} unchanged for 10 iterations with mean change < 1e-4
            # (mixture components can drift slowly without changing the segmentation)
            if it >= min_iter and (dp < tol or stable >= 10):
                self.converged = True; break
        self.iters = it
        if K and self.edges is not None:
            _, _, self.Ew = self._coupling(p_old)
        self._m_step(); L, self.Eu = self._ell()
        self.elbo = self.elbo_independent(L)
        return self

    # ------------------------------------------------------------------ outputs
    def elbo_independent(self, L=None):
        """ELBO of the model without coupling, at the current q"""
        if L is None:
            L, _ = self._ell()
        R = self.R; lg = self.prior_logits()
        with np.errstate(invalid='ignore'):
            t = np.where(R > 0, R * (lg + L - np.log(np.maximum(R, 1e-300))), 0.0)
        e = float(t.sum())
        for j, post in enumerate(self.post):
            e += post.kl_term(self.tp if j < self.K else self.bp[j - self.K])
        if self.K:
            e += _dir_kl_term(self.a_rho, np.full(self.K, self.alpha0))
        for t_ in range(self.allowed.shape[0]):
            al = self.allowed[t_]
            if al.any() and (self.types == t_).any():
                e += _dir_kl_term(self.a_lam[t_, al], np.ones(int(al.sum())))
        return e

    def p(self):
        return self.R[:, :self.K].sum(1) if self.K else np.zeros(self.n)

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
                    means=[[round(float(v), 2) for v in self.post[k].m] for k in range(self.K)])


def background_only(Y, types, allowed, tum_prior, bg_priors, K=4, alpha0=0.25, nu=5.0):
    """the 'background only' model for the Bayes factor, with matched flexibility: K extra background classes allowed everywhere,
    with the tumour components' Normal-Wishart prior and a Dirichlet(1) mixing weight. The two models then differ only in whether
    those K classes are placed by the spatial tumour prior (tumour model) or not (background model); a model with more components
    would otherwise win by flexibility alone (phantom test T3)."""
    allowed = np.asarray(allowed, bool)
    al2 = np.hstack([allowed, np.ones((allowed.shape[0], K), bool)])
    return Model(Y, np.zeros(len(Y)), types, al2, tum_prior, list(bg_priors) + [tum_prior] * K, K=K, alpha0=alpha0, nu=nu,
                 edges=None, tumour=False)


def log_bayes_factor(Y, pi, types, allowed, tum_prior, bg_priors, init_q, K=4, alpha0=0.25, nu=5.0, a_w=1.0, b_w=1.0, edges=None,
                     parity=None, max_iter=200):
    """approximate log BF (tumour + background vs background only) on one set of voxels; returns (log_bf, tumour_model, bg_model)"""
    mt = Model(Y, pi, types, allowed, tum_prior, bg_priors, K=K, alpha0=alpha0, nu=nu, a_w=a_w, b_w=b_w, edges=edges, tumour=True)
    mt.init(init_q).run(max_iter=max_iter, parity=parity)
    mb = background_only(Y, types, allowed, tum_prior, bg_priors, K=K, alpha0=alpha0, nu=nu)
    mb.init(np.zeros(len(Y)))
    # warm start: the K extra classes start from the tumour model's tumour responsibilities
    R = np.zeros((len(Y), mb.C)); C0 = mt.C
    R[:, :C0] = mt.R[:, mt.K:]; R[:, C0:] = mt.R[:, :mt.K]
    mb.R = R / R.sum(1, keepdims=True); mb.run(max_iter=max_iter)
    return mt.elbo - mb.elbo, mt, mb

