"""Ledger v4, R2: observation consistency phi_obs(Z, H; X) of the conditional anatomical reasoning model
p(Z, H | X, C, K) ∝ phi_obs · phi_tumor · phi_host (main.tex, Section 'Reasoning details', Eqs. obs_def - decision).

Every candidate part a that passed host and logic reasoning is grounded in the image, including parts both models propose.
Decision (MAP of the per-part posterior):
        z_a = 1  iff  BF1_a >= tau  and  BF2_a >= tau  and  logit(pi_a) + log BF2_a >= 0,      tau = 3
  pi_a  tumour prior from the support of BP and VT: both propose 0.9; one proposes and the other is silent there 0.75;
        one proposes and the other is active elsewhere 0.5; neither proposes (recall part) 0.25.
  BF1_a one-sided Bayes factor 'a differs from every structure of its local anatomy' (intersection-union test, see gate()):
        for each structure s of A(a), a is compared with the most similar compact block of s, calibrated by the same
        nearest-block distance among blocks of s; p = max_s p_s and BF1 = 1 / (-e p ln p) for p < 1/e, else 1 (Sellke).
  BF2_a two-sided Bayes factor 'a looks like the accepted tumour rather than its local anatomy': s(a) = mean voxel
        log p_T / p_A (Gaussian KDEs on rank features; voxel log-ratios clipped to +-5), calibrated by pseudo-parts drawn from R_T \\ a and from A(a):
        BF2 = g_T(s(a)) / g_A(s(a)). Two-fold cross-fitting: the densities are fitted on one spatial half of each reference
        and the pseudo-parts are drawn from the other half (no in-sample optimism); log BF2 = mean over the two folds.
Features: (intensity, local mean, local SD) in a 3x3 window of the acquisition plane, mapped to their empirical CDF over the
pooled reference voxels of the scan (invariant to monotone intensity transforms: CT and MR, any protocol).
References:
  A(a)  voxels within window_mm of a inside H_h or any TotalSegmentator structure, minus all candidate parts dilated by
        claim_margin_mm, eroded by one voxel in the acquisition plane (host parenchyma, vessels, neighbouring organs, wall and content of hollow
        organs: no organ exceptions).
  R_T   gated agreed parts of the same host (scan if < min_ref voxels), eroded in-plane by one voxel; if none, gated single-support
        parts; a is always left out. If no reference exists, BF2 is undefined and the part is decided by the gate alone.
Order: gate every part -> build R_T from gated parts -> two-sided decision (rejection, recovery, trimming; BP-only parts are
split once by the sign of the voxel log-ratio) -> recall (pi = 0.25) next to accepted lesions.
Pseudo-parts are spatially compact blocks (the k nearest reference voxels to a random seed, k = |a|, on a uniformly thinned
reference), so they reproduce the size and spatial correlation of a; no voxel independence is assumed.
All constants are fixed a priori and shared by every cancer type; nothing is tuned on stage or outcome."""
from __future__ import annotations
import numpy as np
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

R2P = dict(r2=True, r2_recall=True, r2_tau=3.0, pi_two=0.9, pi_silent=0.75, pi_single=0.5, pi_recall=0.25,
           r2_reach_mm=10.0, r2_window_mm=20.0, r2_claim_margin_mm=2.0, r2_min_ref=200, r2_min_atom_ml=0.5,
           r2_M=200, r2_ned=200, r2_kmax=2000, r2_nkde=3000, r2_grid=20, r2_logbf_clip=10.0, r2_lclip=5.0, r2_ref_cap=200000, r2_struct_min_ml=5.0, r2_contact_mm=3.0, r2_contact_frac=0.1, r2_gate_hyp='contact',
           r2_seed=0, img_root=None, fill_other=True)
RECOVERABLE = {'primary_extra_single_no_vt', 'primary_extra_unconfirmed', 'local_single_model', 'distant_needs_bp_and_site_prompt'}
logit = lambda q: float(np.log(q / (1 - q)))
ST = np.ones((3, 3, 3), bool)


# ----------------------------------------------------------------------------------------------------------- geometry
def _pad(sl, mm, sp, shape):
    pad = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    return tuple(slice(max(0, a.start - p), min(n, a.stop + p)) for a, p, n in zip(sl, pad, shape))


def _put(L, sl, ps):
    out = np.zeros(tuple(s.stop - s.start for s in ps), bool)
    out[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(sl, ps))] = L
    return out


def _features(img, ps, ax):
    X = np.asarray(img.dataobj[ps], dtype=np.float32)
    size = [3, 3, 3]; size[ax] = 1
    m = ndimage.uniform_filter(X, size=size)
    s = np.sqrt(np.maximum(ndimage.uniform_filter(X * X, size=size) - m * m, 0))
    return np.stack([X, m, s], -1)


def _within(M, mm, sp):
    return ndimage.distance_transform_edt(~M, sampling=sp) <= mm if M.any() else np.zeros(M.shape, bool)


class _Ref:
    """A reference region as voxel coordinates (mm, global) and feature rows."""

    def __init__(self, xyz, f, lab=None):
        self.xyz, self.f, self.lab = xyz, f, lab

    def __len__(self):
        return len(self.xyz)

    @staticmethod
    def cat(refs):
        refs = [r for r in refs if len(r)]
        if not refs:
            return _Ref(np.zeros((0, 3)), np.zeros((0, 3), np.float32))
        return _Ref(np.concatenate([r.xyz for r in refs]), np.concatenate([r.f for r in refs]))

    def halves(self):
        """Two spatial halves: split at the median of the axis of largest extent."""
        j = int(np.argmax(np.ptp(self.xyz, 0)))
        m = self.xyz[:, j] <= np.median(self.xyz[:, j])
        if m.all() or not m.any():
            m = np.arange(len(self)) % 2 == 0
        lab = (None, None) if self.lab is None else (self.lab[m], self.lab[~m])
        return _Ref(self.xyz[m], self.f[m], lab[0]), _Ref(self.xyz[~m], self.f[~m], lab[1])

    def sample(self, n, rng, exclude=None):
        idx = np.arange(len(self)) if exclude is None else np.setdiff1d(np.arange(len(self)), exclude, assume_unique=False)
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        return self.f[idx]

    def blocks(self, n_vox, M, rng, kmax):
        """M spatially compact pseudo-parts of n_vox voxels (k nearest voxels to a random seed on a thinned reference).
        Returns index arrays into this reference and whether the block size had to be capped."""
        r = min(1.0, kmax / max(n_vox, 1))
        n = len(self)
        sub = np.arange(n) if r >= 1 else rng.choice(n, max(2, int(round(n * r))), replace=False)
        k = max(1, int(round(n_vox * r)))
        capped = k > len(sub) // 2
        k = max(1, min(k, len(sub) // 2))
        tree = cKDTree(self.xyz[sub])
        seeds = rng.choice(len(sub), M, replace=len(sub) < M)
        _, ii = tree.query(self.xyz[sub][seeds], k=k)
        ii = np.asarray(ii).reshape(M, -1)
        return [sub[row] for row in ii], capped


# ------------------------------------------------------------------------------------------------------- statistics
def _cdf(pool):
    q = [np.sort(pool[:, j]) for j in range(pool.shape[1])]
    return lambda v: np.stack([np.searchsorted(qq, v[:, j], side='right') / len(qq) for j, qq in enumerate(q)], -1)


def _edist(x, y):
    return 2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()


def _sellke(pv):
    return 1.0 / (-np.e * pv * np.log(pv)) if pv < 1 / np.e else 1.0


def _sub(f, n, rng):
    return f[rng.choice(len(f), n, replace=False)] if len(f) > n else f


QS = np.array([.1, .25, .5, .75, .9])


def _desc(fr):
    """Block descriptor: deciles/quartiles of each rank feature (15 numbers); distance = mean absolute difference
    (an L1 / Wasserstein-type distance between marginal quantile functions)."""
    return np.quantile(fr, QS, axis=0).ravel()


def gate(af, axyz, n_vox, A, p, rng, vox_ml, hyp=None):
    """One-sided test of 'a differs from every structure of its local anatomy' -> (BF1, p, capped).
    Intersection-union test over the hypothesis structures s: the host parenchyma and each TotalSegmentator structure that
    a lies in or abuts (>= 10 % of the tissue within 3 mm of a), with at least struct_min_ml inside A (smaller fragments are
    dominated by partial-volume voxels). H0_s: 'a is tissue of s', i.e. a
    looks like some region of s. s is tiled by M compact blocks of the size of a (capped at |s|/8); the null statistic of
    block m is its distance to the most similar non-overlapping block of s, d_m. The statistic of a is the median, over
    compact sub-blocks of a of the same size, of the distance to the most similar block of s. Heterogeneous structures (the
    wall, air and content of hollow organs) are thereby handled: a region of s always finds a similar region of s.
    p_s = (1 + #{d_m >= d(a)}) / (M + 1); a must differ from all structures, so p = max_s p_s, BF1 = Sellke bound."""
    rank = _cdf(A.sample(20000, rng))
    ned, M = p['r2_ned'], p['r2_M']
    labs, cnt = np.unique(A.lab, return_counts=True)
    smin = max(p['r2_min_ref'], int(round(p['r2_struct_min_ml'] / vox_ml)))
    S = [(int(l), np.where(A.lab == l)[0]) for l, c in zip(labs, cnt) if c >= smin and (hyp is None or int(l) in hyp)]
    if not S:
        S = [(0, np.arange(len(A)))]
    aR = _Ref(axyz, af)
    worst, capped, det = None, False, []
    for l, ix in S:
        As = _Ref(A.xyz[ix], A.f[ix])
        nb = int(min(n_vox, len(ix) // 8))
        cap = nb < n_vox
        blks, _ = As.blocks(nb, M, rng, p['r2_kmax'])
        Dsc = np.stack([_desc(rank(_sub(As.f[b], ned, rng))) for b in blks])
        cen = np.stack([As.xyz[b].mean(0) for b in blks])
        rad = np.median([np.linalg.norm(As.xyz[b] - c, axis=1).max() for b, c in zip(blks, cen)])
        far = cdist(cen, cen) >= 2 * rad                       # non-overlapping pairs
        dd = cdist(Dsc, Dsc, 'cityblock') / Dsc.shape[1]
        dd[~far] = np.inf
        dm = dd.min(1)
        dm = dm[np.isfinite(dm)]
        if len(dm) < 20:
            continue                                           # structure too small to tile: no hypothesis
        if nb >= n_vox:
            sub = [np.arange(len(aR))]
        else:
            sub, _ = aR.blocks(nb, min(M, 25), rng, p['r2_kmax'])
        da = np.median([(np.abs(Dsc - _desc(rank(_sub(aR.f[b], ned, rng)))).mean(1)).min() for b in sub])
        ps_ = (1 + int((dm >= da).sum())) / (len(dm) + 1)
        det.append((l, len(ix), round(float(da), 4), round(float(np.quantile(dm, .95)), 4), round(ps_, 4)))
        if worst is None or ps_ > worst[0]:
            worst = (ps_, l)
        capped |= cap
    if worst is None:
        worst = (1.0, None)
    pv = worst[0]
    gate.last = dict(closest=worst[1], n_struct=len(det), nA=len(A), per_struct=det)
    return _sellke(pv), pv, capped


class _Table:
    """log p_T - log p_A of Gaussian KDEs (Scott) on rank features, tabulated on a G^3 grid of the rank cube."""

    def __init__(self, Tf, Af, rank, p, rng):
        jit = lambda a: a + rng.normal(0, 1e-3, a.shape)
        kt = stats.gaussian_kde(jit(rank(_sub(Tf, p['r2_nkde'], rng))).T)
        ka = stats.gaussian_kde(jit(rank(_sub(Af, p['r2_nkde'], rng))).T)
        G = p['r2_grid']
        c = (np.arange(G) + 0.5) / G
        grid = np.stack(np.meshgrid(c, c, c, indexing='ij'), 0).reshape(3, -1)
        ev = lambda k: np.concatenate([k.logpdf(grid[:, i:i + 500]) for i in range(0, grid.shape[1], 500)])
        # voxel log-ratios are clipped to +-lclip so that a few voxels in an empty tail of one density cannot dominate s(a)
        self.G, self.rank = G, rank
        self.tab = np.clip(ev(kt) - ev(ka), -p['r2_lclip'], p['r2_lclip']).reshape(G, G, G)

    def llr(self, f):
        b = np.clip((self.rank(f) * self.G).astype(int), 0, self.G - 1)
        return self.tab[b[:, 0], b[:, 1], b[:, 2]]


def _log_density_ratio(s, sT, sA):
    """log g_T(s) - log g_A(s) with 1-D Gaussian KDEs of the pseudo-part statistics."""
    def kde(v):
        v = np.asarray(v, float)
        if np.std(v) < 1e-6:
            v = v + np.random.default_rng(0).normal(0, 1e-3, v.shape)
        return stats.gaussian_kde(v)
    return float(kde(sT).logpdf([s])[0] - kde(sA).logpdf([s])[0])


def two_sided(af, n_vox, T, A, p, rng, folds=(0, 1), tables=None):
    """Two-sided test of 'a looks like R_T rather than A(a)' -> (log BF2, details). Cross-fitted over spatial halves."""
    rank = _cdf(np.concatenate([T.sample(10000, rng), A.sample(10000, rng)]))
    Th, Ah = T.halves(), A.halves()
    a_eval = _sub(af, p['r2_kmax'], rng)
    lbf, det, capped = [], [], False
    for k in folds:
        tab = tables[k] if tables else _Table(Th[k].f, Ah[k].f, rank, p, rng)
        s = float(tab.llr(a_eval).mean())
        o = 1 - k
        bT, cT = Th[o].blocks(n_vox, p['r2_M'] // len(folds), rng, p['r2_kmax'])
        bA, cA = Ah[o].blocks(n_vox, p['r2_M'] // len(folds), rng, p['r2_kmax'])
        sT = [float(tab.llr(Th[o].f[b]).mean()) for b in bT]
        sA = [float(tab.llr(Ah[o].f[b]).mean()) for b in bA]
        l = float(np.clip(_log_density_ratio(s, sT, sA), -p['r2_logbf_clip'], p['r2_logbf_clip']))
        lbf.append(l); capped |= cT or cA
        det.append(dict(s=round(s, 3), sT=round(float(np.median(sT)), 3), sA=round(float(np.median(sA)), 3), logbf=round(l, 3)))
    return float(np.mean(lbf)), dict(folds=det, capped=capped)


# --------------------------------------------------------------------------------------------------------- main step
def apply_r2(out, hosts, ctx, p):
    """Score and update lesions in place (masks, sizes, status). ctx: shape, sp, vox_ml, img, ax, k, bp_u, vt_u, other, ts_map."""
    shape, sp, vox_ml, img, ax = ctx['shape'], ctx['sp'], ctx['vox_ml'], ctx['img'], ctx['ax']
    bp_u, vt_u, other, md = ctx['bp_u'], ctx['vt_u'], ctx['other'], ctx['ts_map']
    sp = np.asarray(sp, float)
    claims = bp_u | vt_u
    ER = np.zeros((3, 3, 3), bool); _e = [slice(None)] * 3; _e[ax] = 1; ER[tuple(_e)] = True   # in-plane erosion (anisotropic voxels)
    min_atom = max(1, int(round(p['r2_min_atom_ml'] / vox_ml)))
    tau = p['r2_tau']; ltau = float(np.log(tau))
    rng = np.random.default_rng(p['r2_seed'])
    lp = dict(two=logit(p['pi_two']), silent=logit(p['pi_silent']), single=logit(p['pi_single']), recall=logit(p['pi_recall']))
    report = dict(n_lesions=0, n_parts=0, n_gated=0, n_rejected=0, n_recovered=0, trimmed_ml=0.0, recalled_ml=0.0)

    # ---- 1. candidate lesions and their parts ------------------------------------------------------------------------
    work = []
    for i, les in enumerate(out):
        if les['host'] is None or les['cls'] is None:
            continue
        cand = les['status'] == 'admitted' or (les['reason'] in RECOVERABLE and les['in_other'] < p['attach_other_max'])
        if not cand:
            continue
        h = hosts[les['host']]
        if not h['H'].any() or h['ref'] == 'kb_prior_region':
            les['r2'] = dict(testable=False, why='no_organ_reference'); continue
        ps = _pad(les['sl'], p['r2_reach_mm'] + p['r2_window_mm'], sp, shape)
        L = _put(les['L'], les['sl'], ps)
        bp = _put(les['bp'], les['sl'], ps); vt = _put(les['vt'], les['sl'], ps)
        Hm = h['H'][ps]
        anat = Hm | (md[ps] > 0)
        cl = claims[ps]
        excl = _within(cl | L, p['r2_claim_margin_mm'], sp)
        origin = np.array([s.start for s in ps], float)
        small = np.zeros(L.shape, bool)

        def comps(M, small=small):
            lab, n = ndimage.label(M, structure=ST)
            if not n:
                return []
            sz = np.bincount(lab.ravel())
            big = [j for j in range(1, n + 1) if sz[j] >= min_atom]
            small[(lab > 0) & ~np.isin(lab, big)] = True
            return [lab == j for j in big]
        parts = [dict(M=M, kind='two', origin='agreed') for M in comps(L & bp & vt)]
        # single-model parts: the other model is silent (no claim there: missing evidence) or active (claims elsewhere)
        env = h['env'][ps]
        oth = tuple(a for a in range(3) if a != ax)
        bshape = [1, 1, 1]; bshape[ax] = -1
        bp_act = np.broadcast_to((bp_u[ps] & env).any(axis=oth).reshape(bshape), L.shape)
        vo = L & vt & ~bp
        parts += [dict(M=M, kind='silent', origin='vt_only') for M in comps(vo & ~bp_act)]
        parts += [dict(M=M, kind='single', origin='vt_only') for M in comps(vo & bp_act)]
        vt_near = _within(vt_u[ps], p['r2_reach_mm'], sp)
        parts += [dict(M=M, kind='single' if (M & vt_near).any() else 'silent', origin='bp_only') for M in comps(L & bp & ~vt)]
        w = dict(i=i, les=les, h=h, ps=ps, L=L, bp=bp, vt=vt, Hm=Hm, anat=anat, excl=excl, origin=origin, small=small,
                 parts=parts, F=_features(img, ps, ax), cl=cl, env=env,
                 lab=np.where(Hm, -1, md[ps]).astype(np.int16))           # structure label: host -1, else TS label
        work.append(w); report['n_lesions'] += 1

    def ref_of(w, M):
        idx = np.argwhere(M)
        return _Ref((idx + w['origin']) * sp, w['F'][M], w['lab'][M])

    def local_anatomy(w, M, extra_excl=None):
        A = w['anat'] & ~w['excl'] & _within(M, p['r2_window_mm'], sp) & ~M
        if extra_excl is not None:
            A &= ~extra_excl
        return ref_of(w, ndimage.binary_erosion(A, structure=ER))

    def hypotheses(w, M):
        # H0 structures: the host parenchyma, and every structure that makes up >= contact_frac of the tissue in contact with
        # a (within contact_mm): a candidate could be mislabelled host tissue or tissue of a structure it lies in or abuts
        shell = _within(M, p['r2_contact_mm'], sp) & ~M
        lab = w['lab'][shell]
        lab = lab[lab != 0]
        hyp = {-1}
        if p['r2_gate_hyp'] == 'host':
            return hyp
        if len(lab):
            u, c = np.unique(lab, return_counts=True)
            hyp |= {int(x) for x, n in zip(u, c) if n >= p['r2_contact_frac'] * shell.sum()}
        return hyp

    def run_gate(w, M):
        A = local_anatomy(w, M)
        if len(A) < p['r2_min_ref']:
            return None, A
        bf1, pv, cap = gate(w['F'][M], (np.argwhere(M) + w['origin']) * sp, int(M.sum()), A, p, rng, vox_ml, hypotheses(w, M))
        return dict(bf1=round(bf1, 3), p=round(pv, 4), capped_gate=cap, **gate.last), A

    # ---- 2. one-sided gate for every part ----------------------------------------------------------------------------
    for w in work:
        for a in w['parts']:
            report['n_parts'] += 1
            g, A = run_gate(w, a['M'])
            a['A'] = A
            a['gate'] = g
            a['gated'] = g is not None and g['bf1'] >= tau
            report['n_gated'] += int(a['gated'])
            if a['gated']:
                E = ndimage.binary_erosion(a['M'], structure=ER)
                if E.sum() < 1:
                    E = a['M']
                r = ref_of(w, E)
                if len(r) > p['r2_ref_cap']:
                    j = rng.choice(len(r), p['r2_ref_cap'], replace=False); r = _Ref(r.xyz[j], r.f[j])
                a['ref'] = r

    # ---- 3. tumour reference R_T (gated parts; a left out) --------------------------------------------------------------
    def tumour_ref(host, exclude):
        gated = [(w['les']['host'], a) for w in work for a in w['parts'] if a['gated'] and a is not exclude]
        for kinds in (('two',), ('silent', 'single')):
            for same_host in (True, False):
                R = _Ref.cat([a['ref'] for hh, a in gated if a['kind'] in kinds and (hh == host or not same_host)])
                if len(R) >= p['r2_min_ref']:
                    return R, ('agreed' if kinds == ('two',) else 'single') + ('_host' if same_host else '_scan')
        return None, 'none'

    def decide(a, kind, T, A, af, n_vox):
        """Eq. decision; BF2 undefined without a tumour reference (gate alone)."""
        if T is None:
            a.update(bf2=None, logbf2=None, accept=bool(a['gated']), why='no_tumour_reference')
            return
        lbf, det = two_sided(af, n_vox, T, A, p, rng)
        acc = a['gated'] and lbf >= ltau and lp[kind] + lbf >= 0
        a.update(logbf2=round(lbf, 3), bf2=round(float(np.exp(lbf)), 3), two_sided=det, accept=bool(acc))

    # ---- 4. two-sided decision: rejection, recovery, trimming -----------------------------------------------------------
    for w in work:
        les = w['les']
        decided = []
        for a in w['parts']:
            if a['gate'] is None:                                    # local anatomy too small: untested, follows the lesion
                a.update(accept=None, why='small_local_anatomy'); decided.append(a); continue
            if not a['gated']:
                a.update(accept=False, why='no_one_sided_evidence'); decided.append(a); continue
            T, a['ref_kind'] = tumour_ref(les['host'], a)
            if a['origin'] == 'bp_only' and T is not None:
                # split the BP-only part once by the sign of the voxel log-ratio (trimming BP excess); each piece is decided
                rank = _cdf(np.concatenate([T.sample(10000, rng), a['A'].sample(10000, rng)]))
                tab = _Table(T.f, a['A'].f, rank, p, rng)
                vl = np.zeros(a['M'].shape, np.float32); vl[a['M']] = tab.llr(w['F'][a['M']])
                pieces = []
                for sgn in (vl > 0, vl <= 0):
                    lab, n = ndimage.label(a['M'] & sgn, structure=ST)
                    for j in range(1, n + 1):
                        Mj = lab == j
                        if Mj.sum() < min_atom:
                            w['small'] |= Mj; continue
                        pieces.append(Mj)
                if len(pieces) > 1:
                    for Mj in pieces:
                        b = dict(M=Mj, kind='single' if (Mj & _within(vt_u[w['ps']], p['r2_reach_mm'], sp)).any() else 'silent',
                                 origin='bp_only_split', ref_kind=a['ref_kind'])
                        g, A = run_gate(w, Mj)
                        b.update(gate=g, gated=g is not None and g['bf1'] >= tau)
                        if g is None:
                            b.update(accept=None, why='small_local_anatomy')
                        else:
                            decide(b, b['kind'], T, A, w['F'][Mj], int(Mj.sum()))
                        decided.append(b)
                    continue
            decide(a, a['kind'], T, a['A'], w['F'][a['M']], int(a['M'].sum()))
            decided.append(a)
        w['decided'] = decided
        keep = (w['L'] & ~w['bp'] & ~w['vt']) | w['small']           # R3 glue and untestable fragments follow the lesion
        for a in decided:
            if a['accept'] is None or a['accept']:
                keep |= a['M']
        core = np.zeros(keep.shape, bool)
        for a in decided:
            if a['accept']:
                core |= a['M']
        n_untested = sum(a['accept'] is None for a in decided)
        old = les['status']
        if core.sum() >= min_atom:
            if old != 'admitted':
                les['status'], les['reason'] = 'admitted', 'r2_image_evidence'; report['n_recovered'] += 1
        elif old == 'admitted' and (n_untested == 0 or not (keep & (w['bp'] | w['vt'])).any()):
            les['status'], les['reason'] = 'rejected', 'r2_no_image_evidence'; report['n_rejected'] += 1
        w['keep'], w['before'] = keep, old

    # ---- 5. recall next to accepted lesions (pi = 0.25) -----------------------------------------------------------------
    for w in work:
        les = w['les']; w['recalled'] = 0
        if not (p['r2_recall'] and les['status'] == 'admitted' and w['keep'].any()):
            continue
        keep = w['keep']
        T, rk = tumour_ref(les['host'], None)
        allowed = w['env'] & ~w['cl'] & ~(other[w['ps']] & ~w['Hm']) & _within(keep, p['r2_reach_mm'], sp) & ~keep
        if T is None or allowed.sum() < min_atom:
            continue
        # selection with fold-0 densities, test with fold 1 only (selection and test use disjoint reference halves)
        A0 = local_anatomy(w, keep)
        if len(A0) < p['r2_min_ref']:
            continue
        rank = _cdf(np.concatenate([T.sample(10000, rng), A0.sample(10000, rng)]))
        Th, Ah = T.halves(), A0.halves()
        sel = _Table(Th[0].f, Ah[0].f, rank, p, rng)
        vl = np.zeros(keep.shape, np.float32); vl[allowed] = sel.llr(w['F'][allowed])
        touch = ndimage.binary_dilation(keep, structure=ST)
        lab, n = ndimage.label(allowed & (vl > 0), structure=ST)
        sz = np.bincount(lab.ravel()) if n else []
        cands = [lab == j for j in range(1, n + 1) if sz[j] >= min_atom and ((lab == j) & touch).any()]
        if not cands:
            continue
        allc = np.zeros(keep.shape, bool)
        for M in cands:
            allc |= M
        allc_d = _within(allc, p['r2_claim_margin_mm'], sp)
        for M in cands:
            A = local_anatomy(w, M, extra_excl=allc_d)
            a = dict(M=M, kind='recall', origin='unclaimed', ref_kind=rk)
            if len(A) < p['r2_min_ref']:
                a.update(accept=False, why='small_local_anatomy'); w['decided'].append(a); continue
            bf1, pv, cap = gate(w['F'][M], (np.argwhere(M) + w['origin']) * sp, int(M.sum()), A, p, rng, vox_ml, hypotheses(w, M))
            a.update(gate=dict(bf1=round(bf1, 3), p=round(pv, 4), capped_gate=cap), gated=bf1 >= tau)
            if a['gated']:
                lbf, det = two_sided(w['F'][M], int(M.sum()), T, A, p, rng, folds=(1,))
                a.update(logbf2=round(lbf, 3), bf2=round(float(np.exp(lbf)), 3), two_sided=det,
                         accept=bool(lbf >= ltau and lp['recall'] + lbf >= 0))
            else:
                a.update(accept=False, why='no_one_sided_evidence')
            w['decided'].append(a)
            if a['accept']:
                w['keep'] |= M; w['recalled'] += int(M.sum())

    # ---- 6. re-assemble lesions ------------------------------------------------------------------------------------------
    for w in work:
        les, keep, ps = w['les'], w['keep'], w['ps']
        trimmed = int((w['L'] & ~keep).sum())
        if les['status'] == 'admitted' and keep.any():
            bb = ndimage.find_objects(keep.astype(np.int8))[0]
            nsl = tuple(slice(b.start + a.start, b.start + a.stop) for a, b in zip(bb, ps))
            les['L'] = keep[bb]; les['bp'] = w['bp'][bb] & les['L']; les['vt'] = w['vt'][bb] & les['L']; les['sl'] = nsl
            les['vox'] = int(les['L'].sum()); les['agree'] = int((les['bp'] & les['vt']).sum())
            ext = [(nsl[a].stop - nsl[a].start) * sp[a] for a in range(3) if a != ctx['k']]
            les['ld_mm'] = float(np.hypot(*ext))
            nz = int(les['L'].any(axis=tuple(a for a in range(3) if a != ctx['k'])).sum())
            les['n_slices'], les['z_mm'] = nz, round(nz * float(sp[ctx['k']]), 1)
            les['frac'] = {q: int((les['L'] & hh['env'][nsl]).sum()) / les['vox'] for q, hh in hosts.items()}
        rec = []
        for a in w['decided']:
            r = dict(kind=a['kind'], origin=a['origin'], ml=round(int(a['M'].sum()) * vox_ml, 2), accept=a.get('accept'),
                     bf1=(a.get('gate') or {}).get('bf1'), p1=(a.get('gate') or {}).get('p'), bf2=a.get('bf2'),
                     logbf2=a.get('logbf2'), ref=a.get('ref_kind'), why=a.get('why'), gate=a.get('gate'))
            if a.get('two_sided'):
                r['folds'] = a['two_sided']['folds']; r['capped'] = a['two_sided']['capped']
            rec.append(r)
        les['r2'] = dict(testable=True, before=w['before'], parts=rec, recalled_ml=round(w['recalled'] * vox_ml, 2),
                         trimmed_ml=round(trimmed * vox_ml, 2))
        report['trimmed_ml'] += trimmed * vox_ml; report['recalled_ml'] += w['recalled'] * vox_ml
    report['trimmed_ml'] = round(report['trimmed_ml'], 2); report['recalled_ml'] = round(report['recalled_ml'], 2)
    return report


def fill_other(md, labels_excluded, ax):
    """Other-organ region: every TotalSegmentator structure except labels_excluded, each filled slice-wise (acquisition
    plane) for enclosed holes; TS removes tumour tissue from organ masks, so an enclosed lesion is a hole."""
    out = np.zeros(md.shape, bool)
    excl = set(int(x) for x in labels_excluded)
    st = np.zeros((3, 3, 3), bool)
    idx = [1, 1, 1]
    for d in range(3):
        if d == ax:
            continue
        for o in (0, 2):
            j = list(idx); j[d] = o; st[tuple(j)] = True
    st[1, 1, 1] = True
    for lab, sl in enumerate(ndimage.find_objects(md), start=1):
        if sl is None or lab in excl:
            continue
        m = md[sl] == lab
        pw = [(0, 0)] * 3; pw[ax] = (1, 1)                         # isolate slices: the pad slices are not border-connected
        f = ndimage.binary_fill_holes(np.pad(m, pw), structure=st)
        cut = [slice(None)] * 3; cut[ax] = slice(1, -1)
        out[sl] |= f[tuple(cut)]
    return out
