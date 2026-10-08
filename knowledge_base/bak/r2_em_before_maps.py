"""Ledger R2 as calibrated EM: posterior tumour inference from the image under the fixed priors of host and logic reasoning
(main.tex, Section 'Reasoning details', Eqs. part_posterior, tumor_prior, obs_def, bf, decision).

Stage 1 (the ledger, unchanged) fixes the candidates and their priors. Stage 2 (this module) infers the tumour posterior.
  candidates  the source components of the ledger: connected components of BP, of initial VT and of every VT tumour prompt
              (the ledger's claims). Never merged or split for inference.
  prior pi_c  0 if the candidate violates the anatomical logic (a dropped claim, or a lesion rejected for a logical reason);
              otherwise from the other model's state on c: proposes (covers >= 1/2 of c) 0.9, silent 0.75, active 0.5.
              Recall regions 0.25. Fixed throughout.
  normal model  label-conditional: every voxel has an anatomical label (hole-filled TotalSegmentator structure, else a host
              region H_hat, else unlabelled tissue); p0(f | label) from normal voxels of that label within window_mm of the
              candidates, all candidates dilated by margin_mm removed, eroded in-plane.
  tumour model  p1(f) from candidate voxels weighted by w_v = (max_{c contains v} q_c) x r_v, where r_v is the voxel
              responsibility of the tumour class in the mixture rho p1 + (1 - rho) p0(.|label) over candidate voxels (inner EM),
              so voxels the normal model explains do not contaminate the tumour model. No leave-one-out: with one dominant tumour, leaving it out leaves only its over-segmentation and
              small false positives. Instead, two-fold spatial cross-fitting: densities from one half of the box, the
              candidate's voxels and all pseudo-candidates from the other half, so no voxel is scored by a density fitted on
              itself. A candidate of normal tissue then gives lambda ~ 0 for itself and for normal blocks alike: BF ~ 1.
  features    intensity, 3x3 in-plane mean and SD, each mapped to its rank within the scan; densities are binned on a G^3 grid
              of the rank cube and Gaussian-smoothed (Scott bandwidth) - a kernel density estimate on a grid.
  evidence    s_c = mean over c of lambda(v) = log p1 - log p0(.|label(v)), clipped to +-5. g0 / g1 = distributions of s for M
              same-volume compact pseudo-candidates from normal tissue with c's label composition / from current tumour
              voxels; two-fold spatial cross-fitting (densities from one half, pseudo-candidates from the other).
              BF_c = g1(s_c) / g0(s_c), log clipped to +-10; BF = 1 when the test has no power (too few reference voxels).
  EM          (damped: posterior log-odds updated as eta <- (1 - d_t) eta + d_t (logit pi + log BF), d_t = 0.5 for the
              first 5 iterations, then 1/(t - 3): a running average, so oscillating tumour models are averaged out)
              q^(0) = pi; M-step: tumour model from weights; E-step: q = sigmoid(logit pi + log BF) for every candidate, then
              recall regions regenerated from the current tumour and scored the same way (selection-aware null). Stop when
              the accepted set is unchanged and max |dq| < tol, or after max_iter. MAP: z = 1 iff q >= 0.5.
  output      ledger lesions keep their identity; mask = lesion n accepted candidates, plus accepted recall regions touching it.
All constants are fixed a priori and shared by every cancer type."""
from __future__ import annotations
import numpy as np
from scipy import ndimage, stats
from scipy.spatial import cKDTree

EMP = dict(r2_mode='em', em_G=24, em_window_mm=20.0, em_margin_mm=2.0, em_reach_mm=10.0, em_min_ref=200, em_min_ml=0.5,
           em_M=200, em_kmax=500, em_lclip=5.0, em_bfclip=10.0, em_max_iter=30, em_damp=0.5, em_damp_iters=5, em_inner=5, em_tol=0.01, em_eps=1e-3,
           em_pi_two=0.9, em_pi_silent=0.75, em_pi_single=0.5, em_pi_recall=0.25, em_seed=0, em_max_box_vox=45e6)
RECOVERABLE = {'primary_extra_single_no_vt', 'primary_extra_unconfirmed', 'local_single_model', 'distant_needs_bp_and_site_prompt'}
logit = lambda q: float(np.log(q / (1 - q)))
sig = lambda x: float(1 / (1 + np.exp(-x)))
ST = np.ones((3, 3, 3), bool)


def _within(M, mm, sp):
    return ndimage.distance_transform_edt(~M, sampling=sp) <= mm if M.any() else np.zeros(M.shape, bool)


def _put(B, sl, crop, shape_B):
    """Crop (global slice sl) into box B; returns a bool array of B's shape."""
    out = np.zeros(shape_B, bool)
    lo = [max(a.start, b.start) for a, b in zip(sl, B)]
    hi = [min(a.stop, b.stop) for a, b in zip(sl, B)]
    if any(l >= h for l, h in zip(lo, hi)):
        return out
    out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, B))] = \
        crop[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, sl))]
    return out


def fill_labels(md, ax):
    """Label map with every TotalSegmentator structure filled slice-wise (acquisition plane) for enclosed holes; filled
    voxels take the structure's label only where no other structure is present."""
    out = md.astype(np.int32).copy()
    st = np.zeros((3, 3, 3), bool)
    for d in range(3):
        if d != ax:
            for o in (0, 2):
                j = [1, 1, 1]; j[d] = o; st[tuple(j)] = True
    st[1, 1, 1] = True
    for lab, sl in enumerate(ndimage.find_objects(md), start=1):
        if sl is None:
            continue
        m = md[sl] == lab
        pw = [(0, 0)] * 3; pw[ax] = (1, 1)
        f = ndimage.binary_fill_holes(np.pad(m, pw), structure=st)
        cut = [slice(None)] * 3; cut[ax] = slice(1, -1)
        f = f[tuple(cut)] & (out[sl] == 0)
        out[sl][f] = lab
    return out


class _Pool:
    """Reference voxels (flat indices into the box) with physical coordinates; thinned trees for compact blocks."""

    def __init__(self, idx, xyz, rng, ratios=(1, 4, 16, 64, 256, 1024)):
        self.idx, self.xyz = idx, xyz
        self.trees = {}
        for r in ratios:
            n = len(idx) // r
            if n < 2 and r > 1:
                break
            sub = np.arange(len(idx)) if r == 1 else rng.choice(len(idx), n, replace=False)
            self.trees[r] = (sub, cKDTree(xyz[sub]))

    def __len__(self):
        return len(self.idx)

    def blocks(self, n_vox, M, rng, kmax, seeds_near=None, seed_p=None):
        """M compact blocks of n_vox voxels: the k nearest points to a random seed on a thinned pool (k <= kmax).
        Returns flat-index arrays (thinned samples representing blocks of n_vox voxels)."""
        if len(self) < 2:
            return []
        n_vox = max(1, min(int(n_vox), len(self) // 2))
        r = 1
        for rr in sorted(self.trees):
            r = rr
            if n_vox / rr <= kmax:
                break
        sub, tree = self.trees[r]
        k = max(1, min(int(round(n_vox / r)), len(sub) // 2))
        if seed_p is not None:
            seeds = self.xyz[rng.choice(len(self.idx), M, p=seed_p)]
        elif seeds_near is not None and len(seeds_near):
            seeds = self.xyz[rng.choice(seeds_near, M)]
        else:
            seeds = self.xyz[sub[rng.integers(len(sub), size=M)]]
        _, ii = tree.query(seeds, k=k)
        ii = np.asarray(ii).reshape(M, -1)
        return [self.idx[sub[row]] for row in ii]

    def near(self, centre, radius):
        sub, tree = self.trees[1]
        return sub[tree.query_ball_point(centre, radius)]


def _density(bins, G, w=None, eps=1e-3):
    """Binned Gaussian KDE on the G^3 rank grid (Scott bandwidth); returns log density on the grid."""
    h = np.bincount(bins, weights=w, minlength=G ** 3).astype(np.float64)
    tot = h.sum()
    if tot <= 0:
        return None
    n_eff = tot ** 2 / (np.sum((w if w is not None else np.ones(len(bins))) ** 2) + 1e-12)
    sd_bins = G / np.sqrt(12.0)                        # rank features are ~uniform on [0, 1]
    sigma = max(0.5, (max(n_eff, 2) ** (-1.0 / 7.0)) * sd_bins)
    d = ndimage.gaussian_filter(h.reshape(G, G, G), sigma, mode='reflect')
    d = d / d.sum()
    return np.log((1 - eps) * d + eps / G ** 3).ravel()


def _accepted(c, out):
    """MAP label: q > 1/2; an exact tie (pi = 1/2 and no image evidence) keeps the stage-1 decision of its lesion."""
    if abs(c['q'] - 0.5) < 1e-9:
        return 'lesion' in c and out[c['lesion']]['status'] == 'admitted'
    return c['q'] > 0.5


def _kde1(v):
    v = np.asarray(v, float)
    if len(v) < 5:
        return None
    if np.std(v) < 1e-6:
        v = v + np.random.default_rng(0).normal(0, 1e-3, v.shape)
    return stats.gaussian_kde(v)


def apply_r2_em(out, hosts, ctx, p):
    shape, sp, vox_ml, img, ax = ctx['shape'], np.asarray(ctx['sp'], float), ctx['vox_ml'], ctx['img'], ctx['ax']
    md, other, claims = ctx['ts_map'], ctx['other'], ctx['claims']
    bp_u, vt_u = ctx['bp_u'], ctx['vt_u']
    G = p['em_G']; rng = np.random.default_rng(p['em_seed'])
    min_vox = max(1, int(round(p['em_min_ml'] / vox_ml)))
    lp = dict(two=logit(p['em_pi_two']), silent=logit(p['em_pi_silent']), single=logit(p['em_pi_single']),
              recall=logit(p['em_pi_recall']))
    report = dict(n_candidates=len(claims), n_feasible=0, n_accepted=0, n_rejected=0, n_recovered_lesions=0,
                  n_rejected_lesions=0, recalled_ml=0.0, iterations=0, converged=False)

    # ---- lesion map and per-claim prior (stage 1 output, fixed) ---------------------------------------------------------
    lesmap = np.zeros(shape, np.int16)
    for i, les in enumerate(out):
        lesmap[les['sl']][les['L']] = i + 1

    def feasible_lesion(les):
        if les['host'] is None or les['cls'] is None:
            return False, 'no_host'
        if les['status'] == 'admitted':
            return True, None
        if les['reason'] in RECOVERABLE:              # support-based rejection: soft prior (7 Oct: no extra logic check here)
            return True, None
        return False, les['reason']

    cands = []
    for ci, c in enumerate(claims):
        rec = dict(ci=ci, rater=c['rater'], source=c['source'], ml=round(c['vox'] * vox_ml, 2), sl=c['sl'], crop=c['crop'])
        if c.get('drop'):
            rec.update(pi=0.0, why=c['drop']); cands.append(rec); continue
        lm = lesmap[c['sl']][c['crop']]
        lm = lm[lm > 0]
        if not len(lm):
            rec.update(pi=0.0, why='no_lesion'); cands.append(rec); continue
        li = int(np.bincount(lm).argmax()) - 1
        ok, why = feasible_lesion(out[li])
        rec['lesion'] = li
        if not ok:
            rec.update(pi=0.0, why=why); cands.append(rec); continue
        # other model's state on c
        oth = vt_u if c['rater'] == 'BP' else bp_u
        cov = int((oth[c['sl']] & c['crop']).sum()) / c['vox']
        if cov >= 0.5:
            state = 'two'
        elif c['rater'] == 'BP':                        # VT is 3D: silent when nothing within reach
            ps = tuple(slice(max(0, s.start - int(np.ceil(p['em_reach_mm'] / q)) - 1), min(n, s.stop + int(np.ceil(p['em_reach_mm'] / q)) + 1))
                       for s, q, n in zip(c['sl'], sp, shape))
            loc = _put(ps, c['sl'], c['crop'], tuple(s.stop - s.start for s in ps))
            state = 'single' if (vt_u[ps] & _within(loc, p['em_reach_mm'], sp)).any() else 'silent'
        else:                                           # BP is 2D: silent on c's acquisition slices within the host envelope
            h = hosts[out[li]['host']]
            zs = np.where(c['crop'].any(axis=tuple(a for a in range(3) if a != ax)))[0] + c['sl'][ax].start
            sl2 = list(h['env'].sl) if h['env'].any() else [slice(0, n) for n in shape]
            act = False
            for z in zs:
                ix = list(sl2); ix[ax] = slice(int(z), int(z) + 1); ix = tuple(ix)
                if (bp_u[ix] & h['env'][ix]).any():
                    act = True; break
            state = 'single' if act else 'silent'
        rec.update(pi={'two': p['em_pi_two'], 'silent': p['em_pi_silent'], 'single': p['em_pi_single']}[state], state=state)
        cands.append(rec)
    feas = [c for c in cands if c['pi'] > 0]
    report['n_feasible'] = len(feas)
    if not feas:
        report['note'] = 'no feasible candidate'
        return report

    # ---- working box: feasible candidates + window + reach --------------------------------------------------------------
    U = np.zeros(shape, bool)
    for c in feas:
        U[c['sl']] |= c['crop']
    bb = ndimage.find_objects(U.astype(np.int8))[0]
    pad = [int(np.ceil((p['em_window_mm'] + p['em_reach_mm']) / q)) + 1 for q in sp]
    B = tuple(slice(max(0, b.start - q), min(n, b.stop + q)) for b, q, n in zip(bb, pad, shape))
    shB = tuple(s.stop - s.start for s in B)
    if np.prod(shB) > p['em_max_box_vox']:
        report['note'] = f'box too large {shB}'
        return report
    org = np.array([s.start for s in B], float)
    # features -> rank bins (flat index on the G^3 grid); features themselves are not kept
    X = np.asarray(img.dataobj[B], dtype=np.float32)
    size = [3, 3, 3]; size[ax] = 1
    m = ndimage.uniform_filter(X, size=size)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(X * X, size=size) - m * m, 0))
    samp = rng.choice(X.size, min(X.size, 200000), replace=False)
    BIN = np.zeros(shB, np.int32)
    for j, A in enumerate((X, m, sd)):
        edges = np.quantile(A.ravel()[samp], np.arange(1, G) / G)
        BIN = BIN * G + np.searchsorted(edges, A, side='right').astype(np.int32)
    del X, m, sd
    BIN = BIN.ravel()
    # labels: hole-filled TS structures, else host region, else unlabelled (0)
    LAB = fill_labels(md[B], ax)
    hostlab = {}
    for k_, (key, h) in enumerate(hosts.items()):
        if h['H'].any():
            Hb = h['H'][B]
            LAB[(LAB == 0) & Hb] = 1000 + k_
            hostlab[key] = 1000 + k_
    LAB = LAB.ravel()
    # candidate voxels (flat) in the box
    for c in feas:
        Mc = _put(B, c['sl'], c['crop'], shB)
        c['idx'] = np.flatnonzero(Mc)
        c['xyz'] = (np.argwhere(Mc) + org) * sp
        c['centre'] = c['xyz'].mean(0); c['radius'] = float(np.linalg.norm(c['xyz'] - c['centre'], axis=1).max())
    U_B = U[B]; del U
    claims_all = np.zeros(shB, bool)
    for c in cands:
        claims_all |= _put(B, c['sl'], c['crop'], shB)
    # normal pools per label (window around the candidates, claims dilated removed, in-plane erosion)
    normal = ~_within(claims_all, p['em_margin_mm'], sp) & _within(U_B, p['em_window_mm'], sp)
    ER = np.zeros((3, 3, 3), bool); e = [slice(None)] * 3; e[ax] = 1; ER[tuple(e)] = True
    LABg = LAB.reshape(shB)
    normal_lab = {}
    for l in np.unique(LABg[normal]):
        Ml = ndimage.binary_erosion(normal & (LABg == l), structure=ER)
        if Ml.sum() >= p['em_min_ref']:
            normal_lab[int(l)] = np.flatnonzero(Ml)
    del normal
    xyz_of = lambda idx: (np.stack(np.unravel_index(idx, shB), -1) + org) * sp

    # spatial halves (one split for the whole box, along its longest physical axis)
    j = int(np.argmax(np.array(shB) * sp))
    cut = shB[j] // 2
    half_of = lambda idx: (np.unravel_index(idx, shB)[j] >= cut).astype(np.int8)
    N0 = {}                                             # label -> fold -> (log density, pool)
    for l, idx in normal_lab.items():
        hv = half_of(idx)
        N0[l] = {}
        for k in (0, 1):
            ik = idx[hv == k]
            if len(ik) >= p['em_min_ref']:
                N0[l][k] = (_density(BIN[ik], G, eps=p['em_eps']), _Pool(ik, xyz_of(ik), rng))
    allN = {l: _density(BIN[idx], G, eps=p['em_eps']) for l, idx in normal_lab.items()}

    def lam_table(logp1, l, k):
        """lambda on the grid for label l, fold k (None = no normal model)."""
        src = allN.get(l) if k is None else (N0.get(l, {}).get(k) or (None,))[0]
        if src is None or logp1 is None:
            return None
        return np.clip(logp1 - src, -p['em_lclip'], p['em_lclip'])

    def lam(idx, logp1, k, cache):
        out_ = np.full(len(idx), np.nan)
        labs = LAB[idx]
        for l in np.unique(labs):
            if (l, k) not in cache:
                cache[(l, k)] = lam_table(logp1, int(l), k)
            t = cache[(l, k)]
            if t is not None:
                sel = labs == l
                out_[sel] = t[BIN[idx[sel]]]
        return out_

    # ---- EM ---------------------------------------------------------------------------------------------------------------
    for c in feas:
        c['q'] = c['pi']; c['hv'] = half_of(c['idx'])
    recall = []
    near10 = _within(U_B, p['em_reach_mm'], sp).ravel()
    env_all = {key: (h['env'][B].ravel() if h['env'].any() else np.zeros(int(np.prod(shB)), bool)) for key, h in hosts.items()}
    oth_B = other[B].ravel()
    H_B = {key: (h['H'][B].ravel() if h['H'].any() else None) for key, h in hosts.items()}
    claims_flat = claims_all.ravel()
    resp = {}
    g0_cache = {}                                       # candidate -> fold -> list of pseudo-candidate index arrays

    def weights():
        """M-step weights: w_v = (max posterior of the candidates covering v) x r_v, where r_v is the voxel responsibility
        of the tumour component in the two-class mixture rho p1 + (1 - rho) p0(.|label) over candidate voxels. Voxels a
        candidate covers but that the normal model of their label explains (over-segmentation, a candidate of normal tissue)
        receive low r_v and do not contaminate the tumour model."""
        wq = np.zeros(int(np.prod(shB)), np.float32)
        for c in feas + recall:
            np.maximum.at(wq, c['idx'], c['q'])
        U_ = np.flatnonzero(wq > 0)
        if not len(U_):
            return wq
        lp0 = np.full(len(U_), np.nan)
        labs = LAB[U_]
        for l in np.unique(labs):
            t = allN.get(int(l))
            if t is not None:
                sel = labs == l
                lp0[sel] = t[BIN[U_[sel]]]
        r = np.ones(len(U_))
        rho = 0.5
        for _ in range(p['em_inner']):
            lp1 = _density(BIN[U_], G, w=(wq[U_] * r).astype(np.float64), eps=p['em_eps'])
            if lp1 is None:
                break
            a1 = np.log(rho) + lp1[BIN[U_]]
            a0 = np.log(1 - rho) + lp0
            r = np.where(np.isnan(a0), 1.0, 1.0 / (1.0 + np.exp(np.clip(np.nan_to_num(a0) - a1, -50, 50))))
            rho = float(np.clip((wq[U_] * r).sum() / wq[U_].sum(), 0.05, 0.95))
        w = np.zeros_like(wq)
        w[U_] = wq[U_] * r
        R_ = np.zeros_like(wq); R_[U_] = r
        resp['tumour_pool'] = np.flatnonzero(w > 0.05)       # g1: candidate voxels; seeds drawn proportional to w
        resp['rho'] = rho; resp['mean_r'] = float(r.mean())
        return w

    def tumour_logp(w, k, leave=None):
        tp = np.flatnonzero(w > 0)
        if leave is not None:
            tp = np.setdiff1d(tp, leave, assume_unique=True)
        tp = tp[half_of(tp) == k] if k is not None else tp
        if w[tp].sum() < p['em_min_ref'] * 0.5:
            return None, tp
        return _density(BIN[tp], G, w=w[tp].astype(np.float64), eps=p['em_eps']), tp

    def evidence(c, w, selection=None):
        """log BF for candidate/region c (two folds), with pseudo-candidates; selection = per-voxel mask function for
        recall (selection-aware null)."""
        lbf, det = [], []
        sid = c['ci'] if 'ci' in c else int(c['idx'][0])        # deterministic pseudo-candidates: EM is a fixed map
        for k in (0, 1):
            o = 1 - k
            lrng = np.random.default_rng([p['em_seed'], sid, k])
            # cross-fitting: densities from half k; the candidate's voxels and all pseudo-candidates from half o
            logp1, _ = tumour_logp(w, k)
            if logp1 is None:
                continue
            idx_o = c['idx'][half_of(c['idx']) == o]
            if len(idx_o) < 20:
                continue
            n_o = len(idx_o)
            cache = {}
            la = lam(idx_o, logp1, k, cache)
            if selection is not None:
                la = la[selection(idx_o)]
            if len(la) == 0 or np.isnan(la).mean() > 0.5:
                continue
            s = float(np.nanmean(la))
            labs, cnt = np.unique(LAB[idx_o], return_counts=True)
            frac = {int(l): n_ / n_o for l, n_ in zip(labs, cnt)}
            # g0: normal pseudo-candidates with the same label composition, near c, from half o
            key = (sid, o, 'r' if 'ci' not in c else 'c')
            if key not in g0_cache:
                parts = []
                for l, f in frac.items():
                    if l not in N0 or o not in N0[l] or f < 0.05:
                        continue
                    pool = N0[l][o][1]
                    near = pool.near(c['centre'], c['radius'] + p['em_window_mm'])
                    parts.append(pool.blocks(f * n_o, p['em_M'], lrng, p['em_kmax'], seeds_near=near))
                g0_cache[key] = [np.concatenate([pp[m_] for pp in parts]) for m_ in range(p['em_M'])] if parts else []
            s0 = []
            for b in g0_cache[key]:
                lb = lam(b, logp1, k, cache)
                if selection is not None:
                    lb = lb[selection(b)]
                if len(lb) and np.isfinite(lb).any():
                    s0.append(float(np.nanmean(lb)))
            # g1: current tumour voxels (q >= 0.5) of half o
            tp = resp.get('tumour_pool', np.zeros(0, int))
            tp = tp[half_of(tp) == o]
            s1 = []
            if len(tp) >= p['em_min_ref'] and w[tp].sum() > 0:
                pool = _Pool(tp, xyz_of(tp), lrng, ratios=(1, 4, 16, 64))
                sp_ = w[tp].astype(np.float64); sp_ /= sp_.sum()
                for b in pool.blocks(n_o, p['em_M'], lrng, p['em_kmax'], seed_p=sp_):
                    lb = lam(b, logp1, k, cache)
                    if selection is not None:
                        lb = lb[selection(b)]
                    if len(lb) and np.isfinite(lb).any():
                        s1.append(float(np.nanmean(lb)))
            k0, k1 = _kde1(s0), _kde1(s1)
            if k0 is None or k1 is None:
                det.append(dict(fold=k, s=round(s, 3), n0=len(s0), n1=len(s1), logbf=None))
                continue
            l_ = float(np.clip(k1.logpdf([s])[0] - k0.logpdf([s])[0], -p['em_bfclip'], p['em_bfclip']))
            lbf.append(l_)
            det.append(dict(fold=k, s=round(s, 3), s0=round(float(np.median(s0)), 3), s1=round(float(np.median(s1)), 3), logbf=round(l_, 3)))
        return (float(np.mean(lbf)) if lbf else 0.0), det, bool(lbf)

    hist = []
    for it in range(p['em_max_iter']):
        w = weights()                                   # M-step input (the tumour model is built inside evidence())
        old = {id(c): c['q'] for c in feas}
        acc_old = {id(c) for c in feas if _accepted(c, out)}
        for c in feas:                                  # E-step over candidates
            lb, det, powered = evidence(c, w)
            c['logbf'], c['det'], c['powered'] = lb, det, powered
            # damped update of the posterior log-odds (relaxed EM): the calibrated E-step is not an exact EM step, and an
            # undamped update can oscillate when a large candidate flips and changes the tumour model
            eta_new = logit(c['pi']) + lb
            # step size: d for the first damp_iters iterations, then 1/(t - damp_iters + 2) (running average of the
            # targets, as in stochastic approximation): a 2-cycle between tumour models is averaged out and q converges
            d = p['em_damp'] if it < p['em_damp_iters'] else 1.0 / (it - p['em_damp_iters'] + 2)
            c['eta'] = eta_new if it == 0 else (1 - d) * c['eta'] + d * eta_new
            c['q'] = sig(c['eta'])
        # recall regions from the current tumour
        w2 = weights()
        Z = np.zeros(int(np.prod(shB)), bool)
        for c in feas:
            if _accepted(c, out):
                Z[c['idx']] = True
        recall = []
        if Z.any():
            hosts_acc = {out[c['lesion']]['host'] for c in feas if _accepted(c, out)}
            allowed = np.zeros_like(Z)
            for key in hosts_acc:
                hb = H_B.get(key)
                hbm = hb if hb is not None else np.zeros_like(oth_B)
                allowed |= env_all[key] & ~(oth_B & ~hbm)
            touch = ndimage.binary_dilation(Z.reshape(shB), structure=ST).ravel()
            allowed &= ~claims_flat & near10
            cand_idx = np.flatnonzero(allowed)
            if len(cand_idx):
                logp1_all, _ = tumour_logp(w2, None)
                cacheA = {}
                la = lam(cand_idx, logp1_all, None, cacheA)
                posmask = np.zeros(int(np.prod(shB)), bool)
                posmask[cand_idx[np.nan_to_num(la, nan=-1) > 0]] = True
                lab_r, nr = ndimage.label(posmask.reshape(shB), structure=ST)
                if nr:
                    lab_r = lab_r.ravel()
                    sz = np.bincount(lab_r)
                    tl = np.unique(lab_r[touch & (lab_r > 0)])
                    selfun = lambda idx: np.nan_to_num(lam(idx, logp1_all, None, cacheA), nan=-1) > 0
                    for r_ in tl:
                        if sz[r_] < min_vox:
                            continue
                        ridx = np.flatnonzero(lab_r == r_)
                        rr = dict(kind='recall', pi=p['em_pi_recall'], idx=ridx, xyz=xyz_of(ridx), ml=round(len(ridx) * vox_ml, 2))
                        rr['centre'] = rr['xyz'].mean(0); rr['radius'] = float(np.linalg.norm(rr['xyz'] - rr['centre'], axis=1).max())
                        lb, det, powered = evidence(rr, w2, selection=selfun)
                        rr.update(logbf=lb, det=det, powered=powered, q=sig(logit(rr['pi']) + lb))
                        recall.append(rr)
        dq = max(abs(c['q'] - old[id(c)]) for c in feas)
        acc = {id(c) for c in feas if _accepted(c, out)}
        hist.append(dict(it=it, rho=round(resp.get('rho', 0), 3), mean_r=round(resp.get('mean_r', 0), 3), n_acc=len(acc), n_recall=sum(r['q'] >= 0.5 for r in recall), dq=round(dq, 4)))
        report['iterations'] = it + 1
        if acc == acc_old and dq < p['em_tol'] and it > 0:
            report['converged'] = True
            break
    report['history'] = hist

    # ---- output: lesion masks from accepted candidates and recall regions ---------------------------------------------
    acc_mask = np.zeros(int(np.prod(shB)), bool)
    for c in feas:
        if _accepted(c, out):
            acc_mask[c['idx']] = True
    rec_mask = np.zeros(int(np.prod(shB)), bool)
    for r in recall:
        if r['q'] >= 0.5:
            rec_mask[r['idx']] = True
    report['recalled_ml'] = round(float(rec_mask.sum()) * vox_ml, 2)
    acc_mask = acc_mask.reshape(shB); rec_mask = rec_mask.reshape(shB)
    lab_rec, _ = ndimage.label(rec_mask, structure=ST)
    for i, les in enumerate(out):
        mine = [c for c in feas if c.get('lesion') == i]
        if not mine:
            continue
        Lb = _put(B, les['sl'], les['L'], shB)
        keep = Lb & acc_mask
        if keep.any():
            ids = np.unique(lab_rec[ndimage.binary_dilation(keep, structure=ST) & (lab_rec > 0)])
            keep |= np.isin(lab_rec, ids) & (lab_rec > 0)
        before = les['status']
        if keep.any():                                   # status follows the posterior only (7 Oct: no size rule here)
            if before != 'admitted':
                les['status'], les['reason'] = 'admitted', 'r2_em_recovered'; report['n_recovered_lesions'] += 1
            bb2 = ndimage.find_objects(keep.astype(np.int8))[0]
            nsl = tuple(slice(b.start + a.start, b.start + a.stop) for a, b in zip(bb2, B))
            bpL = _put(nsl, les['sl'], les['bp'], tuple(s.stop - s.start for s in nsl))
            vtL = _put(nsl, les['sl'], les['vt'], tuple(s.stop - s.start for s in nsl))
            les['L'] = keep[bb2]; les['bp'] = bpL & les['L']; les['vt'] = vtL & les['L']; les['sl'] = nsl
            les['vox'] = int(les['L'].sum()); les['agree'] = int((les['bp'] & les['vt']).sum())
            ext = [(nsl[a].stop - nsl[a].start) * sp[a] for a in range(3) if a != ctx['k']]
            les['ld_mm'] = float(np.hypot(*ext))
            nz = int(les['L'].any(axis=tuple(a for a in range(3) if a != ctx['k'])).sum())
            les['n_slices'], les['z_mm'] = nz, round(nz * float(sp[ctx['k']]), 1)
            les['frac'] = {q: int((les['L'] & hh['env'][nsl]).sum()) / les['vox'] for q, hh in hosts.items()}
        elif before == 'admitted':
            les['status'], les['reason'] = 'rejected', 'r2_em_rejected'; report['n_rejected_lesions'] += 1
        les['r2'] = dict(before=before, candidates=[dict(source=c['source'], rater=c['rater'], ml=c['ml'], state=c['state'],
                                                          pi=c['pi'], q=round(c['q'], 3), logbf=round(c['logbf'], 3),
                                                          powered=c['powered'], folds=c['det']) for c in mine],
                         recall=[dict(ml=r['ml'], q=round(r['q'], 3), logbf=round(r['logbf'], 3)) for r in recall
                                 if (ndimage.binary_dilation(Lb, structure=ST).ravel()[r['idx']]).any()])
    report['n_accepted'] = sum(_accepted(c, out) for c in feas)
    report['n_rejected'] = sum(not _accepted(c, out) for c in feas)
    report['infeasible'] = [dict(source=c['source'], ml=c['ml'], why=c['why']) for c in cands if c['pi'] == 0][:50]
    return report
