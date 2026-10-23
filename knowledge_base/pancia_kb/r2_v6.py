"""R2 v6: normal-model image grounding (design: documents/plan/R2_v6_design.md, 11 Oct 2026). Work in progress.

Anomaly-detection framing: the normal is modelled (organ geometry from expert anatomy annotations, KB
geometry_priors.yaml; appearance from the scan's own host), the tumour weakly (broad architecture prior, per candidate).
r2_em.py is untouched; v6 is selected with em_model='v6' once integrated.

Evidence terms are log Bayes factors log p(.|H1) - log p(.|H0), each capped at +-log(100):
  e_vol    organ volume: H0 the normal organ must explain organ+candidate; H1 the normal organ is organ-minus-tumour.
           Conditioned on the candidate (its existence prior is the ledger pi).
  e_shape  tumour plausibility (elongation): <= 0 only; penalises shapes implausible for a tumour, never adds support.
"""
import numpy as np

CAP = float(np.log(100.0))
Z95 = 1.6448536269514722


def _cap(x):
    return float(np.clip(x, -CAP, CAP))


def lognormal_from_q(q):
    """Log-normal matched to stored quantiles: mu = log q50, sigma from the 5-95 range."""
    q5, q50, q95 = max(q['q5'], 1e-6), max(q['q50'], 1e-6), max(q['q95'], 1e-6)
    return np.log(q50), max((np.log(q95) - np.log(q5)) / (2 * Z95), 1e-3)


def logpdf_log(x, mu, s):
    """Log density of log(x) under N(mu, s) (density on the log scale)."""
    z = (np.log(max(x, 1e-9)) - mu) / s
    return float(-0.5 * z * z - np.log(s * np.sqrt(2 * np.pi)))


def split_normal_from_q(q, transform='logit'):
    """Two-piece normal on a transformed scale matched to q5 / q50 / q95 (asymmetric tails)."""
    f = (lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))) if transform == 'logit' else np.log
    m = f(q['q50']); sl = max((m - f(q['q5'])) / Z95, 1e-3); sh = max((f(q['q95']) - m) / Z95, 1e-3)
    return f, m, sl, sh


def e_vol(organ_ml, tumour_ml, inside_ml, anat_vol_q, tum_ratio_q=None):
    """organ_ml: agent organ estimate (may already contain the tumour), tumour_ml: candidate volume, inside_ml: part of
    the candidate inside the organ estimate. H0: normal organ volume = organ_ml + (tumour_ml - inside_ml).
    H1: normal organ volume = organ_ml - inside_ml. The candidate (and its size) is conditioned on: its existence prior is
    the ledger pi, so no tumour-size term is added here (tum_ratio_q kept only for reporting); a small candidate therefore
    gives e_vol ~ 0 (uninformative), not a penalty."""
    mu, s = lognormal_from_q(anat_vol_q)
    v0 = organ_ml + (tumour_ml - inside_ml); v1 = max(organ_ml - inside_ml, 1e-3)
    r = tumour_ml / (v1 + tumour_ml)
    l1 = logpdf_log(v1, mu, s)
    l0 = logpdf_log(v0, mu, s)
    return _cap(l1 - l0), dict(v0=round(v0, 1), v1=round(v1, 1), ratio=round(r, 4))


def e_shape(elong, tum_elong_q):
    """<= 0: -0.5 z^2 on the logit scale with asymmetric tails (no support for plausible shapes, penalty for implausible)."""
    if elong is None or not np.isfinite(elong): return 0.0
    f, m, sl, sh = split_normal_from_q(tum_elong_q)
    z = (f(elong) - m) / (sl if f(elong) < m else sh)
    return _cap(-0.5 * z * z)


# ----------------------------------------------------------------------------------------------------------------------
# E_app: within-scan appearance (design section 3a)
# ----------------------------------------------------------------------------------------------------------------------
from scipy import ndimage, stats
from scipy.spatial import cKDTree

GB, KAPPA, LCLIP, M_NULL, NS, BLOCK_MM = 32, 500.0, 5.0, 50, 6000, 15.0
TAU_APP = 0.454   # nats/voxel, a priori: scripts/r2_v6_tau_sim.py (median gain of a lesion 1 tissue-SD brighter, 1-80 ml)


def _smooth_hist(cnt, n):
    if cnt.sum() <= 0: return None
    s = max(0.5, (max(n, 2) ** (-1 / 5)) * (GB / np.sqrt(12)))
    d = ndimage.gaussian_filter1d(cnt.astype(float), s, mode='reflect') + 1e-12
    return d / d.sum()


def heldout_gain(bt, pt, bh, ph, per_voxel=False):
    """Two-fold held-out log density ratio of a sample (bins bt, folds pt) vs the reference (bh, ph).
    Returns mean gain (nats/voxel), and optionally per-voxel gains of fold-k voxels scored with fold-k densities
    (used only for core SELECTION on the same fold) as {k: (index_in_sample, gain)}."""
    out, sel = [], {}
    for k in (0, 1):
        F, Ob = bt[pt == k], bt[pt == 1 - k]; hk = bh[ph == k]
        if len(F) < 10 or len(Ob) < 10 or len(hk) < 50: return (None, None) if per_voxel else None
        fh = _smooth_hist(np.bincount(hk, minlength=GB), len(hk)); ft = _smooth_hist(np.bincount(F, minlength=GB), len(F))
        q = (len(F) * ft + KAPPA * fh) / (len(F) + KAPPA)
        lam = np.clip(np.log(q) - np.log(fh), -LCLIP, LCLIP)
        out.append(float(lam[Ob].mean()))
        if per_voxel: sel[k] = (np.where(pt == k)[0], lam[F], q, fh)
    return (float(np.mean(out)), sel) if per_voxel else float(np.mean(out))


def core_gain(co_mm, bt, pt, bh, ph, sigma_mm=3.0):
    """Cross-fitted core: select the core on fold k (smoothed per-voxel gain > 0 at fold-k points, normalised kernel
    smoothing over fold-k points within 2 sigma), apply it to fold-(1-k) points by nearest fold-k point, score those with
    fold-k densities. Returns (mean gain over both directions, core fraction) or (None, None)."""
    g, sel = heldout_gain(bt, pt, bh, ph, per_voxel=True)
    if g is None: return None, None
    out, frac = [], []
    for k in (0, 1):
        ik, lam, q, fh = sel[k]; io = np.where(pt == 1 - k)[0]
        if len(ik) < 10 or len(io) < 10: return None, None
        tk = cKDTree(co_mm[ik]); nb = tk.query_ball_point(co_mm[ik], r=2 * sigma_mm)
        sm = np.array([np.average(lam[j], weights=np.exp(-0.5 * (np.linalg.norm(co_mm[ik][j] - co_mm[ik][a], axis=1) / sigma_mm) ** 2)) for a, j in enumerate(nb)])
        core_k = sm > 0
        if core_k.sum() < 10: out.append(0.0); frac.append(0.0); continue
        _, nn = tk.query(co_mm[io]); inc = core_k[nn]
        if inc.sum() < 10: out.append(0.0); frac.append(float(core_k.mean())); continue
        lo = np.clip(np.log(q[bt[io][inc]]) - np.log(fh[bt[io][inc]]), -LCLIP, LCLIP)
        out.append(float(lo.mean())); frac.append(float(inc.mean()))
    return float(np.mean(out)), float(np.mean(frac))


def logbf_halfnormal(delta, sigma, tau):
    sigma = max(float(sigma), 1e-9); z = float(delta) / sigma; R = float(tau) / sigma; v = 1.0 + R * R
    return float(np.log(2.0) - 0.5 * np.log(v) + z * z * R * R / (2 * v) + stats.norm.logcdf(z * R / np.sqrt(v)))


def _parity(co, sp):
    return (np.floor(co * sp / BLOCK_MM).astype(int).sum(1) % 2).astype(np.int8)


def e_app(F, sp, ax, cand, levels, rng=None, min_ref=500):
    """F: feature volume (smoothed intensity) of the window; cand: candidate mask; levels: ordered list of
    (name, reference mask) already restricted to host-organ interior minus all candidates (+5 mm), e.g.
    [('local', ...), ('organ', ...), ('counterpart', ...)]. Returns dict with log_bf (whole mask), log_bf_core, core_frac,
    trim (bool), ref, n_T, n_H, z, s, sd0; or ref='none'."""
    rng = rng or np.random.default_rng(0)
    st = np.zeros((3, 3, 3), bool); sl = [1, 1, 1]
    for a in range(3):
        if a != ax: sl[a] = slice(None)
    st[tuple(sl)] = True
    T = ndimage.binary_erosion(cand, structure=st)
    if T.sum() < 200: return dict(ref='none', why='candidate_too_small', n_T=int(T.sum()))
    tco = np.argwhere(T)
    if len(tco) > NS: tco = tco[rng.choice(len(tco), NS, replace=False)]
    ref, hco = None, None
    for i, (name, m) in enumerate(levels):
        n = int(m.sum()); last = i == len(levels) - 1
        if n >= max(min_ref, 3 * len(tco)) or (last and n >= min_ref) or (n >= min_ref and all(int(mm.sum()) < min_ref for _, mm in levels[i + 1:])):
            ref, hco = name, np.argwhere(m); break
    if ref is None: return dict(ref='none', why='no_reference', n_T=len(tco))
    if len(hco) > 60000: hco = hco[rng.choice(len(hco), 60000, replace=False)]
    ft, fh = F[tuple(tco.T)], F[tuple(hco.T)]
    e = np.quantile(np.concatenate([ft, fh]), np.arange(1, GB) / GB)
    bt, bh = np.searchsorted(e, ft), np.searchsorted(e, fh); pt, ph = _parity(tco, sp), _parity(hco, sp)
    s = heldout_gain(bt, pt, bh, ph); sc, cf = core_gain(tco * sp, bt, pt, bh, ph)
    if s is None: return dict(ref=ref, why='fold_too_small', n_T=len(tco), n_H=len(hco))
    kk = min(len(tco), len(hco) // 3); tree = cKDTree(hco * sp); s0, c0 = [], []
    for _ in range(M_NULL):
        ii = np.atleast_1d(tree.query(hco[rng.integers(len(hco))] * sp, k=kk)[1]); keep = np.ones(len(hco), bool); keep[ii] = False
        v = heldout_gain(bh[ii], ph[ii], bh[keep], ph[keep])
        if v is not None: s0.append(v)
        vc, _ = core_gain(hco[ii] * sp, bh[ii], ph[ii], bh[keep], ph[keep])
        if vc is not None: c0.append(vc)
    if len(s0) < 20: return dict(ref=ref, why='null_too_small', n_T=len(tco), n_H=len(hco))
    s0 = np.array(s0); sd0 = s0.std() + 1e-9; m0 = np.median(s0)
    out = dict(ref=ref, n_T=len(tco), n_H=len(hco), s=round(s, 4), sd0=round(float(sd0), 4), z=round(float((s - m0) / sd0), 2),
               log_bf_raw=round(logbf_halfnormal(s - m0, sd0, TAU_APP), 3))
    out['log_bf'] = _cap(out['log_bf_raw'])
    if sc is not None and len(c0) >= 20:
        c0 = np.array(c0); sdc = c0.std() + 1e-9
        out.update(s_core=round(sc, 4), z_core=round(float((sc - np.median(c0)) / sdc), 2), core_frac=round(cf, 3),
                   log_bf_core_raw=round(logbf_halfnormal(sc - np.median(c0), sdc, TAU_APP), 3))
        out['log_bf_core'] = _cap(out['log_bf_core_raw'])
        out['trim'] = bool(out['log_bf_core_raw'] - out['log_bf_raw'] >= np.log(3) and cf < 0.9)   # compared uncapped
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Selection-matched null (11 Oct 2026): candidates are SELECTED by agents because they look unlike their surroundings, so
# the null must be normal structures selected the same way, not random host patches. Agent-agnostic procedure:
#   1. blob sites: scale-normalised |LoG| of the image at the candidate's scale (sigma = r_eq / sqrt(3), in-plane only
#      for thick slices), local maxima inside the reference pool (min separation 2 r_eq), top M_SITES by response;
#   2. pseudo-candidate at each site: within a ball of radius 1.5 r_eq, the k reference voxels most atypical of the host
#      (largest |feature - host median|), k = candidate sample size -> an object selected for looking different;
#   3. score each pseudo-candidate exactly like the candidate (held-out gain vs the reference without it).
# The candidate must be more abnormal than the host's most candidate-like normal structures.
# ----------------------------------------------------------------------------------------------------------------------
M_SITES = 30


def e_app_matched(F, X, sp, ax, cand, levels, rng=None, min_ref=500):
    rng = rng or np.random.default_rng(0)
    st = np.zeros((3, 3, 3), bool); sl = [1, 1, 1]
    for a in range(3):
        if a != ax: sl[a] = slice(None)
    st[tuple(sl)] = True
    T = ndimage.binary_erosion(cand, structure=st)
    if T.sum() < 200: return dict(ref='none', why='candidate_too_small', n_T=int(T.sum()))
    tco = np.argwhere(T)
    if len(tco) > NS: tco = tco[rng.choice(len(tco), NS, replace=False)]
    ref, Hm = None, None
    for i, (name, m) in enumerate(levels):
        n = int(m.sum())
        if n >= max(min_ref, 3 * len(tco)) or (n >= min_ref and all(int(mm.sum()) < max(min_ref, 3 * len(tco)) for _, mm in levels[i + 1:])):
            ref, Hm = name, m; break
    if ref is None: return dict(ref='none', why='no_reference', n_T=len(tco))
    hco_all = np.argwhere(Hm); hco = hco_all if len(hco_all) <= 60000 else hco_all[rng.choice(len(hco_all), 60000, replace=False)]
    ft, fh = F[tuple(tco.T)], F[tuple(hco.T)]
    e = np.quantile(np.concatenate([ft, fh]), np.arange(1, GB) / GB)
    bt, bh = np.searchsorted(e, ft), np.searchsorted(e, fh); pt, ph = _parity(tco, sp), _parity(hco, sp)
    s = heldout_gain(bt, pt, bh, ph)
    if s is None: return dict(ref=ref, why='fold_too_small', n_T=len(tco), n_H=len(hco))
    # 1. blob sites at the candidate's scale
    vox = float(np.prod(sp)); r_eq = (3 * cand.sum() * vox / (4 * np.pi)) ** (1 / 3)
    sig = [0 if (a == ax and sp[a] / min(sp) >= 1.5) else max(r_eq / np.sqrt(3) / sp[a], 0.7) for a in range(3)]
    Lg = np.abs(ndimage.gaussian_laplace(X, sig)) * (r_eq / np.sqrt(3)) ** 2
    fp = [max(3, int(round(2 * r_eq / sp[a]))) if sig[a] > 0 else 1 for a in range(3)]
    pk = (Lg == ndimage.maximum_filter(Lg, size=fp)) & Hm
    pc = np.argwhere(pk)
    if len(pc) == 0: return dict(ref=ref, why='no_sites', n_T=len(tco), n_H=len(hco), s=round(s, 4))
    pc = pc[np.argsort(-Lg[tuple(pc.T)])][:M_SITES]
    # 2. pseudo-candidates: k most atypical reference voxels within 1.5 r_eq of each site
    k = len(tco); med = np.median(fh); tree = cKDTree(hco * sp); atyp = np.abs(fh - med); s0 = []
    for c in pc:
        nb = np.array(tree.query_ball_point(c * sp, r=1.5 * r_eq))
        if len(nb) < max(20, k // 4): continue
        kk = min(k, len(nb), len(hco) // 3); ii = nb[np.argsort(-atyp[nb])[:kk]]
        keep = np.ones(len(hco), bool); keep[ii] = False
        v = heldout_gain(bh[ii], ph[ii], bh[keep], ph[keep])
        if v is not None: s0.append(v)
    if len(s0) < 10: return dict(ref=ref, why='few_sites', n_sites=len(s0), n_T=len(tco), n_H=len(hco), s=round(s, 4))
    s0 = np.array(s0); sd0 = s0.std() + 1e-9; m0 = np.median(s0)
    out = dict(ref=ref, n_T=len(tco), n_H=len(hco), n_sites=len(s0), s=round(s, 4), s0_med=round(float(m0), 4), s0_max=round(float(s0.max()), 4),
               sd0=round(float(sd0), 4), z=round(float((s - m0) / sd0), 2), p=round(float((1 + (s0 >= s).sum()) / (len(s0) + 1)), 4),
               log_bf_raw=round(logbf_halfnormal(s - m0, sd0, TAU_APP), 3))
    out['log_bf'] = _cap(out['log_bf_raw'])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# R2Seg-style two-sample test (arXiv 2511.12691, sec. 3.2.3 / App. 1.4), adapted (11 Oct 2026):
#   X = candidate voxels, Y = candidate + its host (H0: p_X = p_Y). Since p_Y = w p_X + (1-w) p_H, w = |X|/|Y|:
#   MMD^2(X, Y) = (1-w)^2 MMD^2(X, H)  -> a candidate that is host-like OR swallows most of its host is not significant.
#   Statistic: MMD^2 with a Gaussian kernel, median-heuristic bandwidth on the pooled sample (1-D feature -> computed on
#   a 256-bin histogram, V-statistic). Nulls: (a) voxel permutation (as in the paper); (b) compact sub-regions of Y with
#   |X| voxels (spatially aware: H0 = X is just another piece of Y).
# ----------------------------------------------------------------------------------------------------------------------
NB_MMD = 256


def _mmd_hist(a, b, K):
    d = a - b
    return float(d @ K @ d)


def mmd_test(xv, yv, yco, sp, n_x, rng=None, B=200, M=100):
    """xv: candidate feature values; yv/yco: Y feature values and voxel coords (Y includes X); n_x: candidate voxel count
    (for the compact sub-region null). Returns dict(mmd, p_perm, p_patch, z_patch, w)."""
    rng = rng or np.random.default_rng(0)
    if len(xv) > 4000: xv = xv[rng.choice(len(xv), 4000, replace=False)]
    sub = rng.choice(len(yv), min(len(yv), 20000), replace=False); yvs, ycs = yv[sub], yco[sub]
    pool = np.concatenate([xv, yvs[:4000]])
    lo, hi = np.percentile(pool, [0.1, 99.9]); edges = np.linspace(lo, hi, NB_MMD + 1); c = 0.5 * (edges[1:] + edges[:-1])
    pp = pool[rng.choice(len(pool), min(len(pool), 2000), replace=False)]
    sig = np.median(np.abs(pp[:, None] - pp[None, :])) / np.sqrt(2) + 1e-9      # median heuristic (pairwise distance)
    K = np.exp(-(c[:, None] - c[None, :]) ** 2 / (2 * sig * sig))
    h = lambda v: np.histogram(np.clip(v, lo, hi), edges)[0].astype(float) / max(len(v), 1)
    ys4 = yvs[rng.choice(len(yvs), min(len(yvs), 4000), replace=False)]
    obs = _mmd_hist(h(xv), h(ys4), K)
    # (a) voxel permutation of the pooled sample
    P = np.concatenate([xv, ys4]); nx = len(xv); cnt = 0
    for _ in range(B):
        rng.shuffle(P); cnt += _mmd_hist(h(P[:nx]), h(P[nx:]), K) >= obs
    p_perm = (cnt + 1) / (B + 1) if B > 0 else float('nan')
    # (b) compact sub-regions of Y with the candidate's voxel count (scaled to the subsample)
    k = int(np.clip(round(n_x * len(ycs) / max(len(yv), 1)), 50, len(ycs) - 1)); tree = cKDTree(ycs * sp); s0 = []
    for _ in range(M):
        ii = np.atleast_1d(tree.query(ycs[rng.integers(len(ycs))] * sp, k=k)[1]); v = yvs[ii]
        if len(v) > 4000: v = v[rng.choice(len(v), 4000, replace=False)]
        s0.append(_mmd_hist(h(v), h(ys4), K))
    s0 = np.array(s0)
    return dict(mmd=round(obs, 6), p_perm=round(p_perm, 4), p_patch=round(float((1 + (s0 >= obs).sum()) / (M + 1)), 4),
                z_patch=round(float((obs - np.median(s0)) / (s0.std() + 1e-12)), 2), mmd_null_med=round(float(np.median(s0)), 6))


# ---------------------------------------------------------------------------------------------------------------
# Lesion-likeness evidence (12 Oct 2026). Candidate-intrinsic: does the candidate look like a lesion (compact, focal,
# internally structured), independent of any host reference? Confirmed on fresh rated sets E3 (AUC 0.85 frozen) and E4
# (B5 variant 0.78). Mapping to a log Bayes factor calibrated on EXPERT data only (R2Seg tumour masks vs expert
# normal-structure candidates from the TotalSegmentator datasets and R2Seg organ labels): see
# clinical/PanCIA_outputs/ledger/v6/expert_ll/ll_calibration.txt. Weak by design: |log BF| rarely exceeds log 3.
LL_A, LL_B = -3.272, 6.331


def e_lesion(img, M, sp, ax_si=2, cap=np.log(100)):
    """Returns (log BF lesion vs non-lesion, feature dict). img may be a nibabel dataobj proxy; M full-size bool mask."""
    from . import lesion_likeness as LLm
    f = LLm.lesion_features(img, M, sp, ax_si); B = LLm.score(f); f['B'] = B
    if not np.isfinite(B): return np.nan, f
    return float(np.clip(LL_A + LL_B * B, -cap, cap)), f
