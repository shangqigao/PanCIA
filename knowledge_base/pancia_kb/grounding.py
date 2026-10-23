"""Image-grounding signs (16 Oct 2026), consolidated from the 12 Oct research scripts (verbatim maths):
  depth_std           <- scripts/norm_feats3.depth_std        (normative, depth-conditioned standardisation vs host parenchyma)
  host_frame          <- scripts/norm_geo30.run_regions body  (smoothed image Xs, host-frame Z, signed host depth D)
  closed30            <- scripts/norm_geo30.closed30          (closedness, 30 mm search)
  level2              <- scripts/norm_geo.geo level 2 + elong (patchy heterogeneity rel_tex, elongation)
  tree_call           <- FROZEN_GEO_20261012.txt rule with fix C (thresholds frozen 12 Oct)
  organ_tests         <- ledger v6 D10 (a) is the organ / (b) contains the organ
Plan: PanCIA/documents/plan/Image_grounding_v6_plan.md (revised 16 Oct: v2, veto-only, outline kept). No parameter here is fitted on TCGA.
16 Oct grounding v2 (ledger/v6/FROZEN_GROUNDING_V2_20261016.txt): veto_v2 = v6.2 shape organ-like OR strict closedness
(open_ratio > T_OPEN, the 98th percentile over R2Seg expert tumours). The 30 mm tree rule is kept for reporting only (not robust:
R2Seg GT veto 7.8%, TCGA CT tumour veto 22%)."""
from __future__ import annotations
import numpy as np
from scipy import ndimage

OPEN_T, CLOSED_T, HET_T, ELONG_T, ZDEV_T = 5.27, 1.42, 1.047, 0.363, 1.5     # frozen 12 Oct (30 mm rule, fix C)


def depth_std(xo, do, xr, dr, s_glob):
    """standardise values by the rest-of-organ median/robust SD at the same depth below the organ surface"""
    E = np.array([0, 1.5, 3, 5, 8, 12, 18, 25, 35, 50, 1e9]); bo = np.digitize(do, E) - 1; br = np.digitize(dr, E) - 1
    m = np.full(len(E), np.nan); sd = np.full(len(E), np.nan)
    for b in range(len(E) - 1):
        v = xo[bo == b]
        if len(v) >= 100:
            q25, q50, q75 = np.percentile(v, [25, 50, 75]); m[b] = q50; sd[b] = max((q75 - q25) / 1.349, 0.25 * s_glob, 1e-3)
    ok = np.where(np.isfinite(m))[0]
    if len(ok) == 0:
        return None, None
    fill = lambda b: ok[np.argmin(np.abs(ok[:, None] - b[None, :]), 0)]
    bo2, br2 = fill(bo), fill(br)
    return (xo - m[bo2]) / sd[bo2], (xr - m[br2]) / sd[br2]


def _edt(m, sp, pe=4):
    return ndimage.distance_transform_edt(np.pad(m, pe, mode='edge'), sampling=sp)[pe:-pe, pe:-pe, pe:-pe]


def smooth(X, sp, thick, ax):
    sg = [0 if (thick and a == ax) else 1.0 / sp[a] for a in range(3)]
    return ndimage.gaussian_filter(X.astype(np.float32), sg)


def host_frame(Xs, sp, h, R, Hc=None, min_ref=200):
    """Z (host frame) and signed depth D on the window. Reference = host parenchyma (Hc, default h) minus R.
    Returns (Z, D, n_ref) or (None, D, n_ref) when the reference is too small."""
    Hc = h if Hc is None else Hc
    Din = _edt(h, sp); Dout = _edt(~h, sp); D = np.where(h, Din, Dout); Du = _edt(h | R, sp)
    par = Hc & ~R; v = Xs[par]
    if len(v) < min_ref:
        return None, D, int(len(v))
    q25, q75 = np.percentile(v, [25, 75])
    zo, za = depth_std(v, Din[par], Xs.ravel(), Du.ravel(), max((q75 - q25) / 1.349, 1.0))
    if za is None:
        return None, D, int(len(v))
    return za.reshape(Xs.shape).astype(np.float32), D, int(len(v))


def closed30(X, Z, host, R, D, sp, thick, ax, radius=30.0):
    out = {}; out['vol_ml'] = float(R.sum() * np.prod(sp) / 1000)
    if X.size > 3e6:
        nd = 2 if thick else 3; f = int(np.ceil((X.size / 3e6) ** (1 / nd)))
        st = tuple(slice(None, None, 1 if (thick and a == ax) else f) for a in range(3))
        Z, host, R, D = Z[st], host[st], R[st], D[st]; sp = np.array([sp[a] if (thick and a == ax) else sp[a] * f for a in range(3)])
    if not R.any():
        return out
    zmed = float(np.median(Z[R])); out['zh_med'] = zmed
    if abs(zmed) >= ZDEV_T:
        sgn = np.sign(zmed); dev = (sgn * Z > ZDEV_T) | R
        dout = ndimage.distance_transform_edt(~R, sampling=sp); dev &= dout <= radius
        hd = ndimage.distance_transform_edt(~host, sampling=sp) <= 2; dev &= hd | R
        lb, n = ndimage.label(dev); ids = np.unique(lb[R]); ids = ids[ids > 0]; Gm = np.isin(lb, ids) & ~R
        out['open_ratio'] = float(Gm.sum() / R.sum()); out['out_ml'] = float(Gm.sum() * np.prod(sp) / 1000)
        out['reach_mm'] = float(dout[Gm].max()) if Gm.any() else 0.0
    return out


def level2(X, R, sp, thick, ax):
    """rel_tex (patchy heterogeneity, candidate frame; >= 1 ml) and elongation, as norm_geo.geo"""
    out = {}
    if R.sum() < 4:
        return out
    xr = X[R]; med = np.median(xr); mad = 1.4826 * np.median(np.abs(xr - med)) + 1e-6; Zc = (X - med) / mad
    if R.sum() * np.prod(sp) / 1000 >= 1:
        core = ndimage.binary_erosion(R)
        if core.sum() >= 20:
            sg = [0 if (thick and a == ax) else 2.0 / sp[a] for a in range(3)]; m = ndimage.gaussian_filter(Zc, sg)
            v = ndimage.gaussian_filter(Zc * Zc, sg) - m * m
            out['rel_tex'] = float(np.median(np.sqrt(np.maximum(v[core], 0))))
    try:
        P = np.argwhere(R) * sp; ev = np.sort(np.linalg.eigvalsh(np.cov(P.T))) + 1e-9; out['elong'] = float(np.sqrt(ev[0] / ev[2]))
    except Exception:
        pass
    return out


def tree_call(f):
    """frozen 30 mm rule + fix C -> 'tumour' | 'organ' | 'uncertain'"""
    o = f.get('open_ratio'); z = f.get('zh_med'); rt = f.get('rel_tex'); el = f.get('elong'); vol = f.get('vol_ml') or 0
    has_o = o is not None and np.isfinite(o)
    op = has_o and o > OPEN_T; cl = has_o and o <= CLOSED_T
    pa = rt is not None and np.isfinite(rt) and vol >= 3 and rt <= HET_T
    elg = el is not None and np.isfinite(el) and el < ELONG_T
    if (cl or pa) and not op and not elg:
        return 'tumour'
    if (op or elg) and not cl and not pa:
        if not has_o and elg and not op and (z is None or abs(z) < ZDEV_T):      # fix C
            return 'uncertain'
        return 'organ'
    return 'uncertain'


def organ_tests(T, Hm, sp, ax, vol_ratio=0.6, dice2d=0.6, inside=0.8, contain_H=0.8, contain_frac=0.5):
    """D10: (a) T is the organ Hm; (b) T contains Hm. Returns 'a' | 'b' | None"""
    tv, hv = int(T.sum()), int(Hm.sum())
    if tv == 0 or hv == 0:
        return None
    inter = int((T & Hm).sum())
    if inter >= contain_H * hv and inter >= contain_frac * tv:
        return 'b'
    if inter >= inside * tv and tv >= vol_ratio * hv:
        ds = []
        for z in np.nonzero(T.any(axis=tuple(a for a in range(3) if a != ax)))[0]:
            t = np.take(T, z, axis=ax); hh = np.take(Hm, z, axis=ax)
            if hh.any():
                ds.append(2 * (t & hh).sum() / (t.sum() + hh.sum()))
        if ds and np.median(ds) >= dice2d:
            return 'a'
    return None


T_OPEN = 17.069        # 98th pct of open_ratio over R2Seg expert tumours with closedness computed (n = 322); expert data only
LR_VETO = 1.0 / 3.0


def veto_v2(f, v62_cls=None):
    """grounding v2 existence evidence (veto-only). f: closed30 features (zh_med, open_ratio); v62_cls: v6.2 shape class.
    Returns (veto: bool, reasons: list)."""
    why = []
    if v62_cls == 'organ-like':
        why.append('shape_v62')
    o = f.get('open_ratio')
    if o is not None and np.isfinite(o) and o > T_OPEN:
        why.append('open')
    return bool(why), why


def apply_veto(p, veto, lr=LR_VETO):
    """odds update of a prior/posterior p by the veto likelihood ratio (neutral when not vetoed)"""
    if not veto or p <= 0 or p >= 1:
        return p
    o = p / (1 - p) * lr; return float(o / (1 + o))


def host_correct(H0, T):
    """corrected host: the anatomy agent's organ minus the (admitted) tumour"""
    return H0 & ~T


# ------------------------------------------------------------------ slice-thickness harmonisation (user, 17 Oct 2026)
# Thick slices change closedness and internal structure for tumours and normals alike (g0_slice_thickness_check_2026-10-17:
# 17.6% of expert-tumour calls change; normals 3-20 ml tumour-like 34% -> 46%). Every scan -- expert data and TCGA -- is therefore put on a
# common ~5 mm through-plane grid BEFORE any sign is computed: the S-I axis is averaged in blocks of k = floor(5/sp + 0.5) slices
# (k >= 1; scans already ~>= 3.4 mm are unchanged). Image: block mean (thick-slice partial volume). Labels: per-block majority.
THICK_MM = 5.0


def thick_k(sp_si: float, target: float = THICK_MM) -> int:
    return max(1, int(np.floor(target / float(sp_si) + 0.5)))


def harmonise_array(X: np.ndarray, ax: int, k: int, label: bool = False) -> np.ndarray:
    """average (image) or majority-vote (label) along axis ax in blocks of k slices; trailing slices that do not fill a block are dropped"""
    if k <= 1: return X
    n = (X.shape[ax] // k) * k; X = np.take(X, np.arange(n), axis=ax)
    sh = list(X.shape); sh[ax:ax + 1] = [n // k, k]
    if not label:
        return X.astype(np.float32).reshape(sh).mean(axis=ax + 1)
    Xi = np.rint(X).astype(np.int16); vals = np.unique(Xi)
    if len(vals) <= 2:
        return ((Xi.reshape(sh) > 0).mean(axis=ax + 1) >= 0.5).astype(np.int16) * (int(vals.max()) if len(vals) else 1)
    out = np.zeros([s for i, s in enumerate(sh) if i != ax + 1], np.int16); best = None
    for v in vals:
        f = (Xi == v).reshape(sh).mean(axis=ax + 1)
        if best is None: best = f; out[:] = v
        else: m = f > best; out[m] = v; best = np.where(m, f, best)
    return out


def harmonise_affine(aff: np.ndarray, ax: int, k: int) -> np.ndarray:
    A = np.array(aff, float).copy()
    if k > 1: A[:3, ax] = aff[:3, ax] * k; A[:3, 3] = aff[:3, 3] + aff[:3, ax] * (k - 1) / 2.0
    return A
