"""Intensity-agnostic geometry descriptors (v1, 10 Oct 2026).

The same functions are used (a) to build the KB geometry priors from expert annotations (TotalSegmentator CT/MR datasets
for anatomy, R2Seg public datasets for tumours) and (b) at inference, so prior and observation are measured identically.
All descriptors are dimensionless or host-relative except vol_ml / r_eq_mm / wall thickness (kept for reference and for
host-relative ratios). Masks are resampled to an isotropic grid before measuring; the original largest spacing is
recorded so priors can be stratified by slice thickness.

shape_descriptors(mask, sp)            -> dict: vol_ml, r_eq_mm, sphericity, solidity, elong, flat, n_comp, frac_main,
                                          irr_rms, irr_p95, irr_max (surface deviation from a size-relative smoothing)
wall_descriptors(mask, img, sp)        -> dict: wall_med_mm, wall_p95_mm, wall_rel_med, lumen_frac (hollow organs; a
                                          within-scan two-class split, no absolute intensities)
relation_descriptors(T, N, sp, lumen)  -> dict: tumour vs host organ (vol_ratio, contact_frac, lumen_contact_frac,
                                          outside_hull_frac, depth_rel, host_irr_near_p95, host_irr_far_p95, bulge_ratio)
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay

ISO_MM = 1.5          # isotropic grid for shape measurement
SMOOTH_REL = 0.25     # smoothing scale = 0.25 * r_eq (size-relative, so irregularity is dimensionless)


def crop(mask, pad=3):
    idx = np.argwhere(mask)
    if len(idx) == 0: return None, None
    lo = np.maximum(idx.min(0) - pad, 0); hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi)), lo


def to_iso(arr, sp, iso=ISO_MM, order=0):
    f = np.asarray(sp, float) / iso
    if np.allclose(f, 1, atol=0.02): return arr
    return ndimage.zoom(arr.astype(np.float32) if order else arr.astype(np.uint8), f, order=order)


def _surface(m):
    return m & ~ndimage.binary_erosion(m)


def _area_mm2(m, iso):
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        v, f, _, _ = marching_cubes(np.pad(m, 1).astype(np.float32), 0.5, spacing=(iso,) * 3)
        return float(mesh_surface_area(v, f))
    except Exception:
        return float(_surface(m).sum() * iso * iso * 1.5)


def _irregularity(m, iso, r_eq, region=None):
    """|distance| of surface voxels of m to the surface of its size-relative smoothed version, / r_eq."""
    s = max(1.0, SMOOTH_REL * r_eq / iso)
    S = ndimage.gaussian_filter(m.astype(np.float32), s) > 0.5
    if S.sum() == 0: return None
    d = np.where(S, ndimage.distance_transform_edt(S), ndimage.distance_transform_edt(~S)) * iso
    surf = _surface(m)
    if region is not None: surf &= region
    v = d[surf] / r_eq
    return v if len(v) else None


def shape_descriptors(mask, sp, min_comp_ml=0.1):
    c, _ = crop(mask)
    if c is None: return None
    m = to_iso(mask[c], sp) > 0
    if m.sum() < 8: return None
    iso = ISO_MM; vox = iso ** 3 / 1000
    lab, n = ndimage.label(m); sz = np.bincount(lab.ravel())[1:] * vox
    keep = sz >= min_comp_ml
    main = lab == (np.argmax(sz) + 1)
    V = float(main.sum() * vox); r_eq = (3 * V * 1000 / (4 * np.pi)) ** (1 / 3)
    A = _area_mm2(main, iso)
    sph = float(np.pi ** (1 / 3) * (6 * V * 1000) ** (2 / 3) / A) if A > 0 else np.nan
    pts = np.argwhere(_surface(main)) * iso
    try:
        hull = ConvexHull(pts); sol = float(V * 1000 / hull.volume)
    except Exception:
        sol = np.nan
    co = np.argwhere(main) * iso
    if len(co) > 20000: co = co[np.random.default_rng(0).choice(len(co), 20000, replace=False)]
    ev = np.sort(np.linalg.eigvalsh(np.cov(co.T)))[::-1]; ev = np.maximum(ev, 1e-9)
    irr = _irregularity(main, iso, r_eq)
    return dict(vol_ml=round(float(sz[keep].sum()), 3), main_ml=round(V, 3), r_eq_mm=round(r_eq, 2),
                sphericity=round(min(sph, 1.0), 4), solidity=round(min(sol, 1.0), 4),
                elong=round(float(np.sqrt(ev[2] / ev[0])), 4), flat=round(float(np.sqrt(ev[1] / ev[0])), 4),
                n_comp=int(keep.sum()), frac_main=round(V / max(float(sz[keep].sum()), 1e-9), 4),
                irr_rms=round(float(np.sqrt(np.mean(irr ** 2))), 4) if irr is not None else np.nan,
                irr_p95=round(float(np.percentile(irr, 95)), 4) if irr is not None else np.nan,
                irr_max=round(float(irr.max()), 4) if irr is not None else np.nan)


def wall_descriptors(mask, img, sp):
    """Hollow organ: within-scan 2-class split of the organ's own intensities; the class dominating the outer 2 mm shell
    is wall. Thickness at wall voxels = distance to outside + distance to lumen (mm)."""
    c, _ = crop(mask, pad=2)
    if c is None: return None
    m = to_iso(mask[c], sp) > 0; x = to_iso(img[c], sp, order=1)
    if m.sum() < 50: return None
    x = ndimage.gaussian_filter(x, 1.0 / ISO_MM)
    v = x[m]
    try:
        from skimage.filters import threshold_otsu
        t = threshold_otsu(v)
    except Exception:
        t = np.median(v)
    hi = (x > t) & m
    dout = ndimage.distance_transform_edt(m) * ISO_MM
    shell = m & (dout <= 2.0)
    wall_hi = hi[shell].mean() >= 0.5
    W = hi if wall_hi else (m & ~hi); L = m & ~W
    if L.sum() < 10 or W.sum() < 10:
        return dict(wall_med_mm=np.nan, wall_p95_mm=np.nan, wall_rel_med=np.nan, lumen_frac=round(float(L.sum() / m.sum()), 4))
    dl = ndimage.distance_transform_edt(~L) * ISO_MM
    th = (dout + dl)[W]
    V = m.sum() * ISO_MM ** 3; r_eq = (3 * V / (4 * np.pi)) ** (1 / 3)
    return dict(wall_med_mm=round(float(np.median(th)), 2), wall_p95_mm=round(float(np.percentile(th, 95)), 2),
                wall_rel_med=round(float(np.median(th) / r_eq), 4), lumen_frac=round(float(L.sum() / m.sum()), 4))


def relation_descriptors(T, N, sp, lumen=None):
    """T tumour mask, N normal host-organ tissue (excluding tumour), lumen optional (hollow organ cavity). Same grid."""
    H = T | N | (lumen if lumen is not None else False)
    c, _ = crop(H, pad=4)
    if c is None or T.sum() == 0 or N.sum() == 0: return None
    t = to_iso(T[c], sp) > 0; n = to_iso(N[c], sp) > 0; h = to_iso(H[c], sp) > 0
    lu = to_iso(lumen[c], sp) > 0 if lumen is not None else None
    if t.sum() < 8: return None
    vox = ISO_MM ** 3
    ring = ndimage.binary_dilation(t) & ~t
    VH = h.sum() * vox; r_eq = (3 * VH / (4 * np.pi)) ** (1 / 3)
    out = dict(vol_ratio=round(float(t.sum() / h.sum()), 4),
               contact_frac=round(float((ring & n).sum() / max(ring.sum(), 1)), 4),
               lumen_contact_frac=round(float((ring & lu).sum() / max(ring.sum(), 1)), 4) if lu is not None else np.nan)
    # exophytic: tumour voxels outside the convex hull of the normal organ
    pn = np.argwhere(_surface(n))
    try:
        if len(pn) > 30000: pn = pn[np.random.default_rng(0).choice(len(pn), 30000, replace=False)]
        dl = Delaunay(pn); pt = np.argwhere(t)
        if len(pt) > 30000: pt = pt[np.random.default_rng(1).choice(len(pt), 30000, replace=False)]
        out['outside_hull_frac'] = round(float((dl.find_simplex(pt) < 0).mean()), 4)
    except Exception:
        out['outside_hull_frac'] = np.nan
    dh = ndimage.distance_transform_edt(h) * ISO_MM
    out['depth_rel'] = round(float(dh[t].max() / r_eq), 4)
    near = ndimage.distance_transform_edt(~t) * ISO_MM <= 5.0
    a = _irregularity(h, ISO_MM, r_eq, region=near); b = _irregularity(h, ISO_MM, r_eq, region=~near)
    out['host_irr_near_p95'] = round(float(np.percentile(a, 95)), 4) if a is not None else np.nan
    out['host_irr_far_p95'] = round(float(np.percentile(b, 95)), 4) if b is not None else np.nan
    out['bulge_ratio'] = round(out['host_irr_near_p95'] / out['host_irr_far_p95'], 3) if (a is not None and b is not None and out['host_irr_far_p95'] > 0) else np.nan
    return out


def components(mask, sp, min_ml=0.1, sat_mm=20.0):
    """Multiplicity: number of components >= min_ml, satellites within sat_mm of the main component, distant ones."""
    lab, n = ndimage.label(mask)
    if n == 0: return dict(n_comp=0, n_satellite=0, n_distant=0)
    vox = float(np.prod(sp)) / 1000; sz = np.bincount(lab.ravel())[1:] * vox
    ok = [i + 1 for i in range(n) if sz[i] >= min_ml]
    if not ok: return dict(n_comp=0, n_satellite=0, n_distant=0)
    main = max(ok, key=lambda i: sz[i - 1])
    d = ndimage.distance_transform_edt(lab != main, sampling=sp)
    sat = sum(1 for i in ok if i != main and d[lab == i].min() <= sat_mm)
    return dict(n_comp=len(ok), n_satellite=sat, n_distant=len(ok) - 1 - sat)


def load_priors(kb_dir=None):
    """Load the KB geometry priors (knowledge_base/geometry_priors.yaml)."""
    import os, yaml
    kb_dir = kb_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_base')
    with open(os.path.join(kb_dir, 'geometry_priors.yaml')) as f:
        return yaml.safe_load(f)


def prior_tail(value, q):
    """Two-sided position of a descriptor value in a stored quantile block {q5..q95}: 0 inside the 25-75 band, 1 beyond
    the 5/95 limits, linear in between (a simple, assumption-light plausibility score for inference)."""
    lo5, lo25, hi75, hi95 = q['q5'], q['q25'], q['q75'], q['q95']
    if lo25 <= value <= hi75: return 0.0
    if value < lo25: return float(min(1.0, (lo25 - value) / max(lo25 - lo5, 1e-9)))
    return float(min(1.0, (value - hi75) / max(hi95 - hi75, 1e-9)))
