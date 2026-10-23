"""KB relational region of an unsegmented organ (host step 2, 20 Oct 2026). Replaces the vertebral slab where landmarks are segmented.
Each KB spatial_prior entry (landmark, relation, dist_mm [d0, d1]) defines a constraint region
    R_c = {p : some landmark point q has |p - q| <= d1 + m + E and angle(p - q, u_rel) <= theta}  AND  dist(p, landmark) >= d0 - m
(directional cone dilation of the landmark, Bloch-style fuzzy directional relation made crisp; E = organ extent beyond its near surface);
an earlier nearest-point variant (angle of p - nearest landmark point) lost coverage on TotalSegmentator and is kept only as parts()),
u_rel = unit direction of the relation (anterior/posterior/superior/inferior/left/right; lateral/medial relative to the body midline;
compounds are normalised sums). 'between' (hip): between the two landmark centroids in x, within the landmarks' z range +- (d1 + m).
Region = slab (own span, optional) AND side AND every available constraint; a constraint is available when its landmark is
segmented (masks given). If the full intersection is empty, voxels satisfying the most constraints are kept. No landmark -> None
(caller falls back to the slab). theta and m are fitted on TotalSegmentator CT (expert labels; scripts/ts_relational.py).
Coordinates: nibabel world (RAS: +x right, +y anterior, +z superior); 'left' uses lsign (+1 if patient left is +x)."""
import numpy as np
from scipy import ndimage

THETA_DEG = 60.0; MARGIN_MM = 10.0          # defaults; overwritten by ledger/v8/testbed/relational_fit.json when present
LM = {'kidney': ['kidney_left', 'kidney_right'], 'hip': ['hip_left', 'hip_right'], 'iliac_artery': ['iliac_artery_left', 'iliac_artery_right'],
      'iliac_vein': ['iliac_vena_left', 'iliac_vena_right'], 'psoas': ['iliopsoas_left', 'iliopsoas_right'], 'adrenal': ['adrenal_gland_left', 'adrenal_gland_right'],
      'splenic_vein': ['portal_vein_and_splenic_vein'], 'lung': ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
      'lung_middle_lobe_right', 'lung_lower_lobe_right', 'lung_left', 'lung_right']}
REL = {'anterior_to': 'A', 'posterior_to': 'P', 'superior_to': 'S', 'inferior_to': 'I', 'left_of': 'L', 'right_of': 'R', 'lateral_to': 'X',
       'medial_to': 'M', 'posterior_superior_to': 'PS', 'posterior_inferior_to': 'PI', 'superior_lateral_to': 'SX', 'superior_medial_to': 'SM',
       'posterior_lateral_to': 'PX', 'posterior_medial_to': 'PM', 'anterior_medial_to': 'AM', 'anterior_right_of': 'AR',
       'inferior_surface_of': 'I', 'lateral_to_head_of': 'X'}


def landmark(name, side, masks):
    """union of the segmented masks of a KB landmark (same side for a sided target when the landmark is paired); None if absent"""
    if name == 'spine':
        ks = [k for k in masks if k.startswith('vertebrae')]
    elif name == 'sacral_promontory':
        s = masks.get('sacrum')
        return None if s is None or not s.any() else ('promontory', s)
    else:
        ks = LM.get(name, [name])
        if side in ('left', 'right'):
            same = [k for k in ks if k.endswith('_' + side)]
            ks = same or ks
    m = None
    for k in ks:
        if k in masks and masks[k] is not None and masks[k].any():
            m = masks[k].copy() if m is None else (m | masks[k])
    return m


def _dirs(code, X, mid_x, lsign):
    """direction field (3, ...) in world coordinates for a relation code; X = world x of the voxels (same shape)"""
    sh = X.shape; u = np.zeros((3,) + sh, np.float32)
    lat = np.sign(X - mid_x).astype(np.float32)
    for c in code:
        if c == 'A': u[1] += 1
        elif c == 'P': u[1] -= 1
        elif c == 'S': u[2] += 1
        elif c == 'I': u[2] -= 1
        elif c == 'L': u[0] += lsign
        elif c == 'R': u[0] -= lsign
        elif c == 'X': u[0] += lat
        elif c == 'M': u[0] -= lat
    n = np.sqrt((u ** 2).sum(0)); n[n == 0] = 1; return u / n


def constraint(L, rel, d01, sp, A3, X, Z, mid_x, lsign, theta, m, T=(0.0, 0.0, 0.0)):
    d0, d1 = float(d01[0]), float(d01[1])
    if rel == 'between':
        parts_ = L if isinstance(L, (list, tuple)) else None
        if parts_ is None:
            lb, n = ndimage.label(L)
            if n < 2: return None
            sz = ndimage.sum(L, lb, range(1, n + 1)); parts_ = [lb == (i + 1) for i in np.argsort(sz)[-2:]]
        parts_ = [q for q in parts_ if q is not None and q.any()]
        if len(parts_) < 2: return None
        xs = sorted(float((A3 @ np.asarray(ndimage.center_of_mass(q)))[0] + T[0]) for q in parts_[:2])
        L = parts_[0] | parts_[1]
        zl = Z[L]; return (X >= xs[0]) & (X <= xs[1]) & (Z >= zl.min() - d1 - m) & (Z <= zl.max() + d1 + m)
    pr = parts(L, rel, sp, A3, X, mid_x, lsign)
    if pr is None: return None
    dist, cosang = pr
    return (dist >= d0 - m) & (dist <= d1 + m) & (dist > 0) & (cosang >= np.cos(np.deg2rad(theta)))


def parts(L, rel, sp, A3, X, mid_x, lsign):
    """directional relation: (distance map to L in mm, cosine between v(p) and the relation direction)"""
    code = REL.get(rel)
    if code is None: return None
    dist, ind = ndimage.distance_transform_edt(~L, sampling=sp, return_indices=True)
    dist = dist.astype(np.float32); cosang = np.zeros(L.shape, np.float32)
    u = _dirs(code, X, mid_x, lsign)
    for z in range(L.shape[2]):                                        # slice-wise (memory)
        g = np.indices(L.shape[:2], dtype=np.float32)
        dv = np.stack([g[0] - ind[0, :, :, z], g[1] - ind[1, :, :, z], z - ind[2, :, :, z].astype(np.float32)]).reshape(3, -1)
        v = (A3.astype(np.float32) @ dv).reshape((3,) + L.shape[:2]); nv = np.sqrt((v ** 2).sum(0)); nv[nv == 0] = 1
        cosang[:, :, z] = (v * u[:, :, :, z]).sum(0) / nv
    del ind
    return dist, cosang


def _fit():
    import json, os
    f = os.path.join(os.path.expanduser('~'), 'mnt', 'clinical', 'PanCIA_outputs', 'ledger', 'v8', 'testbed', 'relational_fit.json')
    try:
        d = json.load(open(f)); return float(d['theta']), float(d['margin_mm']), float(d['extent_mm']), bool(d['shadow'])
    except Exception:
        return 45.0, 10.0, 50.0, True


def region(kb, ent, side, masks, aff, mid_x, lsign, base=None, lr_ok=True, return_info=False):
    """relational region (cone semantics, fitted theta / m / E / shadow) on the grid of `masks` (bool arrays, TS-style names; sided
    organs as <name>_<side>). base: slab AND side AND footprint (same grid). lr_ok False: constraints with an absolute left/right
    direction are skipped (handedness unreliable). Returns None when no constraint is available."""
    theta, m, E, shw = _fit()
    e = kb.entities.get(ent, {}); pri = (e.get('anchor') or {}).get('spatial_prior') or []
    shape = next(iter(v.shape for v in masks.values() if v is not None))
    A = np.asarray(aff, float); A3 = A[:3, :3]; sp = np.sqrt((A3 ** 2).sum(0)); T = A[:3, 3]
    idx = np.indices(shape, dtype=np.float32).reshape(3, -1); W = (A3.astype(np.float32) @ idx).reshape((3,) + shape) + T.reshape(3, 1, 1, 1).astype(np.float32); del idx
    X, Z = W[0], W[2]; del W
    cnt = np.zeros(shape, np.int16); used = []
    for s_ in pri:
        rel = s_['relation']
        if rel != 'between' and rel not in REL: continue
        if not lr_ok and rel != 'between' and any(c in REL[rel] for c in 'LR'): continue
        if mid_x is None and rel != 'between' and any(c in REL[rel] for c in 'XM'): continue     # no midline: lateral/medial undefined
        if rel == 'between':
            Lp = landmark_parts(s_['landmark'], masks)
            if len(Lp) < 2: continue
            c = constraint(Lp, 'between', s_['dist_mm'], sp, A3, X, Z, mid_x, lsign, theta, m + E, T=T)
        else:
            L = landmark(s_['landmark'], side, masks)
            if L is None: continue
            if isinstance(L, tuple): sac = L[1]; L = sac & (Z >= Z[sac].max() - 15.0)
            c = constraint_cone(L, rel, s_['dist_mm'], sp, A3, X, Z, mid_x, lsign, theta, m, E, T=T, shadow=shw)
        if c is None: continue
        cnt += c; used.append(s_['landmark'] + ':' + rel)
    if not used:
        return (None, dict(used=[])) if return_info else None
    if base is not None: cnt[~base] = 0
    k = int(cnt.max()); out = (cnt == k) if k > 0 else None
    info = dict(used=used, n_cons=len(used), n_sat=k, theta=theta, m=m, E=E, shadow=shw)
    return (out, info) if return_info else out

def _cone_kernel(u, theta, R, A3):
    """offsets w (voxel grid) with |A3 w| <= R and angle(A3 w, u) <= theta (u world unit vector)"""
    sp = np.sqrt((A3 ** 2).sum(0)); r = np.ceil(R / sp).astype(int)
    g = np.indices(tuple(2 * r + 1), dtype=np.float32).reshape(3, -1) - r.reshape(3, 1)
    w = A3.astype(np.float32) @ g; nw = np.sqrt((w ** 2).sum(0)); nw[nw == 0] = 1
    k = (np.sqrt((w ** 2).sum(0)) <= R) & ((u.reshape(3, 1) * w).sum(0) / nw >= np.cos(np.deg2rad(theta)))
    return k.reshape(tuple(2 * r + 1)).astype(np.float32)


def cone_dilate(L, u, theta, R, A3):
    from scipy.signal import fftconvolve
    k = _cone_kernel(np.asarray(u, np.float32) / max(np.linalg.norm(u), 1e-6), theta, R, A3)
    return fftconvolve(L.astype(np.float32), k, mode='same') > 0.5


def _code_vec(code, lat_sign, lsign):
    """world unit vector of a relation code; lat_sign = +1 if 'lateral' points to patient left at the voxels considered, -1 right"""
    u = np.zeros(3, np.float32)
    for c in code:
        if c == 'A': u[1] += 1
        elif c == 'P': u[1] -= 1
        elif c == 'S': u[2] += 1
        elif c == 'I': u[2] -= 1
        elif c == 'L': u[0] += lsign
        elif c == 'R': u[0] -= lsign
        elif c == 'X': u[0] += lsign * lat_sign
        elif c == 'M': u[0] -= lsign * lat_sign
    return u


def _shadow(L, u, A3, T, tol=16.0, binmm=6.0):
    """voxels whose projection along u falls on the landmark's projection (dilated by tol mm): the landmark lies along the axis"""
    u = np.asarray(u, float); u = u / max(np.linalg.norm(u), 1e-9)
    e1 = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    idx = np.indices(L.shape, dtype=np.float32).reshape(3, -1)
    W = A3.astype(np.float32) @ idx + np.asarray(T, np.float32).reshape(3, 1); del idx
    s1 = e1.astype(np.float32) @ W; s2 = e2.astype(np.float32) @ W; del W
    lo1, lo2 = s1.min(), s2.min()
    b1 = ((s1 - lo1) / binmm).astype(np.int32); b2 = ((s2 - lo2) / binmm).astype(np.int32)
    G = np.zeros((b1.max() + 1, b2.max() + 1), bool); lf = L.ravel(); G[b1[lf], b2[lf]] = True
    r = int(np.ceil(tol / binmm)); G = ndimage.binary_dilation(G, iterations=r) if r > 0 else G
    return G[b1, b2].reshape(L.shape)


def landmark_parts(name, masks):
    """left and right masks of a paired landmark (for 'between')"""
    ks = LM.get(name, [name])
    return [masks.get(k) for k in ks if masks.get(k) is not None and masks.get(k).any()]


def constraint_cone(L, rel, d01, sp, A3, X, Z, mid_x, lsign, theta, m, E, T=(0.0, 0.0, 0.0), dist=None, shadow=True):
    """crisp directional-cone constraint (see module docstring); 'between' as before with extension d1 + m + E.
    shadow=True: the relation is only informative where the landmark lies along the relation axis (p in L + cone(+-u, theta, any
    distance)); voxels beside the landmark (outside that shadow) satisfy it (the relation does not speak about them)."""
    d0, d1 = float(d01[0]), float(d01[1])
    if rel == 'between':
        return constraint(L, 'between', d01, sp, A3, X, Z, mid_x, lsign, theta, m + E, T=T)
    code = REL.get(rel)
    if code is None: return None
    R = d1 + m + E
    if dist is None: dist = ndimage.distance_transform_edt(~L, sampling=sp)
    near = (dist >= d0 - m) & (dist > 0)
    def one(u):
        c = cone_dilate(L, u, theta, R, A3)
        if shadow:
            c = c | ~_shadow(L, u, A3, T, tol=m + 6.0)
        return c
    if 'X' in code or 'M' in code:
        left = (lsign * (X - mid_x)) > 0
        return near & np.where(left, one(_code_vec(code, +1, lsign)), one(_code_vec(code, -1, lsign)))
    return near & one(_code_vec(code, +1, lsign))
