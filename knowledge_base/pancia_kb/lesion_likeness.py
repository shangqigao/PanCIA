"""Candidate-intrinsic "lesion-likeness" image evidence (R2 v6, frozen 12 Oct 2026; confirmed on E3).

Does the candidate region look like a lesion (a compact, focal, internally structured object), regardless of what
the agent says it is and without comparing it with a host reference? Four terms, each in [0, 1], higher = more
lesion-like; all are invariant to affine intensity changes (no MR normalisation needed):

  elong       sqrt(l3/l1) of the main component's inertia (pancia_kb.geometry.shape_descriptors)
  blobby      mean |l_small| / |l_large| of the Hessian at sigma = clip(r_eq/2, 1.5, 4 mm) over the eroded candidate
              (2-D in-plane when slices are thick, max/min spacing >= 1.5)
  core_focal  max over multiscale LoG extrema (sigma 2, 4, 7, 11 mm, both signs) inside the candidate (dilated 1 voxel) of
              (enclosure + min(cnr_tex, 2)/2) / 2 with core = ball(0.6 r), ring = shell(1.3 r, 2.0 r), r = sqrt(d)*sigma;
              NaN when no extremum lies inside the candidate
  het_s       het/(het+5), het = IQR(candidate) / noise sd (MAD of the Laplacian / sqrt 20, window = bbox + 3 voxels)

B = mean of the available terms (frozen form: all four; E3 AUC 0.85 [0.76, 0.93] with all four, 0.81 with available terms).
Code mirrors clinical/PanCIA_outputs/scripts/v6_features.py, v6_corefocal.py and v6_structmatch.py exactly.
"""
import numpy as np
from scipy import ndimage
from . import geometry as G

SIG = (2.0, 4.0, 7.0, 11.0)


def _crop(shape, M, sp, pad_mm):
    idx = np.argwhere(M); lo, hi = idx.min(0), idx.max(0) + 1
    pad = np.ceil(pad_mm / sp).astype(int) if pad_mm else np.array([3, 3, 3])
    lo = np.maximum(lo - pad, 0); hi = np.minimum(hi + pad, shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _noise(X):
    lap = ndimage.laplace(ndimage.gaussian_filter(X, 0.5))
    return 1.4826 * np.median(np.abs(lap - np.median(lap))) / np.sqrt(20) + 1e-6


def geometry_terms(M, sp):
    s = G.shape_descriptors(M, sp, min_comp_ml=0.05)
    if not s: return {}
    return dict(vol_ml=s['vol_ml'], r_eq_mm=s['r_eq_mm'], sphericity=s['sphericity'], elong=s['elong'], flat=s['flat'])


def blobby(X, M, sp, ax, thick, r_eq):
    st = np.zeros((3, 3, 3), bool); sl = [1, 1, 1]
    for a in range(3):
        if a != ax: sl[a] = slice(None)
    st[tuple(sl)] = True
    core = ndimage.binary_erosion(M, structure=st) if M.sum() > 50 else M
    if core.sum() < 5: core = M
    hs = float(np.clip(r_eq / 2, 1.5, 4.0)); sg = [0 if (thick and a == ax) else hs / sp[a] for a in range(3)]
    Xs = ndimage.gaussian_filter(X, sg); dims = [a for a in range(3) if not (thick and a == ax)]
    Hm = np.zeros(X.shape + (len(dims), len(dims)), np.float32)
    gr = {a: np.gradient(Xs, sp[a], axis=a) for a in dims}
    for ii, a in enumerate(dims):
        for jj, b in enumerate(dims):
            if jj < ii: continue
            h = np.gradient(gr[a], sp[b], axis=b); Hm[..., ii, jj] = h; Hm[..., jj, ii] = h
    ea = np.sort(np.abs(np.linalg.eigvalsh(Hm[core])), 1) + 1e-9
    return float(np.mean(ea[:, 0] / ea[:, -1])), float(np.mean(1 - ea[:, 0] / ea[:, 1]))  # blobby, tube (as v6_features.py)


def core_focal(X, M, sp, ax, thick):
    dims = [a for a in range(3) if not (thick and a == ax)]; k = len(dims)
    best = (-1, None, None, None); n_ext = 0; Mc = ndimage.binary_dilation(M, iterations=1)
    for s in SIG:
        sig = [0 if (thick and a == ax) else s / sp[a] for a in range(3)]
        Lg = ndimage.gaussian_laplace(X, sig) * s * s
        for sign in (1, -1):
            R = -sign * Lg
            pk = (R == ndimage.maximum_filter(R, size=[1 if sg == 0 else max(3, int(2 * s / sp[a])) for a, sg in enumerate(sig)])) & Mc & (R > 0)
            pc = np.argwhere(pk)
            if len(pc) == 0: continue
            pc = pc[np.argsort(-R[tuple(pc.T)])][:4]
            rad = (np.sqrt(2) if k == 2 else np.sqrt(3)) * s
            for c in pc:
                n_ext += 1
                sl = tuple(slice(max(0, int(c[a] - np.ceil(2.2 * rad / sp[a]))), int(c[a] + np.ceil(2.2 * rad / sp[a])) + 1) if a in dims else slice(c[a], c[a] + 1) for a in range(3))
                sub = X[sl]; g = np.indices(sub.shape).reshape(3, -1).T + np.array([s_.start for s_ in sl])
                dd = np.sqrt((((g - c) * sp)[:, dims] ** 2).sum(1)); v = sub.ravel()
                core = v[dd <= 0.6 * rad]; ring = v[(dd >= 1.3 * rad) & (dd <= 2.0 * rad)]
                if len(core) < 5 or len(ring) < 10: continue
                p10, p90 = np.percentile(core, [10, 90]); enc = ((ring < p10) | (ring > p90)).mean()
                ct = abs(np.median(core) - np.median(ring)) / (np.subtract(*np.percentile(ring, [75, 25])) + 1e-6)
                sc = (enc + min(ct, 2) / 2) / 2
                if sc > best[0]:
                    best = (sc, rad, None, sign)
    return (float(best[0]) if best[1] is not None else np.nan), n_ext


def boundary_support(X, M, sp, ax, thick, sigma_mm=3.0, shell=(1.0, 3.0)):
    """Is the candidate boundary a real image edge? (defined 12 Oct 2026, before evaluation)
    Inner shell = candidate voxels 1-3 mm inside the boundary, outer shell = non-candidate voxels 1-3 mm outside (in-plane
    distances when slices are thick). Local means/variances of each shell by normalised Gaussian convolution (sigma 3 mm,
    in-plane if thick), read at the boundary voxels. Local edge z = |m_in - m_out| / sqrt((v_in + v_out)/2 + noise^2).
    Returns bsup = share of boundary voxels with z > 1, and bz = median z. An artificial cut through homogeneous tissue
    gives bsup ~ 0; a lesion (or any real object) bounded by contrast gives bsup ~ 1."""
    dims = [a for a in range(3) if not (thick and a == ax)]
    samp = [sp[a] if a in dims else 1e6 for a in range(3)]
    if thick:
        din = np.zeros(M.shape); dout = np.zeros(M.shape)
        for k in range(M.shape[ax]):
            sl = [slice(None)] * 3; sl[ax] = k; sl = tuple(sl); m = M[sl]
            if m.any(): din[sl] = ndimage.distance_transform_edt(m, sampling=[sp[a] for a in dims]); dout[sl] = ndimage.distance_transform_edt(~m, sampling=[sp[a] for a in dims])
            else: dout[sl] = 1e6
    else:
        din = ndimage.distance_transform_edt(M, sampling=sp); dout = ndimage.distance_transform_edt(~M, sampling=sp)
    Sin = M & (din > shell[0]) & (din <= shell[1]); Sout = ~M & (dout > shell[0]) & (dout <= shell[1])
    if Sin.sum() < 10: Sin = M & (din <= shell[1])
    bd = M & (din <= min(sp[a] for a in dims) * 1.01)
    if Sin.sum() < 5 or Sout.sum() < 5 or bd.sum() < 5: return np.nan, np.nan
    sg = [0 if a not in dims else sigma_mm / sp[a] for a in range(3)]
    g = lambda v: ndimage.gaussian_filter(v, sg)
    w_in, w_out = g(Sin.astype(np.float32)), g(Sout.astype(np.float32))
    m_in = g(X * Sin) / (w_in + 1e-6); m_out = g(X * Sout) / (w_out + 1e-6)
    v_in = np.maximum(g(X * X * Sin) / (w_in + 1e-6) - m_in ** 2, 0); v_out = np.maximum(g(X * X * Sout) / (w_out + 1e-6) - m_out ** 2, 0)
    ok = bd & (w_in > 0.02) & (w_out > 0.02)
    if ok.sum() < 5: return np.nan, np.nan
    nz = _noise(X)
    z = np.abs(m_in[ok] - m_out[ok]) / np.sqrt((v_in[ok] + v_out[ok]) / 2 + nz ** 2)
    return float((z > 1).mean()), float(np.median(z))


def lesion_features(img, M, sp, ax_si=2):
    """img: full 3-D intensity array (or a proxy supporting slicing); M: full-size bool candidate mask; sp: spacing (mm);
    ax_si: the superior-inferior axis index, used when slices are isotropic-ish."""
    sp = np.asarray(sp, float); thick = sp.max() / sp.min() >= 1.5; ax = int(np.argmax(sp)) if thick else int(ax_si)
    out = dict(max_sp=float(sp.max()))
    if M.sum() < 5: out['note'] = 'tiny'; return out
    out.update(geometry_terms(M, sp)); r_eq = out.get('r_eq_mm', 5.0)
    W = _crop(M.shape, M, sp, 60); X = np.asarray(img[W], np.float32); Mw = M[W]
    out['blobby'], out['tube'] = blobby(X, Mw, sp, ax, thick, r_eq)
    W2 = _crop(M.shape, M, sp, 12); X2 = np.asarray(img[W2], np.float32)
    out['bsup'], out['bz'] = boundary_support(X2, M[W2], sp, ax, thick)
    W = _crop(M.shape, M, sp, 45); X = np.asarray(img[W], np.float32)
    out['core_focal'], out['n_ext'] = core_focal(X, M[W], sp, ax, thick)
    W = _crop(M.shape, M, sp, 0); X = np.asarray(img[W], np.float32)
    out['het'] = float(np.subtract(*np.percentile(X[M[W]], [75, 25])) / _noise(X))
    return out


def score(f):
    """Lesion-likeness B in [0, 1]: mean of the available terms."""
    het = f.get('het', np.nan)
    t = [f.get('elong', np.nan), f.get('blobby', np.nan), f.get('core_focal', np.nan), het / (het + 5) if np.isfinite(het) else np.nan]
    t = [v for v in t if v is not None and np.isfinite(v)]
    return float(np.mean(t)) if len(t) >= 3 else np.nan


def tubular_continuity(img, M, sp, ax_si=2, pad_mm=45.0, max_ml=100.0):
    """3-D tubular continuity (frozen in ledger/v6/FROZEN_CONTINUITY_v63_20261012.txt): does the candidate continue beyond its contour
    as a line-like structure of the same intensity? Returns dict(cont_len [mm, 0-40], n_exits, cont_vol). Higher = more normal-like."""
    sp = np.asarray(sp, float); thick = sp.max() / sp.min() >= 1.5; ax = int(np.argmax(sp)) if thick else int(ax_si)
    vol = M.sum() * np.prod(sp) / 1000
    if M.sum() < 5 or vol > max_ml: return dict(cont_len=np.nan, n_exits=np.nan, cont_vol=np.nan)
    W = _crop(M.shape, M, sp, pad_mm); X = np.asarray(img[W], np.float32); m = M[W]
    dims = [a for a in range(3) if not (thick and a == ax)]
    noise = _noise(X)
    Xs = ndimage.gaussian_filter(X, [0 if a not in dims else 1.0 / sp[a] for a in range(3)])
    st = np.zeros((3, 3, 3), bool); sl = [1, 1, 1]
    for a in dims: sl[a] = slice(None)
    st[tuple(sl)] = True
    core = ndimage.binary_erosion(m, structure=st)
    if core.sum() < 5: core = m
    cv = Xs[core]; med = np.median(cv); half = max(np.subtract(*np.percentile(cv, [75, 25])), 1.5 * noise)
    band = np.abs(Xs - med) <= half
    dM = ndimage.distance_transform_edt(~m, sampling=sp)
    roi = (~m) & (dM <= 40) & band
    line = np.zeros(X.shape, bool)
    for s in (1.5, 3.0, 5.0):
        sig = [0 if a not in dims else s / sp[a] for a in range(3)]
        comps = {}
        for i, a in enumerate(dims):
            for j, b in enumerate(dims):
                if j < i: continue
                order = [0, 0, 0]; order[a] += 1; order[b] += 1
                comps[(i, j)] = ndimage.gaussian_filter(X, sig, order=order) / (sp[a] * sp[b])
        k = len(dims); idx = np.nonzero(roi)
        if len(idx[0]) == 0: break
        Hm = np.zeros((len(idx[0]), k, k), np.float32)
        for (i, j), h in comps.items():
            Hm[:, i, j] = h[idx]; Hm[:, j, i] = h[idx]
        ev = np.sort(np.abs(np.linalg.eigvalsh(Hm)), 1)
        ok = (ev[:, 0] / (ev[:, 1] + 1e-9) < 0.4) & (s * s * ev[:, 1] > 3 * noise)
        line[tuple(i_[ok] for i_ in idx)] = True
    T = line & roi
    lab, n = ndimage.label(T, structure=np.ones((3, 3, 3), bool))
    touch = np.unique(lab[ndimage.binary_dilation(m, structure=np.ones((3, 3, 3), bool)) & T]); touch = touch[touch > 0]
    if len(touch) == 0: return dict(cont_len=0.0, n_exits=0, cont_vol=0.0)
    mx = ndimage.maximum(dM, lab, touch); ext = np.isin(lab, touch)
    return dict(cont_len=float(min(np.max(mx), 40.0)), n_exits=int((np.asarray(mx) >= 5).sum()), cont_vol=float(ext.sum() / m.sum()))
