"""Image features y_v and helpers (stage 2).

y_v (D = 2): intensity smoothed in the acquisition plane (as grounding.smooth: sigma 1 voxel in-plane, none across thick slices) and
the local standard deviation at 2 mm (Gaussian window, isotropic in mm; texture). Local means and a second, 6 mm scale were dropped
(23 Oct): on the phantom (s2_selftest.py) they blur tumour boundaries, so normal tissue next to a tumour looked like tumour
(Dice 0.92 with them, 0.99 without).
CT stays in HU (clipped to [-1024, 3071]). MR has no absolute scale: it is normalised per scan by a robust z-score over the body,
z = (x - median_body) / (IQR_body / 1.349). Both are fixed rules; nothing is tuned.
The body is the largest connected component above a threshold (CT: -500 HU; MR: Otsu on the scan), with holes filled slice by slice."""
import numpy as np
from scipy import ndimage

SD_MM = 2.0
D = 2
NAMES = ['xs', 's2']


def edt(m, sp):
    """distance (mm) from every voxel to the nearest True voxel of m (0 inside m); inf if m is empty"""
    m = np.asarray(m, bool)
    if not m.any():
        return np.full(m.shape, np.inf, np.float32)
    return ndimage.distance_transform_edt(~m, sampling=np.asarray(sp, float)).astype(np.float32)


def inner_depth(m, sp):
    """distance (mm) from a voxel inside m to the nearest voxel outside m (0 outside)"""
    m = np.asarray(m, bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    return ndimage.distance_transform_edt(m, sampling=np.asarray(sp, float)).astype(np.float32)


def signed_distance(m, sp):
    """negative inside m (depth), positive outside (distance), mm"""
    return np.where(m, -inner_depth(m, sp), edt(m, sp)).astype(np.float32)


def dil(m, sp, mm):
    return edt(m, sp) <= mm if np.asarray(m).any() else np.zeros(np.shape(m), bool)


def acq_axis(sp, si):
    """thick-slice axis (largest spacing) when anisotropic (>= 1.5x), else the S-I axis"""
    sp = np.asarray(sp, float)
    thick = sp.max() / sp.min() >= 1.5
    return thick, (int(np.argmax(sp)) if thick else int(si))


def _otsu(v, nb=256):
    h, e = np.histogram(v, bins=nb); c = (e[:-1] + e[1:]) / 2; h = h.astype(float)
    w0 = np.cumsum(h); w1 = w0[-1] - w0; m0 = np.cumsum(h * c); mt = m0[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        mu0 = m0 / w0; mu1 = (mt - m0) / w1; s = w0 * w1 * (mu0 - mu1) ** 2
    s[~np.isfinite(s)] = -1
    return float(c[int(np.argmax(s))])


def body_mask(X, mod, ax):
    """largest component above threshold, holes filled per slice along axis ax"""
    X = np.asarray(X, np.float32)
    if mod == 'CT':
        thr = -500.0
    else:
        st = tuple(slice(None, None, max(1, n // 128)) for n in X.shape)
        v = X[st].ravel(); v = v[np.isfinite(v)]
        thr = _otsu(v[v > np.percentile(v, 1)])
    B = X > thr
    lab, n = ndimage.label(B)
    if n == 0:
        return np.ones(X.shape, bool)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    B = lab == int(np.argmax(sz))
    for z in range(B.shape[ax]):
        sl = [slice(None)] * 3; sl[ax] = z
        B[tuple(sl)] = ndimage.binary_fill_holes(B[tuple(sl)])
    return B


def region_body(X, mod, ax):
    """mask that limits stage-2 regions and their growth. CT: the body mask. MR: no limit (all True): an intensity threshold drops
    hypo-intense tissue on MR (T2 tumours and organs fell outside an Otsu body on R2Seg cervix), and the posterior does not spread
    into air anyway."""
    return body_mask(X, mod, ax) if mod == 'CT' else np.ones(np.shape(X), bool)


def norm_params(X, mod, body):
    """(centre, scale) of the intensity normalisation. CT: (0, 1). MR: robust z over the body."""
    if mod == 'CT':
        return 0.0, 1.0
    v = np.asarray(X, np.float32)[body]
    if v.size > 2_000_000:
        v = v[:: v.size // 2_000_000]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    s = (q3 - q1) / 1.349
    return float(med), float(s if s > 0 else (v.std() or 1.0))


def normalise(X, mod, centre, scale):
    X = np.asarray(X, np.float32)
    if mod == 'CT':
        return np.clip(X, -1024.0, 3071.0)
    return (X - centre) / scale


def image_features(Xn, sp, thick, ax):
    """Xn: normalised intensity on a window (pad the window by >= 6 mm to avoid edge effects). Returns (D, *shape) float32."""
    Xn = np.asarray(Xn, np.float32); sp = np.asarray(sp, float)
    sg = [0.0 if (thick and a == ax) else 1.0 for a in range(3)]
    out = np.empty((D,) + Xn.shape, np.float32)
    out[0] = ndimage.gaussian_filter(Xn, sg)
    sig = [SD_MM / q for q in sp]
    m = ndimage.gaussian_filter(Xn, sig); m2 = ndimage.gaussian_filter(Xn * Xn, sig)
    out[1] = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return out


def pad_box(sl, sp, mm, shape):
    p = [int(np.ceil(mm / float(q))) for q in sp]
    return tuple(slice(max(0, s.start - a), min(n, s.stop + a)) for s, a, n in zip(sl, p, shape))


def bbox(m):
    idx = np.nonzero(np.asarray(m))
    if len(idx[0]) == 0:
        return None
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
