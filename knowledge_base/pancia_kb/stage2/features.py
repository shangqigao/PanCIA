"""Image features y_v and helpers (stage 2).

y_v (D = 4; 26 Oct, from the tail study, main.tex Appendix C): statistics of order 0, 1 and 2 in the acquisition plane, exactly as
the tail study that estimated the tail-weight prior (pancia_kb.stage2.tails):
  I     normalised intensity (no smoothing)
  D1a   forward first difference along the first in-plane axis
  D1b   forward first difference along the second in-plane axis
  LoG1  in-plane Laplacian of Gaussian, sigma = 1 mm
In-plane: the two axes other than the acquisition axis (acq_axis). Derivatives describe spatial change and depend less on the
imaging protocol than the intensity level; one Student-t scale u_v is shared by all four (vb), which fits expert data better than
a scale per order. Before 26 Oct: D = 2 (intensity smoothed in-plane by 1 voxel and the local SD at 2 mm).
CT stays in HU (clipped to [-1024, 3071]). MR has no absolute scale: it is normalised per scan by a robust z-score over the body,
z = (x - median_body) / (IQR_body / 1.349). Both are fixed rules; nothing is tuned.
The body is the largest connected component above a threshold (CT: -500 HU; MR: Otsu on the scan), with holes filled slice by slice."""
import numpy as np
from scipy import ndimage

LOG_SIGMA_MM = 1.0
D = 4
NAMES = ['I', 'D1a', 'D1b', 'LoG1']


def edt(m, sp):
    """distance (mm) from every voxel to the nearest True voxel of m (0 inside m); inf if m is empty"""
    m = np.asarray(m, bool)
    if not m.any():
        return np.full(m.shape, np.inf, np.float32)
    return ndimage.distance_transform_edt(~m, sampling=np.asarray(sp, float)).astype(np.float32)


def inner_depth(m, sp):
    """distance (mm) from a voxel inside m to the nearest voxel outside m (0 outside). Computed on m's bounding box + 1 voxel
    (identical values; fast with many small organ masks)."""
    m = np.asarray(m, bool)
    out = np.zeros(m.shape, np.float32)
    b = bbox(m)
    if b is None:
        return out
    W = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(b, m.shape))
    sub = m[W]
    if sub.all():                                   # the mask fills its padded box (touches the volume edge): no outside voxel
        return ndimage.distance_transform_edt(np.pad(m, 1), sampling=np.asarray(sp, float))[1:-1, 1:-1, 1:-1].astype(np.float32)
    out[W] = ndimage.distance_transform_edt(sub, sampling=np.asarray(sp, float))
    return out


def signed_distance(m, sp):
    """negative inside m (depth), positive outside (distance), mm"""
    return np.where(m, -inner_depth(m, sp), edt(m, sp)).astype(np.float32)


def through_plane_distance(m, sp, ax):
    """for every voxel outside m: |offset along axis ax| (mm) to its nearest voxel of m (Euclidean nearest, sampling sp); 0 inside m;
    60 everywhere when m is empty (the prior basis clips at 60 mm)"""
    from scipy import ndimage as _nd
    m = np.asarray(m, bool)
    if not m.any():
        return np.full(m.shape, 60.0, np.float32)
    _, ind = _nd.distance_transform_edt(~m, sampling=np.asarray(sp, float), return_indices=True)
    pos = np.arange(m.shape[ax]).reshape([-1 if k == ax else 1 for k in range(3)])
    out = (np.abs(ind[ax] - pos) * float(sp[ax])).astype(np.float32)
    out[m] = 0.0
    return out


def dil(m, sp, mm):
    return edt(m, sp) <= mm if np.asarray(m).any() else np.zeros(np.shape(m), bool)


MAX_VOX = 3e6     # crops above this many voxels are strided before stage 2 (testbed rule; the same on R2Seg and TCGA)


def stride_rule(sp, nvox, max_vox=MAX_VOX):
    """stride of a crop of nvox voxels at spacing sp: None when nvox <= max_vox; else (f, axes): every f-th voxel along axes (all
    three when near-isotropic, else the in-plane axes), f = ceil((nvox / max_vox) ** (1 / len(axes))). Used by expert._image_crop
    (R2Seg fitting and validation) and interface.stride_cluster (TCGA), so stage 2 runs at the resolution it was validated at."""
    if nvox <= max_vox:
        return None
    sp = np.asarray(sp, float)
    thin = sp.max() / sp.min() < 1.5
    axes = [0, 1, 2] if thin else [k for k in range(3) if k != int(np.argmax(sp))]
    return int(np.ceil((nvox / max_vox) ** (1.0 / len(axes)))), axes


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
    """Xn: normalised intensity on a window (pad the window by >= 3 mm to avoid edge effects). Returns (D, *shape) float32:
    I, D1a, D1b, LoG1 (see module doc). thick is unused (the plane is the one orthogonal to ax in every case, as the tail study)."""
    Xn = np.asarray(Xn, np.float32); sp = np.asarray(sp, float)
    ip = [a for a in range(3) if a != ax]
    out = np.empty((D,) + Xn.shape, np.float32)
    out[0] = Xn
    for i, a in enumerate(ip):
        d = np.zeros(Xn.shape, np.float32); lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[a] = slice(0, -1); hi[a] = slice(1, None)
        d[tuple(lo)] = Xn[tuple(hi)] - Xn[tuple(lo)]
        out[1 + i] = d
    sig = [LOG_SIGMA_MM / sp[a] if a != ax else 0.0 for a in range(3)]
    lg = np.zeros(Xn.shape, np.float32)
    for a in ip:
        o = [0, 0, 0]; o[a] = 2; lg += ndimage.gaussian_filter(Xn, sig, order=o)
    out[3] = lg
    return out


def pad_box(sl, sp, mm, shape):
    p = [int(np.ceil(mm / float(q))) for q in sp]
    return tuple(slice(max(0, s.start - a), min(n, s.stop + a)) for s, a, n in zip(sl, p, shape))


def bbox(m):
    idx = np.nonzero(np.asarray(m))
    if len(idx[0]) == 0:
        return None
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
