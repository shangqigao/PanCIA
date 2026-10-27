"""Tail study (review only; nothing here enters the stage-2 fit): how heavy-tailed are tumour interiors and the normal tissue around
them, by order of the statistic, and is one tail shape shared across orders?

Statistics, in-plane and signed, on a 3 mm grid (pancia_kb.stage2.tails.statistics):
  I         normalised intensity (order 0)
  D1        forward first difference (order 1, no smoothing; both in-plane axes pooled)
  G1_s<k>   Gaussian derivative, sigma = k mm (order 1; k = 1, 2, 4)
  D2        discrete Laplacian (order 2, no smoothing)
  LoG_s<k>  Laplacian of Gaussian, sigma = k mm (order 2; k = 1, 2, 4)
A voxel enters a statistic only when its clearance (distance to the edge of the region's context: the tumour for a tumour interior,
host organ and tumour for normal tissue) is >= max(3 mm, 2 sigma), so no stencil or kernel reaches across an edge.

Fits (one class, every voxel belongs to it; stage-2 VB, within-scan Normal-Wishart prior):
  fit_1d          Gaussian and Student-t (nu learned), all parameters + lower bounds
  fit_components  1..4 Student-t components, K by the lower bound, nu per component (fit.fit_tumour_components)
  model_form      joint features (I, D1 two axes, LoG_s1): Gaussian full / block-diagonal, multivariate t with ONE shared scale u
                  (one nu) vs order-specific t blocks (I | D1x,D1y | LoG, each its own u and nu)."""
import numpy as np
from scipy import ndimage
from . import vb, fit as FT

SCALES = (1.0, 2.0, 4.0)
STATS = ('I', 'D1', 'D2') + tuple(f'G1_s{int(s)}' for s in SCALES) + tuple(f'LoG_s{int(s)}' for s in SCALES)
ORDER = {'I': 0, 'D1': 1, 'D2': 2, **{f'G1_s{int(s)}': 1 for s in SCALES}, **{f'LoG_s{int(s)}': 2 for s in SCALES}}
BLOCKS = ((0,), (1, 2), (3,))          # model form: I | D1 (two in-plane axes) | LoG_s1
JOINT = ('I', 'D1a', 'D1b', 'LoG_s1')


def grid_mask(shape, sp, mm=3.0):
    step = [max(1, int(round(mm / s))) for s in sp]; g = np.zeros(shape, bool); g[::step[0], ::step[1], ::step[2]] = True
    return g


def grid_slices(n, s, mm=3.0):
    """indices of the through-plane slices on the 3 mm grid. Every statistic is in-plane (no filtering across slices), so only these
    slices are filtered (saves the factor step on thin-slice scans)."""
    return np.arange(0, n, max(1, int(round(mm / s))))


def border_mm(shape, sp):
    """distance (mm) to the nearest face of the crop: clearance is capped by it (np.roll and the filters wrap or pad there)"""
    d = np.full(shape, np.inf, np.float32)
    for a, (n, s) in enumerate(zip(shape, sp)):
        i = np.arange(n, dtype=np.float32); v = np.minimum(i, n - 1 - i) * s
        sh = [1, 1, 1]; sh[a] = n; d = np.minimum(d, v.reshape(sh))
    return d


def _diff(X, a):
    D = np.zeros(X.shape, np.float32); lo = [slice(None)] * 3; hi = [slice(None)] * 3; lo[a] = slice(0, -1); hi[a] = slice(1, None)
    D[tuple(lo)] = X[tuple(hi)] - X[tuple(lo)]
    return D


def laplacian(X, ax):
    """in-plane 4-neighbour discrete Laplacian"""
    X = X.astype(np.float32); L = np.zeros(X.shape, np.float32)
    for a in range(3):
        if a != ax:
            L += np.roll(X, 1, a) + np.roll(X, -1, a) - 2 * X
    return L


def filters(X, sp, ax):
    """the statistic maps on the crop (dict name -> list of per-axis maps, or one map), and the clearance each needs (mm)"""
    ip = [a for a in range(3) if a != ax]; X = X.astype(np.float32)
    maps = {'I': ([X], 3.0), 'D1': ([_diff(X, a) for a in ip], 3.0)}
    maps['D2'] = ([laplacian(X, ax)], 3.0)
    for s in SCALES:
        sig = [s / sp[a] if a != ax else 0.0 for a in range(3)]
        g1 = []; lg = np.zeros(X.shape, np.float32)
        for a in ip:
            o = [0, 0, 0]; o[a] = 1; g1.append(ndimage.gaussian_filter(X, sig, order=o))
            o = [0, 0, 0]; o[a] = 2; lg += ndimage.gaussian_filter(X, sig, order=o)
        maps[f'G1_s{int(s)}'] = (g1, max(3.0, 2 * s)); maps[f'LoG_s{int(s)}'] = ([lg], max(3.0, 2 * s))
    return maps


def statistics(maps, R, C, grid):
    """1-D samples per statistic on region R (bool) with clearance map C (mm)"""
    out = {}
    for k, (ms, need) in maps.items():
        V = R & (C >= need) & grid
        out[k] = np.concatenate([m[V] for m in ms]).astype(np.float32)
    return out


def joint(maps, R, C, grid):
    """(n, 4) joint features (I, D1 axis a, D1 axis b, LoG_s1) at the same voxels (clearance >= 3 mm)"""
    V = R & (C >= 3.0) & grid
    return np.stack([maps['I'][0][0][V], maps['D1'][0][0][V], maps['D1'][0][1][V], maps['LoG_s1'][0][0][V]], 1).astype(np.float64)


def noise_sd(maps, R, C):
    """robust noise level of the normalised image in normal tissue (Immerkaer-type: 4-neighbour Laplacian, var = 20 sigma^2)"""
    v = maps['D2'][0][0][R & (C >= 3.0)]
    if len(v) < 50:
        return None
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(20.0))


def _one(Y, learn):
    n = len(Y); tp, bps = vb.class_priors(Y, np.ones(n), np.ones((n, 1)))
    m = vb.Model(Y, np.ones(n), np.zeros(n, int), np.ones((1, 1), bool), tp, bps, K=1, alpha0=0.25, nu=(10.0 if learn else 1e6),
                 tumour=True, learn_nu=learn)
    m.init(np.ones(n)).run(max_iter=2000)
    post = m.post[0]; El = post.nu * post.W                         # E[Lambda]
    return m, post, El


def fit_1d(y):
    """Gaussian vs Student-t on a 1-D sample; same fields as review/tails2_fits.csv"""
    y = np.asarray(y, float); n = len(y); Y = y[:, None]
    med = float(np.median(y)); out = dict(n=n, med=med, mad=float(1.4826 * np.median(np.abs(y - med))))
    m, post, El = _one(Y, False); out.update(g_mu=float(post.m[0]), g_sd=float(1 / np.sqrt(El[0, 0])), g_elbo=float(m.elbo))
    m, post, El = _one(Y, True)
    out.update(nu=float(m.nu_k[0]), t_mu=float(post.m[0]), t_scale=float(1 / np.sqrt(El[0, 0])), t_elbo=float(m.elbo))
    out['dElbo'] = out['t_elbo'] - out['g_elbo']
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def fit_components(Y):
    Y = np.asarray(Y, float); Y = Y[:, None] if Y.ndim == 1 else Y
    r = FT.fit_tumour_components(Y)
    return dict(n=len(Y), K=r['K'], elbos=r['elbos'], components=[{k: round(v, 4) for k, v in c.items()} for c in r['components']])


def model_form(Yj):
    """shared-scale multivariate t vs order-specific t blocks on the joint features. Gaussian full and block-diagonal are fitted too:
    pref = (t_block - g_block) - (t_shared - g_full) is the gain of order-specific tails over one shared tail, with the effect of
    ignoring cross-block correlation (g_full - g_block) taken out (both pairs differ only in the tail model)."""
    n = len(Yj); out = dict(n=n)
    m, post, El = _one(Yj, False); out['g_full'] = float(m.elbo)
    m, post, El = _one(Yj, True); out['t_shared'] = float(m.elbo); out['nu_shared'] = float(m.nu_k[0])
    out['corr'] = np.round(np.corrcoef(Yj.T), 3).tolist()
    gb = tb = 0.0; nus = []
    for b in BLOCKS:
        mg, _, _ = _one(Yj[:, b], False); mt, _, _ = _one(Yj[:, b], True)
        gb += float(mg.elbo); tb += float(mt.elbo); nus.append(float(mt.nu_k[0]))
    out.update(g_block=gb, t_block=tb, nu_I=nus[0], nu_D1=nus[1], nu_LoG=nus[2])
    out['d_shared'] = out['t_shared'] - out['g_full']; out['d_block'] = out['t_block'] - out['g_block']
    out['pref_block'] = out['d_block'] - out['d_shared']; out['corr_gain'] = out['g_full'] - out['g_block']
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
