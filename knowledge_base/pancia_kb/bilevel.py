"""NOT ADOPTED (16 Oct 2026): research code. Versions v1-v5 failed E1 on R2Seg (ledger/v6/grounding/e1_*.csv; clean-mask Dice change
-0.06 to -0.23, no gain on perturbed masks); ceiling: the best voxel classifier on normative Z reaches Dice 0.67-0.85 vs experts.
Bi-level set image grounding (16 Oct 2026), plan: PanCIA/documents/plan/Image_grounding_v6_plan.md.
One run = (lesion, host, agent): tumour T initialised from one agent's outline(s), host H from the anatomy agent's organ mask.
Discrete (morphological) region competition on a local window, alternating T / H updates until balance:
  v4 (16 Oct, after smoke tests v1-v3): front evolution with hysteresis on the normative sign (see evolve); labels T / H / O,
  a voxel belongs to exactly one of T, H (overlap penalty is hard); Z = depth-standardised deviation from the current host
  parenchyma H \\ T (grounding.host_frame), re-estimated every z_every steps.
Global terms (elongation, closedness, D10 complete overlap) act at the decision stage (veto-only), see decide().
No parameter is fitted on TCGA."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from . import grounding as GR

PARAMS = dict(dom_mm=10.0, win_mm=50.0, host_grow_mm=3.0, env_mm=5.0, block_mm=3.0, dv=0.0, db=2.0, core_mm=3.0, t_host=1.0, t_tum=1.5, t_out=3.0, sign_frac=0.15, c_m=0.5, delta=1.0,
              sigma_mm=3.0, min_nb_frac=0.2, maxit=40, z_every=3, tol=0.002, maxvox=2.0e6, min_comp_ml=0.1, vanish_ml=0.1, vanish_frac=0.1)


def _down(shape, sp, thick, ax, maxvox):
    n = int(np.prod(shape))
    if n <= maxvox:
        return tuple(slice(None) for _ in range(3)), np.asarray(sp, float), 1
    nd = 2 if thick else 3; f = int(np.ceil((n / maxvox) ** (1 / nd)))
    st = tuple(slice(None, None, 1 if (thick and a == ax) else f) for a in range(3))
    return st, np.array([sp[a] if (thick and a == ax) else sp[a] * f for a in range(3)]), f


def _up(M, st, shape):
    out = M
    for a, s in enumerate(st):
        if s.step and s.step > 1:
            out = np.repeat(out, s.step, axis=a)
    return out[tuple(slice(0, n) for n in shape)]


def _fill(M, thick, ax):
    M = ndimage.binary_fill_holes(M)
    if thick:
        for z in range(M.shape[ax]):
            sl = [slice(None)] * 3; sl[ax] = z; M[tuple(sl)] = ndimage.binary_fill_holes(M[tuple(sl)])
    return M


def _clean(M, sp, min_ml):
    lb, n = ndimage.label(M)
    if n == 0:
        return M
    sz = np.bincount(lb.ravel()) * np.prod(sp) / 1000; keep = sz >= min_ml; keep[0] = False
    return keep[lb]


def _logdens(z, sample, zclip=30.0, nb=120):
    """log of a KDE-smoothed histogram density of `sample` (clipped), evaluated at z; 2% uniform floor"""
    c = zclip; w = 2 * c / nb
    if len(sample) < 20:
        return np.full(z.shape, -np.log(2 * c), np.float32)
    q25, q75 = np.percentile(sample, [25, 75]); bw = max(1.0, 0.15 * (q75 - q25) / 1.349 / w)
    h, _ = np.histogram(np.clip(sample, -c, c - 1e-6), bins=nb, range=(-c, c)); h = ndimage.gaussian_filter1d(h.astype(float), bw)
    p = 0.98 * h / h.sum() / w + 0.02 / (2 * c)
    k = np.clip(((np.clip(z, -c, c - 1e-6) + c) / w).astype(int), 0, nb - 1); return np.log(p)[k].astype(np.float32)


def _loggauss(z, sample):
    """log density of a robust Gaussian (median, IQR/1.349 with floors 1 and 0.2|median|) fitted to `sample`"""
    if len(sample) < 20:
        return np.zeros(z.shape, np.float32)
    q25, q50, q75 = np.percentile(sample, [25, 50, 75]); sd = max((q75 - q25) / 1.349, 1.0, 0.2 * abs(q50))   # floors: Z units; saturated tumours
    return (-0.5 * ((z - q50) / sd) ** 2 - np.log(sd)).astype(np.float32)


def _local_cv(X, M, sg, s):
    """>0 where the voxel is closer to the local mean inside M than to the local mean outside M (tanh scale, robust scale s)"""
    Mf = M.astype(np.float32); g_in = ndimage.gaussian_filter(Mf, sg) + 1e-4; g_out = ndimage.gaussian_filter(1 - Mf, sg) + 1e-4
    mi = ndimage.gaussian_filter(X * Mf, sg) / g_in; mo = ndimage.gaussian_filter(X * (1 - Mf), sg) / g_out
    return np.tanh((mi - mo) * (X - 0.5 * (mi + mo)) / (s * s))


def _crop_index(C, st):
    """index arrays mapping full-resolution crop C to the coarse (strided) grid"""
    return np.ix_(*[np.minimum(np.arange(c.start, c.stop) // (s.step or 1), 10 ** 9) for c, s in zip(C, st)])


def _blk(A, b, how):
    """reduce a crop array by integer block factors b (padded by edge values): 'mean' or 'frac' (mask fraction)"""
    pad = [(0, (-n) % k) for n, k in zip(A.shape, b)]
    A = np.pad(A.astype(np.float32), pad, mode='edge')
    sh = []
    for n, k in zip(A.shape, b):
        sh += [n // k, k]
    return A.reshape(sh).mean(axis=(1, 3, 5))


def _expand(M, b, shape):
    out = M
    for a, k in enumerate(b):
        out = np.repeat(out, k, axis=a)
    return out[tuple(slice(0, n) for n in shape)]


def evolve(X, sp, ax, T0, H0, les=None, P=None):
    """v5 (16 Oct): front evolution on a block lattice (block_mm), labels T (tumour) / H (host) / O (other), one per block.
    Why blocks: voxel-level decisions moved expert boundaries (partial volume, heterogeneous rims); the effective signs are
    region-level. A block's evidence is its mean Z; a block only changes when the whole block is clearly host or clearly tumour,
    and changes are applied block-wise at full resolution (unchanged blocks keep the initial outline voxel-exactly).
    Z (host frame, grounding.host_frame) is estimated on the strided window with depth measured inside the host envelope.
    Host zone = H0 + host_grow_mm plus the filled closing (env_mm) of H0 u T0 (tumour bed inside the host). Rules, front blocks only:
      in the host zone:   T -> H if log p_H(zb) - log p_T(zb) > delta (robust Gaussians of block-mean Z: p_T from the tumour core
                          blocks >= core_mm deep, p_H from the host blocks); H/O -> T if |zb| >= t_tum in a sign the tumour core
                          shows (>= sign_frac) and (from H) log p_T - log p_H > delta, (from O) local Chan-Vese cv > c_m.
                          A tumour indistinguishable from its host does not move.
      outside the zone:   T -> O if cv < -c_m (unlike the tumour locally, e.g. fat); no growth (leaks into mediastinum).
      host:               H -> O at the host front if |zb| >= t_out; O -> H if |zb| < t_host and log p_H - log p_T > delta (zone).
    Growth needs >= min_nb_frac tumour neighbours. T confined to Omega (T0 + dom_mm); holes filled; fragments < min_comp_ml dropped.
    Balance = no change in the step right after re-estimating Z and the class models (also every z_every steps)."""
    P = dict(PARAMS, **(P or {})); sp = np.asarray(sp, float); thick = sp.max() / sp.min() >= 1.5
    les = np.zeros(X.shape, bool) if les is None else les & ~T0
    Xs = GR.smooth(X, sp, thick, ax)
    st, spd, f = _down(X.shape, sp, thick, ax, P['maxvox'])
    idx = np.argwhere(T0); lo = idx.min(0); hi = idx.max(0) + 1; pd_ = np.ceil((P['dom_mm'] + 3) / sp).astype(int)
    C = tuple(slice(max(0, a - q), min(n, b + q)) for a, b, q, n in zip(lo, hi, pd_, X.shape)); ci = _crop_index(C, st)
    Xc, T0c, H0c, Lc = Xs[C], T0[C], H0[C], les[C]; cs = Xc.shape
    edt = lambda m, s_=sp: GR._edt(m, s_)
    Om = edt(~T0c) <= P['dom_mm']
    dil = edt(~(H0c | T0c)) <= P['env_mm']; env = ndimage.binary_fill_holes(edt(dil) > P['env_mm']) | H0c | T0c
    zone0 = Om & (env | (edt(~H0c) <= P['host_grow_mm']))
    b = [max(1, int(round(P['block_mm'] / v))) for v in sp]; spb = sp * np.array(b); thb = spb.max() / spb.min() >= 1.5
    axb = int(np.argmax(spb)) if thb else ax
    T0b = _blk(T0c, b, 'frac') >= 0.5; H0b = _blk(H0c & ~T0c, b, 'frac') >= 0.5; Lb = _blk(Lc, b, 'frac') >= 0.5
    Omb = _blk(Om, b, 'frac') >= 0.5; zoneb = _blk(zone0, b, 'frac') >= 0.5; Xb = _blk(Xc, b, 'mean')
    T = T0b.copy(); Hm = H0b & ~T0b
    ref = Xc[H0c & ~T0c & ~Lc]; ref = ref if ref.size >= 50 else Xc[Om & ~T0c]
    q25, q75 = np.percentile(ref, [25, 75]) if ref.size else (0, 1); s = max((q75 - q25) / 1.349, 1e-3)
    sg = P['sigma_mm'] / spb; fp = np.ones([1 if (thb and a == axb) else 3 for a in range(3)], bool); nbk = fp.astype(np.float32)
    lrv = np.zeros(cs, np.float32); cvv = np.zeros(cs, np.float32)

    def fullT(Tb):
        # block changes applied with voxel refinement: in a block that left T, drop the voxels that are host-like (lrv < -dv; in the
        # zone) or locally unlike the tumour (cvv < -c_m; outside the zone); in a block that joined T, add the tumour-like voxels
        rem = _expand(T0b & ~Tb, b, cs); add = _expand(Tb & ~T0b, b, cs) & Om
        drop = rem & ((zone0 & (lrv < -P['dv'])) | (~zone0 & (cvv < -P['c_m'])))
        Tc = (T0c & ~drop) | (add & (lrv > P['dv']))
        if P['db'] is not None:      # front blocks (T blocks touching non-T): clear voxel evidence also decides inside them
            fb = _expand(Tb & ~ndimage.binary_erosion(Tb, fp), b, cs); nb_ = _expand(ndimage.binary_dilation(Tb, fp) & ~Tb, b, cs) & Om
            Tc = (Tc & ~(fb & ((zone0 & (lrv < -P['db'])) | (~zone0 & (cvv < -P['c_m']))))) | (nb_ & zone0 & (lrv > P['db']))
        return Tc

    def fullH(Hb, Tc):
        rem = _expand(H0b & ~Hb, b, cs); add = _expand(Hb & ~H0b, b, cs) & Om
        return ((H0c & ~T0c & ~rem) | add) & ~Tc
    flags = set(); hist = []; it = 0; sgn = (True, True); recompute = False; fresh = False
    for it in range(P['maxit']):
        if not T.any():
            break
        if it % P['z_every'] == 0 or recompute:
            recompute = False; fresh = True
            Tc = fullT(T); Hc = fullH(Hm, Tc)
            Tw = T0.copy(); Tw[C] = Tc; Hw = H0 & ~T0; Hw[C] = Hc; Ew = np.zeros(X.shape, bool); Ew[C] = zone0 & ~Hc & ~Tc
            Td, Hd, Ld, Ed = Tw[st], Hw[st], les[st], Ew[st]
            Zd, D, nref = GR.host_frame(Xs[st], spd, Hd, Td | Ed, Hd & ~Ld)     # depth measured inside the host envelope
            if Zd is None:
                flags.add('no_normal_reference'); r_ = Xs[st][~Td & ~Ld]
                q25_, q50_, q75_ = np.percentile(r_, [25, 50, 75]) if r_.size else (0, 0, 1)
                Zd = ((Xs[st] - q50_) / max((q75_ - q25_) / 1.349, 1e-3)).astype(np.float32)
            Zb = _blk(Zd[ci], b, 'mean')
            dT = edt(T, spb); core = T & (dT >= P['core_mm']) & ~Lb
            zc = Zb[core] if core.sum() >= 20 else Zb[T & ~Lb]
            zh = Zb[Hm & ~Lb]; zh = zh if zh.size >= 20 else Zd[Hd & ~Ld]
            sgn = (bool((zc >= P['t_tum']).mean() >= P['sign_frac']), bool((zc <= -P['t_tum']).mean() >= P['sign_frac']))
            lr = np.clip(_loggauss(Zb, zc) - _loggauss(Zb, zh), -20, 20)
            Zv = Zd[ci]; dTv = edt(Tc); cvx = Tc & (dTv >= P['core_mm']) & ~Lc; zcv = Zv[cvx] if cvx.sum() >= 200 else Zv[Tc & ~Lc]
            lrv = np.clip(_loggauss(Zv, zcv) - _loggauss(Zv, Zd[Hd & ~Ld]), -20, 20); cvv = _local_cv(Xc, Tc, P['sigma_mm'] / sp, s)
        az = np.abs(Zb); dev = ((Zb >= P['t_tum']) & sgn[0]) | ((Zb <= -P['t_tum']) & sgn[1])
        cv = _local_cv(Xb, T, sg, s); zone = zoneb | (Hm & Omb)
        nbT = ndimage.convolve(T.astype(np.float32), nbk, mode='constant')
        inner = T & ~ndimage.binary_erosion(T, fp)
        outer = ndimage.binary_dilation(T, fp) & ~T & Omb & (nbT >= P['min_nb_frac'] * nbk.size)
        toH = inner & zone & (lr < -P['delta'])
        toO = inner & ~zone & (cv < -P['c_m'])
        grow = outer & dev & zone & ((Hm & (lr > P['delta'])) | (~Hm & (cv > P['c_m'])))
        Tn = (T & ~toH & ~toO) | grow
        Tn = _clean(_fill(Tn, thb, axb) & Omb, spb, P['min_comp_ml'])
        Hn = (Hm | toH) & ~Tn
        hfront = Hn & ~ndimage.binary_erosion(Hn, fp) & Omb & ~ndimage.binary_dilation(Tn, fp)
        Hn &= ~(hfront & (az >= P['t_out']))
        Hn |= ndimage.binary_dilation(Hn, fp) & ~Hn & ~Tn & zoneb & (az < P['t_host']) & (lr < -P['delta'])
        ch = int((Tn ^ T).sum() + (Hn ^ Hm).sum()); hist.append(ch)
        if P.get('dbg'):
            print(it, 'toH', int(toH.sum()), 'toO', int(toO.sum()), 'grow', int(grow.sum()), 'T', int(T.sum()), 'Tn', int(Tn.sum()), 'ch', ch)
        T, Hm = Tn, Hn
        if ch <= P['tol'] * max(int(T.sum() + (Hm & Omb).sum()), 1):
            if fresh:
                break
            recompute = True
        fresh = False
    Tc = fullT(T); Hc = fullH(Hm, Tc)
    Tf = np.zeros(X.shape, bool); Tf[C] = Tc
    Hf = H0 & ~T0; Hf[C] = Hc; Hf &= ~Tf
    vml = float(Tf.sum() * np.prod(sp) / 1000); v0 = float(T0.sum() * np.prod(sp) / 1000)
    if vml < P['vanish_ml'] or vml < P['vanish_frac'] * v0:
        flags.add('vanished')
    return dict(T=Tf, H=Hf, flags=sorted(flags), iters=it + 1, changes=hist, vol0_ml=v0, vol_ml=vml, down=f, block=b)


def decide(X, sp, ax, T, H, H0, les=None):
    """veto-only existence evidence on the evolved tumour T (window grid). Returns dict(call, organ_test, veto, feats)."""
    sp = np.asarray(sp, float); thick = sp.max() / sp.min() >= 1.5; out = dict(organ_test=None, call='uncertain', veto=False, why='')
    if not T.any():
        out.update(veto=True, why='vanished'); return out
    Xs = GR.smooth(X, sp, thick, ax); Hc = H & ~les if les is not None else H
    Z, D, nref = GR.host_frame(Xs, sp, H, T, Hc)
    f = {}
    if Z is not None:
        idx = np.argwhere(T); lo = idx.min(0); hi = idx.max(0) + 1; p = np.ceil(32 / sp).astype(int)
        Wr = tuple(slice(max(0, a - q), min(n, b + q)) for a, b, q, n in zip(lo, hi, p, T.shape))
        f.update(GR.closed30(X[Wr], Z[Wr], H[Wr], T[Wr], D[Wr], sp, thick, ax))
    else:
        f['vol_ml'] = float(T.sum() * np.prod(sp) / 1000); out['no_ref'] = True
    f.update(GR.level2(X, T, sp, thick, ax)); out['feats'] = f
    out['call'] = GR.tree_call(f); out['organ_test'] = GR.organ_tests(T, H0, sp, ax)
    why = [w for w, c in (('organ_test_' + str(out['organ_test']), out['organ_test'] is not None), ('tree_organ', out['call'] == 'organ')) if c]
    out['veto'] = bool(why); out['why'] = ';'.join(why); return out


def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else 1.0
