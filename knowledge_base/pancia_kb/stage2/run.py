"""One cluster of lesions, end to end (used for both expert validation and TCGA).

Inputs are agent-agnostic stage-1 outputs:
  lesion  dict(key, anchors=[mask objects], q=mask object (float, q^L on its own box), xi=mask object (chosen extent, union),
               PhL={host: P(h|L) renormalised over S}, S=[hosts], tum_cls='solid'|'hollow')
  organs  dict key -> dict(theta=mask object, kb=bool (knowledge-base region of an unsegmented organ), app=appearance key,
               cls='solid'|'hollow')
A 'mask object' is anything with .sl and .crop (ledger_v6.Box, or Patch below); it is read on a window with win(obj, W).

Steps (main.tex Sec. Stage 2): region R_L = (anchors (+) r) U territories of the credible hosts, within the body; lesions whose regions
overlap form one cluster (caller); features and spatial prior on the cluster window; VB; refined tumour = components of {p >= 1/2}
meeting an anchor of the lesion; region growth (+ r) while a refined tumour touches the region boundary inside the body (at most
max_grow times); approximate Bayes factor on N_L = (anchors (+) r) within the region; flag when log BF < 0."""
import time
import numpy as np
from scipy import ndimage
from . import features as F, prior as PR, vb

N_GENERIC = 3            # generic background classes (fat / muscle / other); design value, disclosed
STRUCT26 = np.ones((3, 3, 3), bool)


class Patch:
    def __init__(self, sl, crop):
        self.sl = tuple(sl); self.crop = crop

    @classmethod
    def from_full(cls, M):
        b = F.bbox(M)
        if b is None:
            return cls((slice(0, 1),) * 3, np.zeros((1, 1, 1), M.dtype))
        return cls(b, M[b].copy())

    def any(self):
        return self.crop is not None and bool(np.asarray(self.crop).any())


def win(obj, W, dtype=bool):
    """the object's values on window W (zeros / False outside its box)"""
    shape = tuple(s.stop - s.start for s in W)
    out = np.zeros(shape, dtype)
    if obj is None or getattr(obj, 'crop', None) is None or getattr(obj, 'sl', None) is None:
        return out
    lo = [max(a.start, b.start) for a, b in zip(obj.sl, W)]; hi = [min(a.stop, b.stop) for a, b in zip(obj.sl, W)]
    if any(l >= h for l, h in zip(lo, hi)):
        return out
    out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, W))] = \
        obj.crop[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, obj.sl))]
    return out


def box_union(sls, shape):
    sls = [s for s in sls if s is not None]
    return tuple(slice(max(0, min(s[d].start for s in sls)), min(shape[d], max(s[d].stop for s in sls))) for d in range(3))


def _nw_from(d, D, beta0, nu0):
    return vb.NW.from_moments(np.asarray(d['mean'], float)[:D], np.asarray(d['cov'], float)[:D, :D], beta0=beta0, nu0=nu0)


class Priors:
    """appearance priors from the fit (expert data). Lookup order: organ key -> organ class -> modality pool -> broad."""

    def __init__(self, fit):
        self.app = fit['appearance']; self.beta0 = fit['nw_beta0']; self.nu0 = fit['nw_nu0']; self.D = F.D

    def _get(self, mod, kind, key):
        a = self.app.get(mod, {}).get(kind, {})
        return a.get(key)

    def _broad(self, mod):
        sc = 400.0 if mod == 'CT' else 3.0
        return vb.NW.from_moments(np.zeros(self.D), np.eye(self.D) * sc ** 2, beta0=1e-3, nu0=self.D + 2)

    def tumour(self, mod, cls):
        d = self._get(mod, 'tumour', cls) or self._get(mod, 'tumour', 'pooled')
        return _nw_from(d, self.D, self.beta0, self.nu0) if d else self._broad(mod)

    def organ(self, mod, key, cls):
        d = self._get(mod, 'organ', key) or self._get(mod, 'organ_cls', cls) or self._get(mod, 'organ_cls', 'pooled')
        return _nw_from(d, self.D, self.beta0, self.nu0) if d else self._broad(mod)

    def generic(self, mod):
        d = self._get(mod, 'generic', 'pooled')
        return _nw_from(d, self.D, self.beta0, self.nu0) if d else self._broad(mod)


def lesion_region(les, organs, body_obj, r, shape, sp):
    """R_L on its own bounding window: returns Patch"""
    asl = box_union([a.sl for a in les['anchors']], shape)
    Wa = F.pad_box(asl, sp, r + 1.0, shape)
    sls = [Wa] + [organs[h]['theta'].sl for h in les['S'] if h in organs and organs[h]['theta'].any()]
    W = box_union(sls, shape)
    A = np.zeros(tuple(s.stop - s.start for s in W), bool)
    for a in les['anchors']:
        A |= win(a, W)
    R = F.dil(A, sp, r)
    for h in les['S']:
        if h in organs:
            R |= win(organs[h]['theta'], W)
    R &= win(body_obj, W)
    return Patch(W, R)


def solve_cluster(img, lesions, organs, fit, sp, mod, thick, ax, body_obj, shape, max_grow=5, bayes=True, max_iter=300, log=None):
    """img: callable W -> normalised intensity on window W. Returns dict(per-lesion results, window maps)."""
    t0 = time.time()
    r = float(fit['r_mm']); SP = PR.SpatialPrior(fit['eta']); PRI = Priors(fit)
    K = int(fit['K_max']); alpha0 = float(fit['alpha0']); nu = float(fit['nu']); a_w, b_w = float(fit['a_w']), float(fit['b_w'])
    regs = [lesion_region(L, organs, body_obj, r, shape, sp) for L in lesions]
    extra = None; n_grow = 0
    while True:
        W = box_union([g.sl for g in regs] + ([extra.sl] if extra is not None else []), shape)
        Wp = F.pad_box(W, sp, 30.0, shape)                          # features and organ depths need context
        ROIp = np.zeros(tuple(s.stop - s.start for s in Wp), bool)
        for g in regs:
            ROIp |= win(g, Wp)
        if extra is not None:
            ROIp |= win(extra, Wp)
        res = _solve_window(img, lesions, organs, SP, PRI, sp, mod, thick, ax, body_obj, Wp, ROIp, K, alpha0, nu, a_w, b_w, r, max_iter)
        grow = res.pop('grow')
        if grow is None or n_grow >= max_grow:
            break
        n_grow += 1
        extra = Patch(Wp, (F.dil(grow, sp, r) & win(body_obj, Wp)) | (win(extra, Wp) if extra is not None else False))
        if log: log(f'  region growth {n_grow}')
    res['n_grow'] = n_grow; res['reached_max_grow'] = bool(grow is not None)
    if bayes:
        for i, L in enumerate(lesions):
            res['lesions'][i].update(_bayes_factor(res, L, organs, PRI, mod, K, alpha0, nu, a_w, b_w, sp, r, max_iter))
    res['sec'] = round(time.time() - t0, 1)
    return res


def _types(organs_in, ROI, theta_w):
    """voxel type = set of segmented organs containing it; allowed classes = those organs + the generic classes"""
    n_org = len(organs_in)
    code = np.zeros(ROI.shape, np.int64)
    for j, k in enumerate(organs_in):
        code |= (theta_w[k].astype(np.int64) << j)
    codes, types = np.unique(code[ROI], return_inverse=True)
    allowed = np.zeros((len(codes), n_org + N_GENERIC), bool)
    for t, cval in enumerate(codes):
        for j in range(n_org):
            allowed[t, j] = bool((int(cval) >> j) & 1)
        allowed[t, n_org:] = True
    return types, allowed


def _solve_window(img, lesions, organs, SP, PRI, sp, mod, thick, ax, body_obj, Wp, ROI, K, alpha0, nu, a_w, b_w, r, max_iter):
    shape_w = ROI.shape
    Xn = img(Wp)
    Y4 = F.image_features(Xn, sp, thick, ax)
    theta_w = {k: win(o['theta'], Wp) for k, o in organs.items()}
    theta_w = {k: v for k, v in theta_w.items() if v.any()}
    kb = {k: bool(organs[k]['kb']) for k in theta_w}
    nonkb = [k for k in theta_w if not kb[k]]
    seg = [k for k in nonkb if organs[k].get('bg', True)]           # tissue classes (composite host territories excluded)
    depth = {k: F.inner_depth(theta_w[k], sp) for k in nonkb}
    # spatial prior per lesion, cluster combination
    pi_c = np.zeros(shape_w, np.float64); q_c = np.zeros(shape_w, np.float64); anchors_w = []
    one_minus_pi = np.ones(int(ROI.sum())); one_minus_q = np.ones(int(ROI.sum()))
    for L in lesions:
        q = win(L['q'], Wp, np.float32)
        xi = win(L['xi'], Wp)
        sd = F.signed_distance(xi, sp) if xi.any() else np.full(shape_w, 60.0, np.float32)
        piL = PR.lesion_prior(SP, q, sd, L['PhL'], theta_w, kb, seg, depth, ROI)
        one_minus_pi *= (1.0 - piL[ROI]); one_minus_q *= (1.0 - q[ROI])
        A = np.zeros(shape_w, bool)
        for a in L['anchors']:
            A |= win(a, Wp)
        anchors_w.append(A)
        L['_pi_w'] = piL
    pi_c[ROI] = 1.0 - one_minus_pi; q_c[ROI] = 1.0 - one_minus_q
    organs_in = [k for k in seg if (theta_w[k] & ROI).sum() >= 20]
    types, allowed = _types(organs_in, ROI, theta_w)
    tum_cls = lesions[0].get('tum_cls', 'solid')
    tp = PRI.tumour(mod, tum_cls)
    bps = [PRI.organ(mod, organs[k]['app'], organs[k]['cls']) for k in organs_in] + [PRI.generic(mod)] * N_GENERIC
    Y = Y4[:, ROI].T
    edges = vb.build_edges(ROI, sp)
    m = vb.Model(Y, pi_c[ROI], types, allowed, tp, bps, K=K, alpha0=alpha0, nu=nu, a_w=a_w, b_w=b_w, edges=edges)
    m.init(q_c[ROI]).run(max_iter=max_iter, parity=vb.parity(ROI))
    p = np.zeros(shape_w, np.float32); p[ROI] = m.p()
    Rt, _ = m.components()
    comp = np.zeros(shape_w, np.uint8); comp[ROI] = (np.argmax(Rt, 1) + 1).astype(np.uint8)
    Z = p >= 0.5
    lab, nlab = ndimage.label(Z, structure=STRUCT26)
    body_w = win(body_obj, Wp)
    # region boundary inside the body and inside the volume: ROI voxels with a non-ROI 6-neighbour that is in the body
    edge = ROI & ~ndimage.binary_erosion(ROI, border_value=1)
    outside_ok = ndimage.binary_dilation(~ROI & body_w) & edge
    out = []; grow = np.zeros(shape_w, bool); used = set(); ZLs = []
    vox_ml = float(np.prod(sp)) / 1000.0
    for i, L in enumerate(lesions):
        ids = set(np.unique(lab[anchors_w[i] & Z]).tolist()) - {0}
        used |= ids
        ZL = np.isin(lab, list(ids)) if ids else np.zeros(shape_w, bool)
        ZLs.append(ZL)
        if (ZL & outside_ok).any():
            grow |= ZL
        xi = win(L['xi'], Wp); A = anchors_w[i]
        trim = xi & ~ZL; rec = ZL & ~xi
        rec_by = {}
        for k in theta_w:
            v = float((rec & theta_w[k]).sum() * vox_ml)
            if v > 0: rec_by[k] = round(v, 2)
        out.append(dict(key=L['key'], vol_xi=round(float(xi.sum()) * vox_ml, 2), vol_Z=round(float(ZL.sum()) * vox_ml, 2),
                        vol_anchor=round(float(A.sum()) * vox_ml, 2), trimmed_ml=round(float(trim.sum()) * vox_ml, 2),
                        recalled_ml=round(float(rec.sum()) * vox_ml, 2), recalled_by_territory=rec_by,
                        dice_Z_xi=round(float(F_dice(ZL, xi)), 3),
                        pbar_xi=round(float(p[xi].mean()), 3) if xi.any() else None, pbar_Z=round(float(p[ZL].mean()), 3) if ZL.any() else None,
                        p_anchor_ge_half=round(float((p[A] >= 0.5).mean()), 3) if A.any() else None,
                        comp_frac=[round(float(x), 3) for x in (np.bincount(comp[ZL], minlength=K + 1)[1:] / max(ZL.sum(), 1))]))
    # new candidates: components meeting no anchor
    newc = []
    Aall = np.zeros(shape_w, bool)
    for A in anchors_w: Aall |= A
    dA = F.edt(Aall, sp) if Aall.any() else np.full(shape_w, np.inf)
    for j in range(1, nlab + 1):
        if j in used: continue
        cj = lab == j
        newc.append(dict(ml=round(float(cj.sum()) * vox_ml, 3), d_anchor_mm=round(float(dA[cj].min()), 1),
                         organs=[k for k in theta_w if (cj & theta_w[k]).any()]))
    cs = m.comp_summary()
    return dict(lesions=out, new_candidates=newc, W=Wp, ROI=ROI, p=p, comp=comp, Z=Z, ZL=ZLs, anchors=anchors_w, lab=lab,
                model=dict(iters=m.iters, converged=m.converged, k_eff=cs['k_eff'], rho=cs['rho'], comp_means=cs['means'],
                           n_vox=int(ROI.sum()), organs=organs_in, mean_Ew=float(np.mean(m.Ew)) if getattr(m, 'Ew', None) is not None else None),
                grow=grow if grow.any() else None, _theta=theta_w, _seg=seg, _kb=kb, _depth=depth, _Y4=Y4)


def F_dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * (a & b).sum() / s if s else 0.0


def _bayes_factor(res, L, organs, PRI, mod, K, alpha0, nu, a_w, b_w, sp, r, max_iter):
    """approximate log BF (vb.log_bayes_factor: tumour + background vs background only with matched flexibility, both refitted) on
    N_L = (anchors (+) r) within the region, with the lesion's own prior. N_L must contain normal tissue around the claim: on the
    claim alone the background model explains everything with its extra classes (phantom: log BF < 0 for a true anchor).
    Flag when log BF < 0 (the image prefers background alone)."""
    i = [l['key'] for l in res['lesions']].index(L['key'])
    A = res['anchors'][i]; ROI = res['ROI']
    if not A.any():
        return dict(log_bf=None, flag_bf=None)
    qw = win(L['q'], res['W'], np.float32)
    N = F.dil(A, sp, r) & ROI
    theta_w = res['_theta']; seg = res['_seg']
    organs_in = [k for k in seg if (theta_w[k] & N).sum() >= 20]
    types, allowed = _types(organs_in, N, theta_w)
    tp = PRI.tumour(mod, L.get('tum_cls', 'solid'))
    bps = [PRI.organ(mod, organs[k]['app'], organs[k]['cls']) for k in organs_in] + [PRI.generic(mod)] * N_GENERIC
    Y = res['_Y4'][:, N].T
    lbf, _, _ = vb.log_bayes_factor(Y, L['_pi_w'][N], types, allowed, tp, bps, qw[N], K=K, alpha0=alpha0, nu=nu, a_w=a_w, b_w=b_w,
                                    edges=vb.build_edges(N, sp), parity=vb.parity(N), max_iter=max_iter)
    return dict(log_bf=round(float(lbf), 2), flag_bf=bool(lbf < 0), n_vox_bf=int(N.sum()))
