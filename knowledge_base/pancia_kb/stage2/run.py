"""One cluster of lesions, end to end (used for both expert validation and TCGA).

Inputs are agent-agnostic stage-1 outputs:
  lesion  dict(key, anchors=[mask objects], q=mask object (float, q^L on its own box), xi=mask object (chosen extent, union),
               PhL={host: P(h|L) renormalised over S}, S=[hosts], tum_cls='solid'|'hollow')
  organs  dict key -> dict(theta=mask object, kb=bool (knowledge-base region of an unsegmented organ), cls='solid'|'hollow'
               (reporting only); appearance is inferred within the scan, so no organ appearance key is needed)
A 'mask object' is anything with .sl and .crop (ledger_v6.Box, or Patch below); it is read on a window with win(obj, W).

Steps (main.tex Sec. Stage 2): region R_L = (anchors (+) r) U territories of the credible hosts, within the body; lesions whose regions
overlap form one cluster (caller); features and spatial prior on the cluster window; VB; refined tumour = components of {p >= 1/2}
meeting an anchor of the lesion; region growth (+ r) while a refined tumour touches the region boundary inside the body (at most
max_grow times); approximate Bayes factor on N_L = (anchors (+) r) within the region; flag when log BF < 0.
Tail weights (26 Oct): with fit['nu_prior'] (nuprior.fit_prior) every class's nu is a MAP estimate under its cell prior (modality x
tissue class): tumour components the mixture over the cluster's credible hosts, organ classes their own organ's cell, generic classes
the normal fallback. Without it, one fixed nu (fit['nu']) for every class."""
import time
import numpy as np
from scipy import ndimage
from . import features as F, prior as PR, vb, nuprior as NP

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


N_TUMOUR_AGENTS = 2         # tumour agents in the system (BiomedParse, VoxTell); agent-agnostic count


def lesion_unagreed(L, n=N_TUMOUR_AGENTS):
    """u = 1 - k / n for a lesion: k = the most agents agreeing on any of its anchors (v8.4 kinds: 'two' -> 2, 'single' / 'lobe' -> 1;
    R2Seg agent lesions: L['support']). Lesions without either (simulated lesions) count as full agreement (u = 0)."""
    if 'support' in L:
        k = 2 if L['support'] == 'two' else 1
    elif L.get('kinds'):
        k = max(2 if v == 'two' else 1 for v in L['kinds'].values())
    else:
        k = n
    return 1.0 - min(k, n) / float(n)


TUMOUR_FROM = ['q']          # set per solve from fit['tumour_from'] ('q' or 'anchor'); see _solve_window


def box_union(sls, shape):
    sls = [s for s in sls if s is not None]
    return tuple(slice(max(0, min(s[d].start for s in sls)), min(shape[d], max(s[d].stop for s in sls))) for d in range(3))


def _priors(Y, q, allowed, types, nw):
    """within-scan class priors (vb.class_priors): tumour components from the q-weighted voxels, each background class from its own
    voxels (territory, weighted by 1 - q); generic classes are allowed everywhere. No cross-scan appearance information is used."""
    return vb.class_priors(Y, q, allowed[types].astype(np.float64), beta0=nw[0], nu0=nw[1])


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


def solve_cluster(img, lesions, organs, fit, sp, mod, thick, ax, body_obj, shape, max_grow=5, bayes=True, max_iter=300, log=None,
                  seq=None):
    """img: callable W -> normalised intensity on window W. seq: sequence family, reported only (appearance is inferred within the
    scan). Returns dict(per-lesion results, window maps)."""
    key = seq or ('CT' if mod == 'CT' else 'MR')
    t0 = time.time()
    r = float(fit['r_mm']); SP = PR.SpatialPrior(fit['eta']); PRI = (float(fit['nw_beta0']), float(fit['nw_nu0']))
    TUMOUR_FROM[0] = fit.get('tumour_from', 'q')
    K = int(fit['K_max']); alpha0 = float(fit['alpha0']); nu = float(fit['nu']); a_w, b_w = float(fit['a_w']), float(fit['b_w'])
    CP = NP.ClassPriors(fit['nu_prior'], mod) if fit.get('nu_prior') else None
    regs = [lesion_region(L, organs, body_obj, r, shape, sp) for L in lesions]
    extra = None; n_grow = 0
    while True:
        W = box_union([g.sl for g in regs] + ([extra.sl] if extra is not None else []), shape)
        Wp = F.pad_box(W, sp, max(30.0, r), shape)                  # context for features and organ depths; growth by r fits
        ROIp = np.zeros(tuple(s.stop - s.start for s in Wp), bool)
        for g in regs:
            ROIp |= win(g, Wp)
        if extra is not None:
            ROIp |= win(extra, Wp)
        res = _solve_window(img, lesions, organs, SP, PRI, sp, key, thick, ax, body_obj, Wp, ROIp, K, alpha0, nu, a_w, b_w, r, max_iter,
                            CP)
        grow = res.pop('grow')
        if grow is None or n_grow >= max_grow:
            break
        n_grow += 1
        extra = Patch(Wp, (F.dil(grow, sp, r) & win(body_obj, Wp)) | (win(extra, Wp) if extra is not None else False))
        if log: log(f'  region growth {n_grow}')
    res['n_grow'] = n_grow; res['reached_max_grow'] = bool(grow is not None)
    if bayes:
        for i, L in enumerate(lesions):
            res['lesions'][i].update(_bayes_factor(res, L, organs, PRI, key, K, alpha0, nu, a_w, b_w, sp, r, max_iter, CP))
    res['sec'] = round(time.time() - t0, 1); res['seq'] = key
    return res


def _types(organs_in, ROI, theta_w):
    """voxel type = set of segmented organs containing it; allowed classes = those organs + the generic classes.
    Types are the distinct rows of the voxel x organ membership matrix (no limit on the number of organs)."""
    n_org = len(organs_in)
    n = int(ROI.sum())
    if n_org == 0:
        return np.zeros(n, np.int64), np.ones((1, N_GENERIC), bool)
    Mb = np.stack([theta_w[k][ROI] for k in organs_in], 1)
    rows, types = np.unique(np.packbits(Mb, axis=1), axis=0, return_inverse=True)
    types = np.asarray(types).reshape(-1)
    allowed = np.zeros((len(rows), n_org + N_GENERIC), bool)
    allowed[:, :n_org] = np.unpackbits(rows, axis=1, count=n_org).astype(bool)
    allowed[:, n_org:] = True
    return types, allowed


def cluster_nu_prior(CP, lesions, organs_in, K):
    """per-class mixtures on log(1/nu) in Model order: K tumour components, the organ classes, the generic classes (None without
    a prior). Tumour: the lesions' credible hosts, P(h|L) averaged over the cluster's lesions."""
    if CP is None:
        return None
    ph = {}
    for L in lesions:
        for h, w in (L.get('PhL') or {}).items():
            ph[h] = ph.get(h, 0.0) + float(w) / len(lesions)
    return [CP.tumour(ph)] * K + [CP.organ(k) for k in organs_in] + [CP.generic()] * N_GENERIC


def _solve_window(img, lesions, organs, SP, PRI, sp, mod, thick, ax, body_obj, Wp, ROI, K, alpha0, nu, a_w, b_w, r, max_iter, CP=None):
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
        dtp = F.through_plane_distance(xi, sp, ax) if SP.tp else None
        piL = PR.lesion_prior(SP, q, sd, L['PhL'], theta_w, kb, seg, depth, ROI, dtp=dtp, unagreed=lesion_unagreed(L))
        one_minus_pi *= (1.0 - piL[ROI]); one_minus_q *= (1.0 - q[ROI])
        A = np.zeros(shape_w, bool)
        for a in L['anchors']:
            A |= win(a, Wp)
        anchors_w.append(A)
        L['_pi_w'] = piL
    pi_c[ROI] = 1.0 - one_minus_pi; q_c[ROI] = 1.0 - one_minus_q
    organs_in = [k for k in seg if (theta_w[k] & ROI).sum() >= 20]
    types, allowed = _types(organs_in, ROI, theta_w)
    Y = Y4[:, ROI].T
    # tumour appearance and initialisation (29 Oct): 'q' (default, fits before 29 Oct) learns the tumour classes' within-scan priors
    # from q-weighted voxels and starts them from q; 'anchor' uses the anchors (agreement cores) for the priors and starts from the
    # anchors (1) and the spatial prior elsewhere, so an extent that is mostly normal tissue cannot redefine the tumour (TCGA KIRP)
    if TUMOUR_FROM[0] == 'anchor':
        Aall = np.zeros(shape_w, bool)
        for A_ in anchors_w:
            Aall |= A_
        q_app = Aall[ROI].astype(np.float64); q_init = np.where(Aall[ROI], 0.99, pi_c[ROI])
    else:
        q_app = q_c[ROI]; q_init = q_c[ROI]
    tp, bps = _priors(Y, q_app, allowed, types, PRI)
    edges = vb.build_edges(ROI, sp)
    m = vb.Model(Y, pi_c[ROI], types, allowed, tp, bps, K=K, alpha0=alpha0, nu=nu, a_w=a_w, b_w=b_w, edges=edges,
                 nu_prior=cluster_nu_prior(CP, lesions, organs_in, K))
    m.init(q_init).run(max_iter=max_iter, parity=vb.parity(ROI))
    p = np.zeros(shape_w, np.float32); p[ROI] = m.p()
    Rt, _ = m.components()
    comp = np.zeros(shape_w, np.uint8); comp[ROI] = (np.argmax(Rt, 1) + 1).astype(np.uint8)
    atyp = np.zeros(shape_w, np.float32); atyp[ROI] = m.atypicality()
    Z = p >= 0.5
    lab, nlab = ndimage.label(Z, structure=STRUCT26)
    body_w = win(body_obj, Wp)
    # region boundary inside the body and inside the volume: ROI voxels with a non-ROI 6-neighbour that is in the body
    edge = ROI & ~ndimage.binary_erosion(ROI, border_value=1)
    outside_ok = ndimage.binary_dilation(~ROI & body_w) & edge
    out = []; grow = np.zeros(shape_w, bool); used = set(); ZLs = []
    vox_ml = float(np.prod(sp)) / 1000.0
    # components of {p >= 1/2} met by each lesion's anchors. A component met by several lesions (their refined tumours meet) is
    # split between them by the nearest anchor (Euclidean, mm), so no lesion inherits another's tumour; the merged component is
    # reported once per group (merged_with, vol_Z_group).
    ids_of = [set(np.unique(lab[anchors_w[i] & Z]).tolist()) - {0} for i in range(len(lesions))]
    owners = {}
    for i, ids in enumerate(ids_of):
        for j in ids:
            owners.setdefault(j, []).append(i)
    shared = {j: o for j, o in owners.items() if len(o) > 1}
    near = None
    if shared:
        sh_ids = list(shared)
        Msh = np.isin(lab, sh_ids)
        dist = np.full((len(lesions),) + shape_w, np.inf, np.float32)
        for i in sorted({i for o in shared.values() for i in o}):
            dist[i] = F.edt(anchors_w[i], sp)
        near = np.full(shape_w, -1, np.int16)
        for j, o in shared.items():
            mj = lab == j
            d = np.stack([dist[i][mj] for i in o], 0)
            near[mj] = np.asarray(o, np.int16)[np.argmin(d, 0)]
        del dist, Msh
    for i, L in enumerate(lesions):
        ids = ids_of[i]
        used |= ids
        own = [j for j in ids if j not in shared]
        ZL = np.isin(lab, own) if own else np.zeros(shape_w, bool)
        if near is not None:
            ZL |= near == i
        grp = sorted({k for j in ids if j in shared for k in shared[j]} - {i})
        ZLs.append(ZL)
        if (ZL & outside_ok).any():
            grow |= ZL
        xi = win(L['xi'], Wp); A = anchors_w[i]
        trim = xi & ~ZL; rec = ZL & ~xi
        rec_by = {}
        for k in theta_w:
            v = float((rec & theta_w[k]).sum() * vox_ml)
            if v > 0: rec_by[k] = round(v, 2)
        out.append(dict(key=L['key'], merged_with=[lesions[k]['key'] for k in grp],
                        vol_Z_group=round(float(np.isin(lab, list(ids)).sum()) * vox_ml, 2) if grp else None,
                        vol_xi=round(float(xi.sum()) * vox_ml, 2), vol_Z=round(float(ZL.sum()) * vox_ml, 2),
                        vol_anchor=round(float(A.sum()) * vox_ml, 2), trimmed_ml=round(float(trim.sum()) * vox_ml, 2),
                        recalled_ml=round(float(rec.sum()) * vox_ml, 2), recalled_by_territory=rec_by,
                        dice_Z_xi=round(float(F_dice(ZL, xi)), 3),
                        pbar_xi=round(float(p[xi].mean()), 3) if xi.any() else None, pbar_Z=round(float(p[ZL].mean()), 3) if ZL.any() else None,
                        p_anchor_ge_half=round(float((p[A] >= 0.5).mean()), 3) if A.any() else None,
                        comp_frac=[round(float(x), 3) for x in (np.bincount(comp[ZL], minlength=K + 1)[1:] / max(ZL.sum(), 1))],
                        atyp_median_Z=round(float(np.median(atyp[ZL])), 3) if ZL.any() else None,
                        atyp_q90_Z=round(float(np.quantile(atyp[ZL], 0.9)), 3) if ZL.any() else None))
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
    return dict(lesions=out, new_candidates=newc, W=Wp, ROI=ROI, p=p, comp=comp, atyp=atyp, Z=Z, ZL=ZLs, anchors=anchors_w, lab=lab,
                model=dict(iters=m.iters, converged=m.converged, k_eff=cs['k_eff'], rho=cs['rho'], comp_means=cs['means'],
                           nu_tumour=cs['nu'], nu_background=[round(float(v), 2) for v in m.nu_k[K:]],
                           nu_prior=CP is not None,
                           n_vox=int(ROI.sum()), organs=organs_in, mean_Ew=float(np.mean(m.Ew)) if getattr(m, 'Ew', None) is not None else None),
                grow=grow if grow.any() else None, _theta=theta_w, _seg=seg, _kb=kb, _depth=depth, _Y4=Y4)


def F_dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * (a & b).sum() / s if s else 0.0


def _bayes_factor(res, L, organs, PRI, mod, K, alpha0, nu, a_w, b_w, sp, r, max_iter, CP=None):
    """approximate log BF (vb.log_bayes_factor: tumour + background vs background only with matched flexibility, both refitted) on
    N_L = (anchors (+) r) within the region, with the lesion's own prior. N_L must contain normal tissue around the claim: on the
    claim alone the background model explains everything with its extra classes (phantom: log BF < 0 for a true anchor).
    Flag when log BF < 0 (the image prefers background alone)."""
    i = [l['key'] for l in res['lesions']].index(L['key'])
    A = res['anchors'][i]; ROI = res['ROI']
    if not A.any():
        return dict(log_bf=None, flag_bf=None, n_vox_bf=0, log_bf_per_vox=None)
    qw = win(L['q'], res['W'], np.float32)
    N = F.dil(A, sp, r) & ROI
    theta_w = res['_theta']; seg = res['_seg']
    organs_in = [k for k in seg if (theta_w[k] & N).sum() >= 20]
    types, allowed = _types(organs_in, N, theta_w)
    Y = res['_Y4'][:, N].T
    q_app = A[N].astype(np.float64) if TUMOUR_FROM[0] == 'anchor' else qw[N]
    q_init = np.where(A[N], 0.99, L['_pi_w'][N]) if TUMOUR_FROM[0] == 'anchor' else qw[N]
    tp, bps = _priors(Y, q_app, allowed, types, PRI)          # within-scan priors on N_L, the same for both models
    npr = cluster_nu_prior(CP, [L], organs_in, K)
    npr = None if npr is None else (npr[:K], npr[K:], CP.generic())
    lbf, _, _ = vb.log_bayes_factor(Y, L['_pi_w'][N], types, allowed, tp, bps, q_init, K=K, alpha0=alpha0, nu=nu, a_w=a_w, b_w=b_w,
                                    edges=vb.build_edges(N, sp), parity=vb.parity(N), max_iter=max_iter, nu_prior=npr)
    n = int(N.sum())
    # log BF sums over voxels treated as independent, so its size grows with N_L and overstates the evidence; the per-voxel value is
    # scale-free (comparable across lesions; near 0 = near tie). Only the sign is used (flag).
    return dict(log_bf=round(float(lbf), 2), flag_bf=bool(lbf < 0), n_vox_bf=n, log_bf_per_vox=round(float(lbf) / max(n, 1), 5))
