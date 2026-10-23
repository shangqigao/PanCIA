"""Expert-data simulation for stage 2 (R2Seg only; Ovary_Tumor_00 is never read).

Same construction as the stage-1 expert testbed (clinical/PanCIA_outputs/scripts/tb_build.py, v7/v8): for the largest expert tumour
component G of a case and its true host h* (the labelled organ touching G most),
  cores (agent-agreement-like anchors): partial (deepest 50% of G), sub (random compact 30-70%), cut2d (3 acquisition slices);
  outlines (agent-like claims): dil4, host10, organ, part80, nb15, lobe. The testbed's 'exact' outline (the expert mask itself) is
  left out for stage 2: no agent draws the expert outline, and with it the prior would learn to trust the chosen extent too much
  and the chosen-extent baseline of the validation would be optimistic;
  extents of an anchor: {anchor, outlines after lobe splitting, U, U - non-host organs}, merged at Dice >= 0.95 (grounding_g1),
  with the stage-1 extent prior w0 ~ psi(log |xi| / |a|) (fit_host6.json psi, eta = 1), exactly as ledger v8.4.
Anatomy scenarios for the host territory: excl (organ mask without the tumour), incl (organ mask including the tumour inside its
contour), lost (host unsegmented: its knowledge-base-like region = the organ's axial levels +- 25 mm). Each scenario is used once per
anchor (equal weight; a disclosed design choice). Every simulated outline is drawn around a real expert tumour (the prior is
conditional on the lesion being a tumour; no false outlines).
The truth t_v is ALL expert tumour voxels (other tumour components included), so recall far from the outlines is learned from
expert scans with several tumours.
Crops: tumour bounding box + 60 mm; crops above ~3M voxels are strided in-plane first (testbed rule, memory)."""
import os, zlib
import numpy as np
import nibabel as nib
from scipy import ndimage
from . import features as F

SETS = {'Kidney_Tumor_00': 'CT', 'Liver_Tumor_00': 'CT', 'Pancreas_Tumor_00': 'CT', 'Lung_Tumor_01': 'CT', 'Cervix_Tumor_00': 'MR'}
FLARE = {1: 'liver', 2: 'kidney_right', 3: 'spleen', 4: 'pancreas', 5: 'aorta', 6: 'inferior_vena_cava', 7: 'adrenal_gland_right',
         8: 'adrenal_gland_left', 9: 'gallbladder', 10: 'esophagus', 11: 'stomach', 12: 'duodenum', 13: 'kidney_left'}
TUMOUR_LABEL = {'Kidney_Tumor_00': 2, 'Liver_Tumor_00': 14, 'Pancreas_Tumor_00': 14, 'Lung_Tumor_01': 5, 'Cervix_Tumor_00': 2}
HOLLOW = {'uterus', 'stomach', 'duodenum', 'esophagus', 'gallbladder'}
CORES = ('partial', 'sub', 'cut2d')
SCENS = ('excl', 'incl', 'lost')


def organ_key(name):
    """expert organ name -> appearance key (sides and kidney components merged)"""
    if name.startswith('kidney'):
        return 'kidney'
    if name.startswith('adrenal'):
        return 'adrenal_gland'
    return name


def organ_cls(name):
    return 'hollow' if organ_key(name) in HOLLOW else 'solid'


def organs_of(ds, a):
    if ds == 'Kidney_Tumor_00':
        T = a == 2; K = a == 1; lb, n = ndimage.label(K); org = {}
        for i in range(1, n + 1):
            m = lb == i
            if m.sum() > 1000: org[f'kidney_{i}'] = m
        return org, T
    if ds in ('Liver_Tumor_00', 'Pancreas_Tumor_00'):
        return {nm: a == v for v, nm in FLARE.items() if (a == v).any()}, a == 14
    if ds == 'Lung_Tumor_01':
        return {nm: a == v for v, nm in {1: 'lung', 2: 'heart', 3: 'esophagus'}.items() if (a == v).any()}, a == 5
    if ds == 'Cervix_Tumor_00':
        return {'uterus': a == 1}, a == 2
    raise ValueError(ds)


def true_host(G, org, sp):
    ring = F.dil(G, sp, 3.0) & ~G
    cnt = {k: int((m & ring).sum()) for k, m in org.items()}
    k = max(cnt, key=cnt.get) if cnt else None
    return k if (k is not None and cnt[k] > 0) else None


def sub_region(G, sp, rng, frac):
    idx = np.argwhere(G); c = idx[rng.integers(len(idx))]; d = (((idx - c) * sp) ** 2).sum(1); n = max(int(frac * len(idx)), 20)
    keep = idx[np.argsort(d)[:n]]; M = np.zeros(G.shape, bool); M[tuple(keep.T)] = True; return M


def lobe_outline(G, sp, rng, allowed):
    dG = F.edt(G, sp); cand = np.argwhere((dG >= 15) & (dG <= 30) & allowed)
    if len(cand) == 0:
        return None
    c = cand[rng.integers(len(cand))]; r = float(rng.uniform(6, 9))
    B = np.zeros(G.shape, bool); B[tuple(c)] = True
    gi = np.argwhere(G); p = gi[np.argmin((((gi - c) * sp) ** 2).sum(1))]
    L = np.zeros(G.shape, bool)
    for t in np.linspace(0, 1, 200):
        q = np.rint(p + t * (c - p)).astype(int); L[tuple(q)] = True
    return G | F.dil(B, sp, r) | F.dil(L, sp, 1.5)


def slab(m, si, sp, margin=25.0):
    zz = np.nonzero(m.any(axis=tuple(k for k in range(3) if k != si)))[0]
    out = np.zeros(m.shape, bool)
    if len(zz) == 0:
        return out
    q = int(np.ceil(margin / sp[si])); lo, hi = max(0, zz[0] - q), min(m.shape[si], zz[-1] + 1 + q)
    sl = [slice(None)] * 3; sl[si] = slice(lo, hi); out[tuple(sl)] = True; return out


def case_list(r2, n_per=None):
    out = []
    for ds in SETS:
        d = os.path.join(r2, ds, 'labels')
        fs = sorted(f for f in os.listdir(d) if f.endswith('_seg.nii.gz'))
        fs = list(np.random.default_rng(zlib.crc32((ds + 'tb').encode())).permutation(fs))
        if n_per:
            fs = fs[:n_per]
        out += [(ds, os.path.join(d, f)) for f in fs]
    return out


def load_case(ds, lf, max_vox=3e6):
    """crop, organs, truth, true host, norm params. Returns None when unusable (no tumour >= 0.5 ml, no host)."""
    mod = SETS[ds]; case = os.path.basename(lf).replace('_seg.nii.gz', '')
    lim = nib.load(lf)
    try:
        A = np.asarray(lim.dataobj.get_unscaled()); A = A.astype(np.uint8) if A.max() < 256 else np.rint(A).astype(np.int16)
    except Exception:
        A = np.rint(np.asanyarray(lim.dataobj)).astype(np.int16)
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); si = [k for k, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'SI'][0]
    TL = TUMOUR_LABEL[ds]
    Tidx = [np.nonzero(np.any(A == TL, axis=tuple(k for k in range(3) if k != d)))[0] for d in range(3)]
    if len(Tidx[0]) == 0:
        return None
    pad = np.ceil(60 / sp).astype(int)
    W = tuple(slice(int(max(0, ix[0] - q)), int(min(n_, ix[-1] + 1 + q))) for ix, q, n_ in zip(Tidx, pad, A.shape))
    A = A[W]
    org, T = organs_of(ds, A); del A
    lb, n = ndimage.label(T)
    if n == 0:
        return None
    sz = np.bincount(lb.ravel()); sz[0] = 0; j = int(np.argmax(sz))
    if sz[j] * np.prod(sp) / 1000 < 0.5:
        return None
    G = lb == j; del lb
    hs = true_host(G, org, sp)
    if hs is None:
        return None
    if mod == 'MR':                                         # normalisation on the whole scan, as on TCGA
        Xf = np.asarray(img.dataobj, np.float32); Xf = Xf[..., 0] if Xf.ndim == 4 else Xf
        thick0, ax0 = F.acq_axis(sp, si)
        cen, scl = F.norm_params(Xf, mod, F.body_mask(Xf, mod, ax0)); X = Xf[W]; del Xf
    else:
        cen, scl = 0.0, 1.0
        X = np.asarray(img.dataobj[W], np.float32); X = X[..., 0] if X.ndim == 4 else X
    nvox = G.size
    if nvox > max_vox:
        thin = sp.max() / sp.min() < 1.5
        axes = [0, 1, 2] if thin else [k for k in range(3) if k != int(np.argmax(sp))]
        f = int(np.ceil((nvox / max_vox) ** (1.0 / len(axes))))
        st = tuple(slice(None, None, f) if k in axes else slice(None) for k in range(3))
        G = G[st]; T = T[st]; org = {k: m[st] for k, m in org.items()}; X = X[st]
        sp = np.array([s * f if k in axes else s for k, s in enumerate(sp)])
    org = {k: m for k, m in org.items() if m.any()}
    if hs not in org:
        return None
    thick, ax = F.acq_axis(sp, si)
    Hf = _fill2d(org[hs] | G, ax) | org[hs]; Gin = G & Hf
    others = np.zeros(G.shape, bool)
    for k, m in org.items():
        if k != hs: others |= m
    Xn = F.normalise(X, mod, cen, scl)
    return dict(ds=ds, case=case, mod=mod, sp=sp, si=si, ax=ax, thick=thick, X=Xn, G=G, T=T, org=org, hs=hs, Gin=Gin,
                host_incl=org[hs] | Gin, others=others, cls=organ_cls(hs))


def _fill2d(M, ax):
    M = M.copy()
    for z in range(M.shape[ax]):
        sl = [slice(None)] * 3; sl[ax] = z; M[tuple(sl)] = ndimage.binary_fill_holes(M[tuple(sl)])
    return M


def make_cores(c, rng):
    G, sp, ax = c['G'], c['sp'], c['ax']
    dG = F.inner_depth(G, sp); cores = {}
    cores['partial'] = G & (dG >= np.quantile(dG[G], 0.5))
    cores['sub'] = sub_region(G, sp, rng, float(rng.uniform(0.3, 0.7)))
    area = G.sum(axis=tuple(k for k in range(3) if k != ax)); zs = np.nonzero(area)[0]
    if len(zs) > 3:
        z0 = int(np.argmax(area)); keep = np.zeros(G.shape[ax], bool); keep[max(0, z0 - 1):z0 + 2] = True; shp = [1, 1, 1]; shp[ax] = -1
        cores['cut2d'] = G & keep.reshape(shp)
    return {k: v for k, v in cores.items() if v.sum() >= 20}


def make_outlines(c, rng):
    G, sp = c['G'], c['sp']; dG = F.inner_depth(G, sp); dout = F.edt(G, sp)
    outl = {'dil4': G | (dout <= 4.0), 'host10': G | (c['host_incl'] & (dout <= 10.0)),
            'organ': c['host_incl'] | G, 'part80': G & (dG >= np.quantile(dG[G], 0.2))}
    nbc = c['others'] & (dout <= 15.0)
    if nbc.sum() * np.prod(sp) / 1000 >= 0.5:
        l2, _ = ndimage.label(G | nbc); ids = np.unique(l2[G]); m = np.isin(l2, ids[ids > 0])
        if (m & ~G).sum() * np.prod(sp) / 1000 >= 0.5: outl['nb15'] = m
    allowed = c['host_incl'] | c['others'] | (c['X'] > (-500 if c['mod'] == 'CT' else -1.0))
    lo = lobe_outline(G, sp, rng, allowed)
    if lo is not None: outl['lobe'] = lo
    return outl


def extents(a, outl, others, sp, G1, G2, psi):
    """stage-1 extents of anchor a and their prior w0 (ledger v8.4 rule). Returns list of (name, mask), w0 array."""
    kept = {}
    for on, om in outl.items():
        k2, _ = G1.split_lobes(om, a, sp); kept[on] = k2
    U = a.copy()
    for m in kept.values():
        U |= m
    names = ['anchor'] + list(kept) + ['U', 'U-nh']; masks = [a] + list(kept.values()) + [U, U & ~others]
    ext = G1.merge_extents(names, masks)
    w0 = G2.extent_prior([dict(logratio=float(np.log(max(int(m.sum()), 1) / max(int(a.sum()), 1))), flags=[]) for _, m in ext], psi, {})
    return ext, np.asarray(w0, float)


def q_map(ext, w0, shape):
    q = np.zeros(shape, np.float32)
    for (_, m), w in zip(ext, w0):
        q[m] += np.float32(w)
    return np.clip(q, 0.0, 1.0)


def territories(c, scen):
    """theta (dict), kb flags, segmented organ list, inner depths, for one anatomy scenario"""
    sp, hs = c['sp'], c['hs']
    theta, kb, depth = {}, {}, {}
    for k, m in c['org'].items():
        if k == hs:
            continue
        theta[k] = m; kb[k] = False; depth[k] = F.inner_depth(m, sp)
    if scen == 'excl':
        theta[hs] = c['org'][hs]; kb[hs] = False
    elif scen == 'incl':
        theta[hs] = c['host_incl']; kb[hs] = False
    elif scen == 'lost':
        theta[hs] = slab(c['org'][hs] | c['G'], c['si'], sp); kb[hs] = True
    else:
        raise ValueError(scen)
    if not kb[hs]:
        depth[hs] = F.inner_depth(theta[hs], sp)
    segmented = [k for k in theta if not kb[k]]
    return theta, kb, segmented, depth


def neighbour_sums(T, sp):
    """for every voxel: ns = sum_w s_vw over its 6 neighbours inside the crop; n1 = sum_w s_vw t_w"""
    T = np.asarray(T, np.float32); sp = np.asarray(sp, float)
    ns = np.zeros(T.shape, np.float32); n1 = np.zeros(T.shape, np.float32)
    for a in range(3):
        s = np.float32(sp.min() / sp[a])
        lo = [slice(None)] * 3; hi = [slice(None)] * 3; lo[a] = slice(0, -1); hi[a] = slice(1, None)
        ns[tuple(lo)] += s; ns[tuple(hi)] += s
        n1[tuple(lo)] += s * T[tuple(hi)]; n1[tuple(hi)] += s * T[tuple(lo)]
    return ns, n1


def appearance_stats(c, Y):
    """per-case means / covariances of the image features: tumour (all expert tumour voxels), each organ's normal tissue (organ minus
    tumour + 2 mm), and unlabelled tissue (body, no organ, no tumour, within the crop)."""
    sp = c['sp']; T = c['T']; Tn = F.dil(T, sp, 2.0)
    body = F.region_body(c['X'], c['mod'], c['ax'])
    allorg = np.zeros(T.shape, bool)
    for m in c['org'].values():
        allorg |= m

    def st(M):
        v = Y[:, M].T.astype(np.float64)
        if len(v) < 20:
            return None
        return dict(n=int(len(v)), mean=v.mean(0).tolist(), cov=np.cov(v.T).tolist())
    out = dict(ds=c['ds'], case=c['case'], mod=c['mod'], host=organ_key(c['hs']), cls=c['cls'], tumour=st(T), organs={}, generic=st(body & ~allorg & ~Tn))
    for k, m in c['org'].items():
        s_ = st(m & ~Tn)
        if s_ is not None:
            out['organs'].setdefault(organ_key(k), s_)
    xs = Y[0][T]
    if xs.size > 4000:
        xs = xs[np.random.default_rng(0).choice(xs.size, 4000, replace=False)]
    out['tumour_xs'] = [round(float(v), 3) for v in xs]
    return out
