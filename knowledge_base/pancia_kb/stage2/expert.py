"""Expert-data simulation for stage 2 (R2Seg only; Ovary_Tumor_00 is never read).

Same construction as the stage-1 expert testbed (clinical/PanCIA_outputs/scripts/tb_build.py, v7/v8): for the largest expert tumour
component G of a case and its true host h* (the labelled organ touching G most),
  cores (agent-agreement-like anchors): partial (deepest 50% of G), sub (random compact 30-70%), cut2d (3 acquisition slices);
  outlines (agent-like claims): dil4, host10, organ, part80, nb15, lobe. The testbed's 'exact' outline (the expert mask itself) is
  left out for stage 2: no agent draws the expert outline, and with it the prior would learn to trust the chosen extent too much
  and the chosen-extent baseline of the validation would be optimistic;
  extents of an anchor: {anchor, outlines after lobe splitting, U, U - non-host organs}, merged at Dice >= 0.95 (grounding_g1),
  with the stage-1 extent prior w0 ~ psi(log |xi| / |a|) (fit_host6.json psi, eta = 1), exactly as ledger v8.4.
Anatomy (user decision 25 Oct): the spatial prior is fitted on the TotalSegmentator anatomy only ('ts': territories and the organs
the simulated outlines are drawn around), as on TCGA. Expert organ labels (5 collections) are used only in validation, as the
'excl' scenario (expert label maps are exclusive, so the tumour never lies in the expert organ, while the TS host contains 34-100%
of it; 25 Oct check). Cases: up to n_per = 100 drawn at random from the USABLE cases of each collection (usability index,
s2_r2seg_index.py: largest expert tumour component >= 0.5 ml and a TS mask on the image grid).
Every simulated outline is drawn around a real expert tumour (the prior is
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
HOLLOW = {'uterus', 'cervix', 'stomach', 'duodenum', 'esophagus', 'gallbladder', 'bladder', 'urinary_bladder', 'colon', 'rectum'}
CORES = ('partial', 'sub', 'cut2d')
SCENS = ('excl', 'incl', 'lost', 'ts')       # index stored in the samples (kept for compatibility); 'ts' = TotalSegmentator territories
FIT_SCENS = ('ts',)                           # fitting uses the TS anatomy only (user, 25 Oct); 'excl' is a validation scenario
MIN_ML = 0.5                                  # smallest expert tumour simulated as a lesion (ml); design value, as stage 1
STRUCT26 = np.ones((3, 3, 3), bool)           # tumour components: 26-connectivity (diagonal fragments are one tumour)


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


def _seqtag(f):
    return 'adc' if '_adc_' in f else ('t2' if '_t2_' in f else '')


def index_files(r2):
    """every R2Seg label file (collection, file name) for the usability index"""
    out = []
    for ds in sorted(APPEARANCE):
        d = os.path.join(r2, ds, 'labels')
        if os.path.isdir(d):
            out += [(ds, f) for f in sorted(os.listdir(d)) if f.endswith('_seg.nii.gz')]
    return out


def label_status(ds, lf):
    """usability of one label file: status usable / small / no_tumour (tumour collections), normal_only / no_label (Uterus_Tumor_00,
    benign), the number of tumour components and the largest one's volume (ml). The expert host is not required (fitting uses TS)."""
    A = _read_labels(lf); sp = np.array(nib.load(lf).header.get_zooms()[:3], float)
    TL = APPEARANCE[ds]['tumour']
    if not TL:
        return dict(status='normal_only' if (A > 0).any() else 'no_label', n_comp=0, n_tumours=0, largest_ml=0.0)
    T = np.isin(A, TL); del A
    if not T.any():
        return dict(status='no_tumour', n_comp=0, n_tumours=0, largest_ml=0.0)
    W = F.pad_box(F.bbox(T), sp, 5.0, T.shape)
    lb, n = ndimage.label(T[W], structure=STRUCT26); sz = np.bincount(lb.ravel())[1:] * np.prod(sp) / 1000; ml = float(sz.max())
    return dict(status='usable' if ml >= MIN_ML else 'small', n_comp=int(n), n_tumours=int((sz >= MIN_ML).sum()), largest_ml=round(ml, 3))


def ts_paths(ds, case):
    from . import paths
    root = os.path.join(paths.root('R2Seg_Seg'), 'TotalSegmentator', ds)
    return os.path.join(root, f'{case}_organ.nii.gz'), os.path.join(root, f'{case}_organ_labels.json')


def read_index():
    """usability index (s2_r2seg_index.py): DataFrame ds, file, seq, status, n_comp, largest_ml, ts"""
    import glob
    import pandas as pd
    from . import paths
    fs = glob.glob(os.path.join(paths.out_dir('expert', 'index'), 'r2seg_index*.csv'))
    if not fs:
        raise FileNotFoundError('no usability index: run scripts/stage2/s2_r2seg_index.py first')
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(['ds', 'file'], keep='last')
    d['seq'] = d['seq'].fillna('').astype(str)
    return d


def _draw(d, r2, n_per, salt):
    """at most n_per files drawn at random (fixed seed per group) from the index rows d of one group"""
    fs = sorted(d.file)
    fs = list(np.random.default_rng(zlib.crc32(salt.encode())).permutation(fs)) if fs else []
    if n_per:
        fs = fs[:n_per]
    return fs


def case_list(r2, n_per=None):
    """spatial-prior cases: up to n_per drawn at random from the usable cases of each spatial collection (largest tumour >= MIN_ML,
    TS mask present; Prostate T2 images only). n_per = None: all usable cases (validation)."""
    d = read_index(); d = d[(d.status == 'usable') & d.ts.astype(bool) & (d.seq != 'adc')]
    out = []
    for ds in SPATIAL_SETS:
        out += [(ds, os.path.join(r2, ds, 'labels', f)) for f in _draw(d[d.ds == ds], r2, n_per, ds + 'tb')]
    return out


def negative_list(r2):
    """tumour-free scans of the spatial collections with a TS mask (validation only: false anchors in scans without tumour)"""
    d = read_index(); d = d[(d.status == 'no_tumour') & d.ts.astype(bool) & (d.seq != 'adc') & d.ds.isin(SPATIAL_SETS)]
    out = []
    for ds in SPATIAL_SETS:
        out += [(ds, os.path.join(r2, ds, 'labels', f)) for f in _draw(d[d.ds == ds], r2, None, ds + 'neg')]
    return out


def load_case(ds, lf, max_vox=3e6, kb=None, ts=True):
    """crop (tumour box + 60 mm), truth, TotalSegmentator territories (required: fitting uses TS anatomy only), expert organs and the
    expert host where labelled (validation 'excl' only), normalised image. Returns None when unusable (largest tumour < MIN_ML,
    or no TS mask on the image grid)."""
    mod = APP_MOD.get(ds, 'CT'); case = os.path.basename(lf).replace('_seg.nii.gz', '')
    if kb is None or not ts:
        raise ValueError('load_case needs the knowledge base and the TS masks (fitting uses TS anatomy only)')
    A = _read_labels(lf)
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); si = [k for k, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'SI'][0]
    TL = APPEARANCE[ds]['tumour']
    if not TL:
        return None                                              # UMD: benign, not a tumour case
    Tm = np.isin(A, TL)
    if not Tm.any():
        return None
    W = F.pad_box(F.bbox(Tm), sp, 60.0, A.shape)
    A = A[W]; del Tm
    if ds in SETS:
        org, T = organs_of(ds, A)
    else:
        org, T = {}, np.isin(A, TL)
    del A
    lb, n = ndimage.label(T)
    if n == 0:
        return None
    sz = np.bincount(lb.ravel()); sz[0] = 0; j = int(np.argmax(sz))
    if sz[j] * np.prod(sp) / 1000 < MIN_ML:
        return None
    G = lb == j; del lb
    hs = true_host(G, org, sp) if org else None            # expert host: validation only ('excl'); None is allowed
    md, names, task = _load_ts(ds, case, W, img)
    if md is None:
        return None
    X, sp, st = _image_crop(img, mod, W, sp, si, G.size, max_vox)
    if st is not None:
        G = G[st]; T = T[st]; org = {k: m[st] for k, m in org.items()}; md = md[st]
    org = {k: m for k, m in org.items() if m.any()}
    if hs is not None and hs not in org:
        hs = None
    thick, ax = F.acq_axis(sp, si)
    # every expert tumour >= MIN_ML is simulated as a lesion of its own (user, 25 Oct: R2Seg scans hold several tumours; liver 68%,
    # NSCLC lung 52% of cases); largest first. Smaller specks stay tumour truth but get no anchor.
    lb, n = ndimage.label(T, structure=STRUCT26); sz = np.bincount(lb.ravel()) * np.prod(sp) / 1000; sz[0] = 0
    comps = [lb == j for j in np.argsort(-sz) if j > 0 and sz[j] >= MIN_ML] or [G]
    G = comps[0]; del lb
    from .sequence import r2seg_family
    seq = 'CT' if mod == 'CT' else r2seg_family(ds, case)
    c = dict(ds=ds, case=case, mod=mod, seq=seq, sp=sp, si=si, ax=ax, thick=thick, X=X, G=G, T=T, comps=comps,
             org=org, hs=hs, cls=APPEARANCE[ds]['cls'])
    c['ts'] = ts_territories(c, md, names, task, kb, SITE_ENTITY[ds])
    c['scens'] = ['ts'] + (['excl'] if hs is not None else [])
    _ts_outline_anatomy(c)
    return c


def _image_crop(img, mod, W, sp, si, nvox, max_vox):
    """normalised image on the crop W (MR: normalisation on the whole scan, as on TCGA); in-plane striding above max_vox voxels.
    Returns X, the (possibly coarser) spacing and the stride (None when not strided)."""
    if mod == 'MR':
        Xf = np.asarray(img.dataobj, np.float32); Xf = Xf[..., 0] if Xf.ndim == 4 else Xf
        thick0, ax0 = F.acq_axis(sp, si)
        cen, scl = F.norm_params(Xf, mod, F.body_mask(Xf, mod, ax0)); X = Xf[W]; del Xf
    else:
        cen, scl = 0.0, 1.0
        X = np.asarray(img.dataobj[W], np.float32); X = X[..., 0] if X.ndim == 4 else X
    st = None
    if nvox > max_vox:
        thin = sp.max() / sp.min() < 1.5
        axes = [0, 1, 2] if thin else [k for k in range(3) if k != int(np.argmax(sp))]
        f = int(np.ceil((nvox / max_vox) ** (1.0 / len(axes))))
        st = tuple(slice(None, None, f) if k in axes else slice(None) for k in range(3))
        X = X[st]; sp = np.array([s * f if k in axes else s for k, s in enumerate(sp)])
    return F.normalise(X, mod, cen, scl), sp, st


def _ts_outline_anatomy(c):
    """anatomy the simulated agent outlines are drawn around (host10, organ, nb15, lobe, U - non-host): the TS anatomy for every
    collection. A host that TS does not segment (knowledge-base region) gives no host outline. The host is slice-filled with all
    tumours, so a tumour inside the host's contour counts as part of it (host_incl of a tumour: host_seg | (tumour & Hf))."""
    th = c['ts']['theta']; hk = c['ts']['host']; T = c['T']
    seg = not c['ts']['kb'][hk]
    host_seg = th[hk] if seg else np.zeros(T.shape, bool)
    Hf = (_fill2d(th[hk] | T, c['ax']) | th[hk]) if seg else np.zeros(T.shape, bool)
    others = np.zeros(T.shape, bool)
    for k, m in th.items():
        if k != hk:
            others |= m
    c.update(host_seg=host_seg, Hf=Hf, host_incl=host_seg | (T & Hf), others=others)


def load_negative(ds, lf, kb, max_vox=3e6):
    """a tumour-free scan (validation only): crop = TS host box + 60 mm, TS territories, empty truth. None without a TS host."""
    mod = APP_MOD.get(ds, 'CT'); case = os.path.basename(lf).replace('_seg.nii.gz', '')
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); si = [k for k, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'SI'][0]
    full = tuple(slice(0, n) for n in img.shape[:3])
    md, names, task = _load_ts(ds, case, full, img)
    if md is None:
        return None
    from ..adapter import _host_ts_labels
    hl = list(_host_ts_labels(kb, task, names, SITE_ENTITY[ds]))
    H = np.isin(md, hl) if hl else np.zeros(md.shape, bool)
    if H.sum() * np.prod(sp) / 1000 < 5.0:                     # no segmented host to put a false anchor in
        return None
    W = F.pad_box(F.bbox(H), sp, 60.0, md.shape); md = md[W]; del H
    X, sp, st = _image_crop(img, mod, W, sp, si, md.size, max_vox)
    if st is not None:
        md = md[st]
    thick, ax = F.acq_axis(sp, si)
    from .sequence import r2seg_family
    seq = 'CT' if mod == 'CT' else r2seg_family(ds, case)
    Z = np.zeros(md.shape, bool)
    c = dict(ds=ds, case=case, mod=mod, seq=seq, sp=sp, si=si, ax=ax, thick=thick, X=X, G=Z, T=Z, org={}, hs=None,
             cls=APPEARANCE[ds]['cls'])
    c['ts'] = ts_territories(c, md, names, task, kb, SITE_ENTITY[ds])
    if c['ts']['kb'][c['ts']['host']]:
        return None
    c['scens'] = ['ts']
    return c


# option A (user, 24 Oct): TotalSegmentator organ masks on R2Seg give territories; tumour truth stays expert. The true host of a
# collection is its site organ (collections are organ-specific); when TS has no class for it (breast, cervix) or finds nothing in the
# crop, the host is unsegmented and its territory is the knowledge-base prior region (pancia_kb.adapter.host_prior_region), as on TCGA.
SITE_ENTITY = {'Kidney_Tumor_00': 'kidney', 'Liver_Tumor_00': 'liver', 'Pancreas_Tumor_00': 'pancreas', 'Colon_Tumor_00': 'colon',
               'Lung_Tumor_00': 'lung', 'Lung_Tumor_01': 'lung', 'Cervix_Tumor_00': 'cervix', 'Bladder_Tumor_00': 'urinary_bladder',
               'Breast_Tumor_00': 'breast', 'Prostate_Tumor_00': 'prostate'}
SPATIAL_SETS = tuple(SITE_ENTITY)
SITE_OF = dict(SITE_ENTITY, Uterus_Tumor_00='uterus')     # tumour site of every collection (fit weighting: sites count equally)


def _load_ts(ds, case, W, ref):
    f, fl = ts_paths(ds, case)
    if not (os.path.exists(f) and os.path.exists(fl)):
        return None, None, None
    meta = __import__('json').load(open(fl))
    im = nib.load(f)
    if im.shape[:3] != ref.shape[:3] or not np.allclose(im.affine, ref.affine, atol=1e-3):
        print(f'WARNING {ds}/{case}: TotalSegmentator mask not on the image grid; ts scenario skipped', flush=True)
        return None, None, None
    md = np.asarray(im.dataobj[W]).astype(np.int16)
    return md, {int(k): v for k, v in meta['labels'].items()}, meta['task']


def ts_territories(c, md, names, task, kb, host_ent):
    """TS masks on the crop, grouped by KB entity and side (lung lobes -> lung of that side, as stage 1); unmapped classes keep
    their TS name. Returns dict(theta, kb, host, segmented, depth)."""
    from ..adapter import _host_ts_labels, _side_of, host_prior_region
    sp = c['sp']; G = c['G']
    ts_map = kb.ts_to_entity.get(task, {})
    host_labs = set(_host_ts_labels(kb, task, names, host_ent))
    labs = [int(v) for v in np.unique(md) if v > 0]
    groups = {}
    for lab in labs:
        nm = names.get(lab, str(lab))
        if lab in host_labs:
            key = ('host:' + host_ent, _side_of(kb, task, nm))
        else:
            e = ts_map.get(nm)
            key = (e[0], e[1] if len(e) > 1 and e[1] else _side_of(kb, task, nm)) if e else ('ts:' + nm, None)
        groups.setdefault(key, []).append(lab)
    theta = {k: np.isin(md, v) for k, v in groups.items()}
    hkeys = [k for k in theta if k[0] == 'host:' + host_ent]
    near = F.dil(G, sp, 10.0)
    hk = ('host', host_ent)
    if hkeys:
        best = max(hkeys, key=lambda k: (int((theta[k] & near).sum()), int(theta[k].sum())))
        theta[hk] = theta.pop(best)
        for k in hkeys:
            if k != best:
                theta[('other_side:' + host_ent, k[1])] = theta.pop(k)
        kbf = False
    else:
        box, _ = host_prior_region(kb, host_ent, md, names, task, sp)
        theta[hk] = box if box is not None else slab(G, c['si'], sp)
        kbf = True
    kbd = {k: False for k in theta}; kbd[hk] = kbf
    segmented = [k for k in theta if not kbd[k]]
    depth = {k: F.inner_depth(theta[k], sp) for k in segmented}
    return dict(theta=theta, kb=kbd, host=hk, segmented=segmented, depth=depth, host_found=not kbf)


def _read_labels(lf):
    """integer label map. Some label files are stored scaled (Prostate158: uint16 with slope 1/21845), so the stored integers are
    only used directly when the scaling is the identity (memory on whole-body FLARE volumes)."""
    lim = nib.load(lf)
    try:
        raw = np.asarray(lim.dataobj.get_unscaled())
        sl, ic = lim.dataobj.slope, lim.dataobj.inter
        ident = (sl is None or np.isnan(sl) or sl == 1) and (ic is None or np.isnan(ic) or ic == 0)
        if ident:
            return raw.astype(np.uint8) if raw.max() < 256 else np.rint(raw).astype(np.int16)
        return np.rint(raw.astype(np.float32) * np.float32(sl) + np.float32(ic or 0.0)).astype(np.int16)
    except Exception:
        return np.rint(np.asanyarray(lim.dataobj)).astype(np.int16)


def _fill2d(M, ax):
    M = M.copy()
    for z in range(M.shape[ax]):
        sl = [slice(None)] * 3; sl[ax] = z; M[tuple(sl)] = ndimage.binary_fill_holes(M[tuple(sl)])
    return M


def make_cores(c, rng, G=None):
    """simulated anchors (agent-agreement-like cores) of one expert tumour G (default: the largest)"""
    G = c['G'] if G is None else G
    sp, ax = c['sp'], c['ax']
    dG = F.inner_depth(G, sp); cores = {}
    cores['partial'] = G & (dG >= np.quantile(dG[G], 0.5))
    cores['sub'] = sub_region(G, sp, rng, float(rng.uniform(0.3, 0.7)))
    area = G.sum(axis=tuple(k for k in range(3) if k != ax)); zs = np.nonzero(area)[0]
    if len(zs) > 3:
        z0 = int(np.argmax(area)); keep = np.zeros(G.shape[ax], bool); keep[max(0, z0 - 1):z0 + 2] = True; shp = [1, 1, 1]; shp[ax] = -1
        cores['cut2d'] = G & keep.reshape(shp)
    return {k: v for k, v in cores.items() if v.sum() >= 20}


def make_outlines(c, rng, G=None):
    """agent-like outlines of one expert tumour G that do not depend on the anchor. Over-segmenting / leaking: dil4, host10, organ,
    nb15, lobe; under-segmenting: part80 (and, per anchor, anchor_outlines)."""
    G = c['G'] if G is None else G
    sp = c['sp']; dG = F.inner_depth(G, sp); dout = F.edt(G, sp)
    # part80: the deepest 80% of the tumour by rank (ties broken at random). A depth threshold kept every voxel of thin tumours
    # (many voxels share the minimum depth), i.e. the exact expert mask (self-review 24 Oct).
    gi = np.flatnonzero(G); order = np.lexsort((rng.random(len(gi)), dG.ravel()[gi]))
    p80 = np.zeros(G.size, bool); p80[gi[order[int(np.floor(0.2 * len(gi))):]]] = True
    outl = {'dil4': G | (dout <= 4.0), 'part80': p80.reshape(G.shape)}
    host_incl = c['host_seg'] | (G & c['Hf'])
    if (host_incl & ~G).any():               # host outlines need a segmented host (none when the host is a KB region: breast)
        outl['host10'] = G | (host_incl & (dout <= 10.0)); outl['organ'] = host_incl | G
    nbc = c['others'] & (dout <= 15.0)
    if nbc.sum() * np.prod(sp) / 1000 >= 0.5:
        l2, _ = ndimage.label(G | nbc); ids = np.unique(l2[G]); m = np.isin(l2, ids[ids > 0])
        if (m & ~G).sum() * np.prod(sp) / 1000 >= 0.5: outl['nb15'] = m
    allowed = host_incl | c['others'] | (c['X'] > (-500 if c['mod'] == 'CT' else -1.0))
    lo = lobe_outline(G, sp, rng, allowed)
    if lo is not None: outl['lobe'] = lo
    return {k: m for k, m in outl.items() if (m != G).any()}       # no simulated agent draws the expert mask (design value 4)


def anchor_outlines(c, rng, G, a):
    """under-segmenting outlines of tumour G that depend on its anchor a (user, 25 Oct: agents also miss parts of a tumour):
    sub     a random compact part of G (30-70%, the rule of the 'sub' core) joined with the anchor;
    trunc   G cut at a random slice beyond the anchor at one end of the acquisition stack (agents often miss end slices);
            the cut is uniform over the slices between the anchor's last slice and the tumour's last slice;
    core4   the anchor grown by 4 mm (follows the tumour only where the anchor is)."""
    sp, ax = c['sp'], c['ax']; out = {}
    out['sub'] = sub_region(G, sp, rng, float(rng.uniform(0.3, 0.7))) | a
    zG = np.nonzero(G.any(axis=tuple(k for k in range(3) if k != ax)))[0]
    za = np.nonzero(a.any(axis=tuple(k for k in range(3) if k != ax)))[0]
    ends = []
    if zG[-1] > za[-1]: ends.append('hi')
    if zG[0] < za[0]: ends.append('lo')
    if ends:
        e = ends[int(rng.integers(len(ends)))]; keep = np.zeros(G.shape[ax], bool)
        if e == 'hi':
            cut = int(rng.integers(za[-1], zG[-1])); keep[:cut + 1] = True          # keeps slices <= cut (cut < last tumour slice)
        else:
            cut = int(rng.integers(zG[0] + 1, za[0] + 1)); keep[cut:] = True       # keeps slices >= cut (cut > first tumour slice)
        shp = [1, 1, 1]; shp[ax] = -1
        out['trunc'] = G & keep.reshape(shp)
    out['core4'] = F.dil(a, sp, 4.0)
    return {k: m for k, m in out.items() if m.sum() >= 20 and (m != G).any()}


def agent_claims(outl, rng):
    """the outlines the agents of one anchor return: a random subset of the outline types, every subset equally likely
    (each type kept with probability 1/2). Uninformative: no rate of agent errors is assumed. Real stage-1 anchors carry 0-9
    outline extents after merging (v8.4 pilot; a sanity check only, not used to set this)."""
    return {k: m for k, m in outl.items() if rng.random() < 0.5}


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


def tumour_features(ds, lf, rng, max_vox=3e6, n_max=5000):
    """image features (smoothed intensity, local SD; as stage 2) of every expert tumour >= MIN_ML of one scan, for the population
    prior of the Student-t tail shape nu (s2_nu.py). Same normalisation and striding as load_case. Returns (meta, [(j, ml, Y)]) with
    Y (n, D), at most n_max voxels per tumour (uniform sample); None when unusable."""
    mod = APP_MOD.get(ds, 'CT'); case = os.path.basename(lf).replace('_seg.nii.gz', '')
    TL = APPEARANCE[ds]['tumour']
    if not TL:
        return None
    A = _read_labels(lf)
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); si = [k for k, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'SI'][0]
    T = np.isin(A, TL); del A
    if not T.any():
        return None
    W = F.pad_box(F.bbox(T), sp, 10.0, T.shape); T = T[W]
    X, sp2, st = _image_crop(img, mod, W, sp, si, T.size, max_vox)
    if st is not None:
        T = T[st]
    thick, ax = F.acq_axis(sp2, si)
    Y = F.image_features(X, sp2, thick, ax)
    lb, n = ndimage.label(T, structure=STRUCT26); sz = np.bincount(lb.ravel()) * np.prod(sp2) / 1000; sz[0] = 0
    out = []
    for j, lab in enumerate([k for k in np.argsort(-sz) if k > 0 and sz[k] >= MIN_ML]):
        idx = np.flatnonzero((lb == lab).ravel())
        if len(idx) > n_max:
            idx = rng.choice(idx, n_max, replace=False)
        out.append((j, float(sz[lab]), Y.reshape(Y.shape[0], -1)[:, idx].T.astype(np.float64)))
    from .sequence import r2seg_family
    seq = 'CT' if mod == 'CT' else r2seg_family(ds, case)
    return dict(ds=ds, case=case, mod=mod, seq=seq), out


def _patch(M):
    """(slices, crop) of an array's non-zero box (memory: many lesions per scan)"""
    b = F.bbox(M)
    if b is None:
        b = (slice(0, 1),) * 3
    return (b, M[b].copy())


def full(pt, shape, dtype=bool):
    out = np.zeros(shape, dtype); out[pt[0]] = pt[1]; return out


def simulate_lesions(c, rng, psi, G1, G2):
    """stage-1-like lesions of one scan (user, 25 Oct): every expert tumour >= MIN_ML is a lesion with its own simulated anchor
    (one per core type) and agent claims (agent_claims: a random subset of its outline types, over- and under-segmenting);
    extents and w0 by the frozen stage-1 (v8.4) rule. Returns {core type: [lesion dict]} with patches (_patch) of the anchor a,
    the marginal q, the chosen extent xi and the union of candidate extents U; per lesion also its tumour index j, the chosen
    extent's name, the claims and the farthest distance from the anchor to its own tumour (for r)."""
    sp = c['sp']; out = {}
    cores_all = [make_cores(c, rng, Gj) for Gj in c['comps']]
    types = [cn for cn in CORES if any(cn in cs for cs in cores_all)]
    for j, Gj in enumerate(c['comps']):
        # every tumour is proposed in every core group of the scan: a tumour too small (or too thin) for a core type gets its
        # 'partial' core, or the whole tumour, as that anchor. Otherwise small tumours would be unproposed in some groups and
        # reappear as tumour beyond every outline, at a rate set by voxel size (self-review 25 Oct).
        cores = cores_all[j]
        fb = cores.get('partial', Gj)
        cores = {cn: cores.get(cn, fb) for cn in types}
        base = make_outlines(c, rng, Gj)
        Wg = F.pad_box(F.bbox(Gj), sp, 2.0, Gj.shape)
        for cn, a in cores.items():
            outl = agent_claims({**base, **anchor_outlines(c, rng, Gj, a)}, rng)
            ext, w0 = extents(a, outl, c['others'], sp, G1, G2, psi)
            q = q_map(ext, w0, a.shape); ib = int(np.argmax(w0))
            U = np.zeros(a.shape, bool)
            for _, m in ext:
                U |= m
            far = float(F.edt(a[Wg], sp)[Gj[Wg]].max())
            out.setdefault(cn, []).append(dict(j=j, a=_patch(a), q=_patch(q), xi=_patch(ext[ib][1]), U=_patch(U), xi_name=ext[ib][0],
                                claims=sorted(outl), n_ext=len(ext), far_mm=far, w0=[float(x) for x in w0],
                                q_on_anchor_min=float(q[a].min())))
            del ext, q, U
    return {cn: out[cn] for cn in CORES if out.get(cn)}


def q_map(ext, w0, shape):
    q = np.zeros(shape, np.float32)
    for (_, m), w in zip(ext, w0):
        q[m] += np.float32(w)
    return np.clip(q, 0.0, 1.0)


def territories(c, scen):
    """theta (dict), kb flags, segmented organ list, inner depths and the host key, for one anatomy scenario.
    ts: TotalSegmentator territories (fitting and validation); excl: expert organ labels (validation only, 5 collections)."""
    if scen == 'ts':
        t = c['ts']
        return t['theta'], t['kb'], t['segmented'], t['depth'], t['host']
    if scen != 'excl' or c['hs'] is None:
        raise ValueError(f'scenario {scen} not available (incl / lost were dropped on 25 Oct)')
    sp, hs = c['sp'], c['hs']
    theta = dict(c['org']); kb = {k: False for k in theta}
    depth = {k: F.inner_depth(m, sp) for k, m in theta.items()}
    return theta, kb, list(theta), depth, hs


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


def tumour_sample(c, Y):
    """sample (<= 4000 voxels) of the expert tumour's smoothed intensity, for the Student-t tail shape nu (fit.fit_nu standardises it
    within the case, so only its shape is used)"""
    xs = Y[0][c['T']]
    if xs.size > 4000:
        xs = xs[np.random.default_rng(0).choice(xs.size, 4000, replace=False)]
    return dict(ds=c['ds'], case=c['case'], mod=c['mod'], seq=c.get('seq'), tumour_xs=[round(float(v), 3) for v in xs])


# ------------------------------------------------------------------ label table of the R2Seg collections
# collection -> tumour labels, organ labels {name: [labels]}, labels excluded from every class (benign lesions), tumour host class
# (solid / hollow: a validation stratum; the model itself has no class term). Labels checked on the data (24 Oct): Prostate158 T2
# labels 1-2 = prostate zones, 3 = tumour; ADC label 3 only. UMD (uterine myoma MRI; labels 1 uterine wall, 2 cavity, 3 myoma,
# 4 nabothian cyst, Pan et al., Sci Data 2024): benign, no tumour; indexed only (not used by stage 2 since the appearance pass was
# removed on 25 Oct). The table name is historical.
APPEARANCE = {
    'Kidney_Tumor_00': dict(tumour=[2], organs={'kidney': [1]}, cls='solid'),
    'Liver_Tumor_00': dict(tumour=[14], organs={organ_key(n): [v] for v, n in FLARE.items()}, cls='solid'),
    'Pancreas_Tumor_00': dict(tumour=[14], organs={organ_key(n): [v] for v, n in FLARE.items()}, cls='solid'),
    'Colon_Tumor_00': dict(tumour=[14], organs={}, cls='hollow'),
    'Lung_Tumor_00': dict(tumour=[1], organs={}, cls='solid'),
    'Lung_Tumor_01': dict(tumour=[5], organs={'lung': [1], 'heart': [2], 'esophagus': [3]}, cls='solid'),
    'Cervix_Tumor_00': dict(tumour=[2], organs={'uterus': [1]}, cls='hollow'),
    'Bladder_Tumor_00': dict(tumour=[1], organs={}, cls='hollow'),
    'Breast_Tumor_00': dict(tumour=[1], organs={}, cls='solid'),
    'Prostate_Tumor_00': dict(tumour=[3], organs={'prostate': [1, 2]}, cls='solid'),
    'Uterus_Tumor_00': dict(tumour=[], organs={'uterus': [1, 2]}, exclude=[3, 4], cls='hollow'),
}
APP_MOD = {'Bladder_Tumor_00': 'MR', 'Breast_Tumor_00': 'MR', 'Cervix_Tumor_00': 'MR', 'Prostate_Tumor_00': 'MR', 'Uterus_Tumor_00': 'MR'}
