"""Real agent outlines on R2Seg for the stage-2 validation (scripts/stage2/s2_validate_agents.py). Expert truth is used for
evaluation only.

Masks: BiomedParse and VoxTell tumour masks written by analysis/a02_tumor_segmentation/m_tumor_segmentation.py --dataset R2Seg
--seg_obj tumor into R2Seg_Seg/<agent>/<collection>/<stem>_tumor.nii.gz. No planner runs on R2Seg, so the per-host prompt masks
of TCGA (VoxTell plan masks) are absent: every anchor has at most two agent outlines.

Stage-1-like lesions, by the frozen v8.4 rules where they apply:
  anchor    a 26-connected component of (BiomedParse AND VoxTell) of >= MIN_ML (0.5 ml, ledger min_ml);
  outlines  per anchor, the components of each agent's mask that touch it ('bp', 'vt'); extents and w0 by expert.extents
            (lobes split off, merged at Dice >= dice_merge, U, U - non-host organs; extent prior with psi), as the simulation;
  lesions   anchors chained when they lie within TAU_MM = 10 mm of each other (ledger tau_mm); a lesion's q = 1 - prod(1 - q_a),
            its chosen extent xi = union of the anchors' xi, U = union of their candidate extents.
Single-agent findings are outlines only (they never form an anchor here). Truth (evaluation only): a lesion is true when one of
its anchors touches expert tumour, false otherwise; expert tumours >= MIN_ML touched by no anchor are stage-1 misses."""
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from . import paths, features as F, expert as EX

AGENTS = ('BiomedParse', 'VoxTell')
SHORT = {'BiomedParse': 'bp', 'VoxTell': 'vt'}
TAU_MM = 10.0                     # ledger v8.4 tau_mm: anchors closer than this form one lesion
PAD_MM = 60.0                     # window margin, as expert.load_case


def mask_path(agent, ds, case):
    return os.path.join(paths.root('R2Seg_Seg'), agent, ds, f'{case}_tumor.nii.gz')


def available(ds, case):
    return all(os.path.exists(mask_path(a, ds, case)) for a in AGENTS)


def _on_grid(p, ref):
    """agent mask as a boolean array on the image grid; resampled (nearest) only when the grids differ. Returns (mask, resampled)"""
    im = nib.load(p)
    d = np.asanyarray(im.dataobj)
    d = d[..., 0] if d.ndim == 4 else d
    if d.shape == ref.shape[:3] and np.allclose(im.affine, ref.affine, atol=1e-3):
        return d > 0, False
    from nibabel.processing import resample_from_to
    im3 = nib.Nifti1Image((d > 0).astype(np.uint8), im.affine)
    out = np.asarray(resample_from_to(im3, (ref.shape[:3], ref.affine), order=0).dataobj) > 0
    return out, True


def _small(lab, n):
    """component labels in the smallest integer type (memory: whole-body volumes)"""
    return lab.astype(np.int16) if n < 32767 else lab


def _box_union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return tuple(slice(min(b[k].start for b in boxes), max(b[k].stop for b in boxes)) for k in range(3))


def _dice(P, G):
    tp = float((P & G).sum()); s = float(P.sum() + G.sum())
    return 2 * tp / s if s else np.nan


def scan(ds, lf, single=False):
    """full-grid truth, agent masks, anchors and lesions of one R2Seg scan. None when an agent mask is missing.
    single=True (29 Oct) also makes single-agent anchors, as v8.4 kind 'single': a component (>= MIN_ML) of one agent's mask that
    meets no agreement voxel; its outline is itself. A single-agent anchor joins the nearest two-agent anchor's lesion within TAU_MM
    (ledger v8.4: lesions chain through two-agent anchors; a single-agent anchor joins the nearest group), else forms its own lesion.
    Every anchor has kind 'two' or 'single'; a lesion's support is 'two' when it holds a two-agent anchor, else 'single'.
    Returns dict(img, case, sp, si, T, lab (anchor labels), lab_ag {agent: component labels}, anchors [dict], lesions [dict],
    baseline (scan-level agent Dice / volumes), comps (expert tumour components >= MIN_ML: box, ml, detected), resampled)."""
    case = os.path.basename(lf).replace('_seg.nii.gz', '')
    if not available(ds, case):
        return None
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); si = [k for k, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'SI'][0]
    vml = float(np.prod(sp)) / 1000.0
    TL = EX.APPEARANCE[ds]['tumour']
    T = np.isin(EX._read_labels(lf), TL) if TL else np.zeros(img.shape[:3], bool)
    M, res_ = {}, []
    for a in AGENTS:
        M[a], r = _on_grid(mask_path(a, ds, case), img)
        if r:
            res_.append(a)
        if M[a].shape != T.shape:
            raise ValueError(f'{ds}/{case}: {a} mask shape {M[a].shape} vs labels {T.shape}')
    bp, vt = M['BiomedParse'], M['VoxTell']
    baseline = dict(T_ml=float(T.sum() * vml))
    for nm, P in (('bp', bp), ('vt', vt), ('agree', bp & vt), ('union', bp | vt)):
        baseline[f'{nm}_dice'] = _dice(P, T); baseline[f'{nm}_ml'] = float(P.sum() * vml)
        baseline[f'{nm}_fp_ml'] = float((P & ~T).sum() * vml); baseline[f'{nm}_tp_ml'] = float((P & T).sum() * vml)
    # anchors: components of the agreement >= MIN_ML; outlines: components of each agent touching them
    lab, n = ndimage.label(bp & vt, structure=EX.STRUCT26); lab = _small(lab, n)
    lab_ag = {a: _small(*ndimage.label(M[a], structure=EX.STRUCT26)) for a in AGENTS}
    del M, bp, vt
    objs = ndimage.find_objects(lab)
    obj_ag = {a: ndimage.find_objects(lab_ag[a]) for a in AGENTS}
    anchors = []
    for j in range(1, n + 1):
        sl = objs[j - 1]
        if sl is None:
            continue
        aj = lab[sl] == j
        ml = float(aj.sum() * vml)
        if ml < EX.MIN_ML:
            lab[sl][aj] = 0
            continue
        an = dict(id=j, ml=ml, box=sl, true=bool(T[sl][aj].any()), kind='two')
        boxes = [sl]
        for a in AGENTS:
            ids = [int(x) for x in np.unique(lab_ag[a][sl][aj]) if x > 0]
            an[SHORT[a]] = ids; boxes += [obj_ag[a][i - 1] for i in ids]
        an['obox'] = _box_union(boxes)                         # anchor and its outlines
        anchors.append(an)
    n_two = len(anchors)
    if single:
        # single-agent anchors: components of one agent's mask (>= MIN_ML) with no agreement voxel; ids continue after n
        nid = n
        if lab.dtype == np.int16 and n + sum(len(v) for v in obj_ag.values()) >= 32767:
            lab = lab.astype(np.int32)
        for a in AGENTS:
            for i, sl in enumerate(obj_ag[a], start=1):
                if sl is None:
                    continue
                ci = lab_ag[a][sl] == i
                ml = float(ci.sum() * vml)
                if ml < EX.MIN_ML or (lab[sl][ci] > 0).any():
                    continue
                nid += 1; lab[sl][ci] = nid
                an = dict(id=nid, ml=ml, box=sl, true=bool(T[sl][ci].any()), kind='single', obox=sl)
                an[SHORT[a]] = [i]; an[SHORT[[b for b in AGENTS if b != a][0]]] = []
                anchors.append(an)
    # lesions: two-agent anchors chained within TAU_MM (single linkage); a single-agent anchor joins the nearest two-agent anchor's
    # lesion within TAU_MM, else stays alone (ledger v8.4)
    par = list(range(len(anchors))); pos = {b['id']: k for k, b in enumerate(anchors)}

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i, an in enumerate(anchors):
        Wb = F.pad_box(an['box'], sp, TAU_MM, lab.shape)
        mine = lab[Wb] == an['id']
        if an['kind'] == 'two':
            near = F.dil(mine, sp, TAU_MM)
            for j_ in np.unique(lab[Wb][near]):
                if j_ > 0 and j_ != an['id'] and anchors[pos[int(j_)]]['kind'] == 'two':
                    par[find(pos[int(j_)])] = find(i)
        else:
            d = F.edt(mine, sp); cand = (lab[Wb] > 0) & (lab[Wb] != an['id']) & (d <= TAU_MM)
            ids = [int(j_) for j_ in np.unique(lab[Wb][cand]) if anchors[pos[int(j_)]]['kind'] == 'two']
            if ids:
                best = min(ids, key=lambda j_: float(d[lab[Wb] == j_].min()))
                par[find(i)] = find(pos[best])
    groups = {}
    for i in range(len(anchors)):
        groups.setdefault(find(i), []).append(i)
    lesions = [dict(anchors=g, true=any(anchors[i]['true'] for i in g), ml=sum(anchors[i]['ml'] for i in g),
                    obox=_box_union([anchors[i]['obox'] for i in g]),
                    support='two' if any(anchors[i]['kind'] == 'two' for i in g) else 'single') for g in groups.values()]
    # expert tumour components >= MIN_ML and whether an anchor touches them (stage-1 detection)
    lt, nt = ndimage.label(T, structure=EX.STRUCT26)
    comps = []
    for j, sl in enumerate(ndimage.find_objects(lt), start=1):
        if sl is None:
            continue
        cj = lt[sl] == j; ml = float(cj.sum() * vml)
        if ml >= EX.MIN_ML:
            comps.append(dict(box=sl, ml=ml, detected=bool((lab[sl][cj] > 0).any())))
    baseline.update(n_anchor=len(anchors), n_anchor_true=sum(a['true'] for a in anchors), n_anchor_single=len(anchors) - n_two,
                    n_anchor_single_true=sum(a['true'] for a in anchors[n_two:]), n_lesion=len(lesions),
                    n_lesion_true=sum(L['true'] for L in lesions), n_tumour=len(comps), n_tumour_detected=sum(c['detected'] for c in comps),
                    tumour_ml=sum(c['ml'] for c in comps), tumour_detected_ml=sum(c['ml'] for c in comps if c['detected']),
                    resampled='+'.join(res_))
    return dict(img=img, case=case, sp=sp, si=si, T=T, lab=lab, lab_ag=lab_ag, anchors=anchors, lesions=lesions, comps=comps,
                baseline=baseline)


def window(ds, S, W, kb, G_host=None, max_vox=3e6):
    """the scan on window W (full-grid slices), as expert.load_case builds a case: normalised image (strided above max_vox),
    expert tumour T and its components >= MIN_ML, TotalSegmentator territories (host chosen near G_host: the largest expert tumour in
    W, else the anchors), anchor and agent component labels. None without a TS mask on the image grid."""
    mod = EX.APP_MOD.get(ds, 'CT'); img = S['img']; sp = S['sp']; si = S['si']
    md, names, task = EX._load_ts(ds, S['case'], W, img)
    if md is None:
        return None
    nvox = int(np.prod([s.stop - s.start for s in W]))
    X, sp2, st = EX._image_crop(img, mod, W, sp, si, nvox, max_vox)
    cut = (lambda A: A[W][st]) if st is not None else (lambda A: A[W])
    if st is not None:
        md = md[st]
    T = cut(S['T']); lab = cut(S['lab']); lab_ag = {a: cut(v) for a, v in S['lab_ag'].items()}
    thick, ax = F.acq_axis(sp2, si)
    lb, n = ndimage.label(T, structure=EX.STRUCT26); sz = np.bincount(lb.ravel()) * np.prod(sp2) / 1000; sz[0] = 0
    comps = [lb == j for j in np.argsort(-sz) if j > 0 and sz[j] >= EX.MIN_ML]
    tcomp = lb                                                    # all expert components (any size): truth of a lesion
    G = comps[0] if comps else (G_host if G_host is not None else lab > 0)
    from .sequence import r2seg_family
    seq = 'CT' if mod == 'CT' else r2seg_family(ds, S['case'])
    c = dict(ds=ds, case=S['case'], mod=mod, seq=seq, sp=sp2, si=si, ax=ax, thick=thick, X=X, G=G, T=T, comps=comps, tcomp=tcomp,
             org={}, hs=None, cls=EX.APPEARANCE[ds]['cls'], lab=lab, lab_ag=lab_ag, W=W, st=st)
    c['ts'] = EX.ts_territories(c, md, names, task, kb, EX.SITE_ENTITY[ds])
    c['scens'] = ['ts']
    EX._ts_outline_anatomy(c)
    return c


def build_lesion(c, S, L, G1, G2, psi):
    """stage-1 quantities of lesion L on the window of case c: anchor mask A (union), per-anchor masks, q, xi, U and the agents'
    outlines (bp, vt). None when every anchor of L vanished in the window (striding)."""
    shape = c['T'].shape
    q = np.zeros(shape, np.float32); xi = np.zeros(shape, bool); U = np.zeros(shape, bool); A = np.zeros(shape, bool)
    out = {'bp': np.zeros(shape, bool), 'vt': np.zeros(shape, bool)}; amasks = []; names = []
    for i in L['anchors']:
        an = S['anchors'][i]
        a = c['lab'] == an['id']
        if a.sum() < 20:
            continue
        outl = {s: np.isin(c['lab_ag'][ag], an[s]) for ag, s in SHORT.items()}
        ext, w0 = EX.extents(a, outl, c['others'], c['sp'], G1, G2, psi)
        qa = EX.q_map(ext, w0, shape); q = 1.0 - (1.0 - q) * (1.0 - qa)
        ib = int(np.argmax(w0)); xi |= ext[ib][1]; names.append(ext[ib][0])
        for _, m in ext:
            U |= m
        A |= a; amasks.append(a)
        for s in out:
            out[s] |= outl[s]
    if not A.any():
        return None
    return dict(A=A, amasks=amasks, q=q.astype(np.float32), xi=xi, U=U, bp=out['bp'], vt=out['vt'], xi_name='+'.join(names))


def lesion_truth(c, A):
    """expert tumour of a lesion: every expert component (any size) its anchors touch"""
    ids = [j for j in np.unique(c['tcomp'][A]) if j > 0]
    return np.isin(c['tcomp'], ids) if ids else np.zeros(A.shape, bool)
