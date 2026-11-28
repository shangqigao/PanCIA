"""Stage-1 decision, part 1a: augmented candidates for the evidence distributions (28 Nov 2026; plan
claude/stage1_decision_plan_2026-11-28.md, decision 1 = D, user-approved categories and 20 mm buffer; workflow claude/stage1_workflow.md).

Candidates are samples of a candidate-generating distribution; the agents give some samples, expert labels give the truth, and this
script draws further samples by design. DESIGN SET ONLY: tumour scans of v1 jobs.csv + v2 jobs_v2.csv. The 139 tumour-free scans
(evaluation reference), jobs_v3.csv, CancerVerse and TCGA are never read.

Workflow steps consumed: 1a (raw BiomedParse / VoxTell masks: the real candidates), 2g g2 (anatomy layer: non-tumour seeds, leakage
neighbours), 4g (ledger/graph_v3/design.csv: real candidates' truth label, scope and region category), 0 (expert labels: truth).
No cancer type, no sex, no agreement.

Rows written (one per candidate):
  source 'real': every design-set real candidate (three modes), features recomputed here with the same definitions as the 26 Nov
                 evidence screen (elongation, nsl, sphericity, thickness), truth / scope / category from design.csv.
  source 'aug' : augmented candidates.
     tumour side: seeds = expert tumour components (>= 0.05 ml, 26-connected); variants = the seed itself and 4 perturbed versions.
     non-tumour side: seeds drawn from the step-2g anatomy layer in five categories (user, 28 Nov):
        solid (organ / gland cores), hollow (hollow-organ cores: wall, folds, collapsed loops, lumen content), vessel (vessel cores),
        rim (band of 2 voxels inside a usable organ territory boundary), outside (just beyond a territory boundary, in no territory);
        never lymph stations, excluded or set-aside organs; never within 20 mm of an expert tumour (user, 28 Nov); a variant that
        touches an expert tumour after perturbation is discarded. Category, volume and slab thickness come from a real false-candidate
        exemplar (see 'Two stages' below; D8 size-matched); the seed is intersected with the region; the component at the centre.
     operators (same set on both sides; 1-3 drawn per variant, random order; identity with p 0.2):
        partial (geodesic growth to 20-100% of the seed), fragment (smoothed random field threshold, one component), slices (keep a
        contiguous run of slices or drop slices along the largest-spacing axis), boundary (in-plane dilate / erode 1-2 voxels),
        leak (add neighbouring usable organ core across the boundary, <= 50% of the volume), resolution (block-majority to 3 / 5 /
        7.5 mm along the slice axis and back, only if coarser than native). After the operators: one random component >= 0.05 ml.
     truth of augmented tumour-side candidates: the strict rule on the expert labels (T: in_frac >= 0.5 or dice_t >= 0.1; P dropped).
Two stages (28 Nov, after the smoke test): --stage real writes real.csv (all design-set real candidates; must be complete first);
--stage aug writes aug.csv. Non-tumour seeds are conditioned on a real false-candidate exemplar drawn from the whole design set
(same category and slice-spacing class): its volume and its slab thickness (slices spanned x slice spacing, re-expressed on the
scan's own slices) define a slab of in-plane ellipses (aspect 1 or U(1, 6), random in-plane angle), so slice-wise BiomedParse-like
slivers and compact VoxTell-like blobs are both reproduced. 'outside' seeds lie just outside an organ territory, in no territory.
Output: PanCIA_outputs/ledger/decision_s1/augment[/<tag>]/{real,aug}.csv (resumable by scan); random seed per scan from its name.
--stage real_tf (28 Nov, overnight run) writes real_tf.csv: features of the real candidates of the TUMOUR-FREE design scans (v1 / v2
job files), for evaluation of the decision on the design set only; never used as exemplars, never fitted, never used to choose a
threshold (the 139 tumour-free scans are an evaluation reference, not calibration data).
Usage: python s1_augment_candidates.py --stage real ; --stage real_tf ; --stage aug [--tag smoke] [--only ...]"""
import os, sys, json, time, argparse, csv, zlib
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from pancia_kb.stage2 import paths, expert as EX     # noqa: E402
from pancia_kb import KnowledgeBase                    # noqa: E402
import s1_graph_v3 as G3                               # noqa: E402  (anatomy-layer readers: Bx, unpack_bx, whole_of, box_of)

R = paths.root('R2Seg'); S = paths.root('R2Seg_Seg')
ANAT = os.path.join(paths.root('outputs'), 'anatomy', 'r2seg')
OUTD = os.path.join(paths.ledger_dir(), 'decision_s1', 'augment')
JOBS = [os.path.join(paths.ledger_dir(), 'evidence_screen', f) for f in ('jobs.csv', 'jobs_v2.csv')]
DESIGN = os.path.join(paths.ledger_dir(), 'graph_v3', 'design.csv')
KBD = os.path.abspath(os.path.join(HERE, '..', '..'))
MIN_ML = 0.05; BUFFER_MM = 20.0
EXCL_FLAGS = ('small', 'spatial', 'not_between')
CAT_OF_KB = {'organ': 'solid', 'gland': 'solid', 'hollow_organ': 'hollow', 'vessel_artery': 'vessel', 'vessel_vein': 'vessel',
             'muscle': 'outside', 'bone': 'outside', 'fat': 'outside'}
COLS = ['ds', 'case', 'mod', 'source', 'mode', 'cid', 'category', 'seed', 'ops', 'lab', 'in_scope', 'ml', 'elong', 'nsl', 'sph',
        'thick_mm', 'maxsp', 'note']


def feats(cb, sp):
    """same definitions as s1_evidence_screen.feats (elong, nsl, sph) + thickness (2 x max EDT)"""
    V = cb.sum() * np.prod(sp)
    faces = sum(float((np.diff(np.pad(cb, [(1, 1) if k == j else (0, 0) for k in range(3)]).astype(np.int8), axis=j) != 0).sum())
                * np.prod([sp[k] for k in range(3) if k != j]) for j in range(3))
    xyz = np.argwhere(cb)
    if len(xyz) > 200000: xyz = xyz[::len(xyz) // 200000 + 1]
    xyz = xyz * np.asarray(sp)
    el = np.nan
    if len(xyz) >= 4:
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T))); el = float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))
    ax = int(np.argmax(sp))
    return dict(ml=round(float(cb.sum() * np.prod(sp) / 1000), 4), elong=el,
                nsl=int(np.any(cb, axis=tuple(k for k in range(3) if k != ax)).sum()),
                sph=float((36 * np.pi * V ** 2) ** (1 / 3) / max(faces, 1e-6)),
                thick_mm=round(float(2 * ndimage.distance_transform_edt(np.pad(cb, 1), sampling=sp).max()), 2), maxsp=max(sp))


def sp_class(m):
    return 'thin' if m <= 2.5 else ('mid' if m <= 5.0 else 'thick')


def category_of_real(r):
    if r.dom in ('c_supply_drain', 'c_vessel_vessel'): return 'vessel'
    if r.dom in ('c_adjacent', 'c_vessel_organ') or str(r.R) == 'True': return 'rim'
    if r.dom == 'outside': return 'outside'
    if r.dom == 'core':
        return CAT_OF_KB.get(r.host_cat, 'other') if r.host_cat in CAT_OF_KB else 'other'
    return 'other'


# ------------------------------------------------------------------ operators (local boxes; m bool, sp spacing)
def comps(m, vml, rng):
    lb, n = ndimage.label(m, structure=EX.STRUCT26)
    if n == 0: return None
    sz = np.bincount(lb.ravel())[1:] * vml
    ok = np.flatnonzero(sz >= MIN_ML) + 1
    if len(ok) == 0: return None
    return lb == rng.choice(ok)


def op_partial(m, sp, rng, ctx):
    idx = np.argwhere(m); c = idx[rng.integers(len(idx))]
    f = rng.uniform(0.2, 1.0); target = max(1, int(f * len(idx)))
    g = np.zeros_like(m); g[tuple(c)] = True; st = ndimage.generate_binary_structure(3, 1)
    for _ in range(400):
        if g.sum() >= target: break
        g2 = ndimage.binary_dilation(g, st) & m
        if g2.sum() == g.sum(): break
        g = g2
    return g


def op_fragment(m, sp, rng, ctx):
    f = ndimage.gaussian_filter(rng.standard_normal(m.shape).astype(np.float32), 1.5)
    q = np.quantile(f[m], rng.uniform(0.2, 0.6))
    return m & (f > q)


def op_slices(m, sp, rng, ctx):
    ax = int(np.argmax(sp)); sl = np.flatnonzero(np.any(m, axis=tuple(k for k in range(3) if k != ax)))
    keep = np.zeros(m.shape[ax], bool)
    if rng.random() < 0.5:
        L = int(rng.integers(1, len(sl) + 1)); s0 = int(rng.integers(0, len(sl) - L + 1)); keep[sl[s0:s0 + L]] = True
    else:
        p = rng.uniform(0.1, 0.5); keep[sl[rng.random(len(sl)) > p]] = True
    shape = [1, 1, 1]; shape[ax] = m.shape[ax]
    return m & keep.reshape(shape)


def op_boundary(m, sp, rng, ctx):
    ax = int(np.argmax(sp)); st = np.zeros((3, 3, 3), bool); c = [1, 1, 1]
    for d in range(3):
        if d == ax: continue
        for s in (0, 2):
            q = list(c); q[d] = s; st[tuple(q)] = True
    st[1, 1, 1] = True
    k = int(rng.integers(1, 3))
    return ndimage.binary_dilation(m, st, iterations=k) if rng.random() < 0.5 else ndimage.binary_erosion(m, st, iterations=k)


def op_leak(m, sp, rng, ctx):
    nb = ctx.get('neigh')
    if nb is None or not nb.any(): return m
    nb = nb & ~m
    d = ndimage.distance_transform_edt(~m, sampling=sp)
    lim = 0.5 * m.sum(); out = m
    for r in (1.0, 2.0, 4.0, 6.0, 8.0):
        add = nb & (d <= r * max(sp))
        if add.sum() > lim: break
        out = m | add
    return out


def op_resolution(m, sp, rng, ctx):
    ax = int(np.argmax(sp)); t = float(rng.choice([3.0, 5.0, 7.5]))
    k = int(round(t / sp[ax]))
    if k < 2: return m
    n = m.shape[ax]; out = np.zeros_like(m)
    for s in range(0, n, k):
        sl = [slice(None)] * 3; sl[ax] = slice(s, min(n, s + k)); blk = m[tuple(sl)]
        maj = blk.sum(axis=ax) >= (blk.shape[ax] / 2.0)
        out[tuple(sl)] = np.expand_dims(maj, ax)
    return out


OPS = {'partial': op_partial, 'fragment': op_fragment, 'slices': op_slices, 'boundary': op_boundary, 'leak': op_leak,
       'resolution': op_resolution}


def perturb(m, sp, vml, rng, ctx):
    if rng.random() < 0.2: return comps(m, vml, rng), 'identity'
    names = list(rng.choice(list(OPS), size=int(rng.integers(1, 4)), replace=False))
    for nm in names:
        if not m.any(): return None, '+'.join(names)
        m = OPS[nm](m, sp, rng, ctx)
    return comps(m, vml, rng) if m.any() else None, '+'.join(names)


# ------------------------------------------------------------------ per scan
def window(sl_list, pad, shape):
    return tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(sl_list, pad, shape))


def one(kb, ds, case, design, pool, a):
    img = f'{ds}/{case}'; js = os.path.join(ANAT, img + '.json')
    if not os.path.exists(js): return [dict(ds=ds, case=case, note='no anatomy file')]
    meta = json.load(open(js)); z = np.load(os.path.join(ANAT, img + '.npz')); off = meta.get('offset') or [0, 0, 0]
    mod = meta.get('modality')
    rng = np.random.default_rng(zlib.crc32(img.encode()))
    bp = os.path.join(S, 'BiomedParse', ds, case + '_tumor.nii.gz'); vp = os.path.join(S, 'VoxTell', ds, case + '_tumor.nii.gz')
    lp = os.path.join(R, ds, 'labels', case + '_seg.nii.gz')
    bimg = nib.load(bp); sp = tuple(float(x) for x in bimg.header.get_zooms()[:3]); vml = float(np.prod(sp)) / 1000
    B = np.asanyarray(bimg.dataobj) > 0; V = np.asanyarray(nib.load(vp).dataobj) > 0
    T = np.isin(EX._read_labels(lp), EX.APPEARANCE[ds]['tumour']) if os.path.exists(lp) else np.zeros(B.shape, bool)
    shape = B.shape
    if not (B.shape == V.shape == T.shape == tuple(meta['shape'])): return [dict(ds=ds, case=case, note='grid mismatch')]
    base = dict(ds=ds, case=case, mod=mod, maxsp=max(sp))
    rows = []
    if a.stage in ('real', 'real_tf'):
        return real_rows(ds, case, design, B, V, sp, vml, base) or [dict(ds=ds, case=case, note='no real candidates')]
    return aug_rows(kb, ds, case, meta, z, off, B, T, sp, vml, shape, base, pool, rng, a) or [dict(ds=ds, case=case, note='no rows')]


def real_rows(ds, case, design, B, V, sp, vml, base):
    """real design-set candidates (three modes); features recomputed; truth / scope / category from design.csv"""
    rows = []
    dz = design.get((ds, case))
    for mode, M in (('BP', B), ('VT', V), ('BP+VT', B | V)):
        lb, n = ndimage.label(M, structure=EX.STRUCT26)
        for j, ob in enumerate(ndimage.find_objects(lb), start=1):
            if ob is None: continue
            cb = lb[ob] == j
            if cb.sum() * vml < MIN_ML: continue
            d = dz.get((mode, j)) if dz is not None else None
            if d is None: continue
            rows.append(dict(base, source='real', mode=mode, cid=j, category=d['category'], seed='', ops='', lab=d['lab'],
                             in_scope=d['in_scope'], **feats(cb, sp)))
    return rows


def aug_rows(kb, ds, case, meta, z, off, B, T, sp, vml, shape, base, pool, rng, a):
    rows = []
    # ---- truth components and the 20 mm buffer (on the tumour's box only)
    lbT, nT = ndimage.label(T, structure=EX.STRUCT26); tsz = np.bincount(lbT.ravel(), minlength=nT + 1)
    near = None; nbox = None
    if T.any():
        tb = ndimage.find_objects(T.astype(np.uint8))[0]
        nbox = window(tb, [int(np.ceil(BUFFER_MM / s)) + 1 for s in sp], shape)
        near = ndimage.distance_transform_edt(~T[nbox], sampling=sp) <= BUFFER_MM

    def near_win(w):
        """the 20 mm tumour buffer on window w (all False without an expert tumour)"""
        out = np.zeros(tuple(s.stop - s.start for s in w), bool)
        if near is None: return out
        lo = [max(a_.start, b_.start) for a_, b_ in zip(w, nbox)]; hi = [min(a_.stop, b_.stop) for a_, b_ in zip(w, nbox)]
        if any(l >= h for l, h in zip(lo, hi)): return out
        out[tuple(slice(l - a_.start, h - a_.start) for l, h, a_ in zip(lo, hi, w))] = \
            near[tuple(slice(l - b_.start, h - b_.start) for l, h, b_ in zip(lo, hi, nbox))]
        return out

    def truth(w, m):
        inter = int((T[w] & m).sum()); nc = int(m.sum())
        if inter == 0: return 'F'
        ids = np.unique(lbT[w][m]); ids = ids[ids > 0]; gs = int(tsz[ids].sum())
        return 'T' if (inter / nc >= .5 or 2 * inter / (nc + gs) >= .1) else 'P'

    # ---- anatomy layer: usable whole organs (no lymph stations), as Bx in full-grid coordinates
    full = tuple(slice(0, n) for n in shape)
    W_core, W_terr, W_cat = {}, {}, {}
    for o in meta['organs']:
        if o['mu'] <= 0 or any(f.split(':')[0].split(' ')[0] in EXCL_FLAGS for f in o['flags']): continue
        wk = G3.whole_of(kb, o['entity']) + ('_' + o['side'] if o['side'] else '')
        cat = kb.entities.get(G3.whole_of(kb, o['entity']), {}).get('category')
        if cat == 'lymph_station': continue
        c = G3.unpack_bx(z, f"o{o['i']}_core", off, full); t = G3.unpack_bx(z, f"o{o['i']}_terr", off, full)
        if c is not None: W_core[wk] = (W_core[wk] | c) if wk in W_core else c
        if t is not None: W_terr[wk] = (W_terr[wk] | t) if wk in W_terr else t
        W_cat[wk] = cat

    def neighbours(w, own=None, avoid_tumour=False):
        nb = np.zeros(tuple(s.stop - s.start for s in w), bool)
        for k, c in W_core.items():
            if k != own: nb |= c.win(w)
        if avoid_tumour: nb &= ~T[w]
        return nb

    # ---- tumour side
    for t_id in range(1, nT + 1):
        if tsz[t_id] * vml < MIN_ML: continue
        ob = ndimage.find_objects((lbT == t_id).astype(np.uint8))[0]
        w = window(ob, [12, 12, 12], shape); seed = lbT[w] == t_id; ctx = dict(neigh=neighbours(w))
        for v in range(a.n_tum + 1):
            if v == 0: m, ops = seed.copy(), 'seed'
            else: m, ops = perturb(seed.copy(), sp, vml, rng, ctx)
            if m is None: continue
            lab = truth(w, m)
            if lab == 'P': continue
            rows.append(dict(base, source='aug', mode='', cid='', category='tumour', seed=f'T{t_id}', ops=ops, lab=lab, in_scope=True,
                             **feats(m, sp)))
    # ---- non-tumour side: seeds conditioned on a real false-candidate exemplar (category, volume, slab thickness)
    ax = int(np.argmax(sp))
    regions = {'solid': [], 'hollow': [], 'vessel': [], 'outside': [], 'rim': []}
    for k, c in W_core.items():
        cat = CAT_OF_KB.get(W_cat.get(k))
        if cat and cat != 'outside': regions[cat].append(k)
    regions['rim'] = list(W_terr); regions['outside'] = list(W_terr)   # 'outside': just beyond an organ territory (no organ)
    spc = sp_class(max(sp))
    cats = [c for c in pool['cat_p'] if regions.get(c) and len(pool['ex'].get((c, spc), [])) > 0]
    if cats:
        p = np.array([pool['cat_p'][c] for c in cats]); p = p / p.sum()
        for s_i in range(a.n_neg):
            cat = str(rng.choice(cats, p=p)); org = regions[cat][int(rng.integers(len(regions[cat])))]
            ex = pool['ex'][(cat, spc)]; ml_ex, nsl_ex, sp_ex = ex[int(rng.integers(len(ex)))]
            nsl_h = max(1, int(round(nsl_ex * sp_ex / sp[ax])))          # same physical slab thickness on this scan's slices
            if cat in ('rim', 'outside'):
                t = W_terr[org]; reg = t.m & ~ndimage.binary_erosion(t.m, iterations=2, border_value=1); rb = t.sl
            else:
                c = W_core[org]; reg = c.m; rb = c.sl
            cand = np.argwhere(reg)
            if len(cand) == 0: continue
            ctr = cand[int(rng.integers(len(cand)))] + np.array([s_.start for s_ in rb])
            asp = 1.0 if rng.random() < 0.5 else float(rng.uniform(1, 6)); th = rng.uniform(0, np.pi)
            ip = [d for d in range(3) if d != ax]

            def build(scale):
                """slab of nsl_h slices with in-plane ellipses of area scale x (exemplar volume / slab thickness), cut to the region"""
                area = scale * ml_ex * 1000.0 / (nsl_h * sp[ax])
                r1, r2 = np.sqrt(area / np.pi * asp), np.sqrt(area / np.pi / asp)
                r1, r2 = min(r1, 150.0), min(r2, 150.0)
                ext = [0, 0, 0]; ext[ax] = nsl_h // 2 + 1
                for d in ip: ext[d] = int(np.ceil(r1 / sp[d])) + 2
                w = tuple(slice(max(0, int(c_) - e), min(n, int(c_) + e + 1)) for c_, e, n in zip(ctr, ext, shape))
                G = np.meshgrid(*[(np.arange(s_.start, s_.stop) - c_) * q for s_, c_, q in zip(w, ctr, sp)], indexing='ij')
                u = np.cos(th) * G[ip[0]] + np.sin(th) * G[ip[1]]; v_ = -np.sin(th) * G[ip[0]] + np.cos(th) * G[ip[1]]
                k0 = (np.arange(w[ax].start, w[ax].stop) - ctr[ax]); lo_s = -(nsl_h // 2); keep = (k0 >= lo_s) & (k0 < lo_s + nsl_h)
                sh = [1, 1, 1]; sh[ax] = len(keep)
                ell = ((u / r1) ** 2 + (v_ / r2) ** 2 <= 1.0) & keep.reshape(sh)
                nw = near_win(w)
                if cat == 'outside':                                     # no organ: outside every usable territory
                    anyt = np.zeros(ell.shape, bool)
                    for t_ in W_terr.values(): anyt |= t_.win(w)
                    m = ell & ~anyt & ~nw
                else:
                    regw = np.zeros(ell.shape, bool)
                    lo = [max(a_.start, b_.start) for a_, b_ in zip(w, rb)]; hi = [min(a_.stop, b_.stop) for a_, b_ in zip(w, rb)]
                    if any(l >= h for l, h in zip(lo, hi)): return None, w, nw
                    regw[tuple(slice(l - a_.start, h - a_.start) for l, h, a_ in zip(lo, hi, w))] = \
                        reg[tuple(slice(l - b_.start, h - b_.start) for l, h, b_ in zip(lo, hi, rb))]
                    m = ell & regw & ~nw                                 # never within 20 mm of an expert tumour
                if not m.any(): return None, w, nw
                lb, _ = ndimage.label(m, structure=EX.STRUCT26)
                ci = tuple(np.array(ctr) - np.array([s_.start for s_ in w]))
                cl = lb[ci] if lb[ci] > 0 else (np.bincount(lb.ravel())[1:].argmax() + 1)
                return lb == cl, w, nw

            # volume-preserving (smoke test 28 Nov): cutting to a thin region shrank seeds below the exemplar volume, which biased
            # slab thickness and category within size bands; the in-plane extent is enlarged (slab thickness fixed) up to 4 times
            scale = 1.0
            for _ in range(5):
                seedm, w, nw = build(scale)
                if seedm is None: break
                vol = seedm.sum() * vml
                if vol >= 0.7 * ml_ex or scale >= 64: break
                scale *= min(4.0, ml_ex / max(vol, 1e-6))
            if seedm is None or seedm.sum() * vml < MIN_ML: continue
            ctx = dict(neigh=neighbours(w, own=org, avoid_tumour=True) & ~nw)
            for v in range(2):
                if v == 0: mm, ops = seedm.copy(), 'seed'
                else: mm, ops = perturb(seedm.copy(), sp, vml, rng, ctx)
                if mm is None: continue
                if (nw & mm).any() or (T[w] & mm).any(): continue      # entered the 20 mm buffer: discarded
                rows.append(dict(base, source='aug', mode='', cid='', category=cat, seed=f'N{s_i}:{org}', ops=ops, lab='F',
                                 in_scope=True, **feats(mm, sp)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['real', 'real_tf', 'aug'], required=True)
    ap.add_argument('--only', default=''); ap.add_argument('--budget', type=float, default=1e9); ap.add_argument('--tag', default='')
    ap.add_argument('--n_tum', type=int, default=4); ap.add_argument('--n_neg', type=int, default=12)
    a = ap.parse_args()
    J = pd.concat([pd.read_csv(f) for f in JOBS]).drop_duplicates(['ds', 'case']); J['case'] = J.case.astype(str)
    tfj = J.tumour_free.astype(str) == 'True'
    J = J[tfj] if a.stage == 'real_tf' else J[~tfj]                 # tumour-free scans: real_tf only (evaluation features)
    if a.only: J = J[(J.ds + '/' + J.case).isin(a.only.split(','))]
    od = os.path.join(OUTD, a.tag) if a.tag else OUTD; os.makedirs(od, exist_ok=True)
    outf = os.path.join(od, a.stage + '.csv')
    design, pool = {}, None
    if a.stage in ('real', 'real_tf'):
        D = pd.read_csv(DESIGN, low_memory=False); D = D[D.note.isna() | (D.note == '')].copy(); D['case'] = D.case.astype(str)
        D = D[(D.tumour_free.astype(str) == 'True') == (a.stage == 'real_tf')]
        D['category'] = [category_of_real(r) for r in D.itertuples()]
        for r in D.itertuples():
            design.setdefault((r.ds, r.case), {})[(r.mode, int(r.cid))] = dict(lab=r.lab, in_scope=str(r.in_scope) == 'True', category=r.category)
    else:
        rf = os.path.join(OUTD, 'real.csv')                          # exemplar pool: ALL design-set real false candidates
        Rr = pd.read_csv(rf, low_memory=False); Rr = Rr[Rr.note.isna() | (Rr.note == '')]
        Fr = Rr[(Rr.lab == 'F') & (Rr.in_scope.astype(str) == 'True') & Rr.category.isin(['solid', 'hollow', 'vessel', 'rim', 'outside'])]
        pool = dict(cat_p=Fr.category.value_counts(normalize=True).to_dict(), ex={})
        for (c, k), g in Fr.assign(spc=Fr.maxsp.map(sp_class)).groupby(['category', 'spc']):
            pool['ex'][(c, k)] = list(zip(g.ml, g.nsl, g.maxsp))
        json.dump(dict(cat_p=pool['cat_p'], n={f'{c}|{k}': len(v) for (c, k), v in pool['ex'].items()}, real_scans=int(Rr[['ds', 'case']].drop_duplicates().shape[0])),
                  open(os.path.join(od, 'negative_exemplar_pool.json'), 'w'), indent=1)
    done = set()
    if os.path.exists(outf):
        d = pd.read_csv(outf, usecols=['ds', 'case']); done = set(zip(d.ds, d.case.astype(str)))
    kb = KnowledgeBase.load(f'{KBD}/knowledge_base')
    t0, n = time.time(), 0
    for ds, case in J[['ds', 'case']].itertuples(index=False):
        if (ds, case) in done: continue
        if time.time() - t0 > a.budget: break
        try:
            rows = one(kb, ds, case, design, pool, a)
        except Exception as e:
            import traceback
            rows = [dict(ds=ds, case=case, note='error ' + repr(e)[:120] + ' ' + traceback.format_exc().splitlines()[-2][:80])]
        new = not os.path.exists(outf)
        with open(outf, 'a', newline='') as fh:
            w = csv.DictWriter(fh, COLS, extrasaction='ignore'); new and w.writeheader()
            for r in rows: w.writerow({k: r.get(k, '') for k in COLS})
        n += 1
    left = sum((d, c) not in done for d, c in zip(J.ds, J.case)) - n
    print(f'{a.stage}: processed {n}; remaining {left}; {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
