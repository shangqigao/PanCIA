"""Graph pruning v3 (27 Nov 2026). Rule and gate: claude/graph_pruning_v3_gate_2026-11-27.md (user-approved before any v3 run);
plan and checklist: claude/graph_pruning_v3_plan_2026-11-27.md; workflow: claude/stage1_workflow.md.

Workflow steps consumed: 1a (raw BiomedParse / VoxTell tumour masks -> candidates), 2g g2 (anatomy files
PanCIA_outputs/anatomy/r2seg/<coll>/<stem>.{npz,json}; no raw TS / VT organ file), the image (HU, CT, for the wall clause), and 0
(expert labels, cohort tumour-free file; EVALUATION ONLY: truth and the scope = primary TS organ + 10 mm as in the evidence screen).
Replaces steps 4-6 for pruning only. No sex metadata, no cancer type in any rule quantity, no TS / VT preference, no agreement.

Graph per scan and mode (BP, VT, BP+VT; candidates = 26-connected components >= 0.05 ml of the mode's raw mask, built identically):
  organs  = step-2g organs with mu > 0, no 'small' / 'spatial' / 'not_between' flag (set-aside masks are absent from g2), merged into
            WHOLE organs (an entity's KB has_part ancestors; side kept): core = union of member cores, territory = union of members'
            territories. Sub-structures never explain on their own (user decision 1).
  regions = whole-organ cores; contested regions (U_a & U_b of the step-2g pairs) by the KB relation of the two entities:
            supply_drain / vessel_vessel / same_entity (-> counted as that organ's core) / adjacent / vessel_organ / organ_organ /
            other_edge; excluded organ cores; outside every organ territory.
Member labels (in order): U-size (< 1 ml); U-region (dominant region outside / excluded / organ_organ / other_edge); V, W or R
explanation; else unexplained.
  V vessel: >= 50% of the candidate in vessel whole cores (KB vessel_artery / vessel_vein) or supply_drain / vessel_vessel contested
    regions; clause: candidate thickness (2 x max EDT, mm) <= q90 of the same vessel's thickness (2 x EDT inside its core) outside the
    candidate box +- 10 mm (>= 50 reference voxels).
  W wall (CT only): >= 50% in one hollow-organ whole core (KB hollow_organ); clause: candidate soft-tissue (-20..150 HU) thickness <= q90
    of the same organ's soft-tissue wall thickness outside the candidate box +- 10 mm (>= 50 reference voxels). As v2 E2.
  R rim: >= 50% of the candidate within max(2 mm, voxel diagonal) of a usable whole-organ territory boundary, or >= 50% in adjacent /
    vessel_organ contested regions; clause: thickness <= 2 x largest spacing AND protrusion beyond the host whole organ's territory
    <= largest spacing (host = the whole organ holding most of the candidate's territory overlap). As v2 E3.
    Design fix 1 (27 Nov, design set only, before the gate): R is not applied when the host is a hollow organ (walls only via W).
    Design fix 1 (27 Nov, design set only, before the gate): R is not applied when the host is a hollow organ; a hollow-organ wall is
    explained only by W, which carries the organ-relative wall-thickness protection (a colon tumour 7.7 ml, 15 mm thick, was pruned
    by R on FLARE23_0142 at 7.5 mm slices, where 'thickness <= 2 x largest spacing' allows 15 mm).
Step-2g g3 node (28 Nov, user-approved plan claude/elongation_and_pelvic_fix_plan_2026-11-28.md, B2): an unresolved sex-class dispute
is kept by 2g as one neutral node 'pelvic_reproductive_unresolved' (npz u0_core). Here it is a usable whole organ for LOCATION only
(host_cat 'unresolved'): it can host candidates (core region, member 'unexplained' unless V applies), it never gives an R explanation
(its boundary is not one organ's boundary; nor W). Its overlap with another usable whole organ's territory is a contested region whose
category is the KB relation shared by all of the node's member whole organs with that organ (else other_edge); the overlap is removed
from both cores, as 2g does for contested pairs. Scans without the node are unchanged.
Fragment groups: candidates within tau = max(5 mm, largest spacing) sharing the dominant whole organ (or both outside organs).
Group decision: pruned if every member is V/W/R; uncertain if any member is U-* or members are partly explained; kept otherwise.

Output: PanCIA_outputs/ledger/graph_v3/<tag>.csv (one row per candidate x mode; resumable). --jobs is REQUIRED: design-set files
(jobs.csv, jobs_v2.csv) for debugging; jobs_v3.csv only after the gate is approved.
Usage: python s1_graph_v3.py --jobs jobs.csv --tag design_debug [--only ds/case,...] [--budget 1e9]"""
import os, sys, json, re, time, argparse, csv
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from pancia_kb.stage2 import paths, expert as EX     # noqa: E402
from pancia_kb import KnowledgeBase                    # noqa: E402
import s1_evidence_screen as SC                        # noqa: E402  (PRIMARY, MIN_ML: evaluation scope and candidate floor only)

R = paths.root('R2Seg'); S = paths.root('R2Seg_Seg')
ANAT = os.path.join(paths.root('outputs'), 'anatomy', 'r2seg')
OUTD = os.path.join(paths.ledger_dir(), 'graph_v3')
KBD = os.path.abspath(os.path.join(HERE, '..', '..'))
EXCL_FLAGS = ('small', 'spatial', 'not_between')
VES = ('vessel_artery', 'vessel_vein'); HOLLOW = ('hollow_organ',)
SIZE_FLOOR_ML = 1.0
U_REGIONS = ('outside', 'excl', 'c_organ_organ', 'c_other_edge')
COLS = ['ds', 'case', 'site', 'tumour_free', 'mod', 'mode', 'cid', 'ml', 'lab', 'in_scope', 'in_frac', 'dice_t', 'dom', 'dom_share',
        'host', 'host_cat', 'thick_mm', 'member', 'V', 'W', 'R', 'group', 'group_n', 'decision', 'set_aside', 'note']


def category(kb, a, b):
    if a == b: return 'same_entity'
    r = {t['type'] for t in kb.out_edges.get(a, []) if t['dst'] == b} | {t['type'] for t in kb.out_edges.get(b, []) if t['dst'] == a}
    va = kb.entities.get(a, {}).get('category') in VES; vb = kb.entities.get(b, {}).get('category') in VES
    if r & {'supplied_by', 'drained_by', 'drains_into', 'supplies', 'drains'}: return 'supply_drain'
    if 'adjacent_to' in r: return 'adjacent'
    if r: return 'other_edge'
    if va and vb: return 'vessel_vessel'
    if va or vb: return 'vessel_organ'
    return 'organ_organ'


def whole_of(kb, e, _c={}):
    """top KB has_part ancestor of entity e (e itself if none)"""
    if e in _c: return _c[e]
    cur, seen = e, set()
    while True:
        par = [r['src'] for r in kb.in_edges.get(cur, []) if r['type'] == 'has_part']
        if not par or par[0] in seen: break
        seen.add(cur); cur = par[0]
    _c[e] = cur
    return cur


def box_of(z, key, off):
    if key + '_sl' not in z: return None
    return [(int(a) + o, int(b) + o) for (a, b), o in zip(z[key + '_sl'], off)]


class Bx:
    """mask stored on its own bounding box inside the crop (memory: whole-body CT has ~100 organs; full-crop arrays exceed RAM)"""
    def __init__(self, sl, m, shp):
        self.sl, self.m, self.shp = sl, m, shp

    @staticmethod
    def of(full, shp):
        if full is None or not full.any(): return None
        ob = ndimage.find_objects(full.astype(np.uint8))[0]
        return Bx(ob, full[ob].copy(), shp)

    def win(self, w):
        """this mask on window w (tuple of slices in crop coordinates)"""
        out = np.zeros(tuple(s.stop - s.start for s in w), bool)
        lo = [max(a.start, b.start) for a, b in zip(self.sl, w)]; hi = [min(a.stop, b.stop) for a, b in zip(self.sl, w)]
        if all(l < h for l, h in zip(lo, hi)):
            out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, w))] = \
                self.m[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, self.sl))]
        return out

    def full(self):
        out = np.zeros(self.shp, bool); out[self.sl] = self.m; return out

    def __or__(self, o):
        if o is None: return self
        lo = [min(a.start, b.start) for a, b in zip(self.sl, o.sl)]; hi = [max(a.stop, b.stop) for a, b in zip(self.sl, o.sl)]
        w = tuple(slice(l, h) for l, h in zip(lo, hi))
        return Bx(w, self.win(w) | o.win(w), self.shp)


def unpack_bx(z, key, off, crop):
    """step-2g mask `key` as a Bx in crop coordinates (None if absent / disjoint), without a full-crop array"""
    if key not in z: return None
    sl = box_of(z, key, off); shp_ = tuple(int(x) for x in z[key + '_shape'])
    lo = [max(a, c.start) for (a, _), c in zip(sl, crop)]; hi = [min(b, c.stop) for (_, b), c in zip(sl, crop)]
    if any(l >= h for l, h in zip(lo, hi)): return None
    m = np.unpackbits(z[key])[:int(np.prod(shp_))].reshape(shp_).astype(bool)
    sub = m[tuple(slice(l - a, h - a) for l, h, (a, _) in zip(lo, hi, sl))]
    if not sub.any(): return None
    return Bx(tuple(slice(l - c.start, h - c.start) for l, h, c in zip(lo, hi, crop)), sub, tuple(c.stop - c.start for c in crop))


def unpack(z, key, off, crop):
    if key not in z: return None
    sl = box_of(z, key, off); shp = tuple(int(x) for x in z[key + '_shape'])
    lo = [max(a, c.start) for (a, _), c in zip(sl, crop)]; hi = [min(b, c.stop) for (_, b), c in zip(sl, crop)]
    if any(l >= h for l, h in zip(lo, hi)): return None
    m = np.unpackbits(z[key])[:int(np.prod(shp))].reshape(shp).astype(bool)
    out = np.zeros(tuple(c.stop - c.start for c in crop), bool)
    out[tuple(slice(l - c.start, h - c.start) for l, h, c in zip(lo, hi, crop))] = m[tuple(slice(l - a, h - a) for l, h, (a, _) in zip(lo, hi, sl))]
    return out if out.any() else None


def thick(m, sp):
    return float(2 * ndimage.distance_transform_edt(np.pad(m, 1), sampling=sp).max()) if m.any() else 0.0


def ref_q90(mask, sp, excl_box, hu=None):
    """q90 of 2 x EDT inside `mask` (optionally restricted to soft tissue `hu`), outside excl_box; None if < 50 reference voxels"""
    m = mask if hu is None else (mask & hu)
    ob = ndimage.find_objects(m.astype(np.uint8))
    if not ob or ob[0] is None: return None
    ob = ob[0]; t = 2 * ndimage.distance_transform_edt(m[ob], sampling=sp)
    keep = m[ob].copy()
    lo = [max(e.start - o.start, 0) for e, o in zip(excl_box, ob)]; hi = [min(e.stop - o.start, o.stop - o.start) for e, o in zip(excl_box, ob)]
    if all(l < h for l, h in zip(lo, hi)):
        keep[tuple(slice(l, h) for l, h in zip(lo, hi))] = False
    v = t[keep]
    return float(np.percentile(v, 90)) if v.size >= 50 else None


def one(kb, ds, case, tf):
    site = EX.SITE_ENTITY.get(ds, ''); base = dict(ds=ds, case=case, site=site, tumour_free=tf)
    img = f'{ds}/{case}'; js = os.path.join(ANAT, img + '.json')
    if not os.path.exists(js): return [dict(base, note='no anatomy file')]
    meta = json.load(open(js)); z = np.load(os.path.join(ANAT, img + '.npz')); off = meta.get('offset') or [0, 0, 0]
    mod = meta.get('modality') or EX.APP_MOD.get(ds, 'CT'); base['mod'] = mod
    base['set_aside'] = '|'.join((meta.get('consistency') or {}).get('dropped_classes') or [])
    bp = os.path.join(S, 'BiomedParse', ds, case + '_tumor.nii.gz'); vp = os.path.join(S, 'VoxTell', ds, case + '_tumor.nii.gz')
    tp = os.path.join(S, 'TotalSegmentator', ds, case + '_organ.nii.gz'); lp = os.path.join(R, ds, 'labels', case + '_seg.nii.gz')
    ip = os.path.join(R, ds, 'images', case + '.nii.gz')
    bimg = nib.load(bp); sp = tuple(float(x) for x in bimg.header.get_zooms()[:3]); vml = np.prod(sp) / 1000
    B = np.asanyarray(bimg.dataobj) > 0; V = np.asanyarray(nib.load(vp).dataobj) > 0
    if not (B.shape == V.shape == tuple(meta['shape'])): return [dict(base, note='grid mismatch')]
    U = B | V
    if not U.any(): return [dict(base, note='no candidates')]
    bb = ndimage.find_objects(U.astype(np.uint8))[0]; del U
    mg = [int(np.ceil(15 / s)) for s in sp]
    c0 = [(max(0, b.start - k), min(n, b.stop + k)) for b, k, n in zip(bb, mg, B.shape)]
    # usable organs -> whole organs; crop = candidate box U boxes of usable organs that intersect it (references need the whole organ)
    usable, excluded = [], []
    for o in meta['organs']:
        if o['mu'] <= 0: continue
        bad = any(f.split(':')[0].split(' ')[0] in EXCL_FLAGS for f in o['flags'])
        (excluded if bad else usable).append(o)
    lo, hi = [a for a, _ in c0], [b for _, b in c0]
    for o in usable:
        bx = box_of(z, f"o{o['i']}_terr", off)
        if bx and all(a < h and b > l for (a, b), l, h in zip(bx, [x for x, _ in c0], [y for _, y in c0])):
            lo = [min(l, a) for l, (a, _) in zip(lo, bx)]; hi = [max(h, b) for h, (_, b) in zip(hi, bx)]
    NODE = 'pelvic_reproductive_unresolved'
    nbx = box_of(z, 'u0_core', off)                                   # g3 node (absent in g2 files and undisputed scans)
    if nbx and all(a < h and b > l for (a, b), l, h in zip(nbx, [x for x, _ in c0], [y for _, y in c0])):
        lo = [min(l, a) for l, (a, _) in zip(lo, nbx)]; hi = [max(h, b) for h, (_, b) in zip(hi, nbx)]
    crop = tuple(slice(max(0, l - 2), min(n, h + 2)) for l, h, n in zip(lo, hi, B.shape))
    B, V = B[crop], V[crop]
    T = (np.isin(EX._read_labels(lp), EX.APPEARANCE[ds]['tumour']) if os.path.exists(lp) else np.zeros(meta['shape'], bool))[crop]
    I = None
    if mod == 'CT' and os.path.exists(ip):
        Xf = np.asanyarray(nib.load(ip).dataobj); Xf = Xf[..., 0] if Xf.ndim == 4 else Xf
        I = np.asarray(Xf[crop], dtype=np.float32); del Xf
    names = json.load(open(tp.replace('_organ.nii.gz', '_organ_labels.json')))['labels']   # evaluation scope only
    pid = [int(k) for k, v in names.items() if v in SC.PRIMARY.get(site, [])]
    P = None
    if pid:
        org = np.asanyarray(nib.load(tp).dataobj)[crop]; P0 = np.isin(org, pid); del org
        if P0.any(): P = ndimage.distance_transform_edt(~P0, sampling=sp) <= 10
    shp = B.shape
    # whole organs (Bx boxes, not full-crop arrays)
    W_core, W_terr, W_cat, W_ent = {}, {}, {}, {}
    hk = {}
    for o in usable:
        wk = whole_of(kb, o['entity']) + ('_' + o['side'] if o['side'] else '')
        hk[o['entity'] + ('_' + o['side'] if o['side'] else '')] = (o, wk)
        c = unpack_bx(z, f"o{o['i']}_core", off, crop); t = unpack_bx(z, f"o{o['i']}_terr", off, crop)
        if c is not None: W_core[wk] = (W_core[wk] | c) if wk in W_core else c
        if t is not None: W_terr[wk] = (W_terr[wk] | t) if wk in W_terr else t
        W_cat[wk] = kb.entities.get(whole_of(kb, o['entity']), {}).get('category'); W_ent[wk] = whole_of(kb, o['entity'])
    node = unpack_bx(z, 'u0_core', off, crop)
    if node is not None:
        W_terr[NODE] = node; W_cat[NODE] = 'unresolved'
        node_wholes = sorted({whole_of(kb, e) for e in ((meta.get('unresolved') or {}).get('entities') or [])})
    EXc = np.zeros(shp, bool); anyterr = np.zeros(shp, bool)
    for o in excluded:
        c = unpack_bx(z, f"o{o['i']}_core", off, crop); t = unpack_bx(z, f"o{o['i']}_terr", off, crop)
        if c is not None: EXc[c.sl] |= c.m
        if t is not None: anyterr[t.sl] |= t.m
    for t in W_terr.values(): anyterr[t.sl] |= t.m
    # contested regions by category (pairs among usable organs); same_entity / same whole organ -> that whole organ's core
    C = {k: np.zeros(shp, bool) for k in ('supply_drain', 'vessel_vessel', 'adjacent', 'vessel_organ', 'organ_organ', 'other_edge')}
    Cves = {}                                                         # vessel whole organ -> its supply / vessel-vessel contested voxels (Bx)
    tcache = {}
    def terr_of(name):
        if name not in tcache:
            o = hk[name][0]; tcache[name] = unpack_bx(z, f"o{o['i']}_terr", off, crop)
        return tcache[name]
    for p in meta['pairs']:
        if p['a'] not in hk or p['b'] not in hk: continue
        ta, tb = terr_of(p['a']), terr_of(p['b'])
        if ta is None or tb is None: continue
        lo_ = [max(a.start, b.start) for a, b in zip(ta.sl, tb.sl)]; hi_ = [min(a.stop, b.stop) for a, b in zip(ta.sl, tb.sl)]
        if any(l >= h for l, h in zip(lo_, hi_)): continue
        w_ = tuple(slice(l, h) for l, h in zip(lo_, hi_)); mm = ta.win(w_) & tb.win(w_)
        if not mm.any(): continue
        m = Bx(w_, mm, shp)
        (oa, wa), (ob_, wb) = hk[p['a']], hk[p['b']]
        cat = category(kb, oa['entity'], ob_['entity'])
        if cat == 'same_entity' or wa == wb:
            W_core[wa] = (W_core[wa] | m) if wa in W_core else m; continue
        C[cat][w_] |= mm
        if cat in ('supply_drain', 'vessel_vessel'):
            for w in (wa, wb):
                if W_cat.get(w) in VES: Cves[w] = (Cves[w] | m) if w in Cves else m
    if node is not None:                                               # g3 node: contested overlaps, then its core
        ncore = node
        for w, t in list(W_terr.items()):
            if w == NODE: continue
            ov_ = t.win(node.sl) & node.m
            if not ov_.any(): continue
            cats = {category(kb, e, W_ent[w]) for e in node_wholes}
            cat = cats.pop() if len(cats) == 1 else 'other_edge'
            cat = 'other_edge' if cat == 'same_entity' else cat
            C[cat][node.sl] |= ov_
            ncore = Bx(ncore.sl, ncore.m & ~ov_, shp)
            if w in W_core: W_core[w] = Bx(W_core[w].sl, W_core[w].m & ~Bx(node.sl, ov_, shp).win(W_core[w].sl), shp)
        if ncore.m.any(): W_core[NODE] = ncore
    core_any = np.zeros(shp, bool)
    for c in W_core.values(): core_any[c.sl] |= c.m
    maxsp = max(sp); tau = max(5.0, maxsp); edge_mm = max(2.0, float(np.sqrt(sum(s * s for s in sp))))
    softt = (I >= -20) & (I <= 150) if I is not None else None
    lbT, nT = ndimage.label(T, structure=EX.STRUCT26); tsz = np.bincount(lbT.ravel(), minlength=nT + 1)
    rows = []
    for mode, M in (('BP', B), ('VT', V), ('BP+VT', B | V)):
        lb, n = ndimage.label(M, structure=EX.STRUCT26)
        cands = []
        for j, ob in enumerate(ndimage.find_objects(lb), start=1):
            if ob is None: continue
            pad = tuple(slice(max(0, s.start - 15), min(sh, s.stop + 15)) for s, sh in zip(ob, shp)); cb = lb[pad] == j
            nc = int(cb.sum()); ml = nc * vml
            if ml < SC.MIN_ML: continue
            tr = bool(T[pad][cb].any()); ids = np.unique(lbT[pad][cb]); ids = ids[ids > 0]
            inter = int((T[pad] & cb).sum()); gs = int(tsz[ids].sum()) if len(ids) else 0
            in_frac = inter / nc; dice_t = (2 * inter / (nc + gs)) if gs else 0.0
            lab = 'F' if not tr else ('T' if (in_frac >= .5 or dice_t >= .1) else 'P')
            insc = tr or P is None or float(P[pad][cb].mean()) >= 0.5
            sh = {'core': float((core_any[pad] & cb).sum()) / nc, 'excl': float((EXc[pad] & cb & ~core_any[pad]).sum()) / nc,
                  'outside': float((~anyterr[pad] & cb).sum()) / nc}
            for k, m in C.items(): sh['c_' + k] = float((m[pad] & cb).sum()) / nc
            dom = max(sh, key=sh.get); dsh = sh[dom]
            ov = {w: float((t.win(pad) & cb).sum()) / nc for w, t in W_terr.items()}
            ov = {w: v for w, v in ov.items() if v > 0}
            host = max(ov, key=ov.get) if ov else ''
            th = thick(cb, sp)
            r = dict(base, mode=mode, cid=j, ml=round(ml, 4), lab=lab, in_scope=bool(insc), in_frac=round(in_frac, 3), dice_t=round(dice_t, 3),
                     dom=dom, dom_share=round(dsh, 3), host=host, host_cat=W_cat.get(host, ''), thick_mm=round(th, 2), V=False, W=False, R=False)
            exbox = tuple(slice(max(0, s.start - int(np.ceil(10 / q))), min(n_, s.stop + int(np.ceil(10 / q)))) for s, q, n_ in zip(ob, sp, shp))
            if ml < SIZE_FLOOR_ML:
                r['member'] = 'U-size'
            elif dom in U_REGIONS:
                r['member'] = 'U-region'
            else:
                # V: vessel cores + supply / vessel-vessel contested
                def vwin(w):
                    a_ = W_core[w].win(pad) if w in W_core else np.zeros(cb.shape, bool)
                    return a_ | Cves[w].win(pad) if w in Cves else a_
                vm = {w: float((vwin(w) & cb).sum()) / nc for w in set(W_core) | set(Cves) if W_cat.get(w) in VES}
                vun = C['supply_drain'][pad] | C['vessel_vessel'][pad]
                for w in vm: vun = vun | vwin(w)
                if vm and float((vun & cb).sum()) / nc >= 0.5:
                    for w in sorted(vm, key=vm.get, reverse=True):
                        if vm[w] <= 0: break
                        vb = W_core[w] | Cves.get(w) if w in W_core else Cves[w]
                        q = ref_q90(vb.full(), sp, exbox)
                        if q is not None and th <= q: r['V'] = True; break
                # W: hollow whole core, CT
                if I is not None:
                    hm = {w: float((W_core[w].win(pad) & cb).sum()) / nc for w in W_core if W_cat.get(w) in HOLLOW}
                    if hm:
                        w = max(hm, key=hm.get)
                        if hm[w] >= 0.5:
                            q = ref_q90(W_core[w].full(), sp, exbox, softt)
                            cs = cb & softt[pad]
                            if q is not None and thick(cs, sp) <= q: r['W'] = True
                # R: rim at a whole-organ territory boundary, or adjacent / vessel_organ contested
                bnd = np.zeros(cb.shape, bool)
                for w, t in W_terr.items():
                    if w == NODE: continue                              # g3 node: no rim explanation from its boundary
                    tp_ = t.win(pad)
                    if not tp_.any(): continue
                    bnd |= tp_ & ~ndimage.binary_erosion(tp_, border_value=1)
                near = (ndimage.distance_transform_edt(~bnd, sampling=sp)[cb] <= edge_mm).mean() if bnd.any() else 0.0
                rel = near >= 0.5 or (float(((C['adjacent'][pad] | C['vessel_organ'][pad]) & cb).sum()) / nc) >= 0.5
                if rel and th <= 2 * maxsp and W_cat.get(host) not in HOLLOW and host != NODE:   # design fix 1: hollow-organ walls only via W
                    prot = 0.0
                    if host:
                        ht = W_terr[host].win(pad); out = cb & ~ht
                        if out.any(): prot = float(ndimage.distance_transform_edt(~ht, sampling=sp)[out].max())
                    r['R'] = prot <= maxsp
                r['member'] = 'explained' if (r['V'] or r['W'] or r['R']) else 'unexplained'
            cands.append((r, pad, cb, 'outside' if (dom == 'outside' or not host) else host))
        # fragment groups: tau, same dominant whole organ (host) or both outside
        k = len(cands); parent = list(range(k))
        def f(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        it = [int(np.ceil(tau / s_)) + 1 for s_ in sp]
        for a in range(k):
            ra, pa, ca, ha = cands[a]
            box = tuple(slice(max(0, p_.start - t_), min(n_, p_.stop + t_)) for p_, t_, n_ in zip(pa, it, shp))
            A = np.zeros(tuple(b_.stop - b_.start for b_ in box), bool)
            A[tuple(slice(p_.start - b_.start, p_.stop - b_.start) for p_, b_ in zip(pa, box))] = ca
            Ad = ndimage.distance_transform_edt(~A, sampling=sp) <= tau
            for b in range(a + 1, k):
                rb, pb, cbb, hb = cands[b]
                if hb != ha: continue
                lo_ = [max(p_.start, b_.start) for p_, b_ in zip(pb, box)]; hi_ = [min(p_.stop, b_.stop) for p_, b_ in zip(pb, box)]
                if any(l_ >= h_ for l_, h_ in zip(lo_, hi_)): continue
                sb = cbb[tuple(slice(l_ - p_.start, h_ - p_.start) for l_, h_, p_ in zip(lo_, hi_, pb))]
                sa = Ad[tuple(slice(l_ - b_.start, h_ - b_.start) for l_, h_, b_ in zip(lo_, hi_, box))]
                if (sb & sa).any(): parent[f(b)] = f(a)
        groups = {}
        for i in range(k): groups.setdefault(f(i), []).append(cands[i][0]['member'])
        for i in range(k):
            ms = groups[f(i)]
            if all(m == 'explained' for m in ms): dec = 'pruned'
            elif any(m.startswith('U-') for m in ms) or any(m == 'explained' for m in ms): dec = 'uncertain'
            else: dec = 'kept'
            r = cands[i][0]; r.update(group=f(i), group_n=len(ms), decision=dec); rows.append(r)
    return rows or [dict(base, note='no candidates >= 0.05 ml')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--only', default=''); ap.add_argument('--budget', type=float, default=1e9)
    a = ap.parse_args()
    J = pd.concat([pd.read_csv(f if os.path.isabs(f) else os.path.join(paths.ledger_dir(), 'evidence_screen', f)) for f in a.jobs.split(',')])
    J = J.drop_duplicates(['ds', 'case']); J['case'] = J.case.astype(str)
    if a.only: J = J[(J.ds + '/' + J.case).isin(a.only.split(','))]
    os.makedirs(OUTD, exist_ok=True); out = os.path.join(OUTD, a.tag + '.csv')
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out, usecols=['ds', 'case']); done = set(zip(d.ds, d.case.astype(str)))
    kb = KnowledgeBase.load(f'{KBD}/knowledge_base')
    t0, n = time.time(), 0
    for ds, case, tf in J[['ds', 'case', 'tumour_free']].itertuples(index=False):
        if (ds, case) in done: continue
        if time.time() - t0 > a.budget: break
        try:
            rows = one(kb, ds, case, bool(tf))
        except Exception as e:
            import traceback
            rows = [dict(ds=ds, case=case, note='error ' + repr(e)[:120] + ' ' + traceback.format_exc().splitlines()[-2][:80])]
        new = not os.path.exists(out)
        with open(out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, COLS, extrasaction='ignore'); new and w.writeheader()
            for r in rows: w.writerow({k: r.get(k, '') for k in COLS})
        n += 1
    left = sum((d, c) not in done for d, c in zip(J.ds, J.case)) - n
    print(f'processed {n}; remaining {left}; {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
