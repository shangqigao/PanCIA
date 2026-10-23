"""v8.2 copy (21 Oct 2026): _mind computed with a KD-tree on boundary voxels (speed only; identical values). --- v8.1 docstring ---
anatomy QC v8.1 (20 Oct 2026): copy of anatomy_qc_v8 (kept); expected organs that are absent no longer count as failed and leave the denominator (only segmented organs grade the scan).
anatomy QC v8 (20 Oct 2026) = anatomy_qc_v7 with the reasoning-layer spans (pancia_kb.rspans): measured spans and variability where
TotalSegmentator CT supports them; in-view share against the frame widened by (variability + 0.5) levels.
anatomy_qc_v7 (18 Oct 2026; main.tex v7 Host reasoning): = anatomy_qc (v6a) with (i) an oversized mask flagged (mu 1/2) for EVERY
organ, not infeasible; (ii) the organ CORE returned: intersection of the masks of the highest state (feasible > flagged) minus
contested regions (Eq. organ_reference); (iii) organ confidence mu (feasible 1, flagged 1/2). anatomy_qc.py is unchanged.
Original docstring:
Anatomy feasibility (host reasoning R(A_i, A_j) over the KB organ-organ relations) and image quality control Q.
Rules: PanCIA/documents/plan/Anatomy_QC_rules_draft.md (user approved 17 Oct 2026); main.tex Eq. (organ_reference),
'Image quality control', 'Host reasoning'. Set logic and KB thresholds only; nothing is fitted on TCGA, stage or agent
outputs; sex is never used. Agents are pluggable: every organ agent's mask A_{k,o} is checked on its own.

  H1 unary     volume > 2 max of KB volume_ml -> infeasible; < 0.5 min -> flag 'small' (incomplete; pilot 17 Oct) (not for a
               mask cut by the FOV); components (anchor organ / gland: KB expected_components; components < 5 % ignored)
               beyond the expected number -> flag only (a tumour can split its organ); side (centroid on the
               wrong side of the planner midline by > 10 mm, only with a reliable midline and handedness),
               vertebral span (centroid level outside the KB span +- (span_var + 1), only with >= 3 TS vertebra labels)
               -> infeasible
  H2 pairwise  organs (crisp categories) not related by containment overlap -> contested region C_{o,o'} (any size); > 5 % of the smaller
               mask -> both 'conflicting'; part within 3 mm of its whole >= 90 % (flag); KB 'abuts' neighbours within
               10 mm (flag); spatial priors within dist_max + 10 mm, side_link respected, 'between' = centroid between the
               two landmarks along left-right (flag). Flags never remove a mask.
  H3/H4        H_o = union of the non-infeasible masks minus every contested region; territory = H_o u contested
  H5           expected organ with no non-infeasible mask -> unsegmentable
  Q            (exclude only if anatomy fails >= 0.5 AND spacing or noise fails; user 17 Oct)
               exclude / degraded / good from the share of unsegmentable expected organs (obligatory, >= half the span in
               the frame) away from the tumour-agent agreement (two agents' cancer-prompt masks) (0.5 / 0.25,
               >= 4 organs), slice spacing > 8 mm, and paraspinal-muscle noise beyond the expert 1st percentile
               (thresholds in knowledge_base/qc_thresholds.json, written by scripts/aqc_expert.py)
"""
from __future__ import annotations
from . import rspans as RS      # v8 (20 Oct): reasoning-layer spans; the planner keeps the frozen KB spans
import json, os
import numpy as np
from scipy import ndimage
from .adapter import _labels_by_side, host_prior_region
from .ledger_v6 import Box, _vt_organ

QP = dict(vol_lo=0.5, vol_hi=2.0, comp_frac=0.05, conflict_frac=0.05, part_in=0.9, part_mm=3.0, abut_mm=10.0,
          sp_tol_mm=10.0, side_mm=10.0, span_extra=1, f_exclude=0.5, f_degraded=0.25, min_expected=4,
          max_spacing_mm=8.0, env_mm=10.0, min_noise_vox=200)
CONTAIN = ('has_part', 'lies_in', 'invested_by', 'has_duct')
COMP_CATS = ('organ', 'gland')
CRISP = ('organ', 'hollow_organ', 'gland', 'bone', 'muscle', 'vessel_artery', 'vessel_vein')   # structures with a boundary of
#   their own; lymph stations, fat, fascia, spaces, regions, ducts and nerves are defined around or through organs and may overlap them
VERSION = 'aqc1'


# ------------------------------------------------------------------ KB relations
def _ancestry(kb, e, depth=3):
    """entities linked to e by a has_part chain (up or down, <= depth hops)"""
    out, fr = {e}, {e}
    for _ in range(depth):
        nx = set()
        for x in fr:
            nx |= {r['dst'] for r in kb.out_edges.get(x, []) if r['type'] == 'has_part'}
            nx |= {r['src'] for r in kb.in_edges.get(x, []) if r['type'] == 'has_part'}
        fr = nx - out; out |= nx
    return out


def related(kb, a, b, _c={}):
    """True if K relates organs a and b by containment: a direct has_part / lies_in / invested_by / has_duct relation, or a
    has_part chain. Same entity is not 'related' here (left vs right kidney must not overlap)."""
    if a == b:
        return False
    k = (id(kb), a, b)
    if k not in _c:
        direct = any(r['dst'] == b and r['type'] in CONTAIN for r in kb.out_edges.get(a, [])) or \
                 any(r['dst'] == a and r['type'] in CONTAIN for r in kb.out_edges.get(b, []))
        _c[k] = direct or b in _ancestry(kb, a)
    return _c[k]


def span_inside(kb, span, frame):
    """share of the organ's KB vertebral span (levels) inside the scan frame; None if either is unknown"""
    if not span or not frame or not frame[0] or not frame[1]:
        return None
    try:
        a, b = kb.level_index(span[0]), kb.level_index(span[1]); f0, f1 = kb.level_index(frame[0]), kb.level_index(frame[1])
    except ValueError:
        return None
    return max(0, min(b, f1) - max(a, f0) + 1) / (b - a + 1)


def _lm_entity(kb, lm):
    return kb.landmarks[lm]['entity'] if lm in kb.landmarks else lm


# ------------------------------------------------------------------ geometry helpers
def _inter(a, b):
    """intersection of two Boxes as (sl, crop) or (None, None)"""
    if a.crop is None or b.crop is None:
        return None, None
    lo = [max(x.start, y.start) for x, y in zip(a.sl, b.sl)]; hi = [min(x.stop, y.stop) for x, y in zip(a.sl, b.sl)]
    if any(l >= h for l, h in zip(lo, hi)):
        return None, None
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    c = a[sl] & b[sl]
    return (sl, c) if c.any() else (None, None)


def _union(boxes, shape):
    boxes = [b for b in boxes if b is not None and b.crop is not None]
    if not boxes:
        return Box(None, shape)
    sl = tuple(slice(min(b.sl[d].start for b in boxes), max(b.sl[d].stop for b in boxes)) for d in range(3))
    m = np.zeros(tuple(s.stop - s.start for s in sl), bool)
    for b in boxes:
        m |= b[sl]
    out = Box(None, shape); out.sl, out.crop = sl, m
    return out


def _minus(b, rms, shape):
    if b.crop is None:
        return b
    m = b.crop.copy()
    for sl, c in rms:
        lo = [max(x.start, y.start) for x, y in zip(b.sl, sl)]; hi = [min(x.stop, y.stop) for x, y in zip(b.sl, sl)]
        if any(l >= h for l, h in zip(lo, hi)):
            continue
        m[tuple(slice(l - x.start, h - x.start) for l, h, x in zip(lo, hi, b.sl))] &= \
            ~c[tuple(slice(l - y.start, h - y.start) for l, h, y in zip(lo, hi, sl))]
    out = Box(None, shape)
    if m.any():
        out.sl, out.crop = b.sl, m
    return out


def _edge(b):
    if b.crop is None:
        return False
    for d in range(3):
        if b.sl[d].start == 0 and np.take(b.crop, 0, axis=d).any():
            return True
        if b.sl[d].stop == b.shape[d] and np.take(b.crop, -1, axis=d).any():
            return True
    return False


def _centroid_world(b, aff):
    idx = np.argwhere(b.crop).mean(0) + np.array([s.start for s in b.sl])
    return (np.asarray(aff) @ np.r_[idx, 1.0])[:3]


def _mind(a, b, sp, cap):
    """min surface distance a -> b (mm), searched within cap mm of a; inf beyond"""
    if a.crop is None or b.crop is None:
        return np.nan
    pad = [int(np.ceil(cap / q)) + 1 for q in sp]
    sl = tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(a.sl, pad, a.shape))
    bb = b[sl]
    if not bb.any():
        return np.inf
    aa = a[sl]
    if (aa & bb).any():
        return 0.0
    # v8.2 (speed only, identical result): the closest pair of two voxel sets lies on their 6-connected boundaries -> KD-tree
    # on boundary voxels (true spacing) instead of a full distance transform of the padded box
    from scipy.spatial import cKDTree
    st = ndimage.generate_binary_structure(3, 1)
    ab = np.argwhere(aa & ~ndimage.binary_erosion(aa, st, border_value=0)) * np.asarray(sp, float)
    bbd = np.argwhere(bb & ~ndimage.binary_erosion(bb, st, border_value=0)) * np.asarray(sp, float)
    return float(cKDTree(bbd).query(ab, k=1)[0].min())



def _within(region, full, sp, mm):
    """True iff a voxel of the full-grid mask lies within mm (surface distance, true spacing) of region: identical to
    (full[E.sl] & E.crop).any() with E = region.dilate(mm, sp) (E is a subset of the box padded by mm)."""
    pad = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    sl = tuple(slice(max(0, a.start - p), min(n, a.stop + p)) for a, p, n in zip(region.sl, pad, region.shape))
    tb = full[sl]
    if not tb.any():
        return False
    rb = region[sl]
    if (rb & tb).any():
        return True
    from scipy.spatial import cKDTree
    st = ndimage.generate_binary_structure(3, 1)
    ra = np.argwhere(rb & ~ndimage.binary_erosion(rb, st, border_value=0)) * np.asarray(sp, float)
    ta = np.argwhere(tb & ~ndimage.binary_erosion(tb, st, border_value=0)) * np.asarray(sp, float)
    return bool(cKDTree(ra).query(ta, k=1, distance_upper_bound=mm + 1e-6)[0].min() <= mm)

# ------------------------------------------------------------------ inputs
def _vt_organ_nonempty(plan_dir, stem, shape):
    """v8.1 (21 Oct 2026, D4 check): as ledger_v6._vt_organ (majority of the main phrase and its alias prompts), but an alias prompt
    that returned an EMPTY mask does not vote: no output is not a vote against. Before, one empty alias vetoed a non-empty main
    mask (majority of 2 needs both), e.g. the VoxTell breast mask of BRCA BILATERAL_ASSET_3_3.
    NOT ADOPTED (21 Oct): over the pilot it would recover 34 masks in 32 series, many of them implausible (e.g. seminal vesicle and
    prostate in CESC/UCEC scans): the empty alias also filters hallucinated masks. Kept for the user's decision."""
    import os as _os
    from .ledger_v6 import _bool, _load
    files = [_os.path.join(plan_dir, stem + sfx + '.nii.gz') for sfx in ('', '_a2', '_a3')]
    ms = [m for m in (_bool(_load(f)) for f in files if _os.path.exists(f)) if m.any()]
    if not ms:
        return np.zeros(shape, bool)
    if len(ms) == 1:
        return ms[0]
    return np.sum(ms, axis=0) >= (len(ms) // 2 + 1)


def organ_masks(kb, md, names, task, plan_dir, manifest, shape):
    """{(entity, side): {agent: Box}} for every KB organ with a TS label (incl. anchors found from parts) or a VT organ mask"""
    tmap = kb.ts_to_entity.get(task, {})
    ents = {tmap[n][0] for n in names.values() if n in tmap and tmap[n][0] in kb.entities}
    for e in list(ents):
        for r in kb.in_edges.get(e, []):
            if r['type'] == 'has_part' and kb.entities.get(r['src'], {}).get('found_from_parts'):
                ents.add(r['src'])
    A = {}
    objs = ndimage.find_objects(md)                       # per-label bounding boxes (speed: masks built on crops)
    for e in sorted(ents):
        for sd, labs in _labels_by_side(kb, task, names, e).items():
            sls = [objs[l - 1] for l in labs if 0 < l <= len(objs) and objs[l - 1] is not None]
            if not sls:
                continue
            sl = tuple(slice(min(x[d].start for x in sls), max(x[d].stop for x in sls)) for d in range(3))
            b = Box(None, shape); b.sl, b.crop = sl, np.isin(md[sl], labs)
            if b.crop.any():
                A.setdefault((e, sd), {})['TS'] = b
    seen = set()
    for mm in manifest.get('masks', []):
        if mm.get('seg_obj') != 'organ' or mm.get('list') == 'T':
            continue
        key = (mm['entity'], mm.get('side'))
        if key in seen or mm['entity'] not in kb.entities:
            continue
        seen.add(key)
        v = _vt_organ_fast(plan_dir, f"{key[0]}{'_' + key[1] if key[1] else ''}_organ", shape)   # v8.2: = ledger_v6._vt_organ, faster   # _vt_organ_nonempty: tested on D4 21 Oct, NOT adopted (see docstring)
        bx = _fast_box(v, shape)
        if bx.crop is not None:
            A.setdefault(key, {})['VT'] = bx
    return A



def _vt_organ_fast(plan_dir, stem, shape):
    """identical to ledger_v6._vt_organ (voxel majority of main + alias prompts); uint8 counts instead of np.sum over a list (speed)"""
    import os
    from .ledger_v6 import _bool, _load
    files = [os.path.join(plan_dir, stem + sfx + '.nii.gz') for sfx in ('', '_a2', '_a3')]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return np.zeros(shape, bool)
    if len(files) == 1:
        return _bool(_load(files[0]))
    cnt = None
    for f in files:
        m = _bool(_load(f))
        cnt = m.astype(np.uint8) if cnt is None else cnt + m
    return cnt >= (len(files) // 2 + 1)


def _fast_box(mask, shape):
    """identical to Box(mask, shape) (no padding): bounding box from axis projections instead of find_objects (speed)"""
    b = Box(None, shape)
    p01 = mask.any(axis=2)
    if not p01.any():
        return b
    z = np.flatnonzero(mask.any(axis=(0, 1))); x = np.flatnonzero(p01.any(axis=1)); y = np.flatnonzero(p01.any(axis=0))
    b.sl = (slice(int(x[0]), int(x[-1]) + 1), slice(int(y[0]), int(y[-1]) + 1), slice(int(z[0]), int(z[-1]) + 1))
    b.crop = mask[b.sl].copy()
    return b

def vertebral_levels(names, md, aff):
    """{level: world z of the TS vertebra label centroid} (CT task; MR has one 'vertebrae' label -> empty)"""
    out = {}
    objs = ndimage.find_objects(md)
    for l, n in names.items():
        if n.startswith('vertebrae_') and len(n) > 10 and 0 < l <= len(objs) and objs[l - 1] is not None:
            sl = objs[l - 1]
            idx = np.argwhere(md[sl] == l)
            if len(idx) > 50:
                idx = idx + np.array([x.start for x in sl])
                out[n[len('vertebrae_'):]] = float((np.asarray(aff) @ np.r_[idx.mean(0), 1.0])[2])
    return out


# ------------------------------------------------------------------ H1
def unary(kb, key, b, sp, vox_ml, ctx):
    ent, side = key
    e = kb.entities.get(ent, {})
    fail, flags = [], []
    v = b.sum() * vox_ml
    trunc = _edge(b)
    if trunc:
        flags.append('truncated')
    vr = e.get('volume_ml')
    inside = RS.inside(kb, ent, ctx.get('frame'))
    if vr and not trunc and v > QP['vol_hi'] * vr['max']:
        flags.append('large_primary' if ent in (ctx.get('primary') or ()) else 'large')   # v7: disease can enlarge any organ (mu 1/2)
    elif vr and not trunc and v < QP['vol_lo'] * vr['min']:
        flags.append('small')       # pilot 17 Oct: a too-small mask is incomplete (partial MR coverage, small organ), not wrong
    if e.get('category') in COMP_CATS and e.get('is_anchor'):
        lb, n = ndimage.label(b.crop, np.ones((3, 3, 3)))
        if n > 1:
            sz = np.bincount(lb.ravel())[1:]
            k = int((sz >= QP['comp_frac'] * sz.sum()).sum())
            exp = int(e.get('expected_components', 1)) * (2 if (e.get('laterality') == 'bilateral' and side is None) else 1)
            if k > exp:                          # pilot 17 Oct: flag only. A tumour can split its organ, and dropping (KB qc
                flags.append(f'components {k}>{exp}')   # keep_largest_k) or rejecting the mask cut primaries off their host
    if side in ('left', 'right') and ctx.get('mid_x') is not None and ctx.get('side_ok'):
        dx = ctx['hand'] * (_centroid_world(b, ctx['aff'])[0] - ctx['mid_x'])     # + = patient left
        if (side == 'left' and dx < -QP['side_mm']) or (side == 'right' and dx > QP['side_mm']):
            fail.append(f'side {side} dx={dx:+.0f}mm')
    span = RS.span(kb, ent)
    lv = ctx.get('levels') or {}
    if span and len(lv) >= 3 and not trunc:
        z = _centroid_world(b, ctx['aff'])[2]
        near = min(lv, key=lambda L: abs(lv[L] - z))
        try:
            i = kb.level_index(near); a, c = kb.level_index(span[0]), kb.level_index(span[1])
            tol = RS.var(kb, ent) + QP['span_extra']
            far = abs(lv[near] - z) > 60.0          # centroid beyond the labelled vertebrae: level unknown
            if not far and (i < a - tol or i > c + tol):
                fail.append(f'span {near} not in {span[0]}-{span[1]}+-{tol}')
        except ValueError:
            pass
    return fail, flags


# ------------------------------------------------------------------ main
def analyse(kb, md, names, task, aff, sp, plan, manifest, plan_dir, tumour=None, image=None, modality=None,
            thresholds=None, check_keys=None, primary=None):
    """Host reasoning over all organ masks of the scan + quality state. Returns a dict (see keys at the end)."""
    shape = md.shape
    vox_ml = float(np.prod(sp)) / 1000.0
    A = organ_masks(kb, md, names, task, plan_dir, manifest, shape)
    log = plan.get('log', [])
    mid = next((l for l in log if l.get('step') == 'midline'), {}); hd = next((l for l in log if l.get('step') == 'header_handedness'), {})
    lv = {L: z for L, z in vertebral_levels(names, md, aff).items() if L in kb.vertebral_order}
    side_ok = mid.get('source', 'midline_structures') == 'midline_structures' and hd.get('source', 'anatomy') in ('anatomy', 'ts_sided_pairs') \
        and hd.get('agreement', True) is True        # side checked only with a reliable midline and header handedness
    ctx = dict(aff=aff, mid_x=mid.get('x_mm'), hand=hd.get('sign', 1) or 1, levels=lv, side_ok=side_ok, primary=set(primary or ()),
               frame=(plan.get('frame') or {}).get('span'))
    # H1 per agent mask
    phi = {}
    for key, ag in A.items():
        for k, b in ag.items():
            fail, flags = unary(kb, key, b, sp, vox_ml, ctx)
            phi[(key, k)] = dict(state='infeasible' if fail else ('flagged' if flags else 'feasible'), fail=fail, flags=list(flags))
    U = {key: _union([b for k, b in ag.items() if phi[(key, k)]['state'] != 'infeasible'], shape) for key, ag in A.items()}
    # H2a contested regions (overlaps need a non-exclusive agent: any pair whose masks come from different label maps)
    keys = [k for k in U if U[k].crop is not None]
    contested, pairs = {}, []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if related(kb, k1[0], k2[0]):
                continue
            if kb.entities.get(k1[0], {}).get('category') not in CRISP or kb.entities.get(k2[0], {}).get('category') not in CRISP:
                continue
            if set(A[k1]) == {'TS'} and set(A[k2]) == {'TS'}:
                continue                                     # one TS label map: exclusive by construction
            sl, c = _inter(U[k1], U[k2])
            if sl is None:
                continue
            n = int(c.sum()); small = min(U[k1].sum(), U[k2].sum())
            conf = n >= QP['conflict_frac'] * small
            contested.setdefault(k1, []).append((sl, c)); contested.setdefault(k2, []).append((sl, c))
            pairs.append(dict(a=_hkey(k1), b=_hkey(k2), ml=round(n * vox_ml, 2), frac_small=round(n / max(small, 1), 3), conflicting=conf))
            if conf:
                for key in (k1, k2):
                    for k in A[key]:
                        if phi[(key, k)]['state'] != 'infeasible':
                            phi[(key, k)]['flags'].append(f'conflict:{_hkey(k2 if key == k1 else k1)}')
                            phi[(key, k)]['state'] = 'flagged'
    # H2b-d flags (only for KB anchors: the hosts and the expected organs)
    ck = set(check_keys) if check_keys else {k for k in keys if kb.entities.get(k[0], {}).get('is_anchor')}   # anchors (hosts, expected)
    byent = {}
    for key in keys:
        byent.setdefault(key[0], []).append(key)
    pflags = {}
    for key in keys:
        if key not in ck:
            continue
        ent, side = key; e = kb.entities.get(ent, {}); fl = []
        trunc = _edge(U[key])
        for r in kb.in_edges.get(ent, []):                     # part within 3 mm of its whole
            if r['type'] == 'has_part' and r['src'] in byent and r.get('spatial') == 'inside':
                W = _union([U[k] for k in byent[r['src']]], shape)
                d = ndimage.distance_transform_edt(~W[U[key].sl], sampling=sp) <= QP['part_mm']
                if (d & U[key].crop).sum() < QP['part_in'] * U[key].sum():
                    fl.append(f"part_outside:{r['src']}")
        if not trunc:
            for r in kb.out_edges.get(ent, []):                # KB 'abuts' neighbours in contact
                if r['type'] != 'adjacent_to' or r.get('contact') != 'abuts' or r['dst'] not in byent:
                    continue
                sl_ = r.get('side_link', 'same')
                if sl_ in ('left', 'right') and side not in (sl_, None):
                    continue                                   # relation holds for one side only (kidney - liver: right)
                if sl_ == 'same':
                    nk = [k for k in byent[r['dst']] if side is None or k[1] in (side, None)]
                else:
                    nk = list(byent[r['dst']])
                nb = _union([U[k] for k in nk if not _edge(U[k])], shape)
                if nb.crop is None:
                    continue
                d = _mind(U[key], nb, sp, QP['abut_mm'] + 20)
                if d > QP['abut_mm']:
                    fl.append(f"no_contact:{r['dst']}:{d:.0f}mm")
            for spr in ((e.get('anchor') or {}).get('spatial_prior') or []):
                lm = _lm_entity(kb, spr.get('landmark'))
                if lm not in byent:
                    continue
                if spr.get('side_link') and side not in (spr['side_link'], None):
                    continue
                if spr.get('relation') == 'between':
                    sides = {k[1]: U[k] for k in byent[lm] if k[1] in ('left', 'right')}
                    if len(sides) == 2 and ctx.get('mid_x') is not None:
                        x = _centroid_world(U[key], aff)[0]; xl = _centroid_world(sides['left'], aff)[0]; xr = _centroid_world(sides['right'], aff)[0]
                        if not (min(xl, xr) <= x <= max(xl, xr)):
                            fl.append(f'not_between:{lm}')
                    continue
                dm = spr.get('dist_mm')
                if not dm:
                    continue
                L = _union([U[k] for k in byent[lm]], shape)
                d = _mind(U[key], L, sp, dm[1] + QP['sp_tol_mm'] + 20)
                if d > dm[1] + QP['sp_tol_mm']:
                    fl.append(f"spatial:{lm}:{spr.get('relation')}:{d:.0f}>{dm[1]}")
        if fl:
            pflags[key] = fl
            for k in A[key]:
                if phi[(key, k)]['state'] == 'feasible':
                    phi[(key, k)]['state'] = 'flagged'
                phi[(key, k)]['flags'] += fl
    # H4 references
    organs = {}
    for key in A:
        C = contested.get(key, [])
        ref = _minus(U[key], C, shape)
        Cb = Box(None, shape)
        if C:
            Cb = _union([_box_of(sl, c, shape) for sl, c in C], shape)
        st = {k: phi[(key, k)]['state'] for k in A[key]}
        best = 'feasible' if 'feasible' in st.values() else ('flagged' if 'flagged' in st.values() else None)
        core = Box(None, shape)
        if best is not None:
            bs = [A[key][k] for k in A[key] if st[k] == best]
            core = bs[0]
            for b_ in bs[1:]:
                sl_, c_ = _inter(core, b_)
                core = _box_of(sl_, c_, shape) if sl_ is not None else Box(None, shape)
            core = _minus(core, C, shape) if core.crop is not None else core
        organs[key] = dict(ref=ref, territory=_union([ref, Cb], shape), contested=Cb, core=core,
                           mu=1.0 if best == 'feasible' else (0.5 if best == 'flagged' else 0.0),
                           agents={k: phi[(key, k)]['state'] for k in A[key]},
                           fail={k: phi[(key, k)]['fail'] for k in A[key] if phi[(key, k)]['fail']},
                           flags=sorted({f for k in A[key] for f in phi[(key, k)]['flags']}),
                           conflicting=any(f.startswith('conflict:') for k in A[key] for f in phi[(key, k)]['flags']))
    # H5 + Q2: expected organs
    absent = set(plan.get('anchors_absent_candidates') or [])
    prompted = {m['entity'] for m in manifest.get('masks', []) if m.get('seg_obj') == 'organ'}
    tmap_ents = {v[0] for v in kb.ts_to_entity.get(task, {}).values()}
    exp_keys = []
    for ent in plan.get('anchors_expected') or []:
        if ent in absent or ent not in kb.entities:
            continue
        e = kb.entities[ent]
        asked = ent in prompted or ent in tmap_ents or any(r['type'] == 'has_part' and r['dst'] in tmap_ents
                                                            for r in kb.out_edges.get(ent, [])) and e.get('found_from_parts')
        if not asked:
            continue
        if (e.get('anchor') or {}).get('presence') != 'obligatory':
            continue                                   # surgical / sex / variable presence: absence is not a failure
        ins = RS.inside(kb, ent, ctx['frame'])
        if ins is not None and ins < 0.5:
            continue                                   # QC counts only organs at least half inside the frame
        if e.get('laterality') == 'bilateral':
            exp_keys += [(ent, 'left'), (ent, 'right')]
        else:
            exp_keys.append((ent, None))
    rows = []
    for key in exp_keys:
        o = organs.get(key)
        unseg = o is None or o['ref'].crop is None
        failed = o is not None and unseg   # v8.1 (user 20 Oct): an ABSENT organ never counts (field of view / protocol, e.g. breast
        #                         MR); only a mask an agent produced and that is infeasible counts against image quality
        # pilot 17 Oct: conflicts are mostly one agent's error (VT pelvic organs), not image quality;
        #                         they shape the references (contested regions) but do not grade the scan
        region = None
        if o is not None:
            region = _union(list(A[key].values()), shape)
        elif tumour is not None:
            H, _ = host_prior_region(kb, key[0], md, names, task, sp)
            region = Box(H, shape) if H is not None else None
        touched = None
        if tumour is not None and region is not None and region.crop is not None:
            touched = _within(region, tumour, sp, QP['env_mm'])          # v8.2: same test as dilate + overlap, faster
        rows.append(dict(organ=_hkey(key), unsegmentable=unseg, failed=failed, tumour_touched=touched,
                         conflicting=bool(o and o['conflicting']),
                         why=('missing' if o is None else 'all_infeasible' if unseg else '')))
        if o is not None:
            o['unsegmentable'] = unseg
    # v8.1 (user 20 Oct): absent expected organs leave the denominator too (and do not count towards min_expected): only organs an
    # agent actually segmented, away from the tumour, grade the scan
    away = [r for r in rows if r['tumour_touched'] is False and r['why'] != 'missing']
    f = (sum(r['failed'] for r in away) / len(away)) if len(away) >= QP['min_expected'] else None
    reasons, state = [], 'good'
    anat_bad = f is not None and f >= QP['f_exclude']
    if f is not None and f >= QP['f_degraded']:
        state = 'degraded'; reasons.append(f'anatomy_fail_away={f:.2f}')
    if max(sp) > QP['max_spacing_mm']:
        reasons.append(f'spacing={max(sp):.1f}mm'); state = 'degraded' if state == 'good' else state
    noise = None
    if image is not None:
        mus = [organs[k]['ref'] for k in organs if k[0] in ('autochthon',) and organs[k]['ref'].crop is not None]
        if mus:
            noise = noise_metric(image, _union(mus, shape), aff, modality)
            th = (thresholds or {}).get(modality or '', {})
            if noise and modality == 'CT' and th.get('sigma_p99') and noise['sigma'] > th['sigma_p99']:
                reasons.append(f"noise_sigma={noise['sigma']:.1f}>{th['sigma_p99']:.1f}"); state = 'degraded' if state == 'good' else state
            if noise and modality == 'MR' and th.get('snr_p01') and noise['snr'] < th['snr_p01']:
                reasons.append(f"noise_snr={noise['snr']:.1f}<{th['snr_p01']:.1f}"); state = 'degraded' if state == 'good' else state
    # user, 17 Oct: missing anatomy alone never excludes (dedicated protocols, e.g. breast MR, show little anatomy besides the
    # host); a scan is excluded only when most expected organs fail AND an image measure (spacing or noise) fails as well
    if anat_bad and any(r.startswith(('spacing', 'noise')) for r in reasons):
        state = 'exclude'
    return dict(version=VERSION, organs=organs, pairs=pairs, expected=rows, f_away=f, n_away=len(away),
                state=state, reasons=reasons, noise=noise, phi={f'{_hkey(k)}|{a}': v for (k, a), v in phi.items()},
                levels=len(ctx['levels']))


def _box_of(sl, c, shape):
    b = Box(None, shape); b.sl, b.crop = sl, c; return b


def _hkey(key):
    return key[0] + (f'_{key[1]}' if key[1] else '')


# ------------------------------------------------------------------ Q1 noise
def noise_metric(X, M, aff, modality):
    """Noise in a homogeneous reference tissue (paraspinal muscle), on the common ~5 mm grid (grounding.harmonise_*):
    in-plane block-averaged to ~1.5 mm; sigma = 1.4826 MAD of the in-plane high-pass residual (image minus its 3x3 in-plane median) inside the muscle eroded by 1
    voxel. CT: sigma (HU); MR: SNR = median / sigma. None if fewer than 200 voxels."""
    import nibabel as nib
    from . import grounding as GR
    if M.crop is None:
        return None
    ax = [k for k, c in enumerate(nib.aff2axcodes(aff)) if c in 'SI'][0]
    sp_si = float(np.sqrt((np.asarray(aff)[:3, ax] ** 2).sum()))
    k = GR.thick_k(sp_si)
    pad = [2, 2, 2]
    sl = tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(M.sl, pad, M.shape))
    x = np.asarray(X[sl], np.float32); m = M[sl]
    x = GR.harmonise_array(x, ax, k); m = GR.harmonise_array(m.astype(np.int16), ax, k, label=True) > 0
    for d in range(3):                                   # in-plane: block-average to ~1.5 mm (expert CT grid), so that noise
        if d == ax:                                      # is compared at one resolution
            continue
        s_ = float(np.sqrt((np.asarray(aff)[:3, d] ** 2).sum())); kk = max(1, int(np.floor(1.5 / s_ + 0.5)))
        if kk > 1:
            x = GR.harmonise_array(x, d, kk); m = GR.harmonise_array(m.astype(np.int16), d, kk, label=True) > 0
    size = [3, 3, 3]; size[ax] = 1
    r = x - ndimage.median_filter(x, size=size)
    me = ndimage.binary_erosion(m, iterations=1)
    if me.sum() < QP['min_noise_vox']:
        return None
    v = r[me]; s = float(1.4826 * np.median(np.abs(v - np.median(v))))
    med = float(np.median(x[me]))
    return dict(sigma=round(s, 3), median=round(med, 2), snr=round(med / s, 2) if s > 0 else None, n=int(me.sum()), k=k)
