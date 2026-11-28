"""General anatomy check (27 Nov 2026; plan claude/anatomy_check_general_plan_2026-11-27.md, user-approved; workflow step 2 for the
graph reasoning, claude/stage1_workflow.md). Inputs are ONLY anatomy: the TotalSegmentator label map, the VoxTell plan organ masks
(all tiers; user 27 Nov), the plan (frame, midline, handedness, expected anchors: TS landmarks), the KB and the image (noise only).
No tumour masks, no cancer type, no planner primary host.

The organ reasoning is anatomy_qc_v82.analyse UNCHANGED (H1 unary, H2 pairwise conflicts / contested regions, part within whole,
'abuts', spatial priors, 'between'; H3/H4 cores and territories), called with tumour=None and primary=None:
  - in v82 the tumour mask (BP AND VT) is used only by scan quality Q (organs within 10 mm of it do not grade the scan), and the
    primary host only labels an oversized organ 'large_primary' instead of 'large' (both mu 1/2).
Q here is graded from EVERY segmented expected organ (no tumour exclusion), with v82's thresholds and image measures:
  f = share of segmented, obligatory, in-frame expected organs whose masks are all infeasible (needs >= min_expected organs);
  f >= f_degraded -> degraded; spacing > max_spacing_mm or noise beyond the expert percentile -> degraded;
  exclude only if f >= f_exclude AND an image measure fails (user 17 Oct).
g2 (27 Nov, user decision 3 of claude/graph_pruning_v3_plan_2026-11-27.md): before the organ reasoning, consistency() sets aside the
masks of KB-exclusive presence classes (sex_male / sex_female, incl. their has_part sub-structures) when both occur in one scan; if
only one class passes its single-mask checks the other is set aside, otherwise both (a dispute the masks cannot resolve is an agent
defect; no agent is preferred, no patient metadata). Set-aside masks take no part in the joint checks. Recorded in r['consistency'].
save() persists the result per scan (npz of packed organ cores / territories / contested regions + json of states and flags)."""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from . import anatomy_qc_v82 as AQ

VERSION = 'g2'   # 27 Nov: + logical-consistency step (KB-exclusive presence classes), see consistency()
CODE_HASH = hashlib.md5(b''.join(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), 'rb').read()
                                 for f in ('anatomy_check_g.py', 'anatomy_qc_v82.py', 'rspans.py'))).hexdigest()[:12]


EXCLUSIVE = ('sex_male', 'sex_female')   # KB anchor presence classes that cannot co-occur in one body


def _class_of(kb):
    """entity -> exclusive presence class, for the anchors of those classes and every entity below them by has_part"""
    out = {}
    for e, v in kb.entities.items():
        c = (v.get('anchor') or {}).get('presence')
        if c in EXCLUSIVE:
            out[e] = c
    todo = list(out)
    while todo:
        e = todo.pop()
        for r in kb.out_edges.get(e, []):
            if r['type'] == 'has_part' and r['dst'] not in out:
                out[r['dst']] = out[e]; todo.append(r['dst'])
    return out


def consistency(kb, md, names, task, aff, sp, plan, manifest, plan_dir):
    """Logical-consistency step (user, 27 Nov 2026; claude/graph_pruning_v3_plan_2026-11-27.md, decision 3).
    1. Per-agent presence of the KB-exclusive classes: TS (labels present in the label map) and VT (plan organ masks whose prompt
       majority is non-empty), each class mapped through has_part (e.g. prostate zones -> sex_male, myometrium -> sex_female).
    2. If both classes are present in the scan (within one agent or across agents), it is a contradiction of the agents' output.
       Per-agent single-mask check (anatomy_qc_v82.unary: side / level failure, 'small') on every mask of each class: a class
       'passes' if at least one of its masks passes. If exactly one class passes, the other class is set aside; otherwise
       (both pass, or both fail) BOTH classes are set aside: the dispute cannot be resolved from the masks, and no agent is
       preferred. No patient metadata is used.
    Returns (md2, manifest2, record): the label map with set-aside TS labels zeroed (a copy only if needed), the manifest without
    set-aside VT organ masks, and the record of what was set aside and why."""
    cls = _class_of(kb)
    tmap = kb.ts_to_entity.get(task, {})
    present_labels = set(int(x) for x in np.unique(md) if x > 0)
    ts_keys = {}
    for l in present_labels:
        n = names.get(l)
        if n in tmap and tmap[n][0] in cls:
            ts_keys.setdefault((tmap[n][0], tmap[n][1] if len(tmap[n]) > 1 else None), []).append(l)
    votes = {}
    for mm in manifest.get('masks', []):
        if mm.get('seg_obj') != 'organ' or mm.get('list') == 'T' or mm.get('entity') not in cls:
            continue
        k = (mm['entity'], mm.get('side'))
        votes.setdefault(k, []).append((mm.get('voxels') or 0) > 0)
    vt_keys = [k for k, v in votes.items() if sum(v) >= len(v) // 2 + 1]
    pres = {c: [] for c in EXCLUSIVE}
    for k in ts_keys: pres[cls[k[0]]].append(('TS', k))
    for k in vt_keys: pres[cls[k[0]]].append(('VT', k))
    record = dict(classes_present={c: [f'{a}:{k[0]}' + (f'_{k[1]}' if k[1] else '') for a, k in v] for c, v in pres.items() if v},
                  contradiction=all(pres[c] for c in EXCLUSIVE), set_aside=[], reason='')
    if not record['contradiction']:
        return md, manifest, record
    # per-agent single-mask check of every mask of both classes
    log = plan.get('log', [])
    mid = next((l for l in log if l.get('step') == 'midline'), {}); hd = next((l for l in log if l.get('step') == 'header_handedness'), {})
    lv = {L: z for L, z in AQ.vertebral_levels(names, md, aff).items() if L in kb.vertebral_order}
    side_ok = mid.get('source', 'midline_structures') == 'midline_structures' and hd.get('source', 'anatomy') in ('anatomy', 'ts_sided_pairs') \
        and hd.get('agreement', True) is True
    ctx = dict(aff=aff, mid_x=mid.get('x_mm'), hand=hd.get('sign', 1) or 1, levels=lv, side_ok=side_ok, primary=set(),
               frame=(plan.get('frame') or {}).get('span'))
    vox_ml = float(np.prod(sp)) / 1000.0; shape = md.shape
    passes = {}
    for c in EXCLUSIVE:
        ok = False
        for a, k in pres[c]:
            if a == 'TS':
                b = AQ.Box(None, shape); m = np.isin(md, ts_keys[k])
                b = AQ._fast_box(m, shape); del m
            else:
                b = AQ._fast_box(AQ._vt_organ_fast(plan_dir, f"{k[0]}{'_' + k[1] if k[1] else ''}_organ", shape), shape)
            if b.crop is None:
                continue
            fail, flags = AQ.unary(kb, k, b, sp, vox_ml, ctx)
            if not fail and 'small' not in flags:
                ok = True
        passes[c] = ok
    if passes[EXCLUSIVE[0]] != passes[EXCLUSIVE[1]]:
        drop = [c for c in EXCLUSIVE if not passes[c]]; record['reason'] = 'only one class passes its single-mask checks'
    else:
        drop = list(EXCLUSIVE); record['reason'] = 'both classes pass' if passes[EXCLUSIVE[0]] else 'both classes fail'
    drop_ents = {e for e, c in cls.items() if c in drop}
    record['set_aside'] = sorted({f'{a}:{k[0]}' + (f'_{k[1]}' if k[1] else '') for c in drop for a, k in pres[c]})
    record['dropped_classes'] = drop
    lab = [l for k, ls in ts_keys.items() if k[0] in drop_ents for l in ls]
    md2 = md
    if lab:
        md2 = md.copy(); md2[np.isin(md2, lab)] = 0
    man2 = dict(manifest); man2['masks'] = [mm for mm in manifest.get('masks', [])
                                            if not (mm.get('seg_obj') == 'organ' and mm.get('entity') in drop_ents)]
    return md2, man2, record


def check(kb, md, names, task, aff, sp, plan, manifest, plan_dir, image=None, modality=None, thresholds=None):
    md, manifest, cons = consistency(kb, md, names, task, aff, sp, plan, manifest, plan_dir)
    r = AQ.analyse(kb, md, names, task, aff, sp, plan, manifest, plan_dir, tumour=None, image=image, modality=modality,
                   thresholds=thresholds, check_keys=None, primary=None)
    r['consistency'] = cons
    QP = AQ.QP
    seg = [x for x in r['expected'] if x['why'] != 'missing']
    f = (sum(x['failed'] for x in seg) / len(seg)) if len(seg) >= QP['min_expected'] else None
    img_reasons = [x for x in r['reasons'] if x.startswith(('spacing', 'noise'))]
    reasons, state = [], 'good'
    if f is not None and f >= QP['f_degraded']:
        state = 'degraded'; reasons.append(f'anatomy_fail={f:.2f}')
    if img_reasons:
        reasons += img_reasons; state = 'degraded'
    if f is not None and f >= QP['f_exclude'] and img_reasons:
        state = 'exclude'
    r.update(state=state, reasons=reasons, f_all=f, n_graded=len(seg), version=VERSION, code_hash=CODE_HASH)
    for k in ('f_away', 'n_away'):
        r.pop(k, None)
    # NOT the ledger's anatomy_insufficient (v8.5 G3 exempts scans whose planned primary organ has a mask; that uses the cancer type)
    r['no_mu1_organ'] = not any(o['mu'] == 1.0 for o in r['organs'].values())
    return r


def _vt_files(manifest, plan_dir):
    out = []
    for mm in manifest.get('masks', []):
        if mm.get('seg_obj') != 'organ' or mm.get('list') == 'T':
            continue
        stem = f"{mm['entity']}{'_' + mm['side'] if mm.get('side') else ''}_organ"
        out += [os.path.join(plan_dir, stem + sfx + '.nii.gz') for sfx in ('', '_a2', '_a3')]
    return sorted({f for f in out if os.path.exists(f)})


def check_scan(kb, ts_path, labels_path, plan, manifest, plan_dir, image_path=None, modality=None, thresholds=None, margin=2):
    """check() on a crop (memory, 27 Nov: whole-body CT exceeds the 4 GB VM). Crop = bounding box of every TS label and every
    VoxTell plan organ file, + margin voxels, clipped to the grid. Exactness: masks touching the scan edge still touch the crop edge
    (the box reaches that edge); no other mask can touch it (margin >= 1); world coordinates use the shifted affine; every check is a
    set operation or distance on these masks. Returns (r, offset, full_shape, affine_full); boxes in r are in crop coordinates."""
    import nibabel as nib
    from scipy import ndimage
    ts = nib.load(ts_path); md = np.asanyarray(ts.dataobj)
    md = md if md.dtype.kind in 'iu' else md.astype(np.int16)
    full = md.shape; lo = np.array(full); hi = np.zeros(3, int)
    def grow(m):
        nonlocal lo, hi
        if m.any():
            idx = [np.flatnonzero(m.any(axis=tuple(j for j in range(3) if j != d))) for d in range(3)]
            lo = np.minimum(lo, [i[0] for i in idx]); hi = np.maximum(hi, [i[-1] + 1 for i in idx])
    grow(md > 0)
    for f in _vt_files(manifest, plan_dir):
        grow(np.asanyarray(nib.load(f).dataobj) > 0)
    if (hi <= lo).any():
        lo, hi = np.zeros(3, int), np.array(full)
    lo = np.maximum(lo - margin, 0); hi = np.minimum(hi + margin, full)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    md = np.ascontiguousarray(md[sl])
    T = np.eye(4); T[:3, 3] = lo; aff = np.asarray(ts.affine) @ T
    X = None
    if image_path and os.path.exists(image_path):
        Xf = np.asanyarray(nib.load(image_path).dataobj)
        Xf = Xf[..., 0] if Xf.ndim == 4 else Xf
        X = np.ascontiguousarray(Xf[sl]); del Xf
    names = {int(k): v for k, v in (json.load(open(labels_path)).get('labels') or {}).items()}
    task = json.load(open(labels_path)).get('task') or 'total'
    from .adapter import _spacing
    sp = np.asarray(_spacing(ts.affine), float)
    orig = AQ._vt_organ_fast
    AQ._vt_organ_fast = lambda pdir, stem, shape: orig(pdir, stem, full)[sl]   # VT organs read full-size, cropped one at a time
    try:
        r = check(kb, md, names, task, aff, sp, plan, manifest, plan_dir, image=X, modality=modality, thresholds=thresholds)
    finally:
        AQ._vt_organ_fast = orig
    r['task'] = task
    return r, tuple(int(x) for x in lo), tuple(int(x) for x in full), np.asarray(ts.affine)


def _pack(prefix, box, out):
    if box is None or box.crop is None or not box.crop.any():
        return False
    out[prefix + '_sl'] = np.array([[s.start, s.stop] for s in box.sl], np.int32)
    out[prefix + '_shape'] = np.array(box.crop.shape, np.int32)
    out[prefix] = np.packbits(box.crop.ravel())
    return True


def save(r, path_stem, shape, affine, extra=None, offset=(0, 0, 0)):
    """path_stem.npz (packed masks) + path_stem.json (states). Keys in the npz: o<i>_core / _terr / _cont (+ _sl, _shape).
    shape / affine: the FULL image grid; offset: crop origin of the boxes in r (check_scan), stored and applied by load_mask."""
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    arr, organs = {}, []
    for i, (key, o) in enumerate(sorted(r['organs'].items(), key=lambda kv: (kv[0][0], str(kv[0][1])))):
        has = {nm: _pack(f'o{i}_{nm}', o[src], arr) for nm, src in (('core', 'core'), ('terr', 'territory'), ('cont', 'contested'))}
        vml = float(np.prod(np.linalg.norm(np.asarray(affine)[:3, :3], axis=0))) / 1000.0
        organs.append(dict(i=i, entity=key[0], side=key[1], mu=o['mu'], agents=o['agents'], fail=o['fail'], flags=o['flags'],
                           conflicting=o['conflicting'], unsegmentable=o.get('unsegmentable'),
                           core_ml=round(float(o['core'].sum()) * vml, 3) if o['core'].crop is not None else 0.0,
                           terr_ml=round(float(o['territory'].sum()) * vml, 3) if o['territory'].crop is not None else 0.0,
                           masks=has))
    np.savez_compressed(path_stem + '.npz', **arr)      # written in place: the mounts do not allow replacing (deleting) files
    meta = dict(version=r['version'], code_hash=r['code_hash'], shape=list(map(int, shape)), offset=list(map(int, offset)), affine=np.asarray(affine).tolist(),
                state=r['state'], reasons=r['reasons'], f_all=r['f_all'], n_graded=r['n_graded'],
                no_mu1_organ=r['no_mu1_organ'], consistency=r.get('consistency'), levels=r.get('levels'), noise=r.get('noise'),
                organs=organs, pairs=r['pairs'], expected=r['expected'], **(extra or {}))
    json.dump(meta, open(path_stem + '.json', 'w'), default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def load_mask(path_stem, i, kind='core'):
    """full-grid boolean mask of organ i (kind core / terr / cont) from a saved npz; None if empty"""
    meta = json.load(open(path_stem + '.json')); z = np.load(path_stem + '.npz')
    k = f'o{i}_{kind}'
    if k not in z:
        return None
    off = meta.get('offset') or [0, 0, 0]
    sl = tuple(slice(int(a) + o, int(b) + o) for (a, b), o in zip(z[k + '_sl'], off)); shp = tuple(z[k + '_shape'])
    m = np.zeros(meta['shape'], bool)
    m[sl] = np.unpackbits(z[k])[:int(np.prod(shp))].reshape(shp).astype(bool)
    return m
