"""Stage 3: lesion ledger and admission gate (set / topology reasoning, no statistics).

Inputs per scan (same relative name <rel> = <Mod>/<project>/<uid>/<series> under every root):
    TotalSegmentator/Radiology/<rel>_organ.nii.gz (+ _organ_labels.json)   organ reference
    BiomedParse/Radiology/<rel>_tumor.nii.gz                               rater BP (frozen)
    VoxTell/Radiology/<rel>_tumor.nii.gz                                   rater VT, initial cancer-type prompt
    VoxTell/Radiology/<rel>_plan/*.nii.gz + <rel>_plan_manifest.json       VT planned prompts (tier-T tumour + organ masks)
    Planner/Radiology/<rel>_plan.json                                      hosts, classes, priors, primary visibility

Rules (qc_rules.lesion_*; user decisions 22 + 28 Sep 2026):
  claims      connected components >= 0.5 ml of BP, VT-initial and every VT tumour prompt; VT-initial and VT-prompted are one
              model (VT). A lesion is a connected component of the union of the surviving claims.
  host ref H  TS organ mask U VT organ mask (planned '<entity>[_side]_organ'); KB prior region only when neither exists.
  shape gate  per claim against its host: slab (median in-plane cover >= 0.4 and aspect <= 0.25) -> dropped (2D artefact).
              organ-shaped (|L|/|H| >= 0.6 and median per-slice Dice >= 0.6): if TS and VT organ masks agree on the organ
              (Dice >= 0.7) -> dropped; otherwise kept only when the other model also claims the organ-shaped region, and the
              lesion is down-weighted (x0.3).
  admission   primary: two-model feasible lesion, or the largest feasible single-model lesion; extra lesions (multifocal) if
              two-model and feasible, or single-model, >= 80 % inside H and LD >= 10 mm. Feasible = touches H + 3 mm and >= 50 %
              inside H + 10 mm.
              local invasion: two-model lesion >= 50 % inside H + 3 mm; contiguous extension of an admitted primary lesion is
              recorded on the primary lesion (extends_into) and becomes a tumour-organ edge, not a separate node.
              distant: BP n VT-prompted-for-that-site >= 0.5 ml, >= 50 % inside H + 3 mm, prompt coherent (>= 80 % of the
              prompted mask inside H + 3 mm). VT-initial does not count here.
  weight      factor vector (support, host prior, inside fraction, size, shape) and their product.
v2 (29 Sep 2026, ledger gate audit; no model rerun, reasoning only):
  VT size floor   VT fragments inside the primary envelope are grouped (<= 5 mm apart) before the size floor, which is
                  0.2 ml there (0.5 ml elsewhere): VoxTell fragments small primaries.
  lesions         VoxTell-anchored: core = VT claims U BP voxels within 5 mm of them, grouped by 5 mm; the rest of BP forms
                  separate BP-only candidates (BP satellites no longer fuse onto VT lesions or bridge organs).
  assignment      primary if >= 50 % inside the primary envelope (as v1) OR attached: touches the primary organ (<= 3 mm) and
                  < 50 % of it lies inside any other organ (full TS label map). TS organ masks exclude
                  the tumour, so large / exophytic primaries sit mostly outside them (KIRC: 13 % of scans lost VT's largest
                  lesion, median 251 ml).
  confirmation    a lesion with VT voxels is confirmed; a BP-only lesion is confirmed when raw VT voxels (any size, in gate)
                  lie within 3 mm; otherwise it is admitted only as the scan's sole primary ('primary_bp_unconfirmed',
                  support 0.25) and never as an extra, and confirmed primaries are admitted first. Extras in a host need VT in
                  that host (paired organs: BP noise in the contralateral kidney is no longer a second primary).
v3 (2 Oct 2026, ledger v3 rule proposals; no model rerun; pan-cancer, constants fixed a priori; v2 = params r1=False, r3=False):
  R3 BP extent    the flat 5 mm VT shell is replaced by a slice and anatomy check: on every acquisition slice (largest
                  spacing axis if the spacing ratio is >= 1.5, else S-I) that holds VT, a BP 2D component touching VT
                  (8-connected, within 1 pixel) joins the VT lesion with its full in-plane extent; BP within 5 mm in 3D still
                  joins; BP voxels inside another TotalSegmentator organ (not the primary organ or its parts) never join and
                  stay separate BP candidates. BP's in-plane outline (gastric wall, breast) is kept; spread across slices or
                  into neighbouring organs is not.
  R1 VT silence   a BP-only primary lesion in a primary host where VoxTell produced no claim at all (vt_in_host False) is
                  not 'unconfirmed': VoxTell's silence there is missing evidence, not evidence against. Its support is
                  'BP_vtsilent' (weight 0.5, like any single-model lesion), it is ordered with confirmed lesions, and it may be
                  admitted as an extra lesion under the single-model nodule rule. Where VoxTell is active in the host and still
                  misses the lesion, 'BP' (unconfirmed) rules are unchanged.
Outputs: <rel>_lesions.json (every claim dropped and every lesion with status and reason) and <rel>_lesions.npz (cropped masks of
admitted lesions, with bounding boxes, for the graph stage)."""
from __future__ import annotations
import json, os
import numpy as np
from scipy import ndimage
from .adapter import _spacing, _envelope, _labels_by_side, host_prior_region

P = dict(min_ml=0.5, env_primary_mm=10.0, env_secondary_mm=3.0, contact_mm=3.0, feasible_inside=0.5, host_inside=0.5,
         slab_cover=0.4, slab_aspect=0.25, organ_vol_ratio=0.6, organ_dice2d=0.6, organ_inside=0.8, prompt_gate_inside=0.5, organ_agree_dice=0.7, organ_like_weight=0.3,
         extra_single_inside=0.8, extra_single_ld_mm=10.0, extra_single_min_slices=3, extra_single_z_mm=10.0, coherent_inside=0.8, size_saturation_ml=5.0,
         primary_min_ml=0.2, vt_group_mm=5.0, bp_attach_mm=5.0, attach_other_max=0.5, vt_trace_ml=0.05, trace_mm=3.0,
         r1=True, r3=True, slice_touch_px=1, acq_ratio=1.5)
VERSION = 'v3'
SUPPORT = {'BP+VT': 1.0, 'VT': 0.5, 'BP+VTtrace': 0.5, 'BP_vtsilent': 0.5, 'BP': 0.25}
PRIOR = dict(primary=1.0, local_invasion=0.3, distant=0.15)


class Box:
    """A boolean mask stored as its bounding box (memory: host references and envelopes of large CTs). box[sl] returns the
    mask on any slice tuple of the full grid; outside the box it is False."""

    def __init__(self, mask, shape, pad_mm=0.0, spacing=None):
        self.shape = shape
        if mask is None or not mask.any():
            self.sl, self.crop = None, None
            return
        sl = ndimage.find_objects(mask.astype(np.int8))[0]
        pad = [int(np.ceil(pad_mm / float(q))) + 1 for q in spacing] if pad_mm else [0, 0, 0]
        self.sl = tuple(slice(max(0, a.start - p), min(n, a.stop + p)) for a, p, n in zip(sl, pad, shape))
        self.crop = mask[self.sl].copy()

    def any(self):
        return self.crop is not None and bool(self.crop.any())

    def sum(self):
        return 0 if self.crop is None else int(self.crop.sum())

    def __getitem__(self, sl):
        sl = tuple(slice(0, n) if (s is None or s == slice(None)) else s for s, n in zip(sl, self.shape))
        out = np.zeros(tuple(s.stop - s.start for s in sl), bool)
        if self.crop is None:
            return out
        lo = [max(a.start, b.start) for a, b in zip(self.sl, sl)]
        hi = [min(a.stop, b.stop) for a, b in zip(self.sl, sl)]
        if any(l >= h for l, h in zip(lo, hi)):
            return out
        out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, sl))] = \
            self.crop[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, self.sl))]
        return out

    def dilate(self, mm, spacing):
        """Box of this mask dilated by mm (EDT on the padded box only)."""
        b = Box(None, self.shape)
        if self.crop is None:
            return b
        pad = [int(np.ceil(mm / float(q))) + 1 for q in spacing]
        b.sl = tuple(slice(max(0, a.start - p), min(n, a.stop + p)) for a, p, n in zip(self.sl, pad, self.shape))
        src = self[b.sl]
        b.crop = ndimage.distance_transform_edt(~src, sampling=spacing) <= mm
        return b


def _load(path):
    import nibabel as nib
    return nib.load(path)


def _bool(img):
    return np.asarray(img.dataobj) > 0


def _dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 0.0


def _claims(mask, rater, source, min_vox, small=None):
    """Connected components >= min_vox as (rater, source, bbox, crop). With `small` (region, min_vox, group_mm, spacing), the
    components below min_vox lying mostly in the region are grouped when <= group_mm apart and a group >= small min_vox is
    one claim (a fragmented VoxTell primary)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    out, tiny = [], []
    for j, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        crop = lab[sl] == j
        v = int(crop.sum())
        if v >= min_vox:
            out.append(dict(rater=rater, source=source, sl=sl, crop=crop, vox=v))
        elif small is not None:
            tiny.append((sl, crop))
    if tiny:
        reg, sp = small['region'], small['spacing']
        tm = np.zeros(mask.shape, bool)
        for sl, crop in tiny:
            if 2 * int((crop & reg[sl]).sum()) >= int(crop.sum()):
                tm[sl] |= crop
        if tm.any():
            box = ndimage.find_objects(tm.astype(np.int8))[0]
            pad = [int(np.ceil(small['group_mm'] / float(q))) + 1 for q in sp]
            box = tuple(slice(max(0, a.start - q), min(nn, a.stop + q)) for a, q, nn in zip(box, pad, mask.shape))
            sub = tm[box]
            glab, _ = ndimage.label(ndimage.distance_transform_edt(~sub, sampling=sp) <= small['group_mm'] / 2, structure=np.ones((3, 3, 3)))
            for g, gsl in enumerate(ndimage.find_objects(glab), 1):
                if gsl is None:
                    continue
                crop = (glab[gsl] == g) & sub[gsl]
                v = int(crop.sum())
                if v >= small['min_vox']:
                    osl = tuple(slice(b.start + q.start, b.start + q.stop) for b, q in zip(box, gsl))
                    out.append(dict(rater=rater, source=source, sl=osl, crop=crop, vox=v, grouped=True))
    return out


def _near(L, sl, shape, sp, mm):
    """(padded slice, mask within mm of L) on L's padded bounding box."""
    pad = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    ps = tuple(slice(max(0, a.start - q), min(n, a.stop + q)) for a, q, n in zip(sl, pad, shape))
    sub = np.zeros(tuple(b.stop - b.start for b in ps), bool)
    sub[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(sl, ps))] = L
    return ps, ndimage.distance_transform_edt(~sub, sampling=sp) <= mm


def _shape(c, H, k, ip, sp, vox_ml):
    """Slab / organ-likeness of a claim against host reference H on the axial axis k."""
    sl, L = c['sl'], c['crop']
    ext = [(sl[a].stop - sl[a].start) * sp[a] for a in ip]
    diag = float(np.hypot(*ext))
    z_ext = (sl[k].stop - sl[k].start) * sp[k]
    covers, dices = [], []
    Hc = H[sl]
    for z in range(L.shape[k]):
        Lz = np.take(L, z, axis=k)
        if not Lz.any():
            continue
        zs = [slice(None)] * 3
        zs[k] = slice(z + sl[k].start, z + sl[k].start + 1)
        hz = int(H[tuple(zs)].sum())
        if hz == 0:
            continue
        inter = int((Lz & np.take(Hc, z, axis=k)).sum())
        covers.append(inter / hz)
        dices.append(2 * inter / (hz + int(Lz.sum())))
    hv = H.sum()
    return dict(inside_H=round(int((L & Hc).sum()) / c['vox'], 3), ld_mm=round(diag, 1), aspect=round(z_ext / diag, 3) if diag else None,
                cover_med=round(float(np.median(covers)), 3) if covers else None,
                dice2d_med=round(float(np.median(dices)), 3) if dices else None,
                vol_ratio=round(c['vox'] / hv, 3) if hv else None)


def build_ledger(rel, seg_root, kb, params=None):
    p = dict(P, **(params or {}))
    R = lambda *a: os.path.join(seg_root, *a)
    plan = json.load(open(R('Planner', 'Radiology', rel + '_plan.json')))
    man_path = R('VoxTell', 'Radiology', rel + '_plan_manifest.json')
    manifest = json.load(open(man_path))
    plan_dir = man_path[:-len('_manifest.json')]
    ts_img = _load(R('TotalSegmentator', 'Radiology', rel + '_organ.nii.gz'))
    md = np.asarray(ts_img.dataobj).astype(np.int16)
    meta = json.load(open(R('TotalSegmentator', 'Radiology', rel + '_organ_labels.json')))
    task = meta.get('task') or 'total'
    names = {int(k): v for k, v in (meta.get('labels') or {}).items()}
    sp = _spacing(ts_img.affine)
    vox_ml = float(np.prod(sp)) / 1000.0
    min_vox = max(1, int(round(p['min_ml'] / vox_ml)))
    import nibabel as nib
    k = [i for i, c in enumerate(nib.aff2axcodes(ts_img.affine)) if c in 'SI'][0]
    ip = [i for i in range(3) if i != k]
    shape = md.shape
    primary_visible = not any(l.get('step') == 'primary_not_in_fov' for l in plan.get('log', []))

    # ---- hosts from the tier-T prompts, with their reference H
    hosts = {}
    for m in manifest['masks']:
        if m['list'] != 'T':
            continue
        key = (m['entity'], m.get('side'))
        h = hosts.setdefault(key, dict(entity=m['entity'], side=m.get('side'), cls=m.get('host_class'),
                                       prior=m.get('host_prior'), prompt_files=[]))
        h['prompt_files'].append(m['file'])
    lbs = {}
    for (ent, side), h in hosts.items():
        if ent not in lbs:
            lbs[ent] = _labels_by_side(kb, task, names, ent)
        labs = [l for sd, ls in lbs[ent].items() for l in ls if side is None or sd in (side, None)]
        ts_m = np.isin(md, labs) if labs else np.zeros(shape, bool)
        vt_m = _vt_organ(plan_dir, f"{ent}{'_' + side if side else ''}_organ", shape)
        H, src = ts_m | vt_m, ('TS+VT' if ts_m.any() and vt_m.any() else 'TS' if ts_m.any() else 'VT' if vt_m.any() else None)
        if src is None and h['cls'] == 'primary':
            # a primary organ neither TS nor VoxTell can delineate (e.g. cervix on CT): the coarse KB prior box, flagged. A
            # secondary host without any organ mask gets no reference: nothing can be assigned to it or measured against it
            H, _ = host_prior_region(kb, ent, md, names, task, sp)
            src = 'kb_prior_region' if H is not None else None
        h.update(ref=src, ts_ml=round(float(ts_m.sum()) * vox_ml, 1), vt_organ_ml=round(float(vt_m.sum()) * vox_ml, 1),
                 organ_agree=(round(_dice(ts_m, vt_m), 3) if ts_m.any() and vt_m.any() else None))
        h['H'] = Box(H, shape)
        del ts_m, vt_m, H
        h['env'] = h['H'].dilate(p['env_primary_mm'] if h['cls'] == 'primary' else p['env_secondary_mm'], sp)
        h['env3'] = h['env'] if h['cls'] != 'primary' else h['H'].dilate(p['contact_mm'], sp)

    # ---- claims (v2: VT fragments in the primary envelope grouped before a 0.2 ml floor; raw VT kept as evidence)
    prim = [key for key, h in hosts.items() if h['cls'] == 'primary']
    prim_region = np.zeros(shape, bool)
    for q in prim:
        e = hosts[q]['env']
        if e.any():
            prim_region[e.sl] |= e.crop
    min_vox_small = max(1, int(round(p['primary_min_ml'] / vox_ml)))
    small = dict(region=prim_region, min_vox=min_vox_small, group_mm=p['vt_group_mm'], spacing=sp) if prim_region.any() else None
    claims = []
    vt_raw = np.zeros(shape, bool)      # every VT voxel (initial anywhere, prompts inside their gate), before any size floor
    bp_p = R('BiomedParse', 'Radiology', rel + '_tumor.nii.gz'); vt_p = R('VoxTell', 'Radiology', rel + '_tumor.nii.gz')
    if os.path.exists(bp_p):
        claims += _claims(_bool(_load(bp_p)), 'BP', 'bp', min_vox)
    if os.path.exists(vt_p):
        m = _bool(_load(vt_p))
        vt_raw |= m
        claims += _claims(m, 'VT', 'vt_initial', min_vox, small)
    for key, h in hosts.items():
        for f in h['prompt_files']:
            fp = os.path.join(plan_dir, f)
            if os.path.exists(fp):
                m = _bool(_load(fp))
                if h['env'].any():
                    vt_raw[h['env'].sl] |= m[h['env'].sl] & h['env'].crop
                claims += [dict(c, host=key) for c in _claims(m, 'VT', 'vt_prompt:' + f[:-7], min_vox, small if h['cls'] == 'primary' else None)]

    # reference host of every claim: its prompt host, else the host whose envelope holds most of it
    def best_host(c):
        if c.get('host'):
            return c['host']
        fr = {key: int((c['crop'] & h['env'][c['sl']]).sum()) / c['vox'] for key, h in hosts.items()}
        key = max(fr, key=fr.get) if fr else None
        return key if key and fr[key] > 0 else None

    dropped, organ_like = [], []
    for c in claims:
        c['ref_host'] = best_host(c)
        h = hosts.get(c['ref_host'])
        c['shape'] = _shape(c, h['H'], k, ip, sp, vox_ml) if h is not None and h['H'].any() else {}
        s_ = c['shape']
        c['drop'], c['organ_like'] = None, False
        # the prompt's gate (stage-2 output filter, applied here): a prompted tumour claim must lie in its host region —
        # VoxTell answers 'X tumor' with the most salient tumour anywhere (KIRC: 'left lung tumor' returned the kidney mass)
        if c.get('host') is not None:
            g = hosts[c['host']]['env']
            if int((c['crop'] & g[c['sl']]).sum()) / c['vox'] < p['prompt_gate_inside']:
                c['drop'] = 'outside_prompt_gate'
                continue
        if s_.get('cover_med') is not None and s_['cover_med'] >= p['slab_cover'] and (s_['aspect'] or 1) <= p['slab_aspect']:
            c['slab'] = True
        elif ((s_.get('vol_ratio') or 0) >= p['organ_vol_ratio'] and (s_.get('dice2d_med') or 0) >= p['organ_dice2d']
              and (s_.get('inside_H') or 0) >= p['organ_inside']):
            if (h.get('organ_agree') or 0) >= p['organ_agree_dice']:
                c['drop'] = 'organ_shaped_organ_agreed'        # TS and VT agree on the organ: the claim is the organ
            else:
                organ_like.append(c)
    surviving = [c for c in claims if not c['drop']]
    # slab: a slice-wise 2D artefact unless the other (3D) model claims the same voxels
    for c in surviving:
        if c.get('slab') and not any(o['rater'] != c['rater'] and int((_crop_to(o, c['sl']) & c['crop']).sum()) >= min_vox
                                     for o in surviving if not o.get('slab')):
            c['drop'] = 'slab_2d_artefact'
    # organ-shaped without an agreed organ reference: kept (down-weighted) on the primary host, where tumour may replace the
    # organ; elsewhere only when the other model also claims the organ-shaped region
    for c in organ_like:
        if c['drop']:
            continue
        other = any(o['rater'] != c['rater'] and o['ref_host'] == c['ref_host'] and o in organ_like and not o['drop']
                    and int((_crop_to(o, c['sl']) & c['crop']).sum()) >= min_vox for o in organ_like)
        if other or hosts[c['ref_host']]['cls'] == 'primary':
            c['organ_like'] = True
        else:
            c['drop'] = 'organ_shaped_single_model'
    for c in claims:
        if c['drop']:
            dropped.append(dict(rater=c['rater'], source=c['source'], host=_hk(c['ref_host']), ml=round(c['vox'] * vox_ml, 2),
                                reason=c['drop'], **c['shape']))
    keep = [c for c in claims if not c['drop']]

    # organ tissue other than the primary: full TS label map minus the primary organ(s), their inside-parts and satellites
    # (kidney_cyst). TS only: a VoxTell organ prompt near a tumour often returns the tumour itself (COAD: 'uterus' = the mass)
    prim_ents = set()
    for q in prim:
        todo = [q[0]]
        while todo:
            e = todo.pop()
            if e in prim_ents:
                continue
            prim_ents.add(e)
            todo += [r['dst'] for r in kb.out_edges.get(e, []) if r['type'] == 'has_part' and r.get('spatial', 'inside') in ('inside', 'encloses')]
    prim_ents |= {e for e in kb.entities if any(e.startswith(b + '_') for b in {q[0] for q in prim})}
    prim_sides = {q[1] for q in prim}
    ts_map = kb.ts_to_entity.get(task, {})
    prim_labels = [lab_ for lab_, nm in names.items() if ts_map.get(nm, (None,))[0] in prim_ents
                   and (None in prim_sides or (ts_map.get(nm, (None, None)) + (None,))[1] in prim_sides | {None})]
    other = (md > 0) & ~np.isin(md, prim_labels)

    # ---- lesions (v2): VoxTell-anchored. core = VT claims U BP voxels within bp_attach_mm of them, grouped by vt_group_mm;
    # the remaining BP voxels form separate BP-only candidates
    vt_u = np.zeros(shape, bool); bp_u = np.zeros(shape, bool)
    for c in keep:
        (bp_u if c['rater'] == 'BP' else vt_u)[c['sl']] |= c['crop']
    near = _envelope(vt_u, sp, p['bp_attach_mm']) if vt_u.any() else np.zeros(shape, bool)
    if p['r3'] and vt_u.any() and bp_u.any():
        # v3 R3: BP joins a VT lesion by in-plane connection on the acquisition slices that hold VT, never inside another organ
        ax = int(np.argmax(sp)) if float(max(sp)) / float(min(sp)) >= p['acq_ratio'] else k
        st2 = np.ones((3, 3), bool); add = np.zeros(shape, bool)
        for z in np.where(vt_u.any(axis=tuple(a for a in range(3) if a != ax)))[0]:
            ix = [slice(None)] * 3; ix[ax] = int(z); ix = tuple(ix)
            b = bp_u[ix] & ~other[ix]
            if not b.any():
                continue
            lab2, _ = ndimage.label(b, structure=st2)
            v = ndimage.binary_dilation(vt_u[ix], structure=st2, iterations=p['slice_touch_px'])
            ids = np.unique(lab2[v & (lab2 > 0)])
            if ids.size:
                add[ix] |= np.isin(lab2, ids)
        core = vt_u | (bp_u & near & ~other) | add
        del add
    else:
        core = vt_u | (bp_u & near)
    bp_rest = bp_u & ~core
    del near
    parts = []
    if core.any():
        glab, _ = ndimage.label(_envelope(core, sp, p['vt_group_mm'] / 2), structure=np.ones((3, 3, 3)))
        for j, sl in enumerate(ndimage.find_objects(glab), 1):
            if sl is not None:
                L = (glab[sl] == j) & core[sl]
                if int(L.sum()) >= min_vox_small:
                    parts.append((sl, L))
        del glab
    lab, _ = ndimage.label(bp_rest, structure=np.ones((3, 3, 3)))
    for j, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is not None:
            L = lab[sl] == j
            if int(L.sum()) >= min_vox:
                parts.append((sl, L))
    del lab, bp_rest, core


    lesions = []
    for sl, L in parts:
        members = [c for c in keep if _overlaps(c, sl, L)]
        bp = bp_u[sl] & L; vt = vt_u[sl] & L; vtp = {}
        for c in members:
            if c['rater'] == 'VT' and c['source'].startswith('vt_prompt') and c.get('host'):
                vtp.setdefault(c['host'], np.zeros(L.shape, bool))
                vtp[c['host']] |= _crop_to(c, sl) & L
        vox = int(L.sum())
        fr = {key: int((L & h['env'][sl]).sum()) / vox for key, h in hosts.items()}
        ext = [(sl[a].stop - sl[a].start) * sp[a] for a in ip]
        n_sl = int(L.any(axis=tuple(a for a in range(3) if a != k)).sum())
        les = dict(sl=sl, L=L, n_slices=n_sl, z_mm=round(n_sl * float(sp[k]), 1), vox=vox, bp=bp, vt=vt, vtp=vtp, frac=fr,
                   agree=int((bp & vt).sum()), organ_like=any(c.get('organ_like') for c in members),
                   sources=sorted({c['source'] for c in members}), ld_mm=float(np.hypot(*ext)),
                   in_other=round(int((L & other[sl]).sum()) / vox, 3))
        les['two_model'] = les['agree'] >= min_vox
        if vt.any():
            les['support'] = 'BP+VT' if les['two_model'] else 'VT'
        else:
            ps, nb = _near(L, sl, shape, sp, p['trace_mm'])
            les['vt_trace_ml'] = round(int((nb & vt_raw[ps]).sum()) * vox_ml, 3)
            les['support'] = 'BP+VTtrace' if les['vt_trace_ml'] >= p['vt_trace_ml'] else 'BP'
        les['confirmed'] = les['support'] != 'BP'
        lesions.append(les)
    del other, vt_u, bp_u

    # ---- assignment (v2): inside the primary envelope, or attached to the primary organ and not inside another organ
    out = []
    for les in lesions:
        sl, L = les['sl'], les['L']
        pf = max((les['frac'].get(q, 0) for q in prim), default=0)
        touch = [q for q in prim if hosts[q]['env3'].any() and bool((L & hosts[q]['env3'][sl]).any())]
        les['assign'] = None
        if primary_visible and prim and pf >= p['feasible_inside']:
            key = max(prim, key=lambda q: les['frac'].get(q, 0)); les['assign'] = 'inside'
        elif primary_visible and touch and les['in_other'] < p['attach_other_max']:
            key = max(touch, key=lambda q: (int((L & hosts[q]['H'][sl]).sum()), les['frac'].get(q, 0))); les['assign'] = 'attached'
        else:
            cand = {q: f for q, f in les['frac'].items() if hosts[q]['cls'] != 'primary' and f >= p['host_inside']}
            key = max(cand, key=cand.get) if cand else None
        les['host'] = key
        les['cls'] = hosts[key]['cls'] if key else None
        out.append(les)
    # VT in each primary host (initial or prompted, any claim): a single-model extra lesion in a host needs VT evidence there
    vt_in_host = {q: any(c['rater'] == 'VT' and int((c['crop'] & hosts[q]['env'][c['sl']]).sum()) >= min_vox_small for c in keep)
                  for q in prim}
    vt_in_primary = any(vt_in_host.values())
    if p['r1']:
        # v3 R1: VoxTell silent in the host -> a BP-only primary is not 'unconfirmed' (missing evidence, not evidence against)
        for les in out:
            if les['support'] == 'BP' and les['cls'] == 'primary' and not vt_in_host.get(les['host']):
                les['support'], les['confirmed'] = 'BP_vtsilent', True
    spread = {q['entity']: q for q in kb.spread_hosts(prim[0][0], (plan.get('input') or {}).get('cancer_type'))} if prim else {}
    admitted_primary = []
    out.sort(key=lambda x: (x['cls'] != 'primary', not x['confirmed'], -x['vox']))
    prim_env = None
    for les in out:
        if les['cls'] != 'primary' and prim_env is None:
            prim_env = np.zeros(shape, bool)
            for a in admitted_primary:
                prim_env[a['sl']] |= a['L']
            prim_env = _envelope(prim_env, sp, p['contact_mm']) if prim_env.any() else prim_env
        h = hosts.get(les['host'])
        status, reason = 'rejected', None
        if h is None:
            reason = 'outside_spread_set'
        elif les['cls'] == 'primary':
            contact = bool((les['L'] & h['env3'][les['sl']]).any())
            inside_H = int((les['L'] & h['H'][les['sl']]).sum()) / les['vox']
            feasible = contact and (les['frac'][les['host']] >= p['feasible_inside'] or les['assign'] == 'attached')
            les['inside_H'] = round(inside_H, 3)
            if not feasible:
                reason = 'primary_infeasible'
            elif not admitted_primary:
                status = 'admitted'
                reason = {'BP+VT': 'primary_two_model', 'VT': 'primary_vt', 'BP+VTtrace': 'primary_bp_vt_trace',
                          'BP_vtsilent': 'primary_bp_vt_silent', 'BP': 'primary_bp_unconfirmed'}[les['support']]
            elif les['two_model']:
                status, reason = 'admitted', 'primary_multifocal_two_model'
            elif not vt_in_host.get(les['host']) and not (p['r1'] and les['support'] == 'BP_vtsilent'):
                reason = 'primary_extra_single_no_vt'
            elif not les['confirmed']:
                reason = 'primary_extra_unconfirmed'
            elif ((inside_H >= p['extra_single_inside'] or les['assign'] == 'attached') and les['ld_mm'] >= p['extra_single_ld_mm']
                  and les['n_slices'] >= p['extra_single_min_slices'] and les['z_mm'] >= p['extra_single_z_mm']):
                status, reason = 'admitted', 'primary_multifocal_single'      # a nodule, not a flat 2D fragment
            else:
                reason = 'primary_satellite_single'
            if status == 'admitted':
                admitted_primary.append(les)
        elif les['cls'] == 'local_invasion':
            # invasion = direct, contiguous growth (T staging): the lesion must touch an admitted primary lesion (<= 3 mm).
            # A separate lesion in a neighbour is a metastasis: judged by the distant rule where the organ is a metastatic
            # site of the cancer, rejected otherwise
            contiguous = prim_env is not None and bool((les['L'] & prim_env[les['sl']]).any())
            les['contiguous_with_primary'] = contiguous
            if contiguous and les['two_model']:
                status, reason = 'admitted', 'local_two_model_contiguous'
            elif contiguous:
                reason = 'local_single_model'
            elif spread.get(les['host'][0], {}).get('also_distant'):
                les['cls'] = 'distant'
            else:
                reason = 'local_not_contiguous'
        if les['cls'] == 'distant' and reason is None:
            vp = les['vtp'].get(les['host'])
            agree_site = int((les['bp'] & vp).sum()) if vp is not None else 0
            coh = _coherence(les['host'], hosts, keep, shape)
            les['site_agree_ml'] = round(agree_site * vox_ml, 2); les['prompt_coherence'] = coh
            if agree_site < min_vox:
                reason = 'distant_needs_bp_and_site_prompt'
            elif coh is not None and coh < p['coherent_inside']:
                reason = 'distant_prompt_incoherent'
            else:
                status, reason = 'admitted', 'distant_bp_and_site_prompt'
        les['status'], les['reason'] = status, reason
    # contiguous extension of admitted primary lesions into local-invasion neighbours
    for les in admitted_primary:
        ph = hosts[les['host']]
        beyond = les['L'] & ~ph['env'][les['sl']]
        les['extends_into'] = {_hk(q): round(int((beyond & h['env3'][les['sl']]).sum()) * vox_ml, 2)
                               for q, h in hosts.items() if h['cls'] == 'local_invasion' and h['ref'] in ('TS', 'VT', 'TS+VT')
                               and int((beyond & h['env3'][les['sl']]).sum()) >= min_vox}
    # weights
    for les in out:
        ml = les['vox'] * vox_ml
        f = dict(support=SUPPORT[les['support']],
                 host_prior=PRIOR.get(les['cls'], 0.0),
                 inside=round(les['frac'].get(les['host'], 0.0), 3) if les['host'] else 0.0,
                 size=round(min(1.0, ml / p['size_saturation_ml']), 3),
                 shape=p['organ_like_weight'] if les['organ_like'] else 1.0)
        les['factors'] = f
        les['weight'] = round(float(np.prod(list(f.values()))), 4)

    def row(i, les):
        return dict(id=i, status=les['status'], reason=les['reason'], host=_hk(les['host']), cls=les['cls'],
                    ml=round(les['vox'] * vox_ml, 2), bp_ml=round(int(les['bp'].sum()) * vox_ml, 2),
                    vt_ml=round(int(les['vt'].sum()) * vox_ml, 2), agree_ml=round(les['agree'] * vox_ml, 2),
                    two_model=les['two_model'], support=les['support'], confirmed=les['confirmed'], vt_trace_ml=les.get('vt_trace_ml'),
                    assign=les.get('assign'), in_other=les['in_other'], organ_like=les['organ_like'], ld_mm=round(les['ld_mm'], 1), n_slices=les['n_slices'], z_mm=les['z_mm'],
                    inside_H=les.get('inside_H'), site_agree_ml=les.get('site_agree_ml'), prompt_coherence=les.get('prompt_coherence'),
                    extends_into=les.get('extends_into'), contiguous_with_primary=les.get('contiguous_with_primary'), sources=les['sources'], factors=les['factors'], weight=les['weight'],
                    bbox=[[s.start, s.stop] for s in les['sl']],
                    frac={_hk(q): round(v, 3) for q, v in les['frac'].items() if v > 0})
    rows = [row(i, les) for i, les in enumerate(out)]
    host_rows = [dict(host=_hk(q), cls=h['cls'], prior=h['prior'], ref=h['ref'], ts_ml=h['ts_ml'], vt_organ_ml=h['vt_organ_ml'],
                      organ_agree=h['organ_agree']) for q, h in hosts.items()]
    ledger = dict(rel=rel, version=VERSION, primary_visible=primary_visible, spacing=[float(x) for x in sp], params=p, hosts=host_rows,
                  vt_in_primary=vt_in_primary if prim else None, vt_in_host={_hk(q): v for q, v in vt_in_host.items()}, n_claims=len(claims), dropped_claims=dropped, lesions=rows,
                  n_admitted=sum(r['status'] == 'admitted' for r in rows))
    crops = {f'lesion_{i}': les['L'] for i, les in enumerate(out) if les['status'] == 'admitted'}
    return ledger, crops, ts_img.affine, shape


def _vt_organ(plan_dir, stem, shape):
    """VoxTell organ mask for '<stem>': main phrase, or the voxel majority of main + aliases when aliases were prompted."""
    files = [os.path.join(plan_dir, stem + sfx + '.nii.gz') for sfx in ('', '_a2', '_a3')]
    ms = [_bool(_load(f)) for f in files if os.path.exists(f)]
    if not ms:
        return np.zeros(shape, bool)
    if len(ms) == 1:
        return ms[0]
    return np.sum(ms, axis=0) >= (len(ms) // 2 + 1)


def _hk(key):
    if not key:
        return None
    return key[0] + (f'_{key[1]}' if key[1] else '')


def _overlaps(c, sl, L):
    lo = [max(a.start, b.start) for a, b in zip(c['sl'], sl)]
    hi = [min(a.stop, b.stop) for a, b in zip(c['sl'], sl)]
    if any(l >= h for l, h in zip(lo, hi)):
        return False
    cc = c['crop'][tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, c['sl']))]
    ll = L[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, sl))]
    return bool((cc & ll).any())


def _crop_to(c, sl):
    out = np.zeros(tuple(s.stop - s.start for s in sl), bool)
    lo = [max(a.start, b.start) for a, b in zip(c['sl'], sl)]
    hi = [min(a.stop, b.stop) for a, b in zip(c['sl'], sl)]
    if any(l >= h for l, h in zip(lo, hi)):
        return out
    out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, sl))] = \
        c['crop'][tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, c['sl']))]
    return out


def _coherence(key, hosts, keep, shape):
    """Fraction of the host's prompted tumour mask (surviving claims) inside its envelope H + 3 mm."""
    cl = [c for c in keep if c.get('host') == key]
    tot = sum(c['vox'] for c in cl)
    if not tot:
        return None
    env = hosts[key]['env3']
    return round(sum(int((c['crop'] & env[c['sl']]).sum()) for c in cl) / tot, 3)


def save_ledger(ledger, crops, affine, shape, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with open(out_prefix + '_lesions.json', 'w') as f:
        json.dump(ledger, f, indent=1, default=str)
    boxes = {r['id']: r['bbox'] for r in ledger['lesions'] if r['status'] == 'admitted'}
    np.savez_compressed(out_prefix + '_lesions.npz', affine=np.asarray(affine), shape=np.asarray(shape),
                        **crops, **{f'bbox_{i}': np.asarray(b) for i, b in boxes.items()})
