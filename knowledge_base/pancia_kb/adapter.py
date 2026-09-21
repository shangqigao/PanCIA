"""Adapters from PanCIA on-disk segmentation outputs to the planner's TSOutput.

Reads the files written by ``extract_TotalSegmentator_segmentation`` (m_tumor_segmentation.py):
  <img>_<seg_obj>.nii.gz          multi-label TS mask (only needed for the tumour-host guess)
  <img>_<seg_obj>_labels.json     {task, fast, modality, site, labels}
  <img>_<seg_obj>_structures.csv  label, name, n_voxels, volume_ml, centroid_{x,y,z}_mm, zmin_mm, zmax_mm, touches_boundary

Optionally a tumour mask (BiomedParse / VoxTell NIfTI) gives the tumour centroid and host organ.
Heavy imports (nibabel, numpy) are local so the KB package stays importable without them.
"""
from __future__ import annotations
import csv, json, os
from typing import Optional, Tuple
from .planner import TSOutput, TSStructure
from .kb import KnowledgeBase


def _spacing(affine):
    """Voxel size per array axis = column norms of the affine (valid for oblique / coronal / sagittal acquisitions,
    where the affine diagonal can be zero)."""
    import numpy as np
    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ('1', 'true', 't', 'yes')


def fov_z_from_nifti(path: str) -> Tuple[float, float]:
    """Inferior/superior extent of the volume in RAS mm from header only (no voxel data read)."""
    import numpy as np, nibabel as nib
    img = nib.load(path)
    shape = np.asarray(img.shape[:3]) - 1
    corners = np.array([[i, j, k, 1.0] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])])
    z = (img.affine @ corners.T)[2]
    return float(z.min()), float(z.max())


def fov_z_from_table(rows) -> Tuple[float, float]:
    """Fallback FOV: union of structure z-extents (underestimates the true FOV; flagged in the log)."""
    zs0 = [float(r['zmin_mm']) for r in rows]
    zs1 = [float(r['zmax_mm']) for r in rows]
    return (min(zs0), max(zs1)) if rows else (0.0, 0.0)


def _host_ts_labels(kb, task, label_names, host):
    """TS label ids (all sides) that map to the host entity."""
    ts_map = kb.ts_to_entity.get(task, {})
    return [lab for lab, name in label_names.items() if ts_map.get(name, (None,))[0] == host]


def host_prior_region(kb, host, md, label_names, task, spacing, expand_mm=30.0, depth=2):
    """Validation region for a host WITHOUT a TS class (cervix, uterus, ovary, breast, rectum …), built from the KB:
    the host's spatial-prior landmarks, followed transitively (cervix → uterus → bladder / hips / sacrum), plus the
    members of the region the host lies in; those that TS did segment define a bounding box, expanded by expand_mm.
    Returns (bool mask | None, list of landmark entities used)."""
    import numpy as np
    ts_map = kb.ts_to_entity.get(task, {})
    ent_to_labels = {}
    for lab, name in label_names.items():
        ent = ts_map.get(name)
        if ent:
            ent_to_labels.setdefault(ent[0], []).append(lab)
    # transitive prior landmarks
    frontier, seen = {host}, set()
    for _ in range(depth):
        nxt = set()
        for e in frontier:
            for p in kb.spatial_prior_for(e):
                lm = p['landmark']
                lm_ent = kb.landmarks[lm]['entity'] if lm in kb.landmarks else lm
                if lm_ent not in seen:
                    seen.add(lm_ent); nxt.add(lm_ent)
        frontier = nxt
    # region members
    region = next((r['dst'] for r in kb.out_edges.get(host, []) if r['type'] == 'lies_in'), None)
    if region:
        for r in kb.relations:
            if r['type'] == 'lies_in' and r['dst'] == region:
                seen.add(r['src'])
    used, labs = [], []
    for e in seen:
        if e in ent_to_labels:
            used.append(e); labs += ent_to_labels[e]
    if not labs:
        return None, []
    m = np.isin(md, labs)
    if not m.any():
        return None, []
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0)
    pad = np.ceil(expand_mm / spacing).astype(int)
    lo = np.maximum(lo - pad, 0); hi = np.minimum(hi + pad, np.asarray(md.shape) - 1)
    box = np.zeros(md.shape, bool)
    box[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
    return box, sorted(used)


def host_candidates_evidence(kb, candidates, masks, names, md, label_names, task, spacing, envelope_mm=10.0, min_inside=0.5, min_component_ml=0.5):
    """Data-driven update of the host prior: for each candidate host (KB.spread_hosts) present in the scan, count the
    set-based evidence that a validated tumour component lies in its region — 2 agreement, 1 one rater, 0 none.
    Candidates with a TS class use the TS envelope; candidates without one use the KB prior region; candidates with
    neither region (not in FOV / no landmarks) get evidence None and keep their prior only.
    weight = prior × (1 + evidence). Returns the candidate list with evidence, weight, in_fov, region kind."""
    import numpy as np
    from scipy import ndimage
    nonempty = [m for m in masks if m.any()]
    inter = None
    if len(nonempty) > 1:
        inter = nonempty[0].copy()
        for m in nonempty[1:]:
            inter &= m
        if not inter.any():
            inter = None
    vox_ml = float(np.prod(spacing)) / 1000.0
    min_vox = max(1, int(round(min_component_ml / vox_ml)))
    comp_cache = []
    for m in nonempty:
        lab, k = ndimage.label(m, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel())
        comp_cache.append((lab, [j for j in range(1, k + 1) if sizes[j] >= min_vox]))   # only measurable components count as evidence
    if inter is not None and inter.sum() < min_vox:
        inter = None
    out = []
    for c in candidates:
        c = dict(c)
        labs = _host_ts_labels(kb, task, label_names, c['entity'])
        region, kind = None, None
        hm = np.isin(md, labs) if labs else None
        if hm is not None and hm.any():
            region, kind = ndimage.distance_transform_edt(~hm, sampling=spacing) <= envelope_mm, 'ts_envelope'
        elif not labs and c['cls'] == 'primary':
            # the KB prior box is coarse: acceptable for the primary (no alternative), never as evidence for a secondary host
            region, used = host_prior_region(kb, c['entity'], md, label_names, task, spacing)
            kind = 'kb_prior_region' if region is not None else None
        c['in_fov'] = region is not None
        c['region'] = kind
        if region is None:
            c['evidence'] = None
            c['weight'] = round(c['prior'], 3)
            out.append(c); continue
        ev = 0
        if inter is not None and float((inter & region).sum()) / float(inter.sum()) >= min_inside:
            ev = 2
        else:
            for lab, comps in comp_cache:
                hit = False
                for j in comps:
                    comp = lab == j
                    if float((comp & region).sum()) / float(comp.sum()) >= min_inside:
                        hit = True; break
                ev += int(hit)
            ev = min(ev, 1) if ev else 0            # single-rater support counts once; two raters' separate components ≠ agreement
        c['evidence'] = ev
        c['weight'] = round(c['prior'] * (1 + ev), 3)
        out.append(c)
    return out


def tumour_summary(tumour_mask_path, ts_mask_path: Optional[str], kb: Optional[KnowledgeBase], task: str, label_names: dict,
                   expected_host: Optional[str] = None, envelope_mm: float = 10.0, min_inside: float = 0.5,
                   cancer_type: Optional[str] = None, min_component_ml: float = 0.5) -> Tuple[Optional[Tuple[float, float, float]], Optional[str], dict]:
    """Validated tumour evidence for the planner.

    BP and VT disagree completely on ~60 % of series, so a raw centroid is unreliable. Evidence is therefore taken
    only from tumour components that are anatomically consistent with the EXPECTED host (cancer type / site):
      1. agreement  A = ∩ of all non-empty raters, if ≥ min_inside of A lies in the host envelope (host TS mask
                    dilated by envelope_mm);
      2. rater      else, per rater in the given order (VT first), the union of its connected components that lie
                    ≥ min_inside inside the envelope;
      3. none       no validated component → centroid None; the planner then keeps the cancer-type host, prompts a
                    bilateral host on both sides and skips tumour-near anchors.
    If the expected host has no TS class (cervix, uterus, ovary, breast, rectum …) the envelope is replaced by the
    KB-derived host prior region (bounding box of the TS-found landmarks of the host's spatial prior, followed
    transitively, and of its body region; expanded 30 mm) — evidence tags become 'prior_region_*'. Only when neither
    exists (no TS mask at all) is the evidence 'unverified'.
    The expected host is never replaced by rater evidence that fails the envelope: a lesion both raters place in
    another organ is recorded (`raters_elsewhere`, possible metastasis or leakage) and the cancer-type host is kept.
    Without an expected host (unknown cancer type) the host is the anchor with maximal overlap.
    Returns (centroid RAS mm | None, host entity | None, diagnostics)."""
    import numpy as np, nibabel as nib
    from scipy import ndimage
    paths = [tumour_mask_path] if isinstance(tumour_mask_path, str) else [q for q in tumour_mask_path if q]
    ref = nib.load(paths[0])
    masks, names = [], []
    for i, q in enumerate(paths):
        m = nib.load(q)
        if m.shape[:3] != ref.shape[:3]:
            continue
        masks.append(np.asarray(m.dataobj) > 0); names.append(f'rater_{i}')
    diag = dict(tumour_raters=len(paths), tumour_raters_nonempty=int(sum(m.any() for m in masks)), expected_host=expected_host)
    if not any(m.any() for m in masks):
        diag.update(tumour_voxels=0, tumour_evidence='none')
        return None, None, diag

    # host envelope from TS
    envelope, md = None, None
    if ts_mask_path and os.path.exists(ts_mask_path) and kb is not None:
        tsimg = nib.load(ts_mask_path)
        if tsimg.shape[:3] == ref.shape[:3] and np.allclose(tsimg.affine, ref.affine, atol=1e-2):
            md = np.asarray(tsimg.dataobj)
            if expected_host:
                labs = _host_ts_labels(kb, task, label_names, expected_host)
                hm = np.isin(md, labs) if labs else None
                spacing = _spacing(ref.affine)
                if hm is not None and hm.any():
                    envelope = ndimage.distance_transform_edt(~hm, sampling=spacing) <= envelope_mm
                    diag['host_ts_volume_ml'] = round(float(hm.sum() * np.prod(spacing) / 1000), 1)
                    diag['validation_region'] = 'host_ts_envelope'
                else:
                    envelope, used = host_prior_region(kb, expected_host, md, label_names, task, spacing)
                    if envelope is not None:
                        diag['validation_region'] = 'kb_prior_region'; diag['prior_region_landmarks'] = used
        else:
            diag['host_note'] = 'tumour and TS grids differ'
    diag['host_envelope'] = envelope is not None

    def inside_frac(m):
        n = int(m.sum())
        return float((m & envelope).sum()) / n if (envelope is not None and n) else None

    vox_ml = float(np.prod(_spacing(ref.affine))) / 1000.0
    min_vox = max(1, int(round(min_component_ml / vox_ml)))

    def validated_components(m):
        lab, k = ndimage.label(m, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel())
        keep = np.zeros_like(m)
        n_keep = 0
        for c in range(1, k + 1):
            if sizes[c] < min_vox:                     # sub-measurable fragment: never evidence
                continue
            comp = lab == c
            if inside_frac(comp) >= min_inside:
                keep |= comp; n_keep += 1
        return keep, n_keep, k

    nonempty = [(n, m) for n, m in zip(names, masks) if m.any()]
    td, evidence = None, 'none'
    inter = None
    if len(nonempty) > 1:
        inter = nonempty[0][1].copy()
        for _, m in nonempty[1:]:
            inter &= m
        if not inter.any():
            inter = None
    diag['agreement_voxels'] = int(inter.sum()) if inter is not None else 0
    if inter is not None and inter.sum() < min_vox:
        inter = None                                   # agreement below the measurable floor is not evidence
    diag['min_component_voxels'] = min_vox
    if envelope is None:
        td = inter if inter is not None else nonempty[0][1]
        evidence = 'unverified_agreement' if inter is not None else f'unverified_{nonempty[0][0]}'
    else:
        tag = 'prior_region_' if diag.get('validation_region') == 'kb_prior_region' else ''
        if inter is not None and inside_frac(inter) >= min_inside:
            td, evidence = inter, tag + 'agreement'
            diag['agreement_inside_frac'] = round(inside_frac(inter), 3)
        else:
            for n, m in nonempty:
                keep, nk, k = validated_components(m)
                diag[f'{n}_components'] = k; diag[f'{n}_validated_components'] = nk
                if nk:
                    td, evidence = keep, f'{tag}{n}_validated'
                    break
    diag['tumour_evidence'] = evidence
    if td is None:
        # nothing consistent with the expected host: record where the raters put the tumour (possible metastasis /
        # organ leakage) but keep the cancer-type host — the planner then prompts both sides of a bilateral host
        diag['tumour_voxels'] = 0
        if md is not None and kb is not None:
            src = inter if inter is not None else nonempty[0][1]
            labs = md[src]; labs = labs[labs > 0]
            if labs.size:
                top = int(np.bincount(labs.astype(np.int64)).argmax())
                ent = kb.ts_to_entity.get(task, {}).get(label_names.get(top))
                diag['raters_elsewhere'] = dict(ts_class=label_names.get(top), entity=ent[0] if ent else None,
                                                frac=round(float((labs == top).sum()) / int(src.sum()), 3),
                                                basis='agreement' if inter is not None else nonempty[0][0])
        if md is not None and kb is not None and expected_host:
            spacing = _spacing(ref.affine)
            hosts = host_candidates_evidence(kb, kb.spread_hosts(expected_host, cancer_type), masks, names, md, label_names, task, spacing, envelope_mm, min_inside, min_component_ml)
            for h in hosts:
                h['kept'] = h['cls'] == 'primary' or (h['evidence'] or 0) >= 1
                h['contact_check'] = (not h['kept']) and h['in_fov'] and h['cls'] == 'local_invasion'
            diag['hosts'] = hosts
        return None, expected_host, diag

    n = int(td.sum()); diag['tumour_voxels'] = n
    com = np.argwhere(td).mean(0)
    cen = ref.affine[:3, :3] @ com + ref.affine[:3, 3]

    # host by overlap — only the agreement may override the expected host
    host = expected_host
    if md is not None and kb is not None:
        src = inter if inter is not None else td
        labs = md[src]; labs = labs[labs > 0]
        if labs.size:
            counts = np.bincount(labs.astype(np.int64))
            ts_map = kb.ts_to_entity.get(task, {})
            for lab in np.argsort(counts)[::-1]:
                if counts[lab] == 0: break
                ent = ts_map.get(label_names.get(int(lab)))
                if ent and kb.entities[ent[0]].get('is_anchor'):
                    frac = float(counts[lab]) / int(src.sum())
                    diag.update(overlap_host=ent[0], overlap_host_frac=round(frac, 3), overlap_ts_class=label_names.get(int(lab)))
                    if expected_host is None:
                        host = ent[0]
                    elif ent[0] != expected_host and inter is not None and frac >= 0.5:
                        host = ent[0]; diag['host_override'] = 'agreement lies in another anchor'
                    elif ent[0] != expected_host:
                        diag['host_conflict_ignored'] = f'{ent[0]} ({frac:.2f}) vs expected {expected_host}'
                    break
    if md is not None and kb is not None and expected_host:
        spacing = _spacing(ref.affine)
        cands = kb.spread_hosts(expected_host, cancer_type)
        hosts = host_candidates_evidence(kb, cands, masks, names, md, label_names, task, spacing, envelope_mm, min_inside, min_component_ml)
        for h in hosts:
            h['kept'] = h['cls'] == 'primary' or (h['evidence'] or 0) >= 1          # secondary hosts earn prompts only with data-driven evidence
            h['contact_check'] = (not h['kept']) and h['in_fov'] and h['cls'] == 'local_invasion'   # neighbours: tumour-contact check on TS masks, no prompts
        diag['hosts'] = hosts
    return (float(cen[0]), float(cen[1]), float(cen[2])), host, diag


def load_ts_output(structures_csv: str, labels_json: Optional[str] = None, image_or_mask_nifti: Optional[str] = None,
                   tumour_mask=None, kb: Optional[KnowledgeBase] = None,
                   min_volume_ml: float = 0.05, expected_host: Optional[str] = None, cancer_type: Optional[str] = None) -> Tuple[TSOutput, dict]:
    """Build a TSOutput from the TS structure table. Returns (ts_output, diagnostics)."""
    base = structures_csv[:-len('_structures.csv')]
    labels_json = labels_json or base + '_labels.json'
    ts_mask = base + '.nii.gz'
    with open(structures_csv, newline='') as f:
        rows = [r for r in csv.DictReader(f)]
    meta = json.load(open(labels_json)) if os.path.exists(labels_json) else {}
    task = meta.get('task') or ('total_mr' if str(meta.get('modality', '')).upper().startswith('MR') else 'total')
    label_names = {int(k): v for k, v in (meta.get('labels') or {}).items()}
    diag = dict(task=task, modality=meta.get('modality'), site=meta.get('site'), n_rows=len(rows), dropped_small=[])

    structs = []
    for r in rows:
        vol = float(r['volume_ml'])
        if vol < min_volume_ml:                       # speckle: a few voxels of a class → not a finding
            diag['dropped_small'].append(r['name'])
            continue
        structs.append(TSStructure(ts_name=r['name'], volume_ml=vol,
                                   centroid_mm=(float(r['centroid_x_mm']), float(r['centroid_y_mm']), float(r['centroid_z_mm'])),
                                   truncated=_as_bool(r.get('touches_boundary', False)),
                                   zmin_mm=float(r['zmin_mm']), zmax_mm=float(r['zmax_mm'])))

    src = image_or_mask_nifti or (ts_mask if os.path.exists(ts_mask) else None)
    if src:
        fov = fov_z_from_nifti(src); diag['fov_source'] = 'nifti_header'
    else:
        fov = fov_z_from_table(rows); diag['fov_source'] = 'structure_union'

    cen, host = None, None
    tm = [tumour_mask] if isinstance(tumour_mask, str) else list(tumour_mask or [])
    tm = [q for q in tm if q and os.path.exists(q)]
    if tm:
        cen, host, tdiag = tumour_summary(tm, ts_mask if os.path.exists(ts_mask) else None, kb, task, label_names, expected_host=expected_host, cancer_type=cancer_type)
        diag.update(tdiag)
    hosts = [dict(entity=h['entity'], cls=h['cls'], prior=h['prior'], evidence=h['evidence'], weight=h['weight'], region=h['region'],
                  role='primary' if h['cls'] == 'primary' else ('secondary' if h.get('kept') else 'contact_check'))
             for h in diag.get('hosts', []) if h.get('kept') or h.get('contact_check')]
    return TSOutput(task=task, structures=structs, fov_z_mm=fov, tumour_centroid_mm=cen, tumour_host_guess=host or expected_host, hosts=hosts), diag
