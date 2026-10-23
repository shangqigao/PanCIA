"""Lesion ledger v6a (17 Oct 2026) = v6s + image quality control and anatomy feasibility (pancia_kb.anatomy_qc; rules
documents/plan/Anatomy_QC_rules_draft.md, user approved 17 Oct): host references are the feasible organ masks minus
contested regions, touch uses the territory (reference + contested), and three tumour-organ relations are added
(in_non_host_organ, contested, organ_replaced). ledger_v6s.py is unchanged.
v6s docstring: Lesion ledger v6s (15 Oct 2026): ledger v6 FINAL v3 with the Stage 0 definitions. Set-based only: no image.
Rules: PanCIA/documents/plan/Ledger_v6_rules_final_draft.md (D1-D11, T1-T8, decisions 1-7 accepted 15 Oct).
Units: claim (one agent's connected component, never cut) -> anchor (connected agreement of >= 2 agents >= 0.5 ml, or a
single-agent component of an agent's kept-claim union) -> lesion (anchors assigned to one host, grouped within tau);
outlines of an anchor = the kept claims overlapping it. ledger_v6.py (v3 mode, K1) is untouched; this module reuses its
helpers only."""
from __future__ import annotations
import json, os
import numpy as np
from scipy import ndimage
from .adapter import _spacing, _labels_by_side, host_prior_region
from .ledger_v6 import Box, _load, _bool, _shape, _vt_organ, _hk, _crop_to, _near, _claims
from . import anatomy_qc as AQ

P = dict(min_ml=0.5, crumb_ml=0.2, crumb_mm=5.0, env_mm=10.0, tau_mm=5.0, band_mm=3.0, trace_mm=3.0,
         gate_inside=0.5, slab_cover=0.4, slab_aspect=0.25, organ_vol_ratio=0.6, organ_dice2d=0.6, organ_inside=0.8,
         contain_H=0.8, contain_frac=0.5, box_inside=0.5, nod_ld_mm=10.0, nod_slices=3, nod_z_mm=10.0,
         pi_two=0.9, pi_silent=0.75, pi_single=0.5, pi_none=0.1, multi_host_p=0.1, admit_p=0.5,
         review_prim_s=0.05, review_nb_s=0.30, acq_ratio=1.5, leak_frac=0.5, nonhost_inside=0.5)
VERSION = 'v6a'
PRIOR = dict(primary=1.0, local_invasion=0.3, distant=0.15)
RANK = {'two_agent': 0, 'silent': 1, 'single_confirmed': 2, 'single_unconfirmed': 3}


def _bb(mask):
    s = ndimage.find_objects(mask.astype(np.int8))
    return s[0] if s else None


def _union_box(items):
    """(sl, crop) list -> (sl, crop) of their union."""
    items = [(s, c) for s, c in items if s is not None and c is not None and c.any()]
    if not items:
        return None, None
    sl = tuple(slice(min(s[d].start for s, _ in items), max(s[d].stop for s, _ in items)) for d in range(3))
    out = np.zeros(tuple(x.stop - x.start for x in sl), bool)
    for s, c in items:
        out[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(s, sl))] |= c
    return sl, out


def _ld(U, ax, ipa, sp):
    """Longest in-plane diameter (mm) over the slices of the acquisition axis ax: the maximum Feret diameter of each slice."""
    from scipy.spatial import ConvexHull
    best = 0.0
    for z in np.nonzero(U.any(axis=tuple(ipa)))[0]:
        pts = np.argwhere(np.take(U, z, axis=ax)).astype(float) * np.array([sp[d] for d in ipa])
        if len(pts) > 3:
            try:
                pts = pts[ConvexHull(pts).vertices]
            except Exception:
                pass
        if len(pts) > 400:
            pts = pts[np.linspace(0, len(pts) - 1, 400).astype(int)]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).max() if len(pts) > 1 else 0.0
        best = max(best, float(d) + float(min(sp[d_] for d_ in ipa)))   # + one pixel: extent of the end voxels
    return best


def _get(full, sl):
    return full[sl] if not isinstance(full, Box) else full[sl]


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
    crumb_vox = max(1, int(round(p['crumb_ml'] / vox_ml)))
    import nibabel as nib
    k = [i for i, c in enumerate(nib.aff2axcodes(ts_img.affine)) if c in 'SI'][0]
    ip = [i for i in range(3) if i != k]
    ax = int(np.argmax(sp)) if float(max(sp)) / float(min(sp)) >= p['acq_ratio'] else k
    tau = max(p['tau_mm'], float(sp[ax]))                                   # decision 3
    ipa = [d for d in range(3) if d != ax]          # in-plane axes of the acquisition plane (audit 15 Oct: slab, D10, T4 on ax)
    shape = md.shape
    primary_visible = not any(l.get('step') == 'primary_not_in_fov' for l in plan.get('log', []))
    flags_scan = []

    # ---- (0)-(1) image quality control and organ feasibility (host reasoning R(A_i, A_j), anatomy_qc)
    tum = np.zeros(shape, np.int8)                           # tumour-agent agreement: voxels of >= 2 agents' cancer prompts
    tum_files = [R('BiomedParse', 'Radiology', rel + '_tumor.nii.gz'), R('VoxTell', 'Radiology', rel + '_tumor.nii.gz')]
    # host-prompt answers are not used here: every host in view is asked for a tumour and answers, which would mark every
    # prompted organ as tumour-touched (cancer-prompt claims of every tumour agent only)
    for f_ in tum_files:
        if os.path.exists(f_):
            tum += _bool(_load(f_)).astype(np.int8)
    tum = tum >= 2
    img_root = p.get('image_root') or os.path.join(os.path.dirname(os.path.normpath(seg_root)), 'TCGA_NIFTI')
    img_p = os.path.join(img_root, (manifest.get('img_name') or rel) + '.nii.gz')
    X = np.asanyarray(_load(img_p).dataobj) if os.path.exists(img_p) else None
    thr_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_base', 'qc_thresholds.json')
    thr = json.load(open(thr_p)) if os.path.exists(thr_p) else {}
    modality = (plan.get('input') or {}).get('modality') or rel.split('/')[0]
    AQR = AQ.analyse(kb, md, names, task, ts_img.affine, np.asarray(sp, float), plan, manifest, plan_dir, tumour=tum, image=X,
                     modality=modality, thresholds=thr,
                     primary={m['entity'] for m in manifest['masks'] if m['list'] == 'T' and m.get('host_class') == 'primary'})
    del tum, X

    # ---- D2 hosts: H = union of organ agents' masks; KB box for a primary with none (E = box, decision 2); mask-less hosts
    hosts = {}
    for m in manifest['masks']:
        if m['list'] != 'T':
            continue
        key = (m['entity'], m.get('side'))
        h = hosts.setdefault(key, dict(entity=m['entity'], side=m.get('side'), cls=m.get('host_class'),
                                       prior=m.get('host_prior'), prompt_files=[]))
        h['prompt_files'].append(m['file'])
    prim = [q for q, h in hosts.items() if h['cls'] == 'primary']
    spread = {q['entity']: q for q in kb.spread_hosts(prim[0][0], (plan.get('input') or {}).get('cancer_type'))} if prim else {}
    lbs = {}
    for (ent, side), h in hosts.items():
        if ent not in lbs:
            lbs[ent] = _labels_by_side(kb, task, names, ent)
        labs = [l for sd, ls in lbs[ent].items() for l in ls if side is None or sd in (side, None)]
        o = AQR['organs'].get((ent, side))                     # feasible masks minus contested regions (Eq. organ_reference)
        Href = o['ref'] if o is not None else Box(None, shape)
        feas = sorted(k for k, st in (o['agents'] if o is not None else {}).items() if st != 'infeasible')
        ref = '+'.join(feas) if Href.any() else None
        h['replaced'] = False
        if ref is None and h['cls'] == 'primary':             # H5: only the primary falls back to the KB region (user, 17 Oct)
            Hb, _ = host_prior_region(kb, ent, md, names, task, sp)
            ref = 'kb_box' if Hb is not None else None
            h['H'] = Box(Hb, shape); h['T'] = h['H']; h['C'] = Box(None, shape)
            ts_cls = bool((kb.entities.get(ent, {}).get('ts') or {}).get(task))
            # 'organ replaced' only when an organ agent could segment the organ: its masks were all infeasible, or TS has a
            # class for it on this task and found nothing (an OOD primary without any mask is not evidence of replacement)
            h['replaced'] = bool(o is not None and o['agents'] and all(st == 'infeasible' for st in o['agents'].values()))                 or (ts_cls and (o is None or 'TS' not in o['agents']))
        else:
            h['H'] = Href
            h['T'] = o['territory'] if (o is not None and ref is not None) else Box(None, shape)
            h['C'] = o['contested'] if o is not None else Box(None, shape)
        h.update(ref=ref, w=float(PRIOR.get(h['cls'], 0.0)),
                 also_distant=bool(spread.get(ent, {}).get('also_distant')),
                 aq=dict(agents=(o or {}).get('agents'), fail=(o or {}).get('fail'), flags=(o or {}).get('flags'),
                         contested_ml=round(h['C'].sum() * vox_ml, 2), ref_ml=round(h['H'].sum() * vox_ml, 1)))
        if h['ref'] == 'kb_box':
            h['E'] = h['H']                                                    # decision 2: E(box) = box
        elif h['ref'] is not None:
            h['E'] = h['T'].dilate(p['env_mm'], sp)                            # envelope of the territory (reference + contested)
        else:
            h['E'] = Box(None, shape)                                          # mask-less / unsegmentable: never touched
    maskless = [q for q, h in hosts.items() if h['ref'] is None]
    host_labels = {int(l) for (ent, side) in hosts for sd, ls in lbs[ent].items() for l in ls if side is None or sd in (side, None)}
    if maskless:
        flags_scan.append('host_without_mask_in_scan')
    Eprim = np.zeros(shape, bool)
    for q in (prim if primary_visible else []):      # audit 15 Oct: no phantom envelope when out of view
        e = hosts[q]['E']
        if e.any():
            Eprim[e.sl] |= e.crop

    # ---- D3 claims per agent and prompt; crumbs touching E(primary) grouped per agent and prompt (decision 6)
    small = dict(region=Eprim, min_vox=crumb_vox, group_mm=p['crumb_mm'], spacing=sp, touch=True) if Eprim.any() else None
    claims = []
    raw = {}                                                                   # raw output per agent (all prompts, pre-drop)
    def add(mask, agent, prompt, src):
        """Claims of one agent's prompt, computed on the mask's bounding box (speed); raw output accumulated."""
        if agent not in raw:
            raw[agent] = np.zeros(shape, bool)
        if not mask.any():
            return
        b = tuple(slice(int(np.nonzero(np.any(mask, axis=tuple(x for x in range(3) if x != d)))[0][0]),
                        int(np.nonzero(np.any(mask, axis=tuple(x for x in range(3) if x != d)))[0][-1]) + 1) for d in range(3))
        sub = mask[b]
        raw[agent][b] |= sub
        sm = dict(small, region=small['region'][b]) if small is not None else None
        for c in _claims(sub, agent, src, min_vox, sm):
            c['sl'] = tuple(slice(s.start + o.start, s.stop + o.start) for s, o in zip(c['sl'], b))
            c.update(agent=agent, prompt=prompt)
            claims.append(c)
    bp_p = R('BiomedParse', 'Radiology', rel + '_tumor.nii.gz'); vt_p = R('VoxTell', 'Radiology', rel + '_tumor.nii.gz')
    if os.path.exists(bp_p):
        add(_bool(_load(bp_p)), 'BP', 'cancer', 'bp')
    if os.path.exists(vt_p):
        add(_bool(_load(vt_p)), 'VT', 'cancer', 'vt_initial')
    for key, h in hosts.items():
        for f in h['prompt_files']:
            fp = os.path.join(plan_dir, f)
            if os.path.exists(fp):
                add(_bool(_load(fp)), 'VT', ('host', key), 'vt_prompt:' + f[:-7])
    agents = sorted(raw)
    raw_box = {a: Box(raw[a], shape) for a in agents}
    silent = {a: not bool((raw[a] & Eprim).any()) for a in agents} if Eprim.any() else {a: False for a in agents}
    del raw
    for i, c in enumerate(claims):
        c['id'] = i
        c['drop'] = None
    # prompt gate (host-prompt claims only)
    for c in claims:
        if c['prompt'] != 'cancer':
            E = hosts[c['prompt'][1]]['E']
            ins = int((c['crop'] & E[c['sl']]).sum()) / c['vox'] if E.any() else 0.0
            if ins < p['gate_inside']:
                c['drop'] = 'outside_prompt_gate'
    # slab: flat against the host organ it overlaps most; dropped unless a kept non-flat claim of another agent overlaps it
    for c in claims:
        if c['drop']:
            continue
        best_H, bo = None, 0                          # the HOST organ it overlaps most (as v3; any-TS-organ dropped 260 claims, 15 Oct)
        for q, h in hosts.items():
            if h['ref'] in (None, 'kb_box'):
                continue
            o = int((c['crop'] & h['T'][c['sl']]).sum())
            if o > bo:
                best_H, bo = h['T'], o
        c['flat'] = False
        if best_H is not None:
            s_ = _shape(c, best_H, ax, ipa, sp, vox_ml)
            c['flat'] = bool(s_.get('cover_med') is not None and s_['cover_med'] >= p['slab_cover'] and (s_['aspect'] or 1) <= p['slab_aspect'])
    alive = [c for c in claims if not c['drop']]
    for c in alive:
        if c.get('flat') and not any(o['agent'] != c['agent'] and not o.get('flat')
                                     and int((_crop_to(o, c['sl']) & c['crop']).sum()) >= min_vox for o in alive):
            if primary_visible and Eprim.any() and bool((c['crop'] & Eprim[c['sl']]).any()):
                c['slab_soft'] = True                 # user, 15 Oct: near the primary a flat claim is kept and soft-rejected
            else:
                c['drop'] = 'slab_flat_artefact'
    keep = [c for c in claims if not c['drop']]
    dropped = [dict(id=c['id'], agent=c['agent'], prompt=str(_hk(c['prompt'][1]) if c['prompt'] != 'cancer' else 'cancer'),
                    ml=round(c['vox'] * vox_ml, 2), reason=c['drop']) for c in claims if c['drop']]

    ledger = dict(version=VERSION, rel=rel, spacing=[float(x) for x in sp], tau_mm=tau, primary_visible=primary_visible,
                  hosts=[dict(host=_hk(q), cls=h['cls'], w=h['w'], ref=h['ref'], also_distant=h['also_distant'],
                              aq=h.get('aq'), replaced=h.get('replaced')) for q, h in hosts.items()],
                  silent={a: silent[a] for a in agents}, flags=flags_scan, dropped_claims=dropped,
                  n_claims=len(claims), anchors=[], lesions=[],
                  qc=dict(state=AQR['state'], reasons=AQR['reasons'], f_away=AQR['f_away'], n_away=AQR['n_away'],
                          noise=AQR['noise'], failed=[e for e in AQR['expected'] if e['failed']],
                          n_conflicting_pairs=sum(1 for q_ in AQR['pairs'] if q_['conflicting']),
                          n_infeasible_masks=sum(1 for v in AQR['phi'].values() if v['state'] == 'infeasible'), version=AQ.VERSION))
    if not keep:
        ledger.update(admitted_ml=0.0, admitted_anchor_ml=0.0, primary_admitted=False, primary_ml=0.0, primary_anchor_ml=0.0,
                      claim_boxes={})
        return ledger, {}, ts_img.affine, shape

    # ---- D4 anchors, on the working box W around the kept claims
    W, _ = _union_box([(c['sl'], c['crop']) for c in keep])
    Wshape = tuple(s.stop - s.start for s in W)
    def tofull(sl_local):
        return tuple(slice(a.start + w.start, a.stop + w.start) for a, w in zip(sl_local, W))
    M = {a: np.zeros(Wshape, bool) for a in agents}
    for c in keep:
        M[c['agent']][tuple(slice(s.start - w.start, s.stop - w.start) for s, w in zip(c['sl'], W))] |= c['crop']
    cnt = sum(M[a].astype(np.uint8) for a in agents)
    A = cnt >= 2
    del cnt
    anchors = []
    TA = np.zeros(Wshape, bool)
    lab, n = ndimage.label(A, structure=np.ones((3, 3, 3)))
    for j, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        cr = lab[sl] == j
        if int(cr.sum()) >= min_vox:
            TA[sl] |= cr
            ag = sorted(a for a in agents if bool((M[a][sl] & cr).any()))
            anchors.append(dict(kind='two', agents=ag, sl=tofull(sl), crop=cr, vox=int(cr.sum())))
    del lab, A
    for a in agents:
        lab, n = ndimage.label(M[a], structure=np.ones((3, 3, 3)))
        parent = list(range(n + 1))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for c in keep:                      # a crumb group is one claim made of several components: one anchor
            if c['agent'] == a and c.get('grouped'):
                ls = np.unique(lab[tuple(slice(s.start - w.start, s.stop - w.start) for s, w in zip(c['sl'], W))][c['crop']])
                ls = [x for x in ls if x > 0]
                for x in ls[1:]:
                    parent[find(x)] = find(ls[0])
        roots = {}
        for j, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is not None:
                roots.setdefault(find(j), []).append((j, sl))
        for r, parts in roots.items():
            items = [(sl, lab[sl] == j) for j, sl in parts]
            sl, cr = _union_box(items)
            if sl is None or bool((TA[sl] & cr).any()):
                continue
            anchors.append(dict(kind='single', agents=[a], sl=tofull(sl), crop=cr, vox=int(cr.sum())))
        del lab
    del M
    AU = np.zeros(shape, bool)
    TAf = np.zeros(shape, bool)
    TAf[W] = TA
    del TA
    for i, an in enumerate(anchors):
        an['id'] = i
        AU[an['sl']] |= an['crop']
    # ---- D5 outlines
    for an in anchors:
        an['outlines'] = [c['id'] for c in keep if bool((_crop_to(c, an['sl']) & an['crop']).any())]
    byid = {c['id']: c for c in keep}

    # ---- D10 outline flags (every organ the outline overlaps: TS labels and host H)
    tsvol = np.bincount(md.ravel(), minlength=max(names) + 1 if names else 1)
    def organ_test(obj, H, hv=None):
        """(a) is the organ / (b) contains the organ, or None."""
        hv = H.sum() if hv is None else hv
        if not hv:
            return None
        inter = int((obj['crop'] & H[obj['sl']]).sum())
        if inter >= p['contain_H'] * hv and inter >= p['contain_frac'] * obj['vox']:
            return 'b'
        if inter >= p['organ_inside'] * obj['vox'] and obj['vox'] >= p['organ_vol_ratio'] * hv:
            s_ = _shape(obj, H, ax, ipa, sp, vox_ml)
            if (s_.get('vol_ratio') or 0) >= p['organ_vol_ratio'] and (s_.get('dice2d_med') or 0) >= p['organ_dice2d'] \
                    and (s_.get('inside_H') or 0) >= p['organ_inside']:
                return 'a'
        return None
    for c in keep:
        c['flag'] = []
        sub = md[c['sl']][c['crop']]
        ls, cts = np.unique(sub[sub > 0], return_counts=True)
        for l, ov in zip(ls, cts):
            if int(l) in host_labels:                   # audit 15 Oct: host organs are tested on H below
                continue
            ov, ovol = int(ov), int(tsvol[l])
            pre_b = ov >= p['contain_H'] * ovol and ov >= p['contain_frac'] * c['vox']
            pre_a = ov >= p['organ_inside'] * c['vox'] and c['vox'] >= p['organ_vol_ratio'] * ovol
            if pre_b:
                c['flag'].append(f"{names.get(int(l), l)}:b")
            elif pre_a:
                t = organ_test(c, Box(md == l, shape), ovol)
                if t:
                    c['flag'].append(f"{names.get(int(l), l)}:{t}")
        for q, h in hosts.items():
            if h['ref'] in (None, 'kb_box') or not bool((c['crop'] & h['T'][c['sl']]).any()):
                continue
            t = organ_test(c, h['T'])
            if t:
                c['flag'].append(f"{_hk(q)}:{t}")
        c['flag'] = sorted(set(c['flag'])) + (['slab'] if c.get('slab_soft') else [])

    # ---- v6a: organs that are not hosts of the cancer type and not related to a host by containment (T-unrelated)
    host_ents = {q[0] for q in hosts}
    nonhost = {key: o_ for key, o_ in AQR['organs'].items()
               if key[0] not in host_ents and not any(AQ.related(kb, key[0], he) for he in host_ents)
               and kb.entities.get(key[0], {}).get('category') in AQ.CRISP and o_['ref'].crop is not None}

    # ---- D6 touch, D8 host probabilities, decision 5 exclusion
    def band(an):
        ps, shell = _near(an['crop'], an['sl'], shape, sp, p['band_mm'])
        inner = np.zeros(shell.shape, bool)
        inner[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(an['sl'], ps))] = an['crop']
        shell &= ~inner & ~AU[ps]
        shell |= inner & (ndimage.distance_transform_edt(inner, sampling=sp) <= p['band_mm'])
        return ps, shell, max(int(shell.sum()), 1)
    for an in anchors:
        sl, L = an['sl'], an['crop']
        touched = []
        for q, h in hosts.items():
            if not h['E'].any():
                continue
            if h['ref'] == 'kb_box':
                if int((L & h['H'][sl]).sum()) >= p['box_inside'] * an['vox']:
                    touched.append(q)
            elif bool((L & h['E'][sl]).any()):
                touched.append(q)
        ps, shell, ns = band(an)
        s = {q: int((shell & hosts[q]['T'][ps]).sum()) / ns for q in touched}     # v6a: contested regions count for both organs
        an['flags'], an['excluded'], an['involved'], an['overlap'] = [], [], {}, []
        tp = [q for q in touched if q in prim] if primary_visible else []
        if tp:
            if len(tp) > 1:
                tp.sort(key=lambda q: (-s[q], -int((L & hosts[q]['E'][sl]).sum())))
                if s[tp[0]] == s[tp[1]] and int((L & hosts[tp[0]]['E'][sl]).sum()) == int((L & hosts[tp[1]]['E'][sl]).sum()) \
                        and not (hosts[tp[0]]['ref'] == 'kb_box' and hosts[tp[1]]['ref'] == 'kb_box'):   # boxes overlap by design
                    an['flags'].append('side_unresolved')
            q0 = tp[0]
            an['primary'] = True
            an['P'] = {q0: 1.0}
            an['s'] = {_hk(q0): round(s[q0], 3)}
            inv = [q for q in touched if q not in prim]
            an['involved'] = {_hk(q): round(s[q], 3) for q in inv}
            an['inv_keys'] = inv
            if inv:
                an['flags'].append('primary_involves_other_organs')
                if s[q0] < p['review_prim_s'] and max(s[q] for q in inv) >= p['review_nb_s']:
                    an['flags'].append('host_uncertain_review')
            for q in [q0] + inv:                                               # D10 on the anchor
                if hosts[q]['ref'] == 'kb_box':
                    continue
                t = organ_test(an, hosts[q]['T'])
                if t:
                    an['overlap'].append(f"{_hk(q)}:{t}")
        else:
            an['primary'] = False
            cand = [q for q in touched if q not in prim]
            keepq = []
            for q in cand:                                                     # decision 5
                if hosts[q]['ref'] == 'kb_box':
                    keepq.append(q); continue
                s_ = _shape(an, hosts[q]['T'], ax, ipa, sp, vox_ml)
                is_org = ((s_.get('vol_ratio') or 0) >= p['organ_vol_ratio'] and (s_.get('dice2d_med') or 0) >= p['organ_dice2d']
                          and (s_.get('inside_H') or 0) >= p['organ_inside'])
                if is_org and an['kind'] == 'single':
                    an['excluded'].append(_hk(q)); continue
                if is_org:
                    an['overlap'].append(f"{_hk(q)}:a")
                keepq.append(q)
            ws = {q: hosts[q]['w'] * s[q] for q in keepq}
            if keepq and sum(ws.values()) <= 0:
                ws = {q: hosts[q]['w'] for q in keepq}
            z = sum(ws.values())
            an['P'] = {q: v / z for q, v in ws.items()} if z > 0 else {}
            an['s'] = {_hk(q): round(s[q], 3) for q in keepq}
            top = sorted(an['P'].values(), reverse=True)
            if len(top) > 1 and top[1] >= p['multi_host_p']:
                an['flags'].append('multi_host')
            if not an['P']:
                an['flags'].append('no_host')
                if maskless:
                    an['flags'].append('host_without_mask_possible')
        an['flagged'] = bool(an['overlap'])
        # v6a tumour-organ relations R(a, H_o)
        an['in_non_host'] = None
        for key, o_ in nonhost.items():
            b_ = o_['ref']
            if b_.crop is None or any(min(x.stop, y.stop) <= max(x.start, y.start) for x, y in zip(an['sl'], b_.sl)):
                continue
            if int((L & b_[sl]).sum()) >= p['nonhost_inside'] * an['vox']:
                an['in_non_host'] = AQ._hkey(key); an['flags'].append(f'in_non_host_organ:{AQ._hkey(key)}'); break
        hq = list(an['P']) + list(an.get('inv_keys') or [])
        if any(hosts[q]['C'].any() and bool((L & hosts[q]['C'][sl]).any()) for q in hq):
            an['flags'].append('contested')
        if an['primary'] and any(hosts[q].get('replaced') and an['vox'] * vox_ml >= 0.5 * ((kb.entities.get(q[0], {}).get('volume_ml') or {}).get('min') or 0)
                                 for q in an['P']):
            an['flags'].append('organ_replaced')        # a tumour that replaced its organ is at least organ-sized (0.5 KB min volume)

    # ---- D10b outline leak (user, 15 Oct): the part of an outline beyond its anchor that lies inside organs other than the
    # anchor's assigned hosts is >= 50 % of the outline -> soft, per anchor-outline pair. Involved organs are not exempt:
    # real invasion rarely puts half an outline inside a neighbour, while whole-organ leaks (KIRC liver) do.
    host_keys = list(hosts)
    for an in anchors:
        allowed = set(an['P'])
        an['leak'] = []
        for i in an['outlines']:
            c = byid[i]
            beyond = c['crop'] & ~_crop_to(an, c['sl'])
            if int(beyond.sum()) < p['leak_frac'] * c['vox']:
                continue
            org = md[c['sl']] > 0
            for q in host_keys:
                if hosts[q]['ref'] is not None:                                # KB boxes included (audit 15 Oct)
                    org |= hosts[q]['T'][c['sl']]
            for q in allowed:
                if hosts[q]['ref'] is not None:                                # a KB box host counts as its region
                    org &= ~hosts[q]['T'][c['sl']]
            if int((beyond & org).sum()) >= p['leak_frac'] * c['vox']:
                an['leak'].append(i)

    def ok_outlines(an):
        return [byid[i] for i in an['outlines'] if not byid[i]['flag'] and i not in an['leak']]

    # ---- D9 summary prior per anchor
    def summary_prior(an):
        ol = [byid[i] for i in an['outlines']]
        use = ok_outlines(an) or ol
        if not use:
            return p['pi_single']
        sl, U = _union_box([(c['sl'], c['crop']) for c in use])
        pr = np.zeros(U.shape, np.float32)
        ta = TAf[sl]; ep = Eprim[sl]
        for c in use:
            cr = _crop_to(c, sl)
            sil = an['primary'] and all(silent[a] for a in agents if a != c['agent']) and len(agents) > 1
            v = np.where(ta, p['pi_two'], np.where(sil & ep, p['pi_silent'], p['pi_single'])).astype(np.float32)
            pr = np.maximum(pr, np.where(cr, v, 0))
        return float(pr[U].mean())
    for an in anchors:
        an['pi'] = round(summary_prior(an), 4)

    # ---- D7 lesions: anchors per host grouped within tau (chained)
    assign = {}
    for an in anchors:
        for q in an['P']:
            assign.setdefault(q, []).append(an['id'])
    near_cache = {}
    def near(i):
        if i not in near_cache:
            an = anchors[i]
            near_cache[i] = _near(an['crop'], an['sl'], shape, sp, tau)
        return near_cache[i]
    lesions = []
    for q, ids in assign.items():
        parent = {i: i for i in ids}
        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        pad = [int(np.ceil(tau / float(v))) + 1 for v in sp]
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                A_, B_ = anchors[ids[a_i]], anchors[ids[b_i]]
                if any(A_['sl'][d].stop + pad[d] <= B_['sl'][d].start or B_['sl'][d].stop + pad[d] <= A_['sl'][d].start for d in range(3)):
                    continue
                if f(ids[a_i]) == f(ids[b_i]):
                    continue
                ps, nm = near(ids[a_i])
                if bool((nm & _crop_to(B_, ps)).any()):
                    parent[f(ids[a_i])] = f(ids[b_i])
        groups = {}
        for i in ids:
            groups.setdefault(f(i), []).append(i)
        for g in groups.values():
            lesions.append(dict(host=q, anchors=sorted(g)))
    for an in anchors:
        if not an['P']:
            lesions.append(dict(host=None, anchors=[an['id']]))
    near_cache.clear()

    # ---- lesion attributes (T terms)
    def lesion_attrs(L):
        q = L['host']
        ans = [anchors[i] for i in L['anchors']]
        unf = [a for a in ans if not a['flagged']]
        L['n_anchors'] = len(ans)
        oids = sorted({i for a in ans for i in a['outlines']})
        ol = [byid[i] for i in oids]
        okids = {c['id'] for a in (unf or ans) for c in ok_outlines(a)}   # audit 15 Oct: the lesion continues with its unflagged anchors
        uo = [c for c in ol if c['id'] in okids]
        L['all_outlines_flagged'] = bool(ol) and not uo
        L['flagged'] = (not unf) or L['all_outlines_flagged']
        L['flag_kind'] = 'slab' if (L['flagged'] and unf and any('slab' in c['flag'] for c in ol)) else 'organ_overlap'
        ext = uo or ol
        sl, U = _union_box([(c['sl'], c['crop']) for c in ext]) if ext else _union_box([(a['sl'], a['crop']) for a in ans])
        L['sl'], L['U'] = sl, U
        L['ml'] = round(int(U.sum()) * vox_ml, 2)
        L['ld_mm'] = round(_ld(U, ax, ipa, sp), 1)                          # audit 15 Oct: longest diameter, not bbox diagonal
        L['n_slices'] = int(U.any(axis=tuple(ipa)).sum())
        L['z_mm'] = round(L['n_slices'] * float(sp[ax]), 1)
        L['outlines'] = oids
        L['shared_outline'] = False
        base = unf or ans
        ag = sorted({x for a in base for x in a['agents']})
        L['agents'] = ag
        if any(a['kind'] == 'two' for a in base):
            cls = 'two_agent'
        elif q in prim and len(ag) == 1 and len(agents) > 1 and all(silent[x] for x in agents if x != ag[0]):
            cls = 'silent'
        else:
            ps, nm = _near(U, sl, shape, sp, p['trace_mm'])
            others = [x for x in agents if x not in ag] if len(ag) == 1 else []
            trace = len(ag) > 1 or any(bool((nm & raw_box[x][ps]).any()) for x in others)
            cls = 'single_confirmed' if trace else 'single_unconfirmed'
        L['cls'] = cls
    for L in lesions:
        lesion_attrs(L)
    # shared outline: one claim overlapping two different anchors in different lesions
    ol_les = {}
    for li, L in enumerate(lesions):
        for i in L['anchors']:
            for o in anchors[i]['outlines']:
                ol_les.setdefault(o, set()).add((li, i))
    for o, s_ in ol_les.items():
        anchs = {i for _, i in s_}
        if len(anchs) > 1 and len({li for li, _ in s_}) > 1:
            byles = {}
            for li, i in s_:
                byles.setdefault(li, set()).add(i)
            ls_ = list(byles)
            if any(byles[x] != byles[y] for x in ls_ for y in ls_ if x != y):
                for li in byles:
                    lesions[li]['shared_outline'] = True

    # ---- pass 1: primary lesions (T1-T4)
    def nodule(L):
        return L['ld_mm'] >= p['nod_ld_mm'] and L['n_slices'] >= p['nod_slices'] and L['z_mm'] >= p['nod_z_mm']
    plist = [L for L in lesions if L['host'] in prim]
    plist.sort(key=lambda L: (L['flagged'], RANK[L['cls']], -L['ml']))
    first = True
    for L in plist:
        if L['flagged']:
            L['status'], L['reason'] = 'soft', 'primary_' + L['flag_kind']; continue
        if first:
            L['status'], L['reason'] = 'admitted', 'primary_' + L['cls']; first = False; continue
        if L['cls'] == 'two_agent':
            L['status'], L['reason'] = 'admitted', 'primary_multifocal_two_agent'
        elif L['cls'] not in ('silent', 'single_confirmed'):
            L['status'], L['reason'] = 'soft', 'primary_extra_unconfirmed'
        elif nodule(L):
            L['status'], L['reason'] = 'admitted', 'primary_multifocal_single'
        else:
            L['status'], L['reason'] = 'soft', 'primary_satellite_single'
    Pstar = np.zeros(shape, bool)
    for L in plist:
        if L.get('status') == 'admitted':
            for i in L['anchors']:
                if not anchors[i]['flagged']:
                    Pstar[anchors[i]['sl']] |= anchors[i]['crop']
    Penv = Box(Pstar, shape).dilate(p['env_mm'], sp) if Pstar.any() else Box(None, shape)
    del Pstar

    # ---- pass 2: T6, T7
    site = {}
    def site_agree(L, q):
        unf = [anchors[i] for i in L['anchors'] if not anchors[i]['flagged']]
        if not unf:
            return 0.0
        sl, U = _union_box([(a['sl'], a['crop']) for a in unf])
        agree = np.zeros(U.shape, bool)                # audit 15 Oct: union over all host-prompt claims for h
        for c in keep:
            if c['prompt'] == 'cancer' or c['prompt'][1] != q:
                continue
            sc = _crop_to(c, sl) & U
            if not sc.any():
                continue
            oth = np.zeros(U.shape, bool)
            for o in keep:
                if o['agent'] != c['agent']:
                    oth |= _crop_to(o, sl)
            agree |= sc & oth
        return round(int(agree.sum()) * vox_ml, 2)
    for L in lesions:
        if 'status' in L:
            continue
        q = L['host']
        if q is None:
            L['status'], L['reason'] = 'soft', 'no_host'; continue
        if L['flagged']:
            L['status'], L['reason'] = 'soft', L['flag_kind']; continue
        cls = hosts[q]['cls']
        if cls == 'local_invasion':
            unf = [anchors[i] for i in L['anchors'] if not anchors[i]['flagged']]
            L['contiguous'] = Penv.any() and any(bool((a['crop'] & Penv[a['sl']]).any()) for a in unf)
            if L['contiguous']:
                L['status'], L['reason'] = ('admitted', 'local_two_agent_contiguous') if L['cls'] == 'two_agent' else ('soft', 'local_single_agent')
                continue
            if not hosts[q]['also_distant']:
                L['status'], L['reason'] = 'soft', 'local_not_contiguous'; continue
            cls = 'distant'
        if cls == 'distant':
            L['site_agree_ml'] = site_agree(L, q)
            if L['site_agree_ml'] >= p['min_ml']:
                L['status'], L['reason'] = 'admitted', 'distant_site_agreement'
            else:
                L['status'], L['reason'] = 'soft', 'distant_needs_site_prompt_agreement'
            continue
        L['status'], L['reason'] = 'soft', 'unknown_host_class'

    # ---- T8 per anchor
    les_of = {}
    for li, L in enumerate(lesions):
        for i in L['anchors']:
            les_of.setdefault(i, {})[L['host']] = li
    for an in anchors:
        if not an['P']:
            li = les_of[an['id']][None]
            an.update(p_admit=0.0, p_tumour=p['pi_none'], status='soft', reason='no_host', host=None, lesion=li)
            continue
        adm = {q: lesions[les_of[an['id']][q]]['status'] == 'admitted' and not an['flagged'] for q in an['P']}
        pa = sum(an['P'][q] for q in an['P'] if adm[q])
        adm_h = [q for q in an['P'] if adm[q]]
        mh = max(adm_h or list(an['P']), key=an['P'].get)          # audit 15 Oct: report the admitting host when admitted
        L = lesions[les_of[an['id']][mh]]
        reason = L['reason'] if not (an['flagged'] and L['status'] == 'admitted') else 'anchor_organ_overlap'
        an.update(p_admit=round(pa, 4), p_tumour=an['pi'], status='admitted' if pa >= p['admit_p'] else 'soft',
                  reason=reason, host=mh, lesion=les_of[an['id']][mh])
        if an['status'] == 'soft' and not an['primary'] and an['kind'] == 'single':
            an['p_tumour'] = p['pi_none']      # user, 15 Oct: single-agent fragments away from the primary get the low prior
        if an.get('in_non_host') and not an['primary']:      # v6a T-unrelated: it is that organ (primary protection kept)
            an.update(status='soft', reason='in_non_host_organ', p_tumour=p['pi_none'], p_admit=0.0)

    # ---- outputs
    adm_outlines = {c['id'] for an in anchors if an['status'] == 'admitted' for c in ok_outlines(an)}
    au = np.zeros(shape, bool)
    for i in adm_outlines:
        au[byid[i]['sl']] |= byid[i]['crop']
    ledger['admitted_ml'] = round(int(au.sum()) * vox_ml, 2)               # outline volume (widest reading)
    au[:] = False
    for an in anchors:
        if an['status'] == 'admitted':
            au[an['sl']] |= an['crop']
    ledger['admitted_anchor_ml'] = round(int(au.sum()) * vox_ml, 2)        # anchor volume (agreed / located core)
    del au
    ledger['primary_admitted'] = any(L.get('status') == 'admitted' for L in plist)
    pa_ = [an for an in anchors if an['status'] == 'admitted' and an['primary']]
    au = np.zeros(shape, bool)
    for an in pa_:
        au[an['sl']] |= an['crop']
    ledger['primary_anchor_ml'] = round(int(au.sum()) * vox_ml, 2)
    au[:] = False
    for an in pa_:
        for c in ok_outlines(an):
            au[c['sl']] |= c['crop']
    ledger['primary_ml'] = round(int(au.sum()) * vox_ml, 2)
    del au
    for L in lesions:                                   # D8 lesion weight (reporting only)
        if L['host'] is None:
            L['weight'] = None
        else:
            v = [anchors[i]['vox'] for i in L['anchors']]
            L['weight'] = round(sum(anchors[i]['P'].get(L['host'], 0) * x for i, x in zip(L['anchors'], v)) / max(sum(v), 1), 3)
    for an in anchors:                                  # D4 same-agent confirmation (recorded, not agreement)
        an['same_agent'] = sorted({x for x in agents if len({str(byid[i]['prompt']) for i in an['outlines'] if byid[i]['agent'] == x}) > 1})
    for an in anchors:
        ledger['anchors'].append(dict(
            id=an['id'], kind=an['kind'], agents='+'.join(an['agents']), ml=round(an['vox'] * vox_ml, 2),
            bbox=[[s.start, s.stop] for s in an['sl']], primary=an['primary'],
            hosts={_hk(q): round(v, 3) for q, v in sorted(an['P'].items(), key=lambda kv: -kv[1])}, s=an['s'],
            involved=an['involved'], excluded=an['excluded'], overlap=an['overlap'], flags=an['flags'],
            outlines=[dict(id=i, agent=byid[i]['agent'], prompt='cancer' if byid[i]['prompt'] == 'cancer' else _hk(byid[i]['prompt'][1]),
                           ml=round(byid[i]['vox'] * vox_ml, 2), flag=byid[i]['flag'], leak=i in an['leak']) for i in an['outlines']],
            pi=an['pi'], p_admit=an['p_admit'], p_tumour=an['p_tumour'], status=an['status'], reason=an['reason'],
            host=_hk(an['host']) if an['host'] else None, lesion=an['lesion'], same_agent='+'.join(an['same_agent'])))
    for li, L in enumerate(lesions):
        ledger['lesions'].append(dict(
            id=li, host=_hk(L['host']) if L['host'] else None, cls_host=hosts[L['host']]['cls'] if L['host'] else None,
            anchors=L['anchors'], outlines=L['outlines'], support=L['cls'], agents='+'.join(L['agents']), ml=L['ml'],
            ld_mm=L['ld_mm'], n_slices=L['n_slices'], z_mm=L['z_mm'], flagged=L['flagged'], shared_outline=L['shared_outline'],
            contiguous=L.get('contiguous'), site_agree_ml=L.get('site_agree_ml'), status=L['status'], reason=L['reason'],
            weight=L.get('weight'),
            bbox=[[s.start, s.stop] for s in L['sl']]))
    crops = {f'anchor_{an["id"]}': an['crop'] for an in anchors}
    crops.update({f'claim_{c["id"]}': c['crop'] for c in keep})
    ledger['claim_boxes'] = {c['id']: [[s.start, s.stop] for s in c['sl']] for c in keep}
    if p.get('return_hosts'):                          # review panels only: host organ references (no effect on the ledger)
        crops.update({f'host::{_hk(q)}': h['H'] for q, h in hosts.items() if h['ref'] is not None})
        crops.update({f'contested::{_hk(q)}': h['C'] for q, h in hosts.items() if h['C'].any()})
    return ledger, crops, ts_img.affine, shape
