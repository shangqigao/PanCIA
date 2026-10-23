"""v8.3 (21 Oct 2026; after panel U4, user decisions): copy of ledger_v82 (kept unchanged).
  1. Static expected list = v8.2 list with fix (1) (pancia_kb.expected_v83.static_hosts): an invasion route (or distant site) that ends
     at an unsegmented PART is mapped to its nearest segmented whole (bladder / ovary / uterus -> colon_sigmoid -> colon), class kept.
  2. Entry (role 'proximal'): any SEGMENTED KB anchor that is a tumour host (kb is_anchor and tumour_host; confidence >= 1/2; not on the
     list) joins the lesion's list if it is STRICTLY closer to the lesion than every expected host ('closer'), or if it holds the
     largest share of the lesion union among all segmented organs, above the share outside every organ ('contain'). No 10 mm gate.
     Parts are not anchors, so they never enter (fix (2) of the first v8.3 draft is dropped; v8.2's exclusion for non-host tissue kept).
  3. The 10 mm safeguard after the credible set is unchanged (scripts/g3_run_v83.py).
--- ledger_v82 docstring ---
v8.2 (21 Oct 2026; plan claude/v81_H_panel_host_review_2026-10-21.md, user decisions 1-9): copy of ledger_v81 (kept unchanged).
  Step 2  static expected list = the KB spread list (kb.spread_hosts + Planner.spread_targets side rules, NO agent-based reach test)
          in view (segmented, or a KB region), plus the planner's prompted hosts; class from the KB (planner class kept as cls_planner).
  Step 3  other organs (outside the static list, segmented, confidence >= 1/2) enter the lesion's list with role 'proximal' by two routes:
          (a) containment, always: the organ holds the largest share of the lesion union among all segmented organs (slice-filled
              territories) and more than the share outside every organ mask (plurality; no fraction; ties -> no entry);
          (b) distance gate: only if every static candidate is > host_tol_mm (10 mm) from the lesion, organs within 10 mm enter.
  Step 5  safeguard (applied in scripts/g3_run_v82.py): min distance over the credible set > 10 mm -> host unknown, decision <= uncertain.
  Fallback (anatomy insufficient) unchanged: no entry, no crumb rescue. KB-region footprint built from TotalSegmentator labels AND
  agent organ territories (D4 footprint bug). Distances are lesion-level (union of post-crumb anchors), surface to surface, true spacing.
--- ledger_v81 docstring ---
v8.1 (host step 1, 20 Oct 2026): copy of ledger_v8 (kept unchanged). Candidates: every present host of H_C in view at ANY distance (no 10 mm envelope gate); an organ outside H_C (plausible mask, mu 1)
is a candidate only when the anchor overlaps its mask, and is then reported as 'other' (user decision 20 Oct; no fraction); unsegmented hosts (KB relational
region, step 2) also at any distance (fix 13). phi adds dist_mm / ldist (host_v81: distance from the lesion union to the territory, capped at D_CAP).
The envelope E is still used for the no_confirmation signal and crumb grouping (not for candidacy).
Fallback (user 21 Oct): no organ with confidence 1 -> anatomy_insufficient: host assumed = planner primary (side undetermined),
no host inference, prompt-mismatch and flatness not evaluated, no_confirmation over the whole scan, no U-nh extent.
--- ledger_v8 docstring ---
Lesion ledger v8 (19 Oct 2026) = ledger v7 with the extent flags stored by name (leak, organ, prompt_mismatch, flat) for the v8
extent prior w0 ~ psi * prod eta_f (grounding_g2); the voxel support pibar is still computed but not used by v8. Equal host weights,
support levels and the existence test are applied in scripts/g2_run.py (the 'pi' fields below are v7 quantities kept for ablation only).
Lesion ledger v7 (18 Oct 2026) -- candidate generation for the one-pass conditional inference of main.tex v7 (Sec. 2.1, 2.3).
Built from ledger_v6a (unchanged); differences:
  host reasoning   anatomy_qc_v7: mask confidence mu (feasible 1, flagged 1/2, corrupt/impossible 0); an oversized mask is flagged for
                   every organ; organ CORE (agreement of the most reliable masks, minus contested), uncertain boundary, contested;
                   organ states present / unsegmented (KB region, mu 1/2, ANY host) / out of view (no KB region in frame).
  claims           never dropped: prompt mismatch (< 50% inside the prompted host's envelope) and flatness are warning signals.
  units            outlines are split into extensions and LOBES (grounding_g1.split_lobes); a lobe is an anchor of its own.
  host candidates  touch generates candidates (envelope = territory + 10 mm; a KB region needs >= 50% of the anchor inside);
                   organs outside H_C that hold >= 50% of the anchor are candidates too (attribution 'other', w = 0.15).
                   No forced primary host. Compatibility features phi (grounding_g1.compat_features) on the LESION union.
  lesions          geometric: two-agent anchors chained within tau = 10 mm; a single-agent / lobe anchor joins the nearest group
                   within tau (no chaining through it); host-independent.
  priors           existence prior pi(a,h) = odds^-1(odds(pi_s) rho^n) with signals prompt mismatch, flatness, no confirmation,
                   no site agreement (distant host); extents X_a = {a, every outline (flagged ones kept), U, U - non-host organs}
                   with flags, mean voxel support and log volume ratio (extent prior assembled in g1_run with psi).
Set-based only (no image intensities). Returns (ledger dict, crops dict of Box, affine, shape)."""
from __future__ import annotations
import json, os
import numpy as np
from scipy import ndimage
from .adapter import _spacing, _labels_by_side, host_prior_region
from .ledger_v6 import Box, _load, _bool, _shape, _hk, _crop_to, _near, _claims
from . import anatomy_qc_v82 as AQ   # v8.2: anatomy_qc_v81 with a faster _mind (identical values)
from . import rspans as RS
from . import host_v81 as HV
from . import kb_relational as KR
from . import grounding_g1 as G1
from .planner import Planner as _Planner
from . import expected_v83 as EX

P = dict(inv_mm=3.0, min_ml=0.5, crumb_ml=0.2, crumb_mm=5.0, env_mm=10.0, tau_mm=10.0, band_mm=3.0, trace_mm=3.0,
         gate_inside=0.5, slab_cover=0.4, slab_aspect=0.25, organ_vol_ratio=0.6, organ_dice2d=0.6, organ_inside=0.8,
         contain_H=0.8, contain_frac=0.5, box_inside=0.5, acq_ratio=1.5, leak_frac=0.5, nonhost_inside=0.5, signal_frac=0.5,
         w_other=0.15, host_tol_mm=10.0)
VERSION = 'v8.3'
PRIOR = dict(primary=1.0, local_invasion=0.3, distant=0.15)


def _union_box(items):
    items = [(s, c) for s, c in items if s is not None and c is not None and c.any()]
    if not items:
        return None, None
    sl = tuple(slice(min(s[d].start for s, _ in items), max(s[d].stop for s, _ in items)) for d in range(3))
    out = np.zeros(tuple(x.stop - x.start for x in sl), bool)
    for s, c in items:
        out[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(s, sl))] |= c
    return sl, out


def _sub(c, sl):
    """crop of an item dict(sl, crop) on slice sl"""
    return _crop_to(c, sl)


def _pad(sl, sp, mm, shape):
    p = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    return tuple(slice(max(0, s.start - a), min(n, s.stop + a)) for s, a, n in zip(sl, p, shape))


_KRC = {}                                                                  # per-series cache for kb_region (cleared per build)


def build_ledger(rel, seg_root, kb, params=None):
    _KRC.clear()
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
    ax = int(np.argmax(sp)) if float(max(sp)) / float(min(sp)) >= p['acq_ratio'] else k
    tau = p['tau_mm']                                                     # v7: fixed in mm (not tied to slice spacing)
    ipa = [d for d in range(3) if d != ax]
    shape = md.shape
    primary_visible = not any(l.get('step') == 'primary_not_in_fov' for l in plan.get('log', []))
    flags_scan = []

    # ---- (0)-(1) image quality control and host reasoning (anatomy_qc_v7)
    tum = np.zeros(shape, np.int8)
    tum_files = [R('BiomedParse', 'Radiology', rel + '_tumor.nii.gz'), R('VoxTell', 'Radiology', rel + '_tumor.nii.gz')]
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
    _aqc = p.get('aq_cache')                                              # v8.2: optional cache of the (deterministic) anatomy QC
    if _aqc and os.path.exists(_aqc):
        import pickle; AQR = pickle.load(open(_aqc, 'rb'))
    else:
        AQR = AQ.analyse(kb, md, names, task, ts_img.affine, np.asarray(sp, float), plan, manifest, plan_dir, tumour=tum, image=X,
                         modality=modality, thresholds=thr,
                         primary={m['entity'] for m in manifest['masks'] if m['list'] == 'T' and m.get('host_class') == 'primary'})
        if _aqc:
            import pickle; pickle.dump(AQR, open(_aqc + '.tmp', 'wb'), protocol=5); os.replace(_aqc + '.tmp', _aqc)
    del tum, X

    # ---- hosts of H_C: state, zones, envelope
    hosts = {}
    for m in manifest['masks']:
        if m['list'] != 'T':
            continue
        key = (m['entity'], m.get('side'))
        h = hosts.setdefault(key, dict(entity=m['entity'], side=m.get('side'), cls=m.get('host_class'), prompt_files=[]))
        h['prompt_files'].append(m['file'])
    prim = [q for q, h in hosts.items() if h['cls'] == 'primary']
    # v8.1 fallback (user 21 Oct): no segmented organ in view with confidence 1 -> anatomy insufficient. Host inference is skipped:
    # the host is ASSUMED to be the planner's primary organ, side undetermined; the other planned hosts are not assessable.
    anat_insuff = not any(float(o.get('mu', 0) or 0) >= 1.0 for o in AQR['organs'].values())
    if anat_insuff and prim:
        pent = prim[0][0]; A = (pent, None)
        hosts = {A: dict(entity=pent, side=None, cls='primary', prompt_files=[f for h_ in hosts.values() for f in h_['prompt_files']], state='assumed', core=Box(None, shape), theta=Box(None, shape),
                         C=Box(None, shape), mu=0.0, E=Box(None, shape), w=1.0, also_distant=False, other=False,
                         aq=dict(agents=None, flags=['anatomy_insufficient'], mu=None), not_assessable=[_hk(q) for q in hosts if q[0] != pent])}
        prim = [A]
    spread = {q['entity']: q for q in EX.static_hosts(kb, prim[0][0], (plan.get('input') or {}).get('cancer_type'))} if prim else {}   # v8.3 (1)
    for h_ in hosts.values():
        h_.setdefault('source', 'planner')
    if prim and not anat_insuff:                                          # v8.2 step 2: static expected list from the KB (no reach test)
        pent, pside = prim[0]; PL = _Planner(kb)
        for e_id, h_ in spread.items():
            if e_id == pent:
                continue
            e_ = kb.entities.get(e_id)
            if not e_ or modality not in e_.get('modality', ['CT', 'MR']):
                continue
            tg = PL.spread_targets(dict(h_, reach=None), pent, pside)
            if not primary_visible and h_.get('cls') == 'local_invasion':
                dp = PRIOR['distant']
                tg = [(t[0], 'distant', dp) for t in tg] if h_.get('also_distant') else []
            for sd, cls_, _pr in tg:
                key = (e_id, sd)
                if key in hosts:
                    if hosts[key]['cls'] != cls_:
                        hosts[key]['cls_planner'] = hosts[key]['cls']; hosts[key]['cls'] = cls_
                    hosts[key]['source'] = 'planner+kb'
                else:
                    hosts[key] = dict(entity=e_id, side=sd, cls=cls_, prompt_files=[], source='kb')
    for key, h in hosts.items():
        if h.get('state') == 'assumed':
            continue
        o = AQR['organs'].get(key)
        present = o is not None and o['ref'].crop is not None and o.get('mu', 0) > 0
        h['w'] = float(PRIOR.get(h['cls'], 0.0)); h['also_distant'] = bool(spread.get(key[0], {}).get('also_distant'))
        h['other'] = False
        if present:
            h.update(state='present', core=o['core'], theta=o['territory'], C=o['contested'], mu=float(o['mu']))
            h['E'] = h['theta'].dilate(p['env_mm'], sp)
            h['aq'] = dict(agents=o['agents'], flags=o['flags'], mu=o['mu'])
        else:
            Hb = kb_region(kb, key[0], key[1], md, names, ts_img.affine, plan, k, organs=AQR['organs'])   # v8.2: footprint incl. agent organs
            h['region_src'] = 'slab'; h['region_info'] = None
            if Hb is not None and Hb.any():                                                # v8.1 step 2: KB relations to segmented landmarks
                Hr, info = kb_region_rel(kb, key[0], key[1], md, names, ts_img.affine, plan, AQR['organs'], Hb)
                if Hr is not None and Hr.any():
                    Hb = Hr; h['region_src'] = 'relational'; h['region_info'] = info
            if Hb is not None and Hb.any():
                reg = Box(Hb, shape)
                h.update(state='unsegmented', core=Box(None, shape), theta=reg, C=Box(None, shape), mu=0.5, E=reg)
            else:
                h.update(state='out_of_view', core=Box(None, shape), theta=Box(None, shape), C=Box(None, shape), mu=0.0, E=Box(None, shape))
            h['aq'] = dict(agents=(o or {}).get('agents'), flags=(o or {}).get('flags'), mu=0.5 if h['state'] == 'unsegmented' else 0.0)
    # organs outside H_C (attribution 'other'; also the non-host organs removed from U)
    host_ents = {q[0] for q in hosts}
    others = {}
    for key, o_ in AQR['organs'].items():
        if anat_insuff:
            break                                                        # fallback: no organ outside H_C is a candidate
        if key in hosts or key[0] in host_ents or any(AQ.related(kb, key[0], he) for he in host_ents):   # as v8.2 (non-host tissue)
            continue
        if kb.entities.get(key[0], {}).get('category') not in AQ.CRISP or o_['ref'].crop is None:
            continue
        # fix 3 (18 Oct): an organ outside H_C is a host CANDIDATE only with a fully plausible mask (mu = 1) and when its KB span is at
        # least half inside the scan frame (planner); otherwise its mask still counts as non-host tissue (leak, U - non-host)
        sh = RS.inside(kb, key[0], (plan.get('frame') or {}).get('span'))
        others[key] = dict(entity=key[0], side=key[1], cls='proximal', w=None, other=False, state='present', core=o_['core'],
                           theta=o_['territory'], C=o_['contested'], mu=float(o_.get('mu', 1.0)), E=None, source='proximal',
                           cand_ok=bool(float(o_.get('mu', 1.0)) >= 0.5))   # v8.2: confidence >= 1/2; entry is decided per lesion
    Eprim = np.zeros(shape, bool)
    for q in (prim if primary_visible else []):
        e = hosts[q]['E']
        if e.any():
            Eprim[e.sl] |= e.crop

    # ---- claims (never dropped; warning signals)
    small = dict(region=Eprim, min_vox=crumb_vox, group_mm=p['crumb_mm'], spacing=sp, touch=True) if Eprim.any() else None
    claims = []
    raw = {}

    def add(mask, agent, prompt, src):
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
    del raw
    for i, c in enumerate(claims):
        c['id'] = i; c['sig'] = []
    for c in claims:                                                     # prompt mismatch (was the prompt gate)
        if c['prompt'] != 'cancer' and not anat_insuff:                  # fallback: needs a host mask -> not evaluated
            E = hosts[c['prompt'][1]]['E']
            ins = int((c['crop'] & E[c['sl']]).sum()) / c['vox'] if E.any() else 0.0
            if ins < p['gate_inside']:
                c['sig'].append('prompt_mismatch')
    for c in claims:                                                     # flatness (was the slab rule)
        best_H, bo = None, 0
        for q, h in hosts.items():
            if h['state'] != 'present':
                continue
            o = int((c['crop'] & h['theta'][c['sl']]).sum())
            if o > bo:
                best_H, bo = h['theta'], o
        c['flat'] = False
        if best_H is not None:
            s_ = _shape(c, best_H, ax, ipa, sp, vox_ml)
            c['flat'] = bool(s_.get('cover_med') is not None and s_['cover_med'] >= p['slab_cover'] and (s_['aspect'] or 1) <= p['slab_aspect'])
    for c in claims:
        if c['flat'] and not any(o['agent'] != c['agent'] and not o['flat'] and int((_crop_to(o, c['sl']) & c['crop']).sum()) >= min_vox
                                 for o in claims):
            c['sig'].append('flat')
    keep = claims
    ledger = dict(version=VERSION, rel=rel, spacing=[float(x) for x in sp], tau_mm=tau, primary_visible=primary_visible, anatomy_insufficient=bool(anat_insuff),
                  hosts=[dict(host=_hk(q), cls=h['cls'], cls_planner=h.get('cls_planner'), source=h.get('source'), w=h['w'], state=h['state'], mu=h['mu'],
                              also_distant=h['also_distant'], aq=h.get('aq')) for q, h in hosts.items()],
                  not_assessable=[x for h in hosts.values() for x in (h.get('not_assessable') or [])],
                  flags=flags_scan, n_claims=len(claims), anchors=[], lesions=[],
                  qc=dict(state=AQR['state'], reasons=AQR['reasons'], f_away=AQR['f_away'], n_away=AQR['n_away'], noise=AQR['noise'],
                          failed=[e for e in AQR['expected'] if e['failed']], version='aq_v7'))
    if not keep:
        return ledger, {}, ts_img.affine, shape

    # ---- anchors (unchanged definition)
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
        for c in keep:
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
    TAf = np.zeros(shape, bool); TAf[W] = TA; del TA
    byid = {c['id']: c for c in keep}

    # ---- outlines and lobes (v7)
    # v8.2: resumable cache of the lobe splits (device calls are time-limited); calls are made in a deterministic order, so the
    # i-th call of a rerun gets the stored result of the i-th call (exact; speed only)
    import pickle, time as _t
    _lcf = p.get('lobe_cache'); _LC = dict(rel=rel, res={}, n=0, t=_t.time())
    if _lcf and os.path.exists(_lcf):
        try:
            _d = pickle.load(open(_lcf, 'rb'))
            if _d.get('rel') == rel:
                _LC['res'] = _d['res']
        except Exception:
            pass

    def _slc(om, am, sp_):
        i = _LC['n']; _LC['n'] += 1
        if i in _LC['res']:
            return _LC['res'][i]
        r_ = _split_lobes(om, am, sp_); _LC['res'][i] = r_
        _lc_save()
        return r_

    def _lc_save(force=False):
        if _lcf and (force or _t.time() - _LC['t'] > 4):
            pickle.dump(dict(rel=rel, res=_LC['res']), open(_lcf + '.tmp', 'wb'), protocol=5); os.replace(_lcf + '.tmp', _lcf); _LC['t'] = _t.time()
    def outlines_of(an):
        return [c['id'] for c in keep if bool((_crop_to(c, an['sl']) & an['crop']).any())]
    for an in anchors:
        an['outlines'] = outlines_of(an)
    AU = np.zeros(shape, bool)
    for an in anchors:
        AU[an['sl']] |= an['crop']
    new_lobes = []
    for an in anchors:
        an['ol_mask'] = {}
        for i in an['outlines']:
            c = byid[i]
            sl = _pad(_union_box([(c['sl'], c['crop']), (an['sl'], an['crop'])])[0], sp, 2, shape)
            om = _crop_to(c, sl); am = _crop_to(an, sl)
            kept, lobes = _slc(om, am, sp)
            an['ol_mask'][i] = (sl, kept)
            for lb_ in lobes:
                if int((lb_ & AU[sl]).sum()) >= 0.5 * lb_.sum():
                    continue
                if any(int((_crop_to(L0, sl) & lb_).sum()) >= 0.5 * lb_.sum() for L0 in new_lobes):
                    continue
                lsl = ndimage.find_objects(lb_.astype(np.int8))[0]
                new_lobes.append(dict(kind='lobe', agents=[c['agent']], sl=tuple(slice(s.start + o.start, s.stop + o.start) for s, o in zip(lsl, sl)),
                                      crop=lb_[lsl].copy(), vox=int(lb_.sum()), parent=i))
    for L0 in new_lobes:
        if L0['vox'] >= min_vox:
            L0['outlines'] = outlines_of(L0)
            L0['ol_mask'] = {}
            for i in L0['outlines']:
                c = byid[i]
                sl = _pad(_union_box([(c['sl'], c['crop']), (L0['sl'], L0['crop'])])[0], sp, 2, shape)
                kept, _ = _slc(_crop_to(c, sl), _crop_to(L0, sl), sp)
                L0['ol_mask'][i] = (sl, kept)
            anchors.append(L0)
            AU[L0['sl']] |= L0['crop']
    for i, an in enumerate(anchors):
        an['id'] = i

    # ---- claim organ-overlap flags (as v6a D10: TS organs and host territories)
    tsvol = np.bincount(md.ravel(), minlength=max(names) + 1 if names else 1)
    host_labels = set()
    lbs = {}
    for (ent, side) in hosts:
        if ent not in lbs:
            lbs[ent] = _labels_by_side(kb, task, names, ent)
        host_labels |= {int(l) for sd, ls in lbs[ent].items() for l in ls if side is None or sd in (side, None)}

    def organ_test(obj, H, hv=None):
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

    def organ_flags(obj):
        fl = []
        sub = md[obj['sl']][obj['crop']]
        ls, cts = np.unique(sub[sub > 0], return_counts=True)
        for l, ov in zip(ls, cts):
            if int(l) in host_labels:
                continue
            ov, ovol = int(ov), int(tsvol[l])
            if ov >= p['contain_H'] * ovol and ov >= p['contain_frac'] * obj['vox']:
                fl.append(f"{names.get(int(l), l)}:b")
            elif ov >= p['organ_inside'] * obj['vox'] and obj['vox'] >= p['organ_vol_ratio'] * ovol:
                t = organ_test(obj, Box(md == l, shape), ovol)
                if t:
                    fl.append(f"{names.get(int(l), l)}:{t}")
        for q, h in hosts.items():
            if h['state'] != 'present' or not bool((obj['crop'] & h['theta'][obj['sl']]).any()):
                continue
            t = organ_test(obj, h['theta'])
            if t:
                fl.append(f"{_hk(q)}:{t}")
        return sorted(set(fl))
    for c in keep:
        c['flag'] = organ_flags(c)

    # ---- host candidates per anchor (touch generates candidates)
    # v8.3: entrants = segmented KB anchors that are tumour hosts, not on the expected list (confidence >= 1/2)
    entrants = {}
    for key, o_ in AQR['organs'].items():
        if anat_insuff:
            break
        e_ = kb.entities.get(key[0], {})
        if key in hosts or o_['ref'].crop is None or not o_['territory'].any() or not e_.get('is_anchor') or not e_.get('tumour_host', True):
            continue
        if float(o_.get('mu', 0) or 0) < 0.5:
            continue
        entrants[key] = others.get(key) or dict(entity=key[0], side=key[1], cls='proximal', w=None, other=False, state='present', core=o_['core'],
                                                theta=o_['territory'], C=o_['contested'], mu=float(o_.get('mu', 1.0)), E=None, source='proximal', cand_ok=True)
    allh = dict(hosts); allh.update(others); allh.update(entrants)
    for an in anchors:
        sl, L = an['sl'], an['crop']
        cand = []
        for q, h in hosts.items():
            if h.get('state') == 'assumed':                                             # fallback: the assumed primary only
                cand.append(q)
            elif h['state'] == 'present' and h['theta'].any():                           # v8.1: no envelope gate
                cand.append(q)
            elif h['state'] == 'unsegmented' and h['theta'].any():                     # fix 13 (20 Oct): no inside gate; distance decides
                cand.append(q)
        an['cand'] = cand
        an['contested'] = any(allh[q]['C'].any() and bool((L & allh[q]['C'][sl]).any()) for q in cand)

    # ---- lesions: geometric, two-agent chaining only
    near_cache = {}

    def near(i):
        if i not in near_cache:
            an = anchors[i]; near_cache[i] = _near(an['crop'], an['sl'], shape, sp, tau)
        return near_cache[i]

    def close(i, j):
        A_, B_ = anchors[i], anchors[j]
        pad = [int(np.ceil(tau / float(v))) + 1 for v in sp]
        if any(A_['sl'][d].stop + pad[d] <= B_['sl'][d].start or B_['sl'][d].stop + pad[d] <= A_['sl'][d].start for d in range(3)):
            return False
        ps, nm = near(i)
        return bool((nm & _crop_to(B_, ps)).any())
    two = [an['id'] for an in anchors if an['kind'] == 'two']
    parent = {i: i for i in range(len(anchors))}

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a_i in range(len(two)):
        for b_i in range(a_i + 1, len(two)):
            if f(two[a_i]) != f(two[b_i]) and close(two[a_i], two[b_i]):
                parent[f(two[a_i])] = f(two[b_i])
    for an in anchors:
        if an['kind'] == 'two':
            continue
        best = None
        for j in two:
            if close(an['id'], j):
                d = abs(np.mean([s.start + s.stop for s in an['sl']]) - np.mean([s.start + s.stop for s in anchors[j]['sl']]))
                if best is None or d < best[0]:
                    best = (d, j)
        if best is not None:
            parent[an['id']] = f(best[1])
    groups = {}
    for an in anchors:
        groups.setdefault(f(an['id']), []).append(an['id'])
    lesions = [dict(anchors=sorted(g)) for g in groups.values()]
    near_cache.clear()
    les_of = {}
    for li, Lz in enumerate(lesions):
        for i in Lz['anchors']:
            les_of[i] = li

    _fill_cache = {}

    def _filled(q_, B_):
        if q_ not in _fill_cache:
            c_ = B_.crop.copy()
            for z_ in range(c_.shape[ax]):
                idx = [slice(None)] * 3; idx[ax] = z_
                c_[tuple(idx)] = ndimage.binary_fill_holes(c_[tuple(idx)])
            b_ = Box(None, shape); b_.sl, b_.crop = B_.sl, c_
            _fill_cache[q_] = b_
        return _fill_cache[q_]

    # ---- lesion host candidates and compatibility features (on the lesion union)
    for li, Lz in enumerate(lesions):
        ans = [anchors[i] for i in Lz['anchors']]
        sl, U = _union_box([(a['sl'], a['crop']) for a in ans])
        Lz['sl'], Lz['U'] = sl, U
        if ('L', li) in _LC['res']:                                           # v8.2: resumable (same deterministic computation)
            Lz.update(_LC['res'][('L', li)]['Lz']); allh.update(_LC['res'][('L', li)]['comp']); continue
        Lz['cand'] = sorted({q for a in ans for q in a['cand']}, key=str)
        ps = _pad(sl, sp, p['band_mm'] + 2, shape)
        Um = np.zeros(tuple(s.stop - s.start for s in ps), bool)
        Um[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(sl, ps))] = U
        oth = AU[ps] & ~Um
        Lz['phi'] = {}
        psD = _pad(sl, sp, HV.D_CAP + 1.0, shape)                                       # v8.1: distance map of the lesion union
        UmD = np.zeros(tuple(s.stop - s.start for s in psD), bool)
        UmD[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(sl, psD))] = U
        daD = G1.edt(~UmD, sp); del UmD
        for q in Lz['cand']:
            h = allh[q]
            ph = G1.compat_features(Um, h['core'][ps] if h['core'].any() else None, h['theta'][ps], h['mu'], sp, others=oth,
                                    kb=h.get('state') == 'unsegmented')
            d = HV.distance_mm(daD, h['theta'][psD]); ph['dist_mm'] = d; ph['ldist'] = float(np.log1p(d))
            Lz['phi'][q] = ph
        # ---- v8.2 step 3: entry of organs outside the static list (role 'proximal'), lesion level
        Lz['static'] = list(Lz['cand']); Lz['entry'] = {}
        dst = [Lz['phi'][q]['dist_mm'] for q in Lz['static']]
        Lz['d_static'] = min(dst) if dst else None
        Lz['contain'] = {}
        if not anat_insuff:
            nU = int(U.sum()); anyorg = np.zeros(U.shape, bool); shares = {}
            for q_, o_ in AQR['organs'].items():
                if o_['ref'].crop is None or float(o_.get('mu', 0) or 0) <= 0 or not o_['territory'].any():
                    continue
                fm = _filled(q_, o_['territory'])[sl] & U
                if fm.any():
                    shares[q_] = int(fm.sum()) / nU; anyorg |= fm
            out_share = 1.0 - int(anyorg.sum()) / nU
            Lz['contain'] = dict(sorted(((_hk(q_), round(v, 3)) for q_, v in shares.items()), key=lambda t: -t[1])[:4], outside=round(out_share, 3))
            if shares:
                top = max(shares.values()); win = [q_ for q_, v in shares.items() if v == top]
                if len(win) == 1 and top > out_share and win[0] in entrants and win[0] not in Lz['cand']:
                    Lz['entry'][win[0]] = 'contain'                                  # v8.3: plurality containment (always)
            for q_, h_ in entrants.items():                                         # v8.3: strictly closer than every expected host
                if q_ in Lz['entry'] or q_ in Lz['cand']:
                    continue
                d_ = HV.distance_mm(daD, h_['theta'][psD])
                if Lz['d_static'] is None or d_ < Lz['d_static']:
                    Lz['entry'][q_] = 'closer'
        if len(Lz['entry']) > 1:                                              # a part never enters beside its whole (same organ)
            vol_ = {q_: entrants[q_]['theta'].sum() for q_ in Lz['entry']}
            drop = {q_ for q_ in Lz['entry'] for q2 in Lz['entry'] if q2 != q_ and AQ.related(kb, q_[0], q2[0]) and vol_[q2] > vol_[q_]}
            Lz['entry'] = {q_: r_ for q_, r_ in Lz['entry'].items() if q_ not in drop}
        for q in Lz['entry']:
            h = entrants[q]
            ph = G1.compat_features(Um, h['core'][ps] if h['core'].any() else None, h['theta'][ps], h['mu'], sp, others=oth, kb=False)
            d = HV.distance_mm(daD, h['theta'][psD]); ph['dist_mm'] = d; ph['ldist'] = float(np.log1p(d))
            Lz['phi'][q] = ph; Lz['cand'].append(q)
        del daD
        # v8.1 step 4: local invasion as a RELATION. A lesion touching (<= inv_mm) the primary host p and one of its KB local-invasion
        # hosts n (planner spread_hosts) gets the composite candidate p>n: features on the union of the two organs' zones, same beta,
        # no extra weight. Its origin is p (reported host p, invading n).
        comps = []
        for q1 in Lz['cand']:
            h1 = allh[q1]
            if h1.get('cls') != 'primary' or h1.get('other') or h1.get('composite'): continue
            for q2 in Lz['cand']:
                h2 = allh[q2]
                if q2 == q1 or h2.get('cls') != 'local_invasion' or h2.get('other'): continue
                if Lz['phi'][q1]['dist_mm'] > p['inv_mm'] or Lz['phi'][q2]['dist_mm'] > p['inv_mm']: continue
                ck = (f'{q1[0]}>{q2[0]}', q1[1])
                if ck not in allh:
                    unseg = h1.get('state') == 'unsegmented' or h2.get('state') == 'unsegmented'
                    allh[ck] = dict(entity=ck[0], side=q1[1], cls='primary', composite=(q1, q2), other=False, also_distant=False,
                                    state='unsegmented' if unseg else 'present', w=h1.get('w'),
                                    core=AQ._union([h1['core'], h2['core']], shape), theta=AQ._union([h1['theta'], h2['theta']], shape),
                                    C=AQ._union([h1['C'], h2['C']], shape), mu=float(min(h1['mu'], h2['mu'])), E=None,
                                    aq=dict(agents=None, flags=[f'composite:{_hk(q1)}>{_hk(q2)}'], mu=float(min(h1['mu'], h2['mu']))))
                h = allh[ck]
                ph = G1.compat_features(Um, h['core'][ps] if h['core'].any() else None, h['theta'][ps], h['mu'], sp, others=oth,
                                        kb=h['state'] == 'unsegmented')
                d = min(Lz['phi'][q1]['dist_mm'], Lz['phi'][q2]['dist_mm']); ph['dist_mm'] = d; ph['ldist'] = float(np.log1p(d))
                Lz['phi'][ck] = ph; comps.append(ck)
        Lz['cand'] = Lz['cand'] + comps
        Lz['agents'] = sorted({x for a in ans for x in a['agents']})
        Lz['two'] = any(a['kind'] == 'two' for a in ans)
        Lz['ml'] = round(int(U.sum()) * vox_ml, 2)
        _LC['res'][('L', li)] = dict(Lz={k_: Lz[k_] for k_ in ('cand', 'phi', 'static', 'entry', 'd_static', 'contain', 'agents', 'two', 'ml')},
                                     comp={q_: allh[q_] for q_ in comps})
        _lc_save()

    def site_agree(Lz, q):
        sl, U = Lz['sl'], Lz['U']
        agree = np.zeros(U.shape, bool)
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
        return int(agree.sum()) * vox_ml

    # ---- anchor signals, existence prior per host, extents
    nonhost_ts = md > 0
    for an in anchors:
        if ('A', an['id']) in _LC['res']:                                     # v8.2: resumable (same deterministic computation)
            an.update(_LC['res'][('A', an['id'])]); an.pop('ol_mask', None); continue
        sl, L = an['sl'], an['crop']
        ol = [byid[i] for i in an['outlines']]
        own = [c for c in ol if c['agent'] in an['agents']]
        sig = []
        for s_name in ('prompt_mismatch', 'flat'):
            clean = np.zeros(L.shape, bool)
            for c in own:
                if s_name not in c['sig']:
                    clean |= _crop_to(c, sl)
            if int((clean & L).sum()) < p['signal_frac'] * an['vox']:
                sig.append(s_name)
        an['sig'] = sig
        Lz = lesions[les_of[an['id']]]
        an['pi'] = {}; an['sig_h'] = {}
        for q in Lz['cand'] or [None]:
            s_h = list(sig)
            if an['kind'] != 'two' and q is not None:
                if allh[q].get('E') is not None and allh[q]['E'].any():
                    env = allh[q]['E']
                else:                                                         # v8.2: dilation cached per host (speed only)
                    if allh[q].get('_Ec') is None:
                        allh[q]['_Ec'] = allh[q]['theta'].dilate(p['env_mm'], sp)
                    env = allh[q]['_Ec']
                if anat_insuff:                                                   # fallback: other agents anywhere in the scan
                    env = Box(np.ones(shape, bool), shape)
                others_ag = [x for x in agents if x not in an['agents']]
                present_ = [x for x in others_ag if env.any() and bool((raw_box[x][env.sl] & env.crop).any())]
                if present_:
                    ps, nm = _near(L, sl, shape, sp, p['trace_mm'])
                    if not any(bool((nm & raw_box[x][ps]).any()) for x in present_):
                        s_h.append('no_confirmation')
            if q is not None and allh[q]['cls'] == 'distant':
                if site_agree(Lz, q) < p['min_ml']:
                    s_h.append('no_site_agreement')
            pi = G1.existence_prior(an['kind'] == 'two', len(s_h))
            if q is None:
                pi = float(G1.inv_odds(G1.odds(pi) * G1.GP1['rho']))
            an['pi'][q] = pi; an['sig_h'][q] = s_h
        # extents
        cand_set = [q for q in Lz['cand']]
        eslices = [an['ol_mask'][i][0] for i in an['outlines']] + [sl]
        esl = tuple(slice(min(s[d].start for s in eslices), max(s[d].stop for s in eslices)) for d in range(3))
        a_m = _crop_to(an, esl)
        outl = {}
        for i in an['outlines']:
            s0, km = an['ol_mask'][i]
            m = np.zeros(tuple(s.stop - s.start for s in esl), bool)
            m[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(s0, esl))] = km
            outl[i] = m
        U = a_m.copy()
        for m in outl.values():
            U |= m
        allowed = np.zeros(U.shape, bool)
        for q in cand_set:
            allowed |= allh[q]['theta'][esl]
        nh = (nonhost_ts[esl] | np.any([allh[q]['theta'][esl] for q in allh if q not in cand_set and allh[q]['theta'].any()], axis=0)
              if any(allh[q]['theta'].any() for q in allh if q not in cand_set) else nonhost_ts[esl]) & ~allowed
        names_ = ['anchor'] + [f'o{i}' for i in an['outlines']] + ['U'] + ([] if anat_insuff else ['U-nh'])   # fallback: no non-host organs
        masks_ = [a_m] + [outl[i] for i in an['outlines']] + [U] + ([] if anat_insuff else [U & ~nh])
        ext = G1.merge_extents(names_, masks_)
        ta = TAf[esl]
        an['extents'] = []
        for nm_, m in ext:
            beyond = m & ~a_m
            leak = bool(beyond.sum() > 0 and int((beyond & nh).sum()) >= p['leak_frac'] * int(beyond.sum()))
            if nm_.startswith('o'):
                c = byid[int(nm_[1:])]; oflag = bool(c['flag']); csig = list(c['sig'])
            elif nm_ == 'anchor':
                oflag = False; csig = []
            else:
                obj = dict(sl=esl, crop=m, vox=int(m.sum())); oflag = bool(organ_flags(obj)); csig = []
                for i in an['outlines']:
                    if (outl[i] & m).any():
                        csig += byid[i]['sig']
            nflags = int(leak) + int(oflag) + int('prompt_mismatch' in csig) + int('flat' in csig)
            pib = float((0.9 * (m & ta).sum() + 0.5 * (m & ~ta).sum()) / max(int(m.sum()), 1))
            flags = [f_ for f_, on in (('leak', leak), ('organ', oflag), ('prompt_mismatch', 'prompt_mismatch' in csig),
                                       ('flat', 'flat' in csig)) if on]
            an['extents'].append(dict(name=nm_, sl=esl, crop=m, ml=round(int(m.sum()) * vox_ml, 3), leak=leak, organ=oflag, flags=flags,
                                      nflags=nflags, pibar=pib, logratio=float(np.log(max(int(m.sum()), 1) / max(an['vox'], 1)))))
        del an['ol_mask']
        _LC['res'][('A', an['id'])] = {k_: an[k_] for k_ in ('sig', 'pi', 'sig_h', 'extents')}
        _lc_save()

    # ---- outputs
    for an in anchors:
        Lz = lesions[les_of[an['id']]]
        ledger['anchors'].append(dict(
            id=an['id'], kind=an['kind'], agents='+'.join(an['agents']), ml=round(an['vox'] * vox_ml, 2),
            bbox=[[s.start, s.stop] for s in an['sl']], lesion=les_of[an['id']], cand=[_hk(q) for q in an['cand']],
            contested=an['contested'], sig=an['sig'], pi={(_hk(q) if q is not None else 'none'): round(v, 4) for q, v in an['pi'].items()},
            sig_h={(_hk(q) if q is not None else 'none'): v for q, v in an['sig_h'].items()},
            outlines=[dict(id=i, agent=byid[i]['agent'], prompt='cancer' if byid[i]['prompt'] == 'cancer' else _hk(byid[i]['prompt'][1]),
                           ml=round(byid[i]['vox'] * vox_ml, 2), flag=byid[i]['flag'], sig=byid[i]['sig']) for i in an['outlines']],
            extents=[{k: v for k, v in e.items() if k not in ('sl', 'crop')} for e in an['extents']]))
    for li, Lz in enumerate(lesions):
        ledger['lesions'].append(dict(id=li, anchors=Lz['anchors'], cand=[_hk(q) for q in Lz['cand']], two=Lz['two'],
                                      agents='+'.join(Lz['agents']), ml=Lz['ml'], bbox=[[s.start, s.stop] for s in Lz['sl']],
                                      phi={_hk(q): {k: round(v, 4) for k, v in ph.items()} for q, ph in Lz['phi'].items()},
                                      entry={_hk(q): r for q, r in Lz.get('entry', {}).items()}, d_static=Lz.get('d_static'), contain=Lz.get('contain')))
    crops = dict(anchors={an['id']: an for an in anchors}, lesions=lesions, hosts=allh, claims=byid, primary=prim, anatomy_insufficient=bool(anat_insuff),
                 host_cls={q: allh[q]['cls'] for q in allh}, ax=ax, si=k, vox_ml=vox_ml)
    return ledger, crops, ts_img.affine, shape

def kb_region(kb, ent, side, md, names, aff, plan, si, margin_levels=0.5, organs=None):
    """fix 1 (18 Oct 2026): region of an unsegmented organ = the slab of its OWN KB vertebral span (+ span variability + half a level)
    across the body footprint, restricted to its side for paired organs (midline and handedness reliable). The span is placed with
    the labelled vertebrae (CT: least-squares z per level) or, without them, linearly on the planner frame (scan extent = frame span).
    None (out of view) if the span is less than half inside the planner frame, or cannot be placed."""
    e = kb.entities.get(ent, {}); an = e.get('anchor') or {}; span = RS.span(kb, ent)     # v8 (20 Oct): reasoning-layer span
    frame = (plan.get('frame') or {}).get('span')
    shp = RS.inside(kb, ent, frame)
    if shp is None or shp < 0.5:
        return None
    try:
        a, b = kb.level_index(span[0]), kb.level_index(span[1])
    except ValueError:
        return None
    A = np.asarray(aff, float); n = md.shape[si]
    ctr = [s_ // 2 for s_ in md.shape]
    def zw(kidx):
        v = list(ctr); v[si] = kidx; return float((A @ np.r_[v, 1.0])[2])
    zk = np.array([zw(i) for i in range(n)])
    if 'lv' not in _KRC:
        _KRC['lv'] = AQ.vertebral_levels(names, md, aff)                     # v8.2: once per series (speed only)
    lv = {L: z for L, z in _KRC['lv'].items() if L in kb.vertebral_order}
    if len(lv) >= 2:
        xi = np.array([kb.level_index(L) for L in lv]); zi = np.array(list(lv.values()))
        cf = np.polyfit(xi, zi, 1); f = lambda li: cf[0] * li + cf[1]
    else:
        try:
            f0, f1 = kb.level_index(frame[0]), kb.level_index(frame[1])
        except ValueError:
            return None
        zs, zi_ = max(zk), min(zk)                      # superior scan edge = frame top level, inferior edge = frame bottom level
        f = lambda li: zs + (li - f0) * (zi_ - zs) / max(f1 - f0, 1)
    var = RS.var(kb, ent) + margin_levels
    z1, z2 = sorted([f(a - var), f(b + var)])
    keep = (zk >= z1) & (zk <= z2)
    if not keep.any():
        return None
    if 'foot' not in _KRC:
        lab_ = md > 0
        for o_ in (organs or {}).values():                                # v8.2: agent organ territories count as body (D4 footprint bug)
            t_ = o_.get('territory')
            if t_ is not None and t_.any() and float(o_.get('mu', 0) or 0) > 0:
                lab_[t_.sl] |= t_.crop
        _KRC['foot'] = ndimage.binary_fill_holes(ndimage.binary_closing(lab_.any(axis=si), iterations=5))
    foot = _KRC['foot'].copy()
    if side in ('left', 'right'):
        log = plan.get('log', [])
        mid = next((l for l in log if l.get('step') == 'midline'), {}); hd = next((l for l in log if l.get('step') == 'header_handedness'), {})
        if mid.get('x_mm') is not None and hd.get('agreement', True) is True:
            ip = [d for d in range(3) if d != si]
            g0, g1 = np.meshgrid(np.arange(md.shape[ip[0]]), np.arange(md.shape[ip[1]]), indexing='ij')
            v = np.zeros((3,) + g0.shape); v[ip[0]] = g0; v[ip[1]] = g1; v[si] = ctr[si]
            xw = A[0, 0] * v[0] + A[0, 1] * v[1] + A[0, 2] * v[2] + A[0, 3]
            dx = (hd.get('sign', 1) or 1) * (xw - mid['x_mm'])                 # + = patient left
            foot &= (dx > 0) if side == 'left' else (dx < 0)
    out = np.zeros(md.shape, bool)
    sh = [1, 1, 1]; sh[si] = n
    out |= keep.reshape(sh) & np.expand_dims(foot, si)
    return out if out.any() else None


def kb_region_rel(kb, ent, side, md, names, aff, plan, organs, slab, grid_mm=6.0):
    """v8.1 step 2: relational region (pancia_kb.kb_relational; cone semantics fitted on TotalSegmentator CT) inside the slab region.
    Landmarks: TotalSegmentator labels of the series and present organ-agent hosts (territories, mu > 0; e.g. rectum). Computed on a
    ~6 mm strided grid and expanded back (nearest), then AND the slab. Returns (region or None, info)."""
    A = np.asarray(aff, float); sp = np.sqrt((A[:3, :3] ** 2).sum(0))
    f = np.maximum(1, np.round(grid_mm / sp)).astype(int); st = tuple(slice(None, None, int(q)) for q in f)
    sub = md[st]; masks = {}
    for l, n in names.items():
        mm = sub == l
        if mm.any(): masks[n] = mm
    for key, o in (organs or {}).items():
        nm = key[0] if not key[1] else f'{key[0]}_{key[1]}'
        if nm in masks or o.get('mu', 0) <= 0 or o.get('territory') is None or not o['territory'].any(): continue
        full = np.zeros(md.shape, bool); t = o['territory']; full[t.sl] = t.crop; mm = full[st]; del full
        if mm.any(): masks[nm] = mm
    if not masks:
        return None, dict(used=[])
    log = plan.get('log', [])
    mid = next((l for l in log if l.get('step') == 'midline'), {}); hd = next((l for l in log if l.get('step') == 'header_handedness'), {})
    if mid.get('x_mm') is None:
        return None, dict(used=[], note='no midline')
    lr_ok = hd.get('agreement', True) is True; lsign = float(hd.get('sign', 1) or 1)
    As = A.copy(); As[:3, :3] = A[:3, :3] * f
    base = slab[st]
    try:
        R_, info = KR.region(kb, ent, side, masks, As, float(mid['x_mm']), lsign, base=base, lr_ok=lr_ok, return_info=True)
    except Exception as ex:
        return None, dict(used=[], note=repr(ex)[:80])
    if R_ is None:
        return None, info
    up = R_
    for d, q in enumerate(f):
        up = np.repeat(up, int(q), axis=d)
    up = up[:md.shape[0], :md.shape[1], :md.shape[2]]
    pad = [(0, max(0, n - u)) for n, u in zip(md.shape, up.shape)]
    if any(p_[1] for p_ in pad): up = np.pad(up, pad, mode='edge')
    return up & slab, info


def _split_lobes(omega, a, sp, open_mm=None, min_ml=None):
    """grounding_g1.split_lobes (unchanged there) with the distance map to the anchor computed once instead of once per opened
    component (identical result; speed only, v8.1 pilot 20 Oct)"""
    from scipy import ndimage as _nd
    open_mm = G1.GP1['lobe_open_mm'] if open_mm is None else open_mm; min_ml = G1.GP1['lobe_min_ml'] if min_ml is None else min_ml
    r = omega & ~a
    if not r.any():
        return omega, []
    ro = _nd.binary_opening(r, structure=G1.ball(open_mm, sp))
    if not ro.any():
        return omega, []
    lab, n = _nd.label(ro, structure=np.ones((3, 3, 3)))
    adj = _nd.binary_dilation(a, structure=np.ones((3, 3, 3)))
    lobes = []; keep = omega.copy(); da = None
    for i in range(1, n + 1):
        c = lab == i
        if (c & adj).any() or G1.vol_ml(c, sp) < min_ml:
            continue
        if da is None:
            da = G1.edt(~a, sp)
        dc = G1.edt(~c, sp); other = ro & ~c
        do = G1.edt(~other, sp) if other.any() else np.full(r.shape, np.inf)
        part = r & (dc < da) & (dc <= do)
        if G1.vol_ml(part, sp) >= min_ml:
            lobes.append(part); keep &= ~part
    return keep, lobes
