"""Deterministic per-scan prompt planner.

Input : TS output summary (which classes found, volumes, centroids in mm, truncation), modality, cancer type,
        optional sex (recorded in the plan for validation only; never used to include/exclude anchors — avoids bias
        from missing/incorrect records and from assuming anatomy from a label; existence is decided on the image).
Output: Plan with tiered PromptJobs and a decision log. No model calls; pure KB logic.
SINGLE PASS: the whole plan is compiled from TS + KB and sent to VoxTell in ONE call per scan. Tiers only order and
budget prompts (0 ruler, 1 completion, 2 profile); nothing is re-planned after VoxTell.

Steps
 1. frame        — vertebral span of the FOV from TS vertebrae, else fallback landmarks, else provisional span + tier-0 ruler prompts
 2. anchors      — TS-found classes → KB anchors (with plausibility), expected set from spans/presence, missing = expected − found
 3. tier 1       — completion prompts for missing anchors under spatial-prior gates
 4. tier 2       — profile prompts (List A refine / List B extend) for host, TS-unnameable missing, tumour-near and systemic anchors
 5. budget       — priority cap, dedup, log
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import json
from .kb import KnowledgeBase


@dataclass
class TSStructure:
    ts_name: str                 # e.g. 'kidney_left'
    volume_ml: float
    centroid_mm: Tuple[float, float, float]   # RAS
    truncated: bool = False
    zmin_mm: float = 0.0
    zmax_mm: float = 0.0


@dataclass
class TSOutput:
    task: str                    # 'total' | 'total_mr'
    structures: List[TSStructure]
    fov_z_mm: Tuple[float, float]
    tumour_centroid_mm: Optional[Tuple[float, float, float]] = None
    tumour_host_guess: Optional[str] = None     # PRIMARY host (cancer type / explicit host)
    hosts: List[dict] = field(default_factory=list)   # weighted host set from adapter: [{entity, cls, prior, evidence, weight, region}], primary first

    def by_name(self) -> Dict[str, TSStructure]:
        return {s.ts_name: s for s in self.structures}


@dataclass
class PromptJob:
    entity: str
    side: Optional[str]
    terms: List[str]
    tier: int                    # 0 ruler | 1 completion | 2 profile — ordering/budget only; ONE VoxTell call
    list: str                    # 'A' refine | 'B' extend | 'R' ruler | 'T' tumour
    gate: dict
    priority: int
    reason: List[str]
    relation: Optional[str] = None
    from_anchor: Optional[str] = None


@dataclass
class Plan:
    frame: dict
    hosts: List[dict]
    anchors_found: List[dict]
    anchors_expected: List[str]
    anchors_missing: List[dict]
    anchors_absent_candidates: List[str]
    prompts: List[PromptJob]
    log: List[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(dict(frame=self.frame, hosts=self.hosts, anchors_found=self.anchors_found, anchors_expected=self.anchors_expected,
                               anchors_missing=self.anchors_missing, anchors_absent_candidates=self.anchors_absent_candidates,
                               prompts=[asdict(p) for p in self.prompts], log=self.log), indent=1)


class Planner:
    def __init__(self, kb: KnowledgeBase, tumour_near_mm: float = 40.0):
        self.kb = kb
        self.tumour_near_mm = tumour_near_mm
        self.systemic = ['spine', 'aorta', 'inferior_vena_cava', 'skeletal_muscle', 'subcutaneous_fat', 'visceral_fat']

    # ------------------------------------------------------------ 1. frame
    def estimate_frame(self, ts: TSOutput, log: List[dict]) -> dict:
        kb = self.kb
        found = ts.by_name()
        levels = []
        for lvl in kb.vertebral_order:
            n = f'vertebrae_{lvl}'
            if n in found:
                levels.append((lvl, found[n].centroid_mm[2]))
        if len(levels) >= 2:
            # z increases superiorly in RAS: map FOV z-range to levels by nearest vertebra, extrapolating one level beyond
            levels.sort(key=lambda x: -x[1])
            top, bot = levels[0][0], levels[-1][0]
            log.append(dict(step='frame', method='ts_vertebrae', top=top, bottom=bot, n=len(levels)))
            return dict(span=[top, bot], method='ts_vertebrae', confidence='high')
        # fallback landmarks
        est = []
        for lm in kb.landmarks.values():
            ent = kb.entities[lm['entity']]
            names = []
            for task, v in (ent.get('ts') or {}).items():
                names += list(v.values()) if isinstance(v, dict) else [v]
            hit = [found[n] for n in names if n in found]
            if hit:
                use = lm.get('use', 'centroid')
                zs = [h.zmax_mm if use == 'top' else h.zmin_mm if use == 'bottom' else h.centroid_mm[2] for h in hit]
                if use != 'centroid' and all(z == 0.0 for z in zs):
                    zs = [h.centroid_mm[2] for h in hit]   # zmin/zmax not supplied
                est.append((lm['level'], sum(zs) / len(zs), lm['id']))
        if len(est) >= 1:
            est.sort(key=lambda x: -x[1])
            # extend by FOV extent assuming ~30 mm per level
            zt, zb = ts.fov_z_mm[1], ts.fov_z_mm[0]
            top_i = max(0, kb.level_index(est[0][0]) - int(round((zt - est[0][1]) / 30)))
            bot_i = min(len(kb.vertebral_order) - 1, kb.level_index(est[-1][0]) + int(round((est[-1][1] - zb) / 30)))
            span = [kb.vertebral_order[top_i], kb.vertebral_order[bot_i]]
            log.append(dict(step='frame', method='fallback_landmarks', landmarks=[e[2] for e in est], span=span))
            return dict(span=span, method='fallback_landmarks', confidence='medium')
        log.append(dict(step='frame', method='none', action='tier0_ruler_prompts'))
        return dict(span=None, method='none', confidence='low')

    # ------------------------------------------------------------ 2. anchors
    def map_found(self, ts: TSOutput, log) -> List[dict]:
        kb = self.kb
        out = []
        m = kb.ts_to_entity.get(ts.task, {})
        for s in ts.structures:
            if s.ts_name not in m:
                continue
            eid, side = m[s.ts_name]
            e = kb.entities[eid]
            plaus = True; why = []
            vr = e.get('volume_ml')
            if vr and not (0.5 * vr['min'] <= s.volume_ml <= 2.0 * vr['max']):
                plaus = False; why.append(f'volume {s.volume_ml:.0f} ml outside [{0.5*vr["min"]:.0f},{2*vr["max"]:.0f}]')
            if side and e['laterality'] == 'bilateral':
                x = s.centroid_mm[0]   # RAS: +x = patient left
                if (side == 'left' and x < 0) or (side == 'right' and x > 0):
                    plaus = False; why.append('laterality mismatch vs affine')
            out.append(dict(entity=eid, side=side, ts_name=s.ts_name, volume_ml=s.volume_ml, truncated=s.truncated, plausible=plaus, why=why, is_anchor=e.get('is_anchor', False)))
            if not plaus:
                log.append(dict(step='found_plausibility', entity=eid, side=side, fail=why))
        return out

    def expected_anchors(self, frame: dict, sex: Optional[str], log) -> List[str]:
        kb = self.kb
        if not frame.get('span'):
            return []
        exp = []
        for e in kb.anchors():
            a = e['anchor']
            if not kb.span_overlaps(a['span'], frame['span']):
                continue
            exp.append(e['id'])   # presence class (incl. sex_*) is NOT used to exclude: the image decides, sex is metadata only
        log.append(dict(step='expected', n=len(exp), sex_recorded=sex, sex_used_for_filtering=False, span=frame['span']))
        return exp

    # ------------------------------------------------------------ gates
    def gate_from_prior(self, entity_id: str, side: Optional[str], found_index: Dict[Tuple[str, Optional[str]], dict]) -> dict:
        """Compile the spatial prior into a gate spec referencing found landmarks.
        Gates are OUTPUT filters: VoxTell always runs on the full volume; the gate is applied afterwards to keep only
        connected components inside the region (see prompt_rules.gate_as_output_filter)."""
        priors = self.kb.spatial_prior_for(entity_id)
        usable = []
        for p in priors:
            lm = p['landmark']
            cand = [k for k in found_index if k[0] == lm]
            if cand:
                usable.append(dict(landmark=lm, relation=p['relation'], dist_mm=p['dist_mm'], sides=[c[1] for c in cand]))
        if usable:
            return dict(kind='spatial_prior', constraints=usable, fallback_region=self._region_of(entity_id))
        return dict(kind='region', region=self._region_of(entity_id))

    def _region_of(self, entity_id: str) -> Optional[str]:
        for r in self.kb.out_edges.get(entity_id, []):
            if r['type'] == 'lies_in':
                return r['dst']
        return None

    def gate_from_relation(self, rel: dict, anchor_side: Optional[str], found_index) -> dict:
        rt = self.kb.relation_types[rel['type']]['gate']
        g = dict(rt)
        g['anchor'] = rel['src']; g['anchor_side'] = anchor_side
        if rel['type'] == 'adjacent_to':
            key = [k for k in found_index if k[0] == rel['dst']]
            g['neighbour_found'] = bool(key); g['direction'] = rel.get('direction')
        return g

    # ------------------------------------------------------------ main
    def plan(self, ts: TSOutput, modality: str, cancer_type: Optional[str] = None, sex: Optional[str] = None,
             include_tumour_prompt: bool = False) -> Plan:
        """include_tumour_prompt: add the cancer-type tumour phrase to the anatomy call. Default False because the
        initial VoxTell tumour mask (step 1) already exists and a re-issued VoxTell prompt is NOT an independent rater
        (same model, same image; only the wording differs). Set True only when no initial VT tumour mask is available."""
        kb = self.kb; log = []
        frame = self.estimate_frame(ts, log)
        found = self.map_found(ts, log)
        found_index = {(f['entity'], f['side']): f for f in found if f['plausible']}
        if frame.get('method') == 'ts_vertebrae':
            found_index[('spine', None)] = dict(entity='spine', side=None, ts_name='vertebrae_*', plausible=True, is_anchor=True, volume_ml=None, truncated=False, why=[])
        prompts: List[PromptJob] = []

        # tier 0: ruler if frame unknown
        if not frame.get('span'):
            for eid in ['spine', 'sacrum', 'hip']:
                for side in (['left', 'right'] if kb.entities[eid]['laterality'] == 'bilateral' else [None]):
                    prompts.append(PromptJob(eid, side, kb.voxtell_terms(eid, side), 0, 'R', dict(kind='none'), 1, ['frame unknown → ruler prompt']))
            frame = dict(span=['T10', 'coccyx'], method='assumed_after_ruler', confidence='low')   # provisional wide span
            log.append(dict(step='frame', note='provisional wide span; tier-0 ruler masks are used by the reasoning layer for frame/laterality checks, no re-planning'))

        expected = self.expected_anchors(frame, sex, log)
        found_anchor_ids = {f['entity'] for f in found if f['plausible'] and f['is_anchor']}
        missing, absent_candidates = [], []
        for eid in expected:
            e = kb.entities[eid]
            sides = ['left', 'right'] if e['laterality'] == 'bilateral' else [None]
            if all((eid, s) in found_index for s in sides):
                continue
            if e['anchor']['presence'] in ('surgical', 'sex_female', 'sex_male', 'variable'):   # empty return = absence, not failure
                absent_candidates.append(eid)
            for side in sides:
                if (eid, side) in found_index:
                    continue
                gate = self.gate_from_prior(eid, side, found_index)
                ts_has_class = bool((e.get('ts') or {}).get(ts.task))
                reason = [f'expected in FOV {frame["span"]} (presence={e["anchor"]["presence"]})', 'TS class exists but missing/implausible' if ts_has_class else 'not nameable by TS in this task']
                pr = e.get('priority', 2) + (1 if ts_has_class else 0)   # TS-nameable-but-missing: usually FOV edge; lower priority
                missing.append(dict(entity=eid, side=side, gate=gate, ts_has_class=ts_has_class))
                prompts.append(PromptJob(eid, side, kb.voxtell_terms(eid, side), 1, 'B', gate, min(pr, 3), reason))
        log.append(dict(step='completion', missing=[(m['entity'], m['side']) for m in missing]))

        # host resolution (+ side from tumour centroid when the host is bilateral)
        host = ts.tumour_host_guess
        host_side = None
        if not host and cancer_type and cancer_type in kb.tumour_prompts['cancer_types']:
            host = kb.tumour_prompts['cancer_types'][cancer_type].get('host')
        if host and kb.entities[host]['laterality'] == 'bilateral' and ts.tumour_centroid_mm:
            cands = [(f, ts.by_name()[f['ts_name']]) for f in found if f['entity'] == host and f['plausible']]
            if cands:
                host_side = min(cands, key=lambda fs: sum((a - b) ** 2 for a, b in zip(fs[1].centroid_mm, ts.tumour_centroid_mm)))[0]['side']
            else:
                # no TS mask of the host (e.g. breast): side from the tumour x offset to the midline (spine/aorta/sacrum if found), only when clearly lateral
                mid = [ts.by_name()[f['ts_name']].centroid_mm[0] for f in found if f['entity'] in ('spine', 'aorta', 'sacrum')]
                dx = ts.tumour_centroid_mm[0] - (sum(mid) / len(mid) if mid else 0.0)
                host_side = ('left' if dx > 0 else 'right') if abs(dx) >= 30 else None
        # no validated tumour evidence on a bilateral host → both sides are prompted (never guess a side)
        log.append(dict(step='host', host=host, side=host_side or ('both' if host and kb.entities[host]['laterality'] == 'bilateral' else None),
                        source='tumour_evidence' if ts.tumour_centroid_mm else 'cancer_type'))

        # tumour prompts (only when no initial VT tumour mask exists — see include_tumour_prompt)
        if host and include_tumour_prompt:
            ct = kb.tumour_prompts['cancer_types'].get(cancer_type or '', {})
            phrases = ct.get('phrases') or [t.replace('{host}', kb.entities[host]['name'].lower()) for t in kb.tumour_prompts['generic_templates']]
            sides = [host_side] if host_side else (['left', 'right'] if kb.entities[host]['laterality'] == 'bilateral' else [None])
            for side in sides:
                terms = [p.replace('{side}', side or '').replace('  ', ' ').strip() for p in phrases[:3]]
                prompts.append(PromptJob(host, side, terms, 2, 'T', dict(kind='inside', anchor=host, dilate_mm=20), 1, ['tumour prompt on host']))

        # tier 2: profiles
        # promote only anchors TS cannot name (or the host); TS-nameable-but-missing anchors are usually FOV-edge and get no profile expansion
        promoted = [(m['entity'], m['side']) for m in missing if (not m.get('ts_has_class') or m['entity'] == host) and m['entity'] not in self.systemic]
        near = []
        if ts.tumour_centroid_mm:
            for f in found:
                if not f['is_anchor'] or f['entity'] == host: continue
                s = ts.by_name()[f['ts_name']]
                d = sum((a - b) ** 2 for a, b in zip(s.centroid_mm, ts.tumour_centroid_mm)) ** 0.5
                if d <= self.tumour_near_mm + 60:    # centroid distance is coarse; run-time uses surface distance
                    near.append((f['entity'], f['side']))
        # secondary hosts (spread): kept candidates other than the primary → 1-hop profile each
        hosts_out = []
        secondary = []
        for h in (ts.hosts or []):
            if h['entity'] == host:
                hosts_out.append(dict(h, side=host_side, role='primary')); continue
            role = h.get('role', 'secondary')
            hosts_out.append(dict(h, side=None, role=role))
            if role == 'secondary':
                secondary.append(h)          # contact_check hosts get no prompts: the reasoning layer tests tumour–organ contact on TS masks
        if host and not hosts_out:
            hosts_out.append(dict(entity=host, cls='primary', prior=1.0, evidence=None, weight=1.0, side=host_side, role='primary'))
        log.append(dict(step='hosts', hosts=[(h['entity'], h['role'], h['cls'], h.get('evidence'), h.get('weight')) for h in hosts_out]))
        targets = []
        if host: targets.append((host, host_side, 2, 'host'))
        for h in secondary:
            sides = ['left', 'right'] if kb.entities[h['entity']]['laterality'] == 'bilateral' else [None]
            for sd in sides:
                targets.append((h['entity'], sd, 1, f"secondary_host[{h['cls']},ev={h.get('evidence')}]"))
        for eid, side in promoted: targets.append((eid, side, 1 if eid != host else 2, 'promoted'))
        for eid, side in near: targets.append((eid, side, 1, 'tumour_near'))
        seen = set()
        earlier = {(p.entity, p.side): p for p in prompts}          # tier-0/1/tumour prompts already planned
        ts_only = []
        for eid, side, hops, why in targets:
            # secondary hosts: extent-relevant relations only (capsule/fascia, vessels, nodes) — no sub-parts (e.g. liver segments)
            types = {'invested_by', 'supplied_by', 'drained_by', 'drains_lymph_to', 'adjacent_to'} if why.startswith('secondary_host') else None
            for r in kb.profile(eid, hops=hops, types=types):
                dst = kb.entities.get(r['dst'])
                if not dst: continue
                if modality not in dst.get('modality', ['CT', 'MR']): continue
                bilateral = dst['laterality'] == 'bilateral'
                sl = r.get('side_link', 'same')
                if sl in ('left', 'right'):
                    if side and side != sl: continue            # relation only holds on one side of the anchor
                    dsides = [sl] if bilateral else [None]
                elif sl == 'same' and side and bilateral:
                    dsides = [side]
                else:
                    dsides = ['left', 'right'] if bilateral else [None]
                for ds in dsides:
                    key = (r['dst'], ds)
                    if key in seen: continue
                    seen.add(key)
                    pr = r.get('priority') or kb.relation_types[r['type']]['default_priority']
                    if dst.get('coverage_tier') == 'ood': pr = max(pr, 3)
                    if why == 'host': pr = max(1, pr - 1)          # host-derived structures matter most
                    elif why.startswith('secondary_host') and ('ev=1' in why or 'ev=2' in why): pr = max(1, pr)   # evidenced secondary host keeps its default priority
                    else: pr = max(pr, 2)                           # everything not derived from the host is at most priority 2
                    ts_found = (r['dst'], ds) in found_index
                    if key in earlier:                                # already a completion prompt: merge reason + priority, no duplicate
                        ep = earlier[key]
                        ep.priority = min(ep.priority, pr)
                        ep.reason.append(f"also {why} profile: {r['src']} -{r['type']}-> {r['dst']}")
                        if ep.relation is None: ep.relation, ep.from_anchor = r['type'], eid
                        continue
                    if ts_found and pr > 1:
                        ts_only.append(dict(entity=r['dst'], side=ds, relation=r['type'], from_anchor=eid))   # keep TS mask, no VoxTell prompt
                        continue
                    listing = 'A' if ts_found else 'B'
                    gate = self.gate_from_relation(r, side, found_index) if listing == 'B' else dict(kind='ts_bbox', ts_entity=r['dst'], side=ds, dilate_mm=10)
                    prompts.append(PromptJob(r['dst'], ds, kb.voxtell_terms(r['dst'], ds), 2, listing, gate, pr,
                                             [f'{why} anchor {eid}', f"{r['src']} -{r['type']}-> {r['dst']} (hop {r['hop']})"] + ([f"staging: {','.join(r['staging'])}"] if r.get('staging') else []),
                                             relation=r['type'], from_anchor=eid))
        # budget
        cap = kb.prompt_rules['budget']['max_prompts_per_scan']
        prompts.sort(key=lambda p: (0 if p.list in ('T', 'R') else 1, p.tier, p.priority))
        log.append(dict(step='ts_only', n=len(ts_only), items=[(t['entity'], t['side']) for t in ts_only]))
        log.append(dict(step='tumour_prompt', included=include_tumour_prompt, host=host))
        # completion (tier 0/1) and tumour prompts are never dropped: they decide which anchors exist; the cap trims tier 2 only
        protected = [p for p in prompts if p.tier < 2 or p.list in ('T', 'R')]
        w2 = [p for p in prompts if not (p.tier < 2 or p.list in ('T', 'R'))]
        room = max(0, cap - len(protected))
        kept, dropped = protected + w2[:room], w2[room:]
        if dropped:
            log.append(dict(step='budget', dropped=[(p.entity, p.side, p.priority) for p in dropped]))
        return Plan(frame=frame, hosts=hosts_out, anchors_found=found, anchors_expected=expected, anchors_missing=missing,
                    anchors_absent_candidates=absent_candidates, prompts=kept, log=log)
