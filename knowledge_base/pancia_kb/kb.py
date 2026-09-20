"""Knowledge-base loader, indexer and validator."""
from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class KnowledgeBase:
    meta: dict
    entities: Dict[str, dict]
    relations: List[dict]
    relation_types: Dict[str, dict]
    regions: Dict[str, dict]
    landmarks: Dict[str, dict]
    tumour_prompts: dict
    prompt_rules: dict
    qc_rules: dict
    vertebral_order: List[str]
    # indexes
    out_edges: Dict[str, List[dict]] = field(default_factory=dict)
    in_edges: Dict[str, List[dict]] = field(default_factory=dict)
    ts_to_entity: Dict[str, Dict[str, tuple]] = field(default_factory=dict)  # task -> ts_name -> (entity_id, side)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str) -> 'KnowledgeBase':
        def rd(name):
            with open(os.path.join(path, name)) as f:
                return yaml.safe_load(f)
        meta = rd('kb_meta.yaml')
        ents = {e['id']: e for e in rd('entities.yaml')['entities']}
        rels = rd('relations.yaml')['relations']
        rtypes = {t['id']: t for t in rd('relation_types.yaml')['relation_types']}
        regions = {r['id']: r for r in rd('regions.yaml')['regions']}
        lm = rd('landmarks.yaml')
        kb = cls(meta=meta, entities=ents, relations=rels, relation_types=rtypes, regions=regions,
                 landmarks={l['id']: l for l in lm['landmarks']}, tumour_prompts=rd('tumour_prompts.yaml'),
                 prompt_rules=rd('prompt_rules.yaml'), qc_rules=rd('qc_rules.yaml'), vertebral_order=lm['vertebral_order'])
        kb._index()
        return kb

    def _index(self):
        self.out_edges = {i: [] for i in self.entities}
        self.in_edges = {i: [] for i in self.entities}
        for r in self.relations:
            self.out_edges[r['src']].append(r)
            if r['dst'] in self.in_edges:
                self.in_edges[r['dst']].append(r)
        self.ts_to_entity = {}
        for e in self.entities.values():
            for task, names in (e.get('ts') or {}).items():
                m = self.ts_to_entity.setdefault(task, {})
                if isinstance(names, dict):
                    for side, n in names.items():
                        m[n] = (e['id'], side)
                else:
                    m[names] = (e['id'], None)

    # ------------------------------------------------------------ queries
    def anchors(self) -> List[dict]:
        return [e for e in self.entities.values() if e.get('is_anchor')]

    def level_index(self, lvl: str) -> int:
        return self.vertebral_order.index(lvl)

    def span_overlaps(self, span: List[str], fov: List[str], min_levels: int = 2) -> bool:
        """True if the anchor's canonical span shares at least min(min_levels, span length) vertebral levels with the FOV."""
        a0, a1 = self.level_index(span[0]), self.level_index(span[1])
        f0, f1 = self.level_index(fov[0]), self.level_index(fov[1])
        overlap = min(a1, f1) - max(a0, f0) + 1
        return overlap >= min(min_levels, a1 - a0 + 1)

    def profile(self, anchor_id: str, hops: int = 2, types: Optional[Set[str]] = None) -> List[dict]:
        """Typed edges reachable from the anchor within `hops` (following src→dst), de-duplicated."""
        seen, frontier, out = set(), {anchor_id}, []
        for h in range(hops):
            nxt = set()
            for a in frontier:
                for r in self.out_edges.get(a, []):
                    if r['type'] in ('lies_in', 'landmark_for'):
                        continue
                    if types and r['type'] not in types:
                        continue
                    key = (r['src'], r['type'], r['dst'])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(dict(r, hop=h + 1))
                    if h == 0 and r['type'] in ('has_part', 'supplied_by', 'drained_by', 'has_duct', 'invested_by'):
                        nxt.add(r['dst'])   # only follow structural edges into a second hop
            frontier = nxt
        return out

    def spread_hosts(self, primary: str, cancer_type: Optional[str] = None) -> List[dict]:
        """Weighted candidate hosts from the data-agnostic prior: primary, local-invasion neighbours (adjacent_to /
        invested_by anchors of the primary, derived from the relation graph) and cancer-specific distant sites."""
        sp = self.tumour_prompts.get('spread', {})
        prior = sp.get('prior', dict(primary=1.0, local_invasion=0.3, distant=0.15))
        out = [dict(entity=primary, cls='primary', prior=prior['primary'])]
        seen = {primary}
        for r in self.out_edges.get(primary, []):
            d = self.entities.get(r['dst'])
            if r['type'] in ('adjacent_to', 'invested_by') and d and d.get('is_anchor') and r['dst'] not in seen:
                seen.add(r['dst']); out.append(dict(entity=r['dst'], cls='local_invasion', prior=prior['local_invasion'], via=r['type']))
        distant = (sp.get('distant_by_cancer') or {}).get(cancer_type or '', sp.get('distant_default', []))
        for e in distant:
            if e in self.entities and self.entities[e].get('is_anchor') and e not in seen:
                seen.add(e); out.append(dict(entity=e, cls='distant', prior=prior['distant']))
        return out

    def spatial_prior_for(self, entity_id: str) -> List[dict]:
        e = self.entities[entity_id]
        return (e.get('anchor') or {}).get('spatial_prior', [])

    def voxtell_terms(self, entity_id: str, side: Optional[str] = None) -> List[str]:
        v = self.entities[entity_id]['voxtell']
        terms = [v['main']] + list(v.get('aliases', []))
        n = self.prompt_rules['alias_ensemble']['n_aliases']
        terms = terms[:n]
        if side:
            terms = [t.replace('{side}', side) for t in terms]
        else:
            terms = [t.replace('{side} ', '').replace('{side}', '') for t in terms]
        return terms

    # ------------------------------------------------------------ validation
    def validate(self) -> List[str]:
        errs = []
        ids = set(self.entities)
        vo = set(self.vertebral_order)
        for r in self.relations:
            if r['src'] not in ids: errs.append(f"relation src not found: {r['src']}")
            if r['dst'] not in ids and not (r['type'] == 'lies_in' and r['dst'] in self.regions):
                errs.append(f"relation dst not found: {r['src']} -{r['type']}-> {r['dst']}")
            if r['type'] not in self.relation_types: errs.append(f"unknown relation type {r['type']}")
        for e in self.entities.values():
            if e.get('is_anchor'):
                a = e['anchor']
                for lvl in a['span']:
                    if lvl not in vo: errs.append(f"{e['id']}: bad vertebral level {lvl}")
                if a['presence'] not in ('obligatory', 'sex_female', 'sex_male', 'surgical', 'variable'):
                    errs.append(f"{e['id']}: bad presence class {a['presence']}")
                for p in a.get('spatial_prior', []):
                    if p['landmark'] not in ids and p['landmark'] not in self.landmarks:
                        errs.append(f"{e['id']}: prior landmark {p['landmark']} unknown")
            if '{side}' in e['voxtell']['main'] and e['laterality'] not in ('bilateral', 'left', 'right'):
                errs.append(f"{e['id']}: side template on non-lateral entity")
            if e['laterality'] == 'bilateral' and '{side}' not in e['voxtell']['main']:
                errs.append(f"{e['id']}: bilateral entity without side template")
        for l in self.landmarks.values():
            if l['entity'] not in ids: errs.append(f"landmark entity missing {l['entity']}")
            if l['level'] not in vo: errs.append(f"landmark level bad {l['level']}")
        # every anchor with presence sex_* or surgical must have a spatial prior (needed for completion)
        for e in self.anchors():
            if e['anchor']['presence'] in ('sex_female', 'sex_male') and not e['anchor'].get('spatial_prior'):
                errs.append(f"{e['id']}: sex-dependent anchor needs a spatial prior")
        # tumour prompt hosts exist
        for ct, d in self.tumour_prompts['cancer_types'].items():
            if d.get('host') and d['host'] not in ids: errs.append(f"tumour host missing for {ct}: {d['host']}")
        return errs
