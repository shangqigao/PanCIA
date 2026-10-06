"""For every tumour-hosting anchor: is the KB ready to take a NEW cancer type with this primary (no YAML entry)?
Checks the KB fields the pipeline reads and dry-runs the planner on one real thorax-abdomen-pelvis CT with an explicit host."""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pancia_kb import KnowledgeBase
from pancia_kb.planner import Planner
from pancia_kb.adapter import load_ts_output
kb = KnowledgeBase.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge_base'))
print('KB validate errors:', kb.validate() if hasattr(kb, 'validate') else 'n/a')
hosts = sorted(e['id'] for e in kb.anchors() if e.get('tumour_host', True))
ts_names = {t: {v[0] for v in m.values()} for t, m in kb.ts_to_entity.items()}
csv = sys.argv[1]
rows = []
for h in hosts:
    e = kb.entities[h]
    sp = kb.spread_hosts(h, 'NEW-CANCER')
    loc = [s['entity'] for s in sp if s['cls'] == 'local_invasion']; dis = [s['entity'] for s in sp if s['cls'] == 'distant']
    r = dict(host=h, category=e.get('category'), tumour_phrase=bool(e.get('tumour_phrase')), vt_terms=len(kb.voxtell_terms(h, 'left' if e['laterality'] == 'bilateral' else None)),
             vocab=e.get('coverage_tier'), ts_ct=h in ts_names.get('total', set()), ts_mr=h in ts_names.get('total_mr', set()),
             spatial_prior=bool((e.get('anchor') or {}).get('spatial_prior')), n_local=len(loc), local=';'.join(loc), distant=';'.join(dis))
    try:
        ts, diag = load_ts_output(csv, kb=kb, expected_host=h, cancer_type='NEW-CANCER')
        plan = Planner(kb).plan(ts, modality='CT', cancer_type='NEW-CANCER')
        T = [p for p in plan.prompts if p.list == 'T']
        r.update(plan='ok', n_prompts=len(plan.prompts), tierT=';'.join(f"{p.entity}{'/' + p.side if p.side else ''}:{p.gate.get('host_class','')[:3]}" for p in T),
                 primary_prompted=any(p.entity == h for p in T), primary_in_fov=ts.primary_in_fov)
    except Exception as ex:
        r.update(plan=f'ERROR {type(ex).__name__}: {ex}')
    rows.append(r)
for r in rows:
    print(json.dumps(r))
