"""Export the three planner graphs of one scan for inspection: (1) KB prior graph of the primary host, (2) the same graph
filtered by TS (found / FOV) with tumour evidence and reach, (3) the planned prompt graph. Writes <out>.json."""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pancia_kb import KnowledgeBase, Planner, load_ts_output

ap = argparse.ArgumentParser()
ap.add_argument('--structures_csv', required=True); ap.add_argument('--tumour', action='append', default=[])
ap.add_argument('--host', required=True); ap.add_argument('--cancer_type', required=True); ap.add_argument('--modality', default='CT')
ap.add_argument('--out', required=True)
a = ap.parse_args()
kb = KnowledgeBase.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge_base'))
ent = lambda e: dict(id=e, name=kb.entities[e]['name'], lat=kb.entities[e]['laterality'], anchor=kb.entities[e].get('is_anchor', False),
                     tumour_host=kb.entities[e].get('tumour_host', True), category=kb.entities[e].get('category'))
prof = kb.profile(a.host, hops=2)
spread = kb.spread_hosts(a.host, a.cancer_type)
nodes = {a.host} | {r['dst'] for r in prof} | {r['src'] for r in prof} | {h['entity'] for h in spread}
prior = dict(host=a.host, cancer_type=a.cancer_type, nodes=[ent(n) for n in sorted(nodes) if n in kb.entities],
             edges=[{k: r.get(k) for k in ('src', 'type', 'dst', 'side_link', 'staging', 'hop', 'contact', 'note')} for r in prof],
             spread=spread)
ts, diag = load_ts_output(a.structures_csv, tumour_mask=a.tumour, kb=kb, expected_host=a.host, cancer_type=a.cancer_type)
plan = Planner(kb).plan(ts, modality=a.modality, cancer_type=a.cancer_type)
found = [dict(ts_name=s.ts_name, entity=(kb.ts_to_entity.get(ts.task, {}).get(s.ts_name) or [None, None])[0],
              side=(kb.ts_to_entity.get(ts.task, {}).get(s.ts_name) or [None, None])[1], ml=s.volume_ml, truncated=s.truncated) for s in ts.structures]
tsg = dict(task=ts.task, fov_z_mm=ts.fov_z_mm, found=found, hosts=diag.get('hosts'), primary_sides=diag.get('primary_sides'),
           tumour_evidence=diag.get('tumour_evidence'), tumour_voxels=diag.get('tumour_voxels'),
           raters={k: v for k, v in diag.items() if k.startswith('rater_')})
d = json.loads(plan.to_json())
out = dict(prior=prior, ts=tsg, plan=dict(prompts=d['prompts'], log=d['log'], frame=d.get('frame')))
json.dump(out, open(a.out, 'w'), indent=1, default=str)
print('wrote', a.out, len(prior['edges']), 'prior edges;', len(found), 'TS structures;', len(d['prompts']), 'prompts')
