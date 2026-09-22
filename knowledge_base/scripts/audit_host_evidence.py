"""Which rater supports each secondary host, and with what?  For every plan.json given (or found under --plan_dir),
re-derive the host evidence per rater: for each candidate host with a TS envelope, list the measurable components
(>= min_component_ml) of each rater that lie >= min_inside inside the host envelope, with their volume, their distance
to the primary-host agreement region and the fraction of the component that also lies inside the PRIMARY host envelope.
A secondary host supported only by small components of one rater that sit inside the primary envelope too is leakage
(satellite / boundary spill), not spread.

  python scripts/audit_host_evidence.py --plan_dir /path/Planner/Radiology/MR/TCGA-LIHC [--limit 3]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, nibabel as nib
from scipy import ndimage
from pancia_kb import KnowledgeBase
from pancia_kb.adapter import _spacing, _host_ts_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan_dir', required=True); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--kb', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge_base'))
    ap.add_argument('--envelope_mm', type=float, default=10.0); ap.add_argument('--min_inside', type=float, default=0.5)
    ap.add_argument('--min_component_ml', type=float, default=0.5)
    ap.add_argument('--remap', default=None, help='OLDPREFIX=NEWPREFIX applied to the paths stored in plan.json (plans written on another mount)')
    a = ap.parse_args()
    def rp(x):
        if a.remap and x:
            o, n = a.remap.split('=', 1); x = x.replace(o, n)
        return x
    kb = KnowledgeBase.load(a.kb)
    plans = sorted(glob.glob(os.path.join(a.plan_dir, '**', '*_plan.json'), recursive=True))
    if a.limit: plans = plans[:a.limit]
    for pf in plans:
        p = json.load(open(pf)); inp = p['input']; d = inp['diagnostics']
        base = rp(inp['structures_csv'])[:-len('_structures.csv')]
        meta = json.load(open(base + '_labels.json')); task = meta['task']
        label_names = {int(k): v for k, v in meta['labels'].items()}
        ts = nib.load(base + '.nii.gz'); md = np.asarray(ts.dataobj); sp = _spacing(ts.affine); vox_ml = float(np.prod(sp)) / 1000
        raters = []
        for i, q in enumerate(map(rp, inp['tumour_mask'] or [])):
            m = np.asarray(nib.load(q).dataobj) > 0
            lab, k = ndimage.label(m, structure=np.ones((3, 3, 3)))
            sizes = np.bincount(lab.ravel()) * vox_ml
            raters.append((f'rater_{i}:' + q.split('/Radiology/')[0].split('/')[-1], lab, k, sizes))
        prim = d.get('expected_host')
        def envelope(ent):
            labs = _host_ts_labels(kb, task, label_names, ent); hm = np.isin(md, labs) if labs else None
            return None if hm is None or not hm.any() else ndimage.distance_transform_edt(~hm, sampling=sp) <= a.envelope_mm
        penv = envelope(prim) if prim else None
        # validated primary tumour = agreement of the raters (what the planner's centroid used when evidence is 'agreement')
        tumour = None
        if len(raters) > 1:
            tumour = np.ones(md.shape, bool)
            for _, lab, k, _s in raters: tumour &= lab > 0
            if not tumour.any(): tumour = None
        print(f"\n== {inp['img_name'].split('/')[-1]}  host={prim}  evidence={d.get('tumour_evidence')}  raters={[r[0] for r in raters]}")
        for name, lab, k, sizes in raters:
            big = [j for j in range(1, k + 1) if sizes[j] >= a.min_component_ml]
            print(f"   {name}: {k} components, {len(big)} >= {a.min_component_ml} ml, total {sizes[1:].sum():.1f} ml, largest {sizes[1:].max() if k else 0:.1f} ml")
        for h in d.get('hosts', []):
            if h.get('isolated_lesions'):
                print(f"   {h['entity']:12s} isolated single-rater lesions (not evidence): {h['isolated_lesions']}")
            if h['cls'] == 'primary' or not h.get('evidence'): continue
            env = envelope(h['entity'])
            if env is None: continue
            for name, lab, k, sizes in raters:
                for j in range(1, k + 1):
                    if sizes[j] < a.min_component_ml: continue
                    comp = lab == j; n = comp.sum()
                    fin = (comp & env).sum() / n
                    if fin < a.min_inside: continue
                    fprim = (comp & penv).sum() / n if penv is not None else float('nan')
                    touches = bool((comp & tumour).any()) if tumour is not None else False
                    outside_ml = (comp & env & ~penv).sum() * vox_ml if penv is not None else float('nan')
                    tag = 'SUPPORTS' if (h['evidence'] == 2 or (touches and outside_ml >= a.min_component_ml)) else 'no'
                    print(f"   {h['entity']:12s} ev{h['evidence']}  {tag:8s} {name} comp#{j}: {sizes[j]:.1f} ml, {fin:.0%} in {h['entity']} env, {fprim:.0%} in {prim} env, touches primary tumour={touches}, {outside_ml:.1f} ml beyond primary env")


if __name__ == '__main__':
    main()
