"""Workflow step 2 for the graph reasoning (27 Nov 2026; claude/anatomy_check_general_plan_2026-11-27.md): run the general anatomy
check (pancia_kb.anatomy_check_g: TS + VoxTell plan organs + plan + KB + image noise; NO tumour masks, no cancer type) on R2Seg
scans and persist it per scan -> PanCIA_outputs/anatomy/r2seg/<collection>/<stem>.{npz,json}  (closes workflow gap G-a).
Scan list: cohort/r2seg_scan_meta.csv (img_name, modality only; the sex column is never read).
--verify: also runs anatomy_qc_v82.analyse exactly as ledger_v86 does (tumour = BP AND VT, planner primary) and writes one comparison
row per scan to anatomy/r2seg/_verify/verify.csv (organ states, mu, flags, core voxels; Q). Expert labels are not read.
Memory guard: every attempt is logged in anatomy/r2seg/_heavy/attempts.log (append-only: start / end / error). A scan is skipped as
heavy only after TWO attempts that started and never finished (killed out of memory on the 4 GB VM, e.g. whole-body CT); one
unfinished attempt may be a device-call cut-off. New scans are not started within --guard seconds of the budget end. Heavy scans:
run where memory allows with --retry_heavy. Usage: python s1_anatomy_g.py [--budget 120] [--guard 30] [--only Coll/stem,...] [--jobs jobs.csv,jobs_v2.csv] [--collections A,B] [--sample N --seed S] [--verify]   (resumable)"""
import os, sys, json, time, argparse, csv
import numpy as np, pandas as pd, nibabel as nib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths                                   # noqa: E402
from pancia_kb import KnowledgeBase                                   # noqa: E402
from pancia_kb import anatomy_check_g as G, anatomy_qc_v82 as AQ     # noqa: E402
from pancia_kb.adapter import _spacing                                # noqa: E402

R = paths.root('R2Seg'); S = paths.root('R2Seg_Seg'); OUT = os.path.join(paths.root('outputs'), 'anatomy', 'r2seg')
KBD = os.path.abspath(os.path.join(HERE, '..', '..'))
THR = json.load(open(os.path.join(KBD, 'knowledge_base', 'qc_thresholds.json')))
VCOLS = ['img', 'code_hash', 'n_organs', 'n_same_state', 'n_same_flags', 'n_same_core', 'diff_organs', 'q_v82', 'q_g', 'reasons_v82', 'reasons_g',
         'f_v82', 'f_g', 'noise_v82', 'noise_g', 'noise_same', 'sec_g', 'sec_v82']


def inputs(img):
    coll, stem = img.split('/')
    plan = json.load(open(os.path.join(S, 'Planner', coll, stem + '_plan.json')))
    mp = os.path.join(S, 'VoxTell', coll, stem + '_plan_manifest.json'); man = json.load(open(mp))
    ts = nib.load(os.path.join(S, 'TotalSegmentator', coll, stem + '_organ.nii.gz'))
    md = np.asanyarray(ts.dataobj)                                     # memory (27 Nov): stored dtype (uint8 label map); no int16 copy
    md = md if md.dtype.kind in 'iu' else md.astype(np.int16)
    meta = json.load(open(os.path.join(S, 'TotalSegmentator', coll, stem + '_organ_labels.json')))
    names = {int(k): v for k, v in (meta.get('labels') or {}).items()}
    ip = os.path.join(R, coll, 'images', stem + '.nii.gz')
    X = np.asanyarray(nib.load(ip).dataobj) if os.path.exists(ip) else None
    if X is not None and X.ndim == 4: X = X[..., 0]
    return plan, man, mp[:-len('_manifest.json')], ts, md, names, meta.get('task') or 'total', X


def one(KB, img, mod, verify):
    coll, stem = img.split('/')
    plan = json.load(open(os.path.join(S, 'Planner', coll, stem + '_plan.json')))
    mp = os.path.join(S, 'VoxTell', coll, stem + '_plan_manifest.json'); man = json.load(open(mp)); pdir = mp[:-len('_manifest.json')]
    tsp = os.path.join(S, 'TotalSegmentator', coll, stem + '_organ.nii.gz')
    modality = (plan.get('input') or {}).get('modality') or mod
    t0 = time.time()
    r, off, full, aff_full = G.check_scan(KB, tsp, tsp.replace('_organ.nii.gz', '_organ_labels.json'), plan, man, pdir,
                                         image_path=os.path.join(R, coll, 'images', stem + '.nii.gz'), modality=modality, thresholds=THR)
    tg = time.time() - t0
    G.save(r, os.path.join(OUT, img), full, aff_full, extra=dict(img=img, modality=modality, task=r['task']), offset=off)
    if not verify:
        return None
    plan, man, pdir, ts, md, names, task, X = inputs(img)
    sp = _spacing(ts.affine)
    coll, stem = img.split('/')
    tum = np.zeros(md.shape, np.int8)
    for a in ('BiomedParse', 'VoxTell'):
        f = os.path.join(S, a, coll, stem + '_tumor.nii.gz')
        if os.path.exists(f): tum += (np.asarray(nib.load(f).dataobj) > 0).astype(np.int8)
    prim = {m['entity'] for m in man['masks'] if m['list'] == 'T' and m.get('host_class') == 'primary'}
    t0 = time.time()
    v = AQ.analyse(KB, md, names, task, ts.affine, np.asarray(sp, float), plan, man, pdir, tumour=tum >= 2, image=X,
                   modality=modality, thresholds=THR, primary=prim)
    tv = time.time() - t0
    same_s = same_f = same_c = 0; diff = []
    for k, o in r['organs'].items():
        w = v['organs'].get(k)
        if w is None: diff.append(f'{k}:missing'); continue
        s1 = o['agents'] == w['agents'] and o['mu'] == w['mu']
        fl = [x.replace('large_primary', 'large') for x in w['flags']]
        f1 = sorted(set(fl)) == o['flags']
        osl = None if o['core'].crop is None else tuple(slice(a.start + d, a.stop + d) for a, d in zip(o['core'].sl, off))
        c1 = (o['core'].crop is None and w['core'].crop is None) or (o['core'].crop is not None and w['core'].crop is not None
             and osl == w['core'].sl and np.array_equal(o['core'].crop, w['core'].crop))
        same_s += s1; same_f += f1; same_c += c1
        if not (s1 and f1 and c1): diff.append(f"{k[0]}_{k[1]}:{'s' if not s1 else ''}{'f' if not f1 else ''}{'c' if not c1 else ''}")
    nv, ng = v.get('noise') or {}, r.get('noise') or {}
    same_noise = set(nv) == set(ng) and all(np.isclose(float(nv[k]), float(ng[k]), rtol=1e-6, atol=1e-9, equal_nan=True)
                                            for k in nv if isinstance(nv[k], (int, float, np.floating, np.integer)))
    return dict(img=img, code_hash=G.CODE_HASH, n_organs=len(r['organs']), n_same_state=same_s, n_same_flags=same_f, n_same_core=same_c,
                diff_organs=';'.join(diff), q_v82=v['state'], q_g=r['state'], reasons_v82='|'.join(v['reasons']),
                reasons_g='|'.join(r['reasons']), f_v82=v.get('f_away'), f_g=r['f_all'], noise_v82=json.dumps(nv, default=float),
                noise_g=json.dumps(ng, default=float), noise_same=same_noise,
                sec_g=round(tg, 1), sec_v82=round(tv, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=120); ap.add_argument('--guard', type=float, default=30); ap.add_argument('--only', default=''); ap.add_argument('--jobs', default=''); ap.add_argument('--collections', default='')
    ap.add_argument('--sample', type=int, default=0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--verify', action='store_true'); ap.add_argument('--retry_heavy', action='store_true')
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(paths.root('outputs'), 'cohort', 'r2seg_scan_meta.csv'), usecols=['img_name', 'modality'])
    meta = meta[meta.img_name.map(lambda x: os.path.exists(os.path.join(S, 'VoxTell', x + '_plan_manifest.json')))]
    if a.only: meta = meta[meta.img_name.isin(a.only.split(','))]
    if a.jobs:                                                          # 27 Nov: job files (ds, case) of ledger/evidence_screen
        js_ = set()
        for f in a.jobs.split(','):
            D = pd.read_csv(f if os.path.isabs(f) else os.path.join(paths.ledger_dir(), 'evidence_screen', f), usecols=['ds', 'case'])
            js_ |= {f'{d}/{c}' for d, c in zip(D.ds, D.case.astype(str))}
        miss = js_ - set(meta.img_name)
        if miss: print('jobs without a VoxTell plan manifest (skipped):', len(miss))
        meta = meta[meta.img_name.isin(js_)]
    if a.collections: meta = meta[meta.img_name.str.split('/').str[0].isin(a.collections.split(','))]
    if a.sample: meta = meta.groupby(meta.img_name.str.split('/').str[0], group_keys=False).apply(
        lambda g: g.sample(min(a.sample, len(g)), random_state=a.seed))
    KB = KnowledgeBase.load(f'{KBD}/knowledge_base')
    vf = os.path.join(OUT, '_verify', 'verify_v2.csv'); os.makedirs(os.path.dirname(vf), exist_ok=True)   # v2: + code_hash, reasons, noise
    vdone = set(pd.read_csv(vf).query('code_hash == @G.CODE_HASH').img) if (a.verify and os.path.exists(vf)) else set()
    alog = os.path.join(OUT, '_heavy', 'attempts.log'); os.makedirs(os.path.dirname(alog), exist_ok=True)
    from collections import Counter
    starts, ends = Counter(), Counter()
    if os.path.exists(alog):
        for ln in open(alog):
            k, v = ln.rstrip('\n').split('\t'); (starts if k == 'start' else ends)[v] += 1
    heavy = {v for v in starts if ends[v] == 0 and starts[v] >= 2}      # two unfinished attempts and never finished
    if heavy:
        print(('retrying' if a.retry_heavy else 'skipping (use --retry_heavy to include)') + ' heavy scans (killed twice earlier):',
              ', '.join(sorted(heavy)), flush=True)
    t0, n = time.time(), 0
    for img, mod in meta.itertuples(index=False):
        if time.time() - t0 > a.budget - a.guard: break
        js = os.path.join(OUT, img + '.json')
        if os.path.exists(js) and json.load(open(js)).get('code_hash') == G.CODE_HASH and (not a.verify or img in vdone):
            continue
        if img in heavy and not a.retry_heavy:                         # memory (27 Nov): previous attempt killed (out of memory on the
            continue                                                    # 4 GB VM); listed in _heavy/attempts.log (append-only, no deletes)
        with open(alog, 'a') as fh: fh.write(f'start\t{img}\n')
        try:
            row = one(KB, img, mod, a.verify)
        except Exception as e:
            print('ERR', img, repr(e)[:200]); row = None; err = True
        else:
            err = False
        with open(alog, 'a') as fh: fh.write(f"{'error' if err else 'end'}\t{img}\n")
        if err: continue
        n += 1
        if row:
            new = not os.path.exists(vf)
            with open(vf, 'a', newline='') as fh:
                w = csv.DictWriter(fh, VCOLS); new and w.writeheader(); w.writerow(row)
    left = sum(1 for img in meta.img_name if not os.path.exists(os.path.join(OUT, img + '.json')))
    print(f'processed {n}; remaining {left}; {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
