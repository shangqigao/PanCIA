"""v3 design statistics (27 Nov 2026; plan claude/graph_pruning_v3_plan_2026-11-27.md; workflow claude/stage1_workflow.md).
DESIGN SET ONLY (v1 jobs.csv + v2 jobs_v2.csv; jobs_v3.csv is never read here). Descriptive; nothing is fitted or decided.

Where do agent candidates lie relative to the step-2g anatomy layer, under the user's organ-use decisions (27 Nov):
  usable organ  = mu > 0 and no 'small' / 'spatial' / 'not_between' flag (TS and VT alike), used through its CORE;
  excluded organ = mu > 0 with one of those flags (core reported separately); mu 0 organs have no core;
  contested regions = overlaps of two organs' territories (U_a & U_b, for every pair listed by step 2g), classified by the KB
                      relation between the two entities: supply_drain / vessel_vessel / same_entity / adjacent / vessel_organ /
                      organ_organ (no edge) / other_edge.
Workflow steps consumed: 1a (raw BiomedParse / VoxTell tumour masks -> candidates), 2g (anatomy files), 0 (expert labels and the
cohort tumour-free file, evaluation only). No raw TS / VT organ file is read; TS is read only for the evaluation scope (primary organ
+ 10 mm, as in the evidence screen). No sex, no cancer type in any anatomy quantity.

Candidates: 26-connected components >= 0.05 ml of the raw BP mask, the raw VT mask, and their union (three independent modes).
Truth (evaluation only): lab T = touches an expert tumour and (in_frac >= 0.5 or dice_t >= 0.1); F = touches none; P = partial.
Per candidate: voxel shares in usable cores (sum and top organ), excluded cores, each contested category, outside all organs, and in
'impossible pelvic' cores (VT uterus / vagina / cervix / ovary / myometrium / endometrium cores in scans whose TS segmented a prostate).
Output: PanCIA_outputs/ledger/v3_design/location.csv (append; resumable by (ds, case)).
Usage: python s1_v3_location.py [--budget 1e9] [--only ds/case,...]"""
import os, sys, json, re, time, argparse, csv
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from pancia_kb.stage2 import paths, expert as EX     # noqa: E402
from pancia_kb import KnowledgeBase                    # noqa: E402
import s1_evidence_screen as SC                        # noqa: E402  (PRIMARY, MIN_ML only)

R = paths.root('R2Seg'); S = paths.root('R2Seg_Seg')
ANAT = os.path.join(paths.root('outputs'), 'anatomy', 'r2seg')
OUTD = os.path.join(paths.ledger_dir(), 'v3_design'); OUTF = os.path.join(OUTD, 'location.csv')
JOBS = [os.path.join(paths.ledger_dir(), 'evidence_screen', f) for f in ('jobs.csv', 'jobs_v2.csv')]   # design set only
KBD = os.path.abspath(os.path.join(HERE, '..', '..'))
EXCL_FLAGS = ('small', 'spatial', 'not_between')          # user 27 Nov: mask-error flags; components / no_contact are not
VES = ('vessel_artery', 'vessel_vein')
FEM = {'uterus', 'cervix', 'vagina', 'ovary', 'myometrium', 'endometrium', 'endometrial_cavity'}
CATS = ('supply_drain', 'vessel_vessel', 'same_entity', 'adjacent', 'vessel_organ', 'organ_organ', 'other_edge')
COLS = ['ds', 'case', 'site', 'tumour_free', 'mod', 'mode', 'cid', 'ml', 'lab', 'in_scope', 'in_frac', 'dice_t',
        'f_core', 'top_core', 'top_core_src', 'top_core_share', 'f_excl'] + [f'f_c_{c}' for c in CATS] + \
       ['f_outside', 'f_impossible', 'note']


def base_ent(x):
    return re.sub(r'_(left|right)$', '', x)


def category(kb, a, b):
    if a == b:
        return 'same_entity'
    r = {t['type'] for t in kb.out_edges.get(a, []) if t['dst'] == b} | {t['type'] for t in kb.out_edges.get(b, []) if t['dst'] == a}
    va = kb.entities.get(a, {}).get('category') in VES; vb = kb.entities.get(b, {}).get('category') in VES
    if r & {'supplied_by', 'drained_by', 'drains_into', 'supplies', 'drains'}: return 'supply_drain'
    if 'adjacent_to' in r: return 'adjacent'
    if r: return 'other_edge'
    if va and vb: return 'vessel_vessel'
    if va or vb: return 'vessel_organ'
    return 'organ_organ'


def unpack(z, key, off, crop):
    """organ mask `key` (o<i>_core / _terr) restricted to the crop box `crop` (full-grid slices); None if absent / disjoint"""
    if key not in z:
        return None
    sl = [(int(a) + o, int(b) + o) for (a, b), o in zip(z[key + '_sl'], off)]; shp = tuple(int(x) for x in z[key + '_shape'])
    lo = [max(a, c.start) for (a, _), c in zip(sl, crop)]; hi = [min(b, c.stop) for (_, b), c in zip(sl, crop)]
    if any(l >= h for l, h in zip(lo, hi)):
        return None
    m = np.unpackbits(z[key])[:int(np.prod(shp))].reshape(shp).astype(bool)
    out = np.zeros(tuple(c.stop - c.start for c in crop), bool)
    out[tuple(slice(l - c.start, h - c.start) for l, h, c in zip(lo, hi, crop))] = \
        m[tuple(slice(l - a, h - a) for l, h, (a, _) in zip(lo, hi, sl))]
    return out if out.any() else None


def one(kb, ds, case, tf):
    site = EX.SITE_ENTITY.get(ds, ''); base = dict(ds=ds, case=case, site=site, tumour_free=tf)
    img = f'{ds}/{case}'; js = os.path.join(ANAT, img + '.json')
    if not os.path.exists(js):
        return [dict(base, note='no anatomy file')]
    meta = json.load(open(js)); z = np.load(os.path.join(ANAT, img + '.npz')); off = meta.get('offset') or [0, 0, 0]
    base['mod'] = meta.get('modality')
    bp = os.path.join(S, 'BiomedParse', ds, case + '_tumor.nii.gz'); vp = os.path.join(S, 'VoxTell', ds, case + '_tumor.nii.gz')
    tp = os.path.join(S, 'TotalSegmentator', ds, case + '_organ.nii.gz'); lp = os.path.join(R, ds, 'labels', case + '_seg.nii.gz')
    bimg = nib.load(bp); sp = tuple(float(x) for x in bimg.header.get_zooms()[:3]); vml = np.prod(sp) / 1000
    B = np.asanyarray(bimg.dataobj) > 0; V = np.asanyarray(nib.load(vp).dataobj) > 0
    T = np.isin(EX._read_labels(lp), EX.APPEARANCE[ds]['tumour']) if os.path.exists(lp) else np.zeros(B.shape, bool)
    if not (B.shape == V.shape == T.shape == tuple(meta['shape'])):
        return [dict(base, note='grid mismatch')]
    U = B | V
    if not U.any():
        return [dict(base, note='no candidates')]
    bb = ndimage.find_objects(U.astype(np.uint8))[0]; del U
    mg = [int(np.ceil(15 / s)) for s in sp]
    crop = tuple(slice(max(0, b.start - k), min(n, b.stop + k)) for b, k, n in zip(bb, mg, B.shape))
    B, V, T = B[crop], V[crop], T[crop]
    # evaluation scope (as the evidence screen): primary TS organ + 10 mm; TS read for scope only
    names = json.load(open(tp.replace('_organ.nii.gz', '_organ_labels.json')))['labels']
    pid = [int(k) for k, v in names.items() if v in SC.PRIMARY.get(site, [])]
    P = None
    if pid:
        org = np.asanyarray(nib.load(tp).dataobj)[crop]; P0 = np.isin(org, pid); del org
        if P0.any():
            P = ndimage.distance_transform_edt(~P0, sampling=sp) <= 10
    # anatomy layer (step 2g)
    org_meta = {(o['entity'], o['side']): o for o in meta['organs']}
    hk = {((o['entity'] + ('_' + o['side'] if o['side'] else ''))): o for o in meta['organs']}
    cores, excl, terr = {}, {}, {}
    pros = any(o['entity'] == 'prostate' and 'TS' in o['agents'] for o in meta['organs'])
    imp = np.zeros(B.shape, bool)
    for o in meta['organs']:
        if o['mu'] <= 0:
            continue
        c = unpack(z, f"o{o['i']}_core", off, crop)
        name = o['entity'] + ('_' + o['side'] if o['side'] else '')
        bad = any(f.split(':')[0].split(' ')[0] in EXCL_FLAGS for f in o['flags'])
        if c is not None:
            (excl if bad else cores)[name] = (c, '+'.join(sorted(o['agents'])))
            if pros and o['entity'] in FEM and 'TS' not in o['agents']:
                imp |= c
    pairs = []
    for p in meta['pairs']:
        a, b_ = hk.get(p['a']), hk.get(p['b'])
        if a is None or b_ is None:
            continue
        ta = terr.get(p['a']); ta = ta if ta is not None else unpack(z, f"o{a['i']}_terr", off, crop); terr[p['a']] = ta
        tb = terr.get(p['b']); tb = tb if tb is not None else unpack(z, f"o{b_['i']}_terr", off, crop); terr[p['b']] = tb
        if ta is None or tb is None:
            continue
        m = ta & tb
        if m.any():
            pairs.append((category(kb, a['entity'], b_['entity']), m))
    anyorg = np.zeros(B.shape, bool)
    for t in terr.values():
        if t is not None: anyorg |= t
    for o in meta['organs']:                                   # territories of organs without contested pairs
        name = o['entity'] + ('_' + o['side'] if o['side'] else '')
        if name not in terr and o['mu'] > 0:
            t = unpack(z, f"o{o['i']}_terr", off, crop)
            if t is not None: anyorg |= t
    catm = {c: np.zeros(B.shape, bool) for c in CATS}
    for c, m in pairs:
        catm[c] |= m
    lbT, nT = ndimage.label(T, structure=EX.STRUCT26); tsz = np.bincount(lbT.ravel(), minlength=nT + 1)
    rows = []
    for mode, M in (('BP', B), ('VT', V), ('BP+VT', B | V)):
        lb, n = ndimage.label(M, structure=EX.STRUCT26)
        for j, ob in enumerate(ndimage.find_objects(lb), start=1):
            if ob is None: continue
            cb = lb[ob] == j; nc = int(cb.sum()); ml = nc * vml
            if ml < SC.MIN_ML: continue
            tr = bool(T[ob][cb].any()); ids = np.unique(lbT[ob][cb]); ids = ids[ids > 0]
            inter = int((T[ob] & cb).sum()); gs = int(tsz[ids].sum()) if len(ids) else 0
            in_frac = inter / nc; dice_t = (2 * inter / (nc + gs)) if gs else 0.0
            lab = 'F' if not tr else ('T' if (in_frac >= .5 or dice_t >= .1) else 'P')
            insc = tr or P is None or float(P[ob][cb].mean()) >= 0.5
            sh = {k: float((v[0][ob] & cb).sum()) / nc for k, v in cores.items()}
            sh = {k: v for k, v in sh.items() if v > 0}
            un = np.zeros(cb.shape, bool)
            for k in sh: un |= cores[k][0][ob]
            ex = np.zeros(cb.shape, bool)
            for k, v in excl.items(): ex |= v[0][ob]
            top = max(sh, key=sh.get) if sh else ''
            r = dict(base, mode=mode, cid=j, ml=round(ml, 4), lab=lab, in_scope=bool(insc), in_frac=round(in_frac, 3),
                     dice_t=round(dice_t, 3), f_core=round(float((un & cb).sum()) / nc, 3), top_core=top,
                     top_core_src=cores[top][1] if top else '', top_core_share=round(sh.get(top, 0.0), 3),
                     f_excl=round(float((ex & cb).sum()) / nc, 3),
                     f_outside=round(float((~anyorg[ob] & cb).sum()) / nc, 3), f_impossible=round(float((imp[ob] & cb).sum()) / nc, 3))
            for c in CATS:
                r[f'f_c_{c}'] = round(float((catm[c][ob] & cb).sum()) / nc, 3)
            rows.append(r)
    return rows or [dict(base, note='no candidates >= 0.05 ml')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--budget', type=float, default=1e9); ap.add_argument('--only', default='')
    a = ap.parse_args()
    J = pd.concat([pd.read_csv(f) for f in JOBS]).drop_duplicates(['ds', 'case']); J['case'] = J.case.astype(str)
    if a.only: J = J[(J.ds + '/' + J.case).isin(a.only.split(','))]
    os.makedirs(OUTD, exist_ok=True)
    done = set()
    if os.path.exists(OUTF):
        d = pd.read_csv(OUTF, usecols=['ds', 'case']); done = set(zip(d.ds, d.case.astype(str)))
    kb = KnowledgeBase.load(f'{KBD}/knowledge_base')
    t0, n = time.time(), 0
    for ds, case, tf in J[['ds', 'case', 'tumour_free']].itertuples(index=False):
        if (ds, case) in done: continue
        if time.time() - t0 > a.budget: break
        try:
            rows = one(kb, ds, case, bool(tf))
        except Exception as e:
            rows = [dict(ds=ds, case=case, note='error ' + repr(e)[:150])]
        new = not os.path.exists(OUTF)
        with open(OUTF, 'a', newline='') as fh:
            w = csv.DictWriter(fh, COLS, extrasaction='ignore'); new and w.writeheader()
            for r in rows: w.writerow({k: r.get(k, '') for k in COLS})
        n += 1
    left = sum((d, c) not in done for d, c in zip(J.ds, J.case)) - n
    print(f'processed {n}; remaining {left}; {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
