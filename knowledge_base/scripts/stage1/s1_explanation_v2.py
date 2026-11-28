"""Explanation-based pruning v2 (26 Nov 2026; gate claude/explanation_v2_gate_2026-11-26.md, user-approved incl. VT guards and a
TS-only arm). Three independent reasonings (BP, VT, BP+VT candidates = 26-connected components >= 0.05 ml of raw BP / initial VT /
union). Anatomy layer: arm 'ts' = TotalSegmentator only; arm 'tsvt' = TS + VoxTell plan organs for entities TS lacks (TS precedence),
each VT organ used to explain a candidate only if QC passes: exact Dice(candidate, organ) < 0.8, organ overlaps VT plan tumour masks
< 50% of its volume, and organ volume <= KB normal q95 when a KB prior exists. VT organs are used only for E1 / E3.
Explanations (scan-relative clauses): E1 organ part, E2 hollow wall (CT; TS organs), E3 edge / partial volume, E5 whole organ (TS),
E4 tube (geometry only -> uncertain). Fragment groups: surfaces within tau = max(5 mm, largest spacing), same host (or both none).
Expert labels for evaluation only. Fresh sample: jobs_v2.csv (seed 2; patients disjoint from the v1 jobs and the expert-tumour set).
Output: ledger/evidence_screen/expl_v2.csv (one row per candidate x arm). Usage: python s1_explanation_v2.py [budget_s] [workers]"""
import os, sys, json, glob, re, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib, yaml
from scipy import ndimage
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402
import s1_evidence_screen as SC                     # noqa: E402
OUT = SC.OUT; O = os.path.join(OUT, os.environ.get('EXPL_OUT', 'expl_v2.csv'))
MODES = os.environ.get('MODES', 'BP,VT,BP+VT').split(',')   # 27 Nov: heavy scans run one mode per call
JF = os.path.join(OUT, 'jobs_v2.csv'); MAIN = os.path.join(OUT, 'expl_v2.csv')
MIN_ML = 0.05; TAU_ELONG = 3.3                        # v1 pan-cancer elongation level; E4 only marks 'uncertain'
GP = yaml.safe_load(open(os.path.join(HERE, '..', '..', 'knowledge_base', 'geometry_priors.yaml')))['anatomy']
CAT = dict(SC.CAT_KEYS)
COLS = ['ds', 'case', 'site', 'tumour_free', 'mod', 'arm', 'mode', 'cid', 'ml', 'true', 'in_scope', 'in_frac', 'dice_t', 'group',
        'host', 'thick_mm', 'elong', 'E1', 'E2', 'E3', 'E4', 'E5', 'explained', 'vt_qc_fail', 'note']


def cat_of(name):
    for cat, keys in SC.CAT_KEYS:
        if any(x in name for x in keys): return cat
    return 'organ'


def kb_key(name):
    for k in GP:
        if name == k or name.startswith(k + '_') or (k == 'lung' and name.startswith('lung')) or (k == 'kidney' and name.startswith('kidney')):
            return k
    return None


def elong_of(m, sp):
    xyz = np.argwhere(m)
    if len(xyz) > 200000: xyz = xyz[::len(xyz) // 200000 + 1]
    xyz = xyz * np.asarray(sp)
    if len(xyz) < 4: return np.nan
    ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T))); return float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))


def make_jobs(n_per=20, seed=2):
    J1 = pd.read_csv(os.path.join(OUT, 'jobs.csv')); G = pd.read_csv(os.path.join(OUT, 'expert_tumour_geom.csv'), usecols=['ds', 'case'])
    used = {(d, SC.patient(d, str(c))) for d, c in list(zip(J1.ds, J1.case)) + list(zip(G.ds, G.case))}
    C = pd.read_csv(os.path.join(os.path.dirname(paths.ledger_dir()), 'cohort', 'r2seg_tumour_free.csv'))[['ds', 'case', 'status']]; C['case'] = C.case.astype(str)
    t = pd.read_csv(os.path.join(paths.ledger_dir(), 'v86_r2seg_bp2', 'diag', 'anchor_truth_lite.csv'), usecols=['ds', 'case'], low_memory=False).drop_duplicates()
    t = t[t.ds != 'Uterus_Tumor_00']; t['case'] = t.case.astype(str); t = t.merge(C, on=['ds', 'case'], how='left'); t['status'] = t.status.fillna('tumour')
    neg = t[t.status == 'negative'].assign(tumour_free=True)
    tum = t[t.status == 'tumour'].copy(); tum['site'] = tum.ds.map(EX.SITE_ENTITY); tum['pat'] = [SC.patient(d, c) for d, c in zip(tum.ds, tum.case)]
    tum = tum[[(d, p) not in used for d, p in zip(tum.ds, tum.pat)]]
    out = [g.sample(frac=1, random_state=seed).drop_duplicates('pat').head(n_per) for _, g in tum.groupby('site')]
    tum = pd.concat(out).assign(tumour_free=False)
    return pd.concat([tum[['ds', 'case', 'tumour_free']], neg[['ds', 'case', 'tumour_free']]], ignore_index=True)


def one(job):
    ds, case, tf = job; site = EX.SITE_ENTITY.get(ds, ''); mod = EX.APP_MOD.get(ds, 'CT')
    base = dict(ds=ds, case=case, site=site, tumour_free=tf, mod=mod)
    try:
        ip = os.path.join(SC.R, ds, 'images', case + '.nii.gz'); lp = os.path.join(SC.R, ds, 'labels', case + '_seg.nii.gz')
        bp = os.path.join(SC.S, 'BiomedParse', ds, case + '_tumor.nii.gz'); vp = os.path.join(SC.S, 'VoxTell', ds, case + '_tumor.nii.gz')
        tp = os.path.join(SC.S, 'TotalSegmentator', ds, case + '_organ.nii.gz')
        if not all(os.path.exists(p) for p in (ip, bp, vp, tp)): return [dict(base, note='missing input')]
        img = nib.load(ip); I = np.asanyarray(img.dataobj)
        if I.ndim == 4: I = I[..., 0]
        sp = tuple(float(x) for x in img.header.get_zooms()[:3]); vml = np.prod(sp) / 1000
        B = np.asanyarray(nib.load(bp).dataobj) > 0; V = np.asanyarray(nib.load(vp).dataobj) > 0
        org = np.asanyarray(nib.load(tp).dataobj).astype(np.int16)
        names = {int(k): v for k, v in json.load(open(tp.replace('_organ.nii.gz', '_organ_labels.json')))['labels'].items()}
        T = np.isin(EX._read_labels(lp), EX.APPEARANCE[ds]['tumour']) if os.path.exists(lp) else np.zeros(I.shape, bool)
        if not (B.shape == V.shape == org.shape == T.shape == I.shape): return [dict(base, note='grid mismatch')]
        pid = [k for k, v in names.items() if v in SC.PRIMARY.get(site, [])]
        U = B | V | T
        if pid: U |= np.isin(org, pid)
        if not U.any(): return [dict(base, note='no candidates')]
        bb = ndimage.find_objects(U.astype(np.uint8))[0]; del U
        mg = [int(np.ceil(40 / s)) for s in sp]
        bb = tuple(slice(max(0, b.start - k), min(n, b.stop + k)) for b, k, n in zip(bb, mg, I.shape))
        I = np.asarray(I[bb], dtype=np.float32); B = B[bb]; V = V[bb]; org = org[bb]; T = T[bb]
        # VT plan organs (entities TS lacks) and VT plan tumour union (for QC)
        pdir = os.path.join(SC.S, 'VoxTell', ds, case + '_plan'); tsn = set(names.values()); vto = []; vttum = np.zeros(org.shape, bool)
        # speed (27 Nov): read only VT organ files for entities TS lacks, plus each such entity's own tumour file; the initial VT
        # tumour mask V is also part of the QC tumour union.
        vttum |= V
        def rd(fpath):
            m = np.asanyarray(nib.load(fpath).dataobj)
            if m.ndim == 4: m = m[..., 0]
            if m.shape != org.shape: m = m[bb]
            return m > 0 if m.shape == org.shape else None
        # memory (27 Nov): VT organs are written straight into the TS+VT label map; no per-organ masks are kept.
        org_tsvt = org.copy(); vt_info = {}; nid = 1000
        for fpath in sorted(glob.glob(os.path.join(pdir, '*_organ.nii.gz'))):
            ent = os.path.basename(fpath)[:-len('_organ.nii.gz')]
            if re.search(r'_a\d+$', ent) or ent in tsn: continue
            m = rd(fpath)
            if m is None or not m.any(): continue
            tt = V.copy(); tf_ = os.path.join(pdir, ent + '_tumor.nii.gz')
            if os.path.exists(tf_):
                mt = rd(tf_)
                if mt is not None: tt |= mt
            kk = kb_key(ent); vol = float(m.sum() * vml)
            vt_info[nid] = dict(tum_overlap=float((m & tt).sum()) / max(int(m.sum()), 1), vol_ok=(kk is None or vol <= GP[kk]['vol_ml']['q95']))
            org_tsvt[(org_tsvt == 0) & m] = nid; names[nid] = ent; nid += 1; del m, tt
        maxsp = max(sp); tau = max(5.0, maxsp); edge_mm = max(2.0, float(np.sqrt(sum(s * s for s in sp))))
        lbT, nT = ndimage.label(T, structure=EX.STRUCT26); tsz = np.bincount(lbT.ravel(), minlength=nT + 1)
        P = None
        if pid:
            P0 = np.isin(org, pid)
            if P0.any():
                pb = ndimage.find_objects(P0.astype(np.uint8))[0]; m_ = [int(np.ceil(10 / s_)) + 1 for s_ in sp]
                pb = tuple(slice(max(0, b_.start - k_), min(n_, b_.stop + k_)) for b_, k_, n_ in zip(pb, m_, P0.shape))
                P = np.zeros(P0.shape, bool); P[pb] = ndimage.distance_transform_edt(~P0[pb], sampling=sp) <= 10
        osz = {}
        for lab in (org, org_tsvt):
            vv, cc = np.unique(lab, return_counts=True); osz.update({int(a): int(b) for a, b in zip(vv, cc) if a > 0})
        # boundary distance maps per arm; organ elongation; hollow-wall thickness maps (CT)
        def bdist(lab):
            bnd = np.zeros(lab.shape, bool)
            for ax in range(3):
                d = np.diff(lab, axis=ax) != 0; s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[ax] = slice(0, -1); s1[ax] = slice(1, None)
                bnd[tuple(s0)] |= d; bnd[tuple(s1)] |= d
            return ndimage.distance_transform_edt(~bnd, sampling=sp) if bnd.any() else np.full(lab.shape, 1e9)
        # boundary distance computed locally per candidate (speed): see local_bd below
        oel = {}; wall = {}
        for k in osz:
            if k >= 1000 or cat_of(names.get(k, '')) == 'organ' or k >= 1000:
                pass
        def organ_elong(k, lab):
            if k not in oel: oel[k] = elong_of(lab == k, sp)
            return oel[k]
        def wall_map(k):
            if k not in wall:
                w = (org == k) & (I >= -20) & (I <= 150)
                wb = ndimage.find_objects(w.astype(np.uint8))
                if not wb or wb[0] is None: wall[k] = (np.zeros((0, 3), int), np.zeros(0)); return wall[k]
                wb = wb[0]; tm = 2 * ndimage.distance_transform_edt(w[wb], sampling=sp)
                xyz = np.argwhere(w[wb]); vals = tm[w[wb]]; xyz = xyz + np.array([s_.start for s_ in wb])
                if len(vals) > 300000: st = len(vals) // 300000 + 1; xyz, vals = xyz[::st], vals[::st]
                wall[k] = (xyz, vals)
            return wall[k]
        rows = []
        for mode, M in (('BP', B), ('VT', V), ('BP+VT', B | V)):
            if mode not in MODES: continue
            lb, n = ndimage.label(M, structure=EX.STRUCT26)
            if n == 0: continue
            cands = []
            for j, ob in enumerate(ndimage.find_objects(lb), start=1):
                pad = tuple(slice(max(0, s.start - 15), min(sh, s.stop + 15)) for s, sh in zip(ob, I.shape)); cb = lb[pad] == j
                ml = float(cb.sum() * vml)
                if ml < MIN_ML: continue
                tr = bool(T[pad][cb].any()); ids = np.unique(lbT[pad][cb]); ids = ids[ids > 0]
                inter = int((T[pad] & cb).sum()); nc = int(cb.sum()); gs = int(tsz[ids].sum()) if len(ids) else 0
                insc = tr or (P is None) or (float(P[pad][cb].mean()) >= 0.5)
                th = float(2 * ndimage.distance_transform_edt(np.pad(cb, 1), sampling=sp).max()); el = elong_of(cb, sp)
                cands.append(dict(cid=j, pad=pad, cb=cb, ml=ml, true=tr, in_scope=bool(insc), in_frac=inter / max(nc, 1),
                                  dice_t=(2 * inter / (nc + gs)) if gs else 0.0, thick_mm=th, elong=el))
            for arm, lab in (('ts', org), ('tsvt', org_tsvt)):
                for c in cands:
                    pad, cb = c['pad'], c['cb']; L = lab[pad][cb]; nc = max(int(cb.sum()), 1)
                    vals, cnt = np.unique(L[L > 0], return_counts=True)
                    qcfail = False; host = 0; hshare = 0.0
                    order = sorted(zip(vals, cnt), key=lambda x: -x[1])
                    for v, ct in order:
                        v = int(v)
                        if v >= 1000:
                            m = lab[pad] == v; dice = 2 * ct / (nc + osz.get(v, 1)); inf = vt_info.get(v, {})
                            if dice >= 0.8 or inf.get('tum_overlap', 0) >= 0.5 or not inf.get('vol_ok', True):
                                qcfail = True; continue
                        host = v; hshare = ct / nc; break
                    hname = names.get(host, '') if host else ''; hcat = cat_of(hname) if host else 'none'
                    E1 = E2 = E3 = E5 = False
                    if host and hcat == 'organ' and hshare >= 0.5:
                        E1 = (c['elong'] > organ_elong(host, lab)) and (c['thick_mm'] <= 2 * maxsp)
                    if host and host < 1000 and hcat in ('bowel', 'hollow') and hshare >= 0.5 and mod == 'CT':
                        xyz, vals = wall_map(host)
                        # exclusion zone: the candidate's bounding box +- 10 mm (box approximation of 'candidate +- 10 mm')
                        mgx = [int(np.ceil(10 / s_)) for s_ in sp]
                        inb = np.ones(len(vals), bool)
                        for ax_, (p_, k_) in enumerate(zip(pad, mgx)): inb &= (xyz[:, ax_] >= p_.start - k_) & (xyz[:, ax_] < p_.stop + k_)
                        ref = vals[~inb]
                        if ref.size >= 50:
                            q90 = float(np.percentile(ref, 90)); cs = cb & (I[pad] >= -20) & (I[pad] <= 150)
                            cth = float(2 * ndimage.distance_transform_edt(np.pad(cs, 1), sampling=sp).max()) if cs.any() else 0.0
                            E2 = cth <= q90
                    bd = bdist(lab[pad])[cb]           # pad = candidate box + 15 voxels >= edge_mm on these grids
                    if (bd <= edge_mm).mean() >= 0.5 and c['thick_mm'] <= 2 * maxsp:
                        if host:
                            outside = cb & (lab[pad] != host)
                            prot = float(ndimage.distance_transform_edt(lab[pad] != host, sampling=sp)[outside].max()) if outside.any() else 0.0
                        else:
                            prot = 0.0
                        E3 = prot <= maxsp
                    if host and host < 1000:
                        dice = 2 * float((lab[pad][cb] == host).sum()) / (nc + osz.get(host, 1)); kk = kb_key(hname)
                        E5 = dice >= 0.8 and kk is not None and osz.get(host, 0) * vml <= GP[kk]['vol_ml']['q95']
                    E4 = (not np.isnan(c['elong'])) and c['elong'] > TAU_ELONG
                    c[arm] = dict(host=host, hname=hname, E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, qc=qcfail)
            # fragment groups (same host in the tsvt arm; ts arm uses its own host)
            for arm in ('ts', 'tsvt'):
                k = len(cands); parent = list(range(k))
                def f(i):
                    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
                    return i
                it = [int(np.ceil(tau / s_)) + 1 for s_ in sp]
                for a in range(k):
                    pa = cands[a]['pad']; box = tuple(slice(max(0, p_.start - t_), min(n_, p_.stop + t_)) for p_, t_, n_ in zip(pa, it, I.shape))
                    A = np.zeros(tuple(b_.stop - b_.start for b_ in box), bool)
                    A[tuple(slice(p_.start - b_.start, p_.stop - b_.start) for p_, b_ in zip(pa, box))] = cands[a]['cb']
                    Ad = ndimage.distance_transform_edt(~A, sampling=sp) <= tau
                    for b in range(a + 1, k):
                        if cands[a][arm]['host'] != cands[b][arm]['host']: continue
                        pb = cands[b]['pad']
                        lo = [max(p_.start, b_.start) for p_, b_ in zip(pb, box)]; hi = [min(p_.stop, b_.stop) for p_, b_ in zip(pb, box)]
                        if any(l_ >= h_ for l_, h_ in zip(lo, hi)): continue
                        sb = cands[b]['cb'][tuple(slice(l_ - p_.start, h_ - p_.start) for l_, h_, p_ in zip(lo, hi, pb))]
                        sa = Ad[tuple(slice(l_ - b_.start, h_ - b_.start) for l_, h_, b_ in zip(lo, hi, box))]
                        if (sb & sa).any(): parent[f(b)] = f(a)
                for i, c in enumerate(cands):
                    e = c[arm]; expl = e['E1'] or e['E2'] or e['E3'] or e['E5']
                    rows.append(dict(base, arm=arm, mode=mode, cid=c['cid'], ml=c['ml'], true=c['true'], in_scope=c['in_scope'],
                                     in_frac=c['in_frac'], dice_t=c['dice_t'], group=f(i), host=e['hname'], thick_mm=c['thick_mm'],
                                     elong=c['elong'], E1=e['E1'], E2=e['E2'], E3=e['E3'], E4=e['E4'], E5=e['E5'], explained=expl,
                                     vt_qc_fail=e['qc']))
        return rows or [dict(base, note='no candidates')]
    except Exception as e:
        import traceback
        return [dict(base, note='error ' + repr(e)[:120] + ' ' + traceback.format_exc().splitlines()[-2][:80])]


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150; W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if not os.path.exists(JF): make_jobs().to_csv(JF, index=False)
    J = pd.read_csv(JF); done = set()
    for f_ in {O, MAIN}:
        if os.path.exists(f_):
            d = pd.read_csv(f_, usecols=['ds', 'case']); done |= set(zip(d.ds, d.case.astype(str)))
    todo = [(r.ds, str(r.case), bool(r.tumour_free)) for r in J.itertuples() if (r.ds, str(r.case)) not in done]
    print('jobs', len(J), 'todo', len(todo), flush=True); t0 = time.time(); n = 0
    with Pool(W) as p:
        for rows in p.imap_unordered(one, todo):
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(O, mode='a', index=False, header=not os.path.exists(O)); n += 1
            if time.time() - t0 > budget: p.terminate(); break
    print('processed', n, 'in', round(time.time() - t0), 's', flush=True)
