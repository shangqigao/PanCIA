"""Evidence screen for agent-invariant stage 1 (26 Nov 2026). Three independent reasonings: candidates are the 26-connected components of
the raw BP mask ('BP'), of the raw (initial) VT mask ('VT'), and of their union ('BP+VT'). Every feature is computed from the candidate's
own outline and the image only (no use of the other agent), so it is defined identically in every mode. Expert labels are used for
evaluation only (true = touches an expert tumour; in_scope = touches an expert tumour or lies >= 50% within the primary organ + 10 mm, so
R2Seg's primary-only labels do not score non-primary candidates; breast / cervix MR have no TS organ: whole scan in scope).
Features (robust, per candidate, from a 5 mm ring around it, inside the host organ when the candidate lies in one):
  z_med   |median(in) - median(ring)| / MAD(ring)          local contrast
  ks      KS statistic in vs ring                           distribution difference
  edge    mean |grad| on the boundary / median |grad| in the ring
  het     IQR(in) / IQR(ring)
  z_host  |median(in) - median(host organ)| / MAD(host)     contrast to the whole host organ (NaN without a host)
  sph     sphericity (36 pi V^2)^(1/3) / surface (face count)
  elong   sqrt(largest / smallest PCA eigenvalue) of the voxel coordinates (mm)
  nsl     number of slices along the slice (largest-spacing) axis
  log_ml  log10 volume
  in_org  share of the candidate inside any TS structure
Sampling (26 Nov, corrected): make_jobs. in_scope applies to every scan, tumour and tumour-free alike.
Output: PanCIA_outputs/ledger/evidence_screen/candidates.csv. Usage: python s1_evidence_screen.py [budget_s] [workers] [n_per_set]"""
import os, sys, json, glob, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from scipy.stats import ks_2samp
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402
R = paths.root('R2Seg'); S = paths.root('R2Seg_Seg')
OUT = os.path.join(paths.ledger_dir(), 'evidence_screen'); O = os.path.join(OUT, os.environ.get('SCREEN_OUT', 'candidates.csv'))
PRIMARY = {'kidney': ['kidney_left', 'kidney_right'], 'liver': ['liver'], 'pancreas': ['pancreas'], 'colon': ['colon'],
           'lung': ['lung_left', 'lung_right', 'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                    'lung_middle_lobe_right', 'lung_lower_lobe_right'], 'urinary_bladder': ['urinary_bladder'], 'prostate': ['prostate']}
COLS = ['ds', 'case', 'site', 'tumour_free', 'mode', 'cid', 'ml', 'true', 'in_scope', 'host', 'z_med', 'ks', 'edge', 'het', 'z_host',
        'sph', 'elong', 'nsl', 'log_ml', 'in_org', 'in_frac', 'dice_t', 'note']
MIN_ML = 0.05; RING_MM = 5.0


def load(p):
    return nib.load(p)


def mad(x):
    m = np.median(x); return max(float(np.median(np.abs(x - m))) * 1.4826, 1e-6)


def feats(Ib, cb, ob, names, sp, hstat):
    st = ndimage.generate_binary_structure(3, 1)
    dil = ndimage.binary_dilation(cb, st, iterations=max(1, int(np.ceil(RING_MM / np.mean(sp)))))
    ring = dil & ~cb
    lab_in = ob[cb]; host = int(np.bincount(lab_in[lab_in > 0]).argmax()) if (lab_in > 0).mean() >= 0.5 else 0
    if host:
        rh = ring & (ob == host)
        if rh.sum() >= 20: ring = rh
    vin = Ib[cb].astype(np.float32); vr = Ib[ring].astype(np.float32)
    o = dict(host=names.get(str(host), '') if host else '', in_org=float((lab_in > 0).mean()))
    if vr.size < 10 or vin.size < 3:
        o.update(z_med=np.nan, ks=np.nan, edge=np.nan, het=np.nan)
    else:
        o['z_med'] = abs(float(np.median(vin) - np.median(vr))) / mad(vr)
        rng = np.random.default_rng(0)
        a = vin if vin.size <= 4000 else rng.choice(vin, 4000, replace=False); b = vr if vr.size <= 4000 else rng.choice(vr, 4000, replace=False)
        o['ks'] = float(ks_2samp(a, b).statistic)
        q = lambda v: float(np.subtract(*np.percentile(v, [75, 25])))
        o['het'] = q(vin) / max(q(vr), 1e-6)
        g = np.sqrt(sum(np.square(ndimage.sobel(Ib.astype(np.float32), axis=k) / sp[k]) for k in range(3)))
        bnd = cb & ~ndimage.binary_erosion(cb, st)
        o['edge'] = float(g[bnd].mean()) / max(float(np.median(g[ring])), 1e-6)
    if host and host in hstat:
        o['z_host'] = abs(float(np.median(vin) - hstat[host][0])) / hstat[host][1]
    else:
        o['z_host'] = np.nan
    vml = float(np.prod(sp)) / 1000; V = cb.sum() * np.prod(sp)
    faces = sum(float((np.diff(np.pad(cb, [(1, 1) if k == j else (0, 0) for k in range(3)]).astype(np.int8), axis=j) != 0).sum())
                * np.prod([sp[k] for k in range(3) if k != j]) for j in range(3))
    o['sph'] = float((36 * np.pi * V ** 2) ** (1 / 3) / max(faces, 1e-6))
    xyz = np.argwhere(cb) * np.asarray(sp)
    if len(xyz) >= 4:
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T)))
        o['elong'] = float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))
    else:
        o['elong'] = np.nan
    ax = int(np.argmax(sp)); o['nsl'] = int(np.any(cb, axis=tuple(k for k in range(3) if k != ax)).sum())
    o['ml'] = float(cb.sum() * vml); o['log_ml'] = float(np.log10(max(o['ml'], 1e-3)))
    return o


# ---- error-type audit (26 Nov): AUDIT=1 replaces the screen features by simple TS-relation / geometry / CT-density measures
AUDIT = os.environ.get('AUDIT') == '1'
CAT_KEYS = (('vessel', ('aorta', 'vena_cava', 'portal_vein', 'iliac_artery', 'iliac_vena', 'pulmonary_vein', 'pulmonary_artery',
                        'brachiocephalic', 'subclavian', 'carotid', 'jugular', 'artery', 'vein', 'vena')),
            ('bowel', ('colon', 'small_bowel', 'duodenum', 'stomach', 'esophagus', 'rectum')),
            ('bone', ('vertebra', 'rib_', 'sacrum', 'hip_', 'femur', 'humerus', 'scapula', 'clavicula', 'sternum', 'skull',
                      'costal', 'intervertebral')),
            ('hollow', ('urinary_bladder', 'gallbladder')),
            ('node', ('ln_', 'lymph')))
AUDIT_COLS = ['ds', 'case', 'site', 'tumour_free', 'mode', 'cid', 'ml', 'mod', 'sh_vessel', 'sh_bowel', 'sh_bone', 'sh_hollow',
              'sh_node', 'sh_organ', 'sh_vtonly', 'sh_any', 'top_label', 'top_cov', 'sh_surf2', 'thick_mm', 'flat', 'elong', 'hu_med', 'hu_iqr', 'note']


def lab_cat(names):
    lut = {}
    for k, v in names.items():
        c = 'organ'
        for cat, keys in CAT_KEYS:
            if any(x in v for x in keys): c = cat; break
        lut[int(k)] = c
    return lut


def audit_feats(Ib, cb, ob, names, sp, lut, osz, mod):
    lab = ob[cb]; n = max(int(cb.sum()), 1); o = {}
    for cat in ('vessel', 'bowel', 'bone', 'hollow', 'node', 'organ'):
        ids = [k for k, c in lut.items() if c == cat]
        o['sh_' + cat] = float(np.isin(lab, ids).sum()) / n if ids else 0.0
    o['sh_any'] = float((lab > 0).mean()); o['sh_vtonly'] = float((lab >= 1000).mean())
    vals, cnt = np.unique(lab[lab > 0], return_counts=True)
    if len(vals):
        covs = [(int(v), c / max(osz.get(int(v), 1), 1)) for v, c in zip(vals, cnt)]
        top = max(covs, key=lambda x: x[1]); o['top_label'] = names.get(str(top[0]), ''); o['top_cov'] = float(top[1])
    else:
        o['top_label'] = ''; o['top_cov'] = 0.0
    bnd = np.zeros(ob.shape, bool)
    for ax in range(3):
        d = np.diff(ob, axis=ax) != 0
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3; sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        bnd[tuple(sl0)] |= d; bnd[tuple(sl1)] |= d
    o['sh_surf2'] = float((ndimage.distance_transform_edt(~bnd, sampling=sp)[cb] <= 2.0).mean()) if bnd.any() else 0.0
    o['thick_mm'] = float(2 * ndimage.distance_transform_edt(np.pad(cb, 1), sampling=sp).max())
    xyz = np.argwhere(cb) * np.asarray(sp)
    if len(xyz) >= 4:
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T)))
        o['flat'] = float(np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9))); o['elong'] = float(np.sqrt(ev[-1] / max(ev[0], 1e-6)))
    else:
        o['flat'] = np.nan; o['elong'] = np.nan
    v = Ib[cb]; o['hu_med'] = float(np.median(v)); o['hu_iqr'] = float(np.subtract(*np.percentile(v, [75, 25]))); o['mod'] = mod
    return o


def one(job):
    ds, case, tf = job; site = EX.SITE_ENTITY.get(ds, '')
    base = dict(ds=ds, case=case, site=site, tumour_free=tf)
    try:
        ip = os.path.join(R, ds, 'images', case + '.nii.gz'); lp = os.path.join(R, ds, 'labels', case + '_seg.nii.gz')
        bp = os.path.join(S, 'BiomedParse', ds, case + '_tumor.nii.gz'); vp = os.path.join(S, 'VoxTell', ds, case + '_tumor.nii.gz')
        tp = os.path.join(S, 'TotalSegmentator', ds, case + '_organ.nii.gz')
        if not all(os.path.exists(p) for p in (ip, bp, vp, tp)):
            return [dict(base, note='missing input')]
        # memory (26 Nov): whole-body CT exceeded the 4 GB VM with 3 workers. Keep the image in its stored dtype, and crop every
        # array to the box of (BP | VT | expert tumour | primary organ) + margin after the host statistics are taken.
        img = load(ip); I = np.asanyarray(img.dataobj)
        if I.ndim == 4: I = I[..., 0]
        sp = tuple(float(x) for x in img.header.get_zooms()[:3])
        B = np.asanyarray(load(bp).dataobj) > 0; V = np.asanyarray(load(vp).dataobj) > 0
        org = np.asanyarray(load(tp).dataobj).astype(np.int16)
        names = json.load(open(tp.replace('_organ.nii.gz', '_organ_labels.json')))['labels']
        T = np.isin(EX._read_labels(lp), EX.APPEARANCE[ds]['tumour']) if os.path.exists(lp) else np.zeros(I.shape, bool)
        if not (B.shape == V.shape == org.shape == T.shape == I.shape):
            return [dict(base, note='grid mismatch')]
        hstat = {}
        o1 = org.ravel(); i1 = I.ravel(); step = max(1, o1.size // 3000000)
        os_, is_ = o1[::step], i1[::step].astype(np.float32)
        for k in np.unique(os_):
            if k > 0:
                v = is_[os_ == k]
                if v.size >= 50: hstat[int(k)] = (float(np.median(v)), mad(v))
        del o1, i1, os_, is_
        pid0 = [int(k) for k, v in names.items() if v in PRIMARY.get(site, [])]
        U = B | V | T
        if pid0: U |= np.isin(org, pid0)
        if not U.any():
            return [dict(base, note='no candidates')]
        bb = ndimage.find_objects(U.astype(np.uint8))[0]; del U
        mg = [int(np.ceil(25 / s)) for s in sp]
        bb = tuple(slice(max(0, b.start - k), min(n, b.stop + k)) for b, k, n in zip(bb, mg, I.shape))
        I = np.asarray(I[bb], dtype=np.float32); B = B[bb]; V = V[bb]; org = org[bb]; T = T[bb]
        if AUDIT and os.environ.get('AUDIT_VT') == '1':
            # 26 Nov (user): the anatomy layer is TS + VoxTell plan organ masks; VT supplies the entities TS cannot segment.
            # TS keeps precedence; VT organs fill unlabelled voxels; main prompt file only (no _a2/_a3 aliases).
            import re
            pdir = os.path.join(S, 'VoxTell', ds, case + '_plan'); tsn = set(names.values()); nid = 1000
            for fpath in sorted(glob.glob(os.path.join(pdir, '*_organ.nii.gz'))):
                ent = os.path.basename(fpath)[:-len('_organ.nii.gz')]
                if re.search(r'_a\d+$', ent) or ent in tsn: continue
                try:
                    mvt = np.asanyarray(load(fpath).dataobj)
                    if mvt.shape[:3] != tuple(n.stop - n.start for n in bb) and mvt.shape[:3] != B.shape and len(mvt.shape) >= 3:
                        mvt = mvt[bb] if mvt.ndim == 3 else mvt[..., 0][bb]
                    mvt = mvt > 0
                    if mvt.shape != org.shape: continue
                    org[(org == 0) & mvt] = nid; names[str(nid)] = ent; nid += 1
                except Exception:
                    continue
        pid = [int(k) for k, v in names.items() if v in PRIMARY.get(site, [])]
        if pid:
            P0 = np.isin(org, pid); P = None
            if P0.any():                                       # primary organ + 10 mm (EDT on the organ's bounding box)
                bb = ndimage.find_objects(P0.astype(np.uint8))[0]
                m = [int(np.ceil(10 / s)) + 1 for s in sp]
                bb = tuple(slice(max(0, b.start - k), min(n, b.stop + k)) for b, k, n in zip(bb, m, P0.shape))
                P = np.zeros(P0.shape, bool); P[bb] = ndimage.distance_transform_edt(~P0[bb], sampling=sp) <= 10
        else:
            P = None
        if AUDIT:
            lut = lab_cat(names); vv, cc = np.unique(org, return_counts=True); osz = {int(a): int(b) for a, b in zip(vv, cc) if a > 0}
        lbT, nT = ndimage.label(T, structure=EX.STRUCT26); tsz = np.bincount(lbT.ravel(), minlength=nT + 1)   # strict truth (26 Nov)
        rows = []
        for mode, M in (('BP', B), ('VT', V), ('BP+VT', B | V)):
            lb, n = ndimage.label(M, structure=EX.STRUCT26)
            if n == 0: continue
            objs = ndimage.find_objects(lb); vml = np.prod(sp) / 1000
            for j, ob in enumerate(objs, start=1):
                pad = [(max(0, s.start - 12), min(sh, s.stop + 12)) for s, sh in zip(ob, I.shape)]
                sl = tuple(slice(a, b) for a, b in pad); cb = lb[sl] == j
                ml = float(cb.sum() * vml)
                if ml < MIN_ML: continue
                tr = bool(T[sl][cb].any())
                ids = np.unique(lbT[sl][cb]); ids = ids[ids > 0]
                inter = int((T[sl] & cb).sum()); nc = int(cb.sum()); gs = int(tsz[ids].sum()) if len(ids) else 0
                strict = dict(in_frac=inter / max(nc, 1), dice_t=(2 * inter / (nc + gs)) if gs else 0.0)
                insc = tr or (P is None) or (float(P[sl][cb].mean()) >= 0.5)
                if AUDIT:
                    f = audit_feats(I[sl], cb, org[sl], names, sp, lut, osz, EX.APP_MOD.get(ds, 'CT'))
                    rows.append(dict(base, mode=mode, cid=j, ml=ml, **f)); continue
                f = feats(I[sl], cb, org[sl], names, sp, hstat)
                rows.append(dict(base, mode=mode, cid=j, true=tr, in_scope=bool(insc), **f, **strict))
        return rows or [dict(base, note='no candidates')]
    except Exception as e:
        return [dict(base, note='error ' + repr(e)[:150])]


def patient(ds, case):
    """one image per patient: R2Seg case names -> patient id"""
    import re
    if ds == 'Breast_Tumor_00': return re.sub(r'_\d{4}$', '', case)
    if ds == 'Cervix_Tumor_00': return case.split('_MR')[0]
    if ds == 'Prostate_Tumor_00': return case.split('_')[0]
    return case


def sequence(ds, case):
    return case.split('_MR', 1)[1].split('_', 1)[1] if ds == 'Cervix_Tumor_00' and '_MR' in case else ''


def make_jobs(n_per, seed=0):
    """26 Nov (user): tumour-free = cohort/r2seg_tumour_free.csv status 'negative' only (139: NIH-like pancreas 80, Prostate158
    controls 56, lung 3), all of them. Tumour scans: scans with an expert tumour and not listed in that file (the 222 unannotated
    FLARE23 scans and the ADC-only prostate case are excluded). n_per tumour scans per SITE (Lung_Tumor_00 and _01 pooled), one
    image per patient, spread evenly over sequences where a site has several. Uterus (benign) excluded."""
    cf = os.path.join(os.path.dirname(paths.ledger_dir()), 'cohort', 'r2seg_tumour_free.csv')
    C = pd.read_csv(cf)[['ds', 'case', 'status']]
    t = pd.read_csv(os.path.join(paths.ledger_dir(), 'v86_r2seg_bp2', 'diag', 'anchor_truth_lite.csv'), usecols=['ds', 'case'],
                    low_memory=False).drop_duplicates()
    t = t[t.ds != 'Uterus_Tumor_00']; t['case'] = t.case.astype(str)
    t = t.merge(C, on=['ds', 'case'], how='left'); t['status'] = t.status.fillna('tumour')
    neg = t[t.status == 'negative'].assign(tumour_free=True)
    tum = t[t.status == 'tumour'].copy(); tum['site'] = tum.ds.map(EX.SITE_ENTITY)
    tum['pat'] = [patient(d, c) for d, c in zip(tum.ds, tum.case)]; tum['seq'] = [sequence(d, c) for d, c in zip(tum.ds, tum.case)]
    rng = np.random.default_rng(seed); out = []
    for site, g in tum.groupby('site'):
        g = g.sample(frac=1, random_state=seed).drop_duplicates('pat')           # one image per patient (random)
        seqs = sorted(g.seq.unique()); k = 0; pick = []
        pools = {q: list(g[g.seq == q].index) for q in seqs}
        while len(pick) < min(n_per, len(g)):                                     # round-robin over sequences
            q = seqs[k % len(seqs)]; k += 1
            if pools[q]: pick.append(pools[q].pop(0))
        out.append(g.loc[pick])
    tum = pd.concat(out).assign(tumour_free=False)
    return pd.concat([tum[['ds', 'case', 'tumour_free']], neg[['ds', 'case', 'tumour_free']]], ignore_index=True)


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 160; W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    os.makedirs(OUT, exist_ok=True)
    jf = os.path.join(OUT, 'jobs.csv')
    if not os.path.exists(jf):
        J = make_jobs(n_per); J.to_csv(jf, index=False)
    J = pd.read_csv(jf)
    done = set()
    if os.path.exists(O):
        d = pd.read_csv(O, usecols=['ds', 'case'], low_memory=False); done = set(zip(d.ds, d.case.astype(str)))
    todo = [(r.ds, str(r.case), bool(r.tumour_free)) for r in J.itertuples() if (r.ds, str(r.case)) not in done]
    print('jobs', len(J), 'todo', len(todo), flush=True)
    t0 = time.time(); n = 0
    with Pool(W) as p:
        for rows in p.imap_unordered(one, todo):
            pd.DataFrame(rows).reindex(columns=AUDIT_COLS if AUDIT else COLS).to_csv(O, mode='a', index=False, header=not os.path.exists(O)); n += 1
            if time.time() - t0 > budget:
                p.terminate(); break
    print('processed', n, 'in', round(time.time() - t0), 's', flush=True)
