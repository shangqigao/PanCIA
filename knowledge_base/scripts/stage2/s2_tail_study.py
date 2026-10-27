"""Stage 2, review study (no effect on the fit): tails of tumour interiors and of the normal tissue around them, by order of statistic
(step 1, evidence), and whether one tail shape is shared across orders (step 2, model form). Expert R2Seg data only.

Per usable scan (index: usable, TS mask, not ADC; spatial collections without Breast, which has no host organ), largest tumour G:
  regions   tumour  interior of G (clearance from G's edge)
            ring    host tissue > 5 mm from every tumour, >= 3 mm inside host+tumour, widened until its voxel count matches the tumour
                    interior's (cap 35 mm)  [as review/tails2]
            far     host tissue farther than max(30 mm, ring width) from every tumour (same organ, away from the tumour)
            host = expert organ label where the collection has one (Kidney, Liver, Pancreas, Lung_01, Cervix, Prostate), else the
            TotalSegmentator host (knowledge base host labels)
  per statistic (pancia_kb.stage2.tails.STATS: I, D1, D2, G1 and LoG at sigma 1, 2, 4 mm): Gaussian and Student-t fits at EQUAL n
            (n = min(4000, tumour, ring); far at the same n when it has enough voxels, flagged otherwise)
  components  1..4 Student-t components (K by the lower bound, nu per component) on I, LoG_s1 and the joint features, n <= 2000
  form      joint features (I, D1 two axes, LoG_s1): shared-scale multivariate t vs order-specific t blocks (tails.model_form)
  covariates noise SD (normal tissue), in-plane spacing, slice thickness, stride, tumour volume, number of tumours >= 0.5 ml
Per tumour-free scan (index status no_tumour; at most --n-neg per collection): the host organ interior, fitted at n = 500, 1000,
2000, 4000 (as available) so that the summary can compare with rings at matched n.

Outputs clinical/PanCIA_outputs/stage2/tail_study/: records[_s<i>].jsonl (one per scan, all fit parameters), done[_s<i>].txt,
notes[_s<i>].txt (errors), samples/<ds>__<case>.npz (the fitted samples, float16, for re-plotting).
--summary writes the tidy tables (fits.csv in the review/tails2_fits.csv format plus covariates, components.csv, form.csv) and the
statistics on fits with n >= --min-n voxels (default 200; small tumours give few grid voxels) (stats_paired.csv, stats_negatives.csv, stats_confounds.csv, stats_form.csv, stats_components.csv, report.txt).
Sites count equally in every pooled number (fit.site_weights).

Usage: [SHARD=i/n] python s2_tail_study.py [--n-per N (0 = all)] [--n-neg 50] [--budget SECONDS] [--pause FRACTION]
       [SHARD=i/n] python s2_tail_study.py --tumour-only --n-per 100     (breast: tumour interiors only, 100 random usable scans)
Load: one thread per process; only the through-plane slices on the 3 mm grid are filtered (statistics are in-plane).
       python s2_tail_study.py --summary [--min-n 200]"""
import os, sys, json, time, zlib, glob, argparse, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')      # one thread per process (set before numpy): shards x BLAS threads overloaded the laptop
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import numpy as np, pandas as pd, nibabel as nib   # noqa: E402
from scipy import ndimage, stats as SS   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, tails as TL, fit as FT   # noqa: E402

EXPERT_HOST = {'Kidney_Tumor_00': [1], 'Liver_Tumor_00': [1], 'Pancreas_Tumor_00': [4], 'Lung_Tumor_01': [1], 'Cervix_Tumor_00': [1],
               'Prostate_Tumor_00': [1, 2]}
N_FIT, N_COMP, NEG_N = 4000, 2000, (500, 1000, 2000, 4000)
COMP_FEATS = ('I', 'LoG_s1', 'joint')
KB = None


def _kb():
    global KB
    if KB is None:
        from pancia_kb import KnowledgeBase
        KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    return KB


def _host(ds, case, A, W, img):
    """host organ mask on the crop W, and its source"""
    if ds in EXPERT_HOST and np.isin(A, EXPERT_HOST[ds]).any():
        return np.isin(A[W], EXPERT_HOST[ds]), 'expert'
    from pancia_kb.adapter import _host_ts_labels
    md, names, task = EX._load_ts(ds, case, W, img)
    hl = list(_host_ts_labels(_kb(), task, names, EX.SITE_ENTITY[ds])) if md is not None else []
    return (np.isin(md, hl) if hl else np.zeros(tuple(s.stop - s.start for s in W), bool)), 'TS'


def _sub(v, n, rng):
    return v if len(v) <= n else v[rng.choice(len(v), n, replace=False)]


def _open(ds, f):
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', f); A = EX._read_labels(lf)
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); si = [k for k, c in enumerate(nib.aff2axcodes(img.affine)) if c in 'SI'][0]
    return A, img, sp, si, EX.APP_MOD.get(ds, 'CT'), f.replace('_seg.nii.gz', '')


def _meta(ds, f, mod, sp2, ax, st, X, Rn, Cn):
    """scan covariates; noise from all slices of the crop (before the grid-slice thinning)"""
    ip = [a for a in range(3) if a != ax]
    return dict(ds=ds, file=f, mod=mod, site=EX.SITE_OF[ds], sp_inplane=round(float(np.mean([sp2[a] for a in ip])), 3),
                thick_mm=round(float(sp2[ax]), 3), strided=st is not None, noise=TL.noise_sd({'D2': ([TL.laplacian(X, ax)], 3.0)}, Rn, Cn))


def run_tumour(ds, f, key):
    A, img, sp, si, mod, case = _open(ds, f)
    Tf = np.isin(A, EX.APPEARANCE[ds]['tumour']); W = F.pad_box(F.bbox(Tf), sp, 80.0, A.shape)
    H, src = _host(ds, case, A, W, img); T = Tf[W]; del A, Tf
    X, sp2, st = EX._image_crop(img, mod, W, sp, si, T.size, 3e6)
    if st is not None:
        T = T[st]; H = H[st]
    thick, ax = F.acq_axis(sp2, si); B = TL.border_mm(T.shape, sp2)
    lb, _ = ndimage.label(T, structure=EX.STRUCT26); sz = np.bincount(lb.ravel()) * np.prod(sp2) / 1000; sz[0] = 0
    G = lb == int(np.argmax(sz)); del lb
    Ct = np.minimum(F.inner_depth(G, sp2), B)
    dT = F.edt(T, sp2); IDH = F.inner_depth(H | T, sp2); Cn = np.minimum(np.minimum(IDH, dT), B)
    C = H & ~T & (dT > 5.0) & (IDH >= 3.0)
    nti = int((G & (Ct >= 3.0)).sum()); width = None; ring = np.zeros(T.shape, bool); far = ring
    if C.any() and nti:
        dv = np.sort(dT[C]); k = min(len(dv), nti) - 1
        width = float(min(max(dv[k], 6.0), 35.0)); ring = C & (dT <= width); far = C & (dT > max(30.0, width))
    meta = _meta(ds, f, mod, sp2, ax, st, X, C, Cn)
    sl = TL.grid_slices(T.shape[ax], sp2[ax]); tk = lambda a: np.take(a, sl, axis=ax)     # grid slices only (in-plane statistics)
    G, Ct, Cn, C, ring, far, X = (tk(a) for a in (G, Ct, Cn, C, ring, far, X)); del T, H, dT, IDH, B
    spg = sp2.copy(); spg[ax] = 3.0; grid = TL.grid_mask(X.shape, spg)
    maps = TL.filters(X, sp2, ax)
    rec = dict(kind='tumour_scan', host_src=src, ring_mm=width, tumour_ml=round(float(sz.max()), 3),
               n_tumours=int((sz >= EX.MIN_ML).sum()), **meta)
    reg = {'tumour': (G, Ct), 'ring': (ring, Cn), 'far': (far, Cn)}
    S = {nm: TL.statistics(maps, R, Cc, grid) for nm, (R, Cc) in reg.items() if R.any()}
    J = {nm: TL.joint(maps, R, Cc, grid) for nm, (R, Cc) in reg.items() if R.any()}
    rec['n_vox'] = {nm: int(len(v['I'])) for nm, v in S.items()}
    rng = np.random.default_rng(zlib.crc32(key.encode())); fits = {}; smp = {}
    if 'tumour' in S and 'ring' in S:
        for k in TL.STATS:
            n = min(N_FIT, len(S['tumour'][k]), len(S['ring'][k]))
            if n < 50:
                continue
            for nm in ('tumour', 'ring', 'far'):
                if nm not in S or len(S[nm][k]) < 50:
                    continue
                v = _sub(S[nm][k], n, rng); r = TL.fit_1d(v); r['n_matched'] = len(v) == n
                fits.setdefault(nm, {})[k] = r; smp[f'{nm}__{k}'] = v.astype(np.float16)
    rec['fits'] = fits
    comps = {}; form = {}
    if 'tumour' in J and 'ring' in J:
        nc = min(N_COMP, len(J['tumour']), len(J['ring'])); nm_ = min(N_FIT, len(J['tumour']), len(J['ring']))
        for nm in ('tumour', 'ring', 'far'):
            if nm not in J:
                continue
            if nc >= 100 and len(J[nm]) >= nc:
                Yc = _sub(J[nm], nc, rng)
                comps[nm] = {'I': TL.fit_components(Yc[:, 0]), 'LoG_s1': TL.fit_components(Yc[:, 3]), 'joint': TL.fit_components(Yc)}
            if nm_ >= 100 and len(J[nm]) >= nm_:
                Ym = _sub(J[nm], nm_, rng); form[nm] = TL.model_form(Ym); smp[f'{nm}__joint'] = Ym.astype(np.float16)
    rec['components'] = comps; rec['form'] = form
    np.savez_compressed(os.path.join(paths.out_dir('tail_study', 'samples'), f'{ds}__{case}.npz'), **smp)
    return rec


def run_tumour_only(ds, f, key):
    """tumour interior only (collections without a segmented host, e.g. breast): the tumour-prior data of the tail-weight prior
    (s2_nu_prior.py). Same statistics, fits and joint-feature components as run_tumour, without ring or far tissue."""
    A, img, sp, si, mod, case = _open(ds, f)
    Tf = np.isin(A, EX.APPEARANCE[ds]['tumour']); W = F.pad_box(F.bbox(Tf), sp, 10.0, A.shape); T = Tf[W]; del A, Tf
    X, sp2, st = EX._image_crop(img, mod, W, sp, si, T.size, 3e6)
    if st is not None:
        T = T[st]
    thick, ax = F.acq_axis(sp2, si); B = TL.border_mm(T.shape, sp2)
    lb, _ = ndimage.label(T, structure=EX.STRUCT26); sz = np.bincount(lb.ravel()) * np.prod(sp2) / 1000; sz[0] = 0
    G = lb == int(np.argmax(sz)); del lb
    Ct = np.minimum(F.inner_depth(G, sp2), B)
    ip = [a_ for a_ in range(3) if a_ != ax]
    rec = dict(kind='tumour_only_scan', ds=ds, file=f, mod=mod, site=EX.SITE_OF[ds], host_src=None, ring_mm=None,
               tumour_ml=round(float(sz.max()), 3), n_tumours=int((sz >= EX.MIN_ML).sum()), noise=None,
               sp_inplane=round(float(np.mean([sp2[a_] for a_ in ip])), 3), thick_mm=round(float(sp2[ax]), 3), strided=st is not None)
    sl = TL.grid_slices(T.shape[ax], sp2[ax]); G, Ct, X = (np.take(a_, sl, axis=ax) for a_ in (G, Ct, X))
    spg = sp2.copy(); spg[ax] = 3.0; grid = TL.grid_mask(X.shape, spg); maps = TL.filters(X, sp2, ax)
    S = TL.statistics(maps, G, Ct, grid); J = TL.joint(maps, G, Ct, grid); rec['n_vox'] = {'tumour': int(len(S['I']))}
    rng = np.random.default_rng(zlib.crc32(key.encode())); fits = {}; smp = {}
    for k in TL.STATS:
        if len(S[k]) >= 50:
            v = _sub(S[k], N_FIT, rng); r = TL.fit_1d(v); r['n_matched'] = True
            fits.setdefault('tumour', {})[k] = r; smp[f'tumour__{k}'] = v.astype(np.float16)
    rec['fits'] = fits; rec['components'] = {}; rec['form'] = {}
    if len(J) >= 100:
        Yc = _sub(J, N_COMP, rng)
        rec['components'] = {'tumour': {'I': TL.fit_components(Yc[:, 0]), 'LoG_s1': TL.fit_components(Yc[:, 3]), 'joint': TL.fit_components(Yc)}}
        Ym = _sub(J, N_FIT, rng); rec['form'] = {'tumour': TL.model_form(Ym)}; smp['tumour__joint'] = Ym.astype(np.float16)
    np.savez_compressed(os.path.join(paths.out_dir('tail_study', 'samples'), f'{ds}__{case}.npz'), **smp)
    return rec


def run_negative(ds, f, key):
    A, img, sp, si, mod, case = _open(ds, f)
    full = tuple(slice(0, n) for n in A.shape)
    H, src = _host(ds, case, A, full, img)
    if H.sum() * np.prod(sp) / 1000 < 5.0:
        return dict(kind='negative_scan', ds=ds, file=f, skipped='no host >= 5 ml')
    W = F.pad_box(F.bbox(H), sp, 10.0, A.shape); H = H[W]; del A
    X, sp2, st = EX._image_crop(img, mod, W, sp, si, H.size, 3e6)
    if st is not None:
        H = H[st]
    thick, ax = F.acq_axis(sp2, si); B = TL.border_mm(H.shape, sp2)
    Ch = np.minimum(F.inner_depth(H, sp2), B); meta = _meta(ds, f, mod, sp2, ax, st, X, H, Ch)
    sl = TL.grid_slices(H.shape[ax], sp2[ax]); H, Ch, X = (np.take(a, sl, axis=ax) for a in (H, Ch, X)); del B
    spg = sp2.copy(); spg[ax] = 3.0; grid = TL.grid_mask(X.shape, spg); maps = TL.filters(X, sp2, ax)
    rec = dict(kind='negative_scan', host_src=src, **meta)
    S = TL.statistics(maps, H, Ch, grid); rec['n_vox'] = {'organ': int(len(S['I']))}
    rng = np.random.default_rng(zlib.crc32(key.encode())); fits = {}; smp = {}
    for k in TL.STATS:
        for n in NEG_N:
            if len(S[k]) >= n:
                v = _sub(S[k], n, rng); fits.setdefault(f'organ_n{n}', {})[k] = TL.fit_1d(v)
                if n == max(m for m in NEG_N if len(S[k]) >= m):
                    smp[f'organ__{k}'] = v.astype(np.float16)
    rec['fits'] = fits
    Jo = TL.joint(maps, H, Ch, grid)
    if len(Jo) >= 100:
        Ym = _sub(Jo, N_FIT, rng); rec['form'] = {'organ': TL.model_form(Ym)}
        rec['components'] = {'organ': {'I': TL.fit_components(_sub(Ym[:, 0], N_COMP, rng))}}
    np.savez_compressed(os.path.join(paths.out_dir('tail_study', 'samples'), f'{ds}__{case}.npz'), **smp)
    return rec


TUMOUR_ONLY = ('Breast_Tumor_00',)   # spatial collections without a segmented host organ: tumour interiors only (--tumour-only)


def jobs(n_per, n_neg, tumour_only=False):
    d = EX.read_index()
    if tumour_only:                    # random n_per usable scans per collection (all when n_per = 0)
        d = d[(d.seq != 'adc') & d.ds.isin(TUMOUR_ONLY) & (d.status == 'usable')]
        out = []
        for ds, g in d.groupby('ds'):
            g = g.sample(frac=1.0, random_state=0); out += [('tumour_only', ds, f) for f in (g.file if not n_per else g.file[:n_per])]
        return out
    d = d[(d.seq != 'adc') & d.ds.isin(EX.SPATIAL_SETS) & ~d.ds.isin(TUMOUR_ONLY)]
    out = []
    for ds, g in d[(d.status == 'usable') & d.ts.astype(bool)].groupby('ds'):
        g = g.sample(frac=1.0, random_state=0); out += [('tumour', ds, f) for f in (g.file if not n_per else g.file[:n_per])]
    for ds, g in d[d.status == 'no_tumour'].groupby('ds'):
        if ds not in EXPERT_HOST:
            g = g[g.ts.astype(bool)]
        out += [('negative', ds, f) for f in g.sample(frac=1.0, random_state=0).file[:n_neg]]
    return sorted(out, key=lambda t: zlib.crc32((t[1] + t[2]).encode()))       # sites interleaved: partial runs cover every site


# ------------------------------------------------------------------------------------------------------------------ summary
def _records(O):
    rs = []
    for fn in sorted(glob.glob(os.path.join(O, 'records*.jsonl'))):
        rs += [json.loads(l) for l in open(fn) if l.strip()]
    out, seen = [], set()
    for r in rs:                       # a scan done by two overlapping runs appears twice (identical: seeded); keep the first
        if 'fits' in r and (r['ds'], r['file']) not in seen:
            seen.add((r['ds'], r['file'])); out.append(r)
    return out


def tidy(rs):
    cov = ('kind', 'ds', 'file', 'mod', 'site', 'host_src', 'ring_mm', 'tumour_ml', 'n_tumours', 'noise', 'sp_inplane', 'thick_mm',
           'strided')
    fr, cr, mr = [], [], []
    for r in rs:
        base = {k: r.get(k) for k in cov}
        for reg, d in r['fits'].items():
            for st, v in d.items():
                fr.append(dict(base, region=reg, stat=st, order=TL.ORDER[st],
                               scale=(float(st.split('_s')[1]) if '_s' in st else 0.0), **v))
        for reg, d in r.get('components', {}).items():
            for feat, v in d.items():
                for j, c in enumerate(v['components']):
                    cr.append(dict(base, region=reg, feature=feat, K=v['K'], n_fit=v['n'], comp=j, **c))
        for reg, v in r.get('form', {}).items():
            mr.append(dict(base, region=reg, **{k: x for k, x in v.items() if k != 'corr'},
                           **{f'corr_{a}_{b}': v['corr'][i][j] for i, a in enumerate(TL.JOINT) for j, b in enumerate(TL.JOINT) if i < j}))
    return pd.DataFrame(fr), pd.DataFrame(cr), pd.DataFrame(mr)


def _boot_site_equal(d, col, B=2000, seed=0):
    """site-equal median of per-site medians and its stratified bootstrap 95% CI (scans resampled within site); sign-flip
    permutation p (H0: symmetric about 0) with the same site-equal statistic"""
    rng = np.random.default_rng(seed); groups = [g[col].to_numpy(float) for _, g in d.groupby('site')]
    stat = lambda gs: float(np.mean([np.median(g) for g in gs]))
    obs = stat(groups)
    bs = [stat([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(B)]
    perm = [stat([g * rng.choice([-1.0, 1.0], len(g)) for g in groups]) for _ in range(B)]
    p = (1 + sum(abs(x) >= abs(obs) for x in perm)) / (B + 1)
    return obs, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), p


def _paired_row(v, B=2000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    nz = v[v != 0]
    if len(nz) >= 5:
        w = SS.wilcoxon(nz); rk = SS.rankdata(np.abs(nz)); rb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
        p = float(w.pvalue)
    else:
        p, rb = np.nan, np.nan
    return dict(n=len(v), test='wilcoxon', median=float(np.median(v)), ci_lo=float(np.quantile(bs, 0.025)), ci_hi=float(np.quantile(bs, 0.975)),
                frac_pos=float((v > 0).mean()), p=p, rank_biserial=rb)


def paired(fr):
    """A - B per scan and statistic for (tumour, ring) and (far, ring): log nu (positive = A lighter-tailed) and lower-bound gain of
    the Student-t per voxel (positive = A more heavy-tailed); per site and pooled with sites equal"""
    rows = []
    t = fr[fr.kind == 'tumour_scan'].copy(); t['log_nu'] = np.log(t.nu); t['gain_vox'] = t.dElbo / t.n
    for a, b in (('tumour', 'ring'), ('far', 'ring'), ('tumour', 'far')):
        A = t[(t.region == a) & t.n_matched.astype(bool)]; Bb = t[(t.region == b) & t.n_matched.astype(bool)]
        m = A.merge(Bb, on=['ds', 'file', 'stat'], suffixes=('_a', '_b'))
        if not len(m):
            continue
        m['site'] = m.site_a
        for met in ('log_nu', 'gain_vox'):
            m['d'] = m[f'{met}_a'] - m[f'{met}_b']
            for st, g in m.groupby('stat'):
                for site, gs in g.groupby('site'):
                    rows.append(dict(contrast=f'{a}-{b}', metric=met, stat=st, site=site, **_paired_row(gs.d)))
                obs, lo, hi, p = _boot_site_equal(g, 'd')
                rows.append(dict(contrast=f'{a}-{b}', metric=met, stat=st, site='ALL (sites equal)', n=len(g), test='sign-flip',
                                 median=obs, ci_lo=lo,
                                 ci_hi=hi, frac_pos=float(np.mean([(gs.d > 0).mean() for _, gs in g.groupby('site')])),
                                 p=p, rank_biserial=np.nan))
    return pd.DataFrame(rows)


def negatives(fr):
    """tumour-free organs vs rings of the same collection, at matched n (each negative's fit at the n nearest the rings' median n)"""
    rows = []
    neg = fr[fr.kind == 'negative_scan']; ring = fr[(fr.kind == 'tumour_scan') & (fr.region == 'ring')]
    for (ds, st), g in neg.groupby(['ds', 'stat']):
        rg = ring[(ring.ds == ds) & (ring.stat == st)]
        if len(rg) < 5:
            continue
        nr = float(np.median(rg.n)); pick = []
        for _, gg in g.groupby('file'):
            pick.append(gg.iloc[int(np.argmin(np.abs(np.log(gg.n.to_numpy(float)) - np.log(nr))))])
        pk = pd.DataFrame(pick)
        u = SS.mannwhitneyu(np.log(pk.nu), np.log(rg.nu))
        rows.append(dict(ds=ds, stat=st, n_neg=len(pk), n_ring=len(rg), ring_median_n=nr, neg_median_n=float(np.median(pk.n)),
                         nu_neg_median=float(np.median(pk.nu)), nu_ring_median=float(np.median(rg.nu)),
                         gain_neg=float(np.median(pk.dElbo / pk.n)), gain_ring=float(np.median(rg.dElbo / rg.n)),
                         mannwhitney_p=float(u.pvalue)))
    return pd.DataFrame(rows)


def confounds(fr, B=500):
    """Paired design: tumour and ring share the scan (noise, spacing, thickness) and n, so scan-level covariates cannot create the
    tumour-ring contrast; they can only MODIFY it. Per statistic: d = log nu(tumour) - log nu(ring) ~ site fixed effects + z(log noise)
    + z(log in-plane spacing) + z(log thickness) + z(log n) (z: standardised within site), weighted sites equal. Reported: the
    site-equal mean contrast at average covariates (intercept) and each slope (change of the contrast per within-site SD), with
    stratified bootstrap CIs. Plus per-site Spearman correlations of each region's log nu with the covariates (does the tail track
    noise or resolution?)."""
    rows = []; rng = np.random.default_rng(0); cvs = ('noise', 'sp_inplane', 'thick_mm', 'n')
    t = fr[(fr.kind == 'tumour_scan') & fr.region.isin(['tumour', 'ring']) & fr.n_matched.astype(bool)].dropna(subset=['noise'])
    t = t[t.noise > 0]
    m = t[t.region == 'tumour'].merge(t[t.region == 'ring'][['ds', 'file', 'stat', 'nu']], on=['ds', 'file', 'stat'], suffixes=('', '_r'))
    m['d'] = np.log(m.nu) - np.log(m.nu_r)
    for st, g in m.groupby('stat'):
        g = g.reset_index(drop=True); sites = sorted(g.site.unique())
        if len(g) < 10:
            continue
        Z = []
        for cv in cvs:
            x = np.log(g[cv].to_numpy(float)); z = np.zeros(len(g))
            for s_ in sites:
                k = (g.site == s_).to_numpy(); sd = x[k].std()
                z[k] = (x[k] - x[k].mean()) / sd if sd > 1e-9 else 0.0
            Z.append(z)
        Xm = np.stack([(g.site == s_).to_numpy(float) for s_ in sites] + Z, 1); y = g.d.to_numpy(float)

        def coefs(ix):
            w = FT.site_weights(g.site.iloc[ix].tolist()); sw = np.sqrt(w)[:, None]
            b = np.linalg.lstsq(Xm[ix] * sw, y[ix] * sw[:, 0], rcond=None)[0]
            return np.r_[b[:len(sites)].mean(), b[len(sites):]]
        c0 = coefs(np.arange(len(g)))
        by = {s_: np.flatnonzero((g.site == s_).to_numpy()) for s_ in sites}
        bs = np.array([coefs(np.concatenate([v[rng.integers(0, len(v), len(v))] for v in by.values()])) for _ in range(B)])
        for j, nm in enumerate(('contrast at mean covariates',) + tuple(f'slope per SD log {c}' for c in cvs)):
            rows.append(dict(stat=st, site='ALL (sites equal)', test=nm, value=float(c0[j]), ci_lo=float(np.quantile(bs[:, j], 0.025)),
                             ci_hi=float(np.quantile(bs[:, j], 0.975)), p=np.nan, n=len(g)))
    for (st, reg, s_), gs in t.groupby(['stat', 'region', 'site']):
        if len(gs) < 8:
            continue
        for cv in ('noise', 'sp_inplane', 'thick_mm') + (('tumour_ml',) if reg == 'tumour' else ()):
            if gs[cv].nunique() > 2:
                r = SS.spearmanr(gs[cv], np.log(gs.nu))
                rows.append(dict(stat=st, site=s_, test=f'spearman {reg} log nu ~ {cv}', value=float(r.statistic), ci_lo=np.nan,
                                 ci_hi=np.nan, p=float(r.pvalue), n=len(gs)))
    return pd.DataFrame(rows)


def form_stats(mr):
    rows = []
    mr = mr.copy(); mr['pref_vox'] = mr.pref_block / mr.n
    for (kind, reg), g in mr.groupby(['kind', 'region']):
        for site, gs in g.groupby('site'):
            rows.append(dict(kind=kind, region=reg, site=site, **_paired_row(gs.pref_vox),
                             nu_shared=float(np.median(gs.nu_shared)), nu_I=float(np.median(gs.nu_I)),
                             nu_D1=float(np.median(gs.nu_D1)), nu_LoG=float(np.median(gs.nu_LoG)),
                             corr_gain_vox=float(np.median(gs.corr_gain / gs.n))))
        if g.site.nunique() > 1:
            obs, lo, hi, p = _boot_site_equal(g, 'pref_vox')
            rows.append(dict(kind=kind, region=reg, site='ALL (sites equal)', n=len(g), test='sign-flip', median=obs, ci_lo=lo, ci_hi=hi,
                             frac_pos=float(np.mean([(gs.pref_vox > 0).mean() for _, gs in g.groupby('site')])), p=p))
    return pd.DataFrame(rows)


def comp_stats(cr):
    rows = []
    for (kind, site, reg, feat), g in cr.groupby(['kind', 'site', 'region', 'feature']):
        per = g.drop_duplicates(['ds', 'file'])
        kd = per.K.value_counts(normalize=True)
        big = g[(g.rho >= 0.05) & (g.n >= 100)]
        w = 1.0 / big.groupby(['ds', 'file']).comp.transform('count') if len(big) else None
        rows.append(dict(kind=kind, site=site, region=reg, feature=feat, n_scans=len(per),
                         **{f'K{k}': round(float(kd.get(k, 0.0)), 3) for k in range(1, 5)},
                         nu_comp_median=(FT.wquantile(big.nu, w, 0.5) if len(big) else np.nan),
                         nu_comp_q25=(FT.wquantile(big.nu, w, 0.25) if len(big) else np.nan),
                         nu_comp_q75=(FT.wquantile(big.nu, w, 0.75) if len(big) else np.nan)))
    return pd.DataFrame(rows)


def summary(O, min_n=200):
    rs = _records(O)
    fr, cr, mr = tidy(rs)
    for nm, d in (('fits', fr), ('components', cr), ('form', mr)):
        d.to_csv(os.path.join(O, f'{nm}.csv'), index=False)                 # all fits (re-plotting); statistics use n >= min_n
    if len(fr):
        fr = fr[fr.n >= min_n]
    if len(cr):
        cr = cr[cr.n_fit >= min_n]
    if len(mr):
        mr = mr[mr.n >= min_n]
    out = {}
    out['paired'] = paired(fr) if len(fr) else pd.DataFrame()
    out['negatives'] = negatives(fr) if len(fr) else pd.DataFrame()
    out['confounds'] = confounds(fr) if len(fr) else pd.DataFrame()
    out['form'] = form_stats(mr) if len(mr) else pd.DataFrame()
    out['components'] = comp_stats(cr) if len(cr) else pd.DataFrame()
    for nm, d in out.items():
        d.to_csv(os.path.join(O, f'stats_{nm}.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 500)
    nt = sum(r['kind'] == 'tumour_scan' for r in rs); nn = sum(r['kind'] == 'negative_scan' for r in rs)
    lines = [f'tail study: {nt} tumour scans, {nn} tumour-free scans, {fr.site.nunique() if len(fr) else 0} sites; statistics on fits '
             f'with n >= {min_n} voxels (3 mm grid)', '']
    if len(out['paired']):
        P = out['paired']
        lines += ['== tumour - ring, log nu (positive: tumour LIGHTER-tailed), sites equal ==',
                  P[(P.contrast == 'tumour-ring') & (P.metric == 'log_nu') & (P.site == 'ALL (sites equal)')]
                  .drop(columns=['contrast', 'metric', 'site']).round(4).to_string(index=False), '',
                  '== tumour - ring, log nu, per-site medians (full rows with CI, frac>0, Wilcoxon p in stats_paired.csv) ==',
                  P[(P.contrast == 'tumour-ring') & (P.metric == 'log_nu') & (P.site != 'ALL (sites equal)')]
                  .pivot(index='site', columns='stat', values='median').round(2).to_string(), '',
                  '== far - ring, log nu (peritumoral effect: positive = ring heavier-tailed than far tissue), sites equal ==',
                  P[(P.contrast == 'far-ring') & (P.metric == 'log_nu') & (P.site == 'ALL (sites equal)')]
                  .drop(columns=['contrast', 'metric', 'site']).round(4).to_string(index=False), '']
    if len(out['negatives']):
        lines += ['== tumour-free organs vs rings (matched n) ==', out['negatives'].round(4).to_string(index=False), '']
    if len(out['confounds']):
        C = out['confounds']
        lines += ['== covariates: tumour-ring contrast of log nu ~ site + noise + spacing + thickness + n (sites equal) ==',
                  C[C.site == 'ALL (sites equal)'].drop(columns=['site', 'p']).round(4).to_string(index=False), '']
    if len(out['form']):
        lines += ['== model form: order-specific t blocks vs one shared scale (pref per voxel > 0: blocks better) ==',
                  out['form'].round(4).to_string(index=False), '']
    if len(out['components']):
        lines += ['== components (K by lower bound) and component nu ==', out['components'].round(3).to_string(index=False)]
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per', type=int, default=0); ap.add_argument('--n-neg', type=int, default=50)
    ap.add_argument('--budget', type=float, default=0.0); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--pause', type=float, default=0.0, help='rest this fraction of each scan\'s compute time (cooler laptop)')
    ap.add_argument('--min-n', type=int, default=200)
    ap.add_argument('--tumour-only', action='store_true', help='collections without a host organ (breast): tumour interiors only')
    a = ap.parse_args()
    O = paths.out_dir('tail_study')
    if a.summary:
        return summary(O, a.min_n)
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    done = set()
    for fn in glob.glob(os.path.join(O, 'done*.txt')):
        done |= set(open(fn).read().splitlines())
    todo = [t for i, t in enumerate(jobs(a.n_per, a.n_neg, a.tumour_only)) if i % SN == SI]
    t0 = time.time(); left = sum(f'{d}/{f}' not in done for _, d, f in todo)
    print(f'shard {SI}/{SN}: {len(todo)} scans, {left} to do', flush=True)
    for k, (kind, ds, f) in enumerate(todo):
        key = f'{ds}/{f}'
        if key in done:
            continue
        if a.budget and time.time() - t0 > a.budget:
            break
        t1 = time.time()
        try:
            rec = dict(tumour=run_tumour, negative=run_negative, tumour_only=run_tumour_only)[kind](ds, f, key)
            with open(os.path.join(O, f'records{SUF}.jsonl'), 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        except Exception:
            with open(os.path.join(O, f'notes{SUF}.txt'), 'a') as fh:
                fh.write(key + ' ' + traceback.format_exc()[-400:].replace('\n', ' | ') + '\n')
        with open(os.path.join(O, f'done{SUF}.txt'), 'a') as fh:
            fh.write(key + '\n')
        print(f'{k + 1}/{len(todo)} {kind} {key} {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)', flush=True)
        if a.pause > 0:
            time.sleep(a.pause * (time.time() - t1))


if __name__ == '__main__':
    main()
