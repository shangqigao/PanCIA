"""Stage 2, step 5: held-out validation on R2Seg (leave one tumour site out). Expert data only.

For each fold (tumour site D, e.g. lung = Lung_Tumor_00 + Lung_Tumor_01): fit on the other sites (s2_fit.py --exclude), then on up to --n-val cases of D: every
expert tumour >= 0.5 ml is a simulated lesion (expert.simulate_lesions: anchors, agent claims as random subsets of over- and under-
segmenting outlines); per core type (partial, sub, cut2d) and anatomy scenario (ts: TotalSegmentator territories, all collections;
excl: expert organs, 5 collections) the lesions form one cluster, each built as stage 1 would (extents, w0, q_v, chosen
extent, host posterior over the expert organs with the stage-1 host model, credible set at 0.05) and run stage 2 (run.solve_cluster).
Metrics against the expert tumour (largest component G; all tumour T for recall of other components):
  methods  stage2 (refined tumour), xi (chosen extent), q_half (q_v >= 1/2), anchor, U (union of candidate extents)
  per method: Dice, precision, recall, over-segmented fraction (FP / predicted), under-segmented fraction (FN / |G|)
  calibration of p_v in the region (10 bins, against T); log BF of the true anchor; new candidates that hit tumour specks;
  recall beyond the union of each lesion's candidate extents, by distance band (0-5, 5-10, 10-20, > 20 mm).
Evaluation only (never used to fit anything), log BF under 'not tumour':
  false_anchor      in every positive case, a ball of the anchor's volume in normal host tissue >= 20 mm from any tumour (TS host
                    when segmented, as on TCGA; else the expert host);
  false_anchor_neg  in up to --n-val tumour-free scans of the held-out site (usability index: no_tumour, e.g. FLARE23 liver /
                    pancreas scans, PI-RADS-negative prostates), a ball in the TS host, its volume drawn from the fold's partial cores.
Fit of each fold: TS anatomy only (as the full fit); the 'excl' scenario (expert organs) is a validation comparison only.
Outputs (clinical/PanCIA_outputs/stage2/validation/): folds/fit_<site>.json, rows[_s<i>].csv, calibration[_s<i>].csv, summary.csv, report.txt
Folds are tumour sites (expert.SITE_ENTITY; the two lung collections form one fold). SHARD=i/n runs every n-th fold.
Usage: [SHARD=i/n] python3 s2_validate.py [--folds kidney,lung,...] [--n-val 20]; python3 s2_validate.py --summary"""
import os, sys, json, time, zlib, argparse, subprocess, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import KnowledgeBase, grounding_g1 as G1, grounding_g2 as G2, host_v81 as HV   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, run as RN   # noqa: E402

VAL_SCENS = ('ts', 'excl')      # ts: TotalSegmentator territories (option A, every collection); excl: expert organs (5 collections)
HOST_CATS = ('organ', 'hollow_organ', 'gland')   # host candidates in validation (stage 1 never takes bones, muscles, vessels as hosts)
KB = None


def _ent(k):
    """KB entity of a territory key: expert names ('kidney_1'), TS keys (('host', ent), (ent, side), ('ts:name', None))"""
    if isinstance(k, tuple):
        e = k[1] if k[0] == 'host' else k[0]
        return str(e).split(':')[-1]
    return EX.organ_key(k)


def organ_specs(theta, kbf):
    from pancia_kb.stage2.interface import _app_key
    out = {}
    for k in theta:
        e = _ent(k)
        out[k] = dict(theta=RN.Patch.from_full(theta[k]), kb=kbf[k], app=_app_key(e), cls=G1.organ_class(KB, e))
    return out


def host_candidates(theta, hkey):
    """organ-like territories (KB category organ / hollow organ / gland) and the true host"""
    keep = {}
    for k in theta:
        e = _ent(k)
        if k == hkey or (KB.entities.get(e, {}).get('category') in HOST_CATS) or (not isinstance(k, tuple)):
            keep[k] = theta[k]
    return keep


def metrics(P, G):
    tp = float((P & G).sum()); fp = float((P & ~G).sum()); fn = float((~P & G).sum())
    return dict(dice=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                over=fp / max(tp + fp, 1), under=fn / max(tp + fn, 1))


def host_posterior(a, theta, kb, sp, beta):
    da = F.edt(a, sp)
    cands = {}
    for k, th in theta.items():
        if not th.any():
            continue
        core = None if kb[k] else th
        phi = G1.compat_features(a, core, th, 0.5 if kb[k] else 1.0, sp, kb=kb[k])
        d = float(min(HV.D_CAP, da[th].min())); phi['dist_mm'] = d; phi['ldist'] = float(np.log1p(d))
        cands[k] = phi
    PhL = HV.host_posterior(cands, beta)
    S = G2.credible_set(PhL, 0.05) if PhL else []
    tot = sum(PhL[h] for h in S) or 1.0
    return {h: PhL[h] / tot for h in S}, S, PhL


def lesion_spec(a, ext, w0, PhL, S, cls):
    q = EX.q_map(ext, w0, a.shape); xi = ext[int(np.argmax(w0))][1]
    return dict(key='L0', anchors=[RN.Patch.from_full(a)], q=RN.Patch.from_full(q), xi=RN.Patch.from_full(xi), PhL=PhL, S=S,
                tum_cls=cls), q, xi


BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf))      # mm beyond the union of a lesion's candidate extents


def run_case(c, fit, beta, psi, rng, rows, calib, fold):
    """one expert scan: every tumour >= 0.5 ml is a simulated lesion (expert.simulate_lesions, as the fit); per core type the lesions
    form one stage-1-like cluster and are inferred jointly (run.solve_cluster), per anatomy scenario. One row per lesion and method;
    recall beyond the lesion's candidate extents (tumour no simulated agent claimed) is counted per distance band."""
    sp = c['sp']; shape = c['T'].shape; vml = float(np.prod(sp)) / 1000.0
    body = F.region_body(c['X'], c['mod'], c['ax']); bodyP = RN.Patch.from_full(body)
    img = lambda W: c['X'][W]
    sim = EX.simulate_lesions(c, rng, psi, G1, G2)
    if not sim:
        return
    allc = np.zeros(shape, bool)
    for Gj in c['comps']:
        allc |= Gj
    for cn, les_sim in sim.items():
        A_full = [EX.full(L['a'], shape) for L in les_sim]
        for sc in [x for x in VAL_SCENS if x in c['scens']]:
            theta, kb, seg, depth, hkey = EX.territories(c, sc)
            cand = host_candidates(theta, hkey)
            lesions = []
            for k, L in enumerate(les_sim):
                PhL, S, _ = host_posterior(A_full[k], cand, {h: kb[h] for h in cand}, sp, beta)
                lesions.append(dict(key=f'L{k}', anchors=[RN.Patch(*L['a'])], q=RN.Patch(*L['q']), xi=RN.Patch(*L['xi']), PhL=PhL,
                                    S=S, tum_cls=c['cls']))
            organs = organ_specs(theta, kb)
            t0 = time.time()
            res = RN.solve_cluster(img, lesions, organs, fit, sp, c['mod'], c['thick'], c['ax'], bodyP, shape, seq=c['seq'])
            W = res['W']; ROI = res['ROI']; Tw = c['T'][W]
            common = dict(fold=fold, ds=c['ds'], case=c['case'], mod=c['mod'], seq=c['seq'], host=_ent(hkey), cls=c['cls'], core=cn,
                          scen=sc, host_kb=bool(kb[hkey]), n_lesions=len(les_sim), n_grow=res['n_grow'], iters=res['model']['iters'],
                          converged=res['model']['converged'], k_eff=res['model']['k_eff'], n_new=len(res['new_candidates']),
                          sec=round(time.time() - t0, 1))
            other = Tw & ~allc[W]; hits = 0                          # new candidates on expert tumour specks (< 0.5 ml, no lesion)
            for j in np.unique(res['lab'][res['Z'] & ~np.any(res['ZL'], axis=0)]):
                if j and (other & (res['lab'] == j)).any(): hits += 1
            common['new_hits_tumour'] = hits
            for k, L in enumerate(les_sim):
                Gk = c['comps'][L['j']][W]
                q = EX.full(L['q'], shape, np.float32)[W]; xi = EX.full(L['xi'], shape)[W]; U = EX.full(L['U'], shape)[W]
                L0 = res['lesions'][k]
                base = dict(common, comp=L['j'], claims='+'.join(L['claims']), host_in_S=hkey in lesions[k]['S'],
                            S=json.dumps([str(x) for x in lesions[k]['S']]), log_bf=L0.get('log_bf'), flag_bf=L0.get('flag_bf'),
                            n_vox_bf=L0.get('n_vox_bf'), log_bf_per_vox=L0.get('log_bf_per_vox'),
                            G_ml=float(Gk.sum() * vml), anchor_ml=float(A_full[k].sum() * vml))
                # tumour beyond every candidate extent of this lesion, and how much of it stage 2 recalls, per distance band
                beyond = Gk & ~U
                if beyond.any():
                    dU = F.edt(U, sp) if U.any() else np.full(U.shape, np.inf)
                    for lo, hi in BANDS:
                        m = beyond & (dU > lo) & (dU <= hi); nm = f'{lo:g}_{hi:g}'
                        base[f'miss_{nm}_ml'] = float(m.sum() * vml); base[f'rec_{nm}_ml'] = float((m & res['ZL'][k]).sum() * vml)
                preds = dict(stage2=res['ZL'][k], xi=xi, q_half=q >= 0.5, anchor=A_full[k][W], U=U)
                for nm, P in preds.items():
                    rows.append(dict(base, method=nm, **metrics(P, Gk)))
            p = res['p'][ROI]; t = Tw[ROI]
            hb = np.clip((p * 10).astype(int), 0, 9)
            for b in range(10):
                s_ = hb == b
                if s_.any():
                    calib.append(dict(fold=fold, mod=c['mod'], bin=b, n=int(s_.sum()), sum_p=float(p[s_].sum()), sum_t=float(t[s_].sum())))
        # false anchor (evaluation only): ball of the largest tumour's anchor volume in normal host tissue >= 20 mm from any tumour.
        # Anatomy: TS (as on TCGA) when TS segments the host; else the expert host ('excl') where labelled; else none
        sc_f = 'ts' if not c['ts']['kb'][c['ts']['host']] else ('excl' if 'excl' in c['scens'] else None)
        if sc_f is None:
            continue
        a = A_full[0]
        theta, kb, seg, depth, hkey = EX.territories(c, sc_f)
        host = theta[hkey] & ~c['T']
        dT = F.edt(c['T'], sp); cand = host & (dT >= 20.0)
        if cand.any():
            dh = F.inner_depth(host, sp); ix = np.argmax(np.where(cand, dh, -1).ravel()); ctr = np.unravel_index(ix, shape)
            rad = (3 * a.sum() * np.prod(sp) / (4 * np.pi)) ** (1 / 3)
            pt = np.zeros(shape, bool); pt[ctr] = True; fa = F.dil(pt, sp, rad) & host
            L0 = false_anchor(c, fa, theta, kb, hkey, fit, beta, psi, img, bodyP, shape)
            if L0 is not None:
                rows.append(dict(fold=fold, ds=c['ds'], case=c['case'], mod=c['mod'], seq=c['seq'], host=_ent(hkey), cls=c['cls'], core=cn,
                                 scen='false_anchor', anat=sc_f, method='false_anchor', **L0))


def false_anchor(c, fa, theta, kb, hkey, fit, beta, psi, img, bodyP, shape):
    """stage 2 on a false anchor fa (a ball in normal tissue; extents anchor and anchor + 4 mm with the stage-1 extent prior).
    Returns the Bayes-factor columns, or None when fa is too small. Evaluation only."""
    sp = c['sp']
    if fa.sum() < 20:
        return None
    ext_f = [('anchor', fa), ('dil4', F.dil(fa, sp, 4.0))]
    w0f = G2.extent_prior([dict(logratio=float(np.log(m.sum() / fa.sum())), flags=[]) for _, m in ext_f], psi, {})
    cand_h = host_candidates(theta, hkey)
    PhL, S, _ = host_posterior(fa, cand_h, {k: kb[k] for k in cand_h}, sp, beta)
    les, _, _ = lesion_spec(fa, ext_f, np.asarray(w0f), PhL, S, c['cls'])
    organs = organ_specs(theta, kb)
    res = RN.solve_cluster(img, [les], organs, fit, sp, c['mod'], c['thick'], c['ax'], bodyP, shape, max_grow=0, seq=c['seq'])
    L0 = res['lesions'][0]
    return dict(log_bf=L0.get('log_bf'), flag_bf=L0.get('flag_bf'), n_vox_bf=L0.get('n_vox_bf'), log_bf_per_vox=L0.get('log_bf_per_vox'),
                vol_Z=L0['vol_Z'], vol_anchor=L0['vol_anchor'], p_anchor_ge_half=L0['p_anchor_ge_half'])


def run_negative(c, fit, beta, psi, rng, rows, fold, vols):
    """tumour-free scan (evaluation only): a ball in the TS host, its volume drawn from the anchor volumes (ml) of this fold's
    simulated partial cores, centred at a random host voxel at least its radius deep (else the deepest host voxel)."""
    sp = c['sp']; shape = c['X'].shape
    body = F.region_body(c['X'], c['mod'], c['ax']); bodyP = RN.Patch.from_full(body)
    img = lambda W: c['X'][W]
    theta, kb, seg, depth, hkey = EX.territories(c, 'ts')
    host = theta[hkey]
    ml = float(rng.choice(vols)); rad = (3 * ml * 1000.0 / (4 * np.pi)) ** (1 / 3)
    dh = F.inner_depth(host, sp); cand = np.flatnonzero((host & (dh >= rad)).ravel())
    ix = int(rng.choice(cand)) if len(cand) else int(np.argmax(np.where(host, dh, -1).ravel()))
    pt = np.zeros(shape, bool); pt[np.unravel_index(ix, shape)] = True
    fa = F.dil(pt, sp, rad) & host
    L0 = false_anchor(c, fa, theta, kb, hkey, fit, beta, psi, img, bodyP, shape)
    if L0 is not None:
        rows.append(dict(fold=fold, ds=c['ds'], case=c['case'], mod=c['mod'], seq=c['seq'], host=_ent(hkey), cls=c['cls'], core='ball',
                         scen='false_anchor_neg', anat='ts', method='false_anchor_neg', drawn_ml=round(ml, 3), **L0))


def main():
    sites = list(dict.fromkeys(EX.SITE_ENTITY.values()))
    ap = argparse.ArgumentParser(); ap.add_argument('--folds', default=','.join(sites)); ap.add_argument('--n-val', type=int, default=20)
    ap.add_argument('--skip-fit', action='store_true'); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    O = paths.out_dir('validation'); FD = paths.out_dir('validation', 'folds')
    # SHARD=i/n: this process takes every n-th fold (folds are independent); rows go to rows_s<i>.csv
    si, sn = (int(x) for x in os.environ.get('SHARD', '0/1').split('/')); tag = f'_s{si}' if sn > 1 else ''
    if a.summary:
        summarise(O); return
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(fit6['psi']); beta = fit6['beta_UD']
    global KB
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    done = set()
    for fr in [os.path.join(O, f) for f in os.listdir(O) if f.startswith('rows') and f.endswith('.csv')]:
        d0 = pd.read_csv(fr); done |= set(zip(d0.ds, d0.case.astype(str)))
    # folds are tumour sites (Lung_Tumor_00 and Lung_Tumor_01 are held out together, so no lung case is in the fold's fit)
    for D in a.folds.split(',')[si::sn]:
        held = [ds for ds, e in EX.SITE_ENTITY.items() if e == D] or [D]
        ff = os.path.join(FD, f'fit_{D}.json')
        if not os.path.exists(ff):
            subprocess.run([sys.executable, os.path.join(HERE, 's2_fit.py'), '--exclude', ','.join(held), '--out', ff], check=True)
        fit = json.load(open(ff))
        cases = [t for t in EX.case_list(paths.root('R2Seg')) if t[0] in held]
        cases = [cases[i] for i in np.argsort([zlib.crc32(os.path.basename(lf).encode()) for _, lf in cases], kind='stable')]
        n = 0
        for ds, lf in cases:
            if n >= a.n_val:
                break
            case = os.path.basename(lf).replace('_seg.nii.gz', '')
            if (ds, case) in done:
                n += 1; continue
            rows, calib = [], []
            try:
                c = EX.load_case(ds, lf, kb=KB)
                if c is None:
                    continue
                rng = np.random.default_rng(zlib.crc32((ds + c['case'] + 's2val').encode()))
                run_case(c, fit, beta, psi, rng, rows, calib, D)
            except Exception:
                rows.append(dict(fold=D, ds=ds, case=case, method='error', note=traceback.format_exc()[-400:].replace('\n', ' | ')))
            n += 1
            _append(O, tag, rows, calib)
            print(f'{D} {n}/{a.n_val} {case}', flush=True)
        # negatives (evaluation only): up to --n-val tumour-free scans of the held-out site, false anchor in the TS host
        negs = [t for t in EX.negative_list(paths.root('R2Seg')) if t[0] in held]
        if negs:
            dd = _cat(O, 'rows')
            vols = [] if dd is None or 'anchor_ml' not in dd else \
                dd[(dd.fold == D) & (dd.core == 'partial') & (dd.method == 'stage2')].drop_duplicates(['ds', 'case']).anchor_ml.dropna().tolist()
            n = 0
            for ds, lf in negs if vols else []:
                if n >= a.n_val:
                    break
                case = os.path.basename(lf).replace('_seg.nii.gz', '')
                if (ds, case) in done:
                    n += 1; continue
                rows = []
                try:
                    c = EX.load_negative(ds, lf, KB)
                    if c is None:
                        continue
                    run_negative(c, fit, beta, psi, np.random.default_rng(zlib.crc32((ds + case + 's2neg').encode())), rows, D, vols)
                except Exception:
                    rows.append(dict(fold=D, ds=ds, case=case, method='error', note=traceback.format_exc()[-400:].replace('\n', ' | ')))
                n += 1
                _append(O, tag, rows, [])
                print(f'{D} negative {n}/{a.n_val} {case}', flush=True)
    if sn == 1:
        summarise(O)


def _append(O, tag, rows, calib):
    for nm, rr in ((f'rows{tag}.csv', rows), (f'calibration{tag}.csv', calib)):
        f = os.path.join(O, nm)
        if rr:
            df = pd.DataFrame(rr)
            if os.path.exists(f):
                df = pd.concat([pd.read_csv(f), df], ignore_index=True)
            df.to_csv(f, index=False)


def _cat(O, stem):
    fs = sorted(f for f in os.listdir(O) if f.startswith(stem) and f.endswith('.csv'))
    return pd.concat([pd.read_csv(os.path.join(O, f)) for f in fs], ignore_index=True) if fs else None


def summarise(O):
    d = _cat(O, 'rows')
    m = d[d.method.isin(['stage2', 'xi', 'q_half', 'anchor', 'U'])]
    K = ['scen', 'mod', 'cls', 'method']; V_ = ['dice', 'precision', 'recall', 'over', 'under']
    per_site = m.groupby(K + ['fold'])[V_].mean()                                      # mean within each held-out site
    s = per_site.groupby(level=K).mean().round(3)                                      # then sites count equally
    s['sites'] = per_site.groupby(level=K).size(); s['rows'] = m.groupby(K).size()
    s.to_csv(os.path.join(O, 'summary.csv')); per_site.round(3).to_csv(os.path.join(O, 'summary_per_site.csv'))
    rep = ['held-out (leave one tumour site out), sites count equally (per-site means in summary_per_site.csv); anatomy scenario '
           '(ts = TotalSegmentator territories; excl = expert organs) / modality / host class / method:', s.to_string(), '']
    # go / no-go, site-weighted (user, 25 Oct: sites count equally): paired difference stage2 - xi in Dice per case, averaged within each
    # held-out site, then over sites, per modality and class; GO when the site-weighted mean is > 0. Per-site values are listed.
    m = m.assign(comp=m['comp'].fillna(0) if 'comp' in m else 0)
    pv = m.pivot_table(index=['fold', 'ds', 'case', 'comp', 'core', 'scen', 'mod', 'cls'], columns='method', values='dice').dropna(subset=['stage2', 'xi'])
    pv = pv.xs('ts', level='scen') if 'ts' in pv.index.get_level_values('scen') else pv      # go / no-go on the realistic anatomy
    pv = pv.assign(diff=pv['stage2'] - pv['xi'])
    rep.append('go / no-go (TS anatomy; sites count equally):')
    for (mod, cls), g in pv.groupby(level=['mod', 'cls']):
        ps = g.groupby(level='fold')[['stage2', 'xi', 'diff']].mean()
        sw = ps.mean()
        rep.append(f'{mod} {cls}: site-weighted Dice stage2 {sw["stage2"]:.3f} vs xi {sw["xi"]:.3f}; diff {sw["diff"]:+.3f}; '
                   f'stage2 better in {(ps["diff"] > 0).sum()} of {len(ps)} sites -> {"GO" if sw["diff"] > 0 else "NO-GO (keep stage-1 extent)"}')
        rep += [f'    {f}: stage2 {r["stage2"]:.3f} xi {r["xi"]:.3f} diff {r["diff"]:+.3f} ({int((g.xs(f, level="fold")["diff"]).size)} rows)'
                for f, r in ps.iterrows()]
    # recall beyond the lesion's candidate extents (tumour no simulated agent claimed), per distance band; sites count equally
    st = d[(d.method == 'stage2') & (d.scen == 'ts')]
    bcols = [f'{lo:g}_{hi:g}' for lo, hi in BANDS]
    if len(st) and any(f'miss_{b}_ml' in st for b in bcols):
        rep.append('recall beyond the candidate extents (TS anatomy; recalled / missed volume per site, then sites averaged):')
        for b in bcols:
            if f'miss_{b}_ml' not in st:
                continue
            g = st.groupby('fold')[[f'miss_{b}_ml', f'rec_{b}_ml']].sum()
            g = g[g[f'miss_{b}_ml'] > 0]
            if len(g):
                rr_ = g[f'rec_{b}_ml'] / g[f'miss_{b}_ml']
                rep.append(f'  {b} mm: site-weighted recall {rr_.mean():.1%} over {len(g)} sites (missed volume {g[f"miss_{b}_ml"].sum():.0f} ml); '
                           + '; '.join(f'{k} {v:.0%}' for k, v in rr_.items()))
        tot = st.groupby('fold').apply(lambda x: sum(x[f'miss_{b}_ml'].fillna(0).sum() for b in bcols if f'miss_{b}_ml' in x) / max(x.G_ml.sum(), 1e-9))
        rep.append('  share of tumour volume beyond every candidate extent (simulation): ' + '; '.join(f'{k} {v:.1%}' for k, v in tot.items()))
    # Bayes-factor flags per site and site-weighted (true anchors should not be flagged; false anchors should)
    def flag_table(meth):
        x = d[(d.method == meth)].dropna(subset=['log_bf_per_vox'])
        if not len(x):
            return None
        per = x.groupby('fold').log_bf_per_vox.apply(lambda v: (v < 0).mean())
        return per, x.groupby('fold').size()
    for meth, lab in (('stage2', 'true anchors flagged (should be low)'), ('false_anchor', 'false anchors in tumour scans flagged (should be high)'),
                      ('false_anchor_neg', 'false anchors in tumour-free scans flagged (should be high)')):
        ft = flag_table(meth)
        if ft is not None:
            per, n = ft
            rep.append(f'{lab}: site-weighted {per.mean():.1%} over {len(per)} sites; ' + '; '.join(f'{k} {v:.0%} of {n[k]}' for k, v in per.items()))
    tv = d[(d.method == 'stage2')].log_bf_per_vox.dropna(); fv = d[d.method == 'false_anchor'].log_bf_per_vox.dropna()
    if len(tv) and len(fv):
        rep.append(f'log BF per voxel, true anchors: median {tv.median():.4f} (10-90%: {tv.quantile(0.1):.4f} to {tv.quantile(0.9):.4f})')
        rep.append(f'log BF per voxel, false anchors: median {fv.median():.4f} (10-90%: {fv.quantile(0.1):.4f} to {fv.quantile(0.9):.4f})')
    cc = _cat(O, 'calibration')
    if cc is not None:
        # per held-out site first, then sites averaged with equal weight (bins a site has no voxels in are left out for that site)
        f_ = cc.groupby(['mod', 'bin', 'fold'])[['n', 'sum_p', 'sum_t']].sum()
        f_['mean_p'] = f_.sum_p / f_.n; f_['freq'] = f_.sum_t / f_.n
        c = f_.groupby(level=['mod', 'bin']).agg(mean_p=('mean_p', 'mean'), freq=('freq', 'mean'), n=('n', 'sum'), sites=('n', 'size'))
        rep += ['calibration of p_v in the region, sites count equally (mean p -> tumour frequency):', c.round(3).to_string()]
    errs = d[d.method == 'error']
    if len(errs):
        rep.append(f'{len(errs)} cases failed (see rows*.csv, method == error)')
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
