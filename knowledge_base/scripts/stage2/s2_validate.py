"""Stage 2, step 5: held-out validation on R2Seg (leave one collection out). Expert data only.

For each fold (collection D): fit on the other collections (s2_fit.py --exclude D), then on up to --n-val cases of D: for every
simulated anchor (partial, sub, cut2d) and anatomy scenario (excl, lost) build the stage-1-like lesion (extents, w0, q_v, chosen
extent, host posterior over the expert organs with the stage-1 host model, credible set at 0.05) and run stage 2 (run.solve_cluster).
Metrics against the expert tumour (largest component G; all tumour T for recall of other components):
  methods  stage2 (refined tumour), xi (chosen extent), q_half (q_v >= 1/2), anchor, U (union of candidate extents)
  per method: Dice, precision, recall, over-segmented fraction (FP / predicted), under-segmented fraction (FN / |G|)
  calibration of p_v in the region (10 bins, against T); log BF of the true anchor; new candidates that hit other tumour components.
Evaluation only: a false anchor (a ball of the anchor's volume in normal host tissue >= 20 mm from any tumour) gives the log BF
distribution under 'not tumour'. It is never used to fit anything.
Outputs (clinical/PanCIA_outputs/stage2/validation/): folds/fit_<D>.json, rows.csv, calibration.csv, summary.csv, report.txt
Usage: python3 s2_validate.py [--folds Kidney_Tumor_00,...] [--n-val 20] [--skip-fit]"""
import os, sys, json, time, zlib, argparse, subprocess, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import grounding_g1 as G1, grounding_g2 as G2, host_v81 as HV   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, run as RN   # noqa: E402

VAL_SCENS = ('excl', 'lost')


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


def run_case(c, fit, beta, psi, rng, rows, calib, fold):
    sp = c['sp']; shape = c['G'].shape
    body = F.region_body(c['X'], c['mod'], c['ax']); bodyP = RN.Patch.from_full(body)
    img = lambda W: c['X'][W]
    cores = EX.make_cores(c, rng); outl = EX.make_outlines(c, rng)
    if not cores:
        return
    for cn, a in cores.items():
        ext, w0 = EX.extents(a, outl, c['others'], sp, G1, G2, psi)
        U = np.zeros(shape, bool)
        for _, m in ext: U |= m
        for sc in VAL_SCENS:
            theta, kb, seg, depth = EX.territories(c, sc)
            PhL, S, PhL_all = host_posterior(a, theta, kb, sp, beta)
            hmap = max(PhL, key=PhL.get) if PhL else None
            les, q, xi = lesion_spec(a, ext, w0, PhL, S, EX.organ_cls(hmap) if hmap else 'solid')
            organs = {k: dict(theta=RN.Patch.from_full(theta[k]), kb=kb[k], app=EX.organ_key(k), cls=EX.organ_cls(k)) for k in theta}
            t0 = time.time()
            res = RN.solve_cluster(img, [les], organs, fit, sp, c['mod'], c['thick'], c['ax'], bodyP, shape)
            W = res['W']; ROI = res['ROI']
            Gw, Tw = c['G'][W], c['T'][W]
            preds = dict(stage2=res['ZL'][0], xi=xi[W], q_half=q[W] >= 0.5, anchor=a[W], U=U[W])
            L0 = res['lesions'][0]
            base = dict(fold=fold, ds=c['ds'], case=c['case'], mod=c['mod'], host=c['hs'], cls=c['cls'], core=cn, scen=sc,
                        host_in_S=c['hs'] in S, S=json.dumps(S), log_bf=L0.get('log_bf'), flag_bf=L0.get('flag_bf'),
                        n_grow=res['n_grow'], iters=res['model']['iters'], converged=res['model']['converged'],
                        k_eff=res['model']['k_eff'], n_new=len(res['new_candidates']),
                        G_ml=float(c['G'].sum() * np.prod(sp) / 1000), sec=round(time.time() - t0, 1))
            # new candidates that overlap other expert tumour components (T but not G)
            other = Tw & ~Gw; hits = 0
            for j in np.unique(res['lab'][res['Z'] & ~res['ZL'][0]]):
                if j and (other & (res['lab'] == j)).any(): hits += 1
            base['new_hits_tumour'] = hits
            for nm, P in preds.items():
                rows.append(dict(base, method=nm, **metrics(P, Gw)))
            p = res['p'][ROI]; t = Tw[ROI]
            hb = np.clip((p * 10).astype(int), 0, 9)
            for b in range(10):
                s = hb == b
                if s.any():
                    calib.append(dict(fold=fold, mod=c['mod'], bin=b, n=int(s.sum()), sum_p=float(p[s].sum()), sum_t=float(t[s].sum())))
        # false anchor (evaluation only): ball of the anchor's volume in normal host tissue >= 20 mm from any tumour
        host = c['org'][c['hs']]; dT = F.edt(c['T'], sp); cand = host & (dT >= 20.0)
        if cand.any():
            dh = F.inner_depth(host, sp); ix = np.argmax(np.where(cand, dh, -1).ravel()); ctr = np.unravel_index(ix, shape)
            rad = (3 * a.sum() * np.prod(sp) / (4 * np.pi)) ** (1 / 3)
            pt = np.zeros(shape, bool); pt[ctr] = True; fa = F.dil(pt, sp, rad) & host
            if fa.sum() >= 20:
                ext_f = [('anchor', fa), ('dil4', F.dil(fa, sp, 4.0))]
                w0f = G2.extent_prior([dict(logratio=float(np.log(m.sum() / fa.sum())), flags=[]) for _, m in ext_f], psi, {})
                theta, kb, seg, depth = EX.territories(c, 'excl')
                PhL, S, _ = host_posterior(fa, theta, kb, sp, beta)
                les, _, _ = lesion_spec(fa, ext_f, np.asarray(w0f), PhL, S, c['cls'])
                organs = {k: dict(theta=RN.Patch.from_full(theta[k]), kb=kb[k], app=EX.organ_key(k), cls=EX.organ_cls(k)) for k in theta}
                res = RN.solve_cluster(img, [les], organs, fit, sp, c['mod'], c['thick'], c['ax'], bodyP, shape, max_grow=0)
                L0 = res['lesions'][0]
                rows.append(dict(fold=fold, ds=c['ds'], case=c['case'], mod=c['mod'], host=c['hs'], cls=c['cls'], core=cn, scen='false_anchor',
                                 method='false_anchor', log_bf=L0.get('log_bf'), flag_bf=L0.get('flag_bf'),
                                 vol_Z=L0['vol_Z'], vol_anchor=L0['vol_anchor'], p_anchor_ge_half=L0['p_anchor_ge_half']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--folds', default=','.join(EX.SETS)); ap.add_argument('--n-val', type=int, default=20)
    ap.add_argument('--skip-fit', action='store_true')
    a = ap.parse_args()
    O = paths.out_dir('validation'); FD = paths.out_dir('validation', 'folds')
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(fit6['psi']); beta = fit6['beta_UD']
    done = set()
    fr = os.path.join(O, 'rows.csv')
    if os.path.exists(fr):
        d0 = pd.read_csv(fr); done = set(zip(d0.ds, d0.case.astype(str)))
    for D in a.folds.split(','):
        ff = os.path.join(FD, f'fit_{D}.json')
        if not (a.skip_fit and os.path.exists(ff)) and not os.path.exists(ff):
            subprocess.run([sys.executable, os.path.join(HERE, 's2_fit.py'), '--exclude', D, '--out', ff], check=True)
        fit = json.load(open(ff))
        cases = [t for t in EX.case_list(paths.root('R2Seg')) if t[0] == D]
        n = 0
        for ds, lf in cases:
            if n >= a.n_val:
                break
            case = os.path.basename(lf).replace('_seg.nii.gz', '')
            if (ds, case) in done:
                n += 1; continue
            rows, calib = [], []
            try:
                c = EX.load_case(ds, lf)
                if c is None:
                    continue
                rng = np.random.default_rng(zlib.crc32((ds + c['case'] + 's2val').encode()))
                run_case(c, fit, beta, psi, rng, rows, calib, D)
            except Exception:
                rows.append(dict(fold=D, ds=ds, case=case, method='error', note=traceback.format_exc()[-400:].replace('\n', ' | ')))
            n += 1
            for nm, rr in (('rows.csv', rows), ('calibration.csv', calib)):
                f = os.path.join(O, nm)
                if rr:
                    df = pd.DataFrame(rr)
                    if os.path.exists(f):
                        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
                    df.to_csv(f, index=False)
            print(f'{D} {n}/{a.n_val} {case}', flush=True)
    summarise(O)


def summarise(O):
    d = pd.read_csv(os.path.join(O, 'rows.csv'))
    m = d[d.method.isin(['stage2', 'xi', 'q_half', 'anchor', 'U'])]
    s = m.groupby(['mod', 'cls', 'method'])[['dice', 'precision', 'recall', 'over', 'under']].mean().round(3)
    s['n'] = m.groupby(['mod', 'cls', 'method']).size()
    s.to_csv(os.path.join(O, 'summary.csv'))
    rep = ['held-out (leave one collection out) means per modality / host class / method:', s.to_string(), '']
    # go / no-go: paired difference stage2 - xi in Dice per modality and class
    pv = m.pivot_table(index=['ds', 'case', 'core', 'scen', 'mod', 'cls'], columns='method', values='dice').dropna(subset=['stage2', 'xi'])
    for (mod, cls), g in pv.groupby(level=['mod', 'cls']):
        dd = g['stage2'] - g['xi']
        rep.append(f'{mod} {cls}: mean Dice stage2 {g["stage2"].mean():.3f} vs xi {g["xi"].mean():.3f}; paired diff {dd.mean():+.3f} '
                   f'(stage2 better in {(dd > 0).mean():.0%} of {len(dd)}) -> {"GO" if dd.mean() > 0 else "NO-GO (keep stage-1 extent)"}')
    tb = d[(d.method == 'stage2')].log_bf.dropna(); fb = d[d.method == 'false_anchor'].log_bf.dropna()
    if len(tb) and len(fb):
        rep.append(f'log BF true anchors: median {tb.median():.1f}, flagged (<0) {(tb < 0).mean():.1%} of {len(tb)}')
        rep.append(f'log BF false anchors: median {fb.median():.1f}, flagged (<0) {(fb < 0).mean():.1%} of {len(fb)}')
    fc = os.path.join(O, 'calibration.csv')
    if os.path.exists(fc):
        c = pd.read_csv(fc).groupby(['mod', 'bin'])[['n', 'sum_p', 'sum_t']].sum()
        c['mean_p'] = c.sum_p / c.n; c['freq'] = c.sum_t / c.n
        rep += ['calibration of p_v in the region (mean p -> tumour frequency):', c[['n', 'mean_p', 'freq']].round(3).to_string()]
    errs = d[d.method == 'error']
    if len(errs):
        rep.append(f'{len(errs)} cases failed (see rows.csv, method == error)')
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
