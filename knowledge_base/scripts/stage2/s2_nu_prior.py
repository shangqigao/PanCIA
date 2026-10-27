"""Stage 2: tail-weight prior of the Student-t classes, keyed by modality x tissue class (main.tex Eq. nu_prior; Appendix C).

Data: the tail study's tissue components on the joint features (I, two in-plane first differences, LoG at 1 mm;
s2_tail_study.py, components.csv): tumour components (tumour interiors; breast from --tumour-only) for the tumour prior, and
components of the normal ring for the normal prior. Kept: region fits with >= MIN_N grid voxels (default 100), components with
weight >= 0.05 and >= 100 voxels. Variable x = log(1/nu), nu in [2, 100] (components at a bound enter at the bound).

Cells: modality (CT, MR; one MR row: the spatial-change features are expected to transfer across sequences) x the host's
knowledge-base tissue_class (parenchymal, hollow, aerated; KB v0.5.9). The host is the collection's site entity
(expert.SITE_ENTITY). Sites count equally (expert.SITE_OF: the two lung collections are one site).

Model (normal hierarchy, moment estimates):  x ~ N(mu_site, sw2);  mu_site ~ N(mu_cell, ts2);  mu_cell ~ N(mu0, tc2).
  sw2   within-site variance of x (components equal within a scan, scans equal within a site; sites equal)
  ts2   between-site variance within a cell (cells with >= 2 sites, pooled)
  tc2   between-cell variance (method of moments, >= 0); mu0 = mean of the cell means (cells equal)
  cell mean shrunk to mu0: mu_k = (n_k c_k / ts2 + mu0 / tc2) / (n_k / ts2 + 1 / tc2)   (n_k sites in cell k)
Prior for a component of a new scan:  cell k:  N(mu_k, v_k + ts2 + sw2);  any other host or modality (empty cell, tissue classes
without expert tumours: neural, osseous, muscle, lymphoid, diffuse, vascular, space):  N(mu0, v0 + tc2 + ts2 + sw2).

Validation: leave one site out (refit without it; its cell may become empty and then uses the pan-cancer prior), mean log
density of the held-out site's components (sites equal) under the cell prior and under the single pan-cancer prior.
Split-half (--split-half): per tumour scan, a one-class shared-scale Student-t on two random halves of the saved joint samples
(tail_study/samples); the half-to-half spread of x estimates the noise of a per-scan estimate, compared with the prior variance
(the share of a scan's estimate that would come from the prior).

Outputs: knowledge_base/knowledge_base/stage2/nu_prior.json (the prior; next to stage2_fit.json) and
clinical/PanCIA_outputs/stage2/tail_study/nu_prior/ (report.txt, cells.csv, loso.csv, split_half.csv).
Usage: python s2_nu_prior.py [--split-half] [--min-n 100]"""
import os, sys, json, glob, time, zlib, argparse
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import numpy as np, pandas as pd   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402

from pancia_kb.stage2.nuprior import NU_RANGE, CELL_CLASSES, MODALITIES, FALLBACK_CLASSES, tissue_class_of, load_components, fit, logpdf, loso   # noqa: E402


def split_half(O, min_n=100, max_n=2000):
    """per tumour scan: shared-scale Student-t on two random halves of the saved joint samples -> x_A, x_B (resumable)"""
    from pancia_kb.stage2 import tails as TL
    out = os.path.join(O, 'nu_prior', 'split_half.csv'); done = set()
    if os.path.exists(out):
        done = set(pd.read_csv(out).key)
    files = sorted(glob.glob(os.path.join(O, 'samples', '*.npz'))); t0 = time.time()
    for k, fn in enumerate(files):
        key = os.path.basename(fn)
        if key in done:
            continue
        z = np.load(fn)
        if 'tumour__joint' not in z.files:
            continue
        Y = z['tumour__joint'].astype(np.float64)
        if len(Y) < 2 * min_n:
            continue
        rng = np.random.default_rng(zlib.crc32(key.encode())); idx = rng.permutation(len(Y))[:min(len(Y), 2 * max_n)]
        h = len(idx) // 2; xs = []
        for part in (idx[:h], idx[h:2 * h]):
            m, _, _ = TL._one(Y[part], True); xs.append(float(np.log(1.0 / np.clip(m.nu_k[0], *NU_RANGE))))
        ds = key.split('__')[0]
        pd.DataFrame([dict(key=key, ds=ds, n_half=h, x_a=xs[0], x_b=xs[1])]).to_csv(out, mode='a', index=False, header=not os.path.exists(out))
        if (k + 1) % 50 == 0:
            print(f'split-half {k + 1}/{len(files)} {time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--min-n', type=int, default=100); ap.add_argument('--split-half', action='store_true')
    a = ap.parse_args()
    O = paths.out_dir('tail_study'); R = paths.out_dir('tail_study', 'nu_prior')
    if a.split_half:
        split_half(O, a.min_n)
    c, ver = load_components(O, a.min_n)
    res, lines = {}, [f'tail-weight prior (x = log 1/nu), KB {ver}, min_n {a.min_n}; components: '
                      + ', '.join(f'{r} {int((c.region == r).sum())}' for r in ('tumour', 'ring')), '']
    cells_all = []
    for reg, name in (('tumour', 'tumour'), ('ring', 'normal')):
        cr = c[c.region == reg]; F = fit(cr); L = loso(cr)
        C = F['cells'].assign(prior=name, nu_at_mean=lambda d: np.exp(-d['mean']))
        cells_all.append(C); L.assign(prior=name).to_csv(os.path.join(R, f'loso_{name}.csv'), index=False)
        res[name] = dict(cells={r.cell: dict(mean_x=round(r['mean'], 4), sd_x=round(r.sd, 4), nu_at_mean=round(float(np.exp(-r['mean'])), 2),
                                             n_sites=int(r.n_sites), sites=r.sites, n_scans=int(r.n_scans), n_components=int(r.n_comp))
                                for _, r in C.iterrows()},
                         fallback=dict(mean_x=round(F['fallback']['mean'], 4), sd_x=round(F['fallback']['sd'], 4),
                                       nu_at_mean=round(float(np.exp(-F['fallback']['mean'])), 2)),
                         variance=dict(within_site=round(F['sw2'], 4), between_site=round(F['ts2'], 4), between_cell=round(F['tc2'], 4)))
        lines += [f'== {name} prior ==',
                  C[['cell', 'sites', 'n_sites', 'n_scans', 'n_comp', 'raw', 'mean', 'sd', 'nu_at_mean']].round(3).to_string(index=False),
                  f"fallback (other cells and tissue classes): mean {F['fallback']['mean']:.3f} (nu {np.exp(-F['fallback']['mean']):.1f}), "
                  f"sd {F['fallback']['sd']:.3f}; variances: within site {F['sw2']:.3f}, between sites {F['ts2']:.3f} ({F['ts2_df']:.0f} df), between cells {F['tc2']:.3f}",
                  '-- leave one site out (mean log density per component, sites equal; higher is better) --',
                  L.round(3).to_string(index=False),
                  f"mean over sites: cell prior {L.lp_cell.mean():.3f}, pan-cancer prior {L.lp_pancancer.mean():.3f}; "
                  f"|site mean - prediction|: cell {np.abs(L.site_mean_x - L.pred_cell).mean():.3f}, pan-cancer {np.abs(L.site_mean_x - L.pred_pancancer).mean():.3f}",
                  '']
    pd.concat(cells_all).to_csv(os.path.join(R, 'cells.csv'), index=False)
    sh = os.path.join(R, 'split_half.csv')
    if os.path.exists(sh):
        d = pd.read_csv(sh); d['site'] = d.ds.map(EX.SITE_OF)
        d['bin'] = pd.cut(2 * d.n_half, [0, 400, 1000, 2000, 4001], right=False)
        ptv = res['tumour']['cells']; pv = np.mean([v['sd_x'] ** 2 for v in ptv.values()])
        # the two halves estimate the same x: E[(x_A - x_B)^2] = 2 var_half, and var_full = var_half / 2 -> var_full = E[(x_A - x_B)^2] / 4
        g = d.groupby('bin', observed=True).apply(lambda q: pd.Series(dict(scans=len(q), noise_var_full_n=float(np.mean((q.x_a - q.x_b) ** 2) / 4.0))), include_groups=False)
        g = g[g.scans >= 10]
        g['prior_var'] = pv; g['prior_share'] = g.noise_var_full_n / (g.noise_var_full_n + pv)
        acoef = float(np.mean((d.x_a - d.x_b) ** 2 / 4.0 * 2 * d.n_half))       # noise variance ~ acoef / n (n voxels in the fit)
        ct = c[c.region == 'tumour']
        comp = ct.groupby('cell').n.median().rename('median_component_voxels').to_frame()
        comp['noise_var'] = acoef / comp.median_component_voxels
        comp['prior_var'] = [ptv[k]['sd_x'] ** 2 if k in ptv else np.nan for k in comp.index]
        comp['prior_share'] = comp.noise_var / (comp.noise_var + comp.prior_var)
        lines += ['== split-half at component size: noise variance taken as acoef / n, acoef = mean of (x_A - x_B)^2 / 4 x n ==',
                  f'acoef {acoef:.2f}', comp.round(4).to_string(), '']
        res['split_half_components'] = {k: dict(median_voxels=float(v.median_component_voxels), prior_share=round(float(v.prior_share), 3))
                                        for k, v in comp.iterrows()}
        lines += ['== split-half: noise of a per-scan estimate vs prior variance (share of the estimate from the prior) ==',
                  'one-class shared-scale Student-t on two random halves; noise variance at the full n = mean (x_A - x_B)^2 / 4; bins with >= 10 scans', g.round(4).to_string(), '']
        res['split_half'] = {str(k): dict(scans=int(v.scans), noise_var=round(v.noise_var_full_n, 4), prior_share=round(v.prior_share, 3)) for k, v in g.iterrows()}
    out = dict(version='nu_prior_v1', date=time.strftime('%Y-%m-%d'), kb_version=ver, variable='x = log(1/nu), nu in [2, 100]',
               key='modality (CT | MR) x host tissue_class (parenchymal | hollow | aerated); other -> fallback',
               fallback_tissue_classes=FALLBACK_CLASSES, min_n=a.min_n, data='tail_study components.csv, joint features, sites equal',
               tumour=res['tumour'], normal=res['normal'], **({'split_half': res['split_half'], 'split_half_components': res['split_half_components']} if 'split_half' in res else {}))
    dst = os.path.join(paths.kb_dir(), 'knowledge_base', 'stage2'); os.makedirs(dst, exist_ok=True)
    json.dump(out, open(os.path.join(dst, 'nu_prior.json'), 'w'), indent=1)
    open(os.path.join(R, 'report.txt'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
