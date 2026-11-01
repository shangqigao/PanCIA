"""Held-out calibration of the stage-2 spatial prior on REAL agent outlines (29 Oct). Expert truth for evaluation only.

For every tumour site D: the held-out samples of s2_agent_build.py (collections of D only) are scored with the fold fits that left D
out, for each prior to compare (default: the simulated-claim fit validation/folds/fit_<D>.json and the real-agent fit
validation_agents/folds_agent/fit_<D>.json). The cluster prior is pi = 1 - prod_L (1 - sigma(eta . psi(L))), as run.py.
Zones (from the sampled voxel's cluster quantities):
  anchor        q >= 0.999 (inside an agreement anchor)
  extent        inside the chosen extent (sd <= 0), outside the anchor
  outlines      inside a candidate outline but outside the chosen extent (0 < q, sd > 0)
  outside       outside every outline (q = 0), by distance to the chosen extent (0-5, 5-10, 10-20, 20-60 mm) and direction:
                through-plane when the offset along the acquisition axis (dtp) is at least the in-plane offset, else in-plane
Per zone: mean prior and observed tumour frequency (sampling weights), per site, then sites equal. A calibrated prior has mean
prior ~ frequency in every zone; the simulated fit is expected to under-predict outside the outlines.
Outputs: clinical/PanCIA_outputs/stage2/validation_agents/prior_calibration.csv and prior_calibration.txt
1 Nov: --samples agent_samples_true scores the fits on true-lesion samples (the prior is P(tumour | lesion real)); --out names the outputs.
Usage: python s2_prior_calibration.py [--fits sim=validation/folds,agent=validation_agents/folds_agent] [--samples agent_samples] [--out prior_calibration]"""
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, prior as PR, fit as FT, expert as EX   # noqa: E402

BANDS = ((0, 5), (5, 10), (10, 20), (20, 60))


def zones(q, sd, dtp, anc=None):
    z = np.full(len(q), 'other', object)
    z[q >= 0.999] = 'anchor'
    if anc is not None:                          # anchors of two-agent vs single-agent lesions (29 Oct)
        z[(q >= 0.999) & (anc == 2)] = 'anchor (two-agent lesion)'; z[(q >= 0.999) & (anc == 1)] = 'anchor (single-agent lesion)'
    z[(q < 0.999) & (sd <= 0)] = 'extent'
    z[(q > 0) & (q < 0.999) & (sd > 0)] = 'outlines'
    out = (q <= 0) & (sd > 0)
    ip = np.sqrt(np.maximum(sd * sd - dtp * dtp, 0.0))
    dirn = np.where(dtp >= ip, 'through', 'in-plane')
    for lo, hi in BANDS:
        s = out & (sd > lo) & (sd <= hi)
        z[s] = [f'outside {lo}-{hi} mm {d}' for d in dirn[s]]
    return z


def cluster_prior(z_, eta):
    """pi per voxel row from the long per-lesion rows of one file"""
    qL, sdL = z_['qL'], z_['sdL']; dtpL = z_['dtpL'] if 'dtpL' in z_.files else None
    sgL = z_['uL'] if 'uL' in z_.files else np.zeros_like(qL)
    ok = ~np.isnan(qL); r_, _ = np.nonzero(ok)
    tp = len(eta) in (PR.P_TP, PR.P_TPS, PR.P_TPO); sg = len(eta) in (PR.P_TPS, PR.P_TPO); ob = len(eta) == PR.P_TPO
    X = PR.basis(qL[ok], sdL[ok], z_['in_h'][r_], z_['in_o'][r_], z_['kb_h'][r_], z_['dep_h'][r_], z_['dep_o'][r_],
                 dtp=dtpL[ok] if tp else None, unagreed=sgL[ok] if sg else None, beyond=ob)
    S, _ = FT.noisy_or_S(X.astype(np.float64), r_, len(qL), np.asarray(eta, np.float64))
    return -np.expm1(-S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fits', default='sim=validation/folds,agent=validation_agents/folds_agent')
    ap.add_argument('--samples', default='agent_samples'); ap.add_argument('--out', default='prior_calibration')
    a = ap.parse_args()
    fits = dict(x.split('=') for x in a.fits.split(','))
    O = paths.out_dir('validation_agents'); A = paths.out_dir(a.samples)
    files = sorted(glob.glob(os.path.join(A, 'samples', '*.npz')))
    rows = []
    for D in dict.fromkeys(EX.SITE_ENTITY.values()):
        held = [ds for ds, e in EX.SITE_ENTITY.items() if e == D]
        fs = [f for f in files if os.path.basename(f).split('__')[0] in held]
        if not fs:
            continue
        etas = {}
        for nm, d in fits.items():
            ff = os.path.join(paths.out_dir(*d.split('/')), f'fit_{D}.json')
            if os.path.exists(ff):
                etas[nm] = json.load(open(ff))['eta']
        if len(etas) < len(fits):                 # compare like with like: a site is scored only when every fit has its fold
            print(f'{D}: skipped (fold fit missing for {sorted(set(fits) - set(etas))})')
            continue
        for f in fs:
            z_ = np.load(f, allow_pickle=False)
            zn = zones(z_['q'], z_['sd'], z_['dtp'], z_['anc'] if 'anc' in z_.files else None)
            w = z_['w'].astype(np.float64); w = w / w.sum()               # each case sums to 1 within its site
            t = z_['t'].astype(np.float64)
            base = pd.DataFrame(dict(zone=zn, w=w, t=w * t))
            for nm, eta in etas.items():
                base[f'pi_{nm}'] = w * cluster_prior(z_, eta)
            g = base.groupby('zone').sum(); g['site'] = D; g['case'] = os.path.basename(f)
            rows.append(g.reset_index())
    if not rows:
        print('no samples or fits'); return
    d = pd.concat(rows, ignore_index=True)
    ps = d.groupby(['site', 'zone']).sum(numeric_only=True)
    for c in [c for c in ps.columns if c.startswith('pi_')] + ['t']:
        ps[c + '_mean'] = ps[c] / ps['w']
    keep = [c for c in ps.columns if c.endswith('_mean')]
    ps[keep + ['w']].to_csv(os.path.join(O, a.out + '.csv'))
    sw = ps[keep].groupby(level='zone').mean().rename(columns={'t_mean': 'tumour_freq'})
    sw['sites'] = ps.groupby(level='zone').size()
    order = ['anchor', 'anchor (two-agent lesion)', 'anchor (single-agent lesion)', 'extent', 'outlines'] + [f'outside {lo}-{hi} mm {dd}' for lo, hi in BANDS for dd in ('in-plane', 'through')]
    sw = sw.reindex([o for o in order if o in sw.index])
    rep = [f'Held-out calibration of the spatial prior on real agent outlines (R2Seg; samples {a.samples}; each site scored by the fold fit that left it out;',
           'mean prior vs observed tumour frequency, sampling-weighted, per site then sites equal). Columns pi_<fit>_mean:',
           ', '.join(f'{k} = {v}' for k, v in fits.items()), '', sw.round(4).to_string(), '',
           f'per site: {a.out}.csv']
    open(os.path.join(O, a.out + '.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


if __name__ == '__main__':
    main()
