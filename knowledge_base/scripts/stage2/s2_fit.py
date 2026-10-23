"""Stage 2, step 3: fit r, eta, (a_w, b_w), nu and the appearance priors on the expert samples (s2_expert_build.py).

Writes the fit JSON (default: PanCIA/knowledge_base/knowledge_base/stage2/stage2_fit.json, next to the code) and a text report.
--exclude <collections> leaves collections out (held-out validation folds; then write with --out to the validation folder).
Usage: python3 s2_fit.py [--exclude Kidney_Tumor_00] [--out path.json] [--max-rows 4000000]"""
import os, sys, json, glob, argparse, datetime
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, fit as FT, prior as PR, features as F, VERSION   # noqa: E402


def load(E, excl):
    apps = []
    for f in glob.glob(os.path.join(E, 'appearance*.jsonl')):
        apps += [json.loads(l) for l in open(f) if l.strip()]
    apps = [a for a in apps if a['ds'] not in excl]
    rr = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(E, 'r_rows*.csv'))], ignore_index=True)
    rr = rr[~rr.ds.isin(excl)]
    files = sorted(f for f in glob.glob(os.path.join(E, 'samples', '*.npz')) if os.path.basename(f).split('__')[0] not in excl)
    return apps, rr, files


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--exclude', default=''); ap.add_argument('--out', default=None)
    ap.add_argument('--max-rows', type=float, default=4e6)
    a = ap.parse_args(); excl = [x for x in a.exclude.split(',') if x]
    E = paths.out_dir('expert'); out = a.out or paths.fit_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    apps, rr, files = load(E, excl)
    print(f'{len(files)} cases with samples, {len(apps)} appearance rows, excluded {excl}', flush=True)
    r = FT.fit_r(rr.to_dict('records'))
    print(f'r = {r:.1f} mm (95th percentile of {len(rr)} anchors)', flush=True)
    if r > 60.0:
        print('WARNING: r exceeds the 60 mm sampling radius of s2_expert_build.py; samples beyond 60 mm exist only in host territories')
    rng = np.random.default_rng(0)
    per_case_rows = max(1000, int(a.max_rows / max(len(files), 1)))
    S = {k: [] for k in ('q', 'sd', 'in_h', 'in_o', 'kb_h', 'dep_h', 'dep_o', 'ns', 'n1', 't', 'w')}; case_id = []; ds_id = []
    for ci, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        keep = (z['d_anchor'] <= r) | (z['in_h'] > 0)
        grp = z['core'].astype(np.int64) * 10 + z['scen'].astype(np.int64)
        w = z['w'].astype(np.float64).copy(); w[~keep] = 0.0
        gs = np.unique(grp[keep])
        for g in gs:
            s = keep & (grp == g); w[s] = w[s] / w[s].sum() / len(gs)
        idx = np.flatnonzero(keep)
        if len(idx) > per_case_rows:
            pr = w[idx] / w[idx].sum()
            take = rng.choice(idx, per_case_rows, replace=False, p=pr)        # weighted subsample -> equal weights afterwards
            ww = np.full(len(take), 1.0 / len(take))
        else:
            take = idx; ww = w[idx]
        for k in S:
            if k == 'w':
                S[k].append(ww)
            else:
                S[k].append(z[k][take].astype(np.float32))
        case_id.append(np.full(len(take), ci, np.int32)); ds_id.append(np.full(len(take), os.path.basename(f).split('__')[0]))
    S = {k: np.concatenate(v) for k, v in S.items()}; case_id = np.concatenate(case_id); ds_id = np.concatenate(ds_id)
    print(f'{len(case_id)} rows for the fit', flush=True)
    X = FT.design_matrix(S)
    eta, info = FT.fit_eta(X, S['t'], S['w'])
    z = X.astype(np.float64) @ eta; p = 1.0 / (1.0 + np.exp(-z))
    cal = FT.calibration(p, S['t'], S['w'])
    # coupling: per-case pseudo-likelihood
    om = []
    for ci in np.unique(case_id):
        s = case_id == ci
        if S['t'][s].sum() < 20:
            continue
        om.append(FT.fit_omega_case(z[s], S['ns'][s], S['n1'][s], S['t'][s], S['w'][s]))
    a_w, b_w = FT.gamma_moments(om)
    nu, nus = FT.fit_nu([ap_['tumour_xs'] for ap_ in apps])
    app = FT.fit_appearance(apps)
    res = dict(version=VERSION, date=datetime.date.today().isoformat(), excluded=excl, features=F.NAMES, eta_names=PR.NAMES,
               eta=[float(x) for x in eta], eta_fit=info, r_mm=r, a_w=a_w, b_w=b_w, nu=nu, appearance=app, **FT.DESIGN,
               n_cases={d: int(v) for d, v in pd.Series([os.path.basename(f).split('__')[0] for f in files]).value_counts().items()},
               omega=dict(n=len(om), median=float(np.median(om)) if om else None, q10=float(np.percentile(om, 10)) if om else None,
                          q90=float(np.percentile(om, 90)) if om else None),
               nu_cases=dict(n=len(nus), median=nu), calibration_in_sample=cal)
    json.dump(res, open(out, 'w'), indent=1)
    rep = [f'stage-2 fit {res["date"]} ({VERSION}); excluded {excl}; cases {res["n_cases"]}',
           f'r = {r:.1f} mm; nu = {nu:.2f} ({len(nus)} cases); omega per case n={len(om)} median {res["omega"]["median"]} -> Gamma(a_w={a_w}, b_w={b_w})',
           'eta:'] + [f'  {n:>14s} {v:+.4f}' for n, v in zip(PR.NAMES, eta)] + ['in-sample calibration (mean p -> frequency):'] + \
          [f"  {c['bin']}: {c['mean_p']:.3f} -> {c['freq']:.3f} (weight {c['weight']:.3g})" for c in cal]
    # prior at reference points, per scenario (review aid)
    for nm, (q, sd, ih, io, kb, dh, do) in {'inside anchor (q=1, in host)': (1, -5, 1, 0, 0, 10, 0),
                                            'chosen extent rim (q=0.5, in host)': (0.5, -1, 1, 0, 0, 3, 0),
                                            '5 mm outside extent, in host': (0, 5, 1, 0, 0, 5, 0),
                                            '5 mm outside extent, other organ': (0, 5, 0, 1, 0, 0, 5),
                                            '5 mm outside extent, no organ': (0, 5, 0, 0, 0, 0, 0),
                                            '20 mm outside extent, in host': (0, 20, 1, 0, 0, 10, 0),
                                            '20 mm outside, host unsegmented (KB region)': (0, 20, 1, 0, 1, 0, 0)}.items():
        xb = PR.basis([q], [sd], [ih], [io], kb, [dh], [do]); rep.append(f'  prior {nm}: {float(PR.sigmoid(xb @ eta)[0]):.3f}')
    open(out.replace('.json', '_report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print('wrote', out)


if __name__ == '__main__':
    main()
