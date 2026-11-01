"""Stage 2, step 3: fit r, eta (noisy-OR over the scan's simulated lesions, as run.py combines a cluster), (a_w, b_w) and nu on the
expert samples (s2_expert_build.py). No appearance prior is fitted
(appearance is inferred within each scan).

Writes the fit JSON (default: PanCIA/knowledge_base/knowledge_base/stage2/stage2_fit.json, next to the code) and a text report.
--exclude <collections> leaves collections out (held-out validation folds; then write with --out to the validation folder).
Every estimate weighs tumour sites equally (pancia_kb.stage2.fit.site_weights; user, 25 Oct).
Tail weights (26 Oct): the cell prior of nu (modality x host tissue class; nuprior.fit_prior on the tail study's components,
leaving out the excluded collections) is stored as 'nu_prior'; run.py then estimates every class's nu under it. The single
'nu' is kept as the fallback when no tail-study output exists.
Real-agent samples (29 Oct): --samples agent fits r and eta on s2_agent_build.py's samples (BiomedParse + VoxTell outlines on
R2Seg) instead of the simulated claims; their basis adds the through-plane distance (prior.NAMES_TP). nu, the nu prior and the
tail-study inputs are read from the expert outputs as before. --tumour-from anchor stores tumour_from = 'anchor' (run.py: tumour
appearance and initialisation from the anchors instead of q).
Usage: python s2_fit.py [--samples expert|agent] [--exclude Kidney_Tumor_00] [--out path.json] [--max-rows 4000000] [--tumour-from q|anchor]"""
import os, sys, json, glob, argparse, datetime
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, fit as FT, prior as PR, features as F, expert as EX, nuprior as NP, VERSION   # noqa: E402


def load(E, excl, E_nu=None):
    nus = []
    for f in glob.glob(os.path.join(E_nu or E, 'nu_samples*.jsonl')):           # tumour-intensity samples of the spatial build (for nu)
        nus += [json.loads(l) for l in open(f) if l.strip()]
    nus = [a for a in nus if a['ds'] not in excl]
    rr = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(E, 'r_rows*.csv'))], ignore_index=True)
    rr = rr[~rr.ds.isin(excl)]
    files = sorted(f for f in glob.glob(os.path.join(E, 'samples', '*.npz')) if os.path.basename(f).split('__')[0] not in excl)
    # tumour-free scans are evaluation-only negatives (user, 25 Oct): agent samples of tumour-free scans are not fitted
    def _neg(f):
        m = json.loads(str(np.load(f, allow_pickle=False)['meta']))
        return bool(m.get('negative', (m.get('baseline') or {}).get('T_ml', 1.0) == 0))
    files = [f for f in files if not _neg(f)]
    return nus, rr, files


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--exclude', default=''); ap.add_argument('--out', default=None)
    ap.add_argument('--max-rows', type=float, default=4e6); ap.add_argument('--samples', default='expert', choices=['expert', 'agent', 'agent_true'])
    ap.add_argument('--tumour-from', default='q', choices=['q', 'anchor'])
    ap.add_argument('--beyond', action='store_true', help="add 'in an outline beyond xi' (prior.NAMES_TPO; 1 Nov); agent samples only")
    a = ap.parse_args(); excl = [x for x in a.exclude.split(',') if x]
    # agent_true (1 Nov): true-lesion samples (s2_agent_build.py --true-only), pi = P(tumour at v | the lesion is real)
    E_exp = paths.out_dir('expert'); E = E_exp if a.samples == 'expert' else paths.out_dir({'agent': 'agent_samples', 'agent_true': 'agent_samples_true'}[a.samples])
    out = a.out or paths.fit_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    nu_rows, rr, files = load(E, excl, E_nu=E_exp)
    print(f'{len(files)} cases with samples, {len(nu_rows)} tumour-intensity samples, excluded {excl}', flush=True)
    # sites count equally (user, 25 Oct): rows of r and the cases of eta / omega / nu are site-weighted
    rr = rr[rr.apply(lambda x: (x['ds'], str(x['case'])), axis=1).isin({tuple(os.path.basename(f)[:-4].split('__', 1)) for f in files})] \
        if len(files) else rr                                            # r from the cases that have samples
    r = FT.fit_r(rr.to_dict('records'), site_of=EX.SITE_OF)
    print(f'r = {r:.1f} mm (site-weighted 95th percentile of {len(rr)} anchors)', flush=True)
    if r > 60.0:
        print('WARNING: r exceeds the 60 mm sampling radius of s2_expert_build.py; samples beyond 60 mm exist only in host territories')
    rng = np.random.default_rng(0)
    per_case_rows = max(1000, int(a.max_rows / max(len(files), 1)))
    case_w = FT.site_weights([EX.SITE_OF[os.path.basename(f).split('__')[0]] for f in files])   # each site sums to 1 / n_sites
    S = {k: [] for k in ('q', 'sd', 'in_h', 'in_o', 'kb_h', 'dep_h', 'dep_o', 'ns', 'n1', 't', 'w')}; case_id = []; ds_id = []
    long_r, long_q, long_sd, long_dtp, long_sg = [], [], [], [], []; n_rows = 0
    use_tp = bool(files) and all('dtpL' in np.load(f, allow_pickle=False).files for f in files[:3])
    use_sg = use_tp and all('uL' in np.load(f, allow_pickle=False).files for f in files[:3])     # agreement term (u = 1 - k / n)
    use_ob = bool(a.beyond)
    if use_ob and not use_sg:
        sys.exit('--beyond needs agent samples with dtpL and uL')
    for ci, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        keep = (z['d_anchor'] <= r) | (z['in_h'] > 0)
        grp = z['core'].astype(np.int64) * 10 + z['scen'].astype(np.int64)
        w = z['w'].astype(np.float64).copy(); w[~keep] = 0.0
        gs = np.unique(grp[keep])
        for g in gs:
            s = keep & (grp == g); w[s] = w[s] / w[s].sum() / len(gs)
        idx = np.flatnonzero(keep)
        # uniform subsample (memory); the rows keep their sampling weights, renormalised so that every case sums to 1. (A draw in
        # proportion to the weights followed by equal weights is biased when a large fraction of rows is drawn: self-review 23 Oct.)
        take = idx if len(idx) <= per_case_rows else np.sort(rng.choice(idx, per_case_rows, replace=False))
        ww = w[take] / w[take].sum() * case_w[ci]
        for k in S:
            if k == 'w':
                S[k].append(ww)
            else:
                S[k].append(z[k][take].astype(np.float32))
        # per-lesion q_L, sd_L (noisy-OR over the scan's lesions); samples of builds before 25 Oct have one lesion (q, sd)
        qL = z['qL'][take] if 'qL' in z.files else z['q'][take][:, None]
        sdL = z['sdL'][take] if 'sdL' in z.files else z['sd'][take][:, None]
        ok = ~np.isnan(qL); r_, j_ = np.nonzero(ok)
        long_r.append(r_ + n_rows); long_q.append(qL[ok].astype(np.float32)); long_sd.append(sdL[ok].astype(np.float32))
        if use_tp:
            long_dtp.append(z['dtpL'][take][ok].astype(np.float32))
        if use_sg:
            long_sg.append(z['uL'][take][ok].astype(np.float32))
        n_rows += len(take)
        case_id.append(np.full(len(take), ci, np.int32)); ds_id.append(np.full(len(take), os.path.basename(f).split('__')[0]))
    S = {k: np.concatenate(v) for k, v in S.items()}; case_id = np.concatenate(case_id); ds_id = np.concatenate(ds_id)
    grp = np.concatenate(long_r); Lq = np.concatenate(long_q); Lsd = np.concatenate(long_sd); del long_r, long_q, long_sd
    Ldtp = np.concatenate(long_dtp) if use_tp else None; Lsg = np.concatenate(long_sg) if use_sg else None; del long_dtp, long_sg
    print(f'{len(case_id)} voxel rows, {len(grp)} voxel x lesion rows for the fit '
          f'(lesions per voxel: max {np.bincount(grp).max()}, share with > 1: {(np.bincount(grp) > 1).mean():.1%})', flush=True)
    Xl = PR.basis(Lq, Lsd, S['in_h'][grp], S['in_o'][grp], S['kb_h'][grp], S['dep_h'][grp], S['dep_o'][grp], dtp=Ldtp, unagreed=Lsg, beyond=use_ob); del Lq, Lsd, Ldtp, Lsg
    eta, info = FT.fit_eta_noisy_or(Xl, grp, S['t'], S['w'])
    Ssum, _ = FT.noisy_or_S(Xl.astype(np.float64), grp, len(case_id), eta); del Xl
    p = -np.expm1(-Ssum); z = FT.logit_from_S(Ssum)                       # cluster prior and its logit per voxel
    cal = FT.calibration(p, S['t'], S['w'])
    # coupling: per-case pseudo-likelihood
    om, om_site = [], []
    for ci in np.unique(case_id):
        s = case_id == ci
        if S['t'][s].sum() < 20:
            continue
        om.append(FT.fit_omega_case(z[s], S['ns'][s], S['n1'][s], S['t'][s], S['w'][s])); om_site.append(EX.SITE_OF[ds_id[s][0]])
    a_w, b_w = FT.gamma_moments(om, FT.site_weights(om_site))
    # tail shape nu: population prior from the expert tumours' components (s2_nu.py; pooled on 1/nu, sites equal). Fallback for
    # outputs without the component pass: one Student-t per whole tumour (reads between-tissue heterogeneity as tails; flagged)
    comp_rows = []
    for f in glob.glob(os.path.join(E, 'nu_components*.jsonl')):
        comp_rows += [json.loads(l) for l in open(f) if l.strip()]
    comp_rows = [r_ for r_ in comp_rows if r_['ds'] not in excl]
    nu_pop = FT.pooled_nu(comp_rows, EX.SITE_OF)
    if nu_pop:
        nu = float(nu_pop['nu']); nus = [None] * nu_pop['n_components']; nu_src = 'components (s2_nu.py)'
    else:
        tum = [a_ for a_ in nu_rows if a_.get('tumour_xs')]
        nu, nus = FT.fit_nu([a_['tumour_xs'] for a_ in tum], [EX.SITE_OF[a_['ds']] for a_ in tum]); nu_src = 'WHOLE TUMOUR (no s2_nu.py output)'
        print('WARNING: no nu_components*.jsonl (s2_nu.py); nu from one Student-t per whole tumour')
    nu_prior = NP.fit_prior(exclude=excl)
    if nu_prior is None:
        print('WARNING: no tail-study components.csv (s2_tail_study.py); one fixed nu for every class')
    else:
        print('nu prior (tumour, nu at the cell mean): ' + ', '.join(f"{k} {v['nu_at_mean']}" for k, v in nu_prior['tumour']['cells'].items()), flush=True)
    res = dict(version=VERSION, date=datetime.date.today().isoformat(), excluded=excl, features=F.NAMES,
               eta_names=PR.NAMES_TPO if use_ob else (PR.NAMES_TPS if use_sg else (PR.NAMES_TP if use_tp else PR.NAMES)), samples=a.samples, tumour_from=a.tumour_from,
               nu_prior=nu_prior,
               eta=[float(x) for x in eta], eta_fit=info, r_mm=r, a_w=a_w, b_w=b_w, nu=nu, **FT.DESIGN,
               n_cases={d: int(v) for d, v in pd.Series([os.path.basename(f).split('__')[0] for f in files]).value_counts().items()},
               omega=dict(n=len(om), median=float(np.median(om)) if om else None, q10=float(np.percentile(om, 10)) if om else None,
                          q90=float(np.percentile(om, 90)) if om else None),
               nu_cases=dict(n=len(nus), median=nu), nu_source=nu_src, nu_population=nu_pop, calibration_in_sample=cal)
    sites = pd.Series([EX.SITE_OF[d] for d in ds_id[np.unique(case_id, return_index=True)[1]]]).value_counts().to_dict()
    res['weighting'] = dict(rule='sites count equally (each site weight 1 / n_sites, shared by its cases)', cases_per_site=sites)
    json.dump(res, open(out, 'w'), indent=1)
    rep = [f'stage-2 fit {res["date"]} ({VERSION}); samples {a.samples}; tumour classes from {a.tumour_from}; excluded {excl}; cases {res["n_cases"]}',
           f'weighting: sites count equally; cases per site {sites} (each case weighs 1 / ({len(sites)} x cases of its site))',
           'appearance: no prior across scans; inferred within each scan (vb.class_priors, strength beta0 = '
           f"{FT.DESIGN['nw_beta0']:g}, nu0 = {FT.DESIGN['nw_nu0']:g})",
           f'r = {r:.1f} mm; nu = {nu:.2f} ({nu_src}; {len(nus)} components / cases); omega per case n={len(om)} median {res["omega"]["median"]} -> Gamma(a_w={a_w}, b_w={b_w})',
           ('nu population (1/nu pooled; sites, cases, tumours, components equal): ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in nu_pop.items()})) if nu_pop else 'nu population: none',
           f'features: {F.NAMES} (D = {F.D})',
           ('nu prior (modality x tissue class; nu at the cell mean / sd of log 1/nu): tumour ' + '; '.join(f"{k} {v['nu_at_mean']}/{v['sd_x']}" for k, v in nu_prior['tumour']['cells'].items())
            + f"; fallback {nu_prior['tumour']['fallback']['nu_at_mean']}/{nu_prior['tumour']['fallback']['sd_x']} | normal "
            + '; '.join(f"{k} {v['nu_at_mean']}/{v['sd_x']}" for k, v in nu_prior['normal']['cells'].items())
            + f"; fallback {nu_prior['normal']['fallback']['nu_at_mean']}/{nu_prior['normal']['fallback']['sd_x']}") if nu_prior else 'nu prior: none (fixed nu)',
           'eta:'] + [f'  {n:>14s} {v:+.4f}' for n, v in zip(res['eta_names'], eta)] + ['in-sample calibration (mean p -> frequency):'] + \
          [f"  {c['bin']}: {c['mean_p']:.3f} -> {c['freq']:.3f} (weight {c['weight']:.3g})" for c in cal]
    # prior at reference points, per scenario (review aid)
    for nm, (q, sd, ih, io, kb, dh, do, dt) in {'inside anchor (q=1, in host)': (1, -5, 1, 0, 0, 10, 0, 0),
                                                'inside anchor (q=1, in host), one of two agents': (1, -5, 1, 0, 0, 10, 0, -1),
                                                'chosen extent rim (q=0.5, in host)': (0.5, -1, 1, 0, 0, 3, 0, 0),
                                                'in an outline 3 mm beyond extent (q=0.2, in host)': (0.2, 3, 1, 0, 0, 3, 0, 0),
                                                '5 mm outside extent in plane, in host': (0, 5, 1, 0, 0, 5, 0, 0),
                                                '5 mm outside extent through plane, in host': (0, 5, 1, 0, 0, 5, 0, 5),
                                                '10 mm outside extent through plane, in host': (0, 10, 1, 0, 0, 5, 0, 10),
                                                '5 mm outside extent, other organ': (0, 5, 0, 1, 0, 0, 5, 0),
                                                '5 mm outside extent, no organ': (0, 5, 0, 0, 0, 0, 0, 0),
                                                '20 mm outside extent in plane, in host': (0, 20, 1, 0, 0, 10, 0, 0),
                                                '20 mm outside, host unsegmented (KB region)': (0, 20, 1, 0, 1, 0, 0, 0)}.items():
        if dt < 0 and not use_sg:
            continue
        xb = PR.basis([q], [sd], [ih], [io], kb, [dh], [do], dtp=[max(dt, 0)] if use_tp else None,
                      unagreed=[0.5 if dt < 0 else 0.0] if use_sg else None, beyond=use_ob)
        rep.append(f'  prior {nm}: {float(PR.sigmoid(xb @ eta)[0]):.4f}')
    open(out.replace('.json', '_report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print('wrote', out)


if __name__ == '__main__':
    main()
