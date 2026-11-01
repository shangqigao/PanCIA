"""Stage 2, step 5b: held-out validation with REAL agent outlines on R2Seg (leave one tumour site out). Expert truth for evaluation only.

Same folds and fold fits as s2_validate.py (validation/folds/fit_<site>.json: fitted on expert data of the other sites; fitted here
when missing), but the stage-1 lesions come from the BiomedParse and VoxTell tumour masks (pancia_kb/stage2/agents.py: anchors =
agreement components >= 0.5 ml, outlines = each agent's components touching an anchor, extents and w0 by the frozen v8.4 rule,
anchors chained within 10 mm). TotalSegmentator anatomy only (as on TCGA). Note: both agents were trained on data that overlaps
R2Seg, so their errors here are likely smaller than on unseen data.

Per scan of the held-out site (up to --n-val tumour scans and --n-neg tumour-free scans that have both agent masks, in the fixed
order of s2_validate.py):
  baseline     scan-level Dice / volumes of each agent, their agreement and union against all expert tumour; anchors, lesions,
               expert tumours >= 0.5 ml detected by an anchor (stage-1 detection)
  lesion       per true lesion (an anchor touches expert tumour; anchor_precision = share of its anchor inside tumour) and method (stage2, xi, q_half, U, anchor, bp, vt): Dice, precision,
               recall against the expert components its anchors touch; recall beyond the lesion's candidate extents by distance band;
               log Bayes factor
  false_anchor per false lesion (no anchor touches tumour; up to --max-false per scan, largest first): stage 2 with the same
               settings as true lesions, log Bayes factor, refined volume; in tumour scans and in tumour-free scans. Lesions beyond
               --max-false are counted and add their stage-1 volume (false positives) to the scan level of every method but stage2
  missed       per expert tumour >= 0.5 ml without an anchor: volume and the part stage 2 marks as tumour (new candidates)
  scan_part    per window and method: true- and false-positive volume (scan-level Dice in the summary, false lesions included)
  calibration  of p_v in the region of the tumour window (10 bins)
Outputs: clinical/PanCIA_outputs/stage2/validation_agents/ rows[_s<i>].csv, calibration[_s<i>].csv, summary_*.csv, report.txt
Shards (SHARD=i/n) split the scans of all sites round-robin (30 Oct; was one site per shard).
Usage: [SHARD=i/n] python s2_validate_agents.py [--folds kidney,...] [--n-val 40] [--n-neg 20] [--max-false 3]
       [--fit-dir validation/folds --fit-samples expert --tumour-from q --tag validation_agents]   (defaults: the baseline)
       python s2_validate_agents.py --summary
       python s2_validate_agents.py --check          (agent masks found per site; nothing is run)"""
import os, sys, json, time, zlib, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from pancia_kb import KnowledgeBase, grounding_g1 as G1, grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, run as RN, agents as AG   # noqa: E402
import s2_validate as V   # noqa: E402

METHODS = ('stage2', 'xi', 'q_half', 'U', 'anchor', 'bp', 'vt')
PAD_MM = AG.PAD_MM


def _common(fold, ds, case, c=None):
    d = dict(fold=fold, ds=ds, case=case)
    if c is not None:
        d.update(mod=c['mod'], seq=c['seq'], cls=c['cls'])
    else:
        d.update(mod=EX.APP_MOD.get(ds, 'CT'), cls=EX.APPEARANCE[ds]['cls'])
    return d


def _specs(c, S, Ls, beta, psi):
    """lesion specs for run.solve_cluster and the stage-1 quantities of each lesion (None entries dropped)"""
    theta, kb, seg, depth, hkey = EX.territories(c, 'ts')
    cand = V.host_candidates(theta, hkey); kbc = {h: kb[h] for h in cand}
    specs, built, keep = [], [], []
    for L in Ls:
        B = AG.build_lesion(c, S, L, G1, G2, psi)
        if B is None:
            continue
        PhL, Sset, _ = V.host_posterior(B['A'], cand, kbc, c['sp'], beta)
        specs.append(dict(key=f'L{len(specs)}', anchors=[RN.Patch.from_full(a) for a in B['amasks']], q=RN.Patch.from_full(B['q']),
                          xi=RN.Patch.from_full(B['xi']), PhL=PhL, S=Sset, tum_cls=c['cls'], support=L.get('support', 'two')))
        built.append(B); keep.append(L)
    return specs, built, keep, theta, kb, hkey


def _solve(c, specs, theta, kb, fit, max_grow=5):
    body = F.region_body(c['X'], c['mod'], c['ax']); bodyP = RN.Patch.from_full(body)
    organs = V.organ_specs(theta, kb)
    return RN.solve_cluster(lambda W: c['X'][W], specs, organs, fit, c['sp'], c['mod'], c['thick'], c['ax'], bodyP, c['T'].shape,
                            max_grow=max_grow, seq=c['seq'])


def _bf(L0):
    return dict(log_bf=L0.get('log_bf'), flag_bf=L0.get('flag_bf'), n_vox_bf=L0.get('n_vox_bf'), log_bf_per_vox=L0.get('log_bf_per_vox'))


def _scan_parts(base, res, built, c, vml):
    """per method: true- and false-positive volume of the union over the window's lesions (stage2: refined tumours only)"""
    W_ = res['W']; Tw = c['T'][W_]; out = []
    for m in METHODS:
        P = np.zeros(Tw.shape, bool)
        for k, B in enumerate(built):
            P |= _pred(m, res, k, B, W_)
        out.append(dict(base, kind='scan_part', method=m, tp_ml=float((P & Tw).sum() * vml), fp_ml=float((P & ~Tw).sum() * vml)))
    return out


def _pred(m, res, k, B, W_):
    if m == 'stage2':
        return res['ZL'][k]
    if m == 'q_half':
        return B['q'][W_] >= 0.5
    return {'xi': B['xi'], 'U': B['U'], 'anchor': B['A'], 'bp': B['bp'], 'vt': B['vt']}[m][W_]


def false_lesions(ds, S, Ls, fit, beta, psi, base, rows, max_false, where):
    """each false lesion (largest first, up to max_false) in a window of its own: stage 2 with the same settings as the tumour
    window (region growth included), Bayes factor. Lesions beyond max_false are not inferred; their stage-1 quantities still count
    as false-positive volume for every method except stage2 (kind scan_part, window 'false_not_run')."""
    Ls = sorted(Ls, key=lambda L: -L['ml'])
    for L in Ls[max_false:]:
        W = F.pad_box(L['obox'], S['sp'], PAD_MM, S['T'].shape)
        c = AG.window(ds, S, W, KB)
        if c is None:
            continue
        B = AG.build_lesion(c, S, L, G1, G2, psi)
        if B is None:
            continue
        vml = float(np.prod(c['sp'])) / 1000.0; T = c['T']
        for m in METHODS:
            if m == 'stage2':
                continue
            P = B['q'] >= 0.5 if m == 'q_half' else {'xi': B['xi'], 'U': B['U'], 'anchor': B['A'], 'bp': B['bp'], 'vt': B['vt']}[m]
            rows.append(dict(base, mod=c['mod'], window='false_not_run', kind='scan_part', method=m,
                             tp_ml=float((P & T).sum() * vml), fp_ml=float((P & ~T).sum() * vml)))
    for L in Ls[:max_false]:
        W = F.pad_box(L['obox'], S['sp'], PAD_MM, S['T'].shape)
        c = AG.window(ds, S, W, KB)
        if c is None:
            continue
        specs, built, keep, theta, kb, hkey = _specs(c, S, [L], beta, psi)
        if not specs:
            continue
        t0 = time.time(); res = _solve(c, specs, theta, kb, fit); vml = float(np.prod(c['sp'])) / 1000.0
        L0 = res['lesions'][0]
        rows.append(dict(base, mod=c['mod'], seq=c['seq'], kind='false_anchor', where=where, method='stage2', anchor_ml=L['ml'],
                         support=L.get('support', 'two'),
                         n_anchors=len(L['anchors']), xi_name=built[0]['xi_name'], host=V._ent(hkey), host_in_S=hkey in specs[0]['S'],
                         vol_Z=L0['vol_Z'], vol_anchor=L0['vol_anchor'], p_anchor_ge_half=L0['p_anchor_ge_half'],
                         sec=round(time.time() - t0, 1), **_bf(L0)))
        rows += _scan_parts(dict(base, mod=c['mod'], window='false'), res, built, c, vml)
    if len(Ls) > max_false:
        rows.append(dict(base, kind='false_not_run', where=where, n=len(Ls) - max_false, ml=float(sum(L['ml'] for L in Ls[max_false:]))))


def run_positive(ds, lf, fit, beta, psi, fold, rows, calib, max_false):
    S = AG.scan(ds, lf, single=SINGLE[0])
    if S is None:
        return False
    base = _common(fold, ds, S['case'])
    rows.append(dict(base, kind='baseline', **S['baseline']))
    if not S['T'].any():
        return True
    trueL = [L for L in S['lesions'] if L['true']]
    Wt = AG._box_union([F.bbox(S['T'])] + [L['obox'] for L in trueL])
    W = F.pad_box(Wt, S['sp'], PAD_MM, S['T'].shape)
    inside = lambda b: all(b[k].start >= W[k].start and b[k].stop <= W[k].stop for k in range(3))
    inL = [L for L in S['lesions'] if L['true'] or inside(L['obox'])]          # false lesions in the window are inferred jointly
    outL = [L for L in S['lesions'] if not (L['true'] or inside(L['obox']))]
    c = AG.window(ds, S, W, KB)
    if c is None:
        rows.append(dict(base, kind='skip', note='no TS mask on the image grid'))
        return True
    base = _common(fold, ds, S['case'], c); vml = float(np.prod(c['sp'])) / 1000.0
    specs, built, keep, theta, kb, hkey = _specs(c, S, inL, beta, psi)
    hit = np.zeros(c['T'].shape, bool)
    for B in built:
        hit |= B['A']
    res = None
    if specs:
        t0 = time.time(); res = _solve(c, specs, theta, kb, fit); sec = round(time.time() - t0, 1)
        W_ = res['W']
        common = dict(base, n_lesions=len(specs), n_grow=res['n_grow'], iters=res['model']['iters'], converged=res['model']['converged'],
                      k_eff=res['model']['k_eff'], n_new=len(res['new_candidates']), sec=sec, host=V._ent(hkey))
        for k, (B, L) in enumerate(zip(built, keep)):
            L0 = res['lesions'][k]
            if not L['true']:
                rows.append(dict(common, kind='false_anchor', where='tumour_window', method='stage2', anchor_ml=L['ml'],
                                 support=L.get('support', 'two'),
                                 n_anchors=len(L['anchors']), xi_name=B['xi_name'], host_in_S=hkey in specs[k]['S'],
                                 vol_Z=L0['vol_Z'], vol_anchor=L0['vol_anchor'], p_anchor_ge_half=L0['p_anchor_ge_half'], **_bf(L0)))
                continue
            Gk = AG.lesion_truth(c, B['A'])[W_]
            lb = dict(common, kind='lesion', lesion=k, n_anchors=len(L['anchors']), support=L.get('support', 'two'), xi_name=B['xi_name'],
                      host_in_S=hkey in specs[k]['S'],
                      G_ml=float(Gk.sum() * vml), anchor_ml=float(B['A'].sum() * vml),
                      anchor_precision=float((B['A'] & c['T']).sum() / max(int(B['A'].sum()), 1)), **_bf(L0))
            Uw = B['U'][W_]; beyond = Gk & ~Uw
            if beyond.any():
                dU = F.edt(Uw, c['sp']) if Uw.any() else np.full(Uw.shape, np.inf)
                for lo, hi in V.BANDS:
                    m = beyond & (dU > lo) & (dU <= hi); nm = f'{lo:g}_{hi:g}'
                    lb[f'miss_{nm}_ml'] = float(m.sum() * vml); lb[f'rec_{nm}_ml'] = float((m & res['ZL'][k]).sum() * vml)
            for m in METHODS:
                rows.append(dict(lb, method=m, **V.metrics(_pred(m, res, k, B, W_), Gk)))
        rows += _scan_parts(dict(base, window='tumour'), res, built, c, vml)
        p = res['p'][res['ROI']]; t = c['T'][W_][res['ROI']]
        hb = np.clip((p * 10).astype(int), 0, 9)
        for b in range(10):
            s_ = hb == b
            if s_.any():
                calib.append(dict(fold=fold, mod=c['mod'], bin=b, n=int(s_.sum()), sum_p=float(p[s_].sum()), sum_t=float(t[s_].sum())))
    # expert tumours >= 0.5 ml no anchor touches: stage-1 misses; the part stage 2 marks as tumour (Z includes new candidates)
    for g in c['comps']:
        if (g & hit).any():
            continue
        r = dict(base, kind='missed', ml=float(g.sum() * vml))
        if res is not None:
            W_ = res['W']; r['rec_ml'] = float((g[W_] & res['Z']).sum() * vml)
        rows.append(r)
    false_lesions(ds, S, outL, fit, beta, psi, base, rows, max_false, 'tumour_scan')
    return True


def run_negative(ds, lf, fit, beta, psi, fold, rows, max_false):
    S = AG.scan(ds, lf, single=SINGLE[0])
    if S is None:
        return False
    base = _common(fold, ds, S['case'])
    rows.append(dict(base, kind='baseline_neg', **S['baseline']))
    false_lesions(ds, S, S['lesions'], fit, beta, psi, base, rows, max_false, 'tumour_free_scan')
    return True


def _append(O, tag, rows, calib):
    for nm, rr in ((f'rows{tag}.csv', rows), (f'calibration{tag}.csv', calib)):
        f = os.path.join(O, nm)
        if rr:
            df = pd.DataFrame(rr)
            if os.path.exists(f):
                df = pd.concat([pd.read_csv(f, low_memory=False), df], ignore_index=True)
            df.to_csv(f + '.tmp', index=False); os.replace(f + '.tmp', f)     # atomic: a kill mid-write cannot truncate the file


def _ordered(lst, held):
    cases = [t for t in lst if t[0] in held]
    return [cases[i] for i in np.argsort([zlib.crc32(os.path.basename(lf).encode()) for _, lf in cases], kind='stable')]


KB = None
SINGLE = [False]


def main():
    sites = list(dict.fromkeys(EX.SITE_ENTITY.values()))
    ap = argparse.ArgumentParser(); ap.add_argument('--folds', default=','.join(sites)); ap.add_argument('--n-val', type=int, default=40)
    ap.add_argument('--n-neg', type=int, default=20); ap.add_argument('--max-false', type=int, default=3)
    ap.add_argument('--summary', action='store_true'); ap.add_argument('--check', action='store_true')
    # 29 Oct: which fold fits to use. Default = the simulated-claim fits of s2_validate.py (baseline). The real-agent model:
    # --fit-dir validation_agents/folds_agent --fit-samples agent --tumour-from anchor --tag validation_agents_v2
    ap.add_argument('--fit-dir', default='validation/folds'); ap.add_argument('--fit-samples', default='expert', choices=['expert', 'agent', 'agent_true'])
    ap.add_argument('--tumour-from', default='q', choices=['q', 'anchor']); ap.add_argument('--tag', default='validation_agents')
    ap.add_argument('--fit-beyond', action='store_true', help="fold fits with 'in an outline beyond xi' (s2_fit.py --beyond), when fitted here")
    ap.add_argument('--single', action='store_true', help='single-agent anchors too (v8.4 kind single); default: agreement only')
    ap.add_argument('--psi', default=None, help='stage-1 extent prior JSON with a psi entry (ledger/v85/extent/fit_psi.json); default v8.4')
    a = ap.parse_args()
    O = paths.out_dir(a.tag); FD = paths.out_dir(*a.fit_dir.split('/'))
    SINGLE[0] = a.single
    si, sn = (int(x) for x in os.environ.get('SHARD', '0/1').split('/')); tag = f'_s{si}' if sn > 1 else ''
    if a.summary:
        summarise(O); return
    r2 = paths.root('R2Seg')
    if a.check:
        for D in a.folds.split(','):
            held = [ds for ds, e in EX.SITE_ENTITY.items() if e == D] or [D]
            pos = _ordered(EX.case_list(r2), held); neg = _ordered(EX.negative_list(r2), held)
            av = lambda L: sum(AG.available(ds, os.path.basename(lf).replace('_seg.nii.gz', '')) for ds, lf in L)
            print(f'{D}: tumour scans with both agent masks {av(pos)} of {len(pos)} usable (first {a.n_val} in order: {av(pos[:a.n_val])}); '
                  f'tumour-free {av(neg)} of {len(neg)}')
        return
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(json.load(open(a.psi))['psi'] if a.psi else fit6['psi'])     # --psi: stage-1 extent prior (default v8.4)
    beta = fit6['beta_UD']
    global KB
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base')); V.KB = KB
    done = set()
    for fr in [os.path.join(O, f) for f in os.listdir(O) if f.startswith('rows') and f.endswith('.csv')]:
        d0 = pd.read_csv(fr, low_memory=False)
        d0 = d0[d0.kind != 'error'] if 'kind' in d0 else d0          # scans that failed are retried on the next run
        done |= set(zip(d0.ds, d0.case.astype(str)))
    # 30 Oct: shards split by (site, case), not by site (liver alone was 10 of 34 CPU-h, so site shards waited on it). The cases of
    # every fold are fixed first (the first --n-val / --n-neg scans with both agent masks, in the fixed order, as before), then
    # interleaved across sites by their rank and dealt round-robin to the shards. Same cases and results as the per-site split; a run
    # can resume with a different shard count (finished scans are skipped by ds/case).
    folds = a.folds.split(',')
    tasks = []
    for fi, D in enumerate(folds):
        held = [ds for ds, e in EX.SITE_ENTITY.items() if e == D] or [D]
        for ki, (kind, lst, nmax) in enumerate((('tumour', EX.case_list(r2), a.n_val), ('tumour-free', EX.negative_list(r2), a.n_neg))):
            sel = [(ds, lf) for ds, lf in _ordered(lst, held) if AG.available(ds, os.path.basename(lf).replace('_seg.nii.gz', ''))][:nmax]
            tasks += [(rk, ki, fi, D, kind, ds, lf, len(sel)) for rk, (ds, lf) in enumerate(sel)]
    tasks.sort(key=lambda t: t[:3])
    mine = tasks[si::sn]
    need = sorted({t[3] for t in mine})
    miss = [D for D in need if not os.path.exists(os.path.join(FD, f'fit_{D}.json'))]
    if miss and sn > 1:
        sys.exit(f'fold fits missing in {FD} for {miss}: run once without SHARD (or s2_fit.py --exclude) before sharding')
    fits, psis = {}, {}
    pj = json.load(open(a.psi)) if a.psi else None
    for D in need:
        held = [ds for ds, e in EX.SITE_ENTITY.items() if e == D] or [D]
        ff = os.path.join(FD, f'fit_{D}.json')
        if not os.path.exists(ff):
            subprocess.run([sys.executable, os.path.join(HERE, 's2_fit.py'), '--exclude', ','.join(held), '--out', ff,
                            '--samples', a.fit_samples, '--tumour-from', a.tumour_from] + (['--beyond'] if a.fit_beyond else []), check=True)
        fits[D] = json.load(open(ff))
        psis[D] = psi
        if pj and pj.get('adopted') and D in (pj.get('folds') or {}):   # held-out site: the psi fold fitted without it (s1_fit_psi.py)
            psis[D] = G2.Psi(dict(edges=pj['psi']['edges'], val=pj['folds'][D][pj['method']]))
    print(f'shard {si}/{sn}: {len(mine)} of {len(tasks)} scans ({sum((t[5], os.path.basename(t[6]).replace("_seg.nii.gz", "")) in done for t in mine)} done)', flush=True)
    for rk, _, _, D, kind, ds, lf, nmax in mine:
        case = os.path.basename(lf).replace('_seg.nii.gz', '')
        if (ds, case) in done:
            continue
        rows, calib = [], []; t0 = time.time()
        try:
            if kind == 'tumour':
                ok = run_positive(ds, lf, fits[D], beta, psis[D], D, rows, calib, a.max_false)
            else:
                ok = run_negative(ds, lf, fits[D], beta, psis[D], D, rows, a.max_false)
            if not ok:
                continue
        except Exception:
            rows.append(dict(fold=D, ds=ds, case=case, kind='error', note=traceback.format_exc()[-600:].replace('\n', ' | ')))
        _append(O, tag, rows, calib)
        print(f'{D} {kind} {rk + 1}/{nmax} {case} {time.time() - t0:.0f}s', flush=True)
    if sn == 1:
        summarise(O)


# ------------------------------------------------------------------ summary (sites count equally)
def _sw(per):
    """site-weighted mean of a per-site Series / DataFrame"""
    return per.mean()


def summarise(O):
    d = V._cat(O, 'rows')
    if d is None:
        print('no rows yet'); return
    rep = ['Stage-2 validation with real agent outlines (BiomedParse + VoxTell) on R2Seg, leave one tumour site out; TotalSegmentator '
           'anatomy; sites count equally. Both agents have likely seen much of R2Seg in training.', '']
    S_ = []
    # 1. agents and stage-1 detection
    b = d[d.kind == 'baseline'].copy()
    if len(b):
        b['det_rate'] = b.n_tumour_detected / b.n_tumour.where(b.n_tumour > 0)
        b['det_vol'] = b.tumour_detected_ml / b.tumour_ml.where(b.tumour_ml > 0)
        cols = ['bp_dice', 'vt_dice', 'agree_dice', 'union_dice', 'det_rate', 'det_vol', 'n_lesion', 'n_lesion_true']
        ps = b.groupby('fold')[cols].mean(); ps['scans'] = b.groupby('fold').size()
        ps.loc['(sites equal)'] = ps.mean()
        ps.round(3).to_csv(os.path.join(O, 'summary_agents.csv'))
        rep += ['== agents (scan level, all expert tumour) and stage-1 detection (expert tumours >= 0.5 ml touched by an anchor) ==',
                ps.round(3).to_string(), '']
        if b.resampled.notna().any() and (b.resampled.astype(str).str.len() > 0).any():
            rep.append(f"agent masks resampled to the image grid in {int((b.resampled.astype(str).str.len() > 3).sum())} scans")
    bn = d[d.kind == 'baseline_neg']
    if len(bn):
        pn = bn.groupby('fold')[['n_lesion', 'bp_fp_ml', 'vt_fp_ml', 'agree_fp_ml']].mean(); pn['scans'] = bn.groupby('fold').size()
        pn['scans_with_false_lesion'] = bn.assign(x=bn.n_lesion > 0).groupby('fold').x.mean()
        rep += ['== tumour-free scans: false lesions per scan and false-positive volume (ml) ==', pn.round(3).to_string(), '']
    # 2. lesion level
    L = d[d.kind == 'lesion']
    if len(L):
        K = ['mod', 'cls', 'method']; V_ = ['dice', 'precision', 'recall', 'over', 'under']
        per_site = L.groupby(K + ['fold'])[V_].mean()
        s = per_site.groupby(level=K).mean().round(3); s['sites'] = per_site.groupby(level=K).size(); s['rows'] = L.groupby(K).size()
        s.to_csv(os.path.join(O, 'summary_lesion.csv')); per_site.round(3).to_csv(os.path.join(O, 'summary_lesion_per_site.csv'))
        rep += ['== true lesions: against the expert tumour their anchors touch (per site, then sites equal) ==', s.to_string(), '']
        if 'support' in L and L.support.nunique() > 1:
            sp_ = L.groupby(['support', 'method', 'fold']).dice.mean().groupby(level=['support', 'method']).mean().unstack('method')
            rep += ['lesion Dice by anchor support (two-agent lesions are comparable with an agreement-only run):', sp_.round(3).to_string(), '']
        pv = L.pivot_table(index=['fold', 'ds', 'case', 'lesion', 'mod', 'cls'], columns='method', values='dice')
        for ref in ('xi', 'q_half'):
            q = pv.dropna(subset=['stage2', ref]).assign(diff=lambda x: x['stage2'] - x[ref])
            rep.append(f'go / no-go: stage2 vs {ref} (lesion Dice, paired; sites equal):')
            for (mod, cls), g in q.groupby(level=['mod', 'cls']):
                ps = g.groupby(level='fold')[['stage2', ref, 'diff']].mean(); sw = ps.mean()
                rep.append(f'  {mod} {cls}: stage2 {sw["stage2"]:.3f} vs {ref} {sw[ref]:.3f}; diff {sw["diff"]:+.3f}; better in '
                           f'{(ps["diff"] > 0).sum()} of {len(ps)} sites -> {"GO" if sw["diff"] > 0 else "NO-GO"}')
                rep += [f'      {f}: {r["stage2"]:.3f} vs {r[ref]:.3f} ({r["diff"]:+.3f}; {int(g.xs(f, level="fold").shape[0])} lesions)'
                        for f, r in ps.iterrows()]
            rep.append('')
        ap = L[L.method == 'stage2'].groupby('fold').anchor_precision.agg(['mean', 'median', lambda v: (v < 0.5).mean()])
        ap.columns = ['mean', 'median', 'share_below_0.5']
        rep += ['anchor precision of true lesions (share of the anchor inside expert tumour):', ap.round(3).to_string(), '']
    # 3. scan level (false lesions included): Dice from summed true / false positive volume
    sp_ = d[d.kind == 'scan_part']
    if len(sp_) and len(b):
        # every tumour scan counts: a scan whose tumours all went undetected has no scan_part rows (Dice 0 for every method)
        allm = b[b.T_ml > 0][['fold', 'ds', 'case', 'T_ml']].merge(pd.DataFrame(dict(method=list(METHODS))), how='cross')
        g = sp_.groupby(['ds', 'case', 'method'])[['tp_ml', 'fp_ml']].sum().reset_index()
        g = allm.merge(g, on=['ds', 'case', 'method'], how='left').fillna({'tp_ml': 0.0, 'fp_ml': 0.0})
        g['dice'] = 2 * g.tp_ml / (g.tp_ml + g.fp_ml + g.T_ml)
        ps = g.pivot_table(index='fold', columns='method', values='dice', aggfunc='mean')
        ps.loc['(sites equal)'] = ps.mean()
        ps.round(3).to_csv(os.path.join(O, 'summary_scan.csv'))
        rep += ['== scan level: Dice of the union over all lesions (false lesions included; missed tumours count as false negatives). '
                'False lesions beyond --max-false per scan add their stage-1 volume to every method except stage2, which is not run on '
                'them (stage2 is favoured where they are many; see "false lesions not run") ==',
                ps.round(3).to_string(), '']
    # 4. tumour beyond every outline, and stage-1 misses
    st = L[L.method == 'stage2'] if len(L) else L
    bcols = [f'{lo:g}_{hi:g}' for lo, hi in V.BANDS]
    if len(st) and any(f'miss_{x}_ml' in st for x in bcols):
        rep.append('recall beyond the candidate extents (recalled / missed volume per site, then sites averaged):')
        for x in bcols:
            if f'miss_{x}_ml' not in st:
                continue
            g = st.groupby('fold')[[f'miss_{x}_ml', f'rec_{x}_ml']].sum(); g = g[g[f'miss_{x}_ml'] > 0]
            if len(g):
                rr_ = g[f'rec_{x}_ml'] / g[f'miss_{x}_ml']
                rep.append(f'  {x} mm: {rr_.mean():.1%} over {len(g)} sites (missed {g[f"miss_{x}_ml"].sum():.0f} ml); '
                           + '; '.join(f'{k} {v:.0%}' for k, v in rr_.items()))
        tot = st.groupby('fold').apply(lambda x: sum(x[f'miss_{c}_ml'].fillna(0).sum() for c in bcols if f'miss_{c}_ml' in x)
                                       / max(x.G_ml.sum(), 1e-9), include_groups=False)
        rep.append('  share of lesion tumour volume beyond every candidate extent: ' + '; '.join(f'{k} {v:.1%}' for k, v in tot.items()))
    ms = d[d.kind == 'missed']
    if len(ms):
        g = ms.groupby('fold').agg(tumours=('ml', 'size'), ml=('ml', 'sum'), rec_ml=('rec_ml', 'sum'))
        g['recalled'] = g.rec_ml / g.ml
        rep += ['expert tumours >= 0.5 ml with no anchor (stage-1 misses) and the share stage 2 marks as tumour:', g.round(3).to_string()]
    rep.append('')
    # 5. Bayes factor: true lesions should not be flagged, false lesions should
    def rate(x):
        x = x.dropna(subset=['log_bf_per_vox'])
        if not len(x):
            return None
        per = x.groupby('fold').log_bf_per_vox.apply(lambda v: (v < 0).mean())
        return per, x.groupby('fold').size()
    fa = d[d.kind == 'false_anchor']
    for lab, x in (('true lesions flagged (should be low)', st),
                   ('false lesions in tumour scans flagged (should be high)', fa[fa['where'] != 'tumour_free_scan'] if len(fa) else fa),
                   ('false lesions in tumour-free scans flagged (should be high)', fa[fa['where'] == 'tumour_free_scan'] if len(fa) else fa)):
        r = rate(x) if len(x) else None
        if r is not None:
            per, n = r
            rep.append(f'{lab}: {per.mean():.1%} over {len(per)} sites; ' + '; '.join(f'{k} {v:.0%} of {n[k]}' for k, v in per.items()))
    tv = st.log_bf_per_vox.dropna() if len(st) else pd.Series(dtype=float)
    fv = fa.log_bf_per_vox.dropna() if len(fa) else pd.Series(dtype=float)
    if len(tv) and len(fv):
        from scipy.stats import mannwhitneyu
        auc = mannwhitneyu(tv, fv).statistic / (len(tv) * len(fv))
        rep.append(f'log BF per voxel: true median {tv.median():.4f}, false median {fv.median():.4f}; AUC (true > false) {auc:.3f}')
    nr = d[d.kind == 'false_not_run']
    if len(nr):
        rep.append(f'false lesions not run (beyond --max-false per scan): {int(nr.n.sum())}')
    cc = V._cat(O, 'calibration')
    if cc is not None:
        f_ = cc.groupby(['mod', 'bin', 'fold'])[['n', 'sum_p', 'sum_t']].sum()
        f_['mean_p'] = f_.sum_p / f_.n; f_['freq'] = f_.sum_t / f_.n
        c = f_.groupby(level=['mod', 'bin']).agg(mean_p=('mean_p', 'mean'), freq=('freq', 'mean'), n=('n', 'sum'), sites=('n', 'size'))
        rep += ['', 'calibration of p_v in the tumour window (sites equal):', c.round(3).to_string()]
    okc = set(zip(d.ds[d.kind != 'error'], d.case[d.kind != 'error'].astype(str)))
    errs = d[(d.kind == 'error') & ~pd.Series([k in okc for k in zip(d.ds, d.case.astype(str))], index=d.index)]   # retried and passed: not failed
    if len(errs):
        rep.append(f'{len(errs)} scans failed (rows*.csv, kind == error)')
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
