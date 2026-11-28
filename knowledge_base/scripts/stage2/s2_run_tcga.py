"""Stage 2, step 6: run stage 2 on the TCGA pilot (public data; visual and bug checks only, nothing is tuned here).

Per series of ledger/v84_pilot with accepted or uncertain anchors: rebuild stage 1 (pancia_kb.stage2.interface), group lesions whose
regions overlap into clusters, run stage 2 per cluster (pancia_kb.stage2.run.solve_cluster), and write
  lesions[_s<i>].csv     one row per lesion: stage-1 decisions / hosts, volumes (chosen extent, refined tumour), trimmed and recalled
                         ml (recalled by host territory), Dice(refined, extent), mean p over the extent and the refined tumour, log BF and
                         flag (reported only), reported segmentation (stage 2), components, region growth, stage-1 existence
                         test recomputed on the refined tumour under each host of S (alpha 0.05)
  candidates[_s<i>].csv  new candidates (components of p >= 1/2 meeting no anchor): ml, distance to the nearest anchor, organs
  clusters[_s<i>].csv    per cluster: lesions, voxels, iterations, convergence, effective components, time
  checks[_s<i>].csv      P(h|L) / S recomputed vs the stored v84_pilot rows
  maps/<k>/c<j>.npz      window, p (uint8, x255), component map, refined tumours, anchors, chosen extents (for s2_png.py)
Stage 2 runs at the R2Seg validation resolution: a cluster whose crop (anchors and chosen extents + 60 mm) exceeds 3M voxels is
strided in-plane (features.stride_rule, as expert._image_crop) and its maps are put back on the full grid (clusters.csv: stride).
The Bayes factor is reported only (29 Oct): it no longer switches the reported extent to stage 1 (it flagged about 55 % of lesions on
the TCGA smoke, at the same rate for accepted and uncertain ones, and did not separate true from false anchors on R2Seg).
Stage-1 decisions are not changed. Usage: [SHARD=i/n] python3 s2_run_tcga.py [--tag tcga_v1] [--only substr,...] [--no-bf]"""
import os, sys, json, glob, time, argparse, traceback
import numpy as np, pandas as pd
# 23 Nov (step S): PANCIA_FAST_EDT=1 swaps scipy's EDT for the exact `edt` package (as stage 1; ~1e-6 mm differences). Off by default
# here so that a validation run is not mixed across EDT versions.
if os.environ.get('PANCIA_FAST_EDT', '0') == '1':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
    from pancia_kb import fast_edt as _FE
    print('[stage 2] fast EDT:', _FE.install(), flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import KnowledgeBase, grounding_g1 as G1, grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, interface as IF, run as RN, features as F, VERSION   # noqa: E402


def append(f, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(f):
        cols = list(pd.read_csv(f, nrows=0).columns)
        if set(df.columns) - set(cols):
            df = pd.concat([pd.read_csv(f), df], ignore_index=True); df.to_csv(f, index=False); return
        df = df.reindex(columns=cols)
    df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)


def xi_rule_for(a, B):
    """stage-1 outline rule (23 Nov): fit_psi.json xi_rule ('anchor') for v8.6; v8.4 keeps the psi argmax"""
    if a.ledger == 'v84':
        return 'psi'
    return json.load(open(os.path.join(B, 'v85', 'extent', 'fit_psi.json'))).get('xi_rule', 'psi')


def psi_for(a, B):
    """psi for q and xi (16 Nov): refitted psi (ledger/v85/extent/fit_psi.json, pooled over sites: TCGA is external) for stage 1 v8.6;
    the v8.4 psi of fit_host6.json when reproducing a v8.4 run"""
    if a.ledger == 'v84':
        return G2.Psi(json.load(open(os.path.join(B, 'v8', 'testbed', 'fit_host6.json')))['psi'])
    return G2.Psi(json.load(open(os.path.join(B, 'v85', 'extent', 'fit_psi.json')))['psi'])


def stage1_args(ap):
    ap.add_argument('--stage1', default='v86_tcga_pilot', help='stage-1 rows under ledger/ (v8.4: v84_pilot)')
    ap.add_argument('--ledger', default='v86', choices=('v86', 'v84'), help='ledger module used to rebuild stage 1 (match --stage1)')


def name(q):
    from pancia_kb.ledger_v6 import _hk
    return _hk(q) if isinstance(q, tuple) else str(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='tcga_v1'); ap.add_argument('--only', default=None)
    ap.add_argument('--no-bf', action='store_true'); ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--no-evidence', action='store_true'); stage1_args(ap)
    ap.add_argument('--fit', default=None, help='24 Nov: stage-2 fit JSON under knowledge_base/knowledge_base/stage2/; default for v8.6: '
                    'stage2_fit_agent_beyond.json (true-lesion prior with the beyond-xi term, refitted on bp2 23 Nov); v8.4: stage2_fit.json')
    ap.add_argument('--uncertain-only', action='store_true', help='24 Nov: diagnostic, report-only run: lesions with an uncertain anchor and '
                    'no accepted anchor (their uncertain anchors); use its own --tag; never mixed with the accepted-only results')
    ap.add_argument('--accepted-only', action='store_true', help='23 Nov: OOD sets (user, 16 Nov): only lesions with an accepted anchor, '
                    'and their accepted anchors; uncertain lesions are not passed to stage 2')
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir(a.tag); os.makedirs(os.path.join(O, 'maps'), exist_ok=True)
    fitf = paths.fit_path(a.fit or ('stage2_fit_agent_beyond.json' if a.ledger == 'v86' else 'stage2_fit.json'))
    fit = json.load(open(fitf))
    B = paths.ledger_dir(); P = os.path.join(B, a.stage1)                     # 16 Nov: stage-1 rows (default v86_tcga_pilot)
    fit6 = json.load(open(os.path.join(B, 'v8', 'testbed', 'fit_host6.json')))
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P, 'anchors*.csv')))], ignore_index=True)
    ser = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P, 'series*.csv')))], ignore_index=True)
    if a.accepted_only and a.uncertain_only:
        sys.exit('--accepted-only and --uncertain-only exclude each other')
    ser = ser[(ser.n_acc > 0) if a.accepted_only else ((ser.n_unc > 0) if a.uncertain_only else ((ser.n_acc + ser.n_unc) > 0))].sort_values('rel')
    rels = [r for i, r in enumerate(ser.rel) if i % SN == SI]
    if a.only:
        rels = [r for r in rels if any(s in r for s in a.only.split(','))]
    TAB1 = G1.Tables1(os.path.join(B, 'v6', 'grounding', 'g0', 'h5', 'tables_g1.json'))
    TS = G2.TestSets(os.path.join(B, 'v8', 'grounding', 'test_sets.json'))
    S1 = IF.Stage1(KB, fit6['beta_UD'], psi_for(a, B), rows, paths.root('TCGA_Seg'), paths.root('TCGA_NIFTI'), ledger=a.ledger, xi_rule=xi_rule_for(a, B),
                  keep=('accepted', 'accepted_override') if a.accepted_only else (('uncertain',) if a.uncertain_only else IF.KEEP),
                  skip_if=('accepted', 'accepted_override') if a.uncertain_only else ())
    fd = os.path.join(O, f'done{SUF}.txt'); done = set()
    for f_ in glob.glob(os.path.join(O, 'done*.txt')):                 # all shards
        done |= set(open(f_).read().splitlines())
    from pancia_kb.stage2 import vb as _vb
    _g = _vb.gpu()
    if os.environ.get('PANCIA_GPU', '0') == '1' and _g is None:     # 24 Nov: asked for the GPU but none usable -> stop, no silent CPU run
        sys.exit('[stage 2] PANCIA_GPU=1 but no usable GPU / CuPy here (see the [vb] line above); not running on the CPU. '
                 'Run inside a GPU job, or unset PANCIA_GPU for a CPU run.')
    json.dump(dict(version=VERSION, fit=fitf, fit_date=fit.get('date'), r_mm=fit['r_mm'], stage1=a.stage1, accepted_only=a.accepted_only, uncertain_only=a.uncertain_only,
                   bayes_factor=not a.no_bf, gpu=_g is not None, fast_edt=os.environ.get('PANCIA_FAST_EDT', '0') == '1'),
              open(os.path.join(O, f'run_info{SUF}.json'), 'w'))
    print('[stage 2] fit', os.path.basename(fitf), '| VB on', 'GPU' if _g is not None else 'CPU', '|', len(rels), 'series in this shard', flush=True)
    t0 = time.time()
    for k, rel in enumerate(rels):
        if rel in done:
            continue
        kidx = int(ser.set_index('rel').loc[rel, 'k']) if 'k' in ser.columns else k
        try:
            run_series(rel, kidx, S1, fit, O, SUF, a, TAB1, TS)
        except Exception:
            # 24 Nov: a failed series is NOT marked done, so a resubmission retries it (an environment error, e.g. a missing CUDA
            # library, would otherwise mark every series done)
            append(os.path.join(O, f'errors{SUF}.csv'), [dict(rel=rel, err=traceback.format_exc()[-600:].replace('\n', ' | '))])
            print(f'{k + 1}/{len(rels)} {rel} ERROR (see errors{SUF}.csv)', flush=True)
            continue
        open(fd, 'a').write(rel + '\n')
        print(f'{k + 1}/{len(rels)} {rel} {time.time() - t0:.0f}s', flush=True)


def run_series(rel, kidx, S1, fit, O, SUF, a, TAB1, TS):
    s1 = S1.series(rel)
    if s1 is None or not s1['lesions']:
        return
    append(os.path.join(O, f'checks{SUF}.csv'), s1['checks'])
    X, cen, scl, mod = s1['X'], s1['cen'], s1['scl'], s1['mod']
    img = lambda W: F.normalise(X[W], mod, cen, scl)
    groups, _ = IF.clusters(s1['lesions'], s1['organs'], s1['body'], float(fit['r_mm']), s1['shape'], s1['sp'])
    md = os.path.join(O, 'maps', str(kidx)); os.makedirs(md, exist_ok=True)
    vox_ml = float(np.prod(s1['sp'])) / 1000.0
    for j, g in enumerate(groups):
        Ls = [s1['lesions'][i] for i in g]
        t0 = time.time()
        # stage 2 at the resolution it was validated at: the R2Seg stride rule on the cluster's crop (interface.stride_cluster);
        # maps are put back on the full grid, so the stage-1 extents, the existence test and the panels are unchanged
        g_ = IF.stride_cluster(s1, Ls)
        res = RN.solve_cluster(g_['img'], g_['lesions'], g_['organs'], fit, g_['sp'], mod, g_['thick'], g_['ax'], g_['body'], g_['shape'],
                               bayes=not a.no_bf, max_iter=a.max_iter, seq=s1['seq'])
        res = IF.upsample_result(res, g_['fac'], s1['shape'])
        W = res['W']
        np.savez_compressed(os.path.join(md, f'c{j}.npz'), rel=rel, W=np.array([[s.start, s.stop] for s in W]),
                            p=np.round(res['p'] * 255).astype(np.uint8), comp=res['comp'], ROI=np.packbits(res['ROI']),
                            ZL=np.packbits(np.array(res['ZL'])), anchors=np.packbits(np.array(res['anchors'])),
                            xi=np.packbits(np.array([RN.win(L['xi'], W) for L in Ls])), keys=np.array([L['key'] for L in Ls]),
                            shape=np.array(res['ROI'].shape), n_les=len(Ls))
        lrows = []
        for L, r_, ZL in zip(Ls, res['lesions'], res['ZL']):
            ev = {}
            if not a.no_evidence and ZL.any():
                try:
                    ev = IF.existence_on_mask(s1, RN.Patch(W, ZL), L, TAB1, TS)
                except Exception as ex:
                    ev = {'error': repr(ex)[:80]}
            flag = r_.get('flag_bf')
            lrows.append(dict(rel=rel, cancer=rel.split('/')[1], mod=mod, seq=s1['seq'], k=kidx, cluster=j, lesion=L['lesion'], anchors=json.dumps(L['anchor_ids']),
                              decisions=json.dumps(L['decisions']), kinds=json.dumps(L['kinds']), S=json.dumps(L['S_names']),
                              PhL_S=json.dumps({n_: round(L['PhL'][q], 4) for n_, q in zip(L['S_names'], L['S'])}), h_map=L['h_map'],
                              tum_cls=L['tum_cls'], extents=json.dumps(L['ext']),
                              **{k_: (json.dumps({name(q): v for q, v in val.items()}) if isinstance(val, dict) else val) for k_, val in r_.items() if k_ != 'key'},
                              reported='stage2', bf_note='prior-only (log BF < 0)' if flag else None, evidence_on_Z=json.dumps(ev),
                              n_grow=res['n_grow'], reached_max_grow=res['reached_max_grow']))
        append(os.path.join(O, f'lesions{SUF}.csv'), lrows)
        append(os.path.join(O, f'candidates{SUF}.csv'),
               [dict(rel=rel, k=kidx, cluster=j, organs=json.dumps([name(q) for q in c['organs']]), ml=c['ml'], d_anchor_mm=c['d_anchor_mm'])
                for c in res['new_candidates']])
        md_ = res['model']
        append(os.path.join(O, f'clusters{SUF}.csv'), [dict(rel=rel, k=kidx, cluster=j, lesions=json.dumps([L['key'] for L in Ls]),
                                                            n_vox=md_['n_vox'], iters=md_['iters'], converged=md_['converged'],
                                                            k_eff=md_['k_eff'], rho=json.dumps(md_['rho']), comp_means=json.dumps(md_['comp_means']),
                                                            organs=json.dumps([name(q) for q in md_['organs']]), mean_Ew=md_['mean_Ew'],
                                                            n_new=len(res['new_candidates']), n_grow=res['n_grow'], sec=round(time.time() - t0, 1),
                                                            stride=json.dumps(res['stride']),
                                                            vox_ml=vox_ml)])


if __name__ == '__main__':
    main()
