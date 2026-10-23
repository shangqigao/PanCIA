"""Stage 2, step 6: run stage 2 on the TCGA pilot (public data; visual and bug checks only, nothing is tuned here).

Per series of ledger/v84_pilot with accepted or uncertain anchors: rebuild stage 1 (pancia_kb.stage2.interface), group lesions whose
regions overlap into clusters, run stage 2 per cluster (pancia_kb.stage2.run.solve_cluster), and write
  lesions[_s<i>].csv     one row per lesion: stage-1 decisions / hosts, volumes (chosen extent, refined tumour), trimmed and recalled
                         ml (recalled by host territory), Dice(refined, extent), mean p over the extent and the refined tumour, log BF and
                         flag, reported segmentation (stage-1 extent when flagged), components, region growth, stage-1 existence
                         test recomputed on the refined tumour under each host of S (alpha 0.05)
  candidates[_s<i>].csv  new candidates (components of p >= 1/2 meeting no anchor): ml, distance to the nearest anchor, organs
  clusters[_s<i>].csv    per cluster: lesions, voxels, iterations, convergence, effective components, time
  checks[_s<i>].csv      P(h|L) / S recomputed vs the stored v84_pilot rows
  maps/<k>/c<j>.npz      window, p (uint8, x255), component map, refined tumours, anchors, chosen extents (for s2_png.py)
Stage-1 decisions are not changed. Usage: [SHARD=i/n] python3 s2_run_tcga.py [--tag tcga_v1] [--only substr,...] [--no-bf]"""
import os, sys, json, glob, time, argparse, traceback
import numpy as np, pandas as pd
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


def name(q):
    from pancia_kb.ledger_v6 import _hk
    return _hk(q) if isinstance(q, tuple) else str(q)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='tcga_v1'); ap.add_argument('--only', default=None)
    ap.add_argument('--no-bf', action='store_true'); ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--no-evidence', action='store_true')
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir(a.tag); os.makedirs(os.path.join(O, 'maps'), exist_ok=True)
    fit = json.load(open(paths.fit_path()))
    B = paths.ledger_dir(); P = os.path.join(B, 'v84_pilot')
    fit6 = json.load(open(os.path.join(B, 'v8', 'testbed', 'fit_host6.json')))
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P, 'anchors_s*.csv')))], ignore_index=True)
    ser = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P, 'series_s*.csv')))], ignore_index=True)
    ser = ser[(ser.n_acc + ser.n_unc) > 0].sort_values('rel')
    rels = [r for i, r in enumerate(ser.rel) if i % SN == SI]
    if a.only:
        rels = [r for r in rels if any(s in r for s in a.only.split(','))]
    TAB1 = G1.Tables1(os.path.join(B, 'v6', 'grounding', 'g0', 'h5', 'tables_g1.json'))
    TS = G2.TestSets(os.path.join(B, 'v8', 'grounding', 'test_sets.json'))
    S1 = IF.Stage1(KB, fit6['beta_UD'], G2.Psi(fit6['psi']), rows, paths.root('TCGA_Seg'), paths.root('TCGA_NIFTI'))
    fd = os.path.join(O, f'done{SUF}.txt'); done = set()
    for f_ in glob.glob(os.path.join(O, 'done*.txt')):                 # all shards
        done |= set(open(f_).read().splitlines())
    json.dump(dict(version=VERSION, fit=paths.fit_path(), fit_date=fit.get('date'), r_mm=fit['r_mm']), open(os.path.join(O, 'run_info.json'), 'w'))
    t0 = time.time()
    for k, rel in enumerate(rels):
        if rel in done:
            continue
        kidx = int(ser.set_index('rel').loc[rel, 'k']) if 'k' in ser.columns else k
        try:
            run_series(rel, kidx, S1, fit, O, SUF, a, TAB1, TS)
        except Exception:
            append(os.path.join(O, f'errors{SUF}.csv'), [dict(rel=rel, err=traceback.format_exc()[-600:].replace('\n', ' | '))])
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
        res = RN.solve_cluster(img, Ls, s1['organs'], fit, s1['sp'], mod, s1['thick'], s1['ax'], s1['body'], s1['shape'],
                               bayes=not a.no_bf, max_iter=a.max_iter)
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
            lrows.append(dict(rel=rel, cancer=rel.split('/')[1], mod=mod, k=kidx, cluster=j, lesion=L['lesion'], anchors=json.dumps(L['anchor_ids']),
                              decisions=json.dumps(L['decisions']), kinds=json.dumps(L['kinds']), S=json.dumps(L['S_names']),
                              PhL_S=json.dumps({n_: round(L['PhL'][q], 4) for n_, q in zip(L['S_names'], L['S'])}), h_map=L['h_map'],
                              tum_cls=L['tum_cls'], extents=json.dumps(L['ext']),
                              **{k_: (json.dumps({name(q): v for q, v in val.items()}) if isinstance(val, dict) else val) for k_, val in r_.items() if k_ != 'key'},
                              reported='stage1_extent' if flag else 'stage2', evidence_on_Z=json.dumps(ev),
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
                                                            vox_ml=vox_ml)])


if __name__ == '__main__':
    main()
