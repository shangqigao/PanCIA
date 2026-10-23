"""Stage 2, step 1: check the stage-1 -> stage-2 inputs on the TCGA pilot (no stage-2 model is run).

Per lesion passed on (accepted / uncertain anchors): anchors kept, q_v on each anchor (minimum; < 1 where some candidate extents do
not contain the whole anchor, e.g. U - non-host or an outline that cuts the anchor), chosen extents, credible set, region volume (needs r: from the fit if present, else --r), cluster membership; per series:
P(h|L) and S recomputed vs the stored v84_pilot rows.
Outputs stage2/check_inputs/{lesions,clusters,checks}[_s<i>].csv and a short summary.
Usage: [SHARD=i/n] python3 s2_check_inputs.py [--r 20] [--only substr]"""
import os, sys, json, glob, argparse, traceback, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import KnowledgeBase, grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, interface as IF   # noqa: E402
from s2_run_tcga import append, name   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--r', type=float, default=None); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir('check_inputs')
    r = a.r if a.r is not None else (json.load(open(paths.fit_path()))['r_mm'] if os.path.exists(paths.fit_path()) else 20.0)
    B = paths.ledger_dir(); P = os.path.join(B, 'v84_pilot')
    fit6 = json.load(open(os.path.join(B, 'v8', 'testbed', 'fit_host6.json')))
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P, 'anchors_s*.csv')))], ignore_index=True)
    ser = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P, 'series_s*.csv')))], ignore_index=True)
    ser = ser[(ser.n_acc + ser.n_unc) > 0].sort_values('rel')
    rels = [x for i, x in enumerate(ser.rel) if i % SN == SI]
    if a.only:
        rels = [x for x in rels if any(s in x for s in a.only.split(','))]
    S1 = IF.Stage1(KB, fit6['beta_UD'], G2.Psi(fit6['psi']), rows, paths.root('TCGA_Seg'), paths.root('TCGA_NIFTI'))
    fd = os.path.join(O, f'done{SUF}.txt'); done = set()
    for f_ in glob.glob(os.path.join(O, 'done*.txt')):                 # all shards
        done |= set(open(f_).read().splitlines())
    t0 = time.time()
    for k, rel in enumerate(rels):
        if rel in done:
            continue
        try:
            s1 = S1.series(rel)
            if s1 is not None and s1['lesions']:
                append(os.path.join(O, f'checks{SUF}.csv'), s1['checks'])
                groups, regs = IF.clusters(s1['lesions'], s1['organs'], s1['body'], r, s1['shape'], s1['sp'])
                vox_ml = float(np.prod(s1['sp'])) / 1000
                cl_of = {i: j for j, g in enumerate(groups) for i in g}
                lr = []
                for i, L in enumerate(s1['lesions']):
                    q = L['q'].crop
                    lr.append(dict(rel=rel, lesion=L['lesion'], cluster=cl_of[i], cluster_size=len(groups[cl_of[i]]),
                                   n_anchors=len(L['anchor_ids']), decisions=json.dumps(L['decisions']), S=json.dumps(L['S_names']),
                                   h_map=L['h_map'], anchor_ml=round(sum(int(x.crop.sum()) for x in L['anchors']) * vox_ml, 2),
                                   xi_ml=round(int(L['xi'].crop.sum()) * vox_ml, 2), q_pos_ml=round(int((q > 0).sum()) * vox_ml, 2),
                                   q_anchor_min=json.dumps({str(i_): e['q_anchor_min'] for i_, e in L['ext'].items()}),
                                   chosen=json.dumps({str(i_): e['chosen'] for i_, e in L['ext'].items()}),
                                   roi_ml=round(int(regs[i].crop.sum()) * vox_ml, 1), insufficient=s1['insufficient']))
                append(os.path.join(O, f'lesions{SUF}.csv'), lr)
                append(os.path.join(O, f'clusters{SUF}.csv'), [dict(rel=rel, cluster=j, size=len(g)) for j, g in enumerate(groups)])
        except Exception:
            append(os.path.join(O, f'errors{SUF}.csv'), [dict(rel=rel, err=traceback.format_exc()[-600:].replace('\n', ' | '))])
        open(fd, 'a').write(rel + '\n')
        print(f'{k + 1}/{len(rels)} {rel} {time.time() - t0:.0f}s', flush=True)
    summarise(O)


def summarise(O):
    fs = glob.glob(os.path.join(O, 'lesions*.csv'))
    if not fs:
        return
    L = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    C = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(O, 'checks*.csv'))], ignore_index=True)
    qmin = [v for s in L.q_anchor_min for v in json.loads(s).values() if v is not None]
    rep = [f'lesions passed on: {len(L)} in {L.rel.nunique()} series; anchors {L.n_anchors.sum()}',
           f'q on anchors: < 1 somewhere on the anchor for {np.mean(np.array(qmin) < 0.999):.1%} of anchors (some candidate extents do not contain the whole anchor)',
           f'clusters: {L.groupby("rel").cluster.nunique().sum()}; lesions in clusters of size > 1: {(L.cluster_size > 1).mean():.1%}; '
           f'largest cluster {L.cluster_size.max()}',
           f'region volume ml: median {L.roi_ml.median():.0f}, 90th pct {L.roi_ml.quantile(0.9):.0f}, max {L.roi_ml.max():.0f}',
           f'P(h|L) recomputed = stored: {C.phl_match.mean():.1%} of anchors; S equal: {C.S_match.mean():.1%}']
    open(os.path.join(O, 'summary.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


if __name__ == '__main__':
    main()
