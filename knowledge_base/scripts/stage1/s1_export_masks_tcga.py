"""Export stage-1 anchor masks for TCGA (23 Nov 2026), so that the dashboard's former "ledger" measurements can be redone on stage 1.
s1_run_tcga.py did not save masks (only 2-D panel slices in cache/). This rebuilds each series' candidate set with the same frozen
code and inputs (g3_run_v86 -> ledger_v86.build_ledger, same symlink view), checks that anchor ids and volumes match the stage-1
anchors csv, and saves the accepted and uncertain anchors (xi = anchor, adopted 23 Nov) with their stage-1 decision and host class.
No decision is re-taken here.
Output: ledger/v86_tcga_all_bp2/masks/<rel>.npz: meta (json: rel, shape, spacing, anchors {id: lesion, decision, host_cls, kind,
agents, ml, box}), affine, a<id> (packed bits of the anchor crop). Series with no accepted/uncertain anchor get an npz with no anchors.
Usage: [SHARD=i/n] python s1_export_masks_tcga.py [--only ...] [--budget SECONDS]  |  python s1_export_masks_tcga.py --shards 4
       python s1_export_masks_tcga.py --check   (counts, mismatches)"""
import os, sys, csv, glob, json, time, argparse, subprocess, importlib.util, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from pancia_kb.stage2 import paths   # noqa: E402
TAG = 'v86_tcga_all_bp2'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=None); ap.add_argument('--shards', type=int, default=0)
    ap.add_argument('--budget', type=float, default=0, help='stop after this many seconds (resumable)'); ap.add_argument('--check', action='store_true')
    ap.add_argument('--view', default=os.path.join(os.path.expanduser('~'), '.pancia_tcga_view_bp2'))
    a = ap.parse_args()
    real_out = os.path.realpath(paths.root('outputs')); L = os.path.join(real_out, 'ledger', TAG); OUT = os.path.join(L, 'masks')
    if a.shards:
        os.makedirs(os.path.join(L, 'logs'), exist_ok=True)
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args,
                               env=dict(os.environ, SHARD=f'{i}/{a.shards}', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                               stdout=open(os.path.join(L, 'logs', f'export_masks_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        for p in ps:
            p.wait()
        print('done'); return
    import pandas as pd
    A = pd.concat([pd.read_csv(f, usecols=['rel', 'id', 'lesion', 'kind', 'agents', 'ml', 'decision', 'host_cls']) for f in sorted(glob.glob(L + '/anchors_s*.csv'))])
    Sr = pd.concat([pd.read_csv(f, usecols=['rel', 'k', 'note']) for f in sorted(glob.glob(L + '/series_s*.csv'))])
    if a.check:
        done = {os.path.relpath(f, OUT)[:-4] for f in glob.glob(OUT + '/**/*.npz', recursive=True)}
        bad = []
        for f in glob.glob(L + '/logs/export_masks_s*.txt'):
            bad += [l for l in open(f) if l.startswith('MISMATCH') or l.startswith('ERROR')]
        print(f'{len(done)} / {Sr.rel.nunique()} series exported; {len(bad)} mismatch/error lines'); print(''.join(bad[:20])); return
    keep = A[A.decision.isin(['accepted', 'accepted_override', 'uncertain'])]
    by_rel = {r: g for r, g in keep.groupby('rel')}
    allml = {r: g.set_index('id').ml for r, g in A.groupby('rel')}
    rels = sorted(Sr.rel.unique())
    if a.only:
        rels = [r for r in rels if any(s in r for s in a.only.split(','))]
    os.environ['HOME'] = a.view; os.environ['V8OUT'] = os.path.join(L, '_export_scratch')
    spec = importlib.util.spec_from_file_location('g3_run_v86', os.path.join(HERE, 'g3_run_v86.py'))
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    import numpy as np
    t0 = time.time(); n = 0
    for k, rel in enumerate(rels):
        if k % G.SN != G.SI:
            continue
        f = os.path.join(OUT, rel + '.npz')
        if os.path.exists(f):
            continue
        if a.budget and time.time() - t0 > a.budget:
            print('budget reached', flush=True); break
        try:
            g = by_rel.get(rel)
            meta = dict(rel=rel, anchors={}); arr = {}
            if g is None:
                aff = np.eye(4); meta['shape'] = None
            else:
                led, cr, aff, shp = G.L8.build_ledger(rel, G.S, G.KB, params=dict(aq_cache=f'{G.CKD}/xaq{G.SUF}.pkl', lobe_cache=f'{G.CKD}/xlobe{G.SUF}.pkl'))
                anc = cr.get('anchors') or {}
                vml = float(np.prod(led['spacing'])) / 1000.0
                mis = [(i, round(float(np.asarray(anc[i]['crop'], bool).sum() * vml), 2), allml[rel].get(i)) for i in anc
                       if i in allml[rel].index and abs(float(np.asarray(anc[i]['crop'], bool).sum() * vml) - allml[rel][i]) > max(0.06, 0.02 * allml[rel][i])]
                if len(anc) != len(allml[rel]) or mis:
                    print(f'MISMATCH {rel} n_led={len(anc)} n_csv={len(allml[rel])} {mis[:3]}', flush=True)
                meta.update(shape=list(shp), spacing=[float(x) for x in led['spacing']])
                for r in g.itertuples():
                    an = anc.get(int(r.id))
                    if an is None:
                        continue
                    arr[f'a{int(r.id)}'] = np.packbits(np.asarray(an['crop'], bool).ravel())
                    meta['anchors'][str(int(r.id))] = dict(lesion=int(r.lesion), decision=r.decision, host_cls=r.host_cls, kind=r.kind, agents=r.agents,
                                                           ml=float(r.ml), box=[[int(s.start), int(s.stop)] for s in an['sl']])
            os.makedirs(os.path.dirname(f), exist_ok=True)
            np.savez_compressed(f[:-4] + '.tmp.npz', meta=json.dumps(meta), affine=np.asarray(aff, float), **arr); os.replace(f[:-4] + '.tmp.npz', f)
        except Exception:
            print('ERROR', rel, traceback.format_exc()[-300:].replace('\n', ' | '), flush=True)
        n += 1
        print(f'{n} {rel} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
