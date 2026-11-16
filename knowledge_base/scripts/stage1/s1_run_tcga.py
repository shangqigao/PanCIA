"""Stage 1 v8.6 on TCGA (16 Nov 2026): scripts/stage1/g3_run_v86.py on the TCGA planner / agent outputs, for stage 2 and the external
check (overall plan step H). TCGA data already follow the layout g3 expects (TCGA_Seg/<Agent>/Radiology/<CT|MR>/<cohort>/...,
TCGA_NIFTI/Radiology/...); this runner builds a view of symlinks and points HOME at it, as s1_run_r2seg.py does:
  <view>/mnt/PanCIA -> PanCIA      <view>/mnt/clinical/PanCIA_outputs -> outputs root (~/Datasets/PanCIA_outputs)
  <view>/mnt/TCGA_Seg -> TCGA_Seg  <view>/mnt/TCGA_NIFTI -> TCGA_NIFTI
Series: --set pilot (default): the 152 series of ledger/v84_pilot (stage-2 TCGA smoke and pilot); --set all: every series with a
VoxTell plan manifest. Outputs: ledger/v86_tcga_<set>/{anchors,lesions,series}[_s<i>].csv, cache/ (panel slices). Resumable.
Usage: python s1_run_tcga.py [--set pilot|all] [--only <substring>,...] [--shards 4]"""
import os, sys, csv, glob, time, argparse, subprocess, importlib.util, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, HERE)
from pancia_kb.stage2 import paths   # noqa: E402
from s1_run_r2seg import link          # noqa: E402


def build_view(view, real):
    link(os.path.join(view, 'mnt', 'PanCIA'), real['PanCIA'])
    cl = os.path.join(view, 'mnt', 'clinical')
    if os.path.islink(cl):
        os.remove(cl)
    os.makedirs(cl, exist_ok=True)
    link(os.path.join(cl, 'PanCIA_outputs'), real['outputs'])
    link(os.path.join(view, 'mnt', 'TCGA_Seg'), real['TCGA_Seg'])
    link(os.path.join(view, 'mnt', 'TCGA_NIFTI'), real['TCGA_NIFTI'])


def series_list(real, which):
    if which == 'pilot':
        P = os.path.join(real['outputs'], 'ledger', 'v84_pilot')
        rels = set()
        for f in glob.glob(os.path.join(P, 'series*.csv')):
            rels |= {r['rel'] for r in csv.DictReader(open(f))}
        return sorted(rels)
    root = os.path.join(real['TCGA_Seg'], 'VoxTell', 'Radiology')
    return sorted(os.path.relpath(f, root)[:-len('_plan_manifest.json')] for f in glob.glob(os.path.join(root, '*', '*', '*', '*_plan_manifest.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='pilot', choices=('pilot', 'all')); ap.add_argument('--only', default=None)
    ap.add_argument('--shards', type=int, default=0)
    ap.add_argument('--view', default=os.path.join(os.path.expanduser('~'), '.pancia_tcga_view'))
    a = ap.parse_args()
    real = {nm: os.path.realpath(paths.root(nm)) for nm in ('PanCIA', 'outputs', 'TCGA_Seg', 'TCGA_NIFTI')}
    out_dir = os.path.join(real['outputs'], 'ledger', f'v86_tcga_{a.set}')
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    build_view(a.view, real)
    if a.shards:
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args,
                               env=dict(os.environ, SHARD=f'{i}/{a.shards}', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                               stdout=open(os.path.join(out_dir, 'logs', f's1_tcga_s{i}.txt'), 'a'), stderr=subprocess.STDOUT)
              for i in range(a.shards)]
        try:
            for p in ps:
                p.wait()
        except KeyboardInterrupt:
            for p in ps:
                p.terminate()
        print('done; logs in', os.path.join(out_dir, 'logs')); return
    rels = series_list(real, a.set)
    if a.only:
        rels = [r for r in rels if any(s in r for s in a.only.split(','))]
    os.environ['HOME'] = a.view
    os.environ['V8OUT'] = out_dir
    spec = importlib.util.spec_from_file_location('g3_run_v86', os.path.join(HERE, 'g3_run_v86.py'))
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    done = set()
    for f in glob.glob(os.path.join(out_dir, 'series*.csv')):
        done |= {r['rel'] for r in csv.DictReader(open(f)) if not str(r.get('note', '')).startswith('error')}
    t0 = time.time(); n = 0
    for k, rel in enumerate(rels):
        if k % G.SN != G.SI or rel in done:
            continue
        try:
            rows, lrows, srow = G.run(rel, k)
            G.out('anchors', rows); G.out('lesions', lrows); G.out('series', [srow])
        except Exception:
            G.out('series', [dict(rel=rel, note='error: ' + traceback.format_exc()[-400:].replace('\n', ' | '), k=k)])
        n += 1
        print(f'{n} {rel} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
