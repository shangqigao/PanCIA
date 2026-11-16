"""Stage 1 v8.4 (frozen) on R2Seg (11 Nov 2026): ledger_v84.build_ledger + g3_run_v84.run (scripts/stage1/g3_run_v84.py, a byte-identical
copy of clinical/PanCIA_outputs/scripts/g3_run_v84.py, 12 Nov), UNCHANGED, on the R2Seg planner plans and
VoxTell plan-call outputs (run_r2seg_plan.sh + m_tumor_segmentation.py --planner_dir). Decisions accepted / uncertain / rejected per anchor.

The frozen g3_run_v84.py reads fixed paths below ~ (~/mnt/{PanCIA, clinical, TCGA_Seg, TCGA_NIFTI}). This runner builds a view of
symlinks and points HOME at it before importing the module, so nothing in stage 1 is edited:
  <view>/mnt/PanCIA -> PanCIA            <view>/mnt/clinical -> clinical
  <view>/mnt/TCGA_Seg/<Agent>/Radiology/<CT|MR>/<collection> -> R2Seg_Seg/<Agent>/<collection>   (Agent: Planner, VoxTell, BiomedParse, TotalSegmentator)
  <view>/mnt/TCGA_NIFTI/<collection> -> R2Seg/<collection>/images    (the plan manifest's img_name is '<collection>/<stem>')
Series id rel = '<CT|MR>/<collection>/<stem>' (g3_run_v84 reads the modality from rel[:2] and the cohort from rel.split('/')[1]).
Outputs (as v84_pilot): clinical/PanCIA_outputs/ledger/v84_r2seg/{anchors,lesions,series}[_s<i>].csv and cache/ (panel slices).
Scans without a VoxTell plan manifest are skipped (listed in the log).
Extents (12 Nov, user: decisions from frozen v8.4, extents recomputed with the refitted psi): stage 1 is not changed; after each scan the
runner also saves every anchor's mask and candidate extents (name, ml, log volume ratio, flags, box + packed mask) and its lesion from the
ledger checkpoint, to ledger/v84_r2seg/extents/<collection>/<stem>.npz, so that xi and q can be recomputed with any psi later.
Usage: [SHARD=i/n] python s1_run_r2seg.py [--collections Kidney_Tumor_00,...] [--only case_00000,...] [--view <dir>]   (resumable)
       python s1_run_r2seg.py --shards 6     (runs 6 shards in parallel, low priority, and waits)
       --version v85 (15 Nov): ledger_v85 (F1, G1, G3) via g3_run_v85.py; outputs ledger/v85_r2seg/
       --version v86 (15 Nov): g3_run_v86.py (cascade: existence test at level 1 only; v8.6 decision table, R1, no override); ledger/v86_r2seg/"""
import os, sys, csv, time, glob, argparse, subprocess, importlib.util, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

AGENTS = ('Planner', 'VoxTell', 'BiomedParse', 'TotalSegmentator')


def link(dst, src):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) and os.readlink(dst) == src:
        return
    if os.path.lexists(dst):
        os.remove(dst)                                               # a stale symlink inside the view (never a data file)
    try:
        os.symlink(src, dst)
    except FileExistsError:                                          # another shard made it at the same moment
        if os.readlink(dst) != src:
            raise


def build_view(view, real, meta):
    link(os.path.join(view, 'mnt', 'PanCIA'), real['PanCIA'])
    cl = os.path.join(view, 'mnt', 'clinical')                       # 16 Nov: outputs moved out of OneDrive; the frozen modules read
    if os.path.islink(cl):                                           # ~/mnt/clinical/PanCIA_outputs, so the view's clinical holds only that link
        os.remove(cl)
    os.makedirs(cl, exist_ok=True)
    link(os.path.join(cl, 'PanCIA_outputs'), real['outputs'])
    for coll, mod in sorted({(r['img_name'].split('/')[0], r['modality']) for r in meta}):
        for ag in AGENTS:
            link(os.path.join(view, 'mnt', 'TCGA_Seg', ag, 'Radiology', mod, coll), os.path.join(real['R2Seg_Seg'], ag, coll))
        link(os.path.join(view, 'mnt', 'TCGA_NIFTI', coll), os.path.join(real['R2Seg'], coll, 'images'))


def save_extents(G, rel, path):
    """anchor masks, candidate extents and lesions of one scan from g3_run_v84's ledger checkpoint (built by run() for this rel)"""
    import pickle, json, numpy as np
    d = pickle.load(open(os.path.join(G.CKD, f'led{G.SUF}.pkl'), 'rb'))
    if d.get('rel') != rel:
        raise RuntimeError(f'ledger checkpoint is for {d.get("rel")}, not {rel}')
    led, cr, aff, shp = d['v']
    arr, meta = {}, dict(rel=rel, shape=list(shp), spacing=led['spacing'], anchors={}, lesions=[])
    box = lambda sl: [[int(s.start), int(s.stop)] for s in sl]
    for i, an in (cr.get('anchors') or {}).items():
        arr[f'a{i}'] = np.packbits(np.asarray(an['crop'], bool).ravel())
        ex = []
        for j, e in enumerate(an.get('extents') or []):
            arr[f'a{i}_e{j}'] = np.packbits(np.asarray(e['crop'], bool).ravel())
            ex.append(dict(name=e.get('name'), ml=e.get('ml'), logratio=e.get('logratio'), flags=e.get('flags', []), box=box(e['sl'])))
        meta['anchors'][str(i)] = dict(kind=an.get('kind'), box=box(an['sl']), extents=ex)
    for li, Lz in enumerate(cr.get('lesions') or []):
        meta['lesions'].append(dict(lesion=li, anchors=[int(x) for x in Lz['anchors']]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path + '.tmp.npz', meta=json.dumps(meta), affine=np.asarray(aff, float), **arr)
    os.replace(path + '.tmp.npz', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=os.path.join(os.path.expanduser('~'), '.pancia_r2seg_view'))
    ap.add_argument('--collections', default=None); ap.add_argument('--only', default=None)
    ap.add_argument('--shards', type=int, default=0, help='launch this many shard processes and wait')
    ap.add_argument('--version', default='v84', choices=('v84', 'v85', 'v86'), help='stage-1 version (v85: F1, G1, G3; v86: + cascade, decision rules built in; 15 Nov)')
    a = ap.parse_args()
    real = {nm: os.path.realpath(paths.root(nm)) for nm in ('PanCIA', 'outputs', 'R2Seg', 'R2Seg_Seg')}
    out_dir = os.path.join(real['outputs'], 'ledger', f'{a.version}_r2seg')
    if a.shards:
        os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
        meta0 = list(csv.DictReader(open(os.path.join(real['outputs'], 'cohort', 'r2seg_scan_meta.csv'))))
        build_view(a.view, real, meta0)                              # once, before the shards start (they raced on it, 12 Nov)
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args,
                               env=dict(os.environ, SHARD=f'{i}/{a.shards}', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                               stdout=open(os.path.join(out_dir, 'logs', f's1_r2seg_s{i}.txt'), 'a'), stderr=subprocess.STDOUT)
              for i in range(a.shards)]
        try:
            for p in ps:
                p.wait()
        except KeyboardInterrupt:
            for p in ps:
                p.terminate()
        print('done; logs in', os.path.join(out_dir, 'logs')); return
    meta = list(csv.DictReader(open(os.path.join(real['outputs'], 'cohort', 'r2seg_scan_meta.csv'))))
    if a.collections:
        meta = [r for r in meta if r['img_name'].split('/')[0] in a.collections.split(',')]
    if a.only:
        meta = [r for r in meta if any(s in r['img_name'] for s in a.only.split(','))]
    build_view(a.view, real, meta)
    os.environ['HOME'] = a.view                                      # the frozen module resolves ~/mnt/... through the view
    os.environ['V8OUT'] = out_dir
    spec = importlib.util.spec_from_file_location(f'g3_run_{a.version}', os.path.join(HERE, f'g3_run_{a.version}.py'))   # repo copy (identical to the outputs one)
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    rels = [f"{r['modality']}/{r['img_name']}" for r in meta]
    done = set()
    for f in glob.glob(os.path.join(out_dir, 'series*.csv')):
        done |= {r['rel'] for r in csv.DictReader(open(f))}
    t0 = time.time(); n = 0
    for k, rel in enumerate(rels):
        if k % G.SN != G.SI or rel in done:
            continue
        mod, coll, stem = rel.split('/', 2)
        if not os.path.exists(os.path.join(real['R2Seg_Seg'], 'VoxTell', coll, stem + '_plan_manifest.json')):
            print('no VoxTell plan manifest:', rel, flush=True); continue
        try:
            rows, lrows, srow = G.run(rel, k)
            if rows:
                save_extents(G, rel, os.path.join(out_dir, 'extents', coll, stem + '.npz'))
            G.out('anchors', rows); G.out('lesions', lrows); G.out('series', [srow])
        except Exception:
            G.out('series', [dict(rel=rel, note='error: ' + traceback.format_exc()[-400:].replace('\n', ' | '), k=k)])
        n += 1
        print(f'{n} {rel} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
