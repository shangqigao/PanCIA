"""Extent rule on the ledger's own candidates, step 1: rows (23 Nov 2026; claude/extent_rule_refit_plan_2026-11-23.md).
Why: psi was fitted per TRUE anchor on candidates from the agents' initial masks; in the ledger, xi is chosen among all candidate
sources (incl. VoxTell plan masks, U, U-nh) for true AND false lesions, and scan-level precision suffers (claude/stage1_vs_agents_
diagnosis_2026-11-23.md). This builds one row per candidate extent of every accepted or uncertain anchor of ledger <L> on R2Seg
(tumour scans and the 139 tumour-free scans), from ledger/<L>/extents/<collection>/<stem>.npz:
  identity  ds, case, site, neg, rel, anchor id, lesion, decision, kind, j, name, flags, anchor_ml, ml, logratio
  truth     tp_ml, fp_ml (candidate voxels inside / outside the expert tumour), anchor_tp_ml; T_ml of the scan
            (expert labels: evaluation and fitting only)
  image     one agent-agnostic cue: does the candidate's extension beyond the anchor look like the anchor or like its surroundings?
            ext = candidate minus anchor; shell = voxels 3-6 mm outside the anchor; m_* = medians of the raw image intensities;
            scale = mean of the robust SDs (1.4826 x MAD) in anchor and shell;
            s_ext = (|m_ext - m_anchor| - |m_ext - m_shell|) / scale  (> 0: extension resembles the surroundings), 0 if ext is empty;
            ext_ml; also z_ext = (m_ext - m_anchor) / scale.
Outputs: ledger/<L>/extent_rule/rows[_s<i>].csv, done[_s<i>].txt. Usage: python s1_extent_rule_rows.py --ledger v86_r2seg_bp2 --shards 4"""
import os, sys, glob, json, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s1_eval_r2seg as E   # noqa: E402


def rmad(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def scan_rows(L, ds, case, dec, neg):
    lf = os.path.join(E.paths.root('R2Seg'), ds, 'labels', case + '_seg.nii.gz')
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz')); sp = np.array(img.header.get_zooms()[:3], float)
    vml = float(np.prod(sp)) / 1000.0
    z = np.load(os.path.join(L, 'extents', ds, case + '.npz')); meta = json.loads(str(z['meta']))
    lim = nib.load(lf); shape = tuple(meta['shape'])
    if tuple(lim.shape[:3]) != shape:
        return []
    rows = []
    for i, an in meta['anchors'].items():
        i = int(i)
        if i not in dec.index or dec.loc[i, 'decision'] not in ('accepted', 'uncertain') or not an['extents']:
            continue
        W = an['extents'][0]['box']; pad = np.ceil(6.0 / sp).astype(int) + 1
        Wp = tuple(slice(max(0, b[0] - p), min(n, b[1] + p)) for b, p, n in zip(W, pad, shape))
        off = [Wp[k].start for k in range(3)]; shp = [Wp[k].stop - Wp[k].start for k in range(3)]
        def place(key, box):
            m = np.zeros(shp, bool); sl = tuple(slice(b[0] - o, b[1] - o) for b, o in zip(box, off)); m[sl] = E.unpack(z, key, box); return m
        a = place(f'a{i}', an['box'])
        Lc = np.asarray(lim.dataobj[Wp]); T = np.isin(np.rint(Lc).astype(np.int32), E.EX.APPEARANCE[ds]['tumour'])
        I = np.asarray(img.dataobj[Wp], dtype=np.float32)
        d = ndimage.distance_transform_edt(~a, sampling=sp); shell = (d > 3) & (d <= 6)
        Ia, Ish = I[a], I[shell]; ma, msh = float(np.median(Ia)), (float(np.median(Ish)) if Ish.size else np.nan)
        scale = np.nanmean([rmad(Ia), rmad(Ish)]) or np.nan
        for j, e in enumerate(an['extents']):
            c = place(f'a{i}_e{j}', e['box']); ext = c & ~a
            if ext.any() and np.isfinite(msh) and scale and scale > 0:
                me = float(np.median(I[ext])); s_ext = (abs(me - ma) - abs(me - msh)) / scale; z_ext = (me - ma) / scale
            else:
                s_ext = z_ext = 0.0
            rows.append(dict(ds=ds, case=case, site=E.EX.SITE_ENTITY[ds], neg=neg, anchor=i, lesion=int(dec.loc[i, 'lesion']),
                             decision=dec.loc[i, 'decision'], kind=dec.loc[i, 'kind'], j=j, name=e['name'], flags='|'.join(e.get('flags', [])),
                             anchor_ml=float(a.sum() * vml), ml=float(c.sum() * vml), logratio=float(e['logratio']),
                             tp_ml=float((c & T).sum() * vml), fp_ml=float((c & ~T).sum() * vml), anchor_tp_ml=float((a & T).sum() * vml),
                             ext_ml=float(ext.sum() * vml), s_ext=float(s_ext), z_ext=float(z_ext)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ledger', default='v86_r2seg_bp2'); ap.add_argument('--shards', type=int, default=0)
    ap.add_argument('--only', default=None); a = ap.parse_args()
    E.LEDGER[0] = a.ledger; L = os.path.join(E.paths.ledger_dir(), a.ledger); O = os.path.join(L, 'extent_rule'); os.makedirs(O, exist_ok=True)
    if a.shards:
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args, env=dict(os.environ, SHARD=f'{i}/{a.shards}'),
                               stdout=open(os.path.join(O, f'log_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        for p in ps:
            p.wait()
        print('done ->', O); return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    decs = E.load_decisions()
    tf = pd.read_csv(os.path.join(E.paths.root('outputs'), 'cohort', 'r2seg_tumour_free.csv'), dtype={'case': str})
    negk = set(zip(tf.ds[tf.status == 'negative'], tf.case[tf.status == 'negative']))
    ev = pd.concat([pd.read_csv(f, usecols=['ds', 'case', 'T_ml', 'method'], dtype={'case': str}) for f in glob.glob(os.path.join(L, 'eval_psi_bp2', 'scans*.csv'))])
    ev = ev[ev.method == 'vt']; ev['T_ml'] = pd.to_numeric(ev.T_ml, errors='coerce')
    scans = sorted({(d, c) for d, c, t in zip(ev.ds, ev.case, ev.T_ml) if (t > 0 or (d, c) in negk)})
    if a.only:
        scans = [s for s in scans if any(x in s[1] for x in a.only.split(','))]
    done = set(); fd = os.path.join(O, f'done{SUF}.txt')
    for f in glob.glob(os.path.join(O, 'done*.txt')):
        done |= set(open(f).read().split())
    t0 = time.time(); n = 0
    for k, (ds, case) in enumerate(scans):
        if k % SN != SI or f'{ds}/{case}' in done:
            continue
        rel = [r for r in decs if r.split('/')[1] == ds and r.split('/', 2)[2] == case]
        try:
            rows = scan_rows(L, ds, case, decs[rel[0]], (ds, case) in negk) if rel else []
        except Exception:
            print('error', ds, case, traceback.format_exc()[-300:], flush=True); continue
        if rows:
            f = os.path.join(O, f'rows{SUF}.csv'); pd.DataFrame(rows).to_csv(f, mode='a', index=False, header=not os.path.exists(f))
        open(fd, 'a').write(f'{ds}/{case}\n'); n += 1
        print(n, ds, case, len(rows), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
