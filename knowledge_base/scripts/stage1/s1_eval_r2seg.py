"""Step C (13 Nov 2026): evaluate frozen stage 1 v8.4 on R2Seg with its decisions applied, against the agents alone.
Expert labels are used for evaluation only. Uterus_Tumor_00 (benign) is not evaluated.

Inputs: ledger/v84_r2seg/anchors*.csv (decision per anchor; s1_run_r2seg.py), ledger/v84_r2seg/extents/<collection>/<stem>.npz (every
anchor's mask and candidate extents), the BiomedParse / VoxTell tumour masks and the expert labels.
Extents (user, 12 Nov): decisions from frozen v8.4; xi recomputed with the refitted psi (ledger/v85/extent/fit_psi.json), each site
with the psi fold that left it out. xi with the v8.4 psi is also kept (check: equals the ledger's extent column).
Methods (masks on the image grid):
  vt, bp        each agent's whole tumour mask
  agree, union  BiomedParse AND / OR VoxTell
  s1_all        union of xi (refitted psi) over every anchor (no decision; the unfiltered candidate set)
  s1_acc        ... over accepted anchors (accepted, accepted_override)
  s1_acc_unc    ... over accepted + uncertain anchors
  s1_acc_unc_v84  as s1_acc_unc with the v8.4 psi
Per scan (scan level: false positives count): Dice, TP / FP / FN ml, false-positive components (connected parts touching no
expert tumour). Per expert tumour >= 0.5 ml (sensitivity): detected (any overlap), recall, and Dice of the method's connected parts
that touch it. Tumour-free scans: false-positive components and ml only.
Outputs: clinical/PanCIA_outputs/ledger/v84_r2seg/eval/ scans[_s<i>].csv, tumours[_s<i>].csv; --summary writes report.txt (sites
equal: per collection site, then mean over sites; the two lung collections are one site).
Usage: [SHARD=i/n] python s1_eval_r2seg.py [--collections ...] [--only ...]  (resumable)  |  python s1_eval_r2seg.py --shards 4
       python s1_eval_r2seg.py --summary
       --ledger v85_r2seg --decisions decisions_v85.csv (15 Nov): another stage-1 ledger, with decisions taken from a re-decided file
       (column decision_<name>); outputs go to ledger/<ledger>/eval/.
       --tumour-band 0.1,0.5 (22 Nov): only expert tumours with 0.1 <= ml < 0.5; scans without one are skipped (row method
       'no_band_tumour', for resuming only); outputs ledger/<ledger>/eval_band_0.1_0.5/ (tumours, scans, report.txt)."""
import os, sys, csv, json, glob, time, argparse, subprocess, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, agents as AG   # noqa: E402

METHODS = ('vt', 'bp', 'agree', 'union', 's1_all', 's1_acc', 's1_acc_unc', 's1_acc_unc_v84')
BIT = {m: 1 << k for k, m in enumerate(METHODS)}
ACC = ('accepted', 'accepted_override')
LEDGER = ['v84_r2seg']     # --ledger (15 Nov)
DECFILE = [None]           # --decisions: re-decided file in the ledger dir (rel, id, decision_<name>)
<<<<<<< HEAD
=======
XIRULE = [json.load(open(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json'))).get('xi_rule', 'psi')
          if os.path.exists(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json')) else 'psi']   # 23 Nov
PSIMETHOD = [None]         # --psi-method (23 Nov): extent rule from fit_psi.json folds (M1 | M2 | anchor | v8.4); default = adopted method
>>>>>>> 7271616 (update stage 2)
EVTAG = [None]             # --eval-tag (22 Nov): outputs eval_<tag>/ (e.g. psi_bp2 after the psi refit), old eval kept
BAND = [None]              # --tumour-band lo,hi (22 Nov): evaluate only expert tumours with lo <= ml < hi; outputs eval_band_<lo>_<hi>/
S26 = EX.STRUCT26


def out_dir():
    sub = 'eval' if BAND[0] is None else f'eval_band_{BAND[0][0]:g}_{BAND[0][1]:g}'
    if EVTAG[0]:
        sub += f'_{EVTAG[0]}'
    d = os.path.join(paths.ledger_dir(), LEDGER[0], sub); os.makedirs(d, exist_ok=True); return d


def load_decisions():
    fs = sorted(glob.glob(os.path.join(paths.ledger_dir(), LEDGER[0], 'anchors*.csv')))
    d = pd.concat([pd.read_csv(f, usecols=['rel', 'id', 'kind', 'lesion', 'decision', 'extent']) for f in fs], ignore_index=True)
    d = d.drop_duplicates(['rel', 'id'], keep='last')
    if DECFILE[0]:                                                    # re-decided file (e.g. decisions_v85.csv)
        v = pd.read_csv(os.path.join(paths.ledger_dir(), LEDGER[0], DECFILE[0]))
        col = [c for c in v.columns if c.startswith('decision_') and c != 'decision_v84'][0]
        d = d.drop(columns=['decision']).merge(v[['rel', 'id', col]].rename(columns={col: 'decision'}), on=['rel', 'id'], how='left')
        d['decision'] = d.decision.fillna('rejected')
    return {rel: g.set_index('id') for rel, g in d.groupby('rel')}


def psi_for(site):
    pj = json.load(open(os.path.join(paths.ledger_dir(), 'v85', 'extent', 'fit_psi.json')))
    old = G2.Psi(json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))['psi'])
    if pj.get('adopted') and site in (pj.get('folds') or {}):
        new = G2.Psi(dict(edges=pj['psi']['edges'], val=pj['folds'][site][PSIMETHOD[0] or pj['method']]))
    else:
        new = G2.Psi(pj['psi'])
    return old, new


def unpack(z, key, box):
    shp = [b[1] - b[0] for b in box]
    return np.unpackbits(z[key])[:int(np.prod(shp))].reshape(shp).astype(bool)


def put(M, bit, box, crop):
    sl = tuple(slice(b[0], b[1]) for b in box)
    M[sl] |= np.where(crop, bit, 0).astype(M.dtype)


def comps(P):
    lab, n = ndimage.label(P, structure=S26)
    return lab, n


def eval_scan(ds, stem, dec, psis):
    lf = os.path.join(paths.root('R2Seg'), ds, 'labels', stem + '_seg.nii.gz')
    img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
    sp = np.array(img.header.get_zooms()[:3], float); vml = float(np.prod(sp)) / 1000.0
    T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour'])
    if BAND[0] is not None:                                            # band mode: skip scans without a tumour in the band
        lt0, n0 = comps(T); sz0 = np.bincount(lt0.ravel(), minlength=n0 + 1)[1:] * vml
        if not ((sz0 >= BAND[0][0]) & (sz0 < BAND[0][1])).any():
            return [dict(ds=ds, case=stem, site=EX.SITE_ENTITY[ds], T_ml=float(T.sum() * vml), note=None, xi_v84_check=None, method='no_band_tumour',
                         dice=np.nan, tp_ml=np.nan, fp_ml=np.nan, fn_ml=np.nan, fp_comps=np.nan, pred_ml=np.nan)], []   # same columns as a full row (CSV append)
        del lt0
    M = np.zeros(T.shape, np.uint8)
    for a, m in (('VoxTell', 'vt'), ('BiomedParse', 'bp')):
        A, _ = AG._on_grid(AG.mask_path(a, ds, stem), img)
        M |= np.where(A, BIT[m], 0).astype(np.uint8)
    vt = (M & BIT['vt']) > 0; bp = (M & BIT['bp']) > 0
    M |= np.where(vt & bp, BIT['agree'], 0).astype(np.uint8); M |= np.where(vt | bp, BIT['union'], 0).astype(np.uint8)
    del vt, bp
    note = None; n_chk = n_ok = 0
    fz = os.path.join(paths.ledger_dir(), LEDGER[0], 'extents', ds, stem + '.npz')
    if os.path.exists(fz):
        for _try in range(8):                                            # OneDrive read errors (Errno 35 / 60 / 89) are transient (15-16 Nov)
            try:
                z = np.load(fz); meta = json.loads(str(z['meta'])); break
            except OSError:
                if _try == 7:
                    raise
                time.sleep(min(60, 5 * (_try + 1)))
        if tuple(meta['shape']) != T.shape:
            note = f"ledger grid {tuple(meta['shape'])} != label grid {T.shape}: stage-1 masks skipped"
        else:
            old, new = psis
            for i, an in meta['anchors'].items():
                ex = an['extents']
                if not ex or int(i) not in dec.index:
                    continue
                cands = [dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ex]
                j_new = int(np.argmax(G2.extent_prior(cands, new, {}))); j_old = int(np.argmax(G2.extent_prior(cands, old, {})))
                if PSIMETHOD[0] is None and XIRULE[0] == 'anchor':      # 23 Nov: adopted outline rule xi = anchor (fit_psi.json xi_rule)
                    j_new = ([k for k, e in enumerate(ex) if e['name'] == 'anchor'] or [0])[0]
                n_chk += 1; n_ok += int(ex[j_old]['name'] == dec.loc[int(i), 'extent'])
                d_ = dec.loc[int(i), 'decision']
                xn = unpack(z, f'a{i}_e{j_new}', ex[j_new]['box']); xo = unpack(z, f'a{i}_e{j_old}', ex[j_old]['box'])
                put(M, BIT['s1_all'], ex[j_new]['box'], xn)
                if d_ in ACC:
                    put(M, BIT['s1_acc'], ex[j_new]['box'], xn)
                if d_ in ACC or d_ == 'uncertain':
                    put(M, BIT['s1_acc_unc'], ex[j_new]['box'], xn); put(M, BIT['s1_acc_unc_v84'], ex[j_old]['box'], xo)
    # crop to everything that matters
    nz = np.nonzero(T | (M > 0))
    base = dict(ds=ds, case=stem, site=EX.SITE_ENTITY[ds], T_ml=float(T.sum() * vml), note=note, xi_v84_check=f'{n_ok}/{n_chk}')
    if len(nz[0]) == 0:
        return [dict(base, method=m, dice=np.nan, tp_ml=0.0, fp_ml=0.0, fn_ml=0.0, fp_comps=0, pred_ml=0.0) for m in METHODS], []
    sl = tuple(slice(max(0, int(a.min()) - 2), int(a.max()) + 3) for a in nz)
    T = T[sl]; M = M[sl]
    labT, nT = comps(T)
    tsz = np.bincount(labT.ravel(), minlength=nT + 1)
    lo, hi = (EX.MIN_ML, np.inf) if BAND[0] is None else BAND[0]
    big = [j for j in range(1, nT + 1) if lo <= tsz[j] * vml < hi]
    srows, trows = [], []
    for m in METHODS:
        P = (M & BIT[m]) > 0
        tp = float((P & T).sum()); fp = float((P & ~T).sum()); fn = float((~P & T).sum())
        labP, nP = comps(P)
        if nP:
            hitT = np.zeros(nP + 1, bool); hitT[np.unique(labP[T & P])] = True
            fpc = int((~hitT[1:]).sum())
        else:
            fpc = 0
        srows.append(dict(base, method=m, dice=2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else np.nan, tp_ml=tp * vml, fp_ml=fp * vml,
                          fn_ml=fn * vml, fp_comps=fpc, pred_ml=float(P.sum() * vml)))
        for j in big:
            g = labT == j
            inter = P & g
            if inter.any():
                parts = np.isin(labP, np.unique(labP[inter]))
                d_ = 2 * float((parts & g).sum()) / float(parts.sum() + g.sum())
            else:
                d_ = 0.0
            trows.append(dict(ds=ds, case=stem, site=EX.SITE_ENTITY[ds], tumour=j, ml=float(tsz[j] * vml), method=m,
                              detected=bool(inter.any()), recall=float(inter.sum()) / float(tsz[j]), dice=d_))
    return srows, trows


def append(f, rows):
    if rows:
        pd.DataFrame(rows).to_csv(f, mode='a', index=False, header=not os.path.exists(f))


def summarise(O):
    s = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(O, 'scans*.csv'))], ignore_index=True)
    s = s[pd.to_numeric(s.T_ml, errors='coerce').notna()].copy()     # 15 Nov: error rows (misaligned, T_ml = 'error') are left out
    for c in ('T_ml', 'dice', 'tp_ml', 'fp_ml', 'fn_ml', 'fp_comps', 'pred_ml'):
        s[c] = pd.to_numeric(s[c], errors='coerce')
    s = s.drop_duplicates(['ds', 'case', 'method'], keep='last')
    t = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(O, 'tumours*.csv'))], ignore_index=True)
    t = t.drop_duplicates(['ds', 'case', 'tumour', 'method'], keep='last')
    rep = [f'Stage 1 on R2Seg, ledger {LEDGER[0]}' + (f', decisions {DECFILE[0]}' if DECFILE[0] else ', decisions as stored') + f' (extents: refitted psi, held-out site fold; s1_acc_unc_v84 = accepted + uncertain with the v8.4 psi extents). {s.case.nunique()} scans '
           f'({(s[s.method == "vt"].T_ml > 0).sum()} with tumour), {t.drop_duplicates(["case", "ds", "tumour"]).shape[0]} expert tumours >= 0.5 ml. '
           'Sites equal (site mean, then mean over sites).', '']
    pos = s[s.T_ml > 0]
    sw = lambda df, col: df.groupby(['site', 'method'])[col].mean().unstack('method')[list(METHODS)]
    rep += ['== per expert tumour (sensitivity): detection rate ==', sw(t, 'detected').round(3).to_string(),
            '(sites equal) ' + ', '.join(f'{m} {v:.3f}' for m, v in sw(t, 'detected').mean().items()), '']
    rep += ['== per expert tumour: Dice of the parts touching it (0 if missed) ==', sw(t, 'dice').round(3).to_string(),
            '(sites equal) ' + ', '.join(f'{m} {v:.3f}' for m, v in sw(t, 'dice').mean().items()), '']
    rep += ['== scan level, scans with tumour: Dice ==', sw(pos, 'dice').round(3).to_string(),
            '(sites equal) ' + ', '.join(f'{m} {v:.3f}' for m, v in sw(pos, 'dice').mean().items()), '']
    rep += ['== scan level, scans with tumour: false-positive components per scan ==', sw(pos, 'fp_comps').round(2).to_string(),
            '(sites equal) ' + ', '.join(f'{m} {v:.2f}' for m, v in sw(pos, 'fp_comps').mean().items()), '']
    rep += ['== scan level, scans with tumour: false-positive ml per scan ==', sw(pos, 'fp_ml').round(1).to_string(), '']
    # tumour-free scans (user, 14 Nov): cohort/r2seg_tumour_free.csv status 'negative' only (NIH-like pancreas 80, prostate 57, lung 3);
    # FLARE23 scans without a lesion label are unannotated and left out
    tf = pd.read_csv(os.path.join(paths.root('outputs'), 'cohort', 'r2seg_tumour_free.csv'), dtype={'case': str})
    negk = set(zip(tf.ds[tf.status == 'negative'], tf.case[tf.status == 'negative']))
    s['case'] = s.case.astype(str)
    neg = s[[(a, c) in negk for a, c in zip(s.ds, s.case)]]
    if len(neg):
        rep += [f'== tumour-free scans ({neg.case.nunique()}): false-positive components per scan / share of scans with any ==',
                sw(neg, 'fp_comps').round(2).to_string(),
                neg.assign(any_=neg.fp_comps > 0).groupby(['site', 'method']).any_.mean().unstack('method')[list(METHODS)].round(2).to_string(), '']
    chk = s[s.method == 'vt'].xi_v84_check.dropna().str.split('/', expand=True).astype(int).sum()
    rep.append(f'check: xi recomputed with the v8.4 psi equals the ledger extent for {chk[0]} of {chk[1]} anchors')
    notes = s[s.method == 'vt'].note.dropna()
    if len(notes):
        rep.append(f'{len(notes)} scans with notes (e.g. {notes.iloc[0]})')
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


def summarise_band(O):
    t = pd.concat([pd.read_csv(f, dtype={'case': str}) for f in glob.glob(os.path.join(O, 'tumours*.csv'))], ignore_index=True)
    t = t.drop_duplicates(['ds', 'case', 'tumour', 'method'], keep='last')
    t['detected'] = t.detected.astype(str).isin(['True', '1', '1.0'])
    for c in ('recall', 'dice', 'ml'):
        t[c] = pd.to_numeric(t[c], errors='coerce')
    lo, hi = BAND[0]
    n = t.drop_duplicates(['ds', 'case', 'tumour']).groupby('site').size()
    rep = [f'Stage 1 on R2Seg, ledger {LEDGER[0]}: expert tumours with {lo:g} <= ml < {hi:g} only ({int(n.sum())} tumours in '
           f'{t.drop_duplicates(["ds", "case"]).shape[0]} scans). Detection = any overlap. Sites equal = mean over sites with >= 1 tumour.', '',
           'tumours per site: ' + ', '.join(f'{k} {v}' for k, v in n.items()), '']
    for title, col in (('detection rate', 'detected'), ('recall', 'recall'), ('Dice', 'dice')):
        x = t.groupby(['site', 'method'])[col].mean().unstack('method')[[m for m in METHODS if m in set(t.method)]]
        rep += [f'== {title} ==', x.round(3).to_string(), '(sites equal) ' + ', '.join(f'{m} {v:.3f}' for m, v in x.mean().items()), '']
    open(os.path.join(O, 'report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collections', default=None); ap.add_argument('--only', default=None)
    ap.add_argument('--shards', type=int, default=0); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--ledger', default='v84_r2seg'); ap.add_argument('--decisions', default=None)
    ap.add_argument('--eval-tag', default=None, help='output dir eval_<tag>/ (22 Nov)')
<<<<<<< HEAD
    ap.add_argument('--tumour-band', default=None, help='lo,hi in ml (e.g. 0.1,0.5): per-tumour rows only for tumours in [lo, hi); separate output dir')
    a = ap.parse_args()
    LEDGER[0] = a.ledger; DECFILE[0] = a.decisions; EVTAG[0] = a.eval_tag
=======
    ap.add_argument('--psi-method', default=None, choices=('M1', 'M2', 'anchor', 'v8.4'), help='extent rule (held-out site fold) from fit_psi.json (23 Nov)')
    ap.add_argument('--tumour-band', default=None, help='lo,hi in ml (e.g. 0.1,0.5): per-tumour rows only for tumours in [lo, hi); separate output dir')
    a = ap.parse_args()
    LEDGER[0] = a.ledger; DECFILE[0] = a.decisions; EVTAG[0] = a.eval_tag; PSIMETHOD[0] = a.psi_method
>>>>>>> 7271616 (update stage 2)
    if a.tumour_band:
        BAND[0] = tuple(float(x) for x in a.tumour_band.split(','))
        if a.summary:
            summarise_band(out_dir()); return
    O = out_dir()
    if a.summary:
        summarise(O); return
    if a.shards:
        args = [x for x in sys.argv[1:] if not x.startswith('--shards') and x != str(a.shards)]
        ps = [subprocess.Popen(['nice', '-n', '10', sys.executable, os.path.abspath(__file__)] + args, env=dict(os.environ, SHARD=f'{i}/{a.shards}'),
                               stdout=open(os.path.join(O, f'log_s{i}.txt'), 'a'), stderr=subprocess.STDOUT) for i in range(a.shards)]
        try:
            for p in ps:
                p.wait()
        except KeyboardInterrupt:
            for p in ps:
                p.terminate()
        summarise_band(O) if BAND[0] is not None else summarise(O); return
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    meta = list(csv.DictReader(open(os.path.join(paths.root('outputs'), 'cohort', 'r2seg_scan_meta.csv'))))
    meta = [r for r in meta if r['img_name'].split('/')[0] in EX.SITE_ENTITY]          # no Uterus (benign)
    if a.collections:
        meta = [r for r in meta if r['img_name'].split('/')[0] in a.collections.split(',')]
    if a.only:
        meta = [r for r in meta if any(s_ in r['img_name'] for s_ in a.only.split(','))]
    done = set()
    for f in glob.glob(os.path.join(O, 'scans*.csv')):
        d0 = pd.read_csv(f, usecols=['ds', 'case', 'T_ml'], dtype=str); d0 = d0[d0.T_ml != 'error']   # error rows are retried (15 Nov)
        done |= set(zip(d0.ds, d0.case.astype(str)))
    decs = load_decisions(); psis = {}
    t0 = time.time(); n = 0
    for k, r in enumerate(meta):
        if k % SN != SI:
            continue
        ds, stem = r['img_name'].split('/', 1)
        if (ds, stem) in done:
            continue
        rel = f"{r['modality']}/{ds}/{stem}"
        site = EX.SITE_ENTITY[ds]
        if site not in psis:
            psis[site] = psi_for(site)
        dec = decs.get(rel, pd.DataFrame(columns=['kind', 'lesion', 'decision', 'extent']))
        try:
            srows, trows = eval_scan(ds, stem, dec, psis[site])
        except Exception:
            srows = [dict(ds=ds, case=stem, site=site, method='error', note=traceback.format_exc()[-400:].replace('\n', ' | '))]; trows = []
        append(os.path.join(O, f'tumours{SUF}.csv'), trows); append(os.path.join(O, f'scans{SUF}.csv'), srows)
        n += 1
        print(f'{n} {rel} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
