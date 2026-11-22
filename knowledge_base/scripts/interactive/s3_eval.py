"""Score the interactive-agent pilot (22 Nov 2026). Expert labels: evaluation only.
Per pilot lesion (interactive/pilot_bp2/pilot_lesions.csv): target G = the expert tumour components touched by the lesion extent xi
(as when the pilot was drawn). Compared outlines, each restricted to its 26-connected components that touch xi or G (so other lesions
in the scan do not count): xi (stage-1 extent, bp2 psi), VT and BP (initial agent masks), and each nnInteractive variant P1-P4.
Measures: Dice, recall (share of G covered), precision, volume ratio to G; summaries per variant, per site (sites equal), and for
under-segmented lesions (xi recall < 0.7). Time per scan / lesion from the runner's timing files.
Usage: python s3_eval.py [--masks <outputs>/interactive/pilot_bp2/masks]"""
import os, sys, json, glob, argparse
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX, agents as AG   # noqa: E402
VARIANTS = ('P1', 'P2', 'P3', 'P4')


def local(M, ref):
    if M is None or not M.any():
        return np.zeros_like(ref)
    lab, _ = ndimage.label(M, structure=EX.STRUCT26); ids = set(np.unique(lab[ref]).tolist()) - {0}
    return np.isin(lab, list(ids)) if ids else np.zeros_like(ref)


def scores(P, G):
    tp = float((P & G).sum()); return dict(dice=2 * tp / (P.sum() + G.sum()) if (P.sum() + G.sum()) else np.nan,
                                           recall=tp / G.sum(), precision=tp / P.sum() if P.sum() else np.nan, vol_ratio=float(P.sum() / G.sum()))


def main():
    ap = argparse.ArgumentParser(); O = os.path.join(paths.root('outputs'), 'interactive', 'pilot_bp2')
    ap.add_argument('--masks', default=os.path.join(O, 'masks')); a = ap.parse_args()
    pl = pd.read_csv(os.path.join(O, 'pilot_lesions.csv'), dtype={'case': str}); rows = []
    for (ds, case), g in pl.groupby(['ds', 'case']):
        lf = os.path.join(paths.root('R2Seg'), ds, 'labels', case + '_seg.nii.gz'); img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
        T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour']); lt, _ = ndimage.label(T, structure=EX.STRUCT26)
        cz = np.load(os.path.join(O, 'prompts', f'{ds}__{case}.npz'))
        ag = {}
        for nm, agent in (('VT', 'VoxTell'), ('BP', 'BiomedParse')):
            A, _ = AG._on_grid(AG.mask_path(agent, ds, case), img); ag[nm] = None if A is None else A.astype(bool)
        for r in g.itertuples():
            xi = np.unpackbits(cz[f'xi_{r.lesion}'])[:T.size].reshape(T.shape).astype(bool)
            G = np.isin(lt, list(set(np.unique(lt[xi]).tolist()) - {0})); ref = xi | G
            outs = {'xi': xi, 'VT': local(ag['VT'], ref), 'BP': local(ag['BP'], ref)}
            for V in VARIANTS:
                f = os.path.join(a.masks, ds, f'{case}_les{r.lesion}_{V}.nii.gz')
                if os.path.exists(f):
                    M = np.asarray(nib.load(f).dataobj) > 0
                    outs[V] = M if M.shape == T.shape else None
            for k, P in outs.items():
                if P is not None:
                    rows.append(dict(site=r.site, ds=ds, case=case, lesion=r.lesion, decision=r.decision, under=r.under, outline=k, **scores(P, G)))
        print(ds, case, flush=True)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(O, 'pilot_scores.csv'), index=False)
    cols = [c for c in ('xi', 'VT', 'BP') + VARIANTS if c in set(d.outline)]
    rep = [f'Interactive pilot: {pl.shape[0]} lesions in {pl.case.nunique()} scans; outlines restricted to components touching xi or the tumour. Sites equal.', '']
    for m in ('dice', 'recall', 'precision', 'vol_ratio'):
        x = d.groupby(['site', 'outline'])[m].mean().unstack('outline')[cols]
        rep += [f'== {m} ==', x.round(3).to_string(), '(sites equal) ' + ', '.join(f'{c} {v:.3f}' for c, v in x.mean().items())]
        u = d[d.under].groupby(['site', 'outline'])[m].mean().unstack('outline')[cols].mean()
        rep += ['(under-segmented lesions, xi recall < 0.7; sites equal) ' + ', '.join(f'{c} {v:.3f}' for c, v in u.items()), '']
    tfs = glob.glob(os.path.join(a.masks, 'timing_s*.csv'))
    if tfs:
        t = pd.concat([pd.read_csv(f) for f in tfs])
        rep += ['== time (s) ==', 'set_image per scan: median %.1f, max %.1f' % (t[t.variant == 'set_image'].seconds.median(), t[t.variant == 'set_image'].seconds.max()),
                'per lesion and variant: ' + ', '.join(f"{V} median {t[t.variant == V].seconds.median():.2f}" for V in VARIANTS if (t.variant == V).any()),
                'errors: %d' % t.note.astype(str).str.startswith('error').sum()]
    open(os.path.join(O, 'pilot_report.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))


if __name__ == '__main__':
    main()
