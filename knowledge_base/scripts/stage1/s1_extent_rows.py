"""Stage 1 (v8.5 item brought forward, 29 Oct): candidate-extent rows for refitting the extent prior psi on REAL agent outlines.

Data: R2Seg usable tumour scans (up to --n-per per collection, drawn at random; the same draw as the stage-2 builds), BiomedParse and
VoxTell tumour masks (R2Seg_Seg/<agent>/<collection>/<stem>_tumor.nii.gz), expert labels for evaluation of each candidate only.
Anchors and lesions: pancia_kb.stage2.agents.scan(single=True) (two-agent and single-agent anchors, v8.4 rules). Per TRUE anchor
(it touches expert tumour) its candidate extents by the v8.4 rule (expert.extents: the anchor, every outline touching it after lobe
splitting, U, U - non-host organs; merged at Dice >= 0.95). Every outline is a candidate on equal terms; no agent identity is used.
False anchors are left out (psi is the prior over extents of a tumour).

Rows (one per candidate): ds, case, site, anchor, kind (two / single), name, logratio = log(|xi| / |a|), ml, anchor_ml, G_ml (the expert
components the anchor touches), dice, best (argmax Dice within the anchor), w0_v84 (current psi).
Patterns (one row per anchor, for exact q >= 1/2 Dice under any psi): every voxel of the union of the candidates is coded by the set
of candidates containing it (bitmask over the anchor's candidates); n voxels and n tumour voxels per code, plus tumour voxels outside
every candidate.
Outputs: clinical/PanCIA_outputs/ledger/v85/extent/ extent_rows[_s<i>].csv, patterns[_s<i>].jsonl, done[_s<i>].txt, notes[_s<i>].csv
Usage: [SHARD=i/n] python s1_extent_rows.py [--n-per 100] [--only Kidney_Tumor_00]   (resumable)"""
import os, re, sys, json, time, argparse, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import KnowledgeBase, grounding_g1 as G1, grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, agents as AG   # noqa: E402


def out_dir():
    d = os.path.join(paths.ledger_dir(), 'v85', 'extent'); os.makedirs(d, exist_ok=True); return d


def rows_for_case(ds, lf, kb, psi):
    S = AG.scan(ds, lf, single=True)
    if S is None:
        return None, None, 'no agent masks'
    trueL = [L for L in S['lesions'] if L['true']]
    if not trueL:
        return [], [], 'no true anchor'
    W = F.pad_box(AG._box_union([F.bbox(S['T'])] + [L['obox'] for L in trueL]), S['sp'], AG.PAD_MM, S['T'].shape)
    c = AG.window(ds, S, W, kb)
    if c is None:
        return None, None, 'no TS mask'
    sp = c['sp']; vml = float(np.prod(sp)) / 1000.0
    rows, pats = [], []
    for L in trueL:
        for i in L['anchors']:
            an = S['anchors'][i]
            if not an['true']:
                continue
            a = c['lab'] == an['id']
            if a.sum() < 20:
                continue
            outl = {s: np.isin(c['lab_ag'][ag], an[s]) for ag, s in AG.SHORT.items()}
            ext, w0 = EX.extents(a, outl, c['others'], sp, G1, G2, psi)
            G = AG.lesion_truth(c, a)
            dice = [float(2 * (m & G).sum() / max(int(m.sum() + G.sum()), 1)) for _, m in ext]
            ib = int(np.argmax(dice))
            for j, ((nm, m), w, d) in enumerate(zip(ext, w0, dice)):
                rows.append(dict(ds=ds, case=S['case'], site=EX.SITE_OF[ds], anchor=int(an['id']), kind=an['kind'], cand=j, name=nm,
                                 logratio=float(np.log(max(int(m.sum()), 1) / max(int(a.sum()), 1))), ml=float(m.sum() * vml),
                                 anchor_ml=float(a.sum() * vml), G_ml=float(G.sum() * vml), dice=d, best=int(j == ib), w0_v84=float(w)))
            code = np.zeros(a.shape, np.int64)
            for j, (_, m) in enumerate(ext):
                code[m] |= (1 << j)
            inU = code > 0
            cs, inv = np.unique(code[inU], return_inverse=True)
            n = np.bincount(inv); nt = np.bincount(inv, weights=G[inU].astype(float))
            pats.append(dict(ds=ds, case=S['case'], anchor=int(an['id']), vml=vml, codes=[int(x) for x in cs], n=[int(x) for x in n],
                             nt=[int(x) for x in nt], t_out=int((G & ~inU).sum())))
    return rows, pats, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-per', type=int, default=100); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = out_dir()
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(fit6['psi'])
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    done = set()
    for f in os.listdir(O):
        if re.fullmatch(r'done(_s\d+)?\.txt', f):
            done |= set(open(os.path.join(O, f)).read().splitlines())
    tasks = [t for i, t in enumerate(EX.case_list(paths.root('R2Seg'), a.n_per)) if i % SN == SI]
    if a.only:
        tasks = [t for t in tasks if t[0] in a.only.split(',')]
    t0 = time.time()
    for k, (ds, lf) in enumerate(tasks):
        key = ds + '/' + os.path.basename(lf)
        if key in done:
            continue
        note = None
        try:
            rows, pats, note = rows_for_case(ds, lf, KB, psi)
            if rows:
                f = os.path.join(O, f'extent_rows{SUF}.csv')
                pd.DataFrame(rows).to_csv(f, mode='a', index=False, header=not os.path.exists(f))
                with open(os.path.join(O, f'patterns{SUF}.jsonl'), 'a') as fh:
                    for p_ in pats:
                        fh.write(json.dumps(p_) + '\n')
        except Exception:
            note = traceback.format_exc()[-500:].replace('\n', ' | ')
        if note:
            f = os.path.join(O, f'notes{SUF}.csv')
            pd.DataFrame([dict(key=key, note=note)]).to_csv(f, mode='a', index=False, header=not os.path.exists(f))
        with open(os.path.join(O, f'done{SUF}.txt'), 'a') as fh:
            fh.write(key + '\n')
        print(f'{k + 1}/{len(tasks)} {key} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
