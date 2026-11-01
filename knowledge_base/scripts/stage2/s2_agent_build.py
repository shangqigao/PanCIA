"""Stage 2, step 2b: spatial-prior training samples from REAL agent outlines on R2Seg (29 Oct). Tumour truth: expert labels.
Anatomy: TotalSegmentator. Agents: BiomedParse and VoxTell tumour masks in R2Seg_Seg/<agent>/<collection>/<stem>_tumor.nii.gz.

Why: the spatial prior fitted on simulated claims (s2_expert_build.py) learned "no tumour outside the claims": simulated outlines
bracket the expert tumour, so tumour beyond every outline was rare (0-3 %) and the fitted eta is a function of q alone (pi ~ 0.0035
just outside the outlines at any distance). Real agents fail together, see only part of a tumour (BiomedParse is 2D) or make false
anchors. Fitting on real outlines lets the prior learn how often and how far tumour lies beyond them, and how often an anchor is
wrong; no rate is set by hand.

Per case (usable tumour scans: up to --n-per drawn at random per collection, as s2_expert_build.py; tumour-free scans: up to
--n-neg per collection; EVALUATION ONLY: s2_fit.py skips them, as decided on 25 Oct; s2_prior_calibration.py scores them), stage-1-like lesions from the two agents (pancia_kb.stage2.agents: agreement anchors >= 0.5 ml, each
agent's touching components as outlines, v8.4 extents and w0, anchors chained within 10 mm; and, by default, single-agent
anchors: one agent's component with no agreement, joining the nearest two-agent lesion within 10 mm, as v8.4 kind 'single'; the
lesion's disagreement share u = 1 - k / n (k agreeing agents of n; run.lesion_unagreed) is stored as uL, and the voxel's anchor kind as anc = 2 / 1 / 0). Windows as in s2_validate_agents.py:
the tumour window (expert tumour, true lesions and the false lesions inside it, + 60 mm) and up to --max-false other false lesions
in windows of their own (default: all of them; a size-ranked subset would bias how often anchors look false). Within 60 mm of an anchor or in the host territory, voxels are sampled (tumour <= N_T, other <= N_B per
window, with weights count / sampled). Columns as s2_expert_build.py plus dtp (through-plane distance to the cluster's chosen
extent) and dtpL (per lesion); 'core' holds 0 (one group per case: windows are disjoint parts of one scan).
Also per true lesion: the farthest distance from its anchors to its expert tumour (r_rows, for r).

Outputs (clinical/PanCIA_outputs/stage2/agent_samples/): samples/<collection>__<case>.npz, r_rows[_s<i>].csv, done[_s<i>].txt,
notes[_s<i>].csv. Fit with: python s2_fit.py --samples agent [--exclude ...] [--tumour-from anchor] --out ...
True-lesion samples (1 Nov, --true-only -> agent_samples_true/): the spatial prior describes where a REAL tumour lies relative to
its candidate geometry, P(tumour at v | the lesion is real); whether a lesion is real is the stage-1 decision (and the stage-2 log BF),
not the prior. Fitting on every agent lesion (most of them false: ~28 per colon scan, ~1 true) mixed existence into the prior, which
was then too low inside real tumours (validate_agents_v2: stage 2 trimmed true lesions, recall 0.75 -> 0.68). With --true-only only the
tumour window is built, its clusters are the true lesions only (false lesions inside it are not candidates; their voxels are ordinary
samples), false-lesion windows and tumour-free scans are skipped. Every build also stores tL (1 = true lesion) per lesion column.
Usage: [SHARD=i/n] python s2_agent_build.py [--n-per 100] [--n-neg 50] [--only Kidney_Tumor_00] [--true-only]   (resumable)"""
import os, re, sys, json, time, zlib, argparse, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb import KnowledgeBase, grounding_g1 as G1, grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, prior as PR, agents as AG, run as RN   # noqa: E402

N_T, N_B = 2000, 6000
SINGLE = [True]                  # single-agent anchors too (v8.4 kind 'single'; --two-only turns them off)
TRUE_ONLY = [False]              # --true-only: clusters of true lesions only (1 Nov)
R_MAX = 60.0
COLS = ('core', 'scen', 't', 'q', 'sd', 'dtp', 'anc', 'in_h', 'in_o', 'kb_h', 'dep_h', 'dep_o', 'd_anchor', 'ns', 'n1', 'w', 'qL', 'sdL',
        'dtpL', 'uL', 'tL')


def _sd_window(xi, sp, shape):
    b = F.bbox(xi)
    out = np.full(shape, 60.0, np.float32)
    if b is None:
        return out
    W = F.pad_box(b, sp, 61.0, shape)
    out[W] = np.minimum(F.signed_distance(xi[W], sp), 60.0)
    return out


def _dtp_window(xi, sp, ax, shape):
    b = F.bbox(xi)
    out = np.full(shape, 60.0, np.float32)
    if b is None:
        return out
    W = F.pad_box(b, sp, 61.0, shape)
    out[W] = np.minimum(F.through_plane_distance(xi[W], sp, ax), 60.0)
    return out


def windows(S, max_false):
    """[(W, lesions, kind)] : the tumour window (expert tumour + true lesions + false lesions inside it) and false-lesion windows"""
    shape, sp = S['T'].shape, S['sp']
    out = []
    if S['T'].any():
        trueL = [L for L in S['lesions'] if L['true']]
        W = F.pad_box(AG._box_union([F.bbox(S['T'])] + [L['obox'] for L in trueL]), sp, AG.PAD_MM, shape)
        inside = lambda b: all(b[k].start >= W[k].start and b[k].stop <= W[k].stop for k in range(3))
        inL = [L for L in S['lesions'] if L['true'] or (inside(L['obox']) and not TRUE_ONLY[0])]
        rest = [L for L in S['lesions'] if not (L['true'] or inside(L['obox']))]
        if inL:
            out.append((W, inL, 'tumour'))
    else:
        rest = list(S['lesions'])
    if TRUE_ONLY[0]:
        return out
    for L in sorted(rest, key=lambda L: -L['ml'])[:max_false]:
        out.append((F.pad_box(L['obox'], sp, AG.PAD_MM, shape), [L], 'false'))
    return out


def build_window(ds, S, W, Ls, kb, psi, rng, cols, rrows, case):
    c = AG.window(ds, S, W, kb)
    if c is None:
        return 0
    sp, shape, ax = c['sp'], c['T'].shape, c['ax']
    Bs = [(L, B) for L in Ls for B in [AG.build_lesion(c, S, L, G1, G2, psi)] if B is not None]
    if not Bs:
        return 0
    omq = np.ones(shape, np.float32); xi = np.zeros(shape, bool); A = np.zeros(shape, bool); anc = np.zeros(shape, np.int8)
    for L, B in Bs:
        omq *= 1.0 - B['q']; xi |= B['xi']; A |= B['A']
        anc = np.maximum(anc, np.where(B['A'], 2 if L['support'] == 'two' else 1, 0).astype(np.int8))   # 2/1: anchor of a two/single lesion
        if L['true']:
            G = AG.lesion_truth(c, B['A'])
            if G.any():
                rrows.append(dict(ds=ds, case=case, far_mm=float(F.edt(B['A'], sp)[G].max()), anchor_ml=float(B['A'].sum() * np.prod(sp) / 1000),
                                  G_ml=float(G.sum() * np.prod(sp) / 1000), n_anchors=len(L['anchors'])))
    q = 1.0 - omq; del omq
    sd = _sd_window(xi, sp, shape); dtp = _dtp_window(xi, sp, ax, shape); da = F.edt(A, sp)
    theta, kbf, seg, depth, hkey = EX.territories(c, 'ts')
    in_h, in_o, kb_h, dep_h, dep_o = PR.host_inputs(theta, kbf, seg, depth, hkey, shape)
    ns, n1 = EX.neighbour_sums(c['T'], sp)
    R = (da <= R_MAX) | in_h
    takes = []
    for lab_, nmax in ((True, N_T), (False, N_B)):
        idx = np.flatnonzero((R & (c['T'] == lab_)).ravel())
        if len(idx) == 0:
            continue
        take = idx if len(idx) <= nmax else rng.choice(idx, nmax, replace=False)
        w = len(idx) / len(take)
        cols['core'].append(np.zeros(len(take), np.int8)); cols['scen'].append(np.full(len(take), EX.SCENS.index('ts'), np.int8))
        cols['t'].append(np.full(len(take), lab_, np.int8))
        for nm, arr in (('q', q), ('sd', sd), ('dtp', dtp), ('anc', anc), ('in_h', in_h), ('in_o', in_o), ('dep_h', dep_h), ('dep_o', dep_o),
                        ('d_anchor', da), ('ns', ns), ('n1', n1)):
            cols[nm].append(np.asarray(arr).ravel()[take].astype(np.float32))
        cols['kb_h'].append(np.full(len(take), kb_h, np.float32)); cols['w'].append(np.full(len(take), w, np.float32))
        takes.append(take)
    allt = np.concatenate(takes) if takes else np.zeros(0, np.int64)
    J = len(Bs)
    qL = np.full((len(allt), J), np.nan, np.float32); sdL = qL.copy(); dtpL = qL.copy(); uL = qL.copy(); tL = qL.copy()
    for k, (L, B) in enumerate(Bs):
        qL[:, k] = B['q'].ravel()[allt]; uL[:, k] = RN.lesion_unagreed(L); tL[:, k] = float(L['true'])
        sdL[:, k] = (sd if J == 1 else _sd_window(B['xi'], sp, shape)).ravel()[allt]
        dtpL[:, k] = (dtp if J == 1 else _dtp_window(B['xi'], sp, ax, shape)).ravel()[allt]
    cols['qL'].append(qL); cols['sdL'].append(sdL); cols['dtpL'].append(dtpL); cols['uL'].append(uL); cols['tL'].append(tL)
    return len(allt)


def build(ds, lf, kb, psi, O, max_false):
    S = AG.scan(ds, lf, single=SINGLE[0])
    if S is None:
        return None, 'no agent masks'
    if not S['lesions']:
        return None, 'no anchor'
    if TRUE_ONLY[0] and not any(L['true'] for L in S['lesions']):
        return None, 'no true lesion'
    case = S['case']; rng = np.random.default_rng(zlib.crc32((ds + case + 's2ag').encode()))
    cols = {k: [] for k in COLS}; rrows = []; n = 0
    for W, Ls, kind in windows(S, max_false):
        n += build_window(ds, S, W, Ls, kb, psi, rng, cols, rrows, case)
    if n == 0:
        return None, 'no samples'
    J = max(a.shape[1] for a in cols['qL'])
    for k in ('qL', 'sdL', 'dtpL', 'uL', 'tL'):
        cols[k] = [np.pad(a, ((0, 0), (0, J - a.shape[1])), constant_values=np.nan) for a in cols[k]]
    arr = {k: np.concatenate(v) for k, v in cols.items() if v}
    meta = dict(ds=ds, case=case, source='agents', single_anchors=SINGLE[0], true_only=TRUE_ONLY[0], negative=not bool(S['T'].any()), baseline=S['baseline'], n_lesions=len(S['lesions']),
                n_true=sum(L['true'] for L in S['lesions']))
    np.savez_compressed(os.path.join(O, 'samples', f'{ds}__{case}.npz'), meta=json.dumps(meta), **arr)
    return rrows, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-per', type=int, default=100); ap.add_argument('--n-neg', type=int, default=50)
    ap.add_argument('--max-false', type=int, default=1000); ap.add_argument('--only', default=None)
    ap.add_argument('--two-only', action='store_true')
    ap.add_argument('--true-only', action='store_true', help='clusters of true lesions only -> agent_samples_true (1 Nov)')
    ap.add_argument('--psi', default=None, help='stage-1 extent prior JSON with a psi entry (ledger/v85/extent/fit_psi.json); default v8.4')
    a = ap.parse_args(); SINGLE[0] = not a.two_only; TRUE_ONLY[0] = a.true_only
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir('agent_samples_true' if a.true_only else 'agent_samples'); os.makedirs(os.path.join(O, 'samples'), exist_ok=True)
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(json.load(open(a.psi))['psi'] if a.psi else fit6['psi'])     # --psi: stage-1 extent prior (default v8.4)
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    done = set()
    for f in os.listdir(O):
        if re.fullmatch(r'done(_s\d+)?\.txt', f):
            done |= set(open(os.path.join(O, f)).read().splitlines())
    r2 = paths.root('R2Seg')
    neg = []
    for ds in EX.SPATIAL_SETS:
        neg += [t for t in EX.negative_list(r2) if t[0] == ds][:a.n_neg]
    tasks = EX.case_list(r2, a.n_per) + ([] if a.true_only else neg)     # tumour-free scans have no true lesion
    tasks = [t for i, t in enumerate(tasks) if i % SN == SI]
    if a.only:
        tasks = [t for t in tasks if t[0] in a.only.split(',')]
    t0 = time.time()
    for k, (ds, lf) in enumerate(tasks):
        key = ds + '/' + os.path.basename(lf)
        if key in done:
            continue
        note = None
        try:
            rr, note = build(ds, lf, KB, psi, O, a.max_false)
            if rr:
                f = os.path.join(O, f'r_rows{SUF}.csv')
                pd.DataFrame(rr).to_csv(f, mode='a', index=False, header=not os.path.exists(f))
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
