"""Stage 2, step 2: spatial-prior training samples from R2Seg. Tumour truth: expert labels. Anatomy: TotalSegmentator only (user,
25 Oct): territories and the organs the simulated outlines are drawn around, as on TCGA (masks from m_tumor_segmentation.py
--dataset R2Seg, root PANCIA_R2SEG_SEG). Expert organ labels are not used for fitting (validation 'excl' only).
Cases: up to --n-per drawn at random from the usable cases of each spatial collection (usability index, s2_r2seg_index.py).

Per case (pancia_kb.stage2.expert): crop, every expert tumour >= 0.5 ml as a lesion (expert.simulate_lesions; user, 25 Oct), image
features; per core type (partial, sub, cut2d) each lesion gets an anchor and agent claims (a random subset of over- and
under-segmenting outline types), stage-1 extents and w0 (ledger v8.4 rule) -> q_L, chosen extent; the lesions form one cluster
(q = 1 - prod(1 - q_L), as run.py). Spatial inputs of every voxel under the TS anatomy, sampled (tumour voxels <= N_T, other voxels
<= N_B, with weights count / sampled) within 60 mm of any anchor or in the host territory; per-lesion q_L and sd_L are stored
(qL, sdL; NaN-padded) for the noisy-OR fit.
Also per case: farthest distance from each anchor to the tumour (for r) and a sample of the tumour's smoothed intensity (for the
Student-t tail shape nu; no appearance prior is fitted: appearance is inferred within each scan).

Outputs (clinical/PanCIA_outputs/stage2/expert/):
  samples/<collection>__<case>.npz   columns: core, scen, t, q, sd, in_h, in_o, kb_h, dep_h, dep_o, d_anchor, ns, n1, w, qL, sdL
  nu_samples[_s<i>].jsonl, r_rows[_s<i>].csv, done[_s<i>].txt, notes[_s<i>].csv
Usage: [SHARD=i/n] python3 s2_expert_build.py [--n-per 100] [--only Kidney_Tumor_00]   (resumable; run shards in parallel)"""
import os, re, sys, json, time, zlib, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))      # PanCIA/knowledge_base
from pancia_kb import KnowledgeBase, grounding_g1 as G1, grounding_g2 as G2   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, prior as PR   # noqa: E402

N_T, N_B = 2000, 6000            # samples per (anchor, scenario): tumour / other voxels (memory; weights keep the frequencies)
R_MAX = 60.0                     # sampling radius around the anchor (mm); r is estimated <= this (checked in s2_fit.py)


def _sd_window(xi, sp, shape):
    """signed distance to xi on its box padded by the basis' clip (60 mm); 60 elsewhere (the basis clips sd at 60)"""
    b = F.bbox(xi)
    out = np.full(shape, 60.0, np.float32)
    if b is None:
        return out
    W = F.pad_box(b, sp, 61.0, shape)
    out[W] = np.minimum(F.signed_distance(xi[W], sp), 60.0)
    return out


def build(ds, lf, psi, O, kb=None):
    c = EX.load_case(ds, lf, kb=kb)
    if c is None:
        return None, None, None
    rng = np.random.default_rng(zlib.crc32((ds + c['case'] + 's2').encode()))
    sp = c['sp']; shape = c['T'].shape
    Y = F.image_features(c['X'], sp, c['thick'], c['ax'])
    app = EX.tumour_sample(c, Y); del Y
    sim = EX.simulate_lesions(c, rng, psi, G1, G2)                       # {core type: [lesion]} (every tumour >= 0.5 ml)
    ns, n1 = EX.neighbour_sums(c['T'], sp)
    scen_inputs = {s: EX.territories(c, s) for s in EX.FIT_SCENS}       # TS anatomy only (user, 25 Oct)
    J = max(len(v) for v in sim.values()) if sim else 0
    cols = {k: [] for k in ('core', 'scen', 't', 'q', 'sd', 'in_h', 'in_o', 'kb_h', 'dep_h', 'dep_o', 'd_anchor', 'ns', 'n1', 'w',
                            'qL', 'sdL')}
    rrows = []
    for cn, les in sim.items():
        ci = EX.CORES.index(cn)
        # cluster of the scan's lesions, combined as run.py combines a stage-1 cluster: q = 1 - prod(1 - q_L); per-lesion q_L and
        # sd_L are kept for the noisy-OR fit of eta (pi = 1 - prod_L(1 - pi_L), as at inference)
        omq = np.ones(shape, np.float32); xi = np.zeros(shape, bool); A = np.zeros(shape, bool)
        for L in les:
            omq *= 1.0 - EX.full(L['q'], shape, np.float32); xi |= EX.full(L['xi'], shape); A |= EX.full(L['a'], shape)
            rrows.append(dict(ds=ds, case=c['case'], core=cn, comp=L['j'], n_lesions=len(les), far_mm=L['far_mm'],
                              core_ml=float(L['a'][1].sum() * np.prod(sp) / 1000),
                              G_ml=float(c['comps'][L['j']].sum() * np.prod(sp) / 1000), xi=L['xi_name'], n_ext=L['n_ext'],
                              claims='+'.join(L['claims']), q_on_anchor_min=L['q_on_anchor_min']))
        q = 1.0 - omq; del omq
        sd = _sd_window(xi, sp, shape)
        da = F.edt(A, sp)
        takes = []
        for sc in EX.FIT_SCENS:
            si_ = EX.SCENS.index(sc)
            theta, kbf, seg, depth, hkey = scen_inputs[sc]
            in_h, in_o, kb_h, dep_h, dep_o = PR.host_inputs(theta, kbf, seg, depth, hkey, shape)
            R = (da <= R_MAX) | in_h
            for lab_, nmax in ((True, N_T), (False, N_B)):
                idx = np.flatnonzero((R & (c['T'] == lab_)).ravel())
                if len(idx) == 0:
                    continue
                take = idx if len(idx) <= nmax else rng.choice(idx, nmax, replace=False)
                w = len(idx) / len(take)
                cols['core'].append(np.full(len(take), ci, np.int8)); cols['scen'].append(np.full(len(take), si_, np.int8))
                cols['t'].append(np.full(len(take), lab_, np.int8))
                for nm, arr in (('q', q), ('sd', sd), ('in_h', in_h), ('in_o', in_o), ('dep_h', dep_h), ('dep_o', dep_o),
                                ('d_anchor', da), ('ns', ns), ('n1', n1)):
                    cols[nm].append(np.asarray(arr).ravel()[take].astype(np.float32))
                cols['kb_h'].append(np.full(len(take), kb_h, np.float32)); cols['w'].append(np.full(len(take), w, np.float32))
                takes.append(take)
        # per-lesion q_L and sd_L at the sampled voxels (each lesion's maps computed once)
        allt = np.concatenate(takes) if takes else np.zeros(0, np.int64)
        qL = np.full((len(allt), J), np.nan, np.float32); sdL = np.full((len(allt), J), np.nan, np.float32)
        for k_, L in enumerate(les):
            qL[:, k_] = EX.full(L['q'], shape, np.float32).ravel()[allt]
            sdL[:, k_] = (sd if len(les) == 1 else _sd_window(EX.full(L['xi'], shape), sp, shape)).ravel()[allt]
        cols['qL'].append(qL); cols['sdL'].append(sdL)
    arr = {k: np.concatenate(v) for k, v in cols.items() if v}
    meta = dict(ds=ds, case=c['case'], mod=c['mod'], seq=c['seq'], host=c['hs'] or EX.SITE_ENTITY.get(ds), cls=c['cls'],
                sp=[float(x) for x in sp], cores=list(sim), scens=list(EX.SCENS), scens_used=list(EX.FIT_SCENS),
                n_tumours=len(c['comps']), lesions_per_core={k: len(v) for k, v in sim.items()},
                claims={k: [L['claims'] for L in v] for k, v in sim.items()},
                ts_host_found=(c['ts']['host_found'] if 'ts' in c else None))
    np.savez_compressed(os.path.join(O, 'samples', f"{ds}__{c['case']}.npz"), meta=json.dumps(meta), **arr)
    return meta, app, rrows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-per', type=int, default=100); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir('expert'); os.makedirs(os.path.join(O, 'samples'), exist_ok=True)
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(fit6['psi'])
    KB = KnowledgeBase.load(os.path.join(paths.kb_dir(), 'knowledge_base'))
    done = set()
    for f in os.listdir(O):
        # only this step's resume lists (done.txt, done_s<i>.txt); an earlier version also read the (now removed) appearance pass's
        # done_app*.txt and skipped every case that pass had done (bug found at review point 1, 25 Oct)
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
        try:
            meta, app, rr = build(ds, lf, psi, O, KB)
            if meta is None:
                pd.DataFrame([dict(key=key, note='skip')]).to_csv(os.path.join(O, f'notes{SUF}.csv'), mode='a', index=False,
                                                                 header=not os.path.exists(os.path.join(O, f'notes{SUF}.csv')))
            else:
                with open(os.path.join(O, f'nu_samples{SUF}.jsonl'), 'a') as fh:
                    fh.write(json.dumps(app) + '\n')
                f = os.path.join(O, f'r_rows{SUF}.csv')
                pd.DataFrame(rr).to_csv(f, mode='a', index=False, header=not os.path.exists(f))
        except Exception:
            f = os.path.join(O, f'notes{SUF}.csv')
            pd.DataFrame([dict(key=key, note=traceback.format_exc()[-400:].replace('\n', ' | '))]).to_csv(
                f, mode='a', index=False, header=not os.path.exists(f))
        with open(os.path.join(O, f'done{SUF}.txt'), 'a') as fh:
            fh.write(key + '\n')
        print(f'{k + 1}/{len(tasks)} {key} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
