"""Stage 2, step 2: expert training samples from R2Seg (expert data only; Ovary_Tumor_00 is never read; no agent is run).

Per case (pancia_kb.stage2.expert): crop, largest expert tumour G and true host, image features; per simulated anchor (partial, sub,
cut2d): stage-1 extents and w0 (ledger v8.4 rule) -> q_v, chosen extent; per anatomy scenario (excl, incl, lost): spatial inputs of
every voxel, sampled (tumour voxels <= N_T, other voxels <= N_B, with weights count / sampled) within 60 mm of the anchor or in the
host territory. Also per case: farthest distance from each anchor to the tumour (for r), appearance statistics (tumour, organs,
unlabelled tissue) and a tumour-intensity sample (for nu).

Outputs (clinical/PanCIA_outputs/stage2/expert/):
  samples/<collection>__<case>.npz   columns: core, scen, t, q, sd, in_h, in_o, kb_h, dep_h, dep_o, d_anchor, ns, n1, w
  appearance[_s<i>].jsonl, r_rows[_s<i>].csv, done[_s<i>].txt, notes[_s<i>].csv
Usage: [SHARD=i/n] python3 s2_expert_build.py [--n-per 100] [--only Kidney_Tumor_00]   (resumable; run shards in parallel)"""
import os, sys, json, time, zlib, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))      # PanCIA/knowledge_base
from pancia_kb import grounding_g1 as G1, grounding_g2 as G2            # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, features as F, prior as PR   # noqa: E402

N_T, N_B = 2000, 6000            # samples per (anchor, scenario): tumour / other voxels (memory; weights keep the frequencies)
R_MAX = 60.0                     # sampling radius around the anchor (mm); r is estimated <= this (checked in s2_fit.py)


def build(ds, lf, psi, O):
    c = EX.load_case(ds, lf)
    if c is None:
        return None, None, None
    rng = np.random.default_rng(zlib.crc32((ds + c['case'] + 's2').encode()))
    sp = c['sp']
    Y = F.image_features(c['X'], sp, c['thick'], c['ax'])
    app = EX.appearance_stats(c, Y); del Y
    cores = EX.make_cores(c, rng); outl = EX.make_outlines(c, rng)
    ns, n1 = EX.neighbour_sums(c['T'], sp)
    scen_inputs = {s: EX.territories(c, s) for s in EX.SCENS}
    cols = {k: [] for k in ('core', 'scen', 't', 'q', 'sd', 'in_h', 'in_o', 'kb_h', 'dep_h', 'dep_o', 'd_anchor', 'ns', 'n1', 'w')}
    rrows = []
    for ci, (cn, a) in enumerate(cores.items()):
        ext, w0 = EX.extents(a, outl, c['others'], sp, G1, G2, psi)
        q = EX.q_map(ext, w0, a.shape)
        xi = ext[int(np.argmax(w0))][1]
        sd = F.signed_distance(xi, sp)
        da = F.edt(a, sp)
        rrows.append(dict(ds=ds, case=c['case'], core=cn, far_mm=float(da[c['G']].max()), core_ml=float(a.sum() * np.prod(sp) / 1000),
                          G_ml=float(c['G'].sum() * np.prod(sp) / 1000), xi=ext[int(np.argmax(w0))][0], n_ext=len(ext),
                          q_on_anchor_min=float(q[a].min())))
        for si_, sc in enumerate(EX.SCENS):
            theta, kb, seg, depth = scen_inputs[sc]
            in_h, in_o, kb_h, dep_h, dep_o = PR.host_inputs(theta, kb, seg, depth, c['hs'], a.shape)
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
    arr = {k: np.concatenate(v) for k, v in cols.items() if v}
    meta = dict(ds=ds, case=c['case'], mod=c['mod'], host=c['hs'], cls=c['cls'], sp=[float(x) for x in sp], cores=list(cores),
                scens=list(EX.SCENS), outlines=list(outl))
    np.savez_compressed(os.path.join(O, 'samples', f"{ds}__{c['case']}.npz"), meta=json.dumps(meta), **arr)
    return meta, app, rrows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-per', type=int, default=100); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir('expert'); os.makedirs(os.path.join(O, 'samples'), exist_ok=True)
    fit6 = json.load(open(os.path.join(paths.ledger_dir(), 'v8', 'testbed', 'fit_host6.json')))
    psi = G2.Psi(fit6['psi'])
    done = set()
    for f in os.listdir(O):
        if f.startswith('done'):
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
            meta, app, rr = build(ds, lf, psi, O)
            if meta is None:
                pd.DataFrame([dict(key=key, note='skip')]).to_csv(os.path.join(O, f'notes{SUF}.csv'), mode='a', index=False,
                                                                 header=not os.path.exists(os.path.join(O, f'notes{SUF}.csv')))
            else:
                with open(os.path.join(O, f'appearance{SUF}.jsonl'), 'a') as fh:
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
