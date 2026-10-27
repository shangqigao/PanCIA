"""Stage 2, step 2b: tail shape (Student-t nu) of tumour tissue, from the expert tumours of the spatial build's cases.

Per tumour >= 0.5 ml (pancia_kb.stage2.expert.tumour_features: the two stage-2 features, same normalisation; <= 5000 voxels):
a Student-t mixture as stage 2 models a tumour (fit.fit_tumour_components: 1..4 components chosen by the variational lower bound,
nu learned per component). A tumour may hold a single tissue (e.g. fully necrotic) or several; no tissue type is assumed or named.
s2_fit.py pools the components into one population prior (fit.pooled_nu: on 1/nu; sites, cases, tumours, components equal).
Outputs clinical/PanCIA_outputs/stage2/expert/nu_components[_s<i>].jsonl, done_nu[_s<i>].txt (resumable).
Usage: [SHARD=i/n] python3 s2_nu.py [--n-per 100]"""
import os, sys, json, time, zlib, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import numpy as np   # noqa: E402
from pancia_kb.stage2 import paths, expert as EX, fit as FT   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-per', type=int, default=100)
    a = ap.parse_args()
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir('expert')
    done = set()
    for f in os.listdir(O):
        if f.startswith('done_nu'):
            done |= set(open(os.path.join(O, f)).read().splitlines())
    tasks = [t for i, t in enumerate(EX.case_list(paths.root('R2Seg'), a.n_per)) if i % SN == SI]   # the spatial build's cases
    t0 = time.time()
    for k, (ds, lf) in enumerate(tasks):
        key = ds + '/' + os.path.basename(lf)
        if key in done:
            continue
        try:
            rng = np.random.default_rng(zlib.crc32((key + 'nu').encode()))
            r = EX.tumour_features(ds, lf, rng)
            if r is not None:
                meta, tums = r
                with open(os.path.join(O, f'nu_components{SUF}.jsonl'), 'a') as fh:
                    for j, ml, Y in tums:
                        res = FT.fit_tumour_components(Y)
                        fh.write(json.dumps(dict(meta, j=j, ml=round(ml, 2), n_vox=len(Y), **res)) + '\n')
        except Exception:
            with open(os.path.join(O, f'notes_nu{SUF}.txt'), 'a') as fh:
                fh.write(key + ' ' + traceback.format_exc()[-300:].replace('\n', ' | ') + '\n')
        with open(os.path.join(O, f'done_nu{SUF}.txt'), 'a') as fh:
            fh.write(key + '\n')
        print(f'{k + 1}/{len(tasks)} {key} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
