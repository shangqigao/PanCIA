"""Stage 2, step 2a: usability index of every R2Seg label file (expert labels only; no image is read).

Per file (pancia_kb.stage2.expert.label_status): status
  usable       largest expert tumour component >= 0.5 ml (expert.MIN_ML)
  small        tumour present, largest component < 0.5 ml
  no_tumour    no tumour label (e.g. FLARE23 scans with organ labels only, PI-RADS-negative prostates): validation negatives
  normal_only  Uterus_Tumor_00 (benign myoma; not used by stage 2)
the number of tumour components (26-connectivity), the number of tumours >= 0.5 ml, the largest component's volume, the sequence tag (Prostate t2 / adc) and whether a
TotalSegmentator mask exists (R2Seg_Seg; required for the spatial prior, which uses TS anatomy only).
The spatial build draws at most n_per files at random from the usable files of each collection.
Outputs clinical/PanCIA_outputs/stage2/expert/index/r2seg_index[_s<i>].csv (resumable).
Usage: [SHARD=i/n] python3 s2_r2seg_index.py"""
import os, sys, time, traceback
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402

COLS = ['ds', 'file', 'seq', 'ts', 'status', 'n_comp', 'n_tumours', 'largest_ml', 'note']


def main():
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
    O = paths.out_dir('expert', 'index'); fo = os.path.join(O, f'r2seg_index{SUF}.csv')
    done = set()
    for f in os.listdir(O):
        if f.startswith('r2seg_index') and f.endswith('.csv'):
            d0 = pd.read_csv(f if os.path.isabs(f) else os.path.join(O, f)); done |= set(zip(d0.ds, d0.file))
    r2 = paths.root('R2Seg')
    tasks = [t for i, t in enumerate(EX.index_files(r2)) if i % SN == SI and t not in done]
    t0 = time.time()
    for k, (ds, f) in enumerate(tasks):
        case = f.replace('_seg.nii.gz', '')
        try:
            r = EX.label_status(ds, os.path.join(r2, ds, 'labels', f))
        except Exception:
            r = dict(status='error', note=traceback.format_exc()[-300:].replace('\n', ' | '))
        ts = all(os.path.exists(p) for p in EX.ts_paths(ds, case))
        row = dict(ds=ds, file=f, seq=EX._seqtag(f), ts=ts, status=r['status'], n_comp=r.get('n_comp', 0),
                   n_tumours=r.get('n_tumours', 0), largest_ml=r.get('largest_ml', 0.0), note=r.get('note', ''))
        pd.DataFrame([row], columns=COLS).to_csv(fo, mode='a', index=False, header=not os.path.exists(fo))   # fixed columns
        if (k + 1) % 50 == 0 or k + 1 == len(tasks):
            print(f'{k + 1}/{len(tasks)} {ds}/{f} {time.time() - t0:.0f}s', flush=True)
    if SN == 1:                                   # summary (run once more without SHARD after the shards: nothing left to do)
        d = EX.read_index()
        print(d.groupby(['ds', 'seq', 'status']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()
