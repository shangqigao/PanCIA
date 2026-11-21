"""Do the segmentation agents outline renal cysts as tumour? (16 Nov 2026; claude/rejected_tumours_diagnosis_2026-11-16.md).
R2Seg Kidney_Tumor_00 (KiTS23) labels: 1 kidney, 2 tumour, 3 cyst. For every scan and each agent (BiomedParse, VoxTell), every connected
component (26-connected) of the agent's tumour mask on the image grid is classified by the expert label it overlaps: 'tumour' if it
touches label 2, else 'cyst' if it touches label 3, else 'kidney' if >= 50% inside label 1, else 'outside'. Volumes in ml; overlap
fractions with each label. Expert labels: evaluation only.
Output: ledger/v86_r2seg/diag/agent_components_kidney.csv (resumable; chunk with `timeout`). Usage: timeout 170 python s1_agent_cyst_check.py"""
import os, sys, glob, time, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths, expert as EX, agents as AG   # noqa: E402
DS = 'Kidney_Tumor_00'


def main():
    D = os.path.join(paths.ledger_dir(), 'v86_r2seg', 'diag'); f_out = os.path.join(D, 'agent_components_kidney.csv'); f_done = os.path.join(D, 'cyst_done.txt')
    done = set(open(f_done).read().split()) if os.path.exists(f_done) else set()
    stems = sorted(os.path.basename(f)[:-len('_seg.nii.gz')] for f in glob.glob(os.path.join(paths.root('R2Seg'), DS, 'labels', '*_seg.nii.gz')))
    t0 = time.time()
    for stem in stems:
        if stem in done:
            continue
        rows = []
        try:
            lf = os.path.join(paths.root('R2Seg'), DS, 'labels', stem + '_seg.nii.gz')
            L = EX._read_labels(lf).astype(np.int8); img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz'))
            vml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
            n_cyst = int(ndimage.label(L == 3, structure=EX.STRUCT26)[1])
            for ag in ('BiomedParse', 'VoxTell'):
                A, _ = AG._on_grid(AG.mask_path(ag, DS, stem), img)
                if A is None or not A.any():
                    continue
                lab, n = ndimage.label(A, structure=EX.STRUCT26)
                objs = ndimage.find_objects(lab)
                for k, sl in enumerate(objs, 1):
                    c = lab[sl] == k; l = L[sl][c]; v = c.sum()
                    f = {j: float((l == j).sum()) / v for j in (1, 2, 3)}
                    cls = 'tumour' if f[2] > 0 else 'cyst' if f[3] > 0 else 'kidney' if f[1] >= 0.5 else 'outside'
                    rows.append(dict(case=stem, agent=ag, comp=k, ml=v * vml, cls=cls, f_kidney=round(f[1], 3), f_tumour=round(f[2], 3),
                                     f_cyst=round(f[3], 3), scan_has_cyst=n_cyst > 0))
        except Exception:
            print('error', stem, traceback.format_exc()[-300:], flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(f_out, mode='a', index=False, header=not os.path.exists(f_out))
        open(f_done, 'a').write(stem + '\n')
    print('done', len(stems), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
