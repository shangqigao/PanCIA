"""Assemble KB geometry priors (v1) from the per-case statistics:
  clinical/PanCIA_outputs/geometry_priors/anatomy_cases.csv  (TotalSegmentator CT v3.0.0 / MR v2.0.0 expert labels)
  clinical/PanCIA_outputs/geometry_priors/tumour_cases.csv   (R2Seg public datasets; OV04 excluded)
-> knowledge_base/knowledge_base/geometry_priors.yaml
Quantiles (5, 25, 50, 75, 95). Tumour priors are pooled by host architecture with every dataset weighted equally (so
no single dataset's conventions dominate); per-dataset quantiles are kept under `sources` for transparency only.
Descriptors are defined in pancia_kb/geometry.py (v1); priors and inference must use the same functions."""
import os, yaml, numpy as np, pandas as pd
H = os.path.expanduser('~'); D = f'{H}/mnt/clinical/PanCIA_outputs/geometry_priors'
KB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_base', 'geometry_priors.yaml')
Q = [5, 25, 50, 75, 95]
SHAPE = ['sphericity', 'solidity', 'elong', 'flat', 'irr_rms', 'irr_p95', 'irr_max']
REL = ['vol_ratio', 'contact_frac', 'lumen_contact_frac', 'outside_hull_frac', 'depth_rel']
ENT = {'kidney_left': 'kidney', 'kidney_right': 'kidney', 'lung_left': 'lung', 'lung_right': 'lung', 'thyroid_gland': 'thyroid'}


def wq(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]; c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [round(float(np.interp(p / 100, c, x)), 4) for p in q]


def block(df, cols, wcol=None):
    out = {}
    for c in cols:
        s = df[[c] + ([wcol] if wcol else [])].dropna()
        if len(s) < 10: continue
        w = s[wcol].values if wcol else np.ones(len(s))
        out[c] = dict(zip([f'q{q}' for q in Q], wq(s[c].values, w, Q))) | {'n': int(len(s))}
    return out


a = pd.read_csv(f'{D}/anatomy_cases.csv'); a = a[a.note.isna() & a.organ.notna()]
a['entity'] = a.organ.map(lambda o: ENT.get(o, o))
t = pd.read_csv(f'{D}/tumour_cases.csv'); t = t[t.main_ml.notna()]
t['w'] = 1.0 / t.groupby('dataset').case.transform('size')
anat = {}
for e, g in a.groupby('entity'):
    anat[e] = dict(n_cases=int(len(g)), n_ct=int((g.source == 'CT').sum()), n_mr=int((g.source == 'MR').sum()),
                   hollow=bool(g.hollow.max() == 1), vol_ml=block(g, ['vol_ml'])['vol_ml'], shape=block(g, SHAPE),
                   max_spacing_mm=block(g, ['max_sp'])['max_sp'])
tum = {}
for arch, g in t.groupby('architecture'):
    gr = g[g.contact_frac.notna()]
    tum[arch] = dict(datasets=sorted(g.dataset.unique().tolist()), n_cases=int(len(g)), weighting='equal per dataset',
                     main_ml=block(g, ['main_ml'], 'w')['main_ml'], shape=block(g, SHAPE, 'w'),
                     multiplicity={k: {str(v): round(float(g.loc[g[k] == v, 'w'].sum() / g.w.sum()), 4) for v in sorted(g[k].clip(upper=5).dropna().unique())}
                                   for k in ['n_comp', 'n_satellite', 'n_distant']},
                     relation=dict(datasets=sorted(gr.dataset.unique().tolist()), **block(gr.assign(w=1.0 / gr.groupby('dataset').case.transform('size')), REL, 'w')))
src = {ds: dict(architecture=g.architecture.iloc[0], host=g.host.iloc[0], modality=g.modality.iloc[0], n=int(len(g)),
                main_ml_median=round(float(g.main_ml.median()), 2), shape_median={c: round(float(g[c].median()), 4) for c in SHAPE})
       for ds, g in t.groupby('dataset')}
Y = dict(version='1.0', date='2026-10-10', author='Shangqi Gao (Crispin Lab) with Claude',
         principle=['Priors come only from expert annotations, never from any segmentation model output (agents are pluggable).',
                    'Geometry only (intensity-agnostic); appearance is modelled within each scan at inference.',
                    'Agent errors (any agent, any cohort) are handled at inference by treating agent masks as noisy observations.',
                    'Tumour priors are pooled by host architecture (pan-cancer), datasets weighted equally.',
                    'Sex is never used. Only public data: R2Seg OV04 (ovary) excluded as confidential.'],
         descriptors='pancia_kb/geometry.py v1 (iso grid 1.5 mm; irregularity smoothing 0.25 r_eq)',
         sources=dict(anatomy=['TotalSegmentator CT dataset v3.0.0 (Zenodo)', 'TotalSegmentator MRI dataset v2.0.0 (Zenodo)'],
                      anatomy_filter='CT: pathology none, or outside the organ region and not unclear; MR: no pathology field (kept); organs truncated by the field of view skipped',
                      tumour='R2Seg public datasets (FedBCa, MAMA-MIA, CC-Tumor-Heterogeneity, MSD Colon/Lung/Liver/Pancreas (FLARE23 files), KiTS23, NSCLC-Radiomics, Prostate158, UMD)'),
         caveats=['Hollow-organ wall thickness from a within-scan two-class split was implausible (median colon 14 mm, stomach 20 mm); not stored.',
                  'Organ-outline deformation measured by generic size-relative irregularity is NOT larger near tumours than elsewhere (bulge ratio median < 1 for solid hosts); deformation evidence needs organ-specific shape models with correspondence.',
                  'Tumour shape differs significantly between datasets within an architecture (leave-one-dataset-out KS); the architecture prior is broad. Elongation transfers best.',
                  'Hollow-organ tumour-host relations come only from uterus/cervix datasets (no bladder/colon organ labels).'],
         anatomy=anat, tumour=tum, tumour_sources=src)
yaml.safe_dump(Y, open(KB, 'w'), sort_keys=False, width=120)
print('->', KB, 'anatomy entities', len(anat), 'architectures', list(tum))
