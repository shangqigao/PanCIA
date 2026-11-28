"""Fresh evaluation sample for a future explanation / graph-reasoning v3 gate (27 Nov 2026; user: run the general anatomy check on the
v1/v2 samples and on a fresh v3 sample, claude/stage1_workflow.md). Same recipe as s1_explanation_v2.make_jobs (20 tumour scans per
site, lung pooled, one image per patient, Uterus_Tumor_00 excluded; plus the 139 tumour-free 'negative' scans of
cohort/r2seg_tumour_free.csv), seed 3, with patients disjoint from jobs.csv (v1), jobs_v2.csv (v2) and the expert-tumour calibration
set (expert_tumour_geom.csv). Cervix has no unused patients left. Not used for anything until a v3 gate is approved.
Output: ledger/evidence_screen/jobs_v3.csv (ds, case, tumour_free). Usage: python s1_make_jobs_v3.py   (refuses to overwrite)"""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from pancia_kb.stage2 import paths            # noqa: E402
import s1_evidence_screen as SC                # noqa: E402
import s1_explanation_v2 as X2                 # noqa: E402

OUT = SC.OUT; JF = os.path.join(OUT, 'jobs_v3.csv')


def make_jobs(n_per=20, seed=3):
    used = set()
    for f in ('jobs.csv', 'jobs_v2.csv', 'expert_tumour_geom.csv'):
        D = pd.read_csv(os.path.join(OUT, f), usecols=['ds', 'case'])
        used |= {(d, SC.patient(d, str(c))) for d, c in zip(D.ds, D.case)}
    C = pd.read_csv(os.path.join(os.path.dirname(paths.ledger_dir()), 'cohort', 'r2seg_tumour_free.csv'))[['ds', 'case', 'status']]
    C['case'] = C.case.astype(str)
    t = pd.read_csv(os.path.join(paths.ledger_dir(), 'v86_r2seg_bp2', 'diag', 'anchor_truth_lite.csv'), usecols=['ds', 'case'],
                    low_memory=False).drop_duplicates()
    t = t[t.ds != 'Uterus_Tumor_00']; t['case'] = t.case.astype(str)
    t = t.merge(C, on=['ds', 'case'], how='left'); t['status'] = t.status.fillna('tumour')
    neg = t[t.status == 'negative'].assign(tumour_free=True)
    tum = t[t.status == 'tumour'].copy(); tum['site'] = tum.ds.map(X2.EX.SITE_ENTITY)
    tum['pat'] = [SC.patient(d, c) for d, c in zip(tum.ds, tum.case)]
    tum = tum[[(d, p) not in used for d, p in zip(tum.ds, tum.pat)]]
    out = [g.sample(frac=1, random_state=seed).drop_duplicates('pat').head(n_per) for _, g in tum.groupby('site')]
    tum = pd.concat(out).assign(tumour_free=False)
    return pd.concat([tum[['ds', 'case', 'tumour_free']], neg[['ds', 'case', 'tumour_free']]], ignore_index=True)


if __name__ == '__main__':
    if os.path.exists(JF):
        sys.exit(f'{JF} exists; not overwritten')
    J = make_jobs(); J.to_csv(JF, index=False)
    print(len(J), J.tumour_free.sum()); print(J[~J.tumour_free].groupby('ds').size().to_dict())
