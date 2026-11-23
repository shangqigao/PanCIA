"""Label-free stage-1 profile, R2Seg (in-distribution) vs TCGA (OOD), per site.
Reads ledger anchors/series shards; writes JSON for the dashboard.
Usage: python d1_r2seg_vs_tcga.py [--out PATH]"""
import argparse, glob, json, os
import numpy as np, pandas as pd
OUT = os.path.expanduser('~/mnt/PanCIA_outputs')
LED = {'R2Seg': 'ledger/v86_r2seg_bp2', 'TCGA': 'ledger/v86_tcga_all_bp2'}
R2 = {'Breast_Tumor_00':'breast','Pancreas_Tumor_00':'pancreas','Kidney_Tumor_00':'kidney','Lung_Tumor_01':'lung',
      'Lung_Tumor_00':'lung','Uterus_Tumor_00':'uterus','Bladder_Tumor_00':'bladder','Liver_Tumor_00':'liver',
      'Prostate_Tumor_00':'prostate','Colon_Tumor_00':'colon','Cervix_Tumor_00':'cervix'}
TC = {'BLCA':'bladder','BRCA':'breast','CESC':'cervix','COAD':'colon','ESCA':'oesophagus','KICH':'kidney','KIRC':'kidney',
      'KIRP':'kidney','LIHC':'liver','LUAD':'lung','LUSC':'lung','OV':'ovary','PRAD':'prostate','STAD':'stomach','UCEC':'uterus'}
COLS = ['rel','cancer','mod','kind','agents','ml','decision','level','host_cls','extents']

def site(ds, c): return R2.get(c) if ds == 'R2Seg' else TC.get(str(c).replace('TCGA-', ''))

def load(ds):
    L = os.path.join(OUT, LED[ds])
    a = pd.concat([pd.read_csv(f, usecols=COLS) for f in sorted(glob.glob(L + '/anchors_s*.csv'))], ignore_index=True)
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(L + '/series_s*.csv'))], ignore_index=True)
    s = s[s.qc != 'exclude'].reset_index(drop=True)
    a = a[a.rel.isin(set(s.rel))].reset_index(drop=True)
    s['site'] = [site(ds, c) for c in s.cancer]; a = a.drop(columns=['cancer']).merge(s[['rel','cancer','site']], on='rel', how='left')
    s['group'] = 'tumour'
    if ds == 'R2Seg':
        t = pd.read_csv(os.path.join(OUT, 'cohort/r2seg_tumour_free.csv'))
        st = {(r.ds, r.case): r.status for r in t.itertuples()}
        s['group'] = [{'negative':'negative','unannotated':'unannotated','tumour_adc_only':'unannotated'}.get(
            st.get((c, r.split('/')[-1])), 'tumour') for c, r in zip(s.cancer, s.rel)]
    a = a.merge(s[['rel', 'group']], on='rel', how='left')
    def amax(e):
        try: return max(x['ml'] for x in json.loads(e))
        except Exception: return np.nan
    acc = a.decision == 'accepted'
    a.loc[acc, 'cand_max'] = a.loc[acc, 'extents'].map(amax)
    a['anc_ratio'] = a.ml / a.cand_max
    return a.drop(columns='extents'), s

def prof(a, s):
    n = len(s)
    if n == 0: return None
    two = (a.kind == 'two').groupby(a.rel).any().reindex(s.rel, fill_value=False)
    ml = a.groupby(['rel', 'agents']).ml.sum().unstack(fill_value=0).reindex(s.rel, fill_value=0)
    for k in ['BP', 'VT', 'BP+VT']:
        if k not in ml: ml[k] = 0.0
    tot = ml.sum(1)
    agr = (ml['BP+VT'] / tot.replace(0, np.nan))
    ac = a[a.decision == 'accepted']
    dec = a.decision.value_counts(normalize=True)
    kmix = a.kind.value_counts(normalize=True)
    acc_n = ac.groupby('rel').size().reindex(s.rel, fill_value=0)
    unc_n = a[a.decision == 'uncertain'].groupby('rel').size().reindex(s.rel, fill_value=0)
    r = ac.anc_ratio.dropna()
    return dict(
        n_series=int(n), n_anchors=int(len(a)), anchors_per_series=round(len(a) / n, 1),
        pct_two=round(100 * two.mean(), 1),
        agree_vol_median=round(float(agr.median()), 3) if agr.notna().any() else None,
        bp_ml_median=round(float(ml['BP'].median()), 1), vt_ml_median=round(float(ml['VT'].median()), 1),
        two_ml_median=round(float(ml['BP+VT'].median()), 1),
        kind={k: round(100 * float(kmix.get(k, 0)), 1) for k in ['two', 'single', 'lobe']},
        dec={k: round(100 * float(dec.get(k, 0)), 1) for k in ['accepted', 'uncertain', 'rejected']},
        pct_flagged=round(100 * (acc_n > 0).mean(), 1), pct_any_unc=round(100 * (unc_n > 0).mean(), 1),
        acc_per_series=round(float(acc_n.mean()), 2),
        acc_level2=round(100 * float((ac.level == 2).mean()), 1) if len(ac) else None,
        acc_primary=round(100 * float((ac.host_cls == 'primary').mean()), 1) if len(ac) else None,
        ratio_median=round(float(r.median()), 2) if len(r) else None,
        ratio_lt05=round(100 * float((r < 0.5).mean()), 1) if len(r) else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(OUT, 'dashboard/r2seg_vs_tcga.json'))
    o = ap.parse_args(); res = {}
    for ds in LED:
        a, s = load(ds)
        for grp in sorted(s.group.unique()):
            ss = s[s.group == grp]; aa = a[a.group == grp]
            res.setdefault(ds, {}).setdefault(grp, {})['ALL'] = prof(aa, ss)
            for st in sorted(ss.site.dropna().unique()):
                res[ds][grp][st] = prof(aa[aa.site == st], ss[ss.site == st])
            for m in ['CT', 'MR']:
                res[ds][grp]['mod_' + m] = prof(aa[aa['mod'] == m], ss[ss['mod'] == m])
        print(ds, 'done', flush=True)
    os.makedirs(os.path.dirname(o.out), exist_ok=True)
    json.dump(res, open(o.out, 'w'), indent=1); print(o.out)

if __name__ == '__main__': main()
