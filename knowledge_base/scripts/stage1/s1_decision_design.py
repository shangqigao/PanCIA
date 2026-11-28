"""Stage-1 decision, exploration on the DESIGN SET (28 Nov 2026; plan claude/stage1_decision_plan_2026-11-28.md parts 1b-2; workflow
claude/stage1_workflow.md step 6g). Exploratory and descriptive: nothing here is a frozen rule; the decision gate is written and
approved before jobs_v3.csv is touched. Never reads jobs_v3.csv, CancerVerse or TCGA.

Inputs: ledger/graph_v3/design.csv (4g: members, explanations, fragment groups, truth, scope) and
ledger/decision_s1/elong_lr[/<tag>]/candidates.csv (LOSO elongation logLR per real candidate; tumour-free design scans scored by the
fit that left their site out). Descriptive only: organ roles from the planner manifest (list T, host_class primary / distant; cancer
type enters the decision layer only, never graph building or pruning) and organ enlargement (host whole-organ territory volume from
the step-2g anatomy file / KB volume_ml max).

Group score (per fragment group, per mode): groups whose members are all explained (v3 'pruned') are rejected; otherwise
S = max logLR over the members that are not explained. Reject at threshold t: S < t. Curves over t (and 'v3' = explained-only):
T rejected %, worst site, tumour scans losing all T, F removed %, tumour-free false per scan (in scope as the v3 gate, and all
candidates). Tumour-free numbers are REPORTED, never used to pick a point. Reference point per mode: the largest t with T rejected
<= 5%, worst site <= 10%, scans losing all T <= 2% (the G1 limits; tumour scans only; every lower t must meet them too); common point = the smallest of those.
At the common point: stratified rejection (region class x band x mode, as G3) and the descriptive role / enlargement tables.
Output: ledger/decision_s1/design[/<tag>]/{curves.csv, strata.csv, roles.csv, enlargement.csv, summary.txt}.
28 Nov (plan claude/elongation_and_pelvic_fix_plan_2026-11-28.md, B3 as revised by the user): organ presence comes from the image
(anatomy layer) only; the cancer type only names the primary organ. Host = the step-2g g3 node 'pelvic_reproductive_unresolved' ->
role 'unresolved'; primary in view = 'yes' (the named primary organ is a usable organ of this scan), 'unknown' (it belongs to a
sex class and the scan has the unresolved node), else 'no'.
Usage: python s1_decision_design.py --lrdir elong_lr_e2 --design design_g3 --augdir augment_e2 --outdir design_e2 [--tag smoke]"""
import os, sys, json, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from pancia_kb.stage2 import paths     # noqa: E402
from pancia_kb import KnowledgeBase      # noqa: E402
from s1_graph_v3 import whole_of, EXCL_FLAGS   # noqa: E402
from s1_graph_v3_eval import region_class      # noqa: E402
from pancia_kb.anatomy_check_g import _class_of  # noqa: E402
NODE = 'pelvic_reproductive_unresolved'

L = paths.ledger_dir(); ANAT = os.path.join(paths.root('outputs'), 'anatomy', 'r2seg'); S = paths.root('R2Seg_Seg')
KBD = os.path.abspath(os.path.join(HERE, '..', '..'))
TS = [None] + [round(x, 2) for x in np.arange(-4, 4.01, 0.25)]


def organ_info(kb, ds, case):
    """usable whole organs: name -> (whole entity, side, approx territory ml); manifest T roles"""
    js = os.path.join(ANAT, ds, case + '.json'); W = {}; node = False
    if os.path.exists(js):
        meta = json.load(open(js)); node = bool(meta.get('unresolved'))
        for o in meta['organs']:
            if o['mu'] <= 0 or any(f.split(':')[0].split(' ')[0] in EXCL_FLAGS for f in o['flags']): continue
            we = whole_of(kb, o['entity']); wk = we + ('_' + o['side'] if o['side'] else '')
            w = W.setdefault(wk, dict(we=we, side=o['side'], self_ml=None, parts_ml=0.0))
            if o['entity'] == we: w['self_ml'] = max(w['self_ml'] or 0, o.get('terr_ml') or 0)
            else: w['parts_ml'] += o.get('terr_ml') or 0
    mf = os.path.join(S, 'VoxTell', ds, case + '_plan_manifest.json'); roles = []
    if os.path.exists(mf):
        roles = [(whole_of(kb, x['entity']), x.get('side'), x.get('host_class')) for x in json.load(open(mf)).get('masks', []) if x.get('list') == 'T']
    return W, roles, node


def role_of(h, W, roles, node=False):
    if h == NODE: return 'unresolved'
    if not isinstance(h, str) or h not in W: return 'none'
    we, sd = W[h]['we'], W[h]['side']; got = {c for e, s, c in roles if e == we and (s is None or sd is None or s == sd)}
    return 'primary' if 'primary' in got else ('distant' if got else 'none')


def decide(d, t):
    rej = d.g_all_expl if t is None else (d.g_all_expl | (d.g_score < t))
    return rej


def metrics(g, rej, n_tf):
    x = g[g.in_scope & g.lab.isin(['T', 'F']) & ~g.tumour_free]; rx = rej[x.index]
    T = x[x.lab == 'T']; rT = rx[T.index]; F = x[x.lab == 'F']
    site = rT.groupby(T.site).mean() * 100 if len(T) else pd.Series(dtype=float)
    lost = rT.groupby([T.ds, T.case]).all().mean() * 100 if len(T) else np.nan
    tfx = g[g.tumour_free]; tin = tfx[tfx.in_scope & (tfx.lab == 'F')]
    return dict(T_rej=rT.mean() * 100, worst_site=site.max() if len(site) else np.nan, worst_site_name=site.idxmax() if len(site) else '',
                lost=lost, F_rem=rej[F.index].mean() * 100, tf_fp_scope=(~rej[tin.index]).sum() / max(n_tf, 1),
                tf_fp_all=(~rej[tfx.index]).sum() / max(n_tf, 1), nT=len(T), nF=len(F))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default=''); ap.add_argument('--lrdir', default='elong_lr_e2')
    ap.add_argument('--design', default='design_g3'); ap.add_argument('--augdir', default='augment_e2'); ap.add_argument('--outdir', default='design_e2')
    a = ap.parse_args()
    ld = os.path.join(L, 'decision_s1', a.lrdir, a.tag) if a.tag else os.path.join(L, 'decision_s1', a.lrdir)
    od = os.path.join(L, 'decision_s1', a.outdir, a.tag) if a.tag else os.path.join(L, 'decision_s1', a.outdir); os.makedirs(od, exist_ok=True)
    c = pd.read_csv(os.path.join(ld, 'candidates.csv'), low_memory=False); c['case'] = c.case.astype(str)
    d = pd.read_csv(os.path.join(L, 'graph_v3', a.design + '.csv'), low_memory=False); d['case'] = d.case.astype(str)
    d = d[d.note.isna() | (d.note == '')].copy()
    sc = set(zip(c.ds, c.case)); d = d[[(u, v) in sc for u, v in zip(d.ds, d.case)]].copy()
    for k in ('in_scope', 'tumour_free'): d[k] = d[k].astype(str) == 'True'
    d['cid'] = d.cid.astype(int); c['cid'] = c.cid.astype(int)
    d = d.merge(c[['ds', 'case', 'mode', 'cid', 'llr', 'elong']], on=['ds', 'case', 'mode', 'cid'], how='left')
    out = [f'scans {d[["ds", "case"]].drop_duplicates().shape[0]} (tumour-free {d[d.tumour_free][["ds", "case"]].drop_duplicates().shape[0]}),'
           f' candidate rows {len(d)}, without logLR {int(d.llr.isna().sum())} (scored 0)']
    d['llr'] = d.llr.fillna(0.0); d = d.reset_index(drop=True)
    d['expl'] = d.member == 'explained'
    gk = ['ds', 'case', 'mode', 'group']
    d['g_all_expl'] = d.groupby(gk).expl.transform('all')
    d['g_score'] = d.llr.where(~d.expl, -np.inf).groupby([d[k] for k in gk]).transform('max')
    # tumour-free denominator: every tumour-free design scan in this run, including scans without any candidate (as the v3 gate)
    raw = pd.read_csv(os.path.join(L, 'graph_v3', a.design + '.csv'), usecols=['ds', 'case', 'tumour_free'], low_memory=False)
    raw['case'] = raw.case.astype(str); raw = raw[raw.tumour_free.astype(str) == 'True'][['ds', 'case']].drop_duplicates()
    ad = os.path.join(L, 'decision_s1', a.augdir, a.tag) if a.tag else os.path.join(L, 'decision_s1', a.augdir)
    tfs = pd.read_csv(os.path.join(ad, 'real_tf.csv'), usecols=['ds', 'case']); tfs = set(zip(tfs.ds, tfs.case.astype(str)))
    n_tf = sum((u, v) in tfs for u, v in raw.itertuples(index=False))
    out.append(f'tumour-free denominator {n_tf} scans (design file {len(raw)})')
    rows = []
    for mode, g in d.groupby('mode'):
        for t in TS:
            rows.append(dict(mode=mode, t='v3' if t is None else t, **metrics(g, decide(g, t), n_tf)))
    Cv = pd.DataFrame(rows).round(2); Cv.to_csv(os.path.join(od, 'curves.csv'), index=False)
    pick = {}
    for mode, g in Cv[Cv.t != 'v3'].groupby('mode'):
        g = g.assign(t=g.t.astype(float)).sort_values('t')
        good = ((g.T_rej <= 5) & (g.worst_site.fillna(0) <= 10) & (g.lost.fillna(0) <= 2)).to_numpy()
        k = int(np.argmin(good)) if not good.all() else len(good)   # contiguous from the lowest t: every t' <= t meets G1
        pick[mode] = float(g.t.iloc[k - 1]) if k > 0 else None
    show = ['v3', -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    out += ['\nCurves (selected thresholds; tumour-free columns reported only):',
            Cv[Cv.t.astype(str).isin([str(x) for x in show])].to_string(index=False),
            f'\nlargest t meeting the G1 limits per mode (tumour scans only): {pick}']
    valid = [v for v in pick.values() if v is not None]; tc = min(valid) if valid else None
    out.append(f'common reference point t = {tc}' + ('' if tc is not None else ' (none meets G1: reported at v3)'))
    rej = decide(d, tc)
    for mode, g in Cv[Cv.t.astype(str) == str(tc if tc is not None else 'v3')].groupby('mode'):
        out.append(f'  [{mode}] at common t: ' + ', '.join(f'{k} {v}' for k, v in g.iloc[0].items() if k not in ('mode', 't')))
    # G3-like strata at the common point
    x = d[d.in_scope & d.lab.isin(['T', 'F']) & ~d.tumour_free].copy(); x['rej'] = rej[x.index]
    x['rclass'] = [region_class(r) for r in x.itertuples()]
    x['band'] = pd.cut(x.ml, [0, 1, 10, 1e9], labels=['<1', '1-10', '>10'], right=False)
    st = []
    for (rc, band, lab), h in x.groupby(['rclass', 'band', 'lab'], observed=True):
        cnt = h.groupby('mode').size(); rate = h.groupby('mode').rej.mean() * 100
        tested = all(cnt.get(m, 0) >= 30 for m in ('BP', 'VT', 'BP+VT'))
        st.append(dict(region=rc, band=band, lab=lab, **{f'n_{m}': int(cnt.get(m, 0)) for m in ('BP', 'VT', 'BP+VT')},
                       **{f'rej_{m}': round(rate.get(m, np.nan), 1) for m in ('BP', 'VT', 'BP+VT')},
                       spread=round(rate.max() - rate.min(), 1) if tested else np.nan))
    St = pd.DataFrame(st); St.to_csv(os.path.join(od, 'strata.csv'), index=False)
    out += ['\nStratified rejection at the common point (strata with >= 30 per mode):', St[St.spread.notna()].to_string(index=False)]
    # descriptive: roles and enlargement
    kb = KnowledgeBase.load(f'{KBD}/knowledge_base'); info = {}
    for ds, case in d[['ds', 'case']].drop_duplicates().itertuples(index=False): info[(ds, case)] = organ_info(kb, ds, case)
    d['role'] = [role_of(h, *info[(u, v)]) for h, u, v in zip(d.host, d.ds, d.case)]
    cls = _class_of(kb)
    def enl(h, u, v):
        W = info[(u, v)][0]
        if not isinstance(h, str) or h not in W: return np.nan, np.nan
        w = W[h]; ml = w['self_ml'] if w['self_ml'] else w['parts_ml']; mx = (kb.entities.get(w['we'], {}).get('volume_ml') or {}).get('max')
        return ml, (ml / mx if mx and ml else np.nan)
    e = [enl(h, u, v) for h, u, v in zip(d.host, d.ds, d.case)]; d['host_ml'] = [x[0] for x in e]; d['host_ratio'] = [x[1] for x in e]
    piv = {}
    for (u, v), (W, roles, node) in info.items():
        prim = {(e_, s_) for e_, s_, c_ in roles if c_ == 'primary'}
        if any(w['we'] == e_ and (s_ is None or w['side'] is None or s_ == w['side']) for w in W.values() for e_, s_ in prim): piv[(u, v)] = 'yes'
        elif node and any(e_ in cls for e_, _ in prim): piv[(u, v)] = 'unknown'
        else: piv[(u, v)] = 'no'
    d['primary_in_view'] = [piv[(u, v)] for u, v in zip(d.ds, d.case)]
    d['rej'] = rej
    y = d[d.in_scope & d.lab.isin(['T', 'F']) & ~d.tumour_free]
    R = y.groupby(['mode', 'role', 'lab']).agg(n=('llr', 'size'), median_llr=('llr', 'median'), rejected_pct=('rej', 'mean')).reset_index()
    R['rejected_pct'] = (R.rejected_pct * 100).round(1); R['median_llr'] = R.median_llr.round(2); R.to_csv(os.path.join(od, 'roles.csv'), index=False)
    tfr = d[d.tumour_free].groupby(['mode', 'role']).agg(n=('llr', 'size'), kept=('rej', lambda s: int((~s).sum()))).reset_index()
    sc_ = y[y.lab == 'T'][['ds', 'case', 'primary_in_view']].drop_duplicates()
    y2 = y[y.host_ratio.notna()].copy(); y2['enlarged'] = y2.host_ratio > 1
    E = y2.groupby(['mode', 'role', 'enlarged', 'lab']).size().unstack('lab', fill_value=0).reset_index()
    E.to_csv(os.path.join(od, 'enlargement.csv'), index=False)
    hv = y2.groupby(['host', 'ds', 'case']).host_ratio.first().reset_index()
    out += ['\nOrgan roles (descriptive; manifest host_class; in-scope tumour-scan candidates):', R.to_string(index=False),
            '\nTumour-free design scans by role (kept at the common point):', tfr.to_string(index=False),
            f'\nTumour scans, primary organ in view (from the image): {sc_.primary_in_view.value_counts().to_dict()} of {len(sc_)}; by site: '
            f'{sc_.merge(d[["ds", "case", "site"]].drop_duplicates(), on=["ds", "case"]).groupby("site").primary_in_view.value_counts().unstack(fill_value=0).to_dict("index")}',
            '\nEnlargement (host whole-organ territory / KB volume_ml max; part sums are approximate):', E.to_string(index=False),
            f'hosts with ratio > 1: {(hv.host_ratio > 1).sum()}/{len(hv)}; by host: {hv[hv.host_ratio > 1].host.value_counts().head(10).to_dict()}']
    d[['ds', 'case', 'site', 'tumour_free', 'mode', 'cid', 'group', 'lab', 'in_scope', 'ml', 'member', 'host', 'llr', 'g_score',
       'g_all_expl', 'rej', 'role', 'host_ratio', 'primary_in_view']].to_csv(os.path.join(od, 'candidates_decision.csv'), index=False)
    txt = '\n'.join(out); print(txt); open(os.path.join(od, 'summary.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
