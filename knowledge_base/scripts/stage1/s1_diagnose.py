"""Diagnosis of the v8.4 decisions on R2Seg (13 Nov 2026): joins ledger/v84_r2seg/anchors*.csv (decisions, levels, tests, signals)
with diag/{anchor,extent,tumour}_truth*.csv (s1_anchor_truth.py). Expert labels: evaluation only. Writes diag/report_diagnosis.txt.
Reason of a non-accepted anchor (at alpha 0.05, over its credible host set S):
  host_far_cap     accepted before the 10 mm safeguard, capped (every credible host > 10 mm from the lesion)
  hosts_disagree   the per-host decisions differ over S (decide_set -> uncertain)
  L<l>_<test>      every host gives the same decision: support level l (2 two-agent, 1 single / lobe, minus one per signal) and test
Signals that lowered the level are counted separately."""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

ACC = ('accepted', 'accepted_override')
RANK = {'accepted': 3, 'accepted_override': 3, 'uncertain': 2, 'rejected': 1}


def load():
    L = os.path.join(paths.ledger_dir(), 'v84_r2seg'); D = os.path.join(L, 'diag')
    cols = ['rel', 'id', 'kind', 'agents', 'ml', 'lesion', 'S', 'level', 'test', 'ev', 'decision', 'decision_before_safeguard',
            'host_far', 'host_cls', 'host', 'host_map', 'contested']
    a = pd.concat([pd.read_csv(f, usecols=cols) for f in glob.glob(os.path.join(L, 'anchors*.csv'))], ignore_index=True)
    a = a.drop_duplicates(['rel', 'id'], keep='last')
    p = a.rel.str.split('/', n=2, expand=True); a['ds'] = p[1]; a['case'] = p[2]
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, 'anchor_truth*.csv'))], ignore_index=True)
    t = t[pd.to_numeric(t.id, errors='coerce').notna()].drop_duplicates(['ds', 'case', 'id'], keep='last'); t['id'] = t.id.astype(float).astype(int)
    t['true'] = t['true'].astype(str) == 'True'
    e = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, 'extent_truth*.csv'))], ignore_index=True)
    u = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, 'tumour_truth*.csv'))], ignore_index=True).drop_duplicates(['ds', 'case', 'tumour'])
    a['case'] = a.case.astype(str); t['case'] = t.case.astype(str); e['case'] = e.case.astype(str); u['case'] = u.case.astype(str)
    m = t.merge(a.drop(columns=['kind', 'ml']), on=['ds', 'case', 'id'], how='inner')
    return m, e, u, D


def reason(r):
    if r['decision'] in ACC:
        return 'accepted'
    if r['decision_before_safeguard'] == 'accepted' and bool(r['host_far']):
        return 'host_far_cap'
    S = json.loads(r['S']); ev = json.loads(r['ev'])
    decs = {h: (ev.get(h) or {}).get('dec', {}).get('0.05') for h in S}
    if len(set(decs.values())) > 1:
        return 'hosts_disagree'
    lv = json.loads(r['level']); ts = json.loads(r['test'])
    h = S[0] if S else None
    if h is None:
        return 'no_host'
    return f"L{lv.get(h)}_{ts.get(h)}"


def signals(r):
    S = json.loads(r['S']); ev = json.loads(r['ev'])
    out = set()
    for h in S:
        out |= set((ev.get(h) or {}).get('sig') or [])
    return sorted(out)


def share(df, col, by='site'):
    x = df.groupby([by, col]).size().unstack(col).fillna(0)
    tot = x.sum(1)
    x = x.div(tot, axis=0)
    x['n'] = tot
    return x


def main():
    m, e, u, D = load()
    m['reason'] = m.apply(reason, axis=1)
    m['sigs'] = m.apply(lambda r: ','.join(signals(r)) or '-', axis=1)
    rep = [f'{len(m)} anchors with decisions and truth; {int(m.true.sum())} true anchors; {len(u)} expert tumours >= 0.5 ml.', '']
    # ---- tumours: best decision among the anchors touching each
    best = {}
    for r in m.itertuples(index=False):
        for j in json.loads(r.tumours) if isinstance(r.tumours, str) else []:
            k = (r.ds, r.case, j)
            score = (RANK[r.decision], r.in_frac, r.ml)
            if k not in best or score > best[k][0]:
                best[k] = (score, r)
    rows = []
    for x in u.itertuples(index=False):
        k = (x.ds, x.case, int(x.tumour))
        if k not in best:
            rows.append(dict(site=x.site, status='no_anchor', reason='no_anchor', kind='-', sigs='-', ml=x.ml)); continue
        r = best[k][1]
        st = 'accepted' if r.decision in ACC else r.decision
        rows.append(dict(site=x.site, status=st, reason=r.reason, kind=r.kind, sigs=r.sigs, ml=x.ml, host_cls=r.host_cls))
    T = pd.DataFrame(rows)
    rep += ['== expert tumours: best decision among the anchors touching them (share per site) ==', share(T, 'status').round(3).to_string(), '']
    lost = T[T.status.isin(['uncertain', 'rejected'])]
    rep += [f'== tumours touched by an anchor but not accepted ({len(lost)}): reason of their best anchor (share per site) ==',
            share(lost, 'reason').round(3).to_string(), '',
            'by anchor kind:', share(lost, 'kind').round(3).to_string(), '',
            'signals on those anchors (any host):', lost.sigs.value_counts().head(12).to_string(), '',
            'size of lost tumours (ml, median by site):', lost.groupby('site').ml.median().round(1).to_string(), '']
    # ---- anchors: decision x truth
    m['truth'] = np.where(m.true, 'true', 'false')
    rep += ['== anchors: decision by truth and kind (all sites pooled) ==',
            m.groupby(['truth', 'kind', 'decision']).size().unstack('decision').fillna(0).astype(int).to_string(), '']
    fa = m[(~m.true) & m.decision.isin(ACC)]
    nscan = m.groupby('site').case.nunique()
    rep += [f'== accepted FALSE anchors ({len(fa)}): per scan by site, kind, reason ==',
            (fa.groupby('site').size() / nscan).round(2).to_string(), '',
            share(fa, 'kind').round(3).to_string(), '',
            fa.groupby(['kind']).apply(lambda g: g.apply(lambda r: f"L{json.loads(r['level']).get(json.loads(r['S'])[0]) if json.loads(r['S']) else '?'}_"
                                                         f"{json.loads(r['test']).get(json.loads(r['S'])[0]) if json.loads(r['S']) else '?'}", axis=1).value_counts()).to_string(), '',
            'host class of accepted false anchors:', fa.host_cls.value_counts().to_string(), '',
            'their volume (ml): ' + fa.ml.describe().round(2).to_string().replace('\n', '; '), '']
    # ---- extents of accepted true anchors
    at = m[m.true & m.decision.isin(ACC)][['ds', 'case', 'id', 'site']]
    ee = e.merge(at, on=['ds', 'case', 'id', 'site'])
    g = ee.groupby(['ds', 'case', 'id'])
    ch = ee[ee.is_new].set_index(['ds', 'case', 'id'])
    bst = g.dice.max()
    ch = ch.join(bst.rename('best_dice'))
    ch['kind_name'] = ch.name.str.replace(r'\d+', '', regex=True)
    rep += [f'== extents of accepted true anchors ({len(ch)}; refitted psi) ==',
            'chosen vs best available Dice (per site mean):',
            ch.groupby('site')[['dice', 'best_dice', 'precision', 'recall']].mean().round(3).to_string(), '',
            'chosen extent type (share):', ch.kind_name.value_counts(normalize=True).round(3).to_string(), '',
            'precision of the chosen extent by type:', ch.groupby('kind_name')[['precision', 'recall', 'dice']].mean().round(3).to_string(), '']
    chv = ee[ee.is_v84].set_index(['ds', 'case', 'id']).join(bst.rename('best_dice'))
    rep += ['v8.4 psi for comparison (per site mean):', chv.groupby('site')[['dice', 'precision', 'recall']].mean().round(3).to_string(), '']
    open(os.path.join(D, 'report_diagnosis.txt'), 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))
    T.to_csv(os.path.join(D, 'tumour_status.csv'), index=False)


if __name__ == '__main__':
    main()
