"""Gate evaluation for graph pruning v3 (27 Nov 2026; claude/graph_pruning_v3_gate_2026-11-27.md, section 5). Reads
PanCIA_outputs/ledger/graph_v3/<tag>.csv (s1_graph_v3.py) and prints / writes G1-G4 and the report items per mode.
Same code for the design set (tag 'design') and the frozen evaluation (tag 'v3_eval'). Tumour-free scans: cohort file 'negative'
(the job files' tumour_free column comes from it). Candidates: in-scope, lab T / F (P excluded).
Output: ledger/graph_v3/<tag>_gate.txt and <tag>_strata.csv. Usage: python s1_graph_v3_eval.py --tag design"""
import os, sys, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

D = os.path.join(paths.ledger_dir(), 'graph_v3')
VES = ('vessel_artery', 'vessel_vein')


def region_class(r):
    if r.dom in ('c_supply_drain', 'c_vessel_vessel'): return 'vessel'
    if r.dom in ('c_adjacent', 'c_vessel_organ'): return 'rim_contested'
    if r.dom == 'core':
        if r.host_cat in VES: return 'vessel'
        if r.host_cat == 'hollow_organ': return 'hollow_core'
        if r.host_cat == 'unresolved': return 'unresolved_core'      # step-2g g3 pelvic node (28 Nov)
        return 'solid_core'
    return 'uncertain_region'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    d = pd.read_csv(os.path.join(D, a.tag + '.csv'), low_memory=False)
    raw = d.copy(); raw['tumour_free'] = raw.tumour_free.astype(str) == 'True'
    n_neg = raw[raw.tumour_free][['ds', 'case']].drop_duplicates().shape[0]     # incl. scans whose only row is a note (no candidates)
    notes = d[d.note.notna() & (d.note != '')]
    d = d[d.note.isna() | (d.note == '')].copy()
    for c in ('in_scope', 'tumour_free', 'V', 'W', 'R'): d[c] = d[c].astype(str) == 'True'
    x = d[d.in_scope & d.lab.isin(['T', 'F'])].copy()
    x['rclass'] = [region_class(r) for r in x.itertuples()]
    x['band'] = pd.cut(x.ml, [0, 1, 10, 1e9], labels=['<1', '1-10', '>10'], right=False)
    out = [f'tag {a.tag}: scans {d[["ds", "case"]].drop_duplicates().shape[0]}, notes {len(notes)} {notes.note.value_counts().head(3).to_dict()}']
    res = {}
    for mode, g in x.groupby('mode'):
        T, F = g[g.lab == 'T'], g[g.lab == 'F']
        pT = (T.decision == 'pruned').mean() * 100; uT = (T.decision == 'uncertain').mean() * 100
        pF = (F.decision == 'pruned').mean() * 100; uF = (F.decision == 'uncertain').mean() * 100
        site = T.groupby('site').decision.apply(lambda s: (s == 'pruned').mean() * 100)
        sc = T.groupby(['ds', 'case']).decision.apply(lambda s: (s == 'pruned').all())
        lost = sc.mean() * 100 if len(sc) else np.nan
        neg = g[g.tumour_free & (g.lab == 'F')]
        res[mode] = dict(pT=pT, site_max=site.max(), lost=lost, pF=pF, uT=uT)
        out += [f'\n[{mode}] T {len(T)}  F {len(F)}',
                f'  G1: T pruned {pT:.1f}% (<=5) | worst site {site.idxmax() if len(site) else "-"} {site.max():.1f}% (<=10) | tumour scans losing all T {lost:.1f}% (<=2)',
                f'      per site T pruned: {site.round(1).to_dict()}',
                f'  G2: F pruned {pF:.1f}% (>=30) | tumour-free false/scan {len(neg)}/{n_neg} -> {(neg.decision != "pruned").sum()}/{n_neg}'
                f' = {len(neg) / max(n_neg, 1):.2f} -> {(neg.decision != "pruned").sum() / max(n_neg, 1):.2f} '
                f'(reduction {100 * (1 - (neg.decision != "pruned").sum() / max(len(neg), 1)):.1f}%, >=25)',
                f'  G4: T uncertain {uT:.1f}% (flag > 50) | F uncertain {uF:.1f}% | T kept {(T.decision == "kept").mean() * 100:.1f}%',
                f'  explanations among pruned: V {int(g[g.decision == "pruned"].V.sum())} W {int(g[g.decision == "pruned"].W.sum())} R {int(g[g.decision == "pruned"].R.sum())}',
                f'  members: {g.member.value_counts().to_dict()}',
                f'  report R: candidates in scans with set-aside sex classes: {(g.set_aside.fillna("") != "").mean() * 100:.1f}%']
    # G3 stratified
    rows = []
    for (rc, band), g in x.groupby(['rclass', 'band'], observed=True):
        for lab in ('F', 'T'):
            h = g[g.lab == lab]; cnt = h.groupby('mode').size(); rate = h.groupby('mode').decision.apply(lambda s: (s == 'pruned').mean() * 100)
            ok = all(cnt.get(m, 0) >= 30 for m in ('BP', 'VT', 'BP+VT'))
            rows.append(dict(region=rc, band=band, lab=lab, **{f'n_{m}': int(cnt.get(m, 0)) for m in ('BP', 'VT', 'BP+VT')},
                             **{f'pruned_{m}': round(rate.get(m, np.nan), 1) for m in ('BP', 'VT', 'BP+VT')},
                             tested=ok, spread=round(rate.max() - rate.min(), 1) if ok else np.nan, pass_=(rate.max() - rate.min() <= 15) if ok else None))
    st = pd.DataFrame(rows); st.to_csv(os.path.join(D, a.tag + '_strata.csv'), index=False)
    t = st[st.tested]
    out += ['\nG3 (stratified; strata with >= 30 per mode):', t.to_string(index=False) if len(t) else '  no stratum reaches 30 per mode',
            f'  G3 {"PASS" if len(t) and t.pass_.all() else ("FAIL" if len(t) else "not testable")}']
    out += ['\nSummary per mode: ' + ', '.join(f'{m}: G1 {"PASS" if r["pT"] <= 5 and r["site_max"] <= 10 and r["lost"] <= 2 else "FAIL"}'
                                              f' / G2 F-pruned {r["pF"]:.0f}%' for m, r in res.items())]
    txt = '\n'.join(out); print(txt)
    open(os.path.join(D, a.tag + '_gate.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
