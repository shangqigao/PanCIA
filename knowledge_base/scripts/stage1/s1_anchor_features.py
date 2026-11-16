"""Compact per-anchor feature table for the v8.5 rule study (13 Nov 2026): v8.4 decision inputs on R2Seg joined with the expert
match (s1_anchor_truth.py). Expert labels are used for evaluation and for fitting stage-1 parameters on R2Seg only.
Per anchor: ds, case, site, id, kind, ml, mod, true, in_frac, tumours, decision, S_size, host_cls, host_far, contested,
disagree (per-host decisions at alpha 0.05 differ over S), and at the MAP host: test, tree, appl, lam, cls, level, sigs;
plus the union of signals over S. Scans: diag/scans.csv (every evaluated scan, with or without anchors).
Output: clinical/PanCIA_outputs/ledger/v84_r2seg/diag/anchor_features.csv. Usage: python s1_anchor_features.py"""
import os, sys, json, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402
import s1_diagnose as SD             # noqa: E402


def main():
    m, e, u, D = SD.load()
    rows = []
    for r in m.itertuples(index=False):
        S = json.loads(r.S); ev = json.loads(r.ev)
        h = r.host_map if isinstance(r.host_map, str) and r.host_map in ev else (S[0] if S else None)
        x = ev.get(h) or {}
        decs = {q: (ev.get(q) or {}).get('dec', {}).get('0.05') for q in S}
        sig_any = sorted(set().union(*[set((ev.get(q) or {}).get('sig') or []) for q in S])) if S else []
        rows.append(dict(ds=r.ds, case=r.case, site=r.site, id=r.id, kind=r.kind, ml=r.ml, mod=r.rel.split('/')[0], true=r.true,
                         in_frac=r.in_frac, tumours=r.tumours, decision=r.decision, S_size=len(S), host_cls=r.host_cls,
                         host_far=bool(r.host_far) if r.host_far == r.host_far else False, contested=r.contested,
                         disagree=len(set(decs.values())) > 1, test=(x.get('test') or {}).get('0.05'), tree=x.get('tree'),
                         appl=x.get('appl'), lam=x.get('lam'), cls=x.get('cls'), level=x.get('level'),
                         sigs=','.join(x.get('sig') or []), sigs_any=','.join(sig_any)))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(D, 'anchor_features.csv'), index=False)
    sc = []
    for f in glob.glob(os.path.join(D, 'anchor_truth*.csv')):
        sc.append(pd.read_csv(f, usecols=['ds', 'case', 'site']))
    pd.concat(sc).astype(str).drop_duplicates(['ds', 'case']).to_csv(os.path.join(D, 'scans.csv'), index=False)
    u.to_csv(os.path.join(D, 'tumours.csv'), index=False)
    print(len(F), 'anchors ->', D)


if __name__ == '__main__':
    main()
