"""Stage-1 v8.5 decision layer (14 Nov 2026; user decisions 13-14 Nov). Re-decides every v8.4 anchor on R2Seg from its stored per-host
evidence (anchors*.csv: S, ev[h] = level, test, ...); no agent, ledger or host rerun. Expert labels are not used.
Changes from v8.4 (g3_run_v84 / grounding_g2):
  R3  the primary-not-detected override is OFF: 'accepted_override' no longer exists (tumour-free scans exist in practice).
  R2  a two-agent anchor at support level 2 is not demoted by a rule-out test: DECISION[2]['rule_out'] = accepted (the existence
      test carries no information for two-agent anchors on R2Seg: true rate 0.57 under rule-out, 0.66 under neutral).
  R1  when the per-host decisions differ over the credible host set S, the decision under the MAP host is used instead of
      'uncertain' (host uncertainty is still reported through S; it no longer blocks the existence decision).
Unchanged: support level (2 two-agent, 1 single/lobe, minus one per signal), the other table entries, alpha 0.05, the 10 mm
host-far safeguard (every credible host > 10 mm: decision at most uncertain).
Output: ledger/v84_r2seg/decisions_v85.csv (rel, id, kind, decision_v84, decision_v85, rule). Usage: python s1_v85_decide.py [--ledger v85_r2seg] [--table v86]   (defaults v84_r2seg, v85 table)
--table v86 (15 Nov): v8.6 decisions (= g3_run_v86 output, verified on 12 scans) -> decisions_v86.csv"""
import os, sys, json, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

ALPHA = '0.05'
DECISION_V85 = {2: {'confirm': 'accepted', 'neutral': 'accepted', 'rule_out': 'accepted'},       # R2
                1: {'confirm': 'accepted', 'neutral': 'uncertain', 'rule_out': 'rejected'},
                0: {'confirm': 'uncertain', 'neutral': 'rejected', 'rule_out': 'rejected'}}
RANK = {'rejected': 0, 'uncertain': 1, 'accepted': 2}
# v8.6 table (15 Nov; --table v86): level 2 accepted, level 0 rejected (any test); level 1 confirm accepted, neutral / rule-out uncertain
DECISION_V86 = {2: dict.fromkeys(('confirm', 'neutral', 'rule_out'), 'accepted'),
                1: {'confirm': 'accepted', 'neutral': 'uncertain', 'rule_out': 'uncertain'},
                0: dict.fromkeys(('confirm', 'neutral', 'rule_out'), 'rejected')}
TABLE = DECISION_V86 if '--table' in sys.argv and sys.argv[sys.argv.index('--table') + 1] == 'v86' else DECISION_V85
NAME = 'v86' if TABLE is DECISION_V86 else 'v85'


def host_decision(e):
    lv, ts = e.get('level'), (e.get('test') or {}).get(ALPHA)
    if ts == 'not_run':                                               # a v8.6 ledger: the test was skipped (level 0 / 2); decided by level alone
        return TABLE[int(lv)]['neutral'] if lv in (0, 2) else None
    if lv is None or ts not in ('confirm', 'neutral', 'rule_out'):
        return None
    return TABLE[int(lv)][ts]


def decide(r):
    S = json.loads(r['S']) if isinstance(r['S'], str) else []
    ev = json.loads(r['ev']) if isinstance(r['ev'], str) else {}
    per = {h: host_decision(ev.get(h) or {}) for h in S}
    per = {h: d for h, d in per.items() if d is not None}
    if not per:
        # no host evidence (e.g. no credible host): keep the v8.4 decision, without the override
        d0 = r['decision']
        return ('rejected' if d0 == 'accepted_override' else d0), 'no_host_evidence'
    if len(set(per.values())) == 1:
        d, rule = next(iter(per.values())), 'agree'
    else:
        hm = r['host_map'] if isinstance(r['host_map'], str) and r['host_map'] in per else S[0]
        d, rule = per.get(hm, min(per.values(), key=RANK.get)), 'R1_map_host'
    if d == 'accepted' and str(r['host_far']) in ('True', '1', '1.0'):
        d, rule = 'uncertain', rule + '+host_far_cap'
    return d, rule


def main():
    L = os.path.join(paths.ledger_dir(), sys.argv[sys.argv.index('--ledger') + 1] if '--ledger' in sys.argv else 'v84_r2seg')
    out = []
    for f in sorted(glob.glob(os.path.join(L, 'anchors*.csv'))):
        a = pd.read_csv(f, usecols=['rel', 'id', 'kind', 'S', 'ev', 'decision', 'host_map', 'host_far'])
        for r in a.to_dict('records'):
            d, rule = decide(r)
            out.append(dict(rel=r['rel'], id=r['id'], kind=r['kind'], decision_v84=r['decision'], **{f'decision_{NAME}': d}, rule=rule))
    o = pd.DataFrame(out).drop_duplicates(['rel', 'id'], keep='last')
    o.to_csv(os.path.join(L, f'decisions_{NAME}.csv'), index=False)
    print(len(o), 'anchors ->', os.path.join(L, f'decisions_{NAME}.csv'))
    print(pd.crosstab(o.decision_v84, o[f'decision_{NAME}']).to_string())
    print(o.rule.value_counts().to_string())


if __name__ == '__main__':
    main()
