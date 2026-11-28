"""Structural acceptance of the step-2g g3 change (28 Nov 2026; plan claude/elongation_and_pelvic_fix_plan_2026-11-28.md, B,
acceptance 1-3). Design set only; no truth is used for criteria 1-2 (criterion 3 reports truth-labelled shares, not tuned).
--stage anatomy: for every scan rerun by s1_anatomy_g.py --g3_contradiction (copies of the g2 files in anatomy/r2seg_g2/): the organ
  list, pairs, state and every npz array must be identical to g2; the only addition is the node (u0_core, json 'unresolved').
  Scans not rerun keep their g2 files (g3 returns before any change when there is no unresolved dispute; checked on samples).
--stage graph: graph_v3/<new>.csv vs graph_v3/<old>.csv: rows of scans without the node must be identical; per site, before / after:
  true candidates whose dominant region is 'outside' / U-region, organ_organ and other_edge shares, candidates hosted by the node.
Output: ledger/graph_v3/g3_check_<stage>.txt. Usage: python s1_g3_check.py --stage anatomy|graph [--old design --new design_g3]"""
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

A = os.path.join(paths.root('outputs'), 'anatomy'); GD = os.path.join(paths.ledger_dir(), 'graph_v3')
NODE = 'pelvic_reproductive_unresolved'


def anatomy():
    out, bad, n_node = [], [], 0
    olds = sorted(glob.glob(os.path.join(A, 'r2seg_g2', '*', '*.json')))
    for f in olds:
        img = os.path.relpath(f, os.path.join(A, 'r2seg_g2'))[:-5]
        nf = os.path.join(A, 'r2seg', img + '.json'); o = json.load(open(f)); n = json.load(open(nf))
        if n.get('version') != 'g3': bad.append(f'{img}: not rerun (version {n.get("version")})'); continue
        ok = n['organs'] == o['organs'] and n['pairs'] == o['pairs'] and n['state'] == o['state'] and n['expected'] == o['expected']
        zo = np.load(f[:-5] + '.npz'); zn = np.load(nf[:-5] + '.npz')
        ok_z = all(k in zn.files and np.array_equal(zo[k], zn[k]) for k in zo.files) and set(zn.files) - set(zo.files) <= {'u0_core', 'u0_core_sl', 'u0_core_shape'}
        n_node += bool(n.get('unresolved'))
        if not (ok and ok_z): bad.append(f'{img}: organs/pairs/state {ok}, npz {ok_z}')
    out += [f'scans rerun (g2 copies): {len(olds)}; with the node: {n_node}', f'ACCEPTANCE B1/B2 (identical apart from the node): '
            f'{"PASS" if not bad else "FAIL"} ({len(bad)} differing)'] + bad[:30]
    return out


def graph(old, new):
    o = pd.read_csv(os.path.join(GD, old + '.csv'), low_memory=False); n = pd.read_csv(os.path.join(GD, new + '.csv'), low_memory=False)
    for x in (o, n): x['case'] = x.case.astype(str)
    sc = set(zip(n.ds, n.case)); o = o[[(u, v) in sc for u, v in zip(o.ds, o.case)]]      # same scans (a partial new run compares like with like)
    nodes = set(zip(n[n.host == NODE].ds, n[n.host == NODE].case))
    js = {(u, v) for u, v in zip(n.ds, n.case) if (json.load(open(os.path.join(A, 'r2seg', u, v + '.json'))).get('unresolved') or None)}
    k = ['ds', 'case', 'mode', 'cid']
    cols = [c for c in o.columns if c not in k and c in n.columns]
    m = o.merge(n, on=k, how='outer', suffixes=('_o', '_n'), indicator=True)
    m['has_node'] = [(u, v) in js for u, v in zip(m.ds, m.case)]
    diff = np.zeros(len(m), bool)
    for c in cols: diff |= m[c + '_o'].astype(str).values != m[c + '_n'].astype(str).values
    un = m[~m.has_node]
    out = [f'scans with the node: {len(js)}; scans with a candidate hosted by the node: {len(nodes)}',
           f'ACCEPTANCE (rows of scans without the node identical): {"PASS" if not (diff[~m.has_node.values].any() or (un._merge != "both").any()) else "FAIL"}'
           f' ({int(diff[~m.has_node.values].sum())} rows differ, {int((un._merge != "both").sum())} unmatched)']
    rows = []
    for tag, d in (('before', o), ('after', n)):
        d = d[d.note.isna() | (d.note == '')]
        x = d[(d.in_scope.astype(str) == 'True') & d.lab.isin(['T', 'F'])]
        for (site, mode), g in x.groupby(['site', 'mode']):
            T = g[g.lab == 'T']; F = g[g.lab == 'F']
            rows.append(dict(site=site, mode=mode, when=tag, T_Uregion=round((T.member == 'U-region').mean() * 100, 1),
                             T_outside=round((T.dom == 'outside').mean() * 100, 1), F_Uregion=round((F.member == 'U-region').mean() * 100, 1),
                             organ_organ=round((g.dom == 'c_organ_organ').mean() * 100, 1), other_edge=round((g.dom == 'c_other_edge').mean() * 100, 1),
                             T_node=int((T.host == NODE).sum()), F_node=int((F.host == NODE).sum())))
    R = pd.DataFrame(rows).pivot_table(index=['site', 'mode'], columns='when',
                                       values=['T_Uregion', 'F_Uregion', 'organ_organ', 'other_edge', 'T_node', 'F_node'])
    out += ['\nper site x mode (in scope, before / after; % of candidates, node = counts):', R.round(1).to_string()]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['anatomy', 'graph'], required=True)
    ap.add_argument('--old', default='design'); ap.add_argument('--new', default='design_g3'); a = ap.parse_args()
    out = anatomy() if a.stage == 'anatomy' else graph(a.old, a.new)
    txt = '\n'.join(out); print(txt); open(os.path.join(GD, f'g3_check_{a.stage}.txt'), 'w').write(txt)


if __name__ == '__main__':
    main()
