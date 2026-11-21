"""Post-hoc check of a one-directional no_confirmation rule on stage 1 v8.6 / R2Seg (16 Nov 2026; claude/rejected_tumours_diagnosis_2026-11-16.md).
Current rule: a single-agent / lobe anchor gets no_confirmation under host h when another agent has any tumour mask inside h's envelope
but none near the anchor. Variant A: the signal counts only if the other agent's presence in h is a CONFIRMED lesion, approximated
here as: the scan has a two-agent anchor whose credible host set S contains h. Otherwise the signal is dropped, the level is
recomputed (G2.support_level: 1 for single / lobe minus the remaining signals) and the anchor re-decided with the v8.6 table.
Promoted anchors (level 0 -> 1) have no existence test (not run at level 0), so they are counted as 'uncertain' (an upper bound
on uncertain, a lower bound on accepted). R1 (MAP host when hosts differ) and the 10 mm host-far cap (accepted -> uncertain) as in v8.6.
Phase 1 (--redecide): ledger/v86_r2seg/diag/nc_redecide.csv (rel, id, kind, lesion, decision, decision_A, promoted).
Phase 2 (default; resumable, chunk with `timeout`): for every scan with a promoted anchor, whether each promoted anchor's selected
extent (refitted psi, as in s1_eval_r2seg) touches an expert tumour, and which. Expert labels: evaluation only.
Output: diag/nc_promoted_truth.csv. Usage: python s1_nc_rule_check.py --redecide ; timeout 170 python s1_nc_rule_check.py"""
import os, sys, json, glob, time, argparse, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import s1_eval_r2seg as E   # noqa: E402

TAB = {2: 'accepted', 1: 'uncertain', 0: 'rejected'}


def redecide(L):
    a = pd.concat([pd.read_csv(f, usecols=['rel', 'id', 'kind', 'lesion', 'S', 'ev', 'host_map', 'host_far', 'decision'])
                   for f in sorted(glob.glob(os.path.join(L, 'anchors*.csv')))]).drop_duplicates(['rel', 'id'], keep='last')
    a = a[a.rel.str.split('/').str[1] != 'Uterus_Tumor_00']
    J = lambda s: json.loads(s) if isinstance(s, str) else None
    two_hosts = {}
    for rel, S in zip(a.rel[a.kind == 'two'], a.S[a.kind == 'two']):
        two_hosts.setdefault(rel, set()).update(J(S) or [])
    out = []
    for r in a.itertuples(index=False):
        d, prom = r.decision, False
        if r.kind != 'two' and r.decision == 'rejected':
            ev = J(r.ev) or {}; S = J(r.S) or []; per = {}
            for h in S:
                e = ev.get(h) or {}
                sig = list(e.get('sig') or [])
                if 'no_confirmation' in sig and h not in two_hosts.get(r.rel, set()):
                    sig.remove('no_confirmation')
                lv = max(0, min(2, (2 if r.kind == 'two' else 1) - len(sig)))
                t = (e.get('test') or {}).get('0.05')
                per[h] = 'accepted' if (lv == 1 and t == 'confirm') else TAB[lv]
            if per:
                ds = set(per.values())
                d = ds.pop() if len(ds) == 1 else per.get(r.host_map, 'uncertain')
                if d == 'accepted' and str(r.host_far) in ('True', '1', '1.0'):
                    d = 'uncertain'
                prom = d != 'rejected'
        out.append(dict(rel=r.rel, id=r.id, kind=r.kind, lesion=r.lesion, decision=r.decision, decision_A=d, promoted=prom))
    o = pd.DataFrame(out); o.to_csv(os.path.join(L, 'diag', 'nc_redecide.csv'), index=False)
    print(pd.crosstab(o.decision, o.decision_A)); print('scans with promoted anchors', o[o.promoted].rel.nunique())


def truth(L):
    D = os.path.join(L, 'diag'); f_out = os.path.join(D, 'nc_promoted_truth.csv'); f_done = os.path.join(D, 'nc_done.txt')
    o = pd.read_csv(os.path.join(D, 'nc_redecide.csv')); p = o[o.promoted]
    done = set(open(f_done).read().split()) if os.path.exists(f_done) else set()
    psis = {}; t0 = time.time(); n = 0
    SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1)
    if SH:
        f_out = f_out.replace('.csv', f'_s{SI}.csv'); f_done = f_done.replace('.txt', f'_s{SI}.txt')
        done |= set(open(f_done).read().split()) if os.path.exists(f_done) else set()
    for k, (rel, g) in enumerate(p.groupby('rel')):
        if rel in done or k % SN != SI:
            continue
        _, ds, case = rel.split('/', 2); rows = []
        try:
            site = E.EX.SITE_ENTITY[ds]; psis.setdefault(site, E.psi_for(site)); new = psis[site][1]
            lf = os.path.join(E.paths.root('R2Seg'), ds, 'labels', case + '_seg.nii.gz')
            T = np.isin(E.EX._read_labels(lf), E.EX.APPEARANCE[ds]['tumour']); labT, _ = E.comps(T)
            z = np.load(os.path.join(L, 'extents', ds, case + '.npz')); meta = json.loads(str(z['meta']))
            for r in g.itertuples(index=False):
                m = meta['anchors'].get(str(int(r.id)))
                if not m or not m['extents']:
                    continue
                ex = m['extents']; cands = [dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ex]
                j = int(np.argmax(E.G2.extent_prior(cands, new, {}))); box = ex[j]['box']
                x = E.unpack(z, f'a{r.id}_e{j}', box); hit = sorted(set(np.unique(labT[tuple(slice(b[0], b[1]) for b in box)][x]).tolist()) - {0})
                rows.append(dict(rel=rel, ds=ds, case=case, site=site, id=int(r.id), lesion=r.lesion, decision_A=r.decision_A,
                                 true=bool(hit), tumours='|'.join(map(str, hit))))
        except Exception:
            print('error', rel, traceback.format_exc()[-300:], flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(f_out, mode='a', index=False, header=not os.path.exists(f_out))
        open(f_done, 'a').write(rel + '\n'); n += 1
        if n % 50 == 0:
            print(n, rel, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ledger', default='v86_r2seg'); ap.add_argument('--redecide', action='store_true')
    a = ap.parse_args(); L = os.path.join(E.paths.ledger_dir(), a.ledger)
    redecide(L) if a.redecide else truth(L)
