"""v8.6 (15 Nov 2026; user decisions, claude/stage1_v86_cascade_2026-11-15.md): g3_run_v85.py with ledger_v86 (F1, G3; G1 dropped; fast EDT)
with the decision rules built in and a two-tier cascade.
  Tier 1 (no image evidence): support level per credible host from kind and signals. The existence test is computed ONLY for
         level-1 hosts; level 0 and level 2 do not depend on it (test 'not_run').
  Decision table v8.6 (DECISION_V86): level 2 -> accepted (any test; R2); level 1 -> confirm accepted, neutral uncertain,
         rule-out uncertain (rule-out no longer rejects at level 1: uninformative on real agent outlines, true rate 9.0% vs 9.9%);
         level 0 -> rejected (any test; previously confirm gave uncertain).
  R1: per-host decisions that differ over the credible set S -> the decision under the MAP host.
  R3: the primary-not-detected override is removed (tumour-free scans exist).
  Unchanged: 10 mm host-far safeguard, credible sets at alpha 0.01 / 0.05 / 0.10, extents and extent prior.
  Output and checkpoints: ~/v82/ck86; outputs via V8OUT (s1_run_r2seg.py --version v86 -> ledger/v86_r2seg/).
--- v8.5 docstring ---
v8.5 (15 Nov 2026): g3_run_v84.py with ledger_v85 (F1, G1, G3); checkpoints in ~/v82/ck85. Distance transforms use the `edt`
package via pancia_kb.fast_edt (same distances to float precision; PANCIA_FAST_EDT=0 restores scipy). Decision rules R1/R2/override-off are
applied afterwards (s1_v85_decide.py). --- v8.4 docstring ---
v8.4 (21-22 Oct 2026): copy of g3_run_v83.py with ledger_v84 (position rule, sex-dependent organs never join, has_part tidy-up);
column barred; checkpoints carry VER = ledger_v84.CODE_HASH + md5 of this file and are ignored when it differs. Output ledger/v84_pilot.
--- g3_run_v83 docstring ---
v8.3 (21 Oct 2026): copy of g3_run_v82.py with ledger_v83 (expected list: part -> segmented whole; only parts of expected hosts barred from entry). Output ledger/v83_pilot.
--- g3_run_v82 docstring ---
v8.2 (21 Oct 2026; plan claude/v81_H_panel_host_review_2026-10-21.md): copy of g3_run.py (kept) with ledger_v82 (KB static list,
proximal entry by containment or distance gate). Step 5 safeguard: d(S) = min lesion distance over the credible set S_alpha; d(S) >
HOST_TOL (10 mm) -> host unknown ('none_within_tolerance'), decision capped at uncertain (reason host_beyond_tolerance); not applied
in the anatomy-insufficient fallback (host assumed). Attribution over S is replaced by the host ROLE (primary / local_invasion /
distant / proximal / unknown) and role agreement over S; 'other' and 'uncertain_attribution' are removed. The override (primary not
detected) never uses a lesion whose host is unknown by the safeguard. Output ledger/v82_pilot.
--- g3_run docstring ---
v8.1 host step 1 (20 Oct 2026): copy of g2_run.py (kept) with ledger_v81 (no 10 mm gate, distance feature), NO no-host
candidate (P(h | L) over the lesion's candidates only; beta from ledger/v8/testbed/fit_host6.json (beta_UD); unsegmented hosts use the KB relational region, ledger_v81.kb_region_rel), extent prior psi alone (eta = 1).
Steps 3-4: composite candidates p>n (local-invasion relation, ledger_v81); fix 11: the host choice is over origins, each scored
by its best explanation (p alone or p invading n); the invaded organ is reported in the column invades.
Fallback (user 21 Oct): a scan with no organ of confidence 1 is 'anatomy_insufficient' (ledger_v81): host assumed = planner
primary (host_source), intrinsic test, override only for two-agent anchors.
A lesion with no candidate in view: host unknown -> uncertain (attribution 'unknown'). Output ledger/v81_pilot.
--- g2_run docstring ---
v8 + G2 on the TCGA pilot series (19 Oct 2026; plan claude/v8_execution_plan_2026-10-19.md step 4-5; main.tex v8 md5 3d98e2).
Public TCGA only; the pilot is a bug check (no annotation; nothing is tuned on it).
Per series: ledger v8 (pancia_kb.ledger_v8; candidates, signals, extents with named flags) ->
  normal model  P(h | L) = choice model over the lesion's candidates + the no-host candidate (beta, s0 from ledger/v8/testbed/fit_g2.json),
                equal weights; host credible sets S_alpha for alpha in {0.01, 0.05, 0.10}
  tumour model  existence evidence (grounding_g1.existence_evidence, unchanged) under every host of S_0.01 (stored for sensitivity);
                existence test at alpha (ledger/v8/grounding/test_sets.json); support level per host; decision per host
                (grounding_g2.DECISION); the decision at alpha = 0.05 must agree over S_0.05, else uncertain; degraded scan -> neutral
  extent        w0 ~ psi * prod eta_f (fit_g2.json); xi^ = argmax w0
  policy        reported host = most probable host of S; attribution must agree over S (cancer / other / uncertain_attribution;
                user 20 Oct); no host -> 'unknown';
                no accepted anchor whose S contains the primary -> 'primary_not_detected': the lesion whose S contains the primary with
                the highest support level (then the largest volume) is added as a flagged override with its non-rejected anchors;
                inferred and imposed hosts (and classes) are both written.
Writes ledger/v8_pilot/{anchors,lesions,series}[_s<i>].csv and the panel cache (cache/<k>/a<id>.npz).
Usage: SHARD=i/n python3 g2_run.py [budget_s]"""
import os, sys, json, time, glob, numpy as np, pandas as pd, nibabel as nib
H = os.path.expanduser('~'); KBD = f'{H}/mnt/PanCIA/knowledge_base'; sys.path.insert(0, KBD)
from pancia_kb import fast_edt as FE             # v8.5 speed-up (15 Nov): exact EDT from the `edt` package; PANCIA_FAST_EDT=0 disables
FAST_EDT = os.environ.get('PANCIA_FAST_EDT', '1') != '0' and FE.install()
print(f"stage 1 v8.6: distance transforms via {'edt package (fast)' if FAST_EDT else 'scipy (pip install edt for ~2.5x speed)'}", file=sys.stderr, flush=True)
from pancia_kb import KnowledgeBase, ledger_v86 as L8, grounding_g1 as G1, grounding_g2 as G2, host_v81 as HV
from pancia_kb.ledger_v6 import _hk
S = f'{H}/mnt/TCGA_Seg'; B = f'{H}/mnt/clinical/PanCIA_outputs/ledger'
O = os.environ.get('V8OUT', f'{B}/v84_pilot'); os.makedirs(O + '/cache', exist_ok=True)
TAB1 = G1.Tables1(f'{B}/v6/grounding/g0/h5/tables_g1.json')
TS = G2.TestSets(f'{B}/v8/grounding/test_sets.json')
FIT = json.load(open(f'{B}/v8/testbed/fit_host6.json'))   # fixes 11/13: beta (UD, rel_in not used) refitted on testbed_host6
PSI = G2.Psi(FIT['psi']); ETA = {}; BETA = FIT['beta_UD']
ALPHA = 0.05; ALPHAS = [0.01, 0.05, 0.10]
KB = KnowledgeBase.load(f'{KBD}/knowledge_base')
SH = os.environ.get('SHARD'); SI, SN = (int(x) for x in SH.split('/')) if SH else (0, 1); SUF = f'_s{SI}' if SH else ''
IP = f'{H}/g3v86_inprogress{SUF}.txt'
HOST_TOL = L8.P['host_tol_mm']
CAP = lambda d_: 'uncertain' if d_ == 'accepted' else d_
DECISION_V86 = {2: {'confirm': 'accepted', 'neutral': 'accepted', 'rule_out': 'accepted', 'not_run': 'accepted'},
                1: {'confirm': 'accepted', 'neutral': 'uncertain', 'rule_out': 'uncertain', 'not_run': 'uncertain'},
                0: {'confirm': 'rejected', 'neutral': 'rejected', 'rule_out': 'rejected', 'not_run': 'rejected'}}
decide86 = lambda lv, t: DECISION_V86[int(lv)][t]


def decide_set86(per_host, h_map):
    """R1: same decision over S -> it; else the decision under the MAP host"""
    ds = set(per_host.values())
    if len(ds) == 1:
        return ds.pop()
    return per_host[h_map] if h_map in per_host else 'uncertain'
hk = lambda q: _hk(q) if q is not None else 'unknown'


def out(name, rows):
    if not rows: return
    f = f'{O}/{name}{SUF}.csv'; df = pd.DataFrame(rows)
    if os.path.exists(f):
        cols = list(pd.read_csv(f, nrows=0).columns)
        if set(df.columns) - set(cols):
            pd.concat([pd.read_csv(f), df], ignore_index=True).to_csv(f, index=False); return
        df = df.reindex(columns=cols)
    df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)


def pad(sl, sp, mm, shape):
    p = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    return tuple(slice(max(0, s.start - a), min(n, s.stop + a)) for s, a, n in zip(sl, p, shape))


def ubox(sls):
    return tuple(slice(min(s[d].start for s in sls), max(s[d].stop for s in sls)) for d in range(3))


def place(sl_src, crop, sl_dst):
    out_ = np.zeros(tuple(s.stop - s.start for s in sl_dst), bool)
    lo = [max(a.start, b.start) for a, b in zip(sl_src, sl_dst)]; hi = [min(a.stop, b.stop) for a, b in zip(sl_src, sl_dst)]
    if any(l >= h for l, h in zip(lo, hi)): return out_
    out_[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, sl_dst))] = \
        crop[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, sl_src))]
    return out_


CKD = f'{H}/v82/ck86'; os.makedirs(CKD, exist_ok=True)       # checkpoints (a device call is limited to ~44 s; series take up to ~3 min)
import pickle


import hashlib as _hl
VER = L8.CODE_HASH + _hl.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8] + ('f' if FAST_EDT else 's')


def _ld(f, rel):
    try:
        d = pickle.load(open(f, 'rb'))
        return d if (d.get('rel') == rel and d.get('ver') == VER) else None
    except Exception:
        return None


def _dump(f, d):
    d = dict(d, ver=VER)
    pickle.dump(d, open(f + '.tmp', 'wb'), protocol=5); os.replace(f + '.tmp', f)


def run(rel, k_idx):
    t0 = time.time()
    ckL, ckS, ckA = f'{CKD}/led{SUF}.pkl', f'{CKD}/state{SUF}.pkl', f'{CKD}/aq{SUF}_{k_idx}.pkl'
    st = _ld(ckL, rel)
    if st is not None:
        led, cr, aff, shp = st['v']
    else:
        for f_ in glob.glob(f'{CKD}/aq{SUF}_*.pkl'):
            if f_ != ckA: os.remove(f_)                                 # scratch checkpoints of earlier series (outside the mounts)
        led, cr, aff, shp = L8.build_ledger(rel, S, KB, params=dict(aq_cache=ckA, lobe_cache=f'{CKD}/lobe{SUF}.pkl'))
        _dump(ckL, dict(rel=rel, v=(led, cr, aff, shp)))
    if not cr:
        return [], [], dict(rel=rel, cancer=rel.split('/')[1], mod=rel[:2], qc=led['qc']['state'], n_anchors=0, k=k_idx, sec=round(time.time() - t0, 1))
    man = json.load(open(f'{S}/VoxTell/Radiology/{rel}_plan_manifest.json'))
    X = np.asanyarray(nib.load(f"{H}/mnt/TCGA_NIFTI/{man['img_name']}.nii.gz").dataobj); X = X[..., 0] if X.ndim == 4 else X
    sp = np.array(led['spacing'], float); si = cr['si']; ax = cr['ax']; mod = rel[:2]
    off = led['qc']['state'] != 'good'
    anchors, lesions, allh, claims, prim = cr['anchors'], cr['lesions'], cr['hosts'], cr['claims'], cr['primary']
    claimsN = np.zeros(shp, bool); AU = np.zeros(shp, bool)
    for c in claims.values():
        if not c['flag']:
            claimsN[c['sl']] |= c['crop']
    for an in anchors.values():
        AU[an['sl']] |= an['crop']
    A_led = {a['id']: a for a in led['anchors']}
    hcls = lambda q: 'unknown' if q is None else allh[q]['cls']
    insuff = bool(cr.get('anatomy_insufficient'))
    S0 = _ld(ckS, rel) or dict(rel=rel, done=-1, rows=[], lrows=[], res={})
    rows, lrows, res = S0['rows'], S0['lrows'], S0['res']; tsave = time.time()
    for li, Lz in enumerate(lesions):
        if li <= S0['done']:
            continue
        cands = {q: Lz['phi'][q] for q in Lz['cand']}                                  # equal weights; no no-host candidate (v8.1)
        # fix 11 (20 Oct): the choice is over ORIGINS. A composite p>n (local-invasion relation) is not an extra alternative; it is one
        # explanation of origin p, and p is scored by its best explanation (alone or invading n): no mass added by the relation.
        sc = {q: HV.score(ph, BETA) for q, ph in cands.items()}
        org_of = lambda q: (allh[q].get('composite') or (q,))[0]
        best = {}
        for q, v in sc.items():
            o_ = org_of(q)
            if o_ not in best or v > sc[best[o_]]: best[o_] = q
        PhL = HV.host_posterior({o_: cands[best[o_]] for o_ in best}, BETA)
        inv_of = lambda o_: hk(allh[best[o_]]['composite'][1]) if (o_ in best and allh[best[o_]].get('composite')) else None
        Sets = {a: G2.credible_set(PhL, a) for a in ALPHAS}
        h_map = max(PhL, key=PhL.get) if PhL else None
        dS = {a: (min(Lz['phi'][best[o_]]['dist_mm'] for o_ in Sets[a]) if Sets[a] else None) for a in ALPHAS}
        far = {a: (not insuff) and dS[a] is not None and dS[a] > HOST_TOL for a in ALPHAS}     # v8.2 step 5 safeguard
        q_near = min(Lz['phi'], key=lambda q: Lz['phi'][q]['dist_mm']) if Lz['phi'] else None
        if not any(r_['lesion'] == li for r_ in lrows): lrows.append(dict(rel=rel, lesion=li, anchors=json.dumps(Lz['anchors']), ml=Lz['ml'], two=Lz['two'],
                          PhL=json.dumps({hk(q): round(v, 4) for q, v in PhL.items()}),
                          phi=json.dumps({hk(q): {k: round(v, 4) for k, v in ph.items()} for q, ph in Lz['phi'].items()}),
                          **{f'S_{a}': json.dumps([hk(q) for q in Sets[a]]) for a in ALPHAS}, h_map=hk(h_map), h_map_cls=hcls(h_map), h_map_invades=inv_of(h_map),
                          explanations=json.dumps({hk(o_): hk(q) for o_, q in best.items() if q != o_}),
                          d_S=dS[ALPHA], host_far=far[ALPHA], nearest=hk(q_near), d_nearest=(Lz['phi'][q_near]['dist_mm'] if q_near is not None else None),
                          static=json.dumps([hk(q) for q in Lz.get('static', [])]), d_static=Lz.get('d_static'),
                          entry=json.dumps({hk(q): r for q, r in Lz.get('entry', {}).items()}), barred=json.dumps({hk(q): r for q, r in (Lz.get('barred') or {}).items()}), contain=json.dumps(Lz.get('contain') or {}),
                          regions=json.dumps({hk(q): [allh[q].get('region_src'), (allh[q].get('region_info') or {}).get('used')] for q in Lz['cand'] if allh[q].get('state') == 'unsegmented'})))
        for i in Lz['anchors']:
            if i in res:                                                       # resumed inside a lesion (checkpoint per anchor)
                continue
            an = anchors[i]; al = A_led[i]
            ext = an['extents']
            w0 = G2.extent_prior([dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ext], PSI, ETA)
            xi = ext[int(np.argmax(w0))]
            Wa = pad(an['sl'], sp, 52, shp)
            Xw = np.asarray(X[Wa], np.float32); aw = place(an['sl'], an['crop'], Wa)
            T2 = {}                                                            # v8.6: tier-2 context built only when a level-1 host needs it
            def _t2():
                if not T2:
                    T2.update(near_=G1.dil(aw, sp, G1.GP1['win_mm']), clD=G1.dil(claimsN[Wa], sp, G1.GP1['margin_mm']),
                              auD=G1.dil(AU[Wa], sp, G1.GP1['margin_mm']))
                return T2

            def ntis(core_w):
                t2 = _t2(); near_, clD, auD = t2['near_'], t2['clD'], t2['auD']
                N = core_w & ~clD
                if (N & near_).sum() >= G1.GP1['min_ref']: return N, None
                N = core_w & ~auD
                if (N & near_).sum() >= G1.GP1['min_ref']: return N, 'ref_fallback'
                return None, 'no_ref'
            ev = {}; ecache = {}
            for q in Sets[0.01]:                                                    # evidence for every host of the largest credible set
                if q is None:
                    theta_w = core_w = None; cls = None
                else:
                    h = allh[best[q]]; theta_w = h['theta'][Wa] if h['theta'].any() else None; core_w = h['core'][Wa] if h['core'].any() else None
                    cls = G1.organ_class(KB, q[0].split('>')[0])            # composite p>n: the origin's class
                sig = al['sig_h'].get(hk(best[q]) if q in best else hk(q), al['sig'])
                lv = G2.support_level(an['kind'], sig)
                if lv != 1:                                                         # v8.6 tier 1: the test cannot change the decision
                    ev[q] = dict(tree='not_run', appl=None, ref=None, cls=cls if (theta_w is not None and theta_w.any()) else None,
                                 vol=round(float(al['ml']), 2), lam=None, level=lv, sig=sig,
                                 test={str(a): 'not_run' for a in ALPHAS}, dec={str(a): decide86(lv, 'not_run') for a in ALPHAS})
                    continue
                N, nf = (ntis(core_w) if (core_w is not None and core_w.any()) else (None, 'no_core'))
                ckey = None   # fix 20 Oct: no sharing across unsegmented hosts (the in-host fraction 'appl' depends on each host's own region)
                if ckey is not None and ckey in ecache:
                    e_ = ecache[ckey]
                else:
                    try:
                        e_ = G1.existence_evidence(Xw, sp, si, aw, theta_w, N, TAB1, mod, cls)
                    except Exception as ex_:
                        e_ = dict(tree='error', err=repr(ex_)[:60])
                    if ckey is not None: ecache[ckey] = e_
                cls_used = cls if (theta_w is not None and theta_w.any()) else None
                vol = float(e_.get('vol_ml', al['ml']) or al['ml'])
                tests = {a: ('neutral' if off else TS.test(cls_used, vol, mod, e_.get('tree'), e_.get('appl'), a)) for a in ALPHAS}
                ev[q] = dict(tree=e_.get('tree'), appl=None if e_.get('appl') is None else round(float(e_['appl']), 3), ref=nf,
                             cls=cls_used, vol=round(vol, 2), lam=round(TS.lam(cls_used, vol, mod, e_.get('tree')), 3), level=lv, sig=sig,
                             test={str(a): tests[a] for a in ALPHAS}, dec={str(a): decide86(lv, tests[a]) for a in ALPHAS})
            dec0 = {a: (decide_set86({q: ev[q]['dec'][str(a)] for q in Sets[a]}, h_map) if Sets[a] else 'uncertain') for a in ALPHAS}
            dec = {a: (CAP(dec0[a]) if far[a] else dec0[a]) for a in ALPHAS}                     # v8.2: host beyond tolerance -> <= uncertain
            roles = sorted({hcls(q) for q in Sets[ALPHA]})
            role = 'unknown' if (h_map is None or far[ALPHA]) else hcls(h_map)
            res[i] = dict(dec=dec[ALPHA], role=role, far=far[ALPHA], h=h_map, xi=xi, lesion=li, S=Sets[ALPHA], lvl=ev[h_map]['level'] if h_map in ev else None)
            rows.append(dict(rel=rel, cancer=rel.split('/')[1], mod=mod, qc=led['qc']['state'], id=i, kind=an['kind'], agents=al['agents'],
                             ml=al['ml'], lesion=li, PhL=json.dumps({hk(q): round(v, 3) for q, v in PhL.items()}),
                             S=json.dumps([hk(q) for q in Sets[ALPHA]]), S_size=len(Sets[ALPHA]),
                             level=json.dumps({hk(q): ev[q]['level'] for q in Sets[ALPHA]}),
                             test=json.dumps({hk(q): ev[q]['test'][str(ALPHA)] for q in Sets[ALPHA]}),
                             ev=json.dumps({hk(q): v for q, v in ev.items()}),
                             dec_001=dec[0.01], dec_010=dec[0.10], decision=dec[ALPHA], decision_before_safeguard=dec0[ALPHA],
                             reason='host_beyond_tolerance' if (far[ALPHA] and dec0[ALPHA] == 'accepted') else None,
                             host='none_within_tolerance' if far[ALPHA] else hk(h_map), host_map=hk(h_map), host_cls=role, S_roles=json.dumps(roles), role_agree=len(roles) == 1,
                             d_S=dS[ALPHA], host_far=far[ALPHA], entry=json.dumps({hk(q): r for q, r in Lz.get('entry', {}).items()}),
                             invades=None if far[ALPHA] else inv_of(h_map), host_source='assumed_primary' if insuff else 'inferred', override=False, imposed_host=None, imposed_cls=None,
                             extent=xi['name'], extent_ml=xi['ml'], ratio=round(xi['ml'] / max(al['ml'], 1e-6), 2),
                             extents=json.dumps([dict(name=e['name'], ml=e['ml'], flags=e.get('flags', []), w0=round(float(w), 4)) for e, w in zip(ext, w0)]),
                             contested=al['contested']))
            if time.time() - tsave > 5:
                _dump(ckS, dict(rel=rel, done=li - 1, rows=rows, lrows=lrows, res=res)); tsave = time.time()
        if time.time() - tsave > 5 or li == len(lesions) - 1:
            _dump(ckS, dict(rel=rel, done=li, rows=rows, lrows=lrows, res=res)); tsave = time.time()
    # policy: primary not detected -> flagged override (lesion whose S contains the primary, highest level, then largest volume)
    in_S_prim = lambda S_: [q for q in S_ if q in prim or (allh[q].get('composite') or (None,))[0] in prim]   # composite p>n counts as p
    prim_acc = any(v['dec'] == 'accepted' and in_S_prim(v['S']) for v in res.values())
    note = None
    if False:                                                                 # v8.6 R3: the override is removed
        cand_l = []
        for li, Lz in enumerate(lesions):
            ids = [i for i in Lz['anchors'] if i in res and in_S_prim(res[i]['S']) and res[i]['dec'] != 'rejected' and not res[i]['far']
                   and (not cr.get('anatomy_insufficient') or anchors[i]['kind'] == 'two')]   # fallback: override two-agent anchors only
            if not ids: continue
            lv = max((json.loads(r['level']).get(hk(in_S_prim(res[r['id']]['S'])[0]), 0) for r in rows if r['id'] in ids), default=0)
            cand_l.append((lv, Lz['ml'], li))
        if cand_l:
            note = 'primary_not_detected'
            _, _, lbest = max(cand_l)
            for r in rows:
                if r['lesion'] == lbest and r['decision'] != 'rejected':
                    PhL_ = json.loads(r['PhL']); pq = max(in_S_prim(res[r['id']]['S']), key=lambda q: PhL_.get(hk(q), 0))
                    r.update(override=True, decision='accepted_override', imposed_host=hk(pq), imposed_cls='primary')
                    res[r['id']].update(dec='accepted_override', h=pq)
        else:
            note = 'primary_not_detected_no_candidate'
    # panel cache (2-D slices)
    accepted = np.zeros(shp, bool)
    for i, v in res.items():
        if v['dec'] in ('accepted', 'accepted_override'):
            accepted[anchors[i]['sl']] |= anchors[i]['crop']
    cdir = f'{O}/cache/{k_idx}'; os.makedirs(cdir, exist_ok=True)
    for i, v in res.items():
        if os.path.exists(f'{cdir}/a{i}.npz'):
            continue
        an = anchors[i]; xi = v['xi']
        Wp = pad(ubox([an['sl'], xi['sl']]), sp, 40, shp)
        need = [int(np.ceil(110 / s)) for s in sp]
        Wp = tuple(slice(max(0, s.start - max(0, (nd - (s.stop - s.start)) // 2 + 1)), min(n, s.stop + max(0, (nd - (s.stop - s.start)) // 2 + 1)))
                   for s, nd, n in zip(Wp, need, shp))
        am = place(an['sl'], an['crop'], Wp); xm = place(xi['sl'], xi['crop'], Wp)
        z = int(np.argmax(am.sum(axis=tuple(d for d in range(3) if d != ax))))
        ctr = [int(np.mean(np.nonzero(am.any(axis=tuple(x for x in range(3) if x != d)))[0])) for d in range(3)]
        orth = [d for d in range(3) if d != ax][int(np.argmin([sp[d] for d in range(3) if d != ax]))]
        Xp = np.asarray(X[Wp], np.float32)
        ol = {'BP': np.zeros(am.shape, bool), 'VT': np.zeros(am.shape, bool), 'FL': np.zeros(am.shape, bool)}
        for o in A_led[i]['outlines']:
            c = claims[o['id']]; m = place(c['sl'], c['crop'], Wp)
            ol['FL' if (c['flag'] or c['sig']) else c['agent']] |= m
        hq = v['h']; hcore = allh[hq]['core'][Wp] if (hq is not None and allh[hq]['core'].any()) else np.zeros(am.shape, bool)
        hth = allh[hq]['theta'][Wp] if (hq is not None and allh[hq]['theta'].any()) else np.zeros(am.shape, bool)
        oth = accepted[Wp] & ~am
        d = {}
        for tag, dd, s_ in (('a', ax, z), ('o', orth, ctr[orth])):
            take = lambda M: np.take(M, s_, axis=dd)
            d[f'{tag}_img'] = take(Xp).astype(np.float32); d[f'{tag}_sp'] = np.array([sp[x] for x in range(3) if x != dd])
            for nm, M in (('anchor', am), ('xi', xm), ('bp', ol['BP']), ('vt', ol['VT']), ('fl', ol['FL']), ('core', hcore), ('theta', hth), ('oth', oth)):
                d[f'{tag}_{nm}'] = np.packbits(take(M)); d[f'{tag}_{nm}_shape'] = np.array(take(M).shape)
            d[f'{tag}_info'] = np.array([dd, s_])
        np.savez_compressed(f'{cdir}/a{i}.npz', **d)
    srow = dict(rel=rel, cancer=rel.split('/')[1], mod=rel[:2], qc=led['qc']['state'], n_anchors=len(rows),
                n_acc=sum(r['decision'].startswith('accepted') for r in rows),
                n_unc=sum(r['decision'] == 'uncertain' for r in rows), n_rej=sum(r['decision'] == 'rejected' for r in rows),
                n_acc_proximal=sum(r['decision'].startswith('accepted') and r['host_cls'] == 'proximal' for r in rows),
                n_far=sum(bool(r['host_far']) for r in rows), n_capped=sum(r['reason'] == 'host_beyond_tolerance' for r in rows),
                n_kb_hosts=sum(h.get('source') == 'kb' for h in led['hosts']),
                anatomy_insufficient=insuff, not_assessable=json.dumps(led.get('not_assessable') or []),
                note=note, k=k_idx, sec=round(time.time() - t0, 1))
    return rows, lrows, srow


if __name__ == '__main__':
    bud = float(sys.argv[1]) if len(sys.argv) > 1 else 120; t0 = time.time()
    se = pd.read_csv(f'{B}/v6a_ledger/series.csv')
    rels = [r for r, n in se[['rel', 'note']].itertuples(index=False) if not isinstance(n, str)]
    only = os.environ.get('ONLY')                                    # smoke test: comma-separated series ids (substring match)
    done = set().union(*[set(pd.read_csv(f).rel) for f in glob.glob(f'{O}/series*.csv')]) if glob.glob(f'{O}/series*.csv') else set()
    for k_idx, rel in enumerate(rels):
        if only and not any(s_ in rel for s_ in only.split(',')): continue
        if k_idx % SN != SI or rel in done: continue
        if time.time() - t0 > bud: break
        k = 0
        if os.path.exists(IP):
            r0, _, k0 = open(IP).read().strip().partition('|'); k = int(k0 or 1) if r0 == rel else 0
        if k >= int(os.environ.get('MAXTRY', '40')):                    # attempts per series (a call can kill a long series)
            out('series', [dict(rel=rel, note='timeout', k=k_idx)]); os.remove(IP); continue
        open(IP, 'w').write(f'{rel}|{k + 1}')
        try:
            rows, lrows, srow = run(rel, k_idx); out('anchors', rows); out('lesions', lrows); out('series', [srow])
        except Exception:
            import traceback
            out('series', [dict(rel=rel, note='error: ' + traceback.format_exc()[-400:].replace('\n', ' | '), k=k_idx)])
        os.remove(IP)
        print(rel.split('/')[1], round(time.time() - t0), flush=True)
