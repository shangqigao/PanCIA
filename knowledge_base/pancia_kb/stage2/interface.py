"""Stage 1 (ledger v8.4, frozen) -> stage-2 inputs for one TCGA series. Agent-agnostic: only stage-1 outputs are passed on.

Per series: the ledger is rebuilt deterministically (ledger_v84.build_ledger); P(h|L) and the credible set are recomputed exactly as in
g3_run_v84.py (origins scored by their best explanation, beta_UD of fit_host6.json, alpha 0.05) and checked against the stored
v84_pilot rows (mismatches are reported, not corrected). Decisions are read from the stored rows (stage 1 is not re-decided).
Lesions with at least one accepted / accepted_override / uncertain anchor are passed on, with their non-rejected anchors only.
Per lesion: anchors, q^L_v = 1 - prod_a (1 - q^a_v) with q^a_v = sum_xi w0_a(xi) 1[v in xi], the chosen extent xi_L (union of the
anchors' argmax-w0 extents), S_L with P(h|L) renormalised over S_L (host key = the origin's best explanation, e.g. a composite p>n
whose territory is the union of p and n), and the tumour class of the most probable host."""
import os, json
import numpy as np
import nibabel as nib
from . import features as F, run as RN

ALPHA = 0.05
KEEP = ('accepted', 'accepted_override', 'uncertain')


class Stage1:
    def __init__(self, kb, beta, psi, pilot_rows, seg_root, nifti_root):
        from .. import ledger_v84 as L8, host_v81 as HV, grounding_g1 as G1, grounding_g2 as G2
        from ..ledger_v6 import _hk
        self.L8, self.HV, self.G1, self.G2, self.hk = L8, HV, G1, G2, _hk
        self.kb, self.beta, self.psi, self.rows = kb, beta, psi, pilot_rows
        self.seg_root, self.nifti_root = seg_root, nifti_root

    def series(self, rel, cache=None):
        L8, HV, G1, G2, hk = self.L8, self.HV, self.G1, self.G2, self.hk
        params = dict(aq_cache=os.path.join(cache, 'aq.pkl'), lobe_cache=os.path.join(cache, 'lobe.pkl')) if cache else None
        led, cr, aff, shp = L8.build_ledger(rel, self.seg_root, self.kb, params=params)
        if not cr:
            return None
        man = json.load(open(os.path.join(self.seg_root, 'VoxTell', 'Radiology', f'{rel}_plan_manifest.json')))
        img = nib.load(os.path.join(self.nifti_root, f"{man['img_name']}.nii.gz"))
        sp = np.array(led['spacing'], float); si = cr['si']; mod = rel[:2]
        thick, ax = F.acq_axis(sp, si)
        anchors, lesions, allh = cr['anchors'], cr['lesions'], cr['hosts']
        R = self.rows[self.rows.rel == rel].set_index('id')
        out, checks = [], []
        for li, Lz in enumerate(lesions):
            cands = {q: Lz['phi'][q] for q in Lz['cand']}
            sc = {q: HV.score(ph, self.beta) for q, ph in cands.items()}
            org_of = lambda q: (allh[q].get('composite') or (q,))[0]
            best = {}
            for q, v in sc.items():
                o_ = org_of(q)
                if o_ not in best or v > sc[best[o_]]: best[o_] = q
            PhL = HV.host_posterior({o_: cands[best[o_]] for o_ in best}, self.beta)
            S = G2.credible_set(PhL, ALPHA) if PhL else []
            ids = [i for i in Lz['anchors'] if i in R.index and str(R.loc[i, 'decision']) in KEEP]
            # consistency with the stored stage-1 rows
            for i in Lz['anchors']:
                if i in R.index:
                    stored = json.loads(R.loc[i, 'PhL']); mine = {hk(q): round(v, 3) for q, v in PhL.items()}
                    ok = all(abs(stored.get(k, 0) - mine.get(k, 0)) <= 0.002 for k in set(stored) | set(mine))
                    checks.append(dict(rel=rel, lesion=li, id=i, phl_match=ok, S_match=json.loads(R.loc[i, 'S']) == [hk(q) for q in S]))
            if not ids:
                continue
            dec = {i: str(R.loc[i, 'decision']) for i in ids}
            tot = sum(PhL[o] for o in S) or 1.0
            PhS = {best[o]: PhL[o] / tot for o in S}
            # q^L and chosen extent on the union box of the anchors' extents
            Wq = RN.box_union([anchors[i]['extents'][0]['sl'] for i in ids] + [anchors[i]['sl'] for i in ids], shp)
            shape_q = tuple(s.stop - s.start for s in Wq)
            omq = np.ones(shape_q, np.float32); xi = np.zeros(shape_q, bool); ext_rows = {}
            for i in ids:
                ext = anchors[i]['extents']
                w0 = G2.extent_prior([dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ext], self.psi, {})
                qa = np.zeros(shape_q, np.float32)
                for e, w in zip(ext, w0):
                    qa += float(w) * RN.win(RN.Patch(e['sl'], e['crop']), Wq).astype(np.float32)
                omq *= (1.0 - np.clip(qa, 0, 1))
                ej = ext[int(np.argmax(w0))]; xi |= RN.win(RN.Patch(ej['sl'], ej['crop']), Wq)
                aw_ = RN.win(RN.Patch(anchors[i]['sl'], anchors[i]['crop']), Wq)
                ext_rows[i] = dict(chosen=ej['name'], w0=[round(float(w), 4) for w in w0], names=[e['name'] for e in ext],
                                   q_anchor_min=round(float(qa[aw_].min()), 4) if aw_.any() else None)
            hmap = max(PhL, key=PhL.get) if PhL else None
            ent = (hmap[0].split('>')[0] if isinstance(hmap, tuple) else None)
            out.append(dict(key=f'L{li}', lesion=li, anchor_ids=ids, decisions=dec,
                            anchors=[RN.Patch(anchors[i]['sl'], anchors[i]['crop']) for i in ids],
                            q=RN.Patch(Wq, 1.0 - omq), xi=RN.Patch(Wq, xi), PhL=PhS, S=list(PhS),
                            S_names=[hk(q) for q in PhS], PhL_all={hk(o): round(v, 4) for o, v in PhL.items()},
                            h_map=hk(hmap) if hmap is not None else 'unknown',
                            tum_cls=G1.organ_class(self.kb, ent) if ent else 'solid', ext=ext_rows,
                            kinds={i: anchors[i]['kind'] for i in ids}))
        organs = {}
        for q, h in allh.items():
            th = h.get('theta')
            if th is None or not th.any():
                continue
            comp = bool(h.get('composite'))
            ent = q[0].split('>')[0] if isinstance(q, tuple) else str(q)
            organs[q] = dict(theta=th, kb=(h.get('state') == 'unsegmented'), bg=not comp, name=hk(q), mu=float(h.get('mu', 0) or 0),
                             app=_app_key(ent), cls=G1.organ_class(self.kb, ent))
        X = np.asanyarray(img.dataobj)
        X = np.asarray(X[..., 0] if X.ndim == 4 else X, np.float32)
        cen, scl = F.norm_params(X, mod, F.body_mask(X, mod, ax) if mod == 'MR' else None)
        body = F.region_body(X, mod, ax)
        return dict(rel=rel, kb=self.kb, led=led, cr=cr, aff=aff, shape=shp, sp=sp, si=si, mod=mod, thick=thick, ax=ax, X=X, cen=cen, scl=scl,
                    body=RN.Patch.from_full(body), lesions=out, organs=organs, checks=checks,
                    insufficient=bool(cr.get('anatomy_insufficient')))


def _app_key(ent):
    """TCGA organ entity -> expert appearance key (stage2.expert names); unknown names fall back to the organ class"""
    e = str(ent)
    if e.startswith('kidney'): return 'kidney'
    if e.startswith('lung'): return 'lung'
    if e.startswith('adrenal'): return 'adrenal_gland'
    return e


def clusters(lesions, organs, body, r, shape, sp):
    """lesions whose regions R_L overlap (union-find); returns list of lists of lesion indices, and the regions"""
    regs = [RN.lesion_region(L, organs, body, r, shape, sp) for L in lesions]
    par = list(range(len(lesions)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            if f(i) == f(j):
                continue
            W = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(regs[i].sl, regs[j].sl))
            if any(s.start >= s.stop for s in W):
                continue
            if (RN.win(regs[i], W) & RN.win(regs[j], W)).any():
                par[f(j)] = f(i)
    groups = {}
    for i in range(len(regs)):
        groups.setdefault(f(i), []).append(i)
    return list(groups.values()), regs


def existence_on_mask(s1, Zfull_patch, L, TAB1, TS):
    """stage-1 existence evidence and test (alpha 0.05) recomputed on a mask under each host of S_L (as g3_run_v84)"""
    G1 = s1_mod_G1()
    cr = s1['cr']; sp = s1['sp']; shp = s1['shape']; si = s1['si']; mod = s1['mod']; X = s1['X']
    off = s1['led']['qc']['state'] != 'good'
    b = Zfull_patch.sl
    Wa = F.pad_box(b, sp, 52.0, shp)
    aw = RN.win(Zfull_patch, Wa)
    if not aw.any():
        return {}
    claimsN = np.zeros(aw.shape, bool); AU = np.zeros(aw.shape, bool)
    for c in cr['claims'].values():
        if not c['flag']:
            claimsN |= RN.win(RN.Patch(c['sl'], c['crop']), Wa)
    for an in cr['anchors'].values():
        AU |= RN.win(RN.Patch(an['sl'], an['crop']), Wa)
    near_ = G1.dil(aw, sp, G1.GP1['win_mm'])
    clD = G1.dil(claimsN, sp, G1.GP1['margin_mm']); auD = G1.dil(AU, sp, G1.GP1['margin_mm'])
    Xw = np.asarray(X[Wa], np.float32)
    out = {}
    for q in L['S']:
        h = cr['hosts'][q]
        theta_w = h['theta'][Wa] if h['theta'].any() else None; core_w = h['core'][Wa] if h['core'].any() else None
        ent = q[0].split('>')[0] if isinstance(q, tuple) else None
        cls = G1.organ_class(s1['kb'], ent) if ent else None
        N, nf = None, 'no_core'
        if core_w is not None and core_w.any():
            N = core_w & ~clD
            nf = None
            if (N & near_).sum() < G1.GP1['min_ref']:
                N = core_w & ~auD; nf = 'ref_fallback'
                if (N & near_).sum() < G1.GP1['min_ref']:
                    N, nf = None, 'no_ref'
        try:
            e_ = G1.existence_evidence(Xw, sp, si, aw, theta_w, N, TAB1, mod, cls)
        except Exception as ex_:
            e_ = dict(tree='error', err=repr(ex_)[:60])
        cls_used = cls if (theta_w is not None and theta_w.any()) else None
        vol = float(e_.get('vol_ml') or (aw.sum() * np.prod(sp) / 1000))
        test = 'neutral' if off else TS.test(cls_used, vol, mod, e_.get('tree'), e_.get('appl'), ALPHA)
        out[L['S_names'][L['S'].index(q)]] = dict(tree=e_.get('tree'), test=test, ref=nf)
    return out


def s1_mod_G1():
    from .. import grounding_g1 as G1
    return G1
