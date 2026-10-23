"""Image grounding G0 -- inference (17 Oct 2026). main.tex Sec. 'Image grounding' (Eqs. extents, case_posterior, tree, mstep);
rules documents/plan/Image_grounding_rules_draft.md (G0, user decisions 16-17 Oct).

For every anchor a of a scan the latent variables are its host h in H_a, its existence z_a and its extent xi_a in X_a (a choice among
set operations on the agents' outlines; nothing is redrawn). The evidence E = (tree outcome, S5) of an extent under a host is
computed against the host's current normal tissue N_h (host reference minus every voxel with tumour probability >= pi_none, with a
2 mm margin) on the common ~5 mm grid, and scored with the frozen expert likelihood ratios (tables_final.json, g0/h5):
    Lambda(xi, h) = p(E | T) / p(E | N, h)       class (solid/hollow) x size band x modality; S5 unknown -> marginal;
                                                 clipped [1/100, 100]; < 3 ml: >= 1/3; support (>1) only inside the host (appl >= 0.5)
E-step  (likelihood-ratio form of Eq. case_posterior: the normal likelihood is taken on the same extent)
    r(h, 1, xi) ~ P(h|a) pi(a,h) w(xi) Lambda(xi, h),     r(h, 0) ~ P(h|a) (1 - pi(a,h))
M-step  Eq. mstep: w <- 1/2 w + 1/2 (rho + alpha w0)/(1 + alpha); P(h|a) <- 1/2 P + 1/2 (gamma_bar_L + alpha P0)/sum, alpha = 1;
        P(primary|a) = 1 held fixed; gamma_bar_L = volume-weighted mean over the anchors of a's lesion under h.
        q(v) = max_a p_a sum_{xi ∋ v} rho_a(xi) is updated, N_h recomputed. Stop: no weight change > 0.01 and no argmax change, or 10
        iterations (flag not_converged).
Decision  (user, 17 Oct) the reasoning status stands unless the image is decisive: recover if p_a >= 0.5 and
        Lambda_hat = sum_h P(h|a) Lambda(xi_hat, h) >= 3; demote if p_a < 0.5 and Lambda_hat <= 1/3; the largest reasoning-admitted
        primary anchor is never demoted. Host = argmax P, extent = argmax w. Degraded scans (image quality control): Lambda = 1.
Pure numpy/scipy on (slice, crop) boxes; no ledger import. The driver (scripts/g0_run.py) builds the inputs from ledger v6a.
"""
from __future__ import annotations
import json
import numpy as np
from scipy import ndimage
from . import grounding as GR

GP = dict(alpha=1.0, damp=0.5, max_iter=10, tol=0.01, pi_none=0.1, admit_p=0.5, margin_mm=2.0, win_mm=50.0, close_win_mm=32.0,
          dice_merge=0.95, appl_min=0.5, small_cap=1.0 / 3.0, decisive_hi=3.0, decisive_lo=1.0 / 3.0, min_ref=200, two_prior=0.9, out_prior=0.5)
HOLLOW_EXTRA = {'uterus', 'cervix'}          # user 16 Oct: organ classes solid / hollow only; uterus and cervix -> hollow
TREE = ['tumour', 'uncertain', 'organ']
VERSION = 'g0.1'


# ------------------------------------------------------------------ boxes
class SB:
    """boolean or float mask stored on its bounding box (sl, crop) of a grid of given shape"""
    def __init__(self, sl=None, crop=None, shape=None):
        self.sl, self.crop, self.shape = sl, crop, shape

    @staticmethod
    def of(mask, shape=None, offset=None):
        shape = shape or mask.shape
        if mask is None or not mask.any():
            return SB(None, None, shape)
        s = ndimage.find_objects(mask.astype(np.int8))[0]
        if offset is not None:
            return SB(tuple(slice(a.start + o, a.stop + o) for a, o in zip(s, offset)), mask[s].copy(), shape)
        return SB(s, mask[s].copy(), shape)

    def any(self):
        return self.crop is not None and bool(self.crop.any())

    def sum(self):
        return 0 if self.crop is None else int(np.count_nonzero(self.crop))

    def get(self, sl, fill=False):
        out = np.full(tuple(s.stop - s.start for s in sl), fill, dtype=bool if self.crop is None else self.crop.dtype)
        if self.crop is None:
            return out
        lo = [max(a.start, b.start) for a, b in zip(self.sl, sl)]; hi = [min(a.stop, b.stop) for a, b in zip(self.sl, sl)]
        if any(l >= h for l, h in zip(lo, hi)):
            return out
        out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, sl))] = \
            self.crop[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, self.sl))]
        return out


def _ubox(boxes, shape):
    bs = [b for b in boxes if b is not None and b.any()]
    if not bs:
        return SB(None, None, shape)
    sl = tuple(slice(min(b.sl[d].start for b in bs), max(b.sl[d].stop for b in bs)) for d in range(3))
    m = np.zeros(tuple(s.stop - s.start for s in sl), bool)
    for b in bs:
        m |= b.get(sl)
    return SB(sl, m, shape)


def _pad(sl, sp, mm, shape):
    p = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    return tuple(slice(max(0, s.start - a), min(n, s.stop + a)) for s, a, n in zip(sl, p, shape))


def _dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * (a & b).sum() / s if s else 0.0


# ------------------------------------------------------------------ tables
def organ_class(kb, ent):
    if ent in HOLLOW_EXTRA:
        return 'hollow'
    cat = (kb.entities.get(ent, {}) if kb is not None else {}).get('category')
    return 'hollow' if cat == 'hollow_organ' else 'solid'


def band(vol_ml):
    return '<3' if vol_ml < 3 else ('3-20' if vol_ml <= 20 else '>20')


class Tables:
    def __init__(self, path_or_dict):
        t = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
        self.t = t; self.cells = [tuple(c) for c in t['cells']]
        vals = [v for d in ('lam', 'lam_int', 'lam_marg') for lst in t.get(d, {}).values() for v in lst]
        self.max_lam = float(min(max(vals), float(t.get('cap', 100.0)))) if vals else float(t.get('cap', 100.0))

    def lam(self, cls, vol_ml, mod, tree, s5, appl):
        b = band(vol_ml); mod = mod if mod in ('CT', 'MR') else 'CT'
        if cls is None:                                          # no host frame: intrinsic table (no context level)
            L = self.t['lam_int'][f'{b}|{mod}'][TREE.index(tree)]
        elif s5 in ('yes', 'no'):
            L = self.t['lam'][f'{cls}|{b}|{mod}'][self.cells.index((tree, s5))]
        else:
            L = self.t['lam_marg'][f'{cls}|{b}|{mod}'][TREE.index(tree)]
        cap = float(self.t.get('cap', 100.0)); L = float(np.clip(L, 1 / cap, cap))
        if b == '<3':
            L = max(L, GP['small_cap'])
        if cls is not None and (appl is None or appl < GP['appl_min']) and L > 1:
            L = 1.0                                              # support only inside the host (user, 16 Oct)
        return L


# ------------------------------------------------------------------ evidence of one extent under one host
def evidence(X, sp, si_ax, xi, host, qmap, tabs, mod):
    """X image crop (window), xi bool crop, host dict(H=bool crop or None, cls) or None, qmap float crop (tumour probability).
    Returns dict(tree, s5, appl, zh, open, rel_tex, elong, ref_n, vol_ml, lam). Signs on the ~5 mm grid."""
    k = GR.thick_k(sp[si_ax])
    if k > 1:
        X = GR.harmonise_array(X, si_ax, k); xi = GR.harmonise_array(xi.astype(np.int16), si_ax, k, label=True) > 0
        H = GR.harmonise_array(host['H'].astype(np.int16), si_ax, k, label=True) > 0 if host and host.get('H') is not None else None
        qmap = GR.harmonise_array(qmap.astype(np.float32), si_ax, k) if qmap is not None else None
        sp = np.array(sp, float); sp[si_ax] = sp[si_ax] * k
    else:
        H = host['H'] if host and host.get('H') is not None else None
    sp = np.asarray(sp, float)
    out = dict(vol_ml=float(xi.sum() * np.prod(sp) / 1000))
    if not xi.any():
        out.update(tree='uncertain', lam=1.0); return out
    thick = sp.max() / sp.min() >= 1.5; ax = int(np.argmax(sp)) if thick else si_ax
    l2 = GR.level2(X, xi, sp, thick, ax); out['rel_tex'] = l2.get('rel_tex'); out['elong'] = l2.get('elong')
    zh = op = None; appl = None; s5 = None
    if H is not None and H.any():
        hw = H & ~xi
        excl = (qmap >= GP['pi_none']) & ~xi if qmap is not None else np.zeros_like(xi)
        if excl.any():
            excl = GR._edt(~excl, sp) <= GP['margin_mm']
        Hf = _fill2d(H, ax) | H
        out['appl'] = appl = float((xi & (GR._edt(~Hf, sp) <= 2.0)).sum() / max(xi.sum(), 1))
        Xs = GR.smooth(X, sp, thick, ax)
        Z, D, n = GR.host_frame(Xs, sp, hw, xi, hw & ~excl, min_ref=GP['min_ref']); out['ref_n'] = n
        if Z is not None:
            Wr = _bbox(xi, sp, GP['close_win_mm'], xi.shape)
            f = GR.closed30(X[Wr], Z[Wr], hw[Wr], xi[Wr], D[Wr], sp, thick, ax)
            zh = f.get('zh_med')
            if appl >= GP['appl_min']:
                op = f.get('open_ratio')
        Hd = H | (xi & Hf)
        t = GR.organ_tests(xi, Hd, sp, ax)
        s5 = 'yes' if t in ('a', 'b') else 'no'
    out.update(zh=zh, open=op, s5=s5)
    tree = GR.tree_call(dict(open_ratio=op, zh_med=zh, rel_tex=out['rel_tex'], elong=out['elong'], vol_ml=out['vol_ml']))
    out['tree'] = tree
    cls = host.get('cls') if (host and H is not None and H.any()) else None
    out['lam'] = tabs.lam(cls, out['vol_ml'], mod, tree, s5, appl) if tabs is not None else 1.0
    return out


def _fill2d(M, ax):
    M = M.copy()
    for z in range(M.shape[ax]):
        sl = [slice(None)] * 3; sl[ax] = z; M[tuple(sl)] = ndimage.binary_fill_holes(M[tuple(sl)])
    return M


def _bbox(M, sp, pad, shape):
    idx = np.argwhere(M); lo, hi = idx.min(0), idx.max(0) + 1; p = np.ceil(pad / np.asarray(sp)).astype(int)
    return tuple(slice(int(max(0, a - q)), int(min(n, b + q))) for a, b, q, n in zip(lo, hi, p, shape))


# ------------------------------------------------------------------ extents
def build_extents(anchor, outlines, ok, nonhost, shape):
    """X_a = {a, each unflagged non-leaking outline, U = union of unflagged outlines, U minus organs that are not hosts}; Dice >= 0.95 merged.
    anchor SB; outlines {id: SB}; ok set of unflagged outline ids; nonhost SB (union of non-host organ references) or None."""
    cand = [('anchor', anchor)] + [(f'o{i}', b) for i, b in outlines.items() if i in ok]   # flagged / leaking outlines are not candidates (user, 17 Oct)
    U = _ubox([b for i, b in outlines.items() if i in ok], shape)
    if U.any():
        cand.append(('U', U))
        if nonhost is not None and nonhost.any():
            m = U.crop & ~nonhost.get(U.sl)
            if m.any():
                cand.append(('U-nh', SB(U.sl, m, shape)))
    ext = []
    for name, b in cand:
        if not b.any():
            continue
        dup = False
        for _, e in ext:
            sl = tuple(slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(b.sl, e.sl))
            if _dice(b.get(sl), e.get(sl)) >= GP['dice_merge']:
                dup = True; break
        if not dup:
            ext.append((name, b))
    return ext


# ------------------------------------------------------------------ EM over a scan
def ground_scan(scan, anchors, tabs, log=None):
    """scan: dict(X image array, sp, si_ax, mod, state 'good'|'degraded', hosts {key: dict(H=SB or None, cls, kb_box)},
                  nonhost SB or None, two SB (two-agent agreement union, for the voxel prior)).
    anchors: list of dict(id, mask SB, outlines {id: SB}, ok set, P0 {key: p}, primary_fixed bool, pi float, soft_all bool,
                          admitted0 bool, primary bool, lesion {key: lesion id}, pi_out float (voxel prior outside agreement)).
    Returns list of per-anchor result dicts."""
    X, sp, si_ax, shape = scan['X'], np.asarray(scan['sp'], float), int(scan['si_ax']), scan['X'].shape
    off = scan.get('state') == 'degraded'
    two = scan.get('two')
    st = []
    for a in anchors:
        ext = build_extents(a['mask'], a['outlines'], a['ok'], scan.get('nonhost'), shape)
        w0 = []
        for _, b in ext:
            t = two.get(b.sl) if two is not None else np.zeros(b.crop.shape, bool)
            n = b.crop.sum(); n2 = (b.crop & t).sum()
            w0.append((GP['two_prior'] * n2 + a.get('pi_out', GP['out_prior']) * (n - n2)) / max(n, 1))
        w0 = np.asarray(w0, float); w0 = w0 / w0.sum()
        hk = list(a['P0']) or [None]
        P0 = np.array([a['P0'][h] for h in hk], float) if a['P0'] else np.array([1.0])
        win = _pad(_ubox([b for _, b in ext], shape).sl, sp, GP['win_mm'], shape)
        # an anchor whose prior cannot reach admission even with the largest likelihood ratio of the tables is not grounded
        # (its evidence is set to 1; it still enters q with its prior): pi Lmax / (pi Lmax + 1 - pi) < admit_p
        pi = float(a['pi']); Lm = tabs.max_lam if tabs is not None else 1.0
        fixed = off or (pi * Lm / (pi * Lm + 1 - pi) < GP['admit_p'] and not a.get('admitted0'))
        st.append(dict(a=a, ext=ext, w0=w0, w=w0.copy(), hk=hk, P0=P0 / P0.sum(), P=P0 / P0.sum(), win=win,
                       vox=a['mask'].sum(), lam=None, ev=None, p=None, contrib=None, fixed=fixed))
    # q at t = 0: every kept claim is excluded from the normal tissue (voxel prior >= 0.5 > pi_none)
    for s in st:
        U = _ubox([b for _, b in s['ext']], shape)
        s['contrib'] = SB(U.sl, U.crop.astype(np.float32) * 0.5, shape)
    it, conv = 0, False
    while it < GP['max_iter']:
        it += 1
        # evidence against the current normal tissue
        for s in st:
            W = s['win']; Xw = np.asarray(X[W], np.float32); Xw = Xw[..., 0] if Xw.ndim == 4 else Xw
            q = np.zeros(Xw.shape, np.float32)
            for o in st:
                if o['contrib'].any():
                    q = np.maximum(q, o['contrib'].get(W, fill=0.0).astype(np.float32))
            if s['fixed']:
                if s['lam'] is None:
                    s['lam'] = np.ones((len(s['hk']), len(s['ext']))); s['ev'] = [[{} for _ in s['ext']] for _ in s['hk']]
                continue
            sig = hash(np.packbits(q >= GP['pi_none']).tobytes())     # the normal tissue changes only with the excluded set
            if s.get('sig') == sig and s['lam'] is not None and not off:
                continue
            s['sig'] = sig
            L = np.ones((len(s['hk']), len(s['ext']))); EV = [[None] * len(s['ext']) for _ in s['hk']]
            for i, h in enumerate(s['hk']):
                host = scan['hosts'].get(h) if h is not None else None
                hd = None
                if host is not None and not host.get('kb_box') and host.get('H') is not None and host['H'].any():
                    hd = dict(H=host['H'].get(W), cls=host['cls'])
                for j, (_, b) in enumerate(s['ext']):
                    e = evidence(Xw, sp, si_ax, b.get(W), hd, q, None if off else tabs, scan.get('mod'))
                    EV[i][j] = e; L[i, j] = 1.0 if off else e['lam']
            s['lam'], s['ev'] = L, EV
        # E-step
        for s in st:
            pi = float(s['a']['pi'])
            r1 = s['P'][:, None] * pi * s['w'][None, :] * s['lam']; r0 = s['P'] * (1 - pi)
            Zs = r1.sum() + r0.sum(); r1, r0 = r1 / Zs, r0 / Zs
            s['p'] = float(r1.sum()); s['gamma'] = r1.sum(1) + r0
            s['rho'] = r1.sum(0) / r1.sum() if r1.sum() > 0 else s['w'].copy()
        # M-step
        lesvol = {}
        for s in st:
            for i, h in enumerate(s['hk']):
                key = (h, (s['a'].get('lesion') or {}).get(h, ('a', s['a']['id'])))
                lesvol.setdefault(key, []).append((s['vox'], s['gamma'][i]))
        change, amax = 0.0, False
        for s in st:
            a = s['a']; al = GP['alpha']; d = GP['damp']
            wn = (1 - d) * s['w'] + d * (s['rho'] + al * s['w0']) / (1 + al)
            gb = np.array([np.average([g for _, g in lesvol[(h, (a.get('lesion') or {}).get(h, ('a', a['id'])))]],
                                      weights=[max(v, 1) for v, _ in lesvol[(h, (a.get('lesion') or {}).get(h, ('a', a['id'])))]])
                           for h in s['hk']])
            Pn = s['P'] if a.get('primary_fixed') else (1 - d) * s['P'] + d * (gb + al * s['P0']) / (gb + al * s['P0']).sum()
            change = max(change, float(np.abs(wn - s['w']).max()), float(np.abs(Pn - s['P']).max()))
            amax |= int(np.argmax(wn)) != int(np.argmax(s['w'])) or int(np.argmax(Pn)) != int(np.argmax(s['P']))
            s['w'], s['P'] = wn, Pn
            U = _ubox([b for _, b in s['ext']], shape); c = np.zeros(U.crop.shape, np.float32)
            for (_, b), r in zip(s['ext'], s['rho']):
                c += r * b.get(U.sl).astype(np.float32)
            s['contrib'] = SB(U.sl, c * s['p'], shape)
        if log is not None:
            log.append(dict(iter=it, change=round(change, 4), argmax_change=amax))
        if change <= GP['tol'] and not amax:
            conv = True; break
    # decision
    out = []
    prim_keep = None
    cand = [s for s in st if s['a'].get('primary') and s['a'].get('admitted0')]
    if cand:
        prim_keep = max(cand, key=lambda s: s['vox'])['a']['id']
    for s in st:
        a = s['a']; j = int(np.argmax(s['w'])); i = int(np.argmax(s['P']))
        lam_hat = float((s['P'] * s['lam'][:, j]).sum())
        # user, 17 Oct (first pilot): the set-based status changes only on DECISIVE image evidence (3:1, as the small-band cap):
        # a soft anchor is recovered if its posterior reaches 0.5 AND Lambda_hat >= 3; an admitted one is demoted if its posterior
        # falls below 0.5 AND Lambda_hat <= 1/3. Otherwise the reasoning status stands (priors of 0.5 made p >= 0.5 a knife-edge).
        adm = bool(a.get('admitted0')); why = 'reasoning_kept'
        if adm and s['p'] < GP['admit_p'] and lam_hat <= GP['decisive_lo']:
            adm, why = False, 'image_demoted'
        elif not adm and s['p'] >= GP['admit_p'] and lam_hat >= GP['decisive_hi']:
            adm, why = True, 'image_recovered'
        if not adm and a['id'] == prim_keep:
            adm, why = True, 'largest_primary_kept'
        e = s['ev'][i][j]
        fl = []
        if off: fl.append('grounding_off_degraded')
        elif s['fixed']: fl.append('not_grounded_low_prior')
        if any(ev is not None and ev.get('ref_n') is not None and ev['ref_n'] < GP['min_ref'] for row in s['ev'] for ev in row):
            fl.append('no_normal_reference')
        if e.get('zh') is not None and abs(e['zh']) >= 1.5 and e.get('open') is None:
            fl.append('context_skipped')
        if e.get('zh') is not None and abs(e['zh']) < 1.5:
            fl.append('iso_intense')
        if i != int(np.argmax(s['P0'])): fl.append('host_changed')
        if j != int(np.argmax(s['w0'])): fl.append('extent_changed')
        if not conv: fl.append('not_converged')
        out.append(dict(id=a['id'], p=round(s['p'], 4), admitted=bool(adm), why=why, host=s['hk'][i], extent=s['ext'][j][0],
                        extent_ml=round(s['ext'][j][1].sum() * float(np.prod(sp)) / 1000, 2), lam_hat=round(lam_hat, 3),
                        lam={str(h): [round(x, 3) for x in s['lam'][k]] for k, h in enumerate(s['hk'])},
                        extents=[n for n, _ in s['ext']], w=[round(x, 3) for x in s['w']], w0=[round(x, 3) for x in s['w0']],
                        P={str(h): round(float(x), 3) for h, x in zip(s['hk'], s['P'])},
                        E={k: e.get(k) for k in ('tree', 's5', 'appl', 'zh', 'open', 'rel_tex', 'elong', 'ref_n', 'vol_ml')},
                        flags=fl, iters=it))
    return out
