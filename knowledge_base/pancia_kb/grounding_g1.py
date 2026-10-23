"""Image grounding G1 and the v7 reasoning primitives (18 Oct 2026). main.tex v7: Sec. 2.1 (Eqs. organ_reference, normal_model,
case_posterior, likelihood_ratios, combination) and Sec. 2.3 (compat, existence_prior, extent_prior, extent_evidence, decision policy).
Shared by the expert testbed (scripts/tb_build.py, tb_fit.py) and the TCGA driver (scripts/g1_run.py), so the validated code is the code
that runs. Pure numpy/scipy on full arrays or crops; no ledger import.

One-pass conditional inference (no EM, no feedback):
  normal model  P(h | L) ~ w_h g_h(L)                                   (anatomical compatibility, lesion-level, fixed)
                N_h = core(h) minus every agent's claim (+2 mm), frozen; fallback core minus anchors (flag); else unknown
  tumour model  p(z=1, xi | h) ~ pi(a,h) w0(xi) Lam_ex(h) Lam_ext(xi,h);  p(z=0 | h) ~ 1 - pi(a,h)
                Lam_ex  = expert tree tables (tables_g1.json; tree outcome only, on the ~5 mm grid) computed on the ANCHOR
                Lam_ext = exp( sum_{v in xi \\ a} log f_a(v)/f_h(v) / nu ), the same region R_a for every xi
  combination   p_a = sum_h P(h|L) p(z=1|h);  (h^, xi^) = argmax P(h|L) p(z=1, xi | h)
  policy        accept p >= 0.75, reject p <= 0.25, else uncertain; host outside H_C -> 'other' (reported only)
"""
from __future__ import annotations
import json
import numpy as np
from scipy import ndimage
from . import grounding as GR

GP1 = dict(rho=1.0 / 3.0, pi_two=0.75,   # user 18 Oct: two-agent prior 0.9 -> 0.75 (agreement alone does not reach acceptance)
           pi_single=0.5, accept=0.75, reject=0.25, env_mm=10.0, margin_mm=2.0, win_mm=50.0, min_ref=200,
           R_margin_mm=5.0, lobe_open_mm=3.0, lobe_min_ml=0.5, dice_merge=0.95, ext_clip=100.0, appl_min=0.5, small_cap=1.0 / 3.0,
           close_win_mm=32.0, band_mm=3.0, w_other=0.15, kde_bins=128, kde_eps=0.02)
TREE = ['tumour', 'uncertain', 'organ']
VERSION = 'g1.0'


# ------------------------------------------------------------------ small helpers
def edt(m, sp):
    return GR._edt(m, np.asarray(sp, float))


def dil(m, sp, mm):
    if not m.any():
        return m.copy()
    return edt(~m, sp) <= mm


def ball(mm, sp):
    r = [int(np.ceil(mm / s)) for s in sp]
    g = np.ogrid[tuple(slice(-q, q + 1) for q in r)]
    return sum((gi * s) ** 2 for gi, s in zip(g, sp)) <= mm * mm


def dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * (a & b).sum() / s if s else 0.0


def fill2d(M, ax):
    M = M.copy()
    for z in range(M.shape[ax]):
        sl = [slice(None)] * 3; sl[ax] = z; M[tuple(sl)] = ndimage.binary_fill_holes(M[tuple(sl)])
    return M


def vol_ml(m, sp):
    return float(m.sum() * np.prod(sp) / 1000.0)


# ------------------------------------------------------------------ host reasoning: zones (Eq. organ_reference)
def zones(masks):
    """masks: list of (bool array, mu) for ONE organ (any agents). Returns (core, union) before contested regions are removed:
    core = intersection of the masks with the highest confidence, union = all masks with mu > 0."""
    ok = [(m, mu) for m, mu in masks if mu > 0 and m is not None and m.any()]
    if not ok:
        return None, None, 0.0
    mmax = max(mu for _, mu in ok)
    core = None; uni = None
    for m, mu in ok:
        uni = m.copy() if uni is None else (uni | m)
        if mu == mmax:
            core = m.copy() if core is None else (core & m)
    return core, uni, float(mmax)


# ------------------------------------------------------------------ tumour reasoning: lobes
def split_lobes(omega, a, sp, open_mm=None, min_ml=None):
    """remainder r = omega \\ a, opened with a ball; a component of the opened remainder that is NOT connected to a through the opened
    set (i.e. joined only through voxels removed by the opening, a neck) and holds >= min_ml is a lobe. Returns (omega without lobes,
    [lobes]). Lobes are returned as their full (un-opened) remainder component reached from the opened component."""
    open_mm = GP1['lobe_open_mm'] if open_mm is None else open_mm; min_ml = GP1['lobe_min_ml'] if min_ml is None else min_ml
    r = omega & ~a
    if not r.any():
        return omega, []
    ro = ndimage.binary_opening(r, structure=ball(open_mm, sp))
    if not ro.any():
        return omega, []
    lab, n = ndimage.label(ro, structure=np.ones((3, 3, 3)))
    adj = ndimage.binary_dilation(a, structure=np.ones((3, 3, 3)))     # voxel adjacency (mm dilation misses thick-slice neighbours)
    lobes = []; keep = omega.copy()
    full_lab, _ = ndimage.label(r, structure=np.ones((3, 3, 3)))
    for i in range(1, n + 1):
        c = lab == i
        if (c & adj).any() or vol_ml(c, sp) < min_ml:
            continue
        # region of the remainder that belongs to this opened component: remainder voxels closer to c than to any other opened part
        # or to a (geodesic split approximated by distance)
        dc = edt(~c, sp); da = edt(~a, sp); other = ro & ~c
        do = edt(~other, sp) if other.any() else np.full(r.shape, np.inf)
        part = r & (dc < da) & (dc <= do)
        if vol_ml(part, sp) >= min_ml:
            lobes.append(part); keep &= ~part
    return keep, lobes


# ------------------------------------------------------------------ host compatibility (Eq. compat)
FEATS = ['o_core', 'o_theta', 's_band', 'inside', 'straddle', 'exophytic', 'mu', 'kb']   # kb: the candidate is a KB region (unsegmented)


def compat_features(a, core, theta, mu, sp, others=None, kb=False):
    """a anchor (or lesion union); core / theta of the candidate organ (theta = territory incl. contested); mu organ confidence.
    s_band: share of a's two-sided band (3 mm outside, other anchors excluded, and 3 mm inner rim) inside theta."""
    na = max(int(a.sum()), 1)
    oc = float((a & core).sum() / na) if core is not None else 0.0
    ot = float((a & theta).sum() / na) if theta is not None else 0.0
    bm = GP1['band_mm']
    out_b = dil(a, sp, bm) & ~a
    if others is not None:
        out_b &= ~others
    in_b = a & ~(edt(a, sp) > bm)
    band_ = out_b | in_b
    sb = float((band_ & theta).sum() / max(int(band_.sum()), 1)) if theta is not None else 0.0
    pos = 'inside' if ot >= 0.9 else ('straddle' if ot >= 0.1 else 'exophytic')
    return dict(o_core=oc, o_theta=ot, s_band=sb, inside=float(pos == 'inside'), straddle=float(pos == 'straddle'),
                exophytic=float(pos == 'exophytic'), mu=float(mu), kb=float(bool(kb)))


def compat(phi, beta):
    """g_h = sigmoid(b0 + beta . phi); beta dict with 'b0' and feature weights (fitted on expert data, tb_fit.py)"""
    z = beta.get('b0', 0.0) + sum(beta.get(k, 0.0) * phi[k] for k in FEATS)
    return float(1.0 / (1.0 + np.exp(-z)))


def host_posterior(cands, beta, g_min):
    """cands: dict h -> (w_h, phi or None for no-host). Returns dict h -> P(h | L) ~ w_h g_h."""
    s = {}
    for h, (w, phi) in cands.items():
        s[h] = w * (compat(phi, beta) if phi is not None else g_min)
    Z = sum(s.values())
    return {h: v / Z for h, v in s.items()} if Z > 0 else {h: 1.0 / len(s) for h in s}


# ------------------------------------------------------------------ priors (Eqs. existence_prior, extent_prior)
def odds(p):
    return p / (1 - p)


def inv_odds(o):
    return o / (1 + o)


def existence_prior(two_agent, n_signals, rho=None):
    rho = GP1['rho'] if rho is None else rho
    p0 = GP1['pi_two'] if two_agent else GP1['pi_single']
    return float(inv_odds(odds(p0) * rho ** n_signals))


class Psi:
    """parsimony term psi(log volume ratio of extent to anchor), a piecewise-constant density ratio fitted on expert data (tb_fit.py)"""
    def __init__(self, d=None):
        d = d or dict(edges=[-1e9, 1e9], val=[1.0])
        self.edges = np.asarray(d['edges'], float); self.val = np.asarray(d['val'], float)

    def __call__(self, lr):
        i = int(np.clip(np.searchsorted(self.edges, lr, side='right') - 1, 0, len(self.val) - 1)); return float(self.val[i])


def extent_prior(cands, psi, rho=None):
    """cands: list of dict(pibar, nflags, logratio). Returns normalised w0."""
    rho = GP1['rho'] if rho is None else rho
    w = np.array([c['pibar'] * rho ** c['nflags'] * psi(c['logratio']) for c in cands], float)
    return w / w.sum() if w.sum() > 0 else np.full(len(cands), 1.0 / len(cands))


def merge_extents(names, masks, thr=None):
    thr = GP1['dice_merge'] if thr is None else thr
    keep = []
    for n, m in zip(names, masks):
        if m is None or not m.any():
            continue
        if any(dice(m, k[1]) >= thr for k in keep):
            continue
        keep.append((n, m))
    return keep


# ------------------------------------------------------------------ normal tissue (frozen) and extent evidence (Eq. extent_evidence)
def normal_tissue(core, claims, anchors, a, sp):
    """N_h = core minus every tumour claim (+2 mm), valid if >= min_ref voxels within win_mm of a; else core minus anchors (+2 mm) (flag
    'ref_fallback'); else None ('no_ref'). 'claims' should exclude outlines flagged as BEING the organ (organ overlap): an organ-sized
    outline is the organ, not a tumour proposal, and excluding it would delete the reference (G0 pilot)."""
    if core is None or not core.any():
        return None, 'no_ref'
    near = dil(a, sp, GP1['win_mm'])
    N = core & ~dil(claims, sp, GP1['margin_mm'])
    if (N & near).sum() >= GP1['min_ref']:
        return N, None
    N = core & ~dil(anchors, sp, GP1['margin_mm'])
    if (N & near).sum() >= GP1['min_ref']:
        return N, 'ref_fallback'
    return None, 'no_ref'


def _kde_logpdf(vals, lo, hi, nb):
    h, e = np.histogram(vals, bins=nb, range=(lo, hi)); h = h.astype(float)
    n = max(len(vals), 1); s = np.std(vals) if len(vals) > 1 else 1.0
    iqr = np.subtract(*np.percentile(vals, [75, 25])) if len(vals) > 3 else s
    bw = 0.9 * max(min(s, iqr / 1.349), 1e-6) * n ** (-0.2)
    sig_bins = max(bw / ((hi - lo) / nb), 0.5)
    h = ndimage.gaussian_filter1d(h, sig_bins, mode='constant'); h = h / max(h.sum(), 1e-12)
    p = (1 - GP1['kde_eps']) * h + GP1['kde_eps'] / nb
    return np.log(p)


def extent_score(Xs, a, xi, N, R):
    """S(xi) = sum_{v in xi \\ a} log f_a(v)/f_h(v) on the smoothed image Xs (un-normalised by nu); also returns the number of voxels.
    f_a: density of the anchor's own intensities; f_h: density of the frozen normal tissue N within the region R."""
    ext = xi & ~a
    if not ext.any() or N is None:
        return 0.0, int(ext.sum())
    va = Xs[a]; vn = Xs[N & R] if (N & R).sum() >= GP1['min_ref'] else Xs[N]
    if len(va) < 10 or len(vn) < 10:
        return 0.0, int(ext.sum())
    allv = np.concatenate([va, vn]); lo, hi = np.percentile(allv, [0.5, 99.5]); hi = hi if hi > lo else lo + 1.0
    nb = GP1['kde_bins']; la = _kde_logpdf(np.clip(va, lo, hi), lo, hi, nb); ln = _kde_logpdf(np.clip(vn, lo, hi), lo, hi, nb)
    b = np.clip(((Xs[ext] - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    return float((la[b] - ln[b]).sum()), int(ext.sum())


def lam_ext(S, nu_mm3, vox_mm3):
    """Lam_ext = exp(S / nu) with nu = nu_mm3 / vox_mm3 voxels in one correlation volume (nu_mm3 fitted on expert data)"""
    c = GP1['ext_clip']; return float(np.clip(np.exp(S * vox_mm3 / max(nu_mm3, 1e-9)), 1 / c, c))


def organ_test(T, Hm, sp, ax, hv=None, vol_ratio=0.6, dice2d=0.6, inside=0.8, contain_H=0.8, contain_frac=0.5):
    """D10 organ overlap of a unit T with organ Hm ('a' is the organ, 'b' contains it); hv = voxel count of the WHOLE organ when Hm is a
    crop (default Hm.sum())."""
    tv = int(T.sum()); hv = int(Hm.sum()) if hv is None else int(hv)
    if tv == 0 or hv == 0:
        return None
    inter = int((T & Hm).sum())
    if inter >= contain_H * hv and inter >= contain_frac * tv:
        return 'b'
    if inter >= inside * tv and tv >= vol_ratio * hv:
        ds = []
        for z in np.nonzero(T.any(axis=tuple(k for k in range(3) if k != ax)))[0]:
            t = np.take(T, z, axis=ax); hh = np.take(Hm, z, axis=ax)
            if hh.any():
                ds.append(2 * (t & hh).sum() / (t.sum() + hh.sum()))
        if ds and np.median(ds) >= dice2d:
            return 'a'
    return None


# ------------------------------------------------------------------ existence tables and evidence (on the anchor)
def organ_class(kb, ent):
    if ent in ('uterus', 'cervix'):
        return 'hollow'
    cat = (kb.entities.get(ent, {}) if kb is not None else {}).get('category')
    return 'hollow' if cat == 'hollow_organ' else 'solid'


def band(v):
    return '<3' if v < 3 else ('3-20' if v <= 20 else '>20')


class Tables1:
    def __init__(self, path):
        self.t = json.load(open(path)) if isinstance(path, str) else path
        vals = [v for d in ('lam', 'lam_int') for lst in self.t[d].values() for v in lst]
        self.max_lam = float(min(max(vals), self.t.get('cap', 100.0)))

    def lam(self, cls, v, mod, tree, appl):
        b = band(v); mod = mod if mod in ('CT', 'MR') else 'CT'
        L = self.t['lam_int'][f'{b}|{mod}'][TREE.index(tree)] if cls is None else self.t['lam'][f'{cls}|{b}|{mod}'][TREE.index(tree)]
        cap = float(self.t.get('cap', 100.0)); L = float(np.clip(L, 1 / cap, cap))
        if b == '<3':
            L = max(L, GP1['small_cap'])
        if cls is not None and (appl is None or appl < GP1['appl_min']) and L > 1:
            L = 1.0
        return L


def existence_evidence(X, sp, si_ax, a, theta, N, tabs, mod, cls):
    """signs S0-S3 of the anchor a against the frozen normal tissue N of host h (territory theta for applicability and closedness),
    on the common ~5 mm grid. All arrays are crops of the same window. N None -> no host frame (intrinsic table if theta is None,
    else signs without context)."""
    sp = np.asarray(sp, float).copy(); k = GR.thick_k(sp[si_ax])
    H = lambda m: (GR.harmonise_array(m.astype(np.int16), si_ax, k, label=True) > 0) if (m is not None and k > 1) else m
    if k > 1:
        X = GR.harmonise_array(X, si_ax, k); a = H(a); theta = H(theta); N = H(N); sp[si_ax] *= k
    out = dict(vol_ml=vol_ml(a, sp))
    if not a.any():
        out.update(tree='uncertain', lam=1.0); return out
    thick = sp.max() / sp.min() >= 1.5; ax = int(np.argmax(sp)) if thick else si_ax
    l2 = GR.level2(X, a, sp, thick, ax); out['rel_tex'] = l2.get('rel_tex'); out['elong'] = l2.get('elong')
    zh = op = appl = None
    if theta is not None and theta.any():
        Tf = fill2d(theta, ax) | theta
        out['appl'] = appl = float((a & (edt(~Tf, sp) <= 2.0)).sum() / max(a.sum(), 1))
        if N is not None and N.any():
            Xs = GR.smooth(X, sp, thick, ax)
            hw = theta & ~a
            Z, D, n = GR.host_frame(Xs, sp, hw, a, N & ~a, min_ref=GP1['min_ref']); out['ref_n'] = n
            if Z is not None:
                Wr = _bbox(a, sp, GP1['close_win_mm'], a.shape)
                f = GR.closed30(X[Wr], Z[Wr], hw[Wr], a[Wr], D[Wr], sp, thick, ax)
                zh = f.get('zh_med')
                if appl >= GP1['appl_min']:
                    op = f.get('open_ratio')
    out.update(zh=zh, open=op)
    out['tree'] = GR.tree_call(dict(open_ratio=op, zh_med=zh, rel_tex=out['rel_tex'], elong=out['elong'], vol_ml=out['vol_ml']))
    use_cls = cls if (theta is not None and theta.any()) else None
    out['lam'] = tabs.lam(use_cls, out['vol_ml'], mod, out['tree'], appl) if tabs is not None else 1.0
    return out


def _bbox(M, sp, pad, shape):
    idx = np.argwhere(M); lo, hi = idx.min(0), idx.max(0) + 1; p = np.ceil(pad / np.asarray(sp)).astype(int)
    return tuple(slice(int(max(0, a - q)), int(min(n, b + q))) for a, b, q, n in zip(lo, hi, p, shape))


# ------------------------------------------------------------------ one-pass conditional inference (Eqs. case_posterior, combination)
def infer(PhL, per_host):
    """PhL: dict h -> P(h | L). per_host: dict h -> dict(pi, lam_ex, w0 [list], lam_ext [list], names [list]).
    Returns dict(p, P_h_given_tumour, ext_post {h: list}, h_hat, xi_hat, p_z1_given_h)."""
    pz1 = {}; post_ext = {}; joint = []
    for h, P in PhL.items():
        d = per_host[h]
        u1 = d['pi'] * d['lam_ex'] * np.asarray(d['w0']) * np.asarray(d['lam_ext'])
        u0 = 1.0 - d['pi']
        Z = u0 + u1.sum()
        pz1[h] = float(u1.sum() / Z) if Z > 0 else 0.0
        post_ext[h] = (u1 / u1.sum()).tolist() if u1.sum() > 0 else [1.0 / len(u1)] * len(u1)
        for j, n in enumerate(d['names']):
            joint.append((P * (u1[j] / Z if Z > 0 else 0.0), h, n))
    p = float(sum(PhL[h] * pz1[h] for h in PhL))
    ph1 = {h: PhL[h] * pz1[h] for h in PhL}; s = sum(ph1.values())
    ph1 = {h: (v / s if s > 0 else PhL[h]) for h, v in ph1.items()}
    best = max(joint, key=lambda t: t[0]) if joint else (0, None, None)
    return dict(p=p, P_h_given_tumour=ph1, ext_post=post_ext, h_hat=best[1], xi_hat=best[2], p_z1_given_h=pz1)


def decide(p):
    if p >= GP1['accept']:
        return 'accepted'
    if p <= GP1['reject']:
        return 'rejected'
    return 'uncertain'
