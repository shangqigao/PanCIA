"""Ledger v4, R2: observation consistency phi_obs(Z, H; X) of the conditional anatomical reasoning model
p(Z, H | X, C, K) ∝ phi_obs · phi_tumor · phi_host (main.tex). Statistical rejection and statistical recall in one MAP rule.

Given H, the potentials factorise over candidate regions ("atoms"), so MAP keeps an atom a iff
        logit(pi_a) + lam_obs * l_a > 0
  pi_a  prior from the tumour priors and logic (phi_BP · phi_VT · phi_logic):  agreed by BP and VT 0.9; one model with the
        other silent there (missing evidence) 0.75 = half the agreement log-odds; one model contradicted by the other 0.5;
        unclaimed tissue touching accepted tumour inside the host envelope (recall) 0.1; logic-infeasible regions are not atoms.
  l_a   observation evidence = -E_obs: the mean over the atom's voxels of log p_T(x) - log p_H(x), clipped to [-clip, clip].
        x = (intensity, local mean, local std) in a 3x3 window of the acquisition plane, each mapped to its empirical CDF over
        the pooled reference voxels of the scan (rank features: invariant to scanner, protocol, CT HU or MR units).
        p_T = Gaussian KDE (Scott) of the accepted tumour: voxels both models agree on (BP n VT) in admitted lesions of the same
              host (the scan's agreed tumour if the host has too few); leave-atom-out for agreed atoms. If the scan has no
              agreed tumour, all admitted tumour claims of the host are used (weak reference) and recall is disabled.
        p_H = Gaussian KDE of host parenchyma: the host reference mask (TS / VT organ), eroded by 1 voxel, minus every tumour
              claim of either model dilated by claim_margin_mm, within window_mm of the lesion (hollow organs: wall and lumen).
              Recovery and recall need an accepted (two-model) tumour reference; recovered lesions must not lie mostly inside
              another organ (logic). Hollow organs are included:
              their reference mixes wall and lumen content, and a tumour is expected to differ from both (pan-cancer, no
              organ exceptions).
  The mean (not the sum) of voxel log-ratios is used because neighbouring voxels are strongly correlated: an atom counts
  as one observation, so its size cannot override the prior. All constants are fixed a priori (no tuning, no stage).
Atoms per lesion (in its padded box): agreed pieces (pi 0.9); single-model pieces BP\\VT and VT\\BP, the BP-only part split
once by the sign of the voxel log-ratio (trimming BP excess) (pi 0.5); recall pieces (pi 0.1). Rejected lesions whose only
failure was the lack of a second model are re-scored as single-model atoms (recovery)."""
from __future__ import annotations
import numpy as np
from scipy import ndimage, stats

R2P = dict(r2=True, r2_recall=True, lam_obs=1.0, pi_two=0.9, pi_silent=0.75, pi_single=0.5, pi_recall=0.1, r2_reach_mm=10.0, r2_window_mm=20.0,
           r2_claim_margin_mm=2.0, r2_min_ref=200, r2_min_atom_ml=0.5, r2_nmax=5000, r2_neval=3000, r2_clip=5.0, r2_seed=0, img_root=None)
RECOVERABLE = {'primary_extra_single_no_vt', 'primary_extra_unconfirmed', 'local_single_model', 'distant_needs_bp_and_site_prompt'}
logit = lambda q: float(np.log(q / (1 - q)))


def _pad(sl, mm, sp, shape):
    pad = [int(np.ceil(mm / float(q))) + 1 for q in sp]
    return tuple(slice(max(0, a.start - p), min(n, a.stop + p)) for a, p, n in zip(sl, pad, shape))


def _put(L, sl, ps):
    out = np.zeros(tuple(s.stop - s.start for s in ps), bool)
    out[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(sl, ps))] = L
    return out


def _features(img, ps, ax):
    X = np.asarray(img.dataobj[ps], dtype=np.float32)
    size = [3, 3, 3]; size[ax] = 1
    m = ndimage.uniform_filter(X, size=size)
    s = np.sqrt(np.maximum(ndimage.uniform_filter(X * X, size=size) - m * m, 0))
    return np.stack([X, m, s], -1)


class _Ref:
    """Empirical-CDF rank mapping fitted on pooled reference voxels, and the two KDEs."""

    def __init__(self, F, T, Hm, p, rng):
        def samp(M):
            v = F[M]
            return v[rng.choice(len(v), p['r2_nmax'], replace=False)] if len(v) > p['r2_nmax'] else v
        self.t, self.h = samp(T), samp(Hm)
        pool = np.concatenate([self.t, self.h])
        self.q = [np.sort(pool[:, j]) for j in range(pool.shape[1])]
        jit = lambda a: a + rng.normal(0, 1e-3, a.shape)
        self.kt = stats.gaussian_kde(jit(self.rank(self.t)).T)
        self.kh = stats.gaussian_kde(jit(self.rank(self.h)).T)

        # the log-ratio is tabulated once on a G^3 grid of the rank cube (rank features lie in [0, 1]); voxels are looked up
        # by bin, so scoring millions of voxels costs one KDE evaluation of G^3 points
        G = 24
        c = (np.arange(G) + 0.5) / G
        grid = np.stack(np.meshgrid(c, c, c, indexing='ij'), 0).reshape(3, -1)
        ev = lambda k: np.concatenate([k.logpdf(grid[:, i:i + 500]) for i in range(0, grid.shape[1], 500)])   # chunked: memory
        self.G, self.tab = G, (ev(self.kt) - ev(self.kh)).reshape(G, G, G)

    def rank(self, v):
        return np.stack([np.searchsorted(q, v[:, j], side='right') / len(q) for j, q in enumerate(self.q)], -1)

    def llr(self, v):
        b = np.clip((self.rank(v) * self.G).astype(int), 0, self.G - 1)
        return self.tab[b[:, 0], b[:, 1], b[:, 2]]


def apply_r2(out, hosts, ctx, p):
    """Score and update lesions in place (masks, sizes, status). ctx: shape, sp, vox_ml, img, ax, bp_u, vt_u, other."""
    shape, sp, vox_ml, img, ax = ctx['shape'], ctx['sp'], ctx['vox_ml'], ctx['img'], ctx['ax']
    bp_u, vt_u, other = ctx['bp_u'], ctx['vt_u'], ctx['other']
    claims = bp_u | vt_u
    min_atom = max(1, int(round(p['r2_min_atom_ml'] / vox_ml)))
    rng = np.random.default_rng(p['r2_seed'])
    st = np.ones((3, 3, 3), bool)
    lam = p['lam_obs']; lp = dict(two=logit(p['pi_two']), silent=logit(p['pi_silent']), single=logit(p['pi_single']), recall=logit(p['pi_recall']))
    # accepted tumour per host and scan-wide: agreed voxels (BP n VT) of admitted lesions
    agreed = {}
    for les in out:
        if les['status'] == 'admitted' and les['host'] is not None:
            a = agreed.setdefault(les['host'], [])
            a.append((les['sl'], les['bp'] & les['vt']))
    report = dict(n_scored=0, n_rejected=0, n_recovered=0, trimmed_ml=0.0, recalled_ml=0.0, ref=None)
    for les in out:
        if les['host'] is None or les['cls'] is None:
            continue
        # recovery candidates must satisfy the anatomical logic: not mostly inside another organ (phi_logic exclusion)
        cand = les['status'] == 'admitted' or (les['reason'] in RECOVERABLE and les['in_other'] < p['attach_other_max'])
        if not cand:
            continue
        h = hosts[les['host']]
        if not h['H'].any() or h['ref'] == 'kb_prior_region':
            les['r2'] = dict(testable=False, why='no_organ_reference'); continue
        ps = _pad(les['sl'], p['r2_reach_mm'] + p['r2_window_mm'], sp, shape)
        L = _put(les['L'], les['sl'], ps)
        Hm = ndimage.binary_erosion(h['H'][ps], structure=st)
        cl = claims[ps]
        excl = ndimage.distance_transform_edt(~cl, sampling=sp) <= p['r2_claim_margin_mm']
        win = ndimage.distance_transform_edt(~L, sampling=sp) <= p['r2_window_mm']
        # host reference: the host's own parenchyma near the candidate (for hollow organs wall and lumen content together)
        Pm = Hm & ~excl & win
        # tumour reference: agreed voxels of admitted lesions in the host, else in the scan, else all claims of the host
        def agreed_mask(keys):
            M = np.zeros(L.shape, bool)
            for q in keys:
                for sl, A in agreed.get(q, []):
                    lo = [max(a.start, b.start) for a, b in zip(sl, ps)]; hi = [min(a.stop, b.stop) for a, b in zip(sl, ps)]
                    if all(l < u for l, u in zip(lo, hi)):
                        M[tuple(slice(l - b.start, u - b.start) for l, u, b in zip(lo, hi, ps))] |= \
                            A[tuple(slice(l - a.start, u - a.start) for l, u, a in zip(lo, hi, sl))]
            return ndimage.binary_erosion(M, structure=st)
        Tm, ref = agreed_mask([les['host']]), 'agreed_host'
        if int(Tm.sum()) < p['r2_min_ref']:
            Tm, ref = agreed_mask(list(agreed)), 'agreed_scan'
        weak = ndimage.binary_erosion(cl & Hm, structure=st) | ndimage.binary_erosion(L & (_put(les['bp'], les['sl'], ps) | _put(les['vt'], les['sl'], ps)), structure=st)
        if int(Tm.sum()) < p['r2_min_ref']:
            Tm, ref = weak, 'claims_weak'
        if int(Tm.sum()) < p['r2_min_ref'] or int(Pm.sum()) < p['r2_min_ref']:
            les['r2'] = dict(testable=False, why='small_reference', ref=ref); continue
        if les['status'] != 'admitted' and ref == 'claims_weak':
            # recovery is 'consistent with the accepted tumour': it needs an accepted (two-model) tumour in the scan
            les['r2'] = dict(testable=False, why='no_accepted_tumour', ref=ref); continue
        F = _features(img, ps, ax)
        try:
            Rf = _Ref(F, Tm, Pm, p, rng)
        except Exception as e:                                  # degenerate densities (constant image)
            les['r2'] = dict(testable=False, why=f'kde:{type(e).__name__}', ref=ref); continue

        def loo(M):
            # leave-atom-out tumour reference (an atom must not be judged against itself)
            for T_ in (Tm & ~M, weak & ~M):
                if int(T_.sum()) >= p['r2_min_ref']:
                    try:
                        return _Ref(F, T_, Pm, p, rng)
                    except Exception:
                        return None
            return None

        def score(M, kind, R):
            v = F[M]
            if len(v) > p['r2_neval']:
                v = v[rng.choice(len(v), p['r2_neval'], replace=False)]
            l = float(np.clip(np.mean(R.llr(v)), -p['r2_clip'], p['r2_clip']))
            return l, lp[kind] + lam * l
        bp = _put(les['bp'], les['sl'], ps); vt = _put(les['vt'], les['sl'], ps)
        atoms = []                                              # (mask, kind, origin)
        small = np.zeros(L.shape, bool)                         # pieces below the minimum atom size: untested, unchanged

        def comps(M):
            lab, n = ndimage.label(M, structure=st)
            if not n:
                return []
            sz = np.bincount(lab.ravel())
            big = [j for j in range(1, n + 1) if sz[j] >= min_atom]
            small[(lab > 0) & ~np.isin(lab, big)] = True
            return [lab == j for j in big]
        atoms += [(M, 'two', 'agreed') for M in comps(L & bp & vt)]
        # single-model parts: the other model either abstains (silent: no claim there -> missing evidence, pi_silent) or
        # contradicts (active there but not on these voxels -> pi_single). BP is 2D: silent on acquisition slices where it
        # claims nothing inside the host envelope. VT is 3D: silent when it claims nothing within reach of the part.
        env = h['env'][ps]
        oth = [a for a in range(3) if a != ax]
        bp_act = (bp_u[ps] & env).any(axis=tuple(oth))
        bshape = [1, 1, 1]; bshape[ax] = -1
        bp_act = np.broadcast_to(bp_act.reshape(bshape), L.shape)
        vo = L & vt & ~bp
        for part, kind in ((vo & ~bp_act, 'silent'), (vo & bp_act, 'single')):
            atoms += [(M, kind, 'vt_only') for M in comps(part)]
        bo = L & bp & ~vt
        if bo.any():                                            # split BP excess once by the sign of the voxel log-ratio
            vl = np.zeros(L.shape, np.float32); vl[bo] = Rf.llr(F[bo])
            vt_near = ndimage.distance_transform_edt(~vt_u[ps], sampling=sp) <= p['r2_reach_mm'] if vt_u[ps].any() else np.zeros(L.shape, bool)
            for sgn in (vl > 0, vl <= 0):
                for M in comps(bo & sgn):
                    atoms.append((M, 'single' if (M & vt_near).any() else 'silent', 'bp_only'))
        rest = L & ~bp & ~vt                                    # R3 / grouping glue: follows the lesion
        keep = rest | small; rec = []
        for M, kind, org in atoms:
            nv = int(M.sum())
            if nv < min_atom:
                keep |= M; continue                             # too small to test: unchanged
            R_ = loo(M) if (Tm & M).any() else Rf      # an atom is never judged against a reference that contains it
            if R_ is None:
                keep |= M; rec.append(dict(kind=kind, origin=org, ml=round(nv * vox_ml, 2), l=None, score=None, keep=True, why='no_loo_reference')); continue
            l, s = score(M, kind, R_)
            rec.append(dict(kind=kind, origin=org, ml=round(nv * vox_ml, 2), l=round(l, 3), score=round(s, 3), keep=s > 0))
            if s > 0:
                keep |= M
        recalled = 0
        if p['r2_recall'] and ref != 'claims_weak' and les['status'] == 'admitted' and keep.any():
            reach = ndimage.distance_transform_edt(~keep, sampling=sp) <= p['r2_reach_mm']
            allowed = h['env'][ps] & ~cl & ~(other[ps] & ~h['H'][ps]) & reach
            if allowed.any():
                vl = np.zeros(L.shape, np.float32); vl[allowed] = Rf.llr(F[allowed])
                touch = ndimage.binary_dilation(keep, structure=st)
                lab, n = ndimage.label(allowed & (vl > 0), structure=st)
                sz = np.bincount(lab.ravel()) if n else []
                for j in range(1, n + 1):
                    if sz[j] < min_atom:
                        continue
                    M = lab == j
                    if not (M & touch).any():
                        continue
                    l, s = score(M, 'recall', Rf)
                    rec.append(dict(kind='recall', origin='unclaimed', ml=round(int(M.sum()) * vox_ml, 2), l=round(l, 3), score=round(s, 3), keep=s > 0))
                    if s > 0:
                        keep |= M; recalled += int(M.sum())
        core_left = keep & (bp | vt)
        trimmed = int((L & ~keep).sum())
        old_status = les['status']
        if not core_left.any() or int(core_left.sum()) < min_atom:
            les['status'], les['reason'] = 'rejected', 'r2_parenchyma_like'
            if old_status == 'admitted':
                report['n_rejected'] += 1
        elif old_status != 'admitted':
            les['status'], les['reason'] = 'admitted', 'r2_image_consistent'
            report['n_recovered'] += 1
        if les['status'] == 'admitted':
            bb = ndimage.find_objects(keep.astype(np.int8))[0]
            nsl = tuple(slice(b.start + a.start, b.start + a.stop) for a, b in zip(bb, ps))
            les['L'] = keep[bb]; les['bp'] = bp[bb] & les['L']; les['vt'] = vt[bb] & les['L']; les['sl'] = nsl
            les['vox'] = int(les['L'].sum()); les['agree'] = int((les['bp'] & les['vt']).sum())
            ext = [(nsl[a].stop - nsl[a].start) * sp[a] for a in range(3) if a != ctx['k']]
            les['ld_mm'] = float(np.hypot(*ext))
            nz = int(les['L'].any(axis=tuple(a for a in range(3) if a != ctx['k'])).sum())
            les['n_slices'], les['z_mm'] = nz, round(nz * float(sp[ctx['k']]), 1)
            les['frac'] = {q: int((les['L'] & hh['env'][nsl]).sum()) / les['vox'] for q, hh in hosts.items()}
        lf, _ = score(keep if keep.any() else L, 'single', Rf)
        les['r2'] = dict(testable=True, ref=ref, l_lesion=round(lf, 3), atoms=rec, recalled_ml=round(recalled * vox_ml, 2),
                         trimmed_ml=round(trimmed * vox_ml, 2), before=old_status)
        report['n_scored'] += 1; report['trimmed_ml'] += trimmed * vox_ml; report['recalled_ml'] += recalled * vox_ml
    report['trimmed_ml'] = round(report['trimmed_ml'], 2); report['recalled_ml'] = round(report['recalled_ml'], 2)
    return report
