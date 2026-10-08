"""Ledger R2 as calibrated EM: posterior tumour inference from the image under the fixed priors of host and logic reasoning
(main.tex, Section 'Reasoning details', Eqs. part_posterior, tumor_prior, obs_def, bf, decision).

Stage 1 (the ledger, unchanged) fixes the candidates and their priors. Stage 2 (this module) infers the tumour posterior.
  candidates  the source components of the ledger: connected components of BP, of initial VT and of every VT tumour prompt
              (the ledger's claims). Never merged or split for inference.
  prior pi_c  0 if the candidate violates the anatomical logic (a dropped claim, or a lesion rejected for a logical reason);
              otherwise from the other model's state on c: proposes (covers >= 1/2 of c) 0.9, silent 0.75, active 0.5.
              Recall regions 0.25. Fixed throughout.
  normal model  label-conditional: every voxel has an anatomical label (hole-filled TotalSegmentator structure, else a host
              region H_hat, else unlabelled tissue); p0(f | label) from normal voxels of that label within window_mm of the
              candidates, all candidates dilated by margin_mm removed, eroded in-plane.
  tumour model  p1(f) from candidate voxels weighted by w_v = (max_{c contains v} q_c) x r_v, where r_v is the voxel
              responsibility of the tumour class in the mixture rho p1 + (1 - rho) p0(.|label) over candidate voxels (inner EM),
              so voxels the normal model explains do not contaminate the tumour model. No leave-one-out: with one dominant tumour, leaving it out leaves only its over-segmentation and
              small false positives. Instead, two-fold spatial cross-fitting: densities from one half of the box, the
              candidate's voxels and all pseudo-candidates from the other half, so no voxel is scored by a density fitted on
              itself. A candidate of normal tissue then gives lambda ~ 0 for itself and for normal blocks alike: BF ~ 1.
  features    intensity, 3x3 in-plane mean and SD, each mapped to its rank within the scan; densities are binned on a G^3 grid
              of the rank cube and Gaussian-smoothed (Scott bandwidth) - a kernel density estimate on a grid.
  evidence    s_c = mean over c of lambda(v) = log p1 - log p0(.|label(v)), clipped to +-5. g0 / g1 = distributions of s for M
              same-volume compact pseudo-candidates from normal tissue with c's label composition / from current tumour
              voxels; two-fold spatial cross-fitting (densities from one half, pseudo-candidates from the other).
              BF_c = g1(s_c) / g0(s_c), log clipped to +-10; BF = 1 when the test has no power (too few reference voxels).
  EM          (damped: posterior log-odds updated as eta <- (1 - d_t) eta + d_t (logit pi + log BF), d_t = 0.5 for the
              first 5 iterations, then 1/(t - 3): a running average, so oscillating tumour models are averaged out)
              q^(0) = pi; M-step: tumour model from weights; E-step: q = sigmoid(logit pi + log BF) for every candidate, then
              recall regions regenerated from the current tumour and scored the same way (selection-aware null). Stop when
              the accepted set is unchanged and max |dq| < tol, or after max_iter. MAP: z = 1 iff q >= 0.5.
  output      ledger lesions keep their identity; mask = lesion n accepted candidates, plus accepted recall regions touching it.
All constants are fixed a priori and shared by every cancer type."""
from __future__ import annotations
import numpy as np
from scipy import ndimage, stats
from scipy.spatial import cKDTree

EMP = dict(r2_mode='em', em_G=24, em_window_mm=20.0, em_margin_mm=2.0, em_reach_mm=10.0, em_min_ref=200, em_min_ml=0.5,
           em_M=200, em_kmax=500, em_lclip=5.0, em_bfclip=10.0, em_max_iter=30, em_damp=0.5, em_damp_iters=5, em_inner=5, em_tol=0.01, em_eps=1e-3,
           em_pi_two=0.9, em_pi_silent=0.75, em_pi_single=0.5, em_pi_recall=0.25, em_seed=0, em_max_box_vox=45e6,
           em_compare='label', em_xfit='blocks', em_block_mm=15.0,
           # candidate-specific model (8 Oct, em_model='candidate'): own tumour model per candidate (within-candidate EM, shrunk
           # to the leave-c-out pool), local reference from same-tissue labels (KB part-of / contralateral siblings, depleted
           # labels last), candidate-local cross-fitting, null calibration from normal patches put through the same fit
           em_model='shared', em_ref_max_mm=60.0, em_ref_step_mm=10.0, em_dep=0.10, em_min_own=50, em_kappa=500.0,
           em_rho_ab=2.0, em_cap=4000, em_inner_c=10, em_calib='sellke', em_min_null=20,
           em_M_null=400, em_zscale=10.0, em_tau=0.67, em_calib_c='halfnormal', em_calib_recall='mix',
           em_recover_pi=0.5, em_max_inplane_mm=3.0, em_trim_min_ml=0.5, em_trim_min_frac=0.10,
           em_pi_inside=0.75)
# 9 Oct (3:1 rule): the claim boundary is the voxel prior - inside an accepted claim 0.75, outside 0.25 (em_pi_recall) -
# so a voxel is trimmed only if normal tissue is >= 3x more likely (lambda <= -log 3) and grown only if tumour is
# >= 3x more likely (lambda >= log 3); rho_c is descriptive only
# 9 Oct: em_recover_pi caps the prior of a lesion the ledger rejected for lack of support (recovery needs image evidence);
# em_max_inplane_mm: scans coarser in-plane are scouts / localisers (non-diagnostic): R2 admits nothing on them;
# em_trim_*: an accepted candidate loses a dark sub-region (voxel posterior < 1/2) only if it reaches the candidate's
# border and holds >= max(trim_min_ml, trim_min_frac x candidate); holes enclosed in 3-D or in-plane (necrosis, cavity)
# are kept; remaining pieces < trim_min_ml are dropped as speckle
# em_tau (nats/voxel): contrast scale of the tumour hypothesis, fixed a priori by simulation (scripts/r2_tau_sim.py,
# 8 Oct 2026: median delta of a lesion 1 normal-tissue SD brighter than its tissue, 1-80 ml, = 0.672);
# see _candidate_r2   # em_xfit 'blocks': 3-D checkerboard of em_block_mm blocks   # 'label' (default, restored 7 Oct after QC): each voxel vs normal tissue of its own label
                                 # (TS hole-filled, else VT/TS host region, else unlabelled); 'surround': experimental, not used
RECOVERABLE = {'primary_extra_single_no_vt', 'primary_extra_unconfirmed', 'local_single_model', 'distant_needs_bp_and_site_prompt'}
logit = lambda q: float(np.log(q / (1 - q)))
sig = lambda x: float(1 / (1 + np.exp(-x)))
ST = np.ones((3, 3, 3), bool)


def _within(M, mm, sp):
    return ndimage.distance_transform_edt(~M, sampling=sp) <= mm if M.any() else np.zeros(M.shape, bool)


def _put(B, sl, crop, shape_B):
    """Crop (global slice sl) into box B; returns a bool array of B's shape."""
    out = np.zeros(shape_B, bool)
    lo = [max(a.start, b.start) for a, b in zip(sl, B)]
    hi = [min(a.stop, b.stop) for a, b in zip(sl, B)]
    if any(l >= h for l, h in zip(lo, hi)):
        return out
    out[tuple(slice(l - b.start, h - b.start) for l, h, b in zip(lo, hi, B))] = \
        crop[tuple(slice(l - a.start, h - a.start) for l, h, a in zip(lo, hi, sl))]
    return out


def fill_labels(md, ax):
    """Label map with every TotalSegmentator structure filled slice-wise (acquisition plane) for enclosed holes; filled
    voxels take the structure's label only where no other structure is present."""
    out = md.astype(np.int32).copy()
    st = np.zeros((3, 3, 3), bool)
    for d in range(3):
        if d != ax:
            for o in (0, 2):
                j = [1, 1, 1]; j[d] = o; st[tuple(j)] = True
    st[1, 1, 1] = True
    for lab, sl in enumerate(ndimage.find_objects(md), start=1):
        if sl is None:
            continue
        m = md[sl] == lab
        pw = [(0, 0)] * 3; pw[ax] = (1, 1)
        f = ndimage.binary_fill_holes(np.pad(m, pw), structure=st)
        cut = [slice(None)] * 3; cut[ax] = slice(1, -1)
        f = f[tuple(cut)] & (out[sl] == 0)
        out[sl][f] = lab
    return out


class _Pool:
    """Reference voxels (flat indices into the box) with physical coordinates; thinned trees for compact blocks."""

    def __init__(self, idx, xyz, rng, ratios=(1, 4, 16, 64, 256, 1024)):
        self.idx, self.xyz = idx, xyz
        self.trees = {}
        for r in ratios:
            n = len(idx) // r
            if n < 2 and r > 1:
                break
            sub = np.arange(len(idx)) if r == 1 else rng.choice(len(idx), n, replace=False)
            self.trees[r] = (sub, cKDTree(xyz[sub]))

    def __len__(self):
        return len(self.idx)

    def blocks(self, n_vox, M, rng, kmax, seeds_near=None, seed_p=None):
        """M compact blocks of n_vox voxels: the k nearest points to a random seed on a thinned pool (k <= kmax).
        Returns flat-index arrays (thinned samples representing blocks of n_vox voxels)."""
        if len(self) < 2:
            return []
        n_vox = max(1, min(int(n_vox), len(self) // 2))
        r = 1
        for rr in sorted(self.trees):
            r = rr
            if n_vox / rr <= kmax:
                break
        sub, tree = self.trees[r]
        k = max(1, min(int(round(n_vox / r)), len(sub) // 2))
        if seed_p is not None:
            seeds = self.xyz[rng.choice(len(self.idx), M, p=seed_p)]
        elif seeds_near is not None and len(seeds_near):
            seeds = self.xyz[rng.choice(seeds_near, M)]
        else:
            seeds = self.xyz[sub[rng.integers(len(sub), size=M)]]
        _, ii = tree.query(seeds, k=k)
        ii = np.asarray(ii).reshape(M, -1)
        return [self.idx[sub[row]] for row in ii]

    def near(self, centre, radius):
        sub, tree = self.trees[1]
        return sub[tree.query_ball_point(centre, radius)]


def _density(bins, G, w=None, eps=1e-3):
    """Binned Gaussian KDE on the G^3 rank grid (Scott bandwidth); returns log density on the grid."""
    h = np.bincount(bins, weights=w, minlength=G ** 3).astype(np.float64)
    tot = h.sum()
    if tot <= 0:
        return None
    n_eff = tot ** 2 / (np.sum((w if w is not None else np.ones(len(bins))) ** 2) + 1e-12)
    sd_bins = G / np.sqrt(12.0)                        # rank features are ~uniform on [0, 1]
    sigma = max(0.5, (max(n_eff, 2) ** (-1.0 / 7.0)) * sd_bins)
    d = ndimage.gaussian_filter(h.reshape(G, G, G), sigma, mode='reflect')
    d = d / d.sum()
    return np.log((1 - eps) * d + eps / G ** 3).ravel()


def _accepted(c, out):
    """MAP label: q > 1/2; an exact tie (pi = 1/2 and no image evidence) keeps the stage-1 decision of its lesion.
    Candidate mode (c['strict'], 9 Oct): admission needs a TESTED candidate with non-negative image evidence -
    powered, log BF >= 0 and q > 1/2. The prior can break a tie in favour of the image, never override it, and an
    untested candidate (too small for folds / no reference / no null) is never admitted on its prior alone: it is
    reported as 'no power' for expert review."""
    if c.get('strict'):
        return bool(c.get('powered')) and c['logbf'] >= 0 and c['q'] > 0.5
    if abs(c['q'] - 0.5) < 1e-9:
        return 'lesion' in c and out[c['lesion']]['status'] == 'admitted'
    return c['q'] > 0.5


def _kde1(v):
    v = np.asarray(v, float)
    if len(v) < 5:
        return None
    if np.std(v) < 1e-6:
        v = v + np.random.default_rng(0).normal(0, 1e-3, v.shape)
    return stats.gaussian_kde(v)


def apply_r2_em(out, hosts, ctx, p):
    shape, sp, vox_ml, img, ax = ctx['shape'], np.asarray(ctx['sp'], float), ctx['vox_ml'], ctx['img'], ctx['ax']
    md, other, claims = ctx['ts_map'], ctx['other'], ctx['claims']
    bp_u, vt_u = ctx['bp_u'], ctx['vt_u']
    G = p['em_G']; rng = np.random.default_rng(p['em_seed'])
    min_vox = max(1, int(round(p['em_min_ml'] / vox_ml)))
    lp = dict(two=logit(p['em_pi_two']), silent=logit(p['em_pi_silent']), single=logit(p['em_pi_single']),
              recall=logit(p['em_pi_recall']))
    report = dict(n_candidates=len(claims), n_feasible=0, n_accepted=0, n_rejected=0, n_recovered_lesions=0,
                  n_rejected_lesions=0, recalled_ml=0.0, iterations=0, converged=False)

    if p.get('em_model', 'shared') == 'candidate':
        inplane = float(np.sort(sp)[:2].max())
        if inplane > p['em_max_inplane_mm']:           # scout / localiser: the image cannot support any lesion
            for les in out:
                before = les['status']
                if before == 'admitted':
                    les['status'], les['reason'] = 'rejected', 'non_diagnostic_scan'; report['n_rejected_lesions'] += 1
                les['r2'] = dict(before=before, candidates=[], recall=[], note=f'non-diagnostic scan (in-plane {inplane:.1f} mm)')
            report['note'] = f'non_diagnostic_scan: in-plane spacing {inplane:.2f} mm > {p["em_max_inplane_mm"]} mm'
            return report

    # ---- lesion map and per-claim prior (stage 1 output, fixed) ---------------------------------------------------------
    lesmap = np.zeros(shape, np.int16)
    for i, les in enumerate(out):
        lesmap[les['sl']][les['L']] = i + 1

    def feasible_lesion(les):
        if les['host'] is None or les['cls'] is None:
            return False, 'no_host'
        if les['status'] == 'admitted':
            return True, None
        if les['reason'] in RECOVERABLE:              # support-based rejection: soft prior (7 Oct: no extra logic check here)
            return True, None
        return False, les['reason']

    cands = []
    for ci, c in enumerate(claims):
        rec = dict(ci=ci, rater=c['rater'], source=c['source'], ml=round(c['vox'] * vox_ml, 2), sl=c['sl'], crop=c['crop'])
        if c.get('drop'):
            rec.update(pi=0.0, why=c['drop']); cands.append(rec); continue
        lm = lesmap[c['sl']][c['crop']]
        lm = lm[lm > 0]
        if not len(lm):
            rec.update(pi=0.0, why='no_lesion'); cands.append(rec); continue
        li = int(np.bincount(lm).argmax()) - 1
        ok, why = feasible_lesion(out[li])
        rec['lesion'] = li
        if not ok:
            rec.update(pi=0.0, why=why); cands.append(rec); continue
        # other model's state on c
        oth = vt_u if c['rater'] == 'BP' else bp_u
        cov = int((oth[c['sl']] & c['crop']).sum()) / c['vox']
        if cov >= 0.5:
            state = 'two'
        elif c['rater'] == 'BP':                        # VT is 3D: silent when nothing within reach
            ps = tuple(slice(max(0, s.start - int(np.ceil(p['em_reach_mm'] / q)) - 1), min(n, s.stop + int(np.ceil(p['em_reach_mm'] / q)) + 1))
                       for s, q, n in zip(c['sl'], sp, shape))
            loc = _put(ps, c['sl'], c['crop'], tuple(s.stop - s.start for s in ps))
            state = 'single' if (vt_u[ps] & _within(loc, p['em_reach_mm'], sp)).any() else 'silent'
        else:                                           # BP is 2D: silent on c's acquisition slices within the host envelope
            h = hosts[out[li]['host']]
            zs = np.where(c['crop'].any(axis=tuple(a for a in range(3) if a != ax)))[0] + c['sl'][ax].start
            sl2 = list(h['env'].sl) if h['env'].any() else [slice(0, n) for n in shape]
            act = False
            for z in zs:
                ix = list(sl2); ix[ax] = slice(int(z), int(z) + 1); ix = tuple(ix)
                if (bp_u[ix] & h['env'][ix]).any():
                    act = True; break
            state = 'single' if act else 'silent'
        rec.update(pi={'two': p['em_pi_two'], 'silent': p['em_pi_silent'], 'single': p['em_pi_single']}[state], state=state)
        if p.get('em_model', 'shared') == 'candidate' and out[li]['status'] != 'admitted' and rec['pi'] > p['em_recover_pi']:
            rec['pi'] = p['em_recover_pi']; rec['pi_capped'] = True   # a ledger rejection is recovered only by image evidence
        cands.append(rec)
    feas = [c for c in cands if c['pi'] > 0]
    report['n_feasible'] = len(feas)
    if not feas:
        report['note'] = 'no feasible candidate'
        return report

    # ---- working box: feasible candidates + window + reach --------------------------------------------------------------
    U = np.zeros(shape, bool)
    for c in feas:
        U[c['sl']] |= c['crop']
    bb = ndimage.find_objects(U.astype(np.int8))[0]
    padmm = p['em_window_mm'] + p['em_reach_mm']
    if p.get('em_model', 'shared') == 'candidate':
        padmm = max(padmm, p['em_ref_max_mm'] + p['em_reach_mm'])
    pad = [int(np.ceil(padmm / q)) + 1 for q in sp]
    B = tuple(slice(max(0, b.start - q), min(n, b.stop + q)) for b, q, n in zip(bb, pad, shape))
    shB = tuple(s.stop - s.start for s in B)
    if np.prod(shB) > p['em_max_box_vox']:
        report['note'] = f'box too large {shB}'
        return report
    org = np.array([s.start for s in B], float)
    # features -> rank bins (flat index on the G^3 grid); features themselves are not kept
    X = np.asarray(img.dataobj[B], dtype=np.float32)
    size = [3, 3, 3]; size[ax] = 1
    m = ndimage.uniform_filter(X, size=size)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(X * X, size=size) - m * m, 0))
    samp = rng.choice(X.size, min(X.size, 200000), replace=False)
    BIN = np.zeros(shB, np.int32)
    for j, A in enumerate((X, m, sd)):
        edges = np.quantile(A.ravel()[samp], np.arange(1, G) / G)
        BIN = BIN * G + np.searchsorted(edges, A, side='right').astype(np.int32)
    del X, m, sd
    BIN = BIN.ravel()
    # labels: hole-filled TS structures, else host region, else unlabelled (0)
    LAB = fill_labels(md[B], ax)
    hostlab = {}
    for k_, (key, h) in enumerate(hosts.items()):
        if h['H'].any():
            Hb = h['H'][B]
            LAB[(LAB == 0) & Hb] = 1000 + k_
            hostlab[key] = 1000 + k_
    LAB = LAB.ravel()
    # candidate voxels (flat) in the box
    for c in feas:
        Mc = _put(B, c['sl'], c['crop'], shB)
        c['idx'] = np.flatnonzero(Mc)
        c['xyz'] = (np.argwhere(Mc) + org) * sp
        c['centre'] = c['xyz'].mean(0); c['radius'] = float(np.linalg.norm(c['xyz'] - c['centre'], axis=1).max())
    U_B = U[B]; del U
    claims_all = np.zeros(shB, bool)
    for c in cands:
        claims_all |= _put(B, c['sl'], c['crop'], shB)
    if p.get('em_model', 'shared') == 'candidate':
        recall = _candidate_r2(feas, out, hosts, ctx, p, B, shB, org, BIN, LAB, U_B, claims_all, hostlab, report, min_vox)
        return _finish(out, feas, cands, recall, report, B, shB, hosts, ctx, sp, vox_ml)
    # normal pools per label (window around the candidates, claims dilated removed, in-plane erosion)
    normal = ~_within(claims_all, p['em_margin_mm'], sp) & _within(U_B, p['em_window_mm'], sp)
    ER = np.zeros((3, 3, 3), bool); e = [slice(None)] * 3; e[ax] = 1; ER[tuple(e)] = True
    LABg = LAB.reshape(shB)
    normal_lab = {}
    small_lab = {}
    for l in np.unique(LABg[normal]):
        Ml = ndimage.binary_erosion(normal & (LABg == l), structure=ER)
        if Ml.sum() >= p['em_min_ref']:
            normal_lab[int(l)] = np.flatnonzero(Ml)
        elif Ml.any():
            small_lab[int(l)] = np.flatnonzero(Ml)
    del normal
    # KB part-of grouping (7 Oct): a TS label with too few normal voxels of its own (vertebra T6, a lung lobe, gluteus ...)
    # uses the pooled normal tissue of its top part-of ancestor in the KB (spine, that side's lung, skeletal muscle), following
    # only 'inside' / 'group' has_part links (same tissue; not 'contiguous' such as spine -> spinal cord). Labels that are
    # testable on their own keep their own reference; labels without such an ancestor stay untestable.
    group_names = {}
    kb_, task_ = ctx.get('kb'), ctx.get('ts_task')
    if kb_ is not None and p.get('em_kb_group', True):
        tsn = ctx.get('ts_names') or {}
        tmap = kb_.ts_to_entity.get(task_, {})
        def top_of(e):
            seen = set()
            while True:
                ps = [r['src'] for r in kb_.in_edges.get(e, []) if r['type'] == 'has_part' and r.get('spatial', 'inside') in ('inside', 'group')]
                if not ps or ps[0] in seen:
                    return e
                seen.add(e); e = ps[0]
        gkey = {}
        for l in set(int(x) for x in np.unique(LABg)) - {0}:
            if l >= 1000 or l not in tsn or tsn[l] not in tmap:
                continue
            ent, side = tmap[tsn[l]]
            top = top_of(ent)
            if top == ent:
                continue
            side = side or ('left' if '_left' in ent else 'right' if '_right' in ent else None)
            gkey[l] = (top, side)
        groups = {}
        for l, g in gkey.items():
            groups.setdefault(g, []).append(l)
        # testable = >= min_ref normal voxels in BOTH spatial halves used for cross-fitting (same split as below)
        j_ = int(np.argmax(np.array(shB) * sp)); cut_ = shB[j_] // 2
        bs_ = np.maximum(1, np.round(p['em_block_mm'] / sp).astype(int))
        def testable(idx):
            if idx is None or len(idx) < 2 * p['em_min_ref']:
                return False
            if p.get('em_xfit', 'halves') == 'blocks':
                h_ = (np.add.reduce([u // b for u, b in zip(np.unravel_index(idx, shB), bs_)]) % 2).astype(bool)
            else:
                h_ = np.unravel_index(idx, shB)[j_] >= cut_
            return int(h_.sum()) >= p['em_min_ref'] and int((~h_).sum()) >= p['em_min_ref']
        remap = {}
        for gi, (g, members) in enumerate(sorted(groups.items(), key=lambda t: str(t[0]))):
            need = [l for l in members if not testable(normal_lab.get(l))]   # labels without a usable reference of their own
            if not need:
                continue
            pool = np.unique(np.concatenate([normal_lab.get(l, small_lab.get(l, np.zeros(0, np.int64))) for l in members]))
            if not testable(pool):
                continue
            gl = 2000 + gi
            normal_lab[gl] = pool
            for l in need:
                normal_lab.pop(l, None)
            group_names[gl] = 'group: ' + g[0] + (f' ({g[1]})' if g[1] else '') + ' <- ' + ', '.join(tsn[l] for l in sorted(need))
            for l in need:
                remap[l] = gl
        if remap:
            lut = {int(a): int(b) for a, b in remap.items()}
            for a_, b_ in lut.items():
                LAB[LAB == a_] = b_
            LABg = LAB.reshape(shB)
    xyz_of = lambda idx: (np.stack(np.unravel_index(idx, shB), -1) + org) * sp

    # spatial halves (one split for the whole box, along its longest physical axis)
    j = int(np.argmax(np.array(shB) * sp))
    cut = shB[j] // 2
    if p.get('em_xfit', 'halves') == 'blocks':
        # blocked two-fold cross-fitting: 3-D checkerboard of block_mm cubes; each fold covers the whole box, so a candidate
        # always has local normal tissue in the other fold, and no voxel is scored by a density fitted on itself
        bs = np.maximum(1, np.round(p['em_block_mm'] / sp).astype(int))
        half_of = lambda idx: (np.add.reduce([u // b for u, b in zip(np.unravel_index(idx, shB), bs)]) % 2).astype(np.int8)
    else:
        half_of = lambda idx: (np.unravel_index(idx, shB)[j] >= cut).astype(np.int8)
    N0 = {}                                             # label -> fold -> (log density, pool)
    for l, idx in normal_lab.items():
        hv = half_of(idx)
        N0[l] = {}
        for k in (0, 1):
            ik = idx[hv == k]
            if len(ik) >= p['em_min_ref']:
                N0[l][k] = (_density(BIN[ik], G, eps=p['em_eps']), _Pool(ik, xyz_of(ik), rng))
    allN = {l: _density(BIN[idx], G, eps=p['em_eps']) for l, idx in normal_lab.items()}
    allP = {l: _Pool(idx, xyz_of(idx), rng, ratios=(1,)) for l, idx in normal_lab.items()}
    SUR = p.get('em_compare', 'label') == 'surround'

    def surround(c):
        """Reference classes of c (footprint + surroundings, 7 Oct). Under 'c is normal', c is tissue of a class around it
        or under it:
          surroundings: labels with >= min_ref normal voxels within window_mm of c (normal = outside all candidates +
                        margin, eroded in-plane); share s_l of those surrounding normal voxels;
          footprint:    labels covering >= 5 % of c's own voxels and having a normal model in the box; share f_l of c.
        w_l = (f_l + s_l) / 2 (each part renormalised; if one part is empty the other gets all the weight). The TS label
        of c's voxels is one hypothesis among several, never the only reference. Returns {label: (w, f, s)}."""
        cnt = {}
        for l, pool in allP.items():
            n_ = len(pool.near(c['centre'], c['radius'] + p['em_window_mm']))
            if n_ >= p['em_min_ref']:
                cnt[l] = n_
        tot = sum(cnt.values())
        sur = {l: n_ / tot for l, n_ in cnt.items()} if tot else {}
        labs, nn = np.unique(LAB[c['idx']], return_counts=True)
        fp = {int(l): n_ / len(c['idx']) for l, n_ in zip(labs, nn) if n_ / len(c['idx']) >= 0.05}
        ft = sum(fp.values()); fp = {l: v / ft for l, v in fp.items()} if ft else {}
        if fp and sur:
            ks = set(fp) | set(sur)
            return {l: (0.5 * fp.get(l, 0) + 0.5 * sur.get(l, 0), fp.get(l, 0), sur.get(l, 0)) for l in ks}
        one = fp or sur
        return {l: (v, fp.get(l, 0), sur.get(l, 0)) for l, v in one.items()}

    def lp0_mix(idx, cw, k):
        """log of the surrounding-class mixture density sum_l w_l p0(f | l) (fold k; None = both halves)."""
        terms = []
        for l, wl in cw.items():
            t = allN.get(l) if k is None else (N0.get(l, {}).get(k) or (None,))[0]
            if t is not None:
                terms.append(np.log(wl) + t[BIN[idx]])
        if not terms:
            return None
        return np.logaddexp.reduce(np.stack(terms), axis=0)

    def lam_table(logp1, l, k):
        """lambda on the grid for label l, fold k (None = no normal model)."""
        src = allN.get(l) if k is None else (N0.get(l, {}).get(k) or (None,))[0]
        if src is None or logp1 is None:
            return None
        return np.clip(logp1 - src, -p['em_lclip'], p['em_lclip'])

    def lam(idx, logp1, k, cache):
        out_ = np.full(len(idx), np.nan)
        labs = LAB[idx]
        for l in np.unique(labs):
            if (l, k) not in cache:
                cache[(l, k)] = lam_table(logp1, int(l), k)
            t = cache[(l, k)]
            if t is not None:
                sel = labs == l
                out_[sel] = t[BIN[idx[sel]]]
        return out_

    # ---- EM ---------------------------------------------------------------------------------------------------------------
    for c in feas:
        c['q'] = c['pi']; c['hv'] = half_of(c['idx'])
        c['cls_ref'] = surround(c) if SUR else {}
        c['cls_w'] = {l: v[0] for l, v in c['cls_ref'].items()}
    recall = []
    near10 = _within(U_B, p['em_reach_mm'], sp).ravel()
    env_all = {key: (h['env'][B].ravel() if h['env'].any() else np.zeros(int(np.prod(shB)), bool)) for key, h in hosts.items()}
    oth_B = other[B].ravel()
    H_B = {key: (h['H'][B].ravel() if h['H'].any() else None) for key, h in hosts.items()}
    claims_flat = claims_all.ravel()
    resp = {}
    g0_cache = {}                                       # candidate -> fold -> list of pseudo-candidate index arrays

    def weights():
        """M-step weights: w_v = (max posterior of the candidates covering v) x r_v, where r_v is the voxel responsibility
        of the tumour component in the two-class mixture rho p1 + (1 - rho) p0(.|label) over candidate voxels. Voxels a
        candidate covers but that the normal model of their label explains (over-segmentation, a candidate of normal tissue)
        receive low r_v and do not contaminate the tumour model."""
        wq = np.zeros(int(np.prod(shB)), np.float32)
        for c in feas + recall:
            np.maximum.at(wq, c['idx'], c['q'])
        U_ = np.flatnonzero(wq > 0)
        if not len(U_):
            return wq
        lp0 = np.full(len(U_), np.nan)
        if SUR:
            pos = np.full(int(np.prod(shB)), -1, np.int64); pos[U_] = np.arange(len(U_))
            for c in sorted(feas + recall, key=lambda x: x['q']):       # highest q written last
                if c.get('cls_w'):
                    v_ = lp0_mix(c['idx'], c['cls_w'], None)
                    if v_ is not None:
                        lp0[pos[c['idx']]] = v_
        else:
            labs = LAB[U_]
            for l in np.unique(labs):
                t = allN.get(int(l))
                if t is not None:
                    sel = labs == l
                    lp0[sel] = t[BIN[U_[sel]]]
        r = np.ones(len(U_))
        rho = 0.5
        for _ in range(p['em_inner']):
            lp1 = _density(BIN[U_], G, w=(wq[U_] * r).astype(np.float64), eps=p['em_eps'])
            if lp1 is None:
                break
            a1 = np.log(rho) + lp1[BIN[U_]]
            a0 = np.log(1 - rho) + lp0
            r = np.where(np.isnan(a0), 1.0, 1.0 / (1.0 + np.exp(np.clip(np.nan_to_num(a0) - a1, -50, 50))))
            rho = float(np.clip((wq[U_] * r).sum() / wq[U_].sum(), 0.05, 0.95))
        w = np.zeros_like(wq)
        w[U_] = wq[U_] * r
        R_ = np.zeros_like(wq); R_[U_] = r
        resp['tumour_pool'] = np.flatnonzero(w > 0.05)       # g1: candidate voxels; seeds drawn proportional to w
        resp['rho'] = rho; resp['mean_r'] = float(r.mean())
        return w

    def tumour_logp(w, k, leave=None):
        tp = np.flatnonzero(w > 0)
        if leave is not None:
            tp = np.setdiff1d(tp, leave, assume_unique=True)
        tp = tp[half_of(tp) == k] if k is not None else tp
        if w[tp].sum() < p['em_min_ref'] * 0.5:
            return None, tp
        return _density(BIN[tp], G, w=w[tp].astype(np.float64), eps=p['em_eps']), tp

    def evidence(c, w, selection=None):
        """log BF for candidate/region c (two folds), with pseudo-candidates; selection = per-voxel mask function for
        recall (selection-aware null)."""
        if SUR:
            return evidence_surround(c, w, selection)
        lbf, det = [], []
        sid = c['ci'] if 'ci' in c else int(c['idx'][0])        # deterministic pseudo-candidates: EM is a fixed map
        for k in (0, 1):
            o = 1 - k
            lrng = np.random.default_rng([p['em_seed'], sid, k])
            # cross-fitting: densities from half k; the candidate's voxels and all pseudo-candidates from half o
            logp1, _ = tumour_logp(w, k)
            if logp1 is None:
                det.append(dict(fold=k, logbf=None, why='no_tumour_model')); continue
            idx_o = c['idx'][half_of(c['idx']) == o]
            if len(idx_o) < 20:
                det.append(dict(fold=k, logbf=None, why='too_few_voxels_in_half')); continue
            n_o = len(idx_o)
            cache = {}
            la = lam(idx_o, logp1, k, cache)
            if selection is not None:
                la = la[selection(idx_o)]
            if len(la) == 0 or np.isnan(la).mean() > 0.5:
                labs_, cnt_ = np.unique(LAB[idx_o], return_counts=True)
                det.append(dict(fold=k, logbf=None, why='no_normal_model_for_label',
                                labels={int(a_): round(b_ / n_o, 2) for a_, b_ in zip(labs_, cnt_) if int(a_) not in N0 or k not in N0[int(a_)]}))
                continue
            s = float(np.nanmean(la))
            labs, cnt = np.unique(LAB[idx_o], return_counts=True)
            frac = {int(l): n_ / n_o for l, n_ in zip(labs, cnt)}
            # g0: normal pseudo-candidates with the same label composition, near c, from half o
            key = (sid, o, 'r' if 'ci' not in c else 'c')
            if key not in g0_cache:
                parts = []
                for l, f in frac.items():
                    if l not in N0 or o not in N0[l] or f < 0.05:
                        continue
                    pool = N0[l][o][1]
                    near = pool.near(c['centre'], c['radius'] + p['em_window_mm'])
                    parts.append(pool.blocks(f * n_o, p['em_M'], lrng, p['em_kmax'], seeds_near=near))
                g0_cache[key] = [np.concatenate([pp[m_] for pp in parts]) for m_ in range(p['em_M'])] if parts else []
            s0 = []
            for b in g0_cache[key]:
                lb = lam(b, logp1, k, cache)
                if selection is not None:
                    lb = lb[selection(b)]
                if len(lb) and np.isfinite(lb).any():
                    s0.append(float(np.nanmean(lb)))
            # g1: current tumour voxels (q >= 0.5) of half o
            tp = resp.get('tumour_pool', np.zeros(0, int))
            tp = tp[half_of(tp) == o]
            s1 = []
            if len(tp) >= p['em_min_ref'] and w[tp].sum() > 0:
                pool = _Pool(tp, xyz_of(tp), lrng, ratios=(1, 4, 16, 64))
                sp_ = w[tp].astype(np.float64); sp_ /= sp_.sum()
                for b in pool.blocks(n_o, p['em_M'], lrng, p['em_kmax'], seed_p=sp_):
                    lb = lam(b, logp1, k, cache)
                    if selection is not None:
                        lb = lb[selection(b)]
                    if len(lb) and np.isfinite(lb).any():
                        s1.append(float(np.nanmean(lb)))
            k0, k1 = _kde1(s0), _kde1(s1)
            if k0 is None or k1 is None:
                det.append(dict(fold=k, s=round(s, 3), n0=len(s0), n1=len(s1), logbf=None,
                                why='no_normal_pseudo_candidates' if k0 is None else 'no_tumour_pseudo_candidates'))
                continue
            l_ = float(np.clip(k1.logpdf([s])[0] - k0.logpdf([s])[0], -p['em_bfclip'], p['em_bfclip']))
            lbf.append(l_)
            det.append(dict(fold=k, s=round(s, 3), s0=round(float(np.median(s0)), 3), s1=round(float(np.median(s1)), 3), logbf=round(l_, 3)))
        return (float(np.mean(lbf)) if lbf else 0.0), det, bool(lbf)

    def evidence_surround(c, w, selection=None):
        """Candidate vs each surrounding normal class l, separately: lambda_l(v) = log p1 - log p0(.|l) over ALL voxels of c
        (whatever their own label); g0_l from pseudo-candidates drawn from class l only, near c; g1 from tumour voxels.
        BF_l = g1_l(s_l) / g0_l(s_l). Combination (Bayesian average over which normal class c would be):
        BF_c = 1 / sum_l w_l / BF_l, w_l = share of class l in c's surrounding normal tissue (renormalised over the
        classes with power). Two-fold cross-fitting as before; log BF_c averaged over folds."""
        cw = c.get('cls_w') or {}
        lbf, det = [], []
        sid = c['ci'] if 'ci' in c else int(c['idx'][0])
        for k in (0, 1):
            o = 1 - k
            lrng = np.random.default_rng([p['em_seed'], sid, k])
            logp1, _ = tumour_logp(w, k)
            if logp1 is None or not cw:
                continue
            idx_o = c['idx'][half_of(c['idx']) == o]
            if len(idx_o) < 20:
                continue
            sel_o = selection(idx_o) if selection is not None else np.ones(len(idx_o), bool)
            if not sel_o.any():
                continue
            n_o = len(idx_o)
            # g1 blocks (tumour voxels of half o), shared by all classes
            tp = resp.get('tumour_pool', np.zeros(0, int)); tp = tp[half_of(tp) == o]
            g1b = []
            if len(tp) >= p['em_min_ref'] and w[tp].sum() > 0:
                pool1 = _Pool(tp, xyz_of(tp), lrng, ratios=(1, 4, 16, 64))
                sp_ = w[tp].astype(np.float64); sp_ /= sp_.sum()
                g1b = pool1.blocks(n_o, p['em_M'], lrng, p['em_kmax'], seed_p=sp_)
            per = []
            for l, wl in cw.items():
                ent = N0.get(l, {})
                fr = c.get('cls_ref', {}).get(l, (wl, 0, 0))
                xfit = k in ent and o in ent
                if xfit:
                    dens, pool0 = ent[k][0], ent[o][1]
                elif l in allN:                           # too few voxels in one half: whole-box normal model and pool
                    dens, pool0 = allN[l], allP[l]
                else:                                     # no normal model of this class in the box: untestable (BF = 1)
                    per.append(dict(label=int(l), w=round(wl, 3), fp=round(fr[1], 3), sur=round(fr[2], 3), logbf=None, why='no_normal_model'))
                    continue
                t = np.clip(logp1 - dens, -p['em_lclip'], p['em_lclip'])
                lam_l = lambda b: t[BIN[b]]
                s_l = float(lam_l(idx_o)[sel_o].mean())
                key = (sid, o, l, 'r' if 'ci' not in c else 'c')
                if key not in g0_cache:
                    near = pool0.near(c['centre'], c['radius'] + p['em_window_mm'])
                    if len(near) < 20:                    # e.g. a footprint class whose remaining tissue is farther away
                        sub_, tree_ = pool0.trees[1]
                        _, ii_ = tree_.query(c['centre'], k=min(200, len(sub_)))
                        near = sub_[np.atleast_1d(ii_)]
                    g0_cache[key] = pool0.blocks(n_o, p['em_M'], lrng, p['em_kmax'], seeds_near=near) if len(pool0) >= 20 else []
                def stat(bs):
                    out_ = []
                    for b in bs:
                        lb = lam_l(b)
                        if selection is not None:
                            lb = lb[selection(b)]
                        if len(lb):
                            out_.append(float(lb.mean()))
                    return out_
                s0, s1 = stat(g0_cache[key]), stat(g1b)
                k0, k1 = _kde1(s0), _kde1(s1)
                if k0 is None or k1 is None:
                    per.append(dict(label=int(l), w=round(wl, 3), s=round(s_l, 3), logbf=None))
                    continue
                l_ = float(np.clip(k1.logpdf([s_l])[0] - k0.logpdf([s_l])[0], -p['em_bfclip'], p['em_bfclip']))
                per.append(dict(label=int(l), w=round(wl, 3), fp=round(fr[1], 3), sur=round(fr[2], 3), xfit=xfit, s=round(s_l, 3), s0=round(float(np.median(s0)), 3),
                                s1=round(float(np.median(s1)), 3), logbf=round(l_, 3)))
            pw = [d_ for d_ in per if d_['logbf'] is not None]
            if pw:                                        # untestable classes keep their weight with BF = 1 (neutral)
                ws = np.array([cw[d_['label']] for d_ in per], float); ws /= ws.sum()   # unrounded weights (> 0)
                lb_ = np.array([d_['logbf'] if d_['logbf'] is not None else 0.0 for d_ in per], float)
                comb = float(-np.logaddexp.reduce(np.log(ws) - lb_))      # -log sum_l w_l / BF_l
                comb = float(np.clip(comb, -p['em_bfclip'], p['em_bfclip']))
                lbf.append(comb)
                det.append(dict(fold=k, logbf=round(comb, 3), classes=per))
            else:
                det.append(dict(fold=k, logbf=None, classes=per))
        return (float(np.mean(lbf)) if lbf else 0.0), det, bool(lbf)

    hist = []
    for it in range(p['em_max_iter']):
        w = weights()                                   # M-step input (the tumour model is built inside evidence())
        old = {id(c): c['q'] for c in feas}
        acc_old = {id(c) for c in feas if _accepted(c, out)}
        for c in feas:                                  # E-step over candidates
            lb, det, powered = evidence(c, w)
            c['logbf'], c['det'], c['powered'] = lb, det, powered
            # damped update of the posterior log-odds (relaxed EM): the calibrated E-step is not an exact EM step, and an
            # undamped update can oscillate when a large candidate flips and changes the tumour model
            eta_new = logit(c['pi']) + lb
            # step size: d for the first damp_iters iterations, then 1/(t - damp_iters + 2) (running average of the
            # targets, as in stochastic approximation): a 2-cycle between tumour models is averaged out and q converges
            d = p['em_damp'] if it < p['em_damp_iters'] else 1.0 / (it - p['em_damp_iters'] + 2)
            c['eta'] = eta_new if it == 0 else (1 - d) * c['eta'] + d * eta_new
            c['q'] = sig(c['eta'])
        # recall regions from the current tumour
        w2 = weights()
        Z = np.zeros(int(np.prod(shB)), bool)
        for c in feas:
            if _accepted(c, out):
                Z[c['idx']] = True
        recall = []
        if Z.any():
            hosts_acc = {out[c['lesion']]['host'] for c in feas if _accepted(c, out)}
            allowed = np.zeros_like(Z)
            for key in hosts_acc:
                hb = H_B.get(key)
                hbm = hb if hb is not None else np.zeros_like(oth_B)
                allowed |= env_all[key] & ~(oth_B & ~hbm)
            touch = ndimage.binary_dilation(Z.reshape(shB), structure=ST).ravel()
            allowed &= ~claims_flat & near10
            cand_idx = np.flatnonzero(allowed)
            if len(cand_idx):
                logp1_all, _ = tumour_logp(w2, None)
                cacheA = {}
                la = lam(cand_idx, logp1_all, None, cacheA)
                posmask = np.zeros(int(np.prod(shB)), bool)
                posmask[cand_idx[np.nan_to_num(la, nan=-1) > 0]] = True
                lab_r, nr = ndimage.label(posmask.reshape(shB), structure=ST)
                if nr:
                    lab_r = lab_r.ravel()
                    sz = np.bincount(lab_r)
                    tl = np.unique(lab_r[touch & (lab_r > 0)])
                    selfun = lambda idx: np.nan_to_num(lam(idx, logp1_all, None, cacheA), nan=-1) > 0
                    for r_ in tl:
                        if sz[r_] < min_vox:
                            continue
                        ridx = np.flatnonzero(lab_r == r_)
                        rr = dict(kind='recall', pi=p['em_pi_recall'], idx=ridx, xyz=xyz_of(ridx), ml=round(len(ridx) * vox_ml, 2))
                        rr['centre'] = rr['xyz'].mean(0); rr['radius'] = float(np.linalg.norm(rr['xyz'] - rr['centre'], axis=1).max())
                        rr['cls_ref'] = surround(rr) if SUR else {}
                        rr['cls_w'] = {l: v[0] for l, v in rr['cls_ref'].items()}
                        lb, det, powered = evidence(rr, w2, selection=selfun)
                        rr.update(logbf=lb, det=det, powered=powered, q=sig(logit(rr['pi']) + lb))
                        recall.append(rr)
        dq = max(abs(c['q'] - old[id(c)]) for c in feas)
        acc = {id(c) for c in feas if _accepted(c, out)}
        hist.append(dict(it=it, rho=round(resp.get('rho', 0), 3), mean_r=round(resp.get('mean_r', 0), 3), n_acc=len(acc), n_recall=sum(r['q'] >= 0.5 for r in recall), dq=round(dq, 4)))
        report['iterations'] = it + 1
        if acc == acc_old and dq < p['em_tol'] and it > 0:
            report['converged'] = True
            break
    report['history'] = hist
    tsn = ctx.get('ts_names') or {}
    names_ = {0: 'unlabelled tissue'}
    for l in set(int(x) for x in np.unique(LAB)):
        if l in tsn: names_[l] = tsn[l]
    for key, l in hostlab.items():
        names_[l] = 'host region: ' + (key if isinstance(key, str) else '_'.join(str(x) for x in key if x))
    names_.update(group_names)
    report['class_names'] = {str(k): v for k, v in names_.items()}
    report['kb_groups'] = group_names
    report['compare'] = 'surround' if SUR else 'label'

    # ---- voxel maps for review (outputs only; no decision depends on them) ---------------------------------------------
    # region = feasible candidates + reach_mm, cropped to its bounding box. Encodings (sentinels mark 'no value'):
    #   lam  int8  = round(25 * lambda), lambda = log p1 - log p0(.|label) clipped to +-lclip; -128 = no normal model / outside
    #   resp uint8 = round(200 * r), r = sigmoid(logit rho + log p1 - log p0(.|label)) (voxel responsibility); 255 = none
    #   q    uint8 = round(200 * q) of the candidate or recall region covering the voxel (max if several); 255 = no candidate
    try:
        w_fin = weights()
        logp1_fin, _ = tumour_logp(w_fin, None)
        reg = near10.copy()
        for c in feas:
            reg[c['idx']] = True
        ridx = np.flatnonzero(reg)
        if logp1_fin is not None and len(ridx):
            lam_raw = np.full(len(ridx), np.nan)
            labs = LAB[ridx]
            for l in np.unique(labs):
                t = allN.get(int(l))
                if t is not None:
                    sel = labs == l
                    lam_raw[sel] = logp1_fin[BIN[ridx[sel]]] - t[BIN[ridx[sel]]]
            if SUR:   # candidate voxels: lambda against their candidate's surrounding mixture (highest-q candidate wins)
                pos = np.full(int(np.prod(shB)), -1, np.int64); pos[ridx] = np.arange(len(ridx))
                for c in sorted(feas + recall, key=lambda x: x['q']):
                    if c.get('cls_w'):
                        v_ = lp0_mix(c['idx'], c['cls_w'], None)
                        if v_ is not None:
                            ii = pos[c['idx']]; m_ = ii >= 0
                            lam_raw[ii[m_]] = logp1_fin[BIN[c['idx'][m_]]] - v_[m_]
            rho_ = float(resp.get('rho', 0.5))
            ok = ~np.isnan(lam_raw)
            lam8 = np.full(len(ridx), -128, np.int8)
            lam8[ok] = np.round(25 * np.clip(lam_raw[ok], -p['em_lclip'], p['em_lclip'])).astype(np.int8)
            r8 = np.full(len(ridx), 255, np.uint8)
            r8[ok] = np.round(200 / (1 + np.exp(-np.clip(np.log(rho_ / (1 - rho_)) + lam_raw[ok], -50, 50)))).astype(np.uint8)
            qf = np.full(int(np.prod(shB)), -1.0, np.float32)
            for c in feas + [r for r in recall]:
                np.maximum.at(qf, c['idx'], np.float32(c['q']))
            qv = qf[ridx]
            q8 = np.where(qv < 0, 255, np.round(200 * np.clip(qv, 0, 1))).astype(np.uint8)
            rm = reg.reshape(shB)
            bb_r = ndimage.find_objects(rm.astype(np.int8))[0]
            shR = tuple(b.stop - b.start for b in bb_r)
            loc = np.stack(np.unravel_index(ridx, shB), -1) - np.array([b.start for b in bb_r])
            def _arr(v, fill, dt):
                a_ = np.full(shR, fill, dt); a_[tuple(loc.T)] = v; return a_
            cls8 = LAB[ridx].astype(np.int32)                     # class of each voxel (normal reference classes)
            report['_maps'] = dict(origin=np.array([b.start + q.start for b, q in zip(B, bb_r)], np.int32),
                                   lam=_arr(lam8, -128, np.int8), resp=_arr(r8, 255, np.uint8), q=_arr(q8, 255, np.uint8),
                                   cls=_arr(cls8, -1, np.int32),
                                   scale=np.array([25.0, 200.0, 200.0], np.float32))
    except Exception as e:                                # maps are optional outputs: never fail the ledger for them
        report['maps_error'] = repr(e)[:200]

    return _finish(out, feas, cands, recall, report, B, shB, hosts, ctx, sp, vox_ml)


def _finish(out, feas, cands, recall, report, B, shB, hosts, ctx, sp, vox_ml):
    # ---- output: lesion masks from accepted candidates and recall regions ---------------------------------------------
    acc_mask = np.zeros(int(np.prod(shB)), bool)
    for c in feas:
        if _accepted(c, out):
            acc_mask[c.get('keep_idx', c['idx'])] = True     # candidate mode: trimmed (dark border regions removed)
    rec_mask = np.zeros(int(np.prod(shB)), bool)
    for r in recall:
        if r['q'] >= 0.5:
            rec_mask[r['idx']] = True
    report['recalled_ml'] = round(float(rec_mask.sum()) * vox_ml, 2)
    acc_mask = acc_mask.reshape(shB); rec_mask = rec_mask.reshape(shB)
    # a lesion keeps only the accepted voxels of ITS OWN candidates (9 Oct: ledger lesions can overlap - a 68 ml VT
    # region containing another lesion's BP claim - and a scan-wide accepted mask let a lesion whose candidates were all
    # rejected inherit its neighbour's voxels)
    own_acc = {}
    for i in range(len(out)):
        m_ = np.zeros(int(np.prod(shB)), bool)
        for c in feas:
            if c.get('lesion') == i and _accepted(c, out):
                m_[c.get('keep_idx', c['idx'])] = True
        own_acc[i] = m_.reshape(shB)
    owned_grow = {}                                       # candidate mode: grown regions belong to the candidate's lesion
    for r in recall:
        if 'owner' in r and r['q'] >= 0.5:
            li_ = next((c['lesion'] for c in feas if c['ci'] == r['owner']), None)
            if li_ is not None:
                owned_grow.setdefault(li_, []).append(r)
    lab_rec, _ = ndimage.label(rec_mask, structure=ST)
    # each recall region joins exactly one lesion: the lesion with accepted voxels it touches most (contact voxels after a
    # 1-voxel dilation of the lesion's accepted part); ties -> the larger lesion. (Before 7 Oct a region touching several
    # lesions was added to every one of them, duplicating its volume.)
    owner = {}
    if lab_rec.max():
        best = {}
        for i, les in enumerate(out):
            if not any(c.get('lesion') == i for c in feas):
                continue
            kp = _put(B, les['sl'], les['L'], shB) & own_acc[i]
            if not kp.any():
                continue
            ids_, cnt_ = np.unique(lab_rec[ndimage.binary_dilation(kp, structure=ST) & (lab_rec > 0)], return_counts=True)
            for r_, n_ in zip(ids_, cnt_):
                key_ = (int(n_), int(kp.sum()))
                if r_ not in best or key_ > best[r_][0]:
                    best[r_] = (key_, i)
        owner = {r_: v[1] for r_, v in best.items()}
    for i, les in enumerate(out):
        mine = [c for c in feas if c.get('lesion') == i]
        shared = False
        Lb = _put(B, les['sl'], les['L'], shB)
        if not mine:
            # candidate mode: a lesion none of whose claims was assigned to it (each claim goes to the lesion it overlaps
            # most) is judged by the candidates that claim its voxels, restricted to its own voxels (9 Oct: OV 6.1 L3 and
            # CESC L1 had kept their ledger admission without any test)
            if not any(c.get('strict') for c in feas):
                continue
            Lf = Lb.ravel()
            mine = [c for c in feas if Lf[c['idx']].any()]
            if not mine:
                continue
            shared = True
            m_ = np.zeros(int(np.prod(shB)), bool)
            for c in mine:
                if _accepted(c, out):
                    m_[c.get('keep_idx', c['idx'])] = True
            own_acc[i] = m_.reshape(shB)
        keep = Lb & own_acc[i]
        if keep.any():
            if any('owner' in r for r in recall):          # candidate mode: own grown regions only
                for r in owned_grow.get(i, []):
                    keep.ravel()[r['idx']] = True
            else:
                ids = [r_ for r_, o_ in owner.items() if o_ == i]
                if ids:
                    keep |= np.isin(lab_rec, ids)
        before = les['status']
        if keep.any():                                   # status follows the posterior only (7 Oct: no size rule here)
            if before != 'admitted':
                les['status'], les['reason'] = 'admitted', 'r2_em_recovered'; report['n_recovered_lesions'] += 1
            bb2 = ndimage.find_objects(keep.astype(np.int8))[0]
            nsl = tuple(slice(b.start + a.start, b.start + a.stop) for a, b in zip(bb2, B))
            bpL = _put(nsl, les['sl'], les['bp'], tuple(s.stop - s.start for s in nsl))
            vtL = _put(nsl, les['sl'], les['vt'], tuple(s.stop - s.start for s in nsl))
            les['L'] = keep[bb2]; les['bp'] = bpL & les['L']; les['vt'] = vtL & les['L']; les['sl'] = nsl
            les['vox'] = int(les['L'].sum()); les['agree'] = int((les['bp'] & les['vt']).sum())
            ext = [(nsl[a].stop - nsl[a].start) * sp[a] for a in range(3) if a != ctx['k']]
            les['ld_mm'] = float(np.hypot(*ext))
            nz = int(les['L'].any(axis=tuple(a for a in range(3) if a != ctx['k'])).sum())
            les['n_slices'], les['z_mm'] = nz, round(nz * float(sp[ctx['k']]), 1)
            les['frac'] = {q: int((les['L'] & hh['env'][nsl]).sum()) / les['vox'] for q, hh in hosts.items()}
        else:
            no_power = any(c.get('strict') for c in mine) and not any(c.get('powered') for c in mine)
            if no_power:                                  # nothing tested: not admitted, flagged for expert review
                if before == 'admitted':
                    report['n_rejected_lesions'] += 1
                les['status'], les['reason'] = 'rejected', 'r2_no_power'
                report['n_no_power_lesions'] = report.get('n_no_power_lesions', 0) + 1
            elif before == 'admitted':
                les['status'], les['reason'] = 'rejected', 'r2_em_rejected'; report['n_rejected_lesions'] += 1
        les['r2'] = dict(before=before, candidates=[dict(source=c['source'], rater=c['rater'], ml=c['ml'], state=c['state'], map_i=c.get('map_i'),
                                                          pi=c['pi'], q=round(c['q'], 3), logbf=round(c['logbf'], 3),
                                                          powered=c['powered'], accepted=_accepted(c, out), folds=c['det']) for c in mine],
                         recall=[], shared_claims=shared,
                         review=('no_power' if les.get('reason') == 'r2_no_power' else
                                 'partly_untested' if les['status'] == 'admitted' and any(not c['powered'] for c in mine) else None))
        if any('owner' in r for r in recall):
            les['r2']['recall'] = [dict(ml=r['ml'], q=round(r['q'], 3), logbf=round(r['logbf'], 3)) for r in owned_grow.get(i, [])]
        elif recall:
            Lb_d = ndimage.binary_dilation(Lb, structure=ST).ravel()       # once per lesion (was once per region)
            les['r2']['recall'] = [dict(ml=r['ml'], q=round(r['q'], 3), logbf=round(r['logbf'], 3)) for r in recall if Lb_d[r['idx']].any()]
    report['n_accepted'] = sum(_accepted(c, out) for c in feas)
    report['n_rejected'] = sum(not _accepted(c, out) for c in feas)
    report['infeasible'] = [dict(source=c['source'], ml=c['ml'], why=c['why']) for c in cands if c['pi'] == 0][:50]
    return report


# ======================================================================================================================
# Candidate-specific R2 (em_model='candidate', 8 Oct 2026)
#   Each candidate c is tested on its own scan against its own local normal environment; nothing is shared between
#   candidates or learned across scans. Prior (stage 1) and image evidence (this stage) are combined once:
#   q_c = sigmoid(logit pi_c + log BF_c).
#   reference   for every label l in c's footprint, normal voxels (outside all claims + margin, eroded in-plane) of the same
#               tissue: l itself and its siblings (same top KB part-of ancestor, either side; e.g. all vertebrae, both
#               lungs' lobes, both kidneys), within radius r of c, r grown from window_mm in ref_step_mm steps up to
#               ref_max_mm until both folds hold >= min_ref voxels; sibling instances enter WHOLE (all their normal voxels,
#               if they come within ref_max_mm), so an organ's sub-structures keep their natural proportions. Labels that
#               the claims deplete (>= em_dep of the label's voxels in the box inside claims) are used only as a last
#               resort: the remnant of an organ a candidate covers is not representative of it (e.g. the posterior
#               elements left of a claimed vertebral body).
#               H0 for c: its voxels are drawn from p0mix = sum_l w_l p0_l (w_l = c's label composition).
#   folds       normal tissue: 3-D checkerboard of block_mm cubes. Candidate (and each null patch): its own checkerboard,
#               block size halved from block_mm until each fold has >= min_own voxels (>= 3 voxels in-plane; features are
#               3x3 in-plane, so one slice through-plane is leakage-free).
#   statistic   s_c = held-out density ratio: f_c^(k) = KDE of c's fold-k voxels shrunk to p0mix^(k) (kappa voxels);
#               mean over fold-o voxels of log f_c^(k) - log p0mix^(k) (clipped); averaged over the two folds. It measures
#               (tumour fraction) x (contrast) in nats per voxel; no tumour/normal decomposition is needed for the test.
#   null        M normal patches of c's size and label composition, drawn from c's reference near c, each put through the
#               identical procedure (own folds; its voxels removed from the reference). delta_c = s_c - median(s0),
#               sigma_c = sd(s0): c's own contrast and c's own noise.
#   evidence    H0 true contrast 0, H1 true contrast ~ half-normal(tau), observed delta ~ N(true, sigma_c^2):
#               BF_c closed form (_logbf_halfnormal). tau (nats/voxel) is a fixed pan-cancer constant derived a priori by
#               simulation (a lesion 1 normal-tissue SD brighter than its tissue; scripts/r2_tau_sim.py). Large claims that
#               match their neighbourhood (delta ~ 0, small sigma) get evidence against; small noisy ones BF ~ 1.
#               p (empirical) and the Sellke / mixture calibrations are reported for reference only.
#   description after the decision, within-candidate EM: c's voxels ~ rho p1_c + (1 - rho) p0(.|label) -> tumour fraction
#               rho_c, voxel responsibilities r_v, voxel posterior q_c x r_v (maps). No decision depends on it.
#   recall      for accepted candidates: voxels within reach_mm in the lesion host's envelope, not claimed, with
#               log p1_c - log p0 > 0, connected and touching c; scored against normal patches with the same selection
#               (empirical p, mixture calibrator, a proper Bayes factor); prior pi_recall.
# ======================================================================================================================
_RATIOS = (1, 4, 16, 64, 256, 1024)


def _smooth_prob(h, G, n_eff):
    tot = h.sum()
    if tot <= 0:
        return None
    sigma = max(0.5, (max(n_eff, 2) ** (-1.0 / 7.0)) * (G / np.sqrt(12.0)))
    d = ndimage.gaussian_filter(h.reshape(G, G, G), sigma, mode='reflect').ravel()
    return d / d.sum()


def _logbf_from_z(z, R):
    """One-sided default test on the null-standardised statistic: z ~ N(delta, 1); H0 delta = 0, H1 delta ~ half-normal(R).
    BF10 = 2 (1 + R^2)^(-1/2) exp(z^2 R^2 / (2 (1 + R^2))) Phi(z R / sqrt(1 + R^2)); can give evidence against (z ~ 0)."""
    v = 1.0 + R * R
    return float(np.log(2.0) - 0.5 * np.log(v) + z * z * R * R / (2 * v) + stats.norm.logcdf(z * R / np.sqrt(v)))


def _heldout_stat(bins, par, p0mix, scale, G, eps, kappa, lclip):
    """Held-out density ratio of a voxel set against its reference mixture (two folds).
    Fold k: f^(k) = KDE of the set's fold-k voxels (Scott bandwidth), shrunk to the reference: (n f + kappa p0) / (n + kappa),
    n = voxels represented (thinned points x scale). Score fold-o voxels: lambda = log f^(k) - log p0^(k), clipped.
    Returns the mean of the two folds' mean lambda (nats per voxel), or None."""
    G3 = G ** 3
    out = []
    for k in (0, 1):
        F, O = bins[par == k], bins[par == 1 - k]
        if len(F) < 2 or len(O) < 1:
            return None
        pr = _smooth_prob(np.bincount(F, minlength=G3).astype(np.float64), G, len(F))
        if pr is None:
            return None
        nv = len(F) * scale
        f = (nv * pr + kappa * p0mix[k]) / (nv + kappa)
        lam = np.log((1 - eps) * f[O] + eps / G3) - np.log((1 - eps) * p0mix[k][O] + eps / G3)
        out.append(float(np.mean(np.clip(lam, -lclip, lclip))))
    return float(np.mean(out))


def _logbf_halfnormal(delta, sigma, tau):
    """Bayes factor tumour vs normal for an observed contrast delta with null noise sigma (both nats/voxel):
    H0: true contrast 0; H1: true contrast ~ half-normal(tau); observation ~ N(true, sigma^2)."""
    sigma = max(float(sigma), 1e-9)
    return _logbf_from_z(float(delta) / sigma, float(tau) / sigma)


def _logbf_from_p(pv, how):
    if how == 'sellke':
        return 0.0 if pv >= 1 / np.e else float(-np.log(-np.e * pv * np.log(pv)))
    if pv >= 1 - 1e-12:
        return float(np.log(0.5))
    lp = np.log(pv)
    return float(np.log((1 - pv + pv * lp) / (pv * lp * lp)))


def _candidate_r2(feas, out, hosts, ctx, p, B, shB, org, BIN, LAB, U_B, claims_all, hostlab, report, min_vox):
    for c in feas:
        c['strict'] = True
    sp = np.asarray(ctx['sp'], float); ax = ctx['ax']; G = p['em_G']; G3 = G ** 3; eps = p['em_eps']
    lclip = p['em_lclip']; M = p['em_M']; kappa = p['em_kappa']; a_ = p['em_rho_ab']
    nB = int(np.prod(shB)); LABg = LAB.reshape(shB)
    claims_flat = claims_all.ravel()
    logp = lambda pr: np.log((1 - eps) * pr + eps / G3)
    unr = lambda idx: np.stack(np.unravel_index(idx, shB), -1)

    # ---- label bookkeeping: depletion and sibling keys -------------------------------------------------------------------
    labs_all, cnt_all = np.unique(LAB, return_counts=True)
    tot = dict(zip(labs_all.tolist(), cnt_all.tolist()))
    dep = {l: 0.0 for l in tot}
    lc, nc = np.unique(LAB[claims_flat], return_counts=True)
    for l, n_ in zip(lc.tolist(), nc.tolist()):
        dep[l] = n_ / tot[l]
    dep[0] = 0.0
    tsn = ctx.get('ts_names') or {}
    kb_, task_ = ctx.get('kb'), ctx.get('ts_task')
    tmap = kb_.ts_to_entity.get(task_, {}) if kb_ is not None else {}

    def top_of(e):
        if kb_ is None:
            return e
        seen = set()
        while True:
            ps = [r['src'] for r in kb_.in_edges.get(e, []) if r['type'] == 'has_part' and r.get('spatial', 'inside') in ('inside', 'group')]
            if not ps or ps[0] in seen:
                return e
            seen.add(e); e = ps[0]
    strip = lambda e: str(e).replace('_left', '').replace('_right', '')
    host_of_lab = {v: k for k, v in hostlab.items()}
    sibkey, names = {}, {0: 'unlabelled tissue'}
    for l in tot:
        if l == 0:
            sibkey[l] = ('L', 0)
        elif l in host_of_lab:
            key = host_of_lab[l]; ent = key if isinstance(key, str) else key[0]
            sibkey[l] = ('E', strip(top_of(ent)))
            names[l] = 'host region: ' + (key if isinstance(key, str) else '_'.join(str(x) for x in key if x))
        elif l in tsn and tsn[l] in tmap:
            sibkey[l] = ('E', strip(top_of(tmap[tsn[l]][0]))); names[l] = tsn[l]
        else:
            sibkey[l] = ('N', strip(tsn.get(l, l))); names[l] = tsn.get(l, str(l))

    # ---- normal tissue per label (wide window), global block folds ----------------------------------------------------
    normal = ~_within(claims_all, p['em_margin_mm'], sp) & _within(U_B, p['em_ref_max_mm'], sp)
    ER = np.zeros((3, 3, 3), bool); e_ = [slice(None)] * 3; e_[ax] = 1; ER[tuple(e_)] = True
    nidx = {}
    for l in np.unique(LABg[normal]):
        Ml = ndimage.binary_erosion(normal & (LABg == l), structure=ER)
        if Ml.any():
            nidx[int(l)] = np.flatnonzero(Ml)
    del normal
    bsg = np.maximum(1, np.round(p['em_block_mm'] / sp).astype(int))

    def parity(idx, bsz):
        return (np.add.reduce([u // b for u, b in zip(np.unravel_index(idx, shB), bsz)]) % 2).astype(np.int8)
    nfold = {l: parity(ix, bsg) for l, ix in nidx.items()}
    minpl = np.array([1 if a == ax else 3 for a in range(3)])

    def local_blocks(idx):
        mm = float(p['em_block_mm'])
        while True:
            bsz = np.maximum(minpl, np.round(mm / sp).astype(int))
            par = parity(idx, bsz)
            n1 = int(par.sum())
            if min(n1, len(par) - n1) >= p['em_min_own']:
                return bsz, par
            if np.all(bsz <= minpl):
                return None, None
            mm /= 2.0

    # ---- within-candidate EM ----------------------------------------------------------------------------------------------
    def fit_p1(binsF, lp0F, pooled, scale):
        """Within-candidate EM (descriptive, after the decision): p1 = r-weighted KDE (Scott bandwidth), shrunk to `pooled`
        only if one is given. (8 Oct: shrinking to a flat density with weight kappa / sum(r) made the EM collapse to the rho
        floor - fewer tumour voxels -> flatter p1 -> fewer tumour voxels - so the descriptive EM now uses no shrinkage.)"""
        n = len(binsF); r = np.ones(n); rho = 0.5; lp1 = None
        for it in range(p['em_inner_c'] + 1):
            h = np.bincount(binsF, weights=r, minlength=G3)
            ne = r.sum() ** 2 / max((r * r).sum(), 1e-12)
            pr = _smooth_prob(h, G, ne)
            if pr is None:
                return None, rho, r
            nv = r.sum() * scale
            lp1 = logp(pr if pooled is None else (nv * pr + kappa * pooled) / (nv + kappa))
            if it == p['em_inner_c']:
                break
            a1 = np.log(rho) + lp1[binsF]
            a0 = np.log(1 - rho) + lp0F
            r = np.where(np.isnan(a0), 1.0, 1.0 / (1.0 + np.exp(np.clip(np.nan_to_num(a0) - a1, -50, 50))))
            rho = float(np.clip((r.sum() + a_ - 1) / (n + 2 * a_ - 2), 0.02, 0.98))
        return lp1, rho, r

    # ---- reference selection --------------------------------------------------------------------------------------------
    def make_ref(c, l):
        """Reference of label l for candidate c: dict(idx, fold, H, p0, n, labels, r, tier, dep).
        tier 0: l's own normal remnant within radius r of c (if l is not depleted by the claims) plus WHOLE sibling instances
                (other labels with the same top KB part-of ancestor, e.g. other vertebrae, other lobes, the other kidney) that
                the claims leave intact (depletion < em_dep) and that come within ref_max_mm of c. Whole instances keep the
                organ's sub-structures in their natural proportions (vertebral body and posterior elements), which the
                nearest voxels of a partly claimed neighbour do not.
        tier 1 (last resort): every same-tissue label's remnant within radius r, depleted or not.
        r grows from window_mm in ref_step_mm steps up to ref_max_mm until both folds hold >= min_ref voxels."""
        D = c['D']
        sib = [l2 for l2 in nidx if sibkey.get(l2) == sibkey.get(l)]
        okd = lambda l2: l2 == 0 or dep.get(l2, 0) < p['em_dep']

        def dist_of(idx):
            co = unr(idx) - c['clo']
            inside = np.all((co >= 0) & (co < np.array(D.shape)), axis=1)
            dd = np.full(len(idx), np.inf); dd[inside] = D[tuple(co[inside].T)]
            return dd
        whole = [l2 for l2 in sib if l2 != l and l2 != 0 and okd(l2)]
        whole = [l2 for l2 in whole if dist_of(nidx[l2]).min() <= p['em_ref_max_mm']]
        tiers = [([l] if (l in nidx and okd(l)) else [], whole), (sib, [])]
        for ti, (local, wl) in enumerate(tiers):
            if not local and not wl:
                continue
            if ti == 1 and set(local) <= set(tiers[0][0]) and not tiers[0][1]:
                continue
            li = np.concatenate([nidx[l2] for l2 in local]) if local else np.zeros(0, np.int64)
            lf = np.concatenate([nfold[l2] for l2 in local]) if local else np.zeros(0, np.int8)
            ll = np.concatenate([np.full(len(nidx[l2]), l2) for l2 in local]) if local else np.zeros(0, np.int64)
            dl_ = dist_of(li)
            wi = np.concatenate([nidx[l2] for l2 in wl]) if wl else np.zeros(0, np.int64)
            wf = np.concatenate([nfold[l2] for l2 in wl]) if wl else np.zeros(0, np.int8)
            wlab = np.concatenate([np.full(len(nidx[l2]), l2) for l2 in wl]) if wl else np.zeros(0, np.int64)
            r = p['em_window_mm']
            while True:
                sel = dl_ <= r
                ii = np.concatenate([li[sel], wi]); ff = np.concatenate([lf[sel], wf]); dl = np.concatenate([ll[sel], wlab])
                n1 = int(ff.sum()); n0 = len(ff) - n1
                if min(n0, n1) >= p['em_min_ref']:
                    H = [np.bincount(BIN[ii[ff == k]], minlength=G3).astype(np.float64) for k in (0, 1)]
                    used = {int(a): int(b) for a, b in zip(*np.unique(dl, return_counts=True))}
                    ref = dict(idx=ii, fold=ff, H=H, n=[n0, n1], r=r, tier=ti, labels=used, whole=[int(x) for x in wl],
                               dep={int(l2): round(dep.get(l2, 0), 2) for l2 in sib})
                    ref['p0'] = [_smooth_prob(H[k], G, H[k].sum()) for k in (0, 1)]
                    ref['lp0'] = [logp(x) for x in ref['p0']]
                    ref['lp0all'] = logp(_smooth_prob(H[0] + H[1], G, n0 + n1))
                    return ref
                if r >= p['em_ref_max_mm'] - 1e-6:
                    break
                r = min(r + p['em_ref_step_mm'], p['em_ref_max_mm'])
        return None

    def ratio_for(n):
        for rr in _RATIOS:
            if n / rr <= p['em_cap']:
                return rr
        return _RATIOS[-1]

    def p0mix(refs, comp, adj=None):
        """Reference mixture for a voxel set with label composition comp {label: count}: sum_l w_l p0_l^(k)."""
        tot_ = float(sum(comp.values()))
        return [sum((n_ / tot_) * (adj[l][k] if adj is not None else refs[l]['p0'][k]) for l, n_ in comp.items()) for k in (0, 1)]

    def adj_p0(refs, pidx_, plab, pfold, scale):
        """Reference densities with a null patch's points removed (exact: the patch's points are points of the (thinned)
        reference, each standing for `scale` voxels; pfold = their reference folds)."""
        outd = {}
        for l, ref in refs.items():
            if ref is None:
                continue
            sel = plab == l
            if not sel.any():
                outd[l] = ref['p0']; continue
            pf = pfold[sel]
            lst = []
            for k in (0, 1):
                h = np.maximum(ref['H'][k] - scale * np.bincount(BIN[pidx_[sel][pf == k]], minlength=G3), 0)
                pr = _smooth_prob(h, G, h.sum())
                lst.append(pr if pr is not None else ref['p0'][k])
            outd[l] = lst
        return outd

    # ---- per candidate --------------------------------------------------------------------------------------------------
    refmap = np.zeros(nB, np.uint8)
    padv = np.ceil(p['em_ref_max_mm'] / sp).astype(int) + 1
    for c in feas:
        cidx = c['idx']
        co = unr(cidx)
        lo = np.maximum(co.min(0) - padv, 0); hi = np.minimum(co.max(0) + padv + 1, np.array(shB))
        Mc = np.zeros(tuple(hi - lo), bool); Mc[tuple((co - lo).T)] = True
        c['D'] = ndimage.distance_transform_edt(~Mc, sampling=sp).astype(np.float32); c['clo'] = lo
        del Mc
        raw = LAB[cidx]
        rl, rn = np.unique(raw, return_counts=True)
        canon = {}
        for l in rl[np.argsort(-rn)].tolist():          # one reference per sibling key (largest footprint label names it)
            canon[l] = next((l2 for l2 in canon.values() if sibkey.get(l2) == sibkey.get(l)), l)
        c['canon'] = canon
        clab = np.array([canon[l] for l in raw.tolist()]) if len(raw) else raw
        labs, cnts = np.unique(clab, return_counts=True)
        refs = {}
        for l, n_ in zip(labs.tolist(), cnts.tolist()):
            refs[l] = make_ref(c, l) if (n_ >= 0.05 * len(cidx) or n_ >= p['em_min_own']) else None
        c['refs'] = refs
        det = dict(n_vox=len(cidx), refs=[dict(label=names.get(l, str(l)), share=round(n_ / len(cidx), 3),
                                               ref=None if refs[l] is None else dict(r_mm=refs[l]['r'], tier=refs[l]['tier'], n=refs[l]['n'],
                                                   labels={names.get(a, str(a)): b for a, b in refs[l]['labels'].items()},
                                                   whole=[names.get(a, str(a)) for a in refs[l]['whole']],
                                                   dep={names.get(a, str(a)): b for a, b in refs[l]['dep'].items()}))
                                          for l, n_ in zip(labs.tolist(), cnts.tolist())])
        c['powered'] = False; c['logbf'] = 0.0; c['det'] = [det]
        c['ref_idx'], c['ref_tier'] = [], []
        for l, ref in refs.items():
            if ref is not None:
                refmap[ref['idx']] = np.maximum(refmap[ref['idx']], 1 + ref['tier'])
                c['ref_idx'].append(ref['idx']); c['ref_tier'].append(np.full(len(ref['idx']), 1 + ref['tier'], np.uint8))
        bsz, par_full = local_blocks(cidx)
        if bsz is None:
            det['why'] = 'too_small_for_folds'; c['q'] = c['pi']; continue
        det['block_vox'] = bsz.tolist()
        rc = ratio_for(len(cidx))
        lrng = np.random.default_rng([p['em_seed'], c['ci']])
        okv = np.array([refs.get(l) is not None for l in clab.tolist()]) if len(clab) else np.zeros(0, bool)
        if okv.mean() < 0.5:
            det['why'] = 'no_reference_for_label'; c['q'] = c['pi']; continue
        comp = {l: n_ for l, n_ in zip(labs.tolist(), cnts.tolist()) if refs.get(l) is not None}
        vidx, vpar = cidx[okv], par_full[okv]
        sub = np.sort(lrng.choice(len(vidx), int(np.ceil(len(vidx) / rc)), replace=False)) if rc > 1 else np.arange(len(vidx))
        c['bsz'] = bsz
        # null: patches of c's size and label composition from c's own reference, through the identical procedure (their
        # voxels removed from the reference they are compared with)
        # with thinning (rc > 1) the candidate and its patches use the same thinned reference (each point = rc voxels), so a
        # patch's points are removed from it exactly
        trees, refs_t = {}, {}
        for l in comp:
            ref = refs[l]
            nsub = max(1, len(ref['idx']) // rc)
            si = lrng.choice(len(ref['idx']), nsub, replace=False) if rc > 1 else np.arange(len(ref['idx']))
            trees[l] = (ref['idx'][si], cKDTree((unr(ref['idx'][si]) + org) * sp), ref['fold'][si])
            if rc > 1:
                Ht = [rc * np.bincount(BIN[ref['idx'][si][ref['fold'][si] == k]], minlength=G3).astype(np.float64) for k in (0, 1)]
                refs_t[l] = dict(H=Ht, p0=[_smooth_prob(Ht[k], G, Ht[k].sum()) for k in (0, 1)])
            else:
                refs_t[l] = ref
        s = _heldout_stat(BIN[vidx[sub]], vpar[sub], p0mix(refs_t, comp), rc, G, eps, kappa, lclip)
        if s is None:
            det['why'] = 'no_statistic'; c['q'] = c['pi']; continue
        det.update(s=round(s, 5), thin=rc, n_tested=int(okv.sum()))
        ncomp = sum(comp.values())
        s0 = []
        for m_ in range(p['em_M_null']):
            parts, plabs, pfs = [], [], []
            for l, n_ in comp.items():
                pts, tree, pfo = trees[l]
                k_ = max(1, min(int(round(len(vidx) * n_ / ncomp / rc)), len(pts) // 2))
                seed = tree.data[lrng.integers(len(pts))]
                _, ii = tree.query(seed, k=k_)
                ii = np.atleast_1d(ii)
                parts.append(pts[ii]); plabs.append(np.full(len(ii), l)); pfs.append(pfo[ii])
            pidx_ = np.concatenate(parts); plab = np.concatenate(plabs); pfold = np.concatenate(pfs)
            ppar = parity(pidx_, bsz)
            if min(int(ppar.sum()), len(ppar) - int(ppar.sum())) < max(2, p['em_min_own'] // rc):
                continue
            pcomp = {l: int((plab == l).sum()) for l in comp if (plab == l).any()}
            s_ = _heldout_stat(BIN[pidx_], ppar, p0mix(refs_t, pcomp, adj_p0(refs_t, pidx_, plab, pfold, rc)), rc, G, eps, kappa, lclip)
            if s_ is not None:
                s0.append(s_)
        if len(s0) < p['em_min_null']:
            det['why'] = 'no_null'; det['n0'] = len(s0); c['q'] = c['pi']; continue
        s0 = np.array(s0)
        delta = s - float(np.median(s0)); sigma = float(s0.std())
        pv = (1 + int((s0 >= s).sum())) / (len(s0) + 1)
        lb = dict(halfnormal=_logbf_halfnormal(delta, sigma, p['em_tau']),
                  sellke=_logbf_from_p(pv, 'sellke'), mix=_logbf_from_p(pv, 'mix'))
        c['logbf'] = float(np.clip(lb[p['em_calib_c']], -p['em_bfclip'], p['em_bfclip']))
        c['powered'] = True
        c['q'] = sig(logit(c['pi']) + c['logbf'])
        det.update(n0=len(s0), delta=round(delta, 5), sigma=round(sigma, 5), z=round(delta / max(sigma, 1e-9), 2), p=round(pv, 5),
                   tau=p['em_tau'], logbf=round(c['logbf'], 3), logbf_sellke=round(lb['sellke'], 3), logbf_mix=round(lb['mix'], 3),
                   calib=p['em_calib_c'])
    for c in feas:
        c.setdefault('q', c['pi']); c['eta'] = logit(min(max(c['q'], 1e-9), 1 - 1e-9))


    def getref(c, l):
        cl = c['canon'].get(l)
        if cl is None:
            cl = next((l2 for l2 in c['refs'] if sibkey.get(l2) == sibkey.get(l)), l)
            c['canon'][l] = cl
            if cl not in c['refs']:
                c['refs'][cl] = make_ref(c, cl)
        return c['refs'].get(cl)

    # ---- after the decision: within-candidate EM (descriptive). c's voxels ~ rho p1_c + (1 - rho) p0(.|label): tumour
    # fraction rho_c, voxel responsibilities r_v, and the pure tumour model p1_c used for recall. No decision depends on it.
    for c in feas:
        if not c.get('refs'):
            continue
        lrng = np.random.default_rng([p['em_seed'], c['ci'], 3])
        rc = ratio_for(len(c['idx']))
        sub = lrng.choice(len(c['idx']), int(np.ceil(len(c['idx']) / rc)), replace=False) if rc > 1 else np.arange(len(c['idx']))
        vi = c['idx'][sub]; vl = LAB[vi]
        lp0F = np.full(len(vi), np.nan)
        for l in np.unique(vl).tolist():
            ref = getref(c, l)
            if ref is not None:
                lp0F[vl == l] = ref['lp0all'][BIN[vi[vl == l]]]
        if np.isnan(lp0F).all():
            continue
        lp1, rho_c, rv = fit_p1(BIN[vi], lp0F, None, rc)
        if lp1 is None:
            continue
        c['p1'], c['rho'] = lp1, rho_c
        c['det'][0].update(rho=round(rho_c, 3), mean_r=round(float(np.mean(rv)), 3))

    # ---- delineation from the voxel posterior (9 Oct; replaces region-level recall) --------------------------------------
    # 3:1 rule (9 Oct). The claim boundary is the voxel prior: inside an accepted claim pi_inside = 0.75, outside
    # pi_recall = 0.25. Given c is a tumour (admission already used q_c), a voxel crosses the boundary only on 3:1 image
    # evidence: lambda_v = log p1_c(f_v) - log p0(f_v | label(v)) (smoothed in-plane, 3x3)
    #   trim: inside, sigmoid(logit 0.75 + lambda) < 1/2  <=>  lambda < -log 3 (normal >= 3x more likely)
    #   grow: outside, sigmoid(logit 0.25 + lambda) >= 1/2 <=> lambda >= log 3 (tumour >= 3x more likely)
    # voxel posterior (maps): q_c x sigmoid(logit 0.75 + lambda) inside, q_c x sigmoid(logit 0.25 + lambda) around.
    # rho_c (EM tumour fraction) is descriptive only. An accepted candidate with log BF >= log 3 is never trimmed away
    # entirely (flag voxel_support).
    #   trim (geometry): a connected region failing the rule is removed if it reaches the candidate's border and holds
    #         >= max(trim_min_ml, trim_min_frac x candidate); holes enclosed in 3-D or in-plane are kept; leftover
    #         pieces < trim_min_ml are dropped; voxels referenced only by a last-resort remnant are protected.
    #   grow (geometry): within reach_mm of the kept part, in the lesion host's envelope, unclaimed, connected.
    lp_rec = logit(p['em_pi_recall'])
    lp_in = logit(p['em_pi_inside'])
    near_all = _within(U_B, p['em_reach_mm'], sp).ravel()
    sz2 = [3, 3, 3]; sz2[ax] = 1
    oth_B = ctx['other'][B].ravel()
    min_trim = max(1, int(round(p['em_trim_min_ml'] / ctx['vox_ml'])))

    def lam_of(c, vi):
        vl = LAB[vi]; lv = np.zeros(len(vi))
        for l in np.unique(vl).tolist():
            ref = getref(c, l)
            if ref is not None:
                m_ = vl == l
                lv[m_] = np.clip(c['p1'][BIN[vi[m_]]] - ref['lp0all'][BIN[vi[m_]]], -lclip, lclip)
        return lv

    def smooth_on(vals, vi, lo, shp):
        """In-plane 3x3 mean of vals over the voxel set vi only (normalised convolution) inside the crop [lo, lo + shp)."""
        A = np.zeros(shp); W = np.zeros(shp)
        loc_ = tuple((unr(vi) - lo).T)
        A[loc_] = vals; W[loc_] = 1.0
        A = ndimage.uniform_filter(A, size=sz2); W = ndimage.uniform_filter(W, size=sz2)
        return (A / np.maximum(W, 1e-9))[loc_]

    recall = []
    for c in feas:
        c['keep_idx'] = c['idx']
        if not (_accepted(c, out) and c.get('p1') is not None):
            continue
        co = unr(c['idx'])
        pad_ = np.ceil(p['em_reach_mm'] / sp).astype(int) + 2
        lo = np.maximum(co.min(0) - pad_, 0); hi = np.minimum(co.max(0) + pad_ + 1, np.array(shB)); shp = tuple(hi - lo)
        lin = lam_of(c, c['idx'])
        rh = c['rho']
        # extent is decided GIVEN that c is a tumour (admission already used q_c): r_v, not q_c x r_v
        Pin = 1 / (1 + np.exp(-np.clip(lp_in + lin, -50, 50)))       # 3:1 rule: trimmed iff lambda < -log 3 (smoothed)
        Ps = smooth_on(Pin, c['idx'], lo, shp)
        # voxels compared only with a last-resort remnant (their organ is mostly claimed, so the remnant is likely the same
        # tumour) cannot be judged dark: protected from trimming, and the candidate is flagged
        vl_ = LAB[c['idx']]; prot = np.zeros(len(c['idx']), bool)
        for l in np.unique(vl_).tolist():
            ref = getref(c, l)
            if ref is not None and ref['tier'] == 1:
                prot[vl_ == l] = True
        if prot.any():
            Ps = np.where(prot, 1.0, Ps)
            c['det'][0].update(trim_protected_ml=round(float(prot.sum()) * ctx['vox_ml'], 2))
        Mc = np.zeros(shp, bool); Mc[tuple((co - lo).T)] = True
        D = np.zeros(shp, bool); D[tuple((co[Ps < 0.5] - lo).T)] = True
        border = Mc & ~ndimage.binary_erosion(Mc, structure=ST)
        labd, nd = ndimage.label(D, structure=ST)
        rem = np.zeros(shp, bool)
        need = max(min_trim, int(round(p['em_trim_min_frac'] * len(c['idx']))))
        if nd:
            szd = np.bincount(labd.ravel())
            for k in range(1, nd + 1):
                if szd[k] >= need and (border & (labd == k)).any():
                    rem |= labd == k
        if rem.any():
            keep = Mc & ~rem
            # dark tissue enclosed by what remains (necrosis, a cavity) stays: holes filled in 3-D and slice by slice in
            # the acquisition plane (a cavity open to the next slice is still enclosed in-plane)
            st2 = np.zeros((3, 3, 3), bool)
            for d_ in range(3):
                if d_ != ax:
                    for o_ in (0, 2):
                        j_ = [1, 1, 1]; j_[d_] = o_; st2[tuple(j_)] = True
            st2[1, 1, 1] = True
            pw = [(0, 0)] * 3; pw[ax] = (1, 1)
            k2 = ndimage.binary_fill_holes(np.pad(keep, pw), structure=st2)
            cut_ = [slice(None)] * 3; cut_[ax] = slice(1, -1)
            keep = (ndimage.binary_fill_holes(keep) | k2[tuple(cut_)]) & Mc
            # remaining pieces smaller than the trim size are speckle, not a lesion (same size rule in both directions)
            labk, nk = ndimage.label(keep, structure=ST)
            if nk:
                szk = np.bincount(labk.ravel())
                keep &= np.isin(labk, [k for k in range(1, nk + 1) if szk[k] >= min_trim])
            if not keep.any() and c['logbf'] >= np.log(3):
                # strong lesion-level evidence but no voxel stands out on its own (a subtle diffuse change): keep the
                # candidate as claimed and flag it; only weakly supported candidates can be trimmed away entirely
                keep = Mc; c['det'][0].update(voxel_support='none')
            kidx = np.ravel_multi_index(tuple((np.argwhere(keep) + lo).T), shB)
            c['keep_idx'] = kidx
        else:
            keep = Mc
        c['det'][0].update(trim_ml=round((len(c['idx']) - len(c['keep_idx'])) * ctx['vox_ml'], 2))
        # grow
        les = out[c['lesion']]; h = hosts.get(les['host'])
        if h is None or not h['env'].any():
            continue
        Kb = np.zeros(nB, bool); Kb[c['keep_idx']] = True
        near = _within(Kb.reshape(shB), p['em_reach_mm'], sp).ravel()
        env = h['env'][B].ravel(); Hm = h['H'][B].ravel() if h['H'].any() else np.zeros(nB, bool)
        aidx = np.flatnonzero(near & env & ~(oth_B & ~Hm) & ~claims_flat)
        if not len(aidx):
            continue
        Pout = 1 / (1 + np.exp(-np.clip(lp_rec + lam_of(c, aidx), -50, 50)))     # given c is a tumour
        ac = unr(aidx)
        lo2 = np.maximum(np.minimum(ac.min(0), co.min(0)) - 1, 0); hi2 = np.minimum(np.maximum(ac.max(0), co.max(0)) + 2, np.array(shB))
        shp2 = tuple(hi2 - lo2)
        Pos = smooth_on(Pout, aidx, lo2, shp2)
        G_ = np.zeros(shp2, bool); G_[tuple((ac[Pos >= 0.5] - lo2).T)] = True
        Kc = np.zeros(shp2, bool); Kc[tuple((unr(c['keep_idx']) - lo2).T)] = True
        labg, ng = ndimage.label(G_ | Kc, structure=ST)
        core = np.unique(labg[Kc]); core = core[core > 0]
        grown = np.isin(labg, core) & G_
        if grown.any():
            gidx = np.ravel_multi_index(tuple((np.argwhere(grown) + lo2).T), shB)
            sel_ = np.isin(aidx, gidx)
            recall.append(dict(kind='grow', owner=c['ci'], pi=p['em_pi_recall'], idx=gidx, ml=round(len(gidx) * ctx['vox_ml'], 2),
                               q=float(np.mean(Pos[sel_])), logbf=0.0, powered=True, det=[]))
            c['det'][0].update(grow_ml=round(len(gidx) * ctx['vox_ml'], 2))

    # ---- report + voxel maps -------------------------------------------------------------------------------------------
    report.update(model='candidate', calib=p['em_calib_c'], tau=p['em_tau'], iterations=1, converged=True,
                  class_names={str(k): v for k, v in names.items()}, compare='label+siblings', kb_groups={})
    report['history'] = []
    try:
        # owner of each voxel in the map region = nearest feasible candidate (higher q wins on overlap)
        reg = near_all.copy()
        own = np.zeros(nB, np.int32)
        for j, c in enumerate(sorted(feas, key=lambda x: x['q'])):
            own[c['idx']] = feas.index(c) + 1
            reg[c['idx']] = True
        reg |= refmap > 0
        _, ind = ndimage.distance_transform_edt(own.reshape(shB) == 0, sampling=sp, return_indices=True)
        own_n = own.reshape(shB)[tuple(ind)].ravel(); del ind
        for r_ in recall:                                   # recall voxels belong to the candidate that grew them
            oc = next((i for i, c in enumerate(feas) if c['ci'] == r_['owner']), None)
            if oc is not None:
                own_n[r_['idx']] = oc + 1
        for i, c in enumerate(feas):
            c['map_i'] = i + 1
        ridx = np.flatnonzero(reg)
        lam_raw = np.full(len(ridx), np.nan); rr = np.full(len(ridx), np.nan)
        inc = np.zeros(nB, bool)
        for c in feas:
            inc[c['idx']] = True
        post = np.full(len(ridx), np.nan)
        for ci_, c in enumerate(feas):
            if c.get('p1') is None:
                continue
            sel = (own_n[ridx] == ci_ + 1) & (near_all[ridx] | inc[ridx])
            if not sel.any():
                continue
            vi = ridx[sel]; vl = LAB[vi]
            lv = np.full(len(vi), np.nan)
            for l in np.unique(vl).tolist():
                ref = getref(c, l)
                if ref is not None:
                    m_ = vl == l
                    lv[m_] = c['p1'][BIN[vi[m_]]] - ref['lp0all'][BIN[vi[m_]]]
            rh = c['rho']
            lv0 = np.nan_to_num(lv)
            rv = 1 / (1 + np.exp(-np.clip(np.log(rh / (1 - rh)) + lv0, -50, 50)))
            lam_raw[sel] = lv; rr[sel] = rv
            inside = np.isin(vi, c['idx'])               # inside: q_c x r_v; near (outside): q_c x sigmoid(logit pi_recall + lambda)
            pv_ = np.where(inside, c['q'] / (1 + np.exp(-np.clip(lp_in + lv0, -50, 50))),
                           c['q'] / (1 + np.exp(-np.clip(lp_rec + lv0, -50, 50))))
            cur = post[sel]
            post[sel] = np.where(np.isnan(cur), pv_, np.fmax(cur, pv_))
        ok = ~np.isnan(lam_raw)
        lam8 = np.full(len(ridx), -128, np.int8)
        lam8[ok] = np.round(25 * np.clip(lam_raw[ok], -lclip, lclip)).astype(np.int8)
        r8 = np.full(len(ridx), 255, np.uint8); r8[ok] = np.round(200 * rr[ok]).astype(np.uint8)
        okp = ~np.isnan(post)
        p8 = np.full(len(ridx), 255, np.uint8); p8[okp] = np.round(200 * np.clip(post[okp], 0, 1)).astype(np.uint8)
        qf = np.full(nB, -1.0, np.float32)
        for c in feas + recall:
            np.maximum.at(qf, c['idx'], np.float32(c['q']))
        qv = qf[ridx]
        q8 = np.where(qv < 0, 255, np.round(200 * np.clip(qv, 0, 1))).astype(np.uint8)
        rm = reg.reshape(shB)
        bb_r = ndimage.find_objects(rm.astype(np.int8))[0]
        shR = tuple(b.stop - b.start for b in bb_r)
        loc = unr(ridx) - np.array([b.start for b in bb_r])

        def _arr(v, fill, dt):
            a2 = np.full(shR, fill, dt); a2[tuple(loc.T)] = v; return a2
        rec_all = np.concatenate([r_['idx'] for r_ in recall]) if recall else np.zeros(0, np.int64)
        own_v = np.where(near_all[ridx] | inc[ridx] | np.isin(ridx, rec_all), own_n[ridx], 0).astype(np.int16)
        ed = np.zeros(nB, np.uint8)                        # 1 = trimmed from an accepted candidate, 2 = grown
        for c in feas:
            if len(c.get('keep_idx', c['idx'])) < len(c['idx']):
                ed[np.setdiff1d(c['idx'], c['keep_idx'])] = 1
        for r_ in recall:
            ed[r_['idx']] = 2
        edit_v = ed[ridx]
        refc = {}
        for i, c in enumerate(feas):                        # reference voxels of each candidate (positions in the map box)
            if c.get('ref_idx'):
                ii = np.concatenate(c['ref_idx']); tt = np.concatenate(c['ref_tier'])
                pos_ = np.searchsorted(ridx, ii); ok_ = pos_ < len(ridx); ok_[ok_] = ridx[pos_[ok_]] == ii[ok_]
                lc = loc[pos_[ok_]]
                refc[f'refc_{i + 1}'] = np.ravel_multi_index(tuple(lc.T), shR).astype(np.int32)
                refc[f'reft_{i + 1}'] = tt[ok_]
        report['_maps'] = dict(**refc,origin=np.array([b.start + q.start for b, q in zip(B, bb_r)], np.int32),
                               lam=_arr(lam8, -128, np.int8), resp=_arr(r8, 255, np.uint8), q=_arr(q8, 255, np.uint8),
                               cls=_arr(LAB[ridx].astype(np.int32), -1, np.int32), ref=_arr(refmap[ridx], 0, np.uint8),
                               post=_arr(p8, 255, np.uint8), own=_arr(own_v, 0, np.int16), edit=_arr(edit_v, 0, np.uint8),
                               scale=np.array([25.0, 200.0, 200.0], np.float32))
    except Exception as e:
        report['maps_error'] = repr(e)[:300]
    for c in feas:
        for k_ in ('D', 'refs', 'models', 'clo', 'bsz', 'canon', 'p1', 'rho', 'ref_idx', 'ref_tier'):
            c.pop(k_, None)
    return recall
