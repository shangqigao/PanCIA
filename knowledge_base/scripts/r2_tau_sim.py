"""Derive tau (nats/voxel) for R2 candidate mode a priori, on synthetic data only (no scans).

Definition: a tumour differs from its tissue by one normal-tissue SD in intensity. Normal tissue = Gaussian texture
(white noise smoothed in-plane, sd 1); a spherical lesion with the same texture shifted by +shift SD. The lesion and its
surroundings go through the exact R2 candidate statistic (pancia_kb.r2_em._heldout_stat): rank-binned features (raw,
3x3 in-plane mean and SD), reference = normal tissue within 20-60 mm outside lesion + 2 mm, eroded in-plane, 15 mm
checkerboard folds; lesion-local checkerboard folds; thinning to <= em_cap points; null patches from the reference with
their voxels removed. delta = s - median(s0). tau = median delta over lesion sizes and seeds at shift = 1 SD.
Usage: python r2_tau_sim.py [M_null]"""
import sys, os, json
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pancia_kb.r2_em import EMP, _heldout_stat, _smooth_prob

p = dict(EMP); G = p['em_G']; G3 = G ** 3; eps = p['em_eps']; kappa = p['em_kappa']; lclip = p['em_lclip']
SP = np.array([0.8, 0.8, 2.5]); AX = 2
M = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RATIOS = (1, 4, 16, 64, 256, 1024)


def parity(co, bsz):
    return (np.add.reduce([co[:, a] // bsz[a] for a in range(3)]) % 2).astype(np.int8)


def local_blocks(co):
    minpl = np.array([1 if a == AX else 3 for a in range(3)]); mm = p['em_block_mm']
    while True:
        bsz = np.maximum(minpl, np.round(mm / SP).astype(int)); par = parity(co, bsz); n1 = int(par.sum())
        if min(n1, len(par) - n1) >= p['em_min_own']:
            return bsz, par
        if np.all(bsz <= minpl):
            return None, None
        mm /= 2


def one(ml, shift, seed, frac=1.0):
    rng = np.random.default_rng(seed)
    r_mm = (3 * ml * 1000 / (4 * np.pi)) ** (1 / 3)
    half = r_mm + p['em_ref_max_mm'] + 5
    sh = np.ceil(2 * half / SP).astype(int)
    noise = ndimage.gaussian_filter(rng.normal(size=sh), sigma=(1.0, 1.0, 0.0))
    X = (noise - noise.mean()) / noise.std()
    grid = np.stack(np.meshgrid(*[(np.arange(n) - n / 2) * s for n, s in zip(sh, SP)], indexing='ij'), -1)
    dist_c = np.linalg.norm(grid, axis=-1)
    L = dist_c <= r_mm
    if frac < 1.0:                       # partial: only a sub-sphere of the claim is tumour
        T = np.linalg.norm(grid - np.array([r_mm * (1 - frac ** (1 / 3)), 0, 0]), axis=-1) <= r_mm * frac ** (1 / 3)
        T &= L
    else:
        T = L
    X = X + shift * T
    size = [3, 3, 3]; size[AX] = 1
    m = ndimage.uniform_filter(X, size=size)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(X * X, size=size) - m * m, 0))
    samp = rng.choice(X.size, min(X.size, 200000), replace=False)
    BIN = np.zeros(sh, np.int64)
    for A in (X, m, sd):
        edges = np.quantile(A.ravel()[samp], np.arange(1, G) / G)
        BIN = BIN * G + np.searchsorted(edges, A, side='right')
    BIN = BIN.ravel()
    # reference: normal tissue outside lesion + margin, eroded in-plane; radius grown from 20 mm
    dL = ndimage.distance_transform_edt(~L, sampling=SP)
    ER = np.zeros((3, 3, 3), bool); e = [slice(None)] * 3; e[AX] = 1; ER[tuple(e)] = True
    normal = ndimage.binary_erosion((dL > p['em_margin_mm']) & (dL <= p['em_ref_max_mm']), structure=ER)
    nidx = np.flatnonzero(normal); nco = np.stack(np.unravel_index(nidx, sh), -1)
    bsg = np.maximum(1, np.round(p['em_block_mm'] / SP).astype(int)); nf = parity(nco, bsg)
    dn = dL.ravel()[nidx]
    r = p['em_window_mm']
    while True:
        sel = dn <= r
        if min(int(nf[sel].sum()), int(sel.sum() - nf[sel].sum())) >= p['em_min_ref'] or r >= p['em_ref_max_mm']:
            break
        r += p['em_ref_step_mm']
    ridx, rf, rco = nidx[sel], nf[sel], nco[sel]
    cidx = np.flatnonzero(L.ravel()); cco = np.stack(np.unravel_index(cidx, sh), -1)
    bsz, par = local_blocks(cco)
    if bsz is None:
        return None
    rc = next((rr for rr in RATIOS if len(cidx) / rr <= p['em_cap']), RATIOS[-1])
    si = rng.choice(len(ridx), max(1, len(ridx) // rc), replace=False) if rc > 1 else np.arange(len(ridx))
    pts = ridx[si]; pco = rco[si]; pf_all = rf[si]
    H = [rc * np.bincount(BIN[pts[pf_all == k]], minlength=G3).astype(float) for k in (0, 1)]   # thinned reference
    p0 = [_smooth_prob(H[k], G, H[k].sum()) for k in (0, 1)]
    sub = np.sort(rng.choice(len(cidx), int(np.ceil(len(cidx) / rc)), replace=False)) if rc > 1 else np.arange(len(cidx))
    s = _heldout_stat(BIN[cidx[sub]], par[sub], p0, rc, G, eps, kappa, lclip)
    tree = cKDTree(pco * SP)
    s0 = []
    for _ in range(M):
        k_ = max(1, min(int(round(len(cidx) / rc)), len(pts) // 2))
        _, ii = tree.query(tree.data[rng.integers(len(pts))], k=k_)
        ii = np.atleast_1d(ii)
        pp = parity(pco[ii], bsz)
        if min(int(pp.sum()), len(pp) - int(pp.sum())) < max(2, p['em_min_own'] // rc):
            continue
        pf = pf_all[ii]
        adj = []
        for k in (0, 1):
            h = np.maximum(H[k] - rc * np.bincount(BIN[pts[ii][pf == k]], minlength=G3), 0)
            adj.append(_smooth_prob(h, G, h.sum()))
        v = _heldout_stat(BIN[pts[ii]], pp, adj, rc, G, eps, kappa, lclip)
        if v is not None:
            s0.append(v)
    s0 = np.array(s0)
    return dict(ml=ml, shift=shift, frac=frac, seed=seed, s=s, delta=s - float(np.median(s0)), sigma=float(s0.std()),
                n0=len(s0), ref_r=r, thin=rc)


if __name__ == '__main__':
    rows = []
    for shift in (0.0, 0.5, 1.0, 2.0):
        for ml in (1.0, 5.0, 20.0, 80.0):
            for seed in range(3):
                rr = one(ml, shift, 1000 * seed + int(ml * 10) + int(shift * 7))
                if rr:
                    rows.append(rr); print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rr.items()}), flush=True)
    d1 = [r_['delta'] for r_ in rows if r_['shift'] == 1.0]
    print('TAU (median delta at 1 SD):', round(float(np.median(d1)), 4))
    for sh_ in (0.0, 0.5, 1.0, 2.0):
        dd = [r_['delta'] for r_ in rows if r_['shift'] == sh_]
        print('shift', sh_, 'delta median', round(float(np.median(dd)), 4), 'range', round(min(dd), 4), round(max(dd), 4))
