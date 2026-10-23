"""Spatial tumour prior (main.tex Eq. spatial_prior): pi^L_v = sum_{h in S_L} P(h|L) sigma(eta . psi_v(h)).

psi_v(h) is a fixed basis of four inputs:
  q      stage-1 extent marginal q^L_v (1 on the anchors, 0 outside the union of candidate extents)
  sd     signed distance (mm) to the chosen extent xi_L (negative inside)
  terr   territory of v relative to host h: in h / in another organ / in no segmented organ; kb = h is a knowledge-base region
  depth  depth (mm) of v below the surface of the organ it lies in
Knots (design values, disclosed): sd hinges at -10, -5, 0, 5, 10, 20, 40 mm (sd clipped to [-30, 60]); depth hinges at 5, 15 mm
(depth clipped to [0, 30]). eta is fitted on expert data only (fit.py); with eta = None the prior is undefined (no fallback)."""
import numpy as np

SD_KNOTS = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0)
DEP_KNOTS = (5.0, 15.0)
NAMES = (['1', 'q', 'q2', 'sd'] + [f'sd>{k:g}' for k in SD_KNOTS] + ['in_h', 'in_o', 'in_h_kb']
         + ['dep_h'] + [f'dep_h>{k:g}' for k in DEP_KNOTS] + ['dep_o'] + [f'dep_o>{k:g}' for k in DEP_KNOTS]
         + ['in_h*sdpos', 'in_o*sdpos', 'in_h_kb*sdpos'])
P = len(NAMES)


def basis(q, sd, in_h, in_o, kb_h, dep_h, dep_o):
    """all inputs are 1-D arrays of equal length (kb_h a scalar or array). Returns (n, P) float32 design matrix."""
    q = np.asarray(q, np.float32); n = q.size
    sd = np.clip(np.asarray(sd, np.float32), -30.0, 60.0)
    in_h = np.asarray(in_h, np.float32); in_o = np.asarray(in_o, np.float32) * (1.0 - in_h)
    kb = np.broadcast_to(np.asarray(kb_h, np.float32), (n,))
    dh = np.clip(np.asarray(dep_h, np.float32), 0.0, 30.0) * in_h
    do = np.clip(np.asarray(dep_o, np.float32), 0.0, 30.0) * in_o
    sdp = np.clip(sd, 0.0, 60.0)
    cols = [np.ones(n, np.float32), q, q * q, sd] + [np.maximum(sd - k, 0.0) for k in SD_KNOTS]
    cols += [in_h, in_o, in_h * kb, dh] + [np.maximum(dh - k, 0.0) * in_h for k in DEP_KNOTS]
    cols += [do] + [np.maximum(do - k, 0.0) * in_o for k in DEP_KNOTS]
    cols += [in_h * sdp * (1.0 - kb), in_o * sdp, in_h * kb * sdp]
    X = np.stack(cols, 1).astype(np.float32)
    assert X.shape[1] == P
    return X


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class SpatialPrior:
    def __init__(self, eta):
        self.eta = np.asarray(eta, np.float64)
        assert self.eta.size == P, (self.eta.size, P)

    def prob(self, Xb):
        return sigmoid(Xb.astype(np.float64) @ self.eta)


def host_inputs(theta, kb, organs, depth, h, shape):
    """per-voxel territory inputs for host h on a window.
    theta: dict key -> bool array (territory; for an unsegmented host its knowledge-base region); kb: dict key -> bool;
    organs: list of keys of SEGMENTED organs (territories that count as 'another organ'); depth: dict key -> float array
    (inner depth in mm, segmented organs only). Returns in_h, in_o, kb_h, dep_h, dep_o (arrays of the window shape)."""
    th = theta.get(h)
    in_h = th if th is not None else np.zeros(shape, bool)
    in_o = np.zeros(shape, bool); dep_o = np.zeros(shape, np.float32)
    for o in organs:
        if o == h or theta.get(o) is None:
            continue
        in_o |= theta[o]
        dep_o = np.maximum(dep_o, depth[o])
    dep_h = depth[h] if (h in depth and not kb.get(h, False)) else np.zeros(shape, np.float32)
    return in_h, in_o & ~in_h, float(bool(kb.get(h, False))), dep_h, dep_o


def lesion_prior(SP, q, sd, PhL, theta, kb, organs, depth, M=None):
    """pi^L on a window (mixture over the credible hosts). PhL: dict host -> probability, renormalised over the credible set by the
    caller. M: optional bool mask where pi is needed (elsewhere 0). Hosts with no territory in the window still contribute (in_h = 0)."""
    shape = q.shape; M = np.ones(shape, bool) if M is None else M
    out = np.zeros(int(M.sum()), np.float64)
    if not PhL:   # host unknown: the prior has no host term; treated as a host with no territory in the window
        PhL = {None: 1.0}
    for h, w in PhL.items():
        in_h, in_o, kb_h, dep_h, dep_o = host_inputs(theta, kb, organs, depth, h, shape)
        Xb = basis(q[M], sd[M], in_h[M], in_o[M], kb_h, dep_h[M], dep_o[M])
        out += w * SP.prob(Xb)
    pi = np.zeros(shape, np.float32); pi[M] = out
    return pi
