"""Host reasoning v8.1 (host step 1, user 20 Oct 2026): no 10 mm candidacy gate, no 'none' candidate, distance decides.
Candidates = every organ in view (segmented, or a KB region when unsegmented); P(h | L) = softmax(beta . phi_h) over candidates only;
credible set S_alpha = smallest set with probability >= 1 - alpha. If a lesion has no candidate in view, the host is unknown (the
lesion is then 'uncertain'; no pseudo-host). phi = grounding_g1.compat_features + ldist = log(1 + d), d = distance in mm from the
anchor (the part of the lesion nearest the organ) to the organ territory, 0 if they touch, capped at D_CAP. beta fitted on expert data
only (scripts/tb_fit3.py on ledger/v8/testbed_host3). grounding_g1 / grounding_g2 are unchanged."""
import numpy as np
from . import grounding_g1 as G1
from .grounding_g2 import credible_set  # noqa: F401  (same rule)
D_CAP = 60.0
FEATS = list(G1.FEATS) + ['ldist']


def distance_mm(da, theta):
    """da = euclidean distance map to the anchor (G1.edt(~a, sp)); theta = organ territory (bool). Capped at D_CAP."""
    if theta is None or not theta.any():
        return D_CAP
    return float(min(D_CAP, float(da[theta].min())))


def features(a, core, theta, mu, sp, da=None, kb=False, others=None):
    phi = G1.compat_features(a, core, theta, mu, sp, others=others, kb=kb)
    if da is None:
        da = G1.edt(~a, sp)
    d = distance_mm(da, theta)
    phi['dist_mm'] = d; phi['ldist'] = float(np.log1p(d))
    return phi


def far_features(mu=1.0, kb=False, d=D_CAP):
    """an organ in view but outside the analysis window: no overlap, distance >= the window size"""
    return dict(o_core=0.0, o_theta=0.0, s_band=0.0, inside=0.0, straddle=0.0, exophytic=1.0, mu=float(mu), kb=float(bool(kb)),
                dist_mm=float(d), ldist=float(np.log1p(d)))


def score(phi, beta):
    return float(sum(beta.get(k, 0.0) * phi[k] for k in FEATS))


def host_posterior(cands, beta):
    """cands: dict h -> phi (no 'none'). Returns dict h -> P(h | L); empty dict when there is no candidate."""
    hs = list(cands)
    if not hs:
        return {}
    z = np.array([score(cands[h], beta) for h in hs], float); z = np.exp(z - z.max()); z /= z.sum()
    return {h: float(v) for h, v in zip(hs, z)}
