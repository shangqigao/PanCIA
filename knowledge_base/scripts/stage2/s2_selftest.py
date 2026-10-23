"""Stage 2, step 4: synthetic phantom tests of the VB model (no patient data; runs in a few minutes).

Phantom (1 x 1 x 3 mm grid): generic tissue (fat-like), an organ, and inside the organ a tumour with an enhancing rim and a necrotic
core. Checks:
  T1 the tumour is recovered from a partial anchor (Dice of {p >= 1/2} with the truth >= 0.85) and recall extends beyond the anchor;
  T2 the sparse Dirichlet empties at least one unused component (weight < 0.01); 2-3 components remain (rim, necrosis, and
     possibly partial-volume boundary voxels);
  T3 a false anchor placed on normal organ tissue gives an approximate Bayes factor < 1 (background preferred);
  T4 the true anchor gives an approximate Bayes factor > 1;
  T5 coupling: posterior E[omega] is lower on edges across the tumour boundary than inside the tumour.
Usage: python3 s2_selftest.py"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))      # PanCIA/knowledge_base
from pancia_kb.stage2 import vb, features as F   # noqa: E402

rng = np.random.default_rng(0)
sp = np.array([1.0, 1.0, 3.0]); shape = (80, 80, 30)
g = np.indices(shape).astype(np.float32) * sp[:, None, None, None]
c = np.array([40.0, 40.0, 45.0])
r_org = np.sqrt(((g - np.array([40.0, 40.0, 45.0])[:, None, None, None]) ** 2).sum(0))
organ = r_org <= 32
r_t = np.sqrt(((g - np.array([44.0, 40.0, 45.0])[:, None, None, None]) ** 2).sum(0))
tum = r_t <= 12; necro = r_t <= 6
X = np.full(shape, -100.0, np.float32); X[organ] = 60.0; X[tum] = 130.0; X[necro] = 20.0
X += rng.normal(0, 12, shape).astype(np.float32)
thick, ax = F.acq_axis(sp, 2)
Y4 = F.image_features(X, sp, thick, ax)

ROI = F.dil(tum, sp, 15.0) | organ
M = ROI
Y = Y4[:, M].T


def priors():
    D = F.D
    tum_p = vb.NW.from_moments(np.r_[80.0, 20.0], np.diag([80.0, 20.0]) ** 2)      # broad stand-in priors (no fitted values)
    org_p = vb.NW.from_moments(np.r_[60.0, 10.0], np.diag([40.0, 10.0]) ** 2)
    gen_p = vb.NW.from_moments(np.zeros(D), np.eye(D) * 300.0 ** 2)
    return tum_p, [org_p, gen_p, gen_p]


def run(anchor, extent_big, tumour=True, region=None):
    region = M if region is None else region
    q = np.where(anchor, 1.0, np.where(extent_big, 0.5, 0.0))
    sd = F.signed_distance(anchor | extent_big, sp)
    pi = 0.02 + 0.9 * q                                                                   # stand-in prior (no fitted eta)
    sel = region
    types = organ[sel].astype(int)                     # type 0: no organ; type 1: in organ
    allowed = np.array([[False, True, True], [True, True, True]])
    tp, bps = priors()
    ed = vb.build_edges(sel, sp)
    m = vb.Model(Y4[:, sel].T, pi[sel], types, allowed, tp, bps, K=4, alpha0=0.25, nu=5.0, a_w=2.0, b_w=1.0, edges=ed, tumour=tumour)
    m.init(q[sel]).run(max_iter=150, parity=vb.parity(sel))
    return m, sel


t0 = time.time(); ok = True
dT = F.inner_depth(tum, sp)
anchor = tum & (dT >= 4.0)                                          # partial core
ext = F.dil(anchor, sp, 3.0)
m, sel = run(anchor, ext)
p = np.zeros(shape); p[sel] = m.p(); Z = p >= 0.5
dice = 2 * (Z & tum).sum() / (Z.sum() + tum.sum())
rec = ((Z & tum & ~ext).sum()) / max((tum & ~ext).sum(), 1)
print(f'T1 dice {dice:.3f} recall beyond extent {rec:.3f} iters {m.iters} converged {m.converged}'); ok &= dice >= 0.9 and rec >= 0.5 and m.converged
cs = m.comp_summary(); print('T2 components', cs); ok &= 2 <= cs['k_eff'] <= 3 and min(cs['rho']) < 0.01

def lbf(anchor_, region):
    """approximate log BF (refitted, matched flexibility) on N = anchor (+) 15 mm within the region, as run.py"""
    q = np.where(anchor_, 1.0, np.where(F.dil(anchor_, sp, 3.0), 0.5, 0.0)); pi = 0.02 + 0.9 * q
    N = region
    tp, bps = priors(); allowed = np.array([[False, True, True], [True, True, True]])
    v, _, _ = vb.log_bayes_factor(Y4[:, N].T, pi[N], organ[N].astype(int), allowed, tp, bps, q[N], K=4, alpha0=0.25, nu=5.0,
                                  a_w=2.0, b_w=1.0, edges=vb.build_edges(N, sp), parity=vb.parity(N))
    return v


lb_t = lbf(anchor, F.dil(anchor, sp, 15.0) & M); print(f'T4 true anchor log BF {lb_t:.1f}'); ok &= lb_t > 0
fa = organ & (np.sqrt(((g - np.array([18.0, 30.0, 45.0])[:, None, None, None]) ** 2).sum(0)) <= 5)
lb_f = lbf(fa, F.dil(fa, sp, 15.0) & M); print(f'T3 false anchor log BF {lb_f:.1f}'); ok &= lb_f < 0
I, J, S = vb.build_edges(sel, sp); idx = np.flatnonzero(sel.ravel())
ti = tum.ravel()[idx]
_, _, Ew = m._coupling(m.p())
across = ti[I] != ti[J]; inside = ti[I] & ti[J]
print(f'T5 E[omega] across boundary {Ew[across].mean():.3f} inside tumour {Ew[inside].mean():.3f}'); ok &= Ew[across].mean() < Ew[inside].mean()
print('ALL PASS' if ok else 'SOME FAILED', f'({time.time() - t0:.0f} s)')
sys.exit(0 if ok else 1)
