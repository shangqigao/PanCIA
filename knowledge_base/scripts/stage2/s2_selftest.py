"""Stage 2, step 4: synthetic phantom tests of the VB model (no patient data; runs in a few minutes). Priors are within-scan
(vb.class_priors), as in run.py.

Phantom (1 x 1 x 3 mm grid): generic tissue (fat-like), an organ, and inside the organ a tumour with an enhancing rim and a necrotic
core. Checks:
  T1 the tumour is recovered from a partial anchor (Dice of {p >= 1/2} with the truth >= 0.85) and recall extends beyond the anchor;
  T2 components: on the rim + necrosis tumour, one component near the rim's intensity (130) and one near the necrosis (20) carry
     weight >= 0.05 (further components model partial-volume boundaries); the interior (>= 3 mm deep) of a homogeneous tumour is one
     component by the lower bound (fit.fit_tumour_components, as the tail study). (Since 26 Oct the features include first
     differences and a Laplacian, so a whole homogeneous tumour with fixed nu = 5 also uses components for its edges; with the tail
     prior, T6, one component is emptied.)
  T3 a false anchor placed on normal organ tissue gives an approximate Bayes factor < 1 (background preferred);
  T4 the true anchor gives an approximate Bayes factor > 1;
  T5 coupling: posterior E[omega] is lower on edges across the tumour boundary than inside the tumour;
  T6 tail-weight prior (MAP nu, main.tex Eq. nu_prior; stand-in cell priors): Dice >= 0.9 on both phantoms, on the homogeneous
     tumour the sparse Dirichlet empties a component (weight < 0.01), and the Bayes factor keeps its signs (true > 1, false < 1);
  T7 MAP nu on Student-t samples (nu = 4): with 5000 samples the data win (within 25% of 4); with 50 samples the estimate is pulled
     towards the prior (closer to its mean on log 1/nu than the maximum-likelihood nu).
Usage: python3 s2_selftest.py"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))      # PanCIA/knowledge_base
from pancia_kb.stage2 import vb, features as F, fit as FT   # noqa: E402

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


def priors(Ysel, qsel, types, allowed):
    """within-scan class priors (vb.class_priors) on the modelled voxels, as run.py"""
    return vb.class_priors(Ysel, qsel, allowed[types].astype(float))


def run(anchor, extent_big, tumour=True, region=None):
    region = M if region is None else region
    q = np.where(anchor, 1.0, np.where(extent_big, 0.5, 0.0))
    sd = F.signed_distance(anchor | extent_big, sp)
    pi = 0.02 + 0.9 * q                                                                   # stand-in prior (no fitted eta)
    sel = region
    types = organ[sel].astype(int)                     # type 0: no organ; type 1: in organ
    allowed = np.array([[False, True, True], [True, True, True]])
    tp, bps = priors(Y4[:, sel].T, q[sel], types, allowed)
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
cs = m.comp_summary(); print('T2 components (rim + necrosis)', cs)
big = [mu for mu, w in zip(cs['means'], cs['rho']) if w >= 0.05]
ok &= any(abs(mu[0] - 130.0) <= 20.0 for mu in big) and any(abs(mu[0] - 20.0) <= 20.0 for mu in big)
# homogeneous tumour (no necrosis): an unused component must be emptied
Xh = np.full(shape, -100.0, np.float32); Xh[organ] = 60.0; Xh[tum] = 130.0
Xh += np.random.default_rng(1).normal(0, 12, shape).astype(np.float32)
Yh = F.image_features(Xh, sp, thick, ax)
qh = np.where(anchor, 1.0, np.where(ext, 0.5, 0.0)); pih = 0.02 + 0.9 * qh; th_ = organ[sel].astype(int)
alh = np.array([[False, True, True], [True, True, True]]); tph, bph = priors(Yh[:, sel].T, qh[sel], th_, alh)
mh = vb.Model(Yh[:, sel].T, pih[sel], th_, alh, tph, bph, K=4, alpha0=0.25, nu=5.0, a_w=2.0, b_w=1.0, edges=vb.build_edges(sel, sp))
mh.init(qh[sel]).run(max_iter=150, parity=vb.parity(sel))
csh = mh.comp_summary(); print('T2 components (homogeneous, whole tumour, fixed nu; report only)', csh)
rint = FT.fit_tumour_components(Yh[:, tum & (dT >= 3.0)].T)
print('T2 homogeneous interior: K by lower bound', rint['K'], rint['elbos']); ok &= rint['K'] == 1

def lbf(anchor_, region):
    """approximate log BF (refitted, matched flexibility) on N = anchor (+) 15 mm within the region, as run.py"""
    q = np.where(anchor_, 1.0, np.where(F.dil(anchor_, sp, 3.0), 0.5, 0.0)); pi = 0.02 + 0.9 * q
    N = region
    allowed = np.array([[False, True, True], [True, True, True]]); tp, bps = priors(Y4[:, N].T, q[N], organ[N].astype(int), allowed)
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
# T6: tail-weight prior (stand-in cells: tumour nu 20, organ nu 30, generic nu 22 on log 1/nu, sds as fitted)
NP6 = [[(1.0, np.log(1 / 20.0), 0.95)]] * 4 + [[(1.0, np.log(1 / 30.0), 1.1)]] + [[(1.0, np.log(1 / 22.0), 1.25)]] * 2
for nm_, Yf_, chk in (('rim + necrosis', Y4, False), ('homogeneous', Yh, True)):
    q6 = np.where(anchor, 1.0, np.where(ext, 0.5, 0.0)); pi6 = 0.02 + 0.9 * q6; t6 = organ[sel].astype(int)
    al6 = np.array([[False, True, True], [True, True, True]]); tp6, bp6 = priors(Yf_[:, sel].T, q6[sel], t6, al6)
    m6 = vb.Model(Yf_[:, sel].T, pi6[sel], t6, al6, tp6, bp6, K=4, alpha0=0.25, nu=5.0, a_w=2.0, b_w=1.0, edges=vb.build_edges(sel, sp),
                  nu_prior=NP6)
    m6.init(q6[sel]).run(max_iter=200, parity=vb.parity(sel))
    p6 = np.zeros(shape); p6[sel] = m6.p(); Z6 = p6 >= 0.5; d6 = 2 * (Z6 & tum).sum() / (Z6.sum() + tum.sum()); c6 = m6.comp_summary()
    print(f'T6 nu prior, {nm_}: dice {d6:.3f} rho {c6["rho"]} nu {c6["nu"]} iters {m6.iters}')
    ok &= d6 >= 0.9 and (min(c6['rho']) < 0.01 if chk else True)
# T6b: Bayes factor with the tail priors (tumour / organ+generic / extra classes = normal fallback), true and false anchor
def lbf_nu(anchor_, region):
    q = np.where(anchor_, 1.0, np.where(F.dil(anchor_, sp, 3.0), 0.5, 0.0)); pi = 0.02 + 0.9 * q; N = region
    allowed = np.array([[False, True, True], [True, True, True]]); tp, bps = priors(Y4[:, N].T, q[N], organ[N].astype(int), allowed)
    v, _, _ = vb.log_bayes_factor(Y4[:, N].T, pi[N], organ[N].astype(int), allowed, tp, bps, q[N], K=4, alpha0=0.25, nu=5.0,
                                  a_w=2.0, b_w=1.0, edges=vb.build_edges(N, sp), parity=vb.parity(N),
                                  nu_prior=(NP6[:4], NP6[4:], NP6[-1]))
    return v


lt6, lf6 = lbf_nu(anchor, F.dil(anchor, sp, 15.0) & M), lbf_nu(fa, F.dil(fa, sp, 15.0) & M)
print(f'T6 nu prior: true anchor log BF {lt6:.1f}, false anchor log BF {lf6:.1f}'); ok &= lt6 > 0 and lf6 < 0
# T7: MAP nu
from scipy import stats as _st
for n7 in (5000, 50):
    y7 = _st.t.rvs(4, size=(n7, 1), random_state=np.random.default_rng(3)); tp7, bp7 = vb.class_priors(y7, np.ones(n7), np.ones((n7, 1)))
    nus7 = []
    for pri in (None, [[(1.0, np.log(1 / 30.0), 0.5)]] * 2):
        m7 = vb.Model(y7, np.ones(n7), np.zeros(n7, int), np.ones((1, 1), bool), tp7, bp7, K=1, alpha0=0.25, nu=10.0, tumour=True,
                      learn_nu=pri is None, nu_prior=pri)
        m7.init(np.ones(n7)).run(max_iter=500); nus7.append(float(m7.nu_k[0]))
    print(f'T7 n={n7}: nu ML {nus7[0]:.2f}, MAP {nus7[1]:.2f} (prior centre 30)')
    if n7 == 5000:
        ok &= abs(nus7[1] - 4.0) / 4.0 <= 0.25
    else:
        ok &= abs(np.log(1 / nus7[1]) - np.log(1 / 30.0)) < abs(np.log(1 / nus7[0]) - np.log(1 / 30.0))
print('ALL PASS' if ok else 'SOME FAILED', f'({time.time() - t0:.0f} s)')
sys.exit(0 if ok else 1)
