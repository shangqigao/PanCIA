"""A priori tau for the v6 appearance statistic (same definition as v5's r2_tau_sim.py): the median held-out gain s of a
lesion 1 normal-tissue SD brighter than its tissue, lesion volumes 1-80 ml (at 1 mm^3 voxels: 1,000-80,000 voxels,
sampled to <= 6,000 as in E_app), reference 20,000 voxels. Smooth Gaussian intensities (feature = smoothed intensity),
32 quantile bins over sample + reference, random 2-fold parity. No data; nothing tuned on outcomes."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pancia_kb import r2_v6 as V
rng = np.random.default_rng(0); S = []
for _ in range(200):
    nT = int(min(np.exp(rng.uniform(np.log(1000), np.log(80000))), V.NS)); nH = 20000
    t = rng.normal(1.0, 1.0, nT); h = rng.normal(0.0, 1.0, nH)
    e = np.quantile(np.concatenate([t, h]), np.arange(1, V.GB) / V.GB)
    s = V.heldout_gain(np.searchsorted(e, t), rng.integers(0, 2, nT), np.searchsorted(e, h), rng.integers(0, 2, nH))
    S.append(s)
print('tau_app (median s, 1 SD brighter, 1-80 ml) =', round(float(np.median(S)), 3), ' IQR', np.round(np.percentile(S, [25, 75]), 3))
