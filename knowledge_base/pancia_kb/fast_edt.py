"""Fast exact Euclidean distance transform for stage 1 (15 Nov 2026; claude/stage1_runtime_profile_2026-11-15.md).
scipy.ndimage.distance_transform_edt took about half of stage 1's run time. The `edt` package (seung-lab; multi-threaded C++,
anisotropic) computes the same Euclidean distance to the nearest zero voxel, 5-8x faster single-threaded, in float32
(max |difference| ~1e-6 mm on test volumes; no threshold flips at 2 mm or 10 mm).
install() replaces scipy.ndimage.distance_transform_edt in this process by a wrapper with scipy's semantics; it falls back to
scipy for anything the wrapper does not cover (return_indices, distances/indices outputs, no zero voxel, unusual arguments).
Used by scripts/stage1/g3_run_v85.py only; stage 1 v8.4 is untouched."""
import numpy as np
from scipy import ndimage

_SCIPY = ndimage.distance_transform_edt
_STATE = dict(installed=False, calls=0, fallback=0)


def _fast(input, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None):
    import edt as _edt
    if return_indices or not return_distances or distances is not None or indices is not None:
        _STATE['fallback'] += 1
        return _SCIPY(input, sampling=sampling, return_distances=return_distances, return_indices=return_indices,
                      distances=distances, indices=indices)
    m = np.asarray(input) != 0
    if m.ndim not in (2, 3) or m.size == 0 or m.all():
        _STATE['fallback'] += 1                       # no zero voxel: scipy's own convention
        return _SCIPY(input, sampling=sampling)
    if sampling is None:
        an = (1.0,) * m.ndim
    else:
        s = np.atleast_1d(np.asarray(sampling, float))
        an = tuple(float(x) for x in (np.repeat(s, m.ndim) if s.size == 1 else s))
    _STATE['calls'] += 1
    d = _edt.edt(np.ascontiguousarray(m, dtype=np.uint8), anisotropy=an, black_border=False, parallel=1)
    return d.astype(np.float64)


def install():
    """patch scipy.ndimage.distance_transform_edt in this process (idempotent); returns True when `edt` is available"""
    try:
        import edt  # noqa: F401
    except ImportError:
        return False
    if not _STATE['installed']:
        ndimage.distance_transform_edt = _fast
        _STATE['installed'] = True
    return True


def selftest(n=20, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        shp = tuple(int(x) for x in rng.integers(5, 60, 3)); sp = tuple(float(x) for x in rng.uniform(0.4, 5.0, 3))
        m = rng.random(shp) < rng.uniform(0.3, 0.99)
        a = _SCIPY(m, sampling=sp); b = _fast(m, sampling=sp)
        worst = max(worst, float(np.abs(a - b).max()))
    return worst
