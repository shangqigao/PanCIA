"""Stage 2 panels (TCGA pilot; visual check). Agreed layout: 2 x 3 whole slices (axial / coronal / sagittal through the lesion's anchor
centroid, LPS display, never cropped). Top: image only (dashed box = anchors). Bottom: cyan anchors, orange stage-1 chosen extent,
red stage-2 refined tumour, magenta new candidates in the cluster, and the voxel tumour posterior p_v (heat, p >= 0.05).
Usage: python3 s2_png.py [--tag tcga_v1] [--n 60] [--only substr]   -> stage2/<tag>/png/<cancer>_<k>_L<lesion>.png"""
import os, sys, json, glob, argparse
import numpy as np, pandas as pd, nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from pancia_kb.stage2 import paths   # noqa: E402

VIEWS = [('axial', 2), ('coronal', 1), ('sagittal', 0)]


def slice2d(M3, d, idx, t):
    M = np.take(M3, idx, axis=d); rem = [x for x in range(3) if x != d]
    for j, x in enumerate(rem):
        if t[x][1] == -1:
            M = np.flip(M, axis=j)
    if t[rem[0]][0] > t[rem[1]][0]:
        M = M.T
    return M


def unpack(a, shape, n=None):
    v = np.unpackbits(a)[: int(np.prod(shape)) * (n or 1)].astype(bool)
    return v.reshape(((n,) if n else ()) + tuple(shape))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='tcga_v1'); ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    O = paths.out_dir(a.tag); OP = paths.out_dir(a.tag, 'png')
    L = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(O, 'lesions*.csv'))], ignore_index=True)
    if a.only:
        L = L[L.rel.str.contains(a.only)]
    sel = L.sample(min(a.n, len(L)), random_state=0) if a.n and len(L) > a.n else L
    segr = paths.root('TCGA_Seg'); nif = paths.root('TCGA_NIFTI')
    for rel, g in sel.groupby('rel'):
        man = json.load(open(os.path.join(segr, 'VoxTell', 'Radiology', f'{rel}_plan_manifest.json')))
        img = nib.load(os.path.join(nif, f"{man['img_name']}.nii.gz"))
        X = np.asanyarray(img.dataobj); X = np.asarray(X[..., 0] if X.ndim == 4 else X, np.float32)
        sp = np.array(img.header.get_zooms()[:3], float); aff = img.affine
        t = ornt_transform(io_orientation(aff), axcodes2ornt(('L', 'P', 'S')))
        lps_sp = np.zeros(3)
        for x in range(3): lps_sp[int(t[x][0])] = sp[x]
        orig_of = {int(t[x][0]): x for x in range(3)}
        for r in g.itertuples():
            f = os.path.join(OP, f"{r.cancer}_{r.k}_L{r.lesion}.png")
            if os.path.exists(f):
                continue
            z = np.load(os.path.join(O, 'maps', str(r.k), f'c{r.cluster}.npz'))
            Wb = z['W']; shape = tuple(z['shape']); n = int(z['n_les']); keys = list(z['keys'])
            i = keys.index(f'L{r.lesion}')
            full = lambda M: _place(M, Wb, X.shape)
            p = full(z['p'].astype(np.float32) / 255.0)
            ZL = unpack(z['ZL'], shape, n); A = unpack(z['anchors'], shape, n); XI = unpack(z['xi'], shape, n)
            Zi, Ai, XIi = full(ZL[i]), full(A[i]), full(XI[i])
            newc = full((p[tuple(slice(a_, b_) for a_, b_ in Wb)] >= 0.5) & ~ZL.any(0))
            c_orig = [int(np.mean(np.nonzero(Ai.any(axis=tuple(y for y in range(3) if y != d)))[0])) for d in range(3)]
            fig, Ax = plt.subplots(2, 3, figsize=(19, 12.8))
            for col, (vname, Fx) in enumerate(VIEWS):
                d = orig_of[Fx]; idx = c_orig[d]
                im = slice2d(X, d, idx, t); others = [x for x in range(3) if x != Fx]
                sx, sy = lps_sp[others[0]], lps_sp[others[1]]
                if rel.startswith('MR'):
                    v = im[im > np.percentile(im, 5)]; lo, hi = np.percentile(v, [1, 99.5]) if v.size else (0, 1)
                else:
                    lo, hi = -160, 240
                disp = lambda M: M.T; origin = 'upper' if Fx == 2 else 'lower'; asp = sy / sx
                for rr in (0, 1):
                    Ax[rr, col].imshow(disp(im), cmap='gray', vmin=lo, vmax=hi, origin=origin, aspect=asp, interpolation='bilinear'); Ax[rr, col].axis('off')
                    Ax[rr, col].text(0.01, 0.5, {2: 'R', 1: 'R', 0: 'A'}[Fx], color='yellow', transform=Ax[rr, col].transAxes, fontsize=9, va='center')
                    Ax[rr, col].text(0.98, 0.5, {2: 'L', 1: 'L', 0: 'P'}[Fx], color='yellow', transform=Ax[rr, col].transAxes, fontsize=9, va='center', ha='right')
                Ax[0, col].set_title(f'{vname} (LPS) - image only', fontsize=10); Ax[1, col].set_title(f'{vname} with stage-1 / stage-2', fontsize=10)
                am = slice2d(Ai, d, idx, t)
                if am.any():
                    ys, xs = np.nonzero(disp(am)); px, py = 12 / sx, 12 / sy
                    Ax[0, col].add_patch(Rectangle((xs.min() - px, ys.min() - py), xs.max() - xs.min() + 2 * px, ys.max() - ys.min() + 2 * py,
                                                   fill=False, ec='cyan', lw=1.0, ls='--'))
                ps = slice2d(p, d, idx, t)
                Ax[1, col].imshow(np.ma.masked_less(disp(ps), 0.05), cmap='hot', vmin=0, vmax=1, alpha=0.35, origin=origin, aspect=asp)
                for M3, colr, lw, st in ((XIi, 'darkorange', 1.6, 'solid'), (Zi, 'red', 1.6, 'solid'), (newc, 'magenta', 1.2, 'dotted'), (Ai, 'cyan', 1.2, 'solid')):
                    M = slice2d(M3, d, idx, t)
                    if M.any(): Ax[1, col].contour(disp(M).astype(float), [0.5], colors=colr, linewidths=lw, linestyles=st)
            fig.suptitle(f"{r.cancer} {r.mod} k{r.k} lesion {r.lesion} | stage-1 {r.decisions} | S {r.S} (h_map {r.h_map})\n"
                         f"extent {r.vol_xi} ml -> refined {r.vol_Z} ml (trimmed {r.trimmed_ml}, recalled {r.recalled_ml}; by territory {r.recalled_by_territory}) | "
                         f"Dice(refined, extent) {r.dice_Z_xi} | mean p extent {r.pbar_xi}, refined {r.pbar_Z} | log BF {r.log_bf} -> reported {r.reported}\n"
                         f"components {r.comp_frac} | region growth {r.n_grow} | stage-1 test on refined tumour {r.evidence_on_Z}\n"
                         "Bottom: cyan anchors, orange stage-1 chosen extent, red stage-2 refined tumour, magenta dotted new candidates, heat = p_v",
                         fontsize=8.5)
            plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(f, dpi=90); plt.close()
            print(f, flush=True)


def _place(M, Wb, shape):
    out = np.zeros(shape, M.dtype)
    out[tuple(slice(int(a_), int(b_)) for a_, b_ in Wb)] = M
    return out


if __name__ == '__main__':
    main()
