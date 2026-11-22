"""Interactive-agent pilot, step D (22 Nov 2026; claude/interactive_agent_loop_plan_2026-11-07.md, claude/interactive_pilot_2026-11-22.md).
Builds nnInteractive prompts for stage-1 v8.6 lesions (ledger v86_r2seg_bp2), from the stage-1 output only (never from one agent):
  P1  one positive point per anchor, at the anchor voxel deepest inside the anchor (largest distance to its boundary, mm);
  P2  P1 + a 2D box: the lesion extent xi's bounding box on the acquisition-plane slice where xi is largest;
  P3  P2 + a box on the slice through each anchor centre (xi's bounding box on that slice; skipped if xi is empty there);
  P4  P1 + a lasso: xi's own outline on its largest acquisition-plane slice (the README documents no initial-mask call, so the
      planned 'xi as initial mask' is replaced by the documented 2D lasso of xi).
Lesion xi = union of the lesion's anchors' chosen extents (psi from ledger/v85/extent/fit_psi.json; per held-out site fold, as in the
evaluation). Acquisition axis = the axis with the largest voxel spacing (axis 2 if isotropic). Coordinates are voxel indices of the
NIfTI array as nibabel loads it (x, y, z); the runner passes the image in the same order.
Pilot selection (expert labels used ONLY to choose and later score the pilot lesions): accepted or uncertain lesions whose xi touches
an expert tumour, per site up to --per-site (default 7), drawn from random tumour scans (seed 0); half of each site's quota is reserved
for under-segmented lesions (xi recall of the touched tumour < 0.7) when available.
Outputs (outputs root): interactive/pilot_bp2/pilot_lesions_<site>.csv, prompts/<ds>__<case>.json and .npz (lasso crops, xi mask boxes).
Usage: python s3_prompts.py [--per-site 7] [--max-scans-per-site 60]"""
import os, sys, json, glob, argparse, time
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, os.path.join(HERE, '..', 'stage1'))
from pancia_kb.stage2 import paths, expert as EX   # noqa: E402
import s1_eval_r2seg as E   # noqa: E402
LEDGER = 'v86_r2seg_bp2'


def lesion_masks(z, meta, ids, psi, shape):
    xi = np.zeros(shape, bool); anchors = []
    for i in ids:
        m = meta['anchors'].get(str(int(i)))
        if not m:
            continue
        box = m['box']; sl = tuple(slice(b[0], b[1]) for b in box)
        a = np.zeros(shape, bool); a[sl] = E.unpack(z, f'a{i}', box); anchors.append((int(i), a))
        ex = m['extents']
        if ex:
            j = int(np.argmax(E.G2.extent_prior([dict(logratio=e['logratio'], flags=e.get('flags', [])) for e in ex], psi, {})))
            xi[tuple(slice(b[0], b[1]) for b in ex[j]['box'])] |= E.unpack(z, f'a{i}_e{j}', ex[j]['box'])
        else:
            xi |= a
    return xi, anchors


def box_on_slice(mask, ax, s):
    sl = [slice(None)] * 3; sl[ax] = s; m2 = mask[tuple(sl)]
    if not m2.any():
        return None
    nz = np.nonzero(m2); other = [d for d in range(3) if d != ax]; bb = [None] * 3
    for k, d in enumerate(other):
        bb[d] = [int(nz[k].min()), int(nz[k].max()) + 1]
    bb[ax] = [int(s), int(s) + 1]
    return bb


def prompts_for(xi, anchors, sp):
    ax = int(np.argmax(sp)) if np.ptp(sp) > 1e-3 else 2
    pts = []
    for i, a in anchors:
        sl = ndimage.find_objects(a.astype(np.uint8))[0]
        pad = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        d = ndimage.distance_transform_edt(a[pad], sampling=sp)
        p = np.unravel_index(int(np.argmax(d)), d.shape)
        pts.append([int(p[k] + pad[k].start) for k in range(3)])
    area = xi.sum(axis=tuple(d for d in range(3) if d != ax))
    s_big = int(np.argmax(area))
    box_big = box_on_slice(xi, ax, s_big)
    boxes_anchor = []
    for i, a in anchors:
        c = int(round(np.argwhere(a)[:, ax].mean()))
        b = box_on_slice(xi, ax, c)
        if b is not None and b != box_big and b not in boxes_anchor:
            boxes_anchor.append(b)
    crop = xi[tuple(slice(b[0], b[1]) for b in box_big)].astype(np.uint8)
    return dict(axis=ax, points=pts, box_largest=box_big, boxes_anchor=boxes_anchor, lasso_bbox=box_big), crop


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per-site', type=int, default=7); ap.add_argument('--max-scans-per-site', type=int, default=60)
    ap.add_argument('--sites', default=None, help='comma list; outputs pilot_lesions_<site>.csv per site (resumable by site)')
    a = ap.parse_args()
    L = os.path.join(paths.ledger_dir(), LEDGER); O = os.path.join(paths.root('outputs'), 'interactive', 'pilot_bp2'); P = os.path.join(O, 'prompts')
    os.makedirs(P, exist_ok=True)
    an = pd.concat([pd.read_csv(f, usecols=['rel', 'id', 'lesion', 'decision', 'kind']) for f in glob.glob(os.path.join(L, 'anchors*.csv'))]).drop_duplicates(['rel', 'id'], keep='last')
    an['ds'] = an.rel.str.split('/').str[1]; an['case'] = an.rel.str.split('/', n=2).str[2]
    an = an[an.ds.isin(EX.SITE_ENTITY) & an.decision.isin(['accepted', 'uncertain'])]
    tt = pd.read_csv(os.path.join(L, 'diag', 'tumour_tiers.csv'), dtype={'case': str})
    rng = np.random.default_rng(0); rows = []
    sites = a.sites.split(',') if a.sites else sorted(set(EX.SITE_ENTITY.values()))
    for site in sites:
        if os.path.exists(os.path.join(O, f'pilot_lesions_{site}.csv')):
            print(site, 'done'); continue
        dss = [d for d, s in EX.SITE_ENTITY.items() if s == site]
        scans = sorted(set(zip(tt.ds[tt.ds.isin(dss)], tt.case[tt.ds.isin(dss)])))
        scans = [scans[k] for k in rng.permutation(len(scans))][:a.max_scans_per_site]
        under, ok = [], []; psi = E.psi_for(site)[1]; t0 = time.time()
        for ds, case in scans:
            if len(under) >= a.per_site and len(ok) >= a.per_site:
                break
            g = an[(an.ds == ds) & (an.case == case)]
            if g.empty:
                continue
            lf = os.path.join(paths.root('R2Seg'), ds, 'labels', case + '_seg.nii.gz')
            img = nib.load(lf.replace('/labels/', '/images/').replace('_seg.nii.gz', '.nii.gz')); sp = np.array(img.header.get_zooms()[:3], float)
            T = np.isin(EX._read_labels(lf), EX.APPEARANCE[ds]['tumour']); lt, _ = ndimage.label(T, structure=EX.STRUCT26)
            z = np.load(os.path.join(L, 'extents', ds, case + '.npz')); meta = json.loads(str(z['meta']))
            if tuple(meta['shape']) != T.shape:
                continue
            for les, gl in g.groupby('lesion'):
                xi, anchors = lesion_masks(z, meta, gl.id, psi, T.shape)
                if not anchors or not xi.any():
                    continue
                hit = set(np.unique(lt[xi]).tolist()) - {0}
                if not hit:
                    continue
                G = np.isin(lt, list(hit)); rec = float((xi & G).sum() / G.sum()); dice = float(2 * (xi & G).sum() / (xi.sum() + G.sum()))
                dec = 'accepted' if (gl.decision == 'accepted').any() else 'uncertain'
                r = dict(site=site, ds=ds, case=case, lesion=int(les), decision=dec, n_anchors=len(anchors), kinds='|'.join(sorted(set(gl.kind))),
                         xi_ml=float(xi.sum() * np.prod(sp) / 1000), tumour_ml=float(G.sum() * np.prod(sp) / 1000), xi_recall=rec, xi_dice=dice,
                         under=rec < 0.7, _xi=xi, _anchors=anchors, _sp=sp, _img=f'{ds}/images/{case}.nii.gz')
                (under if r['under'] else ok).append(r)
        half = a.per_site // 2 + a.per_site % 2
        pick = under[:half]; pick += ok[:a.per_site - len(pick)]; pick += under[half:half + a.per_site - len(pick)]
        print(site, f'scanned {len(scans)} scans in {time.time() - t0:.0f}s; under {len(under)} ok {len(ok)} -> {len(pick)}', flush=True)
        rows += pick
        write(rows[-len(pick):] if pick else [], O, P, site)
    print(len(rows), 'lesions ->', O)


def write(rows, O, P, site):
    per_scan = {}
    for r in rows:
        pr, crop = prompts_for(r['_xi'], r['_anchors'], r['_sp'])
        key = f"{r['ds']}__{r['case']}"
        d = per_scan.setdefault(key, dict(image=r['_img'], ds=r['ds'], case=r['case'], lesions=[], crops={}))
        d['lesions'].append(dict(lesion=r['lesion'], decision=r['decision'], **pr)); d['crops'][f"lasso_{r['lesion']}"] = crop
        d['crops'][f"xi_{r['lesion']}"] = np.packbits(r['_xi'])
    for key, d in per_scan.items():
        np.savez_compressed(os.path.join(P, key + '.npz'), **d.pop('crops'))
        json.dump(d, open(os.path.join(P, key + '.json'), 'w'), indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]).to_csv(os.path.join(O, f'pilot_lesions_{site}.csv'), index=False)


if __name__ == '__main__':
    main()
