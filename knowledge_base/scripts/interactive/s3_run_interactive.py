"""Interactive-agent pilot runner (22 Nov 2026), for the lab server (GPU). Runs nnInteractive on the prompts written by s3_prompts.py.
Per scan: the image is loaded with nibabel (raw intensities as float32; no windowing or normalisation, as the nnInteractive README
requires) in the array order (x, y, z) used by the prompts, and set once. Per lesion and variant: reset_interactions, add the prompts,
save the target buffer as uint8 NIfTI with the image affine. Variants (see s3_prompts.py): P1 points; P2 points + box on xi's largest
slice; P3 P2 + boxes through the anchor centres; P4 points + lasso of xi on its largest slice. The outline is a CANDIDATE only.
Needs: pip install nnInteractive (>= 2.5; PyTorch first), nibabel. Weights: Hugging Face MIC-DKFZ/nnInteractive (CC BY-NC-SA 4.0).
Outputs: <out>/<ds>/<case>_les<k>_<V>.nii.gz, <out>/timing.csv (resumable: existing masks are skipped).
Usage: python s3_run_interactive.py --prompts <dir>/prompts --r2seg-root <R2Seg> --out <dir>/masks [--variants P1,P2,P3,P4] [--device cuda:0]
       [--only <substring>] [--shard i/n]"""
import os, sys, json, glob, time, argparse, csv
import numpy as np, nibabel as nib


def session_for(device):
    import torch
    from nnInteractive.model_management import ensure_model_available, get_default_model_id
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    s = nnInteractiveInferenceSession(device=torch.device(device), use_torch_compile=False, verbose=False,
                                      torch_n_threads=min(8, os.cpu_count() or 1), do_autozoom=True)
    s.initialize_from_trained_model_folder(str(ensure_model_available(get_default_model_id())))
    print('nnInteractive model licence:', getattr(s, 'license', 'n/a'), flush=True)
    return s


def run_variant(s, les, V, crops):
    import torch
    s.reset_interactions()
    for p in les['points']:
        s.add_point_interaction(tuple(int(x) for x in p), include_interaction=True)
    if V in ('P2', 'P3'):
        s.add_bbox_interaction(les['box_largest'], include_interaction=True)
    if V == 'P3':
        for b in les['boxes_anchor']:
            s.add_bbox_interaction(b, include_interaction=True)
    if V == 'P4':
        s.add_lasso_interaction(crops[f"lasso_{les['lesion']}"].astype(np.uint8), include_interaction=True, interaction_bbox=les['lasso_bbox'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True); ap.add_argument('--r2seg-root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--variants', default='P1,P2,P3,P4'); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--only', default=None); ap.add_argument('--shard', default='0/1')
    a = ap.parse_args(); SI, SN = (int(x) for x in a.shard.split('/'))
    import torch
    files = sorted(glob.glob(os.path.join(a.prompts, '*.json')))
    if a.only:
        files = [f for f in files if any(x in f for x in a.only.split(','))]
    files = [f for k, f in enumerate(files) if k % SN == SI]
    os.makedirs(a.out, exist_ok=True); tf = os.path.join(a.out, f'timing_s{SI}.csv'); new = not os.path.exists(tf)
    tw = csv.writer(open(tf, 'a', buffering=1)); new and tw.writerow(['ds', 'case', 'lesion', 'variant', 'seconds', 'voxels', 'note'])
    s = session_for(a.device); V_ = a.variants.split(',')
    for f in files:
        d = json.load(open(f)); crops = np.load(f[:-5] + '.npz'); od = os.path.join(a.out, d['ds']); os.makedirs(od, exist_ok=True)
        todo = [(les, V) for les in d['lesions'] for V in V_ if not os.path.exists(os.path.join(od, f"{d['case']}_les{les['lesion']}_{V}.nii.gz"))]
        if not todo:
            continue
        t0 = time.time(); img = nib.load(os.path.join(a.r2seg_root, d['image'])); arr = np.asarray(img.get_fdata(dtype=np.float32))
        s.set_image(arr[None]); tgt = torch.zeros(arr.shape, dtype=torch.uint8); s.set_target_buffer(tgt)
        t_img = time.time() - t0
        for les, V in todo:
            t1 = time.time(); note = ''
            try:
                run_variant(s, les, V, crops)
                m = tgt.cpu().numpy().astype(np.uint8)
            except Exception as ex:
                m = np.zeros(arr.shape, np.uint8); note = 'error: ' + repr(ex)[:200]
            nib.save(nib.Nifti1Image(m, img.affine), os.path.join(od, f"{d['case']}_les{les['lesion']}_{V}.nii.gz"))
            tw.writerow([d['ds'], d['case'], les['lesion'], V, round(time.time() - t1, 2), int(m.sum()), note])
        tw.writerow([d['ds'], d['case'], -1, 'set_image', round(t_img, 2), int(arr.size), ''])
        print(d['ds'], d['case'], f'{len(todo)} masks, image {t_img:.1f}s, total {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
