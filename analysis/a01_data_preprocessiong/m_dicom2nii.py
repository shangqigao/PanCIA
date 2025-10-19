from joblib import Parallel, delayed
import subprocess
from pathlib import Path
import json
import argparse
import logging
import nibabel as nib
import numpy as np
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def check_3D_affine(nii_path):
    """ check if image is 3D and affine is valid
    """
    try:
        nii = nib.load(nii_path)
    except Exception:
        return False, "Error loading nifti"

    try:
        data = nii.get_fdata()
    except Exception:
        return False, "Error getting nii array"
        
    affine = nii.affine
    shape = np.squeeze(data).shape
    if len(shape) == 4 or len(shape) < 3:
        return False, "image is not 3D"
    # Must be 4x4
    if affine.shape != (4, 4):
        return False, "Affine is not 4x4"

    # Must be finite numbers
    if not np.isfinite(affine).all():
        return False, "Affine contains NaN or Inf"

    # Rotation/scale block must be full rank
    A = affine[:3, :3]
    rank = np.linalg.matrix_rank(A)
    if rank < 3:
        return False, f"Affine rank-deficient (rank={rank})"

    # Voxel sizes should be positive
    voxel_sizes = np.sqrt((A ** 2).sum(axis=0))
    if np.any(voxel_sizes <= 0):
        return False, f"Invalid voxel sizes: {voxel_sizes}"

    # Try to derive axis codes
    try:
        axcodes = nib.aff2axcodes(affine)
        if None in axcodes:
            return False, f"Cannot derive axis codes: {axcodes}"
    except Exception:
        return False, "Error deriving axis codes from affine"

    return True, f"shape: {shape}, voxel sizes: {voxel_sizes}, axcodes: {axcodes}"

def convert_series(idx, dicom_dir, output_dir):
    logging.info(f"Processing [{idx + 1} / {len(series_dirs)}] ...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "dcm2niix",
            "-b", "y", 
            "-z", "y",
            "-f", "%p_%s",
            "-o", str(output_dir),
            str(dicom_dir)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] dcm2niix failed for: {dicom_dir}")
        print(f"Return code: {e.returncode}")
    except Exception as e:
        print(f"[Exception] Unexpected error for: {dicom_dir} → {e}")
    
    img_paths = Path(output_dir).glob('*.nii.gz')
    for img_path in img_paths:
        img_path = str(img_path)
        json_path = img_path.replace('.nii.gz', '.json')
        valid, mess = check_3D_affine(img_path)
        if not valid:
            print(f">>>>Warning: {mess}, will remove nifti and json files")
            os.remove(img_path)
            os.remove(json_path)
        else:
            print(f"Got valid image of {mess}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--series', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="TCGA", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    args = parser.parse_args()

    with open(f"{args.series}", "r") as f:
        series = json.load(f)

    series_dirs = [d for d in series["included series"] if Path(d).is_dir()]

    output_dirs = [d.replace(f"/{args.dataset}/", f"/{args.dataset}_NIFTI/") for d in series_dirs]

    Parallel(n_jobs=32)(
        delayed(convert_series)(idx, d, o) for idx, (d, o) in enumerate(zip(series_dirs, output_dirs))
    )

    included_nifit = []
    excluded_nifit = []
    for d in output_dirs:
        niis = list(Path(d).glob('*.nii.gz'))
        if len(niis) > 1: 
            excluded_nifit += niis
        elif len(niis) == 1:
            included_nifit += niis
        else:
            print("No nifti file in this folder")
    save_path = f"{args.save_dir}/{args.dataset}_included_nifti.json"
    data_dict = {"included nifti": included_nifit, "excluded nifti": excluded_nifit}
    with open(save_path, "w") as f:
        json.dump(data_dict, f, indent=4)
    print(f"Included {len(included_nifit)} nifti, excluded {len(excluded_nifit)}")