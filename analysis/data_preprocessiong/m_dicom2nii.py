from joblib import Parallel, delayed
import subprocess
from pathlib import Path
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
        niis = Path(d).glob('*.nii.gz')
        niis = [str(p) for p in niis]
        if len(niis) > 1: 
            excluded_nifit += niis
        elif len(niis) == 1:
            included_nifit += niis
    save_path = f"{args.save_dir}/{args.dataset}_included_nifti.json"
    data_dict = {"included nifti": included_nifit, "excluded nifti": excluded_nifit}
    with open(save_path, "w") as f:
        json.dump(data_dict, f, indent=4)
    print(f"Included {len(included_nifit)} nifti, excluded {len(excluded_nifit)}")