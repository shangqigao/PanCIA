import sys
import os
# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import pydicom
import argparse
import pathlib
import json
import logging
import joblib

from tiatoolbox.wsicore.wsireader import WSIReader 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def is_included_dicom(ds):
    desc = ds.get("SeriesDescription", "").lower()
    image_type = [s.lower() for s in ds.get("ImageType", [])]
    
    keywords = ["scout", "summary", "survey", "topogram", "loc", "prep", "localizer", "recon", "mip"]
    if any(k in desc for k in keywords):
        return False
    keytypes = ["derived", "secondary", "mpr", "mip"]
    if any(k in image_type for k in keytypes ):
        return False
    return True

def is_included_wsi(wsi_path):
    wsi_name = pathlib.Path(wsi_path).stem
    try:
        wsi = WSIReader.open(wsi_path)
        if wsi.info.mpp is None and wsi.info.objective_power is None:
            logging.info(f"No required mpp or power info for {wsi_name}")
            del wsi
            return False
        else:
            del wsi
            return True
    except:
        logging.info(f"Cannot open {wsi_name}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="TCGA", type=str)
    parser.add_argument('--modality', default="radiology", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    args = parser.parse_args()

    if args.modality == 'radiology':
        series_paths = pathlib.Path(f"{args.data_dir}/{args.dataset}").rglob('1.3.6*')
    elif args.modality == 'pathology':
        series_paths = pathlib.Path(f"{args.data_dir}/{args.dataset}").rglob('*.svs')
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")
    series_paths = [p for p in series_paths]

    def _inclusion_exclusion(idx, path):
        logging.info(f"Processing [{idx + 1} / {len(series_paths)}] ...")
        if args.modality == 'radiology':
            dicom_files = path.glob('*.dcm')
            raw_dicoms = []
            for dicom in dicom_files:
                ds = pydicom.dcmread(dicom, stop_before_pixels=True)
                if is_included_dicom(ds):
                    raw_dicoms.append(True)
                else:
                    raw_dicoms.append(False)
            if all(raw_dicoms):
                return ("included", str(path))
            else:
                logging.info(f"Excluding series {path.name}")
                return ("excluded", str(path))
        else:
            valid_wsi = is_included_wsi(path)
            if valid_wsi:
                return ("included", str(path))
            else:
                logging.info(f"Excluding wsi {path.name}")
                return ("excluded", str(path))

    # process in parallel
    results = joblib.Parallel(n_jobs=32, backend="threading")(
        joblib.delayed(_inclusion_exclusion)(idx, path)
        for idx, path in enumerate(series_paths)
    )
    # Merge results
    included_series = [p for t, p in results if t == "included"]
    excluded_series = [p for t, p in results if t == "excluded"]
    print(f"Totally {len(included_series)} raw series included")
    print(f"Totally {len(excluded_series)} series excluded")
    if args.modality == 'radiology':
        save_path = f"{args.save_dir}/{args.dataset}_included_raw_series.json"
    else:
        save_path = f"{args.save_dir}/{args.dataset}_included_wsi.json"
    data_dict = {"included series": included_series, "excluded series": excluded_series}
    with open(save_path, "w") as f:
        json.dump(data_dict, f, indent=4)

