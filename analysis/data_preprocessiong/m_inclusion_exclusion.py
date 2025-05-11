import pydicom
import os
import argparse
import pathlib
import pandas as pd

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    args = parser.parse_args()

    series_paths = pathlib.Path(args.data_dir).rglob('1.3.6*')
    included_series = []
    excluded_series = []
    for path in series_paths:
        dicom_files = path.glob('*.dcm')
        raw_dicoms = []
        for dicom in dicom_files:
            ds = pydicom.dcmread(dicom, stop_before_pixels=True)
            if is_included_dicom(ds):
                raw_dicoms.append(True)
            else:
                raw_dicoms.append(False)
        if any(raw_dicoms):
            included_series.append(f"{path}")
        else:
            excluded_series.append(f"{path}")
            print(f"Excluding series {path.name}")
    print(f"Totally {len(included_series)} raw series included")
    print(f"Totally {len(excluded_series)} series excluded")
    save_path = f"{args.save_dir}/included_raw_series.csv"
    data_dict = {"included series": included_series, "excluded series": excluded_series}
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, index=False)

