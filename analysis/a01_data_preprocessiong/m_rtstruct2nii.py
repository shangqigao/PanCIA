import os
import glob
import argparse
import logging
import pathlib
import joblib

import pandas as pd
import numpy as np
import SimpleITK as sitk

from rt_utils import RTStructBuilder, ds_helper, image_helper


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core conversion
# -----------------------------------------------------------------------------
def DICOMRTSTRUCT2NIFTI(dicom_series_path: pathlib.Path,
                       rtstruct_path: pathlib.Path,
                       output_seg_path: pathlib.Path):

    output_seg_path.mkdir(parents=True, exist_ok=True)

    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_series_path))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Load RTSTRUCT
    rtstruct = RTStructBuilder.create_from(
        str(dicom_series_path),
        str(rtstruct_path)
    )
    roi_names = rtstruct.get_roi_names()
    logger.info(roi_names)

    x, y, z = image.GetSize()
    fused_mask = np.zeros((z, y, x), dtype=np.uint8)

    for i, roi in enumerate(rtstruct.ds.StructureSetROISequence):
        contour_sequence = ds_helper.get_contour_sequence_by_roi_number(
            rtstruct.ds, roi.ROINumber
        )
        mask = image_helper.create_series_mask_from_contour_sequence(
            rtstruct.series_data, contour_sequence
        )
        mask = np.transpose(mask, (2, 1, 0))
        fused_mask[mask > 0] = i + 1

    fused_mask = np.flip(fused_mask, axis=2)
    fused_mask = np.rot90(fused_mask, axes=(1, 2))

    final_mask = fused_mask.astype(np.uint8)

    mask_image = sitk.GetImageFromArray(final_mask)
    mask_image.SetSpacing(image.GetSpacing())
    mask_image.SetOrigin(image.GetOrigin())
    mask_image.SetDirection(image.GetDirection())

    seg_name = "_".join(roi_names) if roi_names else "NoFindings"
    out_file = output_seg_path / f"{seg_name}.nii.gz"
    logger.info(f"Writing segmentation: {out_file.resolve()}")
    sitk.WriteImage(mask_image, str(out_file))


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def process_series(idx, series_path, df_meta, args, total):

    logger.info(f"[{idx + 1}/{total}] Processing {series_path.name}")

    df_idx = df_meta[
        (df_meta["ReferencedSeriesInstanceUID"] == series_path.name) &
        (df_meta["Annotation Type"].isin(["Segmentation", "No Findings"]))
    ]

    if df_idx.empty:
        return

    for row in df_idx.itertuples(index=False):
        rt_uid = row.SeriesInstanceUID
        modality = row.ReferencedSeriesModality

        parts = list(series_path.parent.parts)
        mod_idx = parts.index(modality)
        parts[mod_idx] = "RTSTRUCT"

        rt_path = pathlib.Path(*parts) / rt_uid / "1-1.dcm"

        if not rt_path.exists():
            logger.warning(f"Missing RTSTRUCT: {rt_path}")
            continue

        rel = rt_path.parts[rt_path.parts.index(args.dataset) + 1:]
        out_dir = pathlib.Path(args.save_dir) / pathlib.Path(*rel).parent

        DICOMRTSTRUCT2NIFTI(series_path, rt_path, out_dir)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--meta_dir", required=True)
    parser.add_argument("--dataset", default="CPTAC")
    parser.add_argument("--modality", default="radiology")
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_dir) / args.dataset

    if args.modality != "radiology":
        raise ValueError(f"Unsupported modality: {args.modality}")

    series_paths = list(data_root.rglob("1.3.6*"))

    csv_files = glob.glob(os.path.join(args.meta_dir, "*.csv"))
    df_meta = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    uids = set(df_meta["ReferencedSeriesInstanceUID"].dropna().astype(str))
    series_paths = [p for p in series_paths if p.name in uids]

    logger.info(f"Found {len(series_paths)} annotated series")

    joblib.Parallel(n_jobs=1, backend="threading")(
        joblib.delayed(process_series)(
            idx, path, df_meta, args, len(series_paths)
        )
        for idx, path in enumerate(series_paths)
    )
