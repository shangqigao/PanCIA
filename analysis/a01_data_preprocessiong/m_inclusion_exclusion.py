import sys
import os
# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import pydicom
import argparse
import csv
import pathlib
import tarfile
import json
import joblib

from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader 


OV04_ALLOWED_MODALITIES = {"CT", "MR"}


def _parse_modalities(value):
    """Return the modalities present in a metadata field such as ``CT\\MR``."""
    normalized = str(value or "").upper()
    for separator in ("\\", ",", ";", "|", "/"):
        normalized = normalized.replace(separator, " ")
    return {item for item in normalized.split() if item}


def _resolve_ov04_folder(row, dataset_root):
    """Resolve ``RDS folder`` relative to ``<data_dir>/OV04``."""
    folder = (row.get("RDS folder") or "").strip()
    if not folder:
        return None

    folder_path = pathlib.Path(folder).expanduser()
    if folder_path.is_absolute():
        return folder_path

    return dataset_root / folder_path


def _safe_extract_tar(archive_path, output_dir):
    """Extract an archive while rejecting links and paths outside output_dir."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = output_dir.resolve()

    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (output_dir / member.name).resolve()
            if resolved_output not in member_path.parents and member_path != resolved_output:
                raise ValueError(
                    f"Unsafe path in OV04 archive {archive_path}: {member.name}"
                )
            if member.issym() or member.islnk():
                raise ValueError(
                    f"Links are not allowed in OV04 archive {archive_path}: "
                    f"{member.name}"
                )
        archive.extractall(output_dir, members=members)


def _ov04_archive_paths(row, archive_dir):
    """Recursively find the CT/MR archives represented by one metadata row."""
    patient_id = (row.get("AnonPatientID") or "").strip()
    dir_name = (row.get("dirName") or "").strip()
    modalities = _parse_modalities(row.get("ModalitiesInStudy"))
    if not patient_id or not dir_name:
        return []

    archive_paths = set()
    for modality in sorted(modalities.intersection(OV04_ALLOWED_MODALITIES)):
        archive_name = f"{patient_id}_{dir_name}_{modality}.tar"
        archive_paths.update(archive_dir.rglob(archive_name))
    return sorted((path for path in archive_paths if path.is_file()), key=str)


def get_ov04_series_paths(data_dir, dataset, csv_path, extract_dir):
    """Extract and find CT/MR DICOM series referenced by the OV04 CSV."""
    data_root = pathlib.Path(data_dir)
    dataset_root = data_root / dataset
    csv_path = pathlib.Path(csv_path)
    extract_root = pathlib.Path(extract_dir).expanduser()
    extract_root.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        raise FileNotFoundError(f"OV04 CSV does not exist: {csv_path}")

    series_paths = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"ModalitiesInStudy", "RDS folder"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"OV04 CSV is missing required column(s): {sorted(missing_columns)}"
            )

        for row in reader:
            modalities = _parse_modalities(row.get("ModalitiesInStudy"))
            if not modalities.intersection(OV04_ALLOWED_MODALITIES):
                continue

            archive_dir = _resolve_ov04_folder(row, dataset_root)
            if archive_dir is None or not archive_dir.exists():
                logger.warning(
                    "Skipping OV04 row %s: folder does not exist (%s)",
                    row.get("OV04_ID", "<unknown>"),
                    archive_dir,
                )
                continue
            if archive_dir.is_file():
                archive_dir = archive_dir.parent

            try:
                rds_folder_relative = archive_dir.relative_to(dataset_root)
            except ValueError:
                # Absolute folders outside <data_dir>/OV04 retain their final
                # folder name without allowing an absolute extraction path.
                rds_folder_relative = pathlib.Path(archive_dir.name)

            archive_paths = _ov04_archive_paths(row, archive_dir)
            if not archive_paths:
                logger.warning(
                    "Skipping OV04 row %s: no matching CT/MR archive found "
                    "under %s",
                    row.get("OV04_ID", "<unknown>"),
                    archive_dir,
                )
                continue

            for archive_path in archive_paths:
                # Never create generated data beside archives in the shared folder.
                relative_archive = archive_path.relative_to(archive_dir)
                extraction_dir = (
                    extract_root
                    / rds_folder_relative
                    / relative_archive.with_suffix("")
                )
                has_extracted_dicoms = (
                    extraction_dir.exists()
                    and any(extraction_dir.rglob("*.dcm"))
                )
                if not has_extracted_dicoms:
                    logger.info("Extracting OV04 archive %s", archive_path)
                    _safe_extract_tar(archive_path, extraction_dir)

                # A tar contains multiple series, potentially at arbitrary depths.
                series_paths.update(
                    dicom_path.parent
                    for dicom_path in extraction_dir.rglob("*.dcm")
                )

    return sorted(series_paths, key=str)

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
            logger.info(f"No required mpp or power info for {wsi_name}")
            del wsi
            return False
        else:
            del wsi
            return True
    except:
        logger.info(f"Cannot open {wsi_name}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="TCGA", type=str)
    parser.add_argument('--modality', default="radiology", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument(
        '--csv_path', '--csv',
        default=None,
        type=str,
        help=(
            "Optional dataset metadata CSV. For OV04 radiology, only CT/MR "
            "series referenced by this CSV are considered."
        ),
    )
    parser.add_argument(
        '--extract_dir',
        default=None,
        type=str,
        help=(
            "Directory for extracted OV04 DICOM archives. Defaults to "
            "<save_dir>/<dataset>_extracted; the shared archive directory is "
            "never modified."
        ),
    )
    args = parser.parse_args()

    if args.modality == 'radiology':
        if args.dataset.upper() == "OV04" and args.csv_path is not None:
            extract_dir = args.extract_dir or str(
                pathlib.Path(args.save_dir) / f"{args.dataset}_extracted"
            )
            series_paths = get_ov04_series_paths(
                args.data_dir, args.dataset, args.csv_path, extract_dir
            )
        else:
            series_paths = pathlib.Path(
                f"{args.data_dir}/{args.dataset}"
            ).rglob('1.3.6*')
    elif args.modality == 'pathology':
        series_paths = pathlib.Path(f"{args.data_dir}/{args.dataset}").rglob('*.svs')
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")
    series_paths = [p for p in series_paths]

    def _inclusion_exclusion(idx, path):
        logger.info(f"Processing [{idx + 1} / {len(series_paths)}] ...")
        if args.modality == 'radiology':
            dicom_files = path.glob('*.dcm')
            raw_dicoms = []
            for dicom in dicom_files:
                ds = pydicom.dcmread(dicom, stop_before_pixels=True)
                is_allowed_modality = (
                    args.dataset.upper() != "OV04"
                    or args.csv_path is None
                    or str(ds.get("Modality", "")).upper()
                    in OV04_ALLOWED_MODALITIES
                )
                if is_allowed_modality and is_included_dicom(ds):
                    raw_dicoms.append(True)
                else:
                    raw_dicoms.append(False)
            if raw_dicoms and all(raw_dicoms):
                return ("included", str(path))
            else:
                logger.info(f"Excluding series {path.name}")
                return ("excluded", str(path))
        else:
            valid_wsi = is_included_wsi(path)
            if valid_wsi:
                return ("included", str(path))
            else:
                logger.info(f"Excluding wsi {path.name}")
                return ("excluded", str(path))

    # process in parallel
    results = joblib.Parallel(n_jobs=32, backend="threading")(
        joblib.delayed(_inclusion_exclusion)(idx, path)
        for idx, path in enumerate(series_paths)
    )
    # Merge results
    included_series = [p for t, p in results if t == "included"]
    excluded_series = [p for t, p in results if t == "excluded"]
    logger.info(f"Totally {len(included_series)} raw series included")
    logger.info(f"Totally {len(excluded_series)} series excluded")
    if args.modality == 'radiology':
        save_path = f"{args.save_dir}/{args.dataset}_included_raw_series.json"
    else:
        save_path = f"{args.save_dir}/{args.dataset}_included_wsi.json"
    data_dict = {"included series": included_series, "excluded series": excluded_series}
    with open(save_path, "w") as f:
        json.dump(data_dict, f, indent=4)
