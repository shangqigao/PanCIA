"""Measure pre/post physical volumes for all EndoMRI scans and models."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from m_segmentation_model_evaluation import DEFAULT_ALL_ROOT


CHANNELS = {"endometrioma": 0, "ovary": 1, "uterus": 2}
RAW_DIRS = {
    "baseline": "segmentations",
    "r6": "segmentations_r6",
    "r7": "segmentations_r7",
}


def measure_one(task: tuple[str, str, str, Path, float]) -> list[dict]:
    model, stage, scan_name, path, threshold = task
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 4 or data.shape[-1] < 3:
        raise ValueError(f"Expected >=3 channels for {path}, got {data.shape}")
    probability = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        probability /= float(np.iinfo(data.dtype).max)
    probability = np.clip(probability, 0.0, 1.0)
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    rows = []
    for class_name, channel in CHANNELS.items():
        values = probability[..., channel]
        rows.append(
            {
                "model": model,
                "stage": stage,
                "scan_name": scan_name,
                "class": class_name,
                "threshold": threshold,
                "voxel_volume_mm3": voxel_volume,
                "volume_mm3": float(np.count_nonzero(values >= threshold) * voxel_volume),
                "probability_weighted_volume_mm3": float(values.sum(dtype=np.float64) * voxel_volume),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-root", type=Path, default=DEFAULT_ALL_ROOT)
    parser.add_argument(
        "--reasoning-root",
        type=Path,
        default=None,
        help="Comparison output root (defaults to <all-root>/reasoning_model_comparison).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    reasoning_root = args.reasoning_root or args.all_root / "reasoning_model_comparison"
    metadata = pd.read_csv(args.all_root / "class_presence.csv")
    tasks = []
    for model, raw_subdir in RAW_DIRS.items():
        directories = {
            "pre": args.all_root / raw_subdir / "BiomedParse",
            "post": reasoning_root / model / "post_probability_maps",
        }
        for stage, directory in directories.items():
            for scan_name in metadata.scan_name:
                path = directory / f"{scan_name}_endometrioma.nii.gz"
                if not path.exists():
                    raise FileNotFoundError(path)
                tasks.append((model, stage, scan_name, path, args.threshold))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(measure_one, tasks, chunksize=4):
            rows.extend(result)
    volumes = pd.DataFrame(rows)
    volumes = volumes.merge(
        metadata[["scan_name", "center", "case_id", "endometrioma_label"]],
        on="scan_name", how="left", validate="many_to_one",
    )
    volumes["domain"] = volumes.scan_name.str[:2]
    volumes["modality"] = volumes.scan_name.str.rsplit("_", n=1).str[-1]
    gt_path = args.all_root / "endometriosis_detection" / "gt_endometrioma_case_volumes.csv"
    gt = pd.read_csv(gt_path)[["case_id", "gt_endometrioma_case_volume_mm3"]]
    volumes = volumes.merge(gt, on="case_id", how="left", validate="many_to_one")
    volumes["gt_endometrioma_case_volume_mm3"] = volumes["gt_endometrioma_case_volume_mm3"].fillna(0.0)
    output = reasoning_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "all_scan_pre_post_volumes.csv"
    volumes.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
