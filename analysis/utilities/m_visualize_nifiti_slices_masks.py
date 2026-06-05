import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize


# json_file = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology_NIFTI/output_index.json"
# data_root = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology_NIFTI"
# output_root = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology_NIFTI/vis_check"
# metadata_dir = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology/Metadata"

json_file = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology_NIFTI/output_index.json"
data_root = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology_NIFTI"
output_root = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology_NIFTI/vis_check"
metadata_dir = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Metadata"

def get_mask_center(mask):
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return None

    return np.round(coords.mean(axis=0)).astype(int)


def normalize(img):
    p1, p99 = np.percentile(img, [1, 99])

    img = np.clip(img, p1, p99)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return (img * 255).astype(np.uint8)


def resize_keep_aspect(img, target_size=256):
    h, w = img.shape[:2]

    scale = target_size / max(h, w)

    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    if img.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    y0 = (target_size - nh) // 2
    x0 = (target_size - nw) // 2

    canvas[y0:y0+nh, x0:x0+nw] = resized

    return canvas


def resize_to_isotropic(slice2d, spacing_y, spacing_x, order=1):
    """
    Resample a 2D slice so displayed pixels are isotropic.

    spacing_y -> row direction spacing
    spacing_x -> column direction spacing
    """

    h, w = slice2d.shape

    new_h = int(round(h * spacing_y))
    new_w = int(round(w * spacing_x))

    return resize(
        slice2d,
        (new_h, new_w),
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0)
    )


def extract_views_isotropic(image, mask, center, spacing):

    x, y, z = center

    sx, sy, sz = spacing

    views = []

    #
    # Sagittal (YZ plane)
    #
    img_slice = np.rot90(image[x, :, :])
    mask_slice = np.rot90(mask[x, :, :])

    img_slice = resize_to_isotropic(
        img_slice,
        sz,   # rows = Z
        sy,   # cols = Y
        order=1
    )

    mask_slice = resize_to_isotropic(
        mask_slice.astype(float),
        sz,
        sy,
        order=0
    ) > 0.5

    views.append((img_slice, mask_slice))

    #
    # Coronal (XZ plane)
    #
    img_slice = np.rot90(image[:, y, :])
    mask_slice = np.rot90(mask[:, y, :])

    img_slice = resize_to_isotropic(
        img_slice,
        sz,   # rows = Z
        sx,   # cols = X
        order=1
    )

    mask_slice = resize_to_isotropic(
        mask_slice.astype(float),
        sz,
        sx,
        order=0
    ) > 0.5

    views.append((img_slice, mask_slice))

    #
    # Axial (XY plane)
    #
    img_slice = np.rot90(image[:, :, z])
    mask_slice = np.rot90(mask[:, :, z])

    img_slice = resize_to_isotropic(
        img_slice,
        sy,   # rows = Y
        sx,   # cols = X
        order=1
    )

    mask_slice = resize_to_isotropic(
        mask_slice.astype(float),
        sy,
        sx,
        order=0
    ) > 0.5

    views.append((img_slice, mask_slice))

    return views

def make_overlay(img, mask):

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    red = np.zeros_like(rgb)
    red[..., 2] = 255

    idx = mask > 0

    rgb[idx] = (
        0.5 * rgb[idx] +
        0.5 * red[idx]
    ).astype(np.uint8)

    return rgb


def load_tracking_lookup(metadata_dir):
    lookup = {}

    if metadata_dir is None:
        return lookup

    metadata_dir = Path(metadata_dir)

    for csv_file in metadata_dir.glob("*.csv"):

        try:
            df = pd.read_csv(csv_file, dtype=str)

            if (
                "SeriesInstanceUID" not in df.columns
                or "TrackingID" not in df.columns
            ):
                continue

            for _, row in df.iterrows():

                series_uid = str(row["SeriesInstanceUID"]).strip()
                tracking_id = str(row["TrackingID"]).strip()

                if series_uid and tracking_id:
                    lookup[series_uid] = tracking_id

        except Exception as e:
            print(f"Failed loading metadata: {csv_file}")
            print(e)

    print(f"Loaded {len(lookup)} SeriesInstanceUID mappings")

    return lookup

def save_qc(image, mask, spacing, save_path):

    center = get_mask_center(mask)

    if center is None:
        return

    views = extract_views_isotropic(
        image,
        mask,
        center,
        spacing
    )

    top_row = []
    bottom_row = []

    for img_slice, mask_slice in views:

        img_slice = normalize(img_slice)

        top_row.append(
            resize_keep_aspect(img_slice, 256)
        )

        overlay = make_overlay(
            img_slice,
            mask_slice
        )

        bottom_row.append(
            resize_keep_aspect(overlay, 256)
        )

    top_row = np.hstack(top_row)
    bottom_row = np.hstack(bottom_row)

    canvas = np.vstack([
        cv2.cvtColor(top_row, cv2.COLOR_GRAY2BGR),
        bottom_row
    ])

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(save_path),
        canvas,
        [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )


with open(json_file) as f:
    dataset = json.load(f)

tracking_lookup = load_tracking_lookup(metadata_dir)

for patient_id, items in dataset.items():

    for item in items:

        image_path = Path(data_root) / item["image"]
        label_path = Path(data_root) / item["label"]

        try:

            img_nii = nib.load(str(image_path))
            mask_nii = nib.load(str(label_path))

            image = img_nii.get_fdata(dtype=np.float32)

            mask = mask_nii.get_fdata() > 0

            spacing = img_nii.header.get_zooms()[:3]

            rel_path = Path(item["label"])

            parts = list(rel_path.parts[-2:-1])

            # case UID is the last directory before the file
            case_uid = parts[-1]

            tracking_id = tracking_lookup.get(case_uid)

            if tracking_id:
                parts.append(tracking_id)
            else:
                parts.append(
                    rel_path.name.replace(".nii.gz", "")
                )

            save_name = "-".join(parts) + ".png"

            save_path = Path(output_root) / save_name

            save_qc(
                image,
                mask,
                spacing,
                save_path
            )
            print(save_name)

        except Exception as e:
            print("FAILED:", label_path)
            print(e)