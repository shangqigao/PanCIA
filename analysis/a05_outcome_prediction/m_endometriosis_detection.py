"""Scan-level endometrioma detection from six-channel segmentation volumes.

For every probability NIfTI, this script resamples each channel to 1 mm
isotropic resolution, integrates its probabilities to obtain a physical volume,
and evaluates that volume against ``endometrioma_label``. D1 and D2 are analysed
separately to distinguish training-domain from domain-generalization results.

It also applies an interpretable uterus → ovary → endometrioma hierarchy,
learns relative spatial bounds from available D1 annotations, performs
scan-specific block-permutation tests with FDR control, and writes a local HTML
dashboard for candidate-level review.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage
from scipy.stats import energy_distance
from sklearn.metrics import roc_auc_score


LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(
    "/Users/sg2162/Datasets/CancerDatasets/Endometriosis/EndoMRI_All"
)

# Channel order used by prepare_EndoMRI_info().
CLASS_NAMES = (
    "endometrioma",
    "ovary",
    "uterus",
    "tumor_pelvis_cervix",
    "tumor_pelvis_uterus",
    "tumor_pelvis_ovaries",
)

CLASS_TITLES = (
    "Endometrioma in pelvis",
    "Ovary",
    "Uterus",
    "Tumor in pelvis and cervix",
    "Tumor in pelvis and uterus",
    "Tumor in pelvis and ovaries",
)

ANATOMY_CHANNELS = {"endometrioma": 0, "ovary": 1, "uterus": 2}
GT_LABELS = {"endometrioma": 1, "ovary": 3, "uterus": 4}


@dataclass
class SpatialPriors:
    """Robust D1 annotation-derived bounds; no absolute coordinates are stored."""

    uterus_ovary_distance_min_mm: float
    uterus_ovary_distance_max_mm: float
    uterus_ovary_distance_median_mm: float
    uterus_ovary_delta_r_min_mm: float
    uterus_ovary_delta_r_max_mm: float
    uterus_ovary_delta_a_min_mm: float
    uterus_ovary_delta_a_max_mm: float
    uterus_ovary_delta_s_min_mm: float
    uterus_ovary_delta_s_max_mm: float
    uterus_ovary_abs_delta_r_median_mm: float
    uterus_ovary_delta_a_median_mm: float
    uterus_ovary_delta_s_median_mm: float
    endometrioma_uterus_distance_min_mm: float
    endometrioma_uterus_distance_max_mm: float
    endometrioma_uterus_delta_r_min_mm: float
    endometrioma_uterus_delta_r_max_mm: float
    endometrioma_uterus_delta_a_min_mm: float
    endometrioma_uterus_delta_a_max_mm: float
    endometrioma_uterus_delta_s_min_mm: float
    endometrioma_uterus_delta_s_max_mm: float
    endometrioma_ovary_delta_r_min_mm: float
    endometrioma_ovary_delta_r_max_mm: float
    endometrioma_ovary_delta_a_min_mm: float
    endometrioma_ovary_delta_a_max_mm: float
    endometrioma_ovary_delta_s_min_mm: float
    endometrioma_ovary_delta_s_max_mm: float
    endometrioma_ovary_surface_max_mm: float
    endometrioma_volume_min_mm3: float
    endometrioma_volume_max_mm3: float
    endometrioma_sphericity_min: float
    endometrioma_sphericity_max: float
    endometrioma_elongation_min: float
    endometrioma_elongation_max: float
    endometrioma_superior_extent_min_mm: float
    endometrioma_superior_extent_max_mm: float
    n_uterus_ovary_pairs: int
    n_endometrioma_uterus_pairs: int
    n_endometrioma_ovary_pairs: int


def scan_name_from_probability_path(path: Path, suffix: str) -> str:
    """Return the scan name after removing the configured compound suffix."""
    if not path.name.endswith(suffix):
        raise ValueError(f"{path.name!r} does not end with {suffix!r}")
    return path.name[: -len(suffix)]


def probability_scale(nii: nib.spatialimages.SpatialImage) -> float:
    """Return the divisor needed to convert stored values to [0, 1]."""
    dtype = np.dtype(nii.get_data_dtype())
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def measure_probability_map(
    path: Path,
    target_spacing: tuple[float, float, float],
) -> dict[str, float]:
    """Measure probability-weighted physical volumes after isotropic resampling."""
    nii = nib.load(str(path))
    if len(nii.shape) != 4 or nii.shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected a 4D six-channel map, but {path.name} has shape {nii.shape}"
        )

    divisor = probability_scale(nii)
    measurements: dict[str, float] = {
        "source_voxel_volume_mm3": float(np.prod(nii.header.get_zooms()[:3])),
        "field_of_view_volume_mm3": float(
            np.prod(nii.shape[:3])
            * abs(np.linalg.det(nii.affine[:3, :3]))
        ),
    }

    probability_map = np.asanyarray(nii.dataobj, dtype=np.float32) / divisor
    resampling_target = None
    for channel, class_name in enumerate(CLASS_NAMES):
        channel_nii = nib.Nifti1Image(probability_map[..., channel], nii.affine)
        if resampling_target is None:
            resampled_nii = resample_to_output(
                channel_nii,
                voxel_sizes=target_spacing,
                order=1,
            )
            resampling_target = (resampled_nii.shape, resampled_nii.affine)
            measurements["resampled_voxel_volume_mm3"] = float(
                np.prod(resampled_nii.header.get_zooms()[:3])
            )
        else:
            resampled_nii = resample_from_to(
                channel_nii,
                resampling_target,
                order=1,
            )

        resampled_probability = np.clip(
            np.asanyarray(resampled_nii.dataobj), 0.0, 1.0
        )
        voxel_volume_mm3 = float(np.prod(resampled_nii.header.get_zooms()[:3]))
        physical_volume_mm3 = (
            np.sum(resampled_probability, dtype=np.float64)
            * voxel_volume_mm3
        )
        measurements[f"{class_name}_physical_volume_mm3"] = physical_volume_mm3

    return measurements


def collect_measurements(
    probability_dir: Path,
    suffix: str,
    target_spacing: tuple[float, float, float],
) -> pd.DataFrame:
    """Measure all probability maps in a directory."""
    paths = sorted(probability_dir.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(
            f"No probability maps ending in {suffix!r} found in {probability_dir}"
        )

    rows = []
    for index, path in enumerate(paths, start=1):
        LOGGER.info("Measuring %d/%d: %s", index, len(paths), path.name)
        rows.append(
            {
                "scan_name": scan_name_from_probability_path(path, suffix),
                "probability_map": str(path),
                **measure_probability_map(path, target_spacing),
            }
        )
    return pd.DataFrame(rows)


def join_labels(measurements: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Join scan measurements to the binary endometrioma label."""
    labels = pd.read_csv(csv_path)
    required = {"scan_name", "endometrioma_label"}
    missing_columns = required.difference(labels.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
    if labels["scan_name"].duplicated().any():
        duplicates = labels.loc[labels["scan_name"].duplicated(), "scan_name"].tolist()
        raise ValueError(f"CSV contains duplicate scan_name values: {duplicates[:5]}")

    metadata_columns = [
        column
        for column in ("center", "case_id", "scan_name", "endometrioma_label")
        if column in labels.columns
    ]
    merged = measurements.merge(
        labels[metadata_columns],
        on="scan_name",
        how="left",
        validate="one_to_one",
    )

    missing_labels = merged["endometrioma_label"].isna()
    if missing_labels.any():
        scans = merged.loc[missing_labels, "scan_name"].tolist()
        raise ValueError(f"No endometrioma_label found for scans: {scans[:10]}")

    csv_scans = set(labels["scan_name"])
    map_scans = set(measurements["scan_name"])
    if csv_scans - map_scans:
        LOGGER.warning(
            "%d labelled scans have no probability map and will be excluded",
            len(csv_scans - map_scans),
        )

    merged["endometrioma_label"] = merged["endometrioma_label"].astype(int)
    if not merged["endometrioma_label"].isin([0, 1]).all():
        raise ValueError("endometrioma_label must contain only 0 and 1")
    merged["domain"] = merged["scan_name"].str.extract(r"^(D[12])", expand=False)
    if merged["domain"].isna().any():
        scans = merged.loc[merged["domain"].isna(), "scan_name"].tolist()
        raise ValueError(f"Cannot assign D1/D2 domain for scans: {scans[:10]}")
    return merged


def sensitivity_specificity_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 201,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate sensitivity and specificity over volume thresholds."""
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size != scores.size:
        raise ValueError("Volume scores contain non-finite values")

    # Quantiles retain resolution where observed volumes are concentrated.
    thresholds = np.unique(
        np.quantile(finite_scores, np.linspace(0.0, 1.0, n_thresholds))
    )
    epsilon = max(float(np.max(np.abs(finite_scores))) * 1e-9, 1e-12)
    thresholds = np.concatenate(
        ([finite_scores.min() - epsilon], thresholds, [finite_scores.max() + epsilon])
    )

    positive = y_true == 1
    negative = ~positive
    sensitivity = np.empty(thresholds.size, dtype=float)
    specificity = np.empty(thresholds.size, dtype=float)
    for index, threshold in enumerate(thresholds):
        prediction = scores >= threshold
        sensitivity[index] = np.mean(prediction[positive])
        specificity[index] = np.mean(~prediction[negative])
    return thresholds, sensitivity, specificity


def bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    """Return a percentile 95% bootstrap confidence interval for AUROC."""
    if n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, y_true.size, y_true.size)
        if np.unique(y_true[indices]).size == 2:
            values.append(roc_auc_score(y_true[indices], scores[indices]))
    if not values:
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def plot_detection_curves(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Plot six sensitivity/specificity panels and return AUROC results."""
    y_true = data["endometrioma_label"].to_numpy(dtype=int)
    if np.unique(y_true).size != 2:
        raise ValueError("AUROC requires both positive and negative labels")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    result_rows = []

    for class_index, (class_name, class_title, axis) in enumerate(
        zip(CLASS_NAMES, CLASS_TITLES, axes.flat)
    ):
        score_column = f"{class_name}_physical_volume_mm3"
        scores = data[score_column].to_numpy(dtype=float)
        auc = roc_auc_score(y_true, scores)
        ci_low, ci_high = bootstrap_auc_ci(
            y_true,
            scores,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + class_index,
        )
        thresholds, sensitivity, specificity = sensitivity_specificity_curve(
            y_true, scores
        )

        axis.plot(thresholds, sensitivity, linewidth=2, label="Sensitivity")
        axis.plot(
            thresholds,
            specificity,
            linewidth=2,
            linestyle="--",
            label="Specificity",
        )
        axis.set_title(class_title)
        axis.set_xlabel("Predicted volume threshold (mm³)")
        axis.set_xlim(thresholds.min(), thresholds.max())
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        ci_text = (
            f"AUROC = {auc:.3f}\n95% CI {ci_low:.3f}–{ci_high:.3f}"
            if np.isfinite(ci_low)
            else f"AUROC = {auc:.3f}"
        )
        axis.text(
            0.04,
            0.08,
            ci_text,
            transform=axis.transAxes,
            verticalalignment="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.legend(loc="best", frameon=False)
        result_rows.append(
            {
                "class": class_name,
                "domain": domain,
                "score": score_column,
                "auroc": auc,
                "ci_2.5%": ci_low,
                "ci_97.5%": ci_high,
                "n_scans": len(data),
                "n_positive": int(y_true.sum()),
                "n_negative": int((1 - y_true).sum()),
            }
        )

    axes[0, 0].set_ylabel("Sensitivity / specificity")
    axes[1, 0].set_ylabel("Sensitivity / specificity")
    fig.suptitle(
        f"{domain}: endometrioma detection from 1 mm isotropic physical volumes",
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path.suffix.lower() != ".svg":
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(result_rows)


def plot_volume_distributions(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
) -> None:
    """Plot the six physical-volume distributions as violin plots."""
    distributions = [
        data[f"{class_name}_physical_volume_mm3"].to_numpy(dtype=float)
        for class_name in CLASS_NAMES
    ]
    if any(not np.isfinite(values).all() for values in distributions):
        raise ValueError(f"{domain} physical volumes contain non-finite values")

    fig, axis = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(1, len(CLASS_NAMES) + 1)
    violins = axis.violinplot(
        distributions,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        points=200,
    )
    colors = plt.get_cmap("tab10").colors[: len(CLASS_NAMES)]
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(1.5)

    # Explicit IQR bars make the centre and spread readable within each violin.
    quartiles = np.array(
        [np.percentile(values, [25, 50, 75]) for values in distributions]
    )
    axis.vlines(
        positions,
        quartiles[:, 0],
        quartiles[:, 2],
        color="black",
        linewidth=3,
        zorder=3,
    )
    axis.scatter(
        positions,
        quartiles[:, 1],
        color="white",
        edgecolor="black",
        linewidth=0.8,
        s=34,
        zorder=4,
        label="Median and IQR",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(CLASS_TITLES, rotation=20, ha="right")
    axis.set_ylabel("Probability-weighted physical volume (mm³)")
    axis.set_title(
        f"{domain}: predicted class-volume distributions (n={len(data)})"
    )
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path.suffix.lower() != ".svg":
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _world_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return a component centroid in world coordinates for relative geometry."""
    voxel_centroid = np.mean(np.argwhere(mask), axis=0)
    return nib.affines.apply_affine(affine, voxel_centroid)


def _component_shape_features(mask: np.ndarray, affine: np.ndarray) -> dict:
    """Return physical morphology features that do not depend on modality."""
    spacing = nib.affines.voxel_sizes(affine)
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    volume_mm3 = float(mask.sum() * voxel_volume_mm3)
    padded = np.pad(mask.astype(np.int8), 1)
    surface_area_mm2 = 0.0
    for axis in range(3):
        exposed_faces = int(np.abs(np.diff(padded, axis=axis)).sum())
        face_area = float(np.prod(np.delete(spacing, axis)))
        surface_area_mm2 += exposed_faces * face_area
    sphericity = (
        float(
            np.pi ** (1.0 / 3.0)
            * (6.0 * volume_mm3) ** (2.0 / 3.0)
            / surface_area_mm2
        )
        if surface_area_mm2 > 0 and volume_mm3 > 0
        else 0.0
    )
    coordinates = np.argwhere(mask)
    if len(coordinates) >= 3:
        physical = coordinates * spacing
        eigenvalues = np.linalg.eigvalsh(np.cov(physical, rowvar=False))
        elongation = float(
            np.sqrt(max(eigenvalues[-1], 1e-8) / max(eigenvalues[0], 1e-8))
        )
    else:
        elongation = np.inf
    occupied_slices = np.unique(coordinates[:, 2]) if len(coordinates) else []
    return {
        "surface_area_mm2": surface_area_mm2,
        "sphericity": sphericity,
        "elongation": elongation,
        "slice_count": int(len(occupied_slices)),
        "superior_extent_mm": float(len(occupied_slices) * spacing[2]),
    }


def _component_metadata(
    label_map: np.ndarray,
    component_id: int,
    probability: np.ndarray,
    affine: np.ndarray,
    voxel_volume_mm3: float,
) -> dict:
    mask = label_map == component_id
    values = probability[mask]
    centroid_voxel = np.mean(np.argwhere(mask), axis=0)
    centroid_world = nib.affines.apply_affine(affine, centroid_voxel)
    return {
        "component_id": int(component_id),
        "voxel_count": int(mask.sum()),
        "volume_mm3": float(mask.sum() * voxel_volume_mm3),
        "mean_probability": float(values.mean()),
        "max_probability": float(values.max()),
        "confidence": float(0.7 * values.mean() + 0.3 * values.max()),
        "centroid_voxel": centroid_voxel.tolist(),
        "centroid_world": centroid_world.tolist(),
        **_component_shape_features(mask, affine),
    }


def extract_components(
    probability: np.ndarray,
    affine: np.ndarray,
    threshold: float,
    minimum_volume_mm3: float,
) -> tuple[np.ndarray, list[dict]]:
    """Extract 26-connected components using a fixed probability threshold."""
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(probability >= threshold, structure=structure)
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    components = []
    retained = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for component_id in range(1, count + 1):
        mask = labels == component_id
        volume_mm3 = float(mask.sum() * voxel_volume_mm3)
        if volume_mm3 < minimum_volume_mm3:
            continue
        retained[mask] = next_id
        components.append(
            _component_metadata(
                retained,
                next_id,
                probability,
                affine,
                voxel_volume_mm3,
            )
        )
        next_id += 1
    return retained, components


def select_uterus(
    label_map: np.ndarray,
    components: list[dict],
    probability: np.ndarray,
    affine: np.ndarray,
    minimum_uterus_volume_mm3: float,
) -> tuple[dict | None, np.ndarray]:
    """Select the largest individual uterus component above the volume minimum."""
    empty = np.zeros(label_map.shape, dtype=bool)
    viable = [
        item
        for item in components
        if item["volume_mm3"] >= minimum_uterus_volume_mm3
    ]
    if not viable:
        return None, empty
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    primary = max(viable, key=lambda item: item["volume_mm3"])
    uterus_mask = label_map == primary["component_id"]

    values = probability[uterus_mask]
    centroid_voxel = np.mean(np.argwhere(uterus_mask), axis=0)
    uterus = {
        "component_id": primary["component_id"],
        "selection_rule": "largest_component_above_minimum_volume",
        "voxel_count": int(uterus_mask.sum()),
        "volume_mm3": float(uterus_mask.sum() * voxel_volume_mm3),
        "mean_probability": float(values.mean()),
        "max_probability": float(values.max()),
        "confidence": float(0.7 * values.mean() + 0.3 * values.max()),
        "centroid_voxel": centroid_voxel.tolist(),
        "centroid_world": nib.affines.apply_affine(
            affine, centroid_voxel
        ).tolist(),
    }
    return uterus, uterus_mask


def _physical_ball(
    radius_mm: float,
    spacing_mm: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Create an N-dimensional ellipsoidal footprint in physical units."""
    spacing = np.asarray(spacing_mm, dtype=float)
    radii = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    grids = np.ogrid[
        tuple(slice(-radius, radius + 1) for radius in radii)
    ]
    distance = sum(
        (grid * spacing[axis] / radius_mm) ** 2
        for axis, grid in enumerate(grids)
    )
    return distance <= 1.0


def refine_uterus_mask_fast(
    label_map: np.ndarray,
    uterus: dict | None,
    uterus_mask: np.ndarray,
    probability: np.ndarray,
    affine: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    closing_radius_mm: float = 2.5,
    crop_margin_mm: float = 15.0,
    maximum_gap_slices: int = 2,
    trajectory_dilation_mm: float = 5.0,
    maximum_centroid_shift_mm: float = 10.0,
    maximum_volume_ratio: float = 1.5,
) -> tuple[dict | None, np.ndarray, dict]:
    """Conservatively repair a selected uterus using fast binary operations."""
    qc = {
        "uterus_refinement_attempted": False,
        "uterus_refinement_applied": False,
        "uterus_refinement_accepted": False,
        "uterus_refinement_reason": "no_reliable_initial_uterus",
        "uterus_initial_volume_mm3": 0.0,
        "uterus_refined_volume_mm3": 0.0,
        "uterus_refinement_volume_ratio": np.nan,
        "uterus_refinement_centroid_shift_mm": np.nan,
        "uterus_refinement_initial_dice": np.nan,
        "uterus_refinement_slices_recovered": 0,
    }
    if uterus is None or not uterus_mask.any():
        return uterus, uterus_mask, qc

    qc["uterus_refinement_attempted"] = True
    qc["uterus_initial_volume_mm3"] = uterus["volume_mm3"]
    coordinates = np.argwhere(uterus_mask)
    spacing = np.asarray(voxel_spacing_mm, dtype=float)
    margin_voxels = np.ceil(crop_margin_mm / spacing).astype(int)
    lower = np.maximum(0, coordinates.min(axis=0) - margin_voxels)
    upper = np.minimum(
        np.asarray(uterus_mask.shape),
        coordinates.max(axis=0) + margin_voxels + 1,
    )
    slices = tuple(slice(int(low), int(high)) for low, high in zip(lower, upper))
    initial_crop = uterus_mask[slices].copy()
    candidate_crop = (label_map[slices] > 0)

    # Retain only candidate fragments physically near the selected component.
    distance_to_initial = ndimage.distance_transform_edt(
        ~initial_crop,
        sampling=voxel_spacing_mm,
    )
    candidate_crop &= distance_to_initial <= crop_margin_mm
    repaired = initial_crop | candidate_crop

    # Interpolate only short empty gaps whose neighbouring masks follow the
    # same in-plane trajectory after a small physical dilation.
    occupied = np.flatnonzero(repaired.any(axis=(0, 1)))
    dilation_2d = _physical_ball(
        trajectory_dilation_mm, voxel_spacing_mm[:2]
    )
    slices_recovered = 0
    for lower_z, upper_z in zip(occupied[:-1], occupied[1:]):
        gap = int(upper_z - lower_z - 1)
        if gap < 1 or gap > maximum_gap_slices:
            continue
        lower_mask = repaired[:, :, lower_z]
        upper_mask = repaired[:, :, upper_z]
        lower_dilated = ndimage.binary_dilation(
            lower_mask, structure=dilation_2d
        )
        upper_dilated = ndimage.binary_dilation(
            upper_mask, structure=dilation_2d
        )
        if not np.logical_and(lower_dilated, upper_dilated).any():
            continue
        lower_signed = ndimage.distance_transform_edt(lower_mask) - (
            ndimage.distance_transform_edt(~lower_mask)
        )
        upper_signed = ndimage.distance_transform_edt(upper_mask) - (
            ndimage.distance_transform_edt(~upper_mask)
        )
        for offset in range(1, gap + 1):
            weight = offset / (gap + 1)
            repaired[:, :, lower_z + offset] = (
                (1.0 - weight) * lower_signed + weight * upper_signed
            ) >= 0
            slices_recovered += 1

    repaired = ndimage.binary_closing(
        repaired,
        structure=_physical_ball(closing_radius_mm, voxel_spacing_mm),
    )
    repaired = ndimage.binary_fill_holes(repaired)

    # Keep the repaired component with greatest overlap with the original core.
    repaired_labels, repaired_count = ndimage.label(
        repaired, structure=ndimage.generate_binary_structure(3, 3)
    )
    if repaired_count:
        overlaps = [
            int(np.logical_and(repaired_labels == component_id, initial_crop).sum())
            for component_id in range(1, repaired_count + 1)
        ]
        repaired = repaired_labels == (int(np.argmax(overlaps)) + 1)
    refined_mask = np.zeros_like(uterus_mask, dtype=bool)
    refined_mask[slices] = repaired

    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    initial_volume = float(uterus_mask.sum() * voxel_volume_mm3)
    refined_volume = float(refined_mask.sum() * voxel_volume_mm3)
    volume_ratio = refined_volume / max(initial_volume, 1e-8)
    initial_centroid = _world_centroid(uterus_mask, affine)
    refined_centroid = _world_centroid(refined_mask, affine)
    centroid_shift = float(np.linalg.norm(refined_centroid - initial_centroid))
    dice = float(
        2.0
        * np.logical_and(uterus_mask, refined_mask).sum()
        / max(uterus_mask.sum() + refined_mask.sum(), 1)
    )
    initial_shape = _component_shape_features(uterus_mask, affine)
    refined_shape = _component_shape_features(refined_mask, affine)
    shape_not_worse = (
        refined_shape["elongation"]
        <= max(initial_shape["elongation"] * 1.25, 3.0)
        and refined_shape["sphericity"]
        >= initial_shape["sphericity"] * 0.8
    )
    accepted = bool(
        refined_mask.any()
        and 0.8 <= volume_ratio <= maximum_volume_ratio
        and centroid_shift <= maximum_centroid_shift_mm
        and shape_not_worse
    )
    qc.update(
        {
            "uterus_refinement_applied": bool(
                not np.array_equal(refined_mask, uterus_mask)
            ),
            "uterus_refinement_accepted": accepted,
            "uterus_refinement_reason": (
                "accepted_fast_binary_repair"
                if accepted
                else "rejected_refinement_quality_control"
            ),
            "uterus_refined_volume_mm3": refined_volume,
            "uterus_refinement_volume_ratio": volume_ratio,
            "uterus_refinement_centroid_shift_mm": centroid_shift,
            "uterus_refinement_initial_dice": dice,
            "uterus_refinement_slices_recovered": slices_recovered,
        }
    )
    if not accepted:
        return uterus, uterus_mask, qc

    values = probability[refined_mask]
    centroid_voxel = np.mean(np.argwhere(refined_mask), axis=0)
    refined_uterus = {
        **uterus,
        "selection_rule": "largest_component_plus_fast_binary_refinement",
        "voxel_count": int(refined_mask.sum()),
        "volume_mm3": refined_volume,
        "mean_probability": float(values.mean()),
        "max_probability": float(values.max()),
        "confidence": float(0.7 * values.mean() + 0.3 * values.max()),
        "centroid_voxel": centroid_voxel.tolist(),
        "centroid_world": nib.affines.apply_affine(
            affine, centroid_voxel
        ).tolist(),
        **refined_shape,
    }
    return refined_uterus, refined_mask, qc


def _robust_bounds(
    values: list[float],
    fallback: tuple[float, float],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    margin_fraction: float = 0.10,
) -> tuple[float, float]:
    """Estimate conservative bounds without treating missing labels as negatives."""
    if len(values) < 5:
        return fallback
    low, high = np.quantile(values, [lower_quantile, upper_quantile])
    margin = max((high - low) * margin_fraction, 1.0)
    return float(low - margin), float(high + margin)


def _robust_positive_bounds(
    values: list[float],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Estimate broad positive bounds in log space."""
    positive = np.asarray(
        [value for value in values if np.isfinite(value) and value > 0],
        dtype=float,
    )
    if positive.size < 5:
        return fallback
    low, high = np.quantile(np.log(positive), [0.01, 0.99])
    margin = max((high - low) * 0.10, 0.05)
    return float(np.exp(low - margin)), float(np.exp(high + margin))


def learn_spatial_priors(labels_dir: Path) -> SpatialPriors:
    """Learn relative anatomical bounds from D1 scans with required masks present."""
    distances: list[float] = []
    delta_r: list[float] = []
    delta_a: list[float] = []
    delta_s: list[float] = []
    endo_uterus_distances: list[float] = []
    endo_uterus_delta_r: list[float] = []
    endo_uterus_delta_a: list[float] = []
    endo_uterus_delta_s: list[float] = []
    endo_ovary_delta_r: list[float] = []
    endo_ovary_delta_a: list[float] = []
    endo_ovary_delta_s: list[float] = []
    endometrioma_surface_distances: list[float] = []
    endometrioma_volumes: list[float] = []
    endometrioma_sphericities: list[float] = []
    endometrioma_elongations: list[float] = []
    endometrioma_superior_extents: list[float] = []

    for label_path in sorted(labels_dir.glob("D1-*_seg.nii.gz")):
        nii = nib.as_closest_canonical(nib.load(str(label_path)))
        labels = np.asanyarray(nii.dataobj)
        zooms = nii.header.get_zooms()[:3]
        uterus = labels == GT_LABELS["uterus"]
        ovaries = labels == GT_LABELS["ovary"]
        endometriomas = labels == GT_LABELS["endometrioma"]
        largest_ovary_ids: list[int] = []
        if ovaries.any():
            ovary_labels, ovary_count = ndimage.label(ovaries)
            # A scan can contain label islands from partial annotation. Because
            # a patient has at most two ovaries, only its two largest annotated
            # ovarian objects contribute to the physical-volume prior.
            ranked_ovary_ids = sorted(
                range(1, ovary_count + 1),
                key=lambda ovary_id: int((ovary_labels == ovary_id).sum()),
                reverse=True,
            )
            largest_voxel_count = int(
                (ovary_labels == ranked_ovary_ids[0]).sum()
            )
            largest_ovary_ids = [
                ovary_id
                for ovary_id in ranked_ovary_ids[:2]
                if int((ovary_labels == ovary_id).sum())
                >= 0.05 * largest_voxel_count
            ]
        if uterus.any() and ovaries.any():
            uterus_world = _world_centroid(uterus, nii.affine)
            for ovary_id in largest_ovary_ids:
                ovary_mask = ovary_labels == ovary_id
                ovary_world = _world_centroid(ovary_mask, nii.affine)
                delta = ovary_world - uterus_world
                distances.append(float(np.linalg.norm(delta)))
                delta_r.append(float(delta[0]))
                delta_a.append(float(delta[1]))
                delta_s.append(float(delta[2]))
        if uterus.any() and endometriomas.any():
            uterus_world = _world_centroid(uterus, nii.affine)
            endo_labels, endo_count = ndimage.label(endometriomas)
            for endo_id in range(1, endo_count + 1):
                endo_mask = endo_labels == endo_id
                component_volume = float(
                    endo_mask.sum() * abs(np.linalg.det(nii.affine[:3, :3]))
                )
                if component_volume >= 20.0:
                    morphology = _component_shape_features(
                        endo_mask, nii.affine
                    )
                    endometrioma_volumes.append(component_volume)
                    endometrioma_sphericities.append(
                        morphology["sphericity"]
                    )
                    if np.isfinite(morphology["elongation"]):
                        endometrioma_elongations.append(
                            morphology["elongation"]
                        )
                    endometrioma_superior_extents.append(
                        morphology["superior_extent_mm"]
                    )
                endo_world = _world_centroid(endo_mask, nii.affine)
                delta = endo_world - uterus_world
                endo_uterus_distances.append(float(np.linalg.norm(delta)))
                endo_uterus_delta_r.append(float(delta[0]))
                endo_uterus_delta_a.append(float(delta[1]))
                endo_uterus_delta_s.append(float(delta[2]))
        if ovaries.any() and endometriomas.any():
            distance_to_ovary = ndimage.distance_transform_edt(~ovaries, sampling=zooms)
            endo_labels, endo_count = ndimage.label(endometriomas)
            ovary_centroids = [
                _world_centroid(ovary_labels == ovary_id, nii.affine)
                for ovary_id in largest_ovary_ids
            ]
            for endo_id in range(1, endo_count + 1):
                endo_mask = endo_labels == endo_id
                endometrioma_surface_distances.append(
                    float(distance_to_ovary[endo_mask].min())
                )
                if ovary_centroids:
                    endo_world = _world_centroid(endo_mask, nii.affine)
                    nearest_ovary = min(
                        ovary_centroids,
                        key=lambda centroid: float(
                            np.linalg.norm(endo_world - centroid)
                        ),
                    )
                    delta = endo_world - nearest_ovary
                    endo_ovary_delta_r.append(float(delta[0]))
                    endo_ovary_delta_a.append(float(delta[1]))
                    endo_ovary_delta_s.append(float(delta[2]))

    distance_bounds = _robust_bounds(distances, (0.0, 120.0))
    r_bounds = _robust_bounds(delta_r, (-120.0, 120.0))
    a_bounds = _robust_bounds(delta_a, (-100.0, 100.0))
    s_bounds = _robust_bounds(delta_s, (-100.0, 100.0))
    endo_uterus_distance_bounds = _robust_bounds(
        endo_uterus_distances, (0.0, 140.0)
    )
    endo_uterus_r_bounds = _robust_bounds(
        endo_uterus_delta_r, (-120.0, 120.0)
    )
    endo_uterus_a_bounds = _robust_bounds(
        endo_uterus_delta_a, (-100.0, 100.0)
    )
    endo_uterus_s_bounds = _robust_bounds(
        endo_uterus_delta_s, (-100.0, 100.0)
    )
    endo_ovary_r_bounds = _robust_bounds(endo_ovary_delta_r, (-100.0, 100.0))
    endo_ovary_a_bounds = _robust_bounds(endo_ovary_delta_a, (-100.0, 100.0))
    endo_ovary_s_bounds = _robust_bounds(endo_ovary_delta_s, (-100.0, 100.0))
    _, endo_surface_max = _robust_bounds(
        endometrioma_surface_distances,
        (0.0, 30.0),
        lower_quantile=0.0,
        upper_quantile=0.99,
    )
    endo_volume_bounds = _robust_positive_bounds(
        endometrioma_volumes, (20.0, 500000.0)
    )
    endo_sphericity_bounds = _robust_bounds(
        endometrioma_sphericities, (0.05, 1.0)
    )
    endo_elongation_bounds = _robust_positive_bounds(
        endometrioma_elongations, (1.0, 20.0)
    )
    endo_extent_bounds = _robust_positive_bounds(
        endometrioma_superior_extents, (1.0, 200.0)
    )
    return SpatialPriors(
        uterus_ovary_distance_min_mm=max(0.0, distance_bounds[0]),
        uterus_ovary_distance_max_mm=distance_bounds[1],
        uterus_ovary_distance_median_mm=float(np.median(distances)) if distances else 60.0,
        uterus_ovary_delta_r_min_mm=r_bounds[0],
        uterus_ovary_delta_r_max_mm=r_bounds[1],
        uterus_ovary_delta_a_min_mm=a_bounds[0],
        uterus_ovary_delta_a_max_mm=a_bounds[1],
        uterus_ovary_delta_s_min_mm=s_bounds[0],
        uterus_ovary_delta_s_max_mm=s_bounds[1],
        uterus_ovary_abs_delta_r_median_mm=(
            float(np.median(np.abs(delta_r))) if delta_r else 35.0
        ),
        uterus_ovary_delta_a_median_mm=(
            float(np.median(delta_a)) if delta_a else 0.0
        ),
        uterus_ovary_delta_s_median_mm=(
            float(np.median(delta_s)) if delta_s else 0.0
        ),
        endometrioma_uterus_distance_min_mm=max(
            0.0, endo_uterus_distance_bounds[0]
        ),
        endometrioma_uterus_distance_max_mm=endo_uterus_distance_bounds[1],
        endometrioma_uterus_delta_r_min_mm=endo_uterus_r_bounds[0],
        endometrioma_uterus_delta_r_max_mm=endo_uterus_r_bounds[1],
        endometrioma_uterus_delta_a_min_mm=endo_uterus_a_bounds[0],
        endometrioma_uterus_delta_a_max_mm=endo_uterus_a_bounds[1],
        endometrioma_uterus_delta_s_min_mm=endo_uterus_s_bounds[0],
        endometrioma_uterus_delta_s_max_mm=endo_uterus_s_bounds[1],
        endometrioma_ovary_delta_r_min_mm=endo_ovary_r_bounds[0],
        endometrioma_ovary_delta_r_max_mm=endo_ovary_r_bounds[1],
        endometrioma_ovary_delta_a_min_mm=endo_ovary_a_bounds[0],
        endometrioma_ovary_delta_a_max_mm=endo_ovary_a_bounds[1],
        endometrioma_ovary_delta_s_min_mm=endo_ovary_s_bounds[0],
        endometrioma_ovary_delta_s_max_mm=endo_ovary_s_bounds[1],
        endometrioma_ovary_surface_max_mm=max(0.0, endo_surface_max),
        endometrioma_volume_min_mm3=max(20.0, endo_volume_bounds[0]),
        endometrioma_volume_max_mm3=min(500000.0, endo_volume_bounds[1]),
        endometrioma_sphericity_min=max(0.05, endo_sphericity_bounds[0]),
        endometrioma_sphericity_max=min(1.0, endo_sphericity_bounds[1]),
        endometrioma_elongation_min=max(1.0, endo_elongation_bounds[0]),
        endometrioma_elongation_max=min(20.0, endo_elongation_bounds[1]),
        endometrioma_superior_extent_min_mm=max(
            0.0, endo_extent_bounds[0]
        ),
        endometrioma_superior_extent_max_mm=min(200.0, endo_extent_bounds[1]),
        n_uterus_ovary_pairs=len(distances),
        n_endometrioma_uterus_pairs=len(endo_uterus_distances),
        n_endometrioma_ovary_pairs=len(endometrioma_surface_distances),
    )


def normalize_image(image: np.ndarray, modality: str) -> np.ndarray:
    """Apply scan-specific MR normalization or the fixed CT pelvis window."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if modality == "CT":
        low, high = -55.0, 200.0
    else:
        foreground = image[finite & (image != 0)]
        if foreground.size == 0:
            foreground = image[finite]
        if foreground.size == 0:
            raise ValueError("Image contains no finite intensities")
        low, high = np.percentile(foreground, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def within_scan_rank_image(
    normalized_image: np.ndarray,
    original_image: np.ndarray,
    modality: str,
    bins: int = 4096,
) -> np.ndarray:
    """Map intensities to an approximate within-scan empirical CDF."""
    finite = np.isfinite(original_image)
    if modality == "CT":
        reference = finite & (normalized_image > 0)
    else:
        reference = finite & (original_image != 0)
    values = normalized_image[reference]
    if values.size == 0:
        values = normalized_image[np.isfinite(normalized_image)]
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    cumulative = np.cumsum(histogram, dtype=np.float64)
    midpoint_cdf = (cumulative - 0.5 * histogram) / max(values.size, 1)
    indices = np.clip(
        np.searchsorted(edges, normalized_image, side="right") - 1,
        0,
        bins - 1,
    )
    ranks = midpoint_cdf[indices].astype(np.float32)
    ranks[~finite] = 0.0
    return ranks


def infer_modality(scan_name: str) -> str:
    """Infer CT versus MR from a scan name; current EndoMRI scans resolve to MR."""
    return "CT" if "CT" in scan_name.upper().split("_") else "MR"


def _relation_to_uterus(component: dict, uterus: dict, priors: SpatialPriors) -> dict:
    delta = np.asarray(component["centroid_world"]) - np.asarray(
        uterus["centroid_world"]
    )
    distance = float(np.linalg.norm(delta))
    geometry_plausible = (
        priors.uterus_ovary_distance_min_mm
        <= distance
        <= priors.uterus_ovary_distance_max_mm
        and priors.uterus_ovary_delta_r_min_mm
        <= delta[0]
        <= priors.uterus_ovary_delta_r_max_mm
        and priors.uterus_ovary_delta_a_min_mm
        <= delta[1]
        <= priors.uterus_ovary_delta_a_max_mm
        and priors.uterus_ovary_delta_s_min_mm
        <= delta[2]
        <= priors.uterus_ovary_delta_s_max_mm
    )
    def scaled_deviation(value: float, target: float, low: float, high: float) -> float:
        return abs(value - target) / max((high - low) / 2.0, 1.0)

    distance_cost = scaled_deviation(
        distance,
        priors.uterus_ovary_distance_median_mm,
        priors.uterus_ovary_distance_min_mm,
        priors.uterus_ovary_distance_max_mm,
    )
    lateral_cost = scaled_deviation(
        abs(float(delta[0])),
        priors.uterus_ovary_abs_delta_r_median_mm,
        0.0,
        max(
            abs(priors.uterus_ovary_delta_r_min_mm),
            abs(priors.uterus_ovary_delta_r_max_mm),
        ),
    )
    anterior_cost = scaled_deviation(
        float(delta[1]),
        priors.uterus_ovary_delta_a_median_mm,
        priors.uterus_ovary_delta_a_min_mm,
        priors.uterus_ovary_delta_a_max_mm,
    )
    superior_cost = scaled_deviation(
        float(delta[2]),
        priors.uterus_ovary_delta_s_median_mm,
        priors.uterus_ovary_delta_s_min_mm,
        priors.uterus_ovary_delta_s_max_mm,
    )
    prior_cost = (
        distance_cost
        + lateral_cost
        + anterior_cost
        + superior_cost
    )
    return {
        "distance_to_uterus_mm": distance,
        "delta_r_mm": float(delta[0]),
        "delta_a_mm": float(delta[1]),
        "delta_s_mm": float(delta[2]),
        "uterus_relation_plausible": bool(geometry_plausible),
        "uterus_relative_prior_cost": float(prior_cost),
        "uterus_relative_prior_score": float(1.0 / (1.0 + prior_cost)),
        "ovary_selection_plausible": bool(geometry_plausible),
    }


def _endometrioma_relation_to_uterus(
    component: dict,
    uterus: dict,
    priors: SpatialPriors,
) -> dict:
    """Evaluate a candidate using only relative uterus geometry."""
    delta = np.asarray(component["centroid_world"]) - np.asarray(
        uterus["centroid_world"]
    )
    distance = float(np.linalg.norm(delta))
    plausible = (
        priors.endometrioma_uterus_distance_min_mm
        <= distance
        <= priors.endometrioma_uterus_distance_max_mm
        and priors.endometrioma_uterus_delta_r_min_mm
        <= delta[0]
        <= priors.endometrioma_uterus_delta_r_max_mm
        and priors.endometrioma_uterus_delta_a_min_mm
        <= delta[1]
        <= priors.endometrioma_uterus_delta_a_max_mm
        and priors.endometrioma_uterus_delta_s_min_mm
        <= delta[2]
        <= priors.endometrioma_uterus_delta_s_max_mm
    )
    return {
        "distance_to_uterus_mm": distance,
        "endometrioma_delta_r_mm": float(delta[0]),
        "endometrioma_delta_a_mm": float(delta[1]),
        "endometrioma_delta_s_mm": float(delta[2]),
        "endometrioma_uterus_relation_plausible": bool(plausible),
    }


def _endometrioma_relation_to_ovaries(
    component: dict,
    ovaries: list[dict],
    priors: SpatialPriors,
) -> dict:
    """Require plausible R/A/S displacement from at least one selected ovary."""
    relations = []
    endometrioma_world = np.asarray(component["centroid_world"])
    for ovary in ovaries:
        delta = endometrioma_world - np.asarray(ovary["centroid_world"])
        distance = float(np.linalg.norm(delta))
        plausible = (
            priors.endometrioma_ovary_delta_r_min_mm
            <= delta[0]
            <= priors.endometrioma_ovary_delta_r_max_mm
            and priors.endometrioma_ovary_delta_a_min_mm
            <= delta[1]
            <= priors.endometrioma_ovary_delta_a_max_mm
            and priors.endometrioma_ovary_delta_s_min_mm
            <= delta[2]
            <= priors.endometrioma_ovary_delta_s_max_mm
        )
        relations.append(
            {
                "matched_ovary_component_id": ovary["component_id"],
                "distance_to_matched_ovary_centroid_mm": distance,
                "endometrioma_ovary_delta_r_mm": float(delta[0]),
                "endometrioma_ovary_delta_a_mm": float(delta[1]),
                "endometrioma_ovary_delta_s_mm": float(delta[2]),
                "endometrioma_ovary_relation_plausible": bool(plausible),
            }
        )
    plausible_relations = [
        relation
        for relation in relations
        if relation["endometrioma_ovary_relation_plausible"]
    ]
    candidates = plausible_relations or relations
    return min(
        candidates,
        key=lambda relation: relation["distance_to_matched_ovary_centroid_mm"],
    )


def _endometrioma_morphology_plausible(
    component: dict,
    priors: SpatialPriors,
) -> bool:
    """Apply broad D1-derived, modality-independent morphology bounds."""
    return bool(
        priors.endometrioma_volume_min_mm3
        <= component["volume_mm3"]
        <= priors.endometrioma_volume_max_mm3
        and priors.endometrioma_sphericity_min
        <= component["sphericity"]
        <= priors.endometrioma_sphericity_max
        and priors.endometrioma_elongation_min
        <= component["elongation"]
        <= priors.endometrioma_elongation_max
        and priors.endometrioma_superior_extent_min_mm
        <= component["superior_extent_mm"]
        <= priors.endometrioma_superior_extent_max_mm
    )


def _block_values(
    image: np.ndarray,
    mask: np.ndarray,
    block_size_mm: float,
    voxel_spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    """Average intensities in spatial blocks to reduce voxel-dependence."""
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return np.array([], dtype=float)
    block_shape = np.maximum(
        1,
        np.ceil(block_size_mm / np.asarray(voxel_spacing_mm)).astype(int),
    )
    block_coordinates = coordinates // block_shape
    _, inverse = np.unique(block_coordinates, axis=0, return_inverse=True)
    values = image[tuple(coordinates.T)]
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return sums / counts


def adaptive_reference_values(
    image: np.ndarray,
    reference_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    block_size_mm: float,
    voxel_spacing_mm: tuple[float, float, float],
    minimum_blocks: int = 3,
) -> tuple[np.ndarray, int, int]:
    """Use the strongest erosion that retains enough spatial reference blocks."""
    for erosion_iterations in (2, 1, 0):
        if erosion_iterations:
            candidate_reference = ndimage.binary_erosion(
                reference_mask,
                iterations=erosion_iterations,
            )
        else:
            candidate_reference = reference_mask.copy()
        candidate_reference &= ~exclusion_mask
        values = _block_values(
            image,
            candidate_reference,
            block_size_mm,
            voxel_spacing_mm,
        )
        if len(values) >= minimum_blocks:
            return values, erosion_iterations, len(values)
    return np.array([], dtype=float), 0, 0


def permutation_energy_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    maximum_samples: int,
) -> tuple[float, float]:
    """Two-sided distribution test with an empirical permutation p-value."""
    if len(sample_a) < 3 or len(sample_b) < 3:
        return np.nan, np.nan
    if len(sample_a) > maximum_samples:
        sample_a = rng.choice(sample_a, maximum_samples, replace=False)
    if len(sample_b) > maximum_samples:
        sample_b = rng.choice(sample_b, maximum_samples, replace=False)
    observed = float(energy_distance(sample_a, sample_b))
    pooled_scale = float(np.std(np.concatenate([sample_a, sample_b])))
    effect_size = observed / max(pooled_scale, 1e-8)
    combined = np.concatenate([sample_a, sample_b])
    n_a = len(sample_a)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(combined)
        statistic = energy_distance(permuted[:n_a], permuted[n_a:])
        exceedances += statistic >= observed
    p_value = (exceedances + 1.0) / (permutations + 1.0)
    return float(p_value), float(effect_size)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjust finite p-values while preserving missing comparisons as NaN."""
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if finite_indices.size == 0:
        return adjusted.tolist()
    finite_values = np.asarray(p_values)[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.clip(corrected, 0.0, 1.0)
    adjusted[finite_indices] = restored
    return adjusted.tolist()


def _save_candidate_preview(
    image: np.ndarray,
    uterus_mask: np.ndarray,
    ovary_mask: np.ndarray,
    candidate_mask: np.ndarray,
    path: Path,
) -> None:
    """Save a compact overlay through the candidate centroid for local review."""
    slice_axis = int(np.argmin(image.shape))
    coordinates = np.argwhere(candidate_mask)
    slice_index = int(round(np.mean(coordinates[:, slice_axis])))

    def plane(array: np.ndarray) -> np.ndarray:
        return np.rot90(np.take(array, slice_index, axis=slice_axis))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(plane(image), cmap="gray", vmin=0, vmax=1)
    for mask, color, label in (
        (uterus_mask, "cyan", "uterus"),
        (ovary_mask, "lime", "ovary"),
        (candidate_mask, "red", "endometrioma candidate"),
    ):
        mask_plane = plane(mask)
        if mask_plane.any():
            axis.contour(mask_plane, levels=[0.5], colors=[color], linewidths=1.4)
            axis.plot([], [], color=color, label=label)
    axis.set_title(f"Slice {slice_index} · axis {slice_axis}")
    axis.axis("off")
    axis.legend(loc="lower right", framealpha=0.75, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyze_scan_hierarchy(
    image_path: Path,
    probability_path: Path,
    priors: SpatialPriors,
    output_dir: Path,
    threshold: float,
    minimum_volume_mm3: float,
    minimum_uterus_volume_mm3: float,
    local_ring_mm: float,
    reference_ovary_surface_max_mm: float,
    intra_equivalence_effect_max: float,
    permutations: int,
    fdr_alpha: float,
    minimum_effect_size: float,
    block_size_mm: float,
    maximum_samples: int,
    random_seed: int,
    save_previews: bool,
) -> list[dict]:
    """Apply uterus → ovary → endometrioma reasoning to one scan."""
    scan_name = probability_path.name.removesuffix("_endometrioma.nii.gz")
    image_nii = nib.as_closest_canonical(nib.load(str(image_path)))
    probability_nii = nib.as_closest_canonical(nib.load(str(probability_path)))
    image = np.asanyarray(image_nii.dataobj)
    probability = np.asanyarray(probability_nii.dataobj, dtype=np.float32)
    probability /= probability_scale(probability_nii)
    if probability.shape[:3] != image.shape[:3] or probability.shape[-1] != 6:
        raise ValueError(
            f"Image/probability shape mismatch for {scan_name}: "
            f"{image.shape} versus {probability.shape}"
        )
    if not np.allclose(image_nii.affine, probability_nii.affine, atol=1e-4):
        raise ValueError(f"Image/probability affines do not match for {scan_name}")
    modality = infer_modality(scan_name)
    normalized_image = normalize_image(image, modality)
    rank_image = within_scan_rank_image(normalized_image, image, modality)
    voxel_spacing_mm = probability_nii.header.get_zooms()[:3]

    maps: dict[str, np.ndarray] = {}
    components: dict[str, list[dict]] = {}
    for class_name, channel in ANATOMY_CHANNELS.items():
        maps[class_name], components[class_name] = extract_components(
            probability[..., channel],
            probability_nii.affine,
            threshold,
            minimum_volume_mm3,
        )

    uterus, uterus_mask = select_uterus(
        maps["uterus"],
        components["uterus"],
        probability[..., ANATOMY_CHANNELS["uterus"]],
        probability_nii.affine,
        minimum_uterus_volume_mm3,
    )
    uterus, uterus_mask, uterus_refinement_qc = refine_uterus_mask_fast(
        maps["uterus"],
        uterus,
        uterus_mask,
        probability[..., ANATOMY_CHANNELS["uterus"]],
        probability_nii.affine,
        voxel_spacing_mm,
    )
    uterus_equivalent_radius_mm = (
        float((3.0 * uterus["volume_mm3"] / (4.0 * np.pi)) ** (1.0 / 3.0))
        if uterus is not None
        else np.nan
    )

    selected_ovaries: list[dict] = []
    if uterus is not None:
        plausible_ovaries = []
        for ovary in components["ovary"]:
            relation = _relation_to_uterus(ovary, uterus, priors)
            ovary.update(relation)
            if relation["ovary_selection_plausible"]:
                plausible_ovaries.append(ovary)
        # At most one ovary on each side of the selected uterus. Absolute world
        # position is never used; side is defined only by the relative R axis.
        best_by_side: dict[str, dict] = {}
        for ovary in plausible_ovaries:
            side = "right" if ovary["delta_r_mm"] >= 0 else "left"
            if side not in best_by_side or (
                ovary["uterus_relative_prior_score"]
                > best_by_side[side]["uterus_relative_prior_score"]
            ):
                best_by_side[side] = ovary
        selected_ovaries = sorted(
            best_by_side.values(),
            key=lambda item: item["uterus_relative_prior_score"],
            reverse=True,
        )

    selected_ovary_ids = [item["component_id"] for item in selected_ovaries]
    ovary_mask = np.isin(maps["ovary"], selected_ovary_ids)
    ovary_distance = (
        ndimage.distance_transform_edt(
            ~ovary_mask,
            sampling=probability_nii.header.get_zooms()[:3],
        )
        if ovary_mask.any()
        else None
    )
    uterus_surface_distance = (
        ndimage.distance_transform_edt(
            ~uterus_mask,
            sampling=probability_nii.header.get_zooms()[:3],
        )
        if uterus_mask.any()
        else None
    )

    records = []
    raw_p_values: list[float] = []
    rng = np.random.default_rng(random_seed)
    for candidate in components["endometrioma"]:
        candidate_mask = maps["endometrioma"] == candidate["component_id"]
        record = {
            "scan_name": scan_name,
            "domain": scan_name[:2],
            "candidate_id": candidate["component_id"],
            "candidate_volume_mm3": candidate["volume_mm3"],
            "candidate_mean_probability": candidate["mean_probability"],
            "candidate_max_probability": candidate["max_probability"],
            "candidate_surface_area_mm2": candidate["surface_area_mm2"],
            "candidate_sphericity": candidate["sphericity"],
            "candidate_elongation": candidate["elongation"],
            "candidate_slice_count": candidate["slice_count"],
            "candidate_superior_extent_mm": candidate["superior_extent_mm"],
            "morphology_plausible": _endometrioma_morphology_plausible(
                candidate, priors
            ),
            "uterus_detected": uterus is not None,
            "selected_uterus_component_id": (
                uterus["component_id"] if uterus is not None else np.nan
            ),
            "uterus_volume_mm3": (
                uterus["volume_mm3"] if uterus is not None else 0.0
            ),
            "uterus_selection_rule": (
                uterus["selection_rule"] if uterus is not None else "none"
            ),
            **uterus_refinement_qc,
            "ovaries_detected": len(selected_ovaries),
            "ovary_selection_rule": "best_uterus_relative_prior_per_side",
            "selected_ovary_component_ids": json.dumps(
                [item["component_id"] for item in selected_ovaries]
            ),
            "selected_ovary_uterus_relative_scores": json.dumps(
                [
                    round(item["uterus_relative_prior_score"], 6)
                    for item in selected_ovaries
                ]
            ),
            "reasoning_path": "none",
            "anatomically_plausible": False,
            "distance_to_uterus_mm": np.nan,
            "distance_to_uterus_normalized": np.nan,
            "endometrioma_delta_r_mm": np.nan,
            "endometrioma_delta_a_mm": np.nan,
            "endometrioma_delta_s_mm": np.nan,
            "surface_distance_to_ovary_mm": np.nan,
            "surface_distance_to_uterus_mm": np.nan,
            "joint_anchor_surface_distance_mm": np.nan,
            "matched_ovary_component_id": np.nan,
            "distance_to_matched_ovary_centroid_mm": np.nan,
            "distance_to_matched_ovary_normalized": np.nan,
            "endometrioma_ovary_delta_r_mm": np.nan,
            "endometrioma_ovary_delta_a_mm": np.nan,
            "endometrioma_ovary_delta_s_mm": np.nan,
            "p_vs_ovary": np.nan,
            "q_vs_ovary": np.nan,
            "effect_vs_ovary": np.nan,
            "p_vs_uterus": np.nan,
            "q_vs_uterus": np.nan,
            "effect_vs_uterus": np.nan,
            "p_vs_local_ring": np.nan,
            "q_vs_local_ring": np.nan,
            "effect_vs_local_ring": np.nan,
            "local_ring_blocks": 0,
            "uterus_reference_erosion_iterations": np.nan,
            "uterus_reference_blocks": 0,
            "ovary_reference_erosion_iterations": np.nan,
            "ovary_reference_blocks": 0,
            "accepted": False,
            "preliminary_accepted": False,
            "reference_credible": False,
            "stage_1_anatomical_pass": False,
            "stage_2_inter_class_pass": False,
            "stage_3_intra_class_pass": False,
            "intra_reference_candidate_id": np.nan,
            "p_vs_reference_endometrioma": np.nan,
            "q_vs_reference_endometrioma": np.nan,
            "effect_vs_reference_endometrioma": np.nan,
            "rejection_reason": "",
            "preview": "",
        }
        if uterus is None:
            record["rejection_reason"] = "missing_or_low_confidence_uterus"
        else:
            uterus_surface = float(
                uterus_surface_distance[candidate_mask].min()
            )
            record["surface_distance_to_uterus_mm"] = uterus_surface
            record["joint_anchor_surface_distance_mm"] = uterus_surface
            uterus_relation = _endometrioma_relation_to_uterus(
                candidate, uterus, priors
            )
            record.update(
                {
                    key: value
                    for key, value in uterus_relation.items()
                    if key != "endometrioma_uterus_relation_plausible"
                }
            )
            record["distance_to_uterus_normalized"] = (
                record["distance_to_uterus_mm"]
                / max(uterus_equivalent_radius_mm, 1e-6)
            )
            if not record["morphology_plausible"]:
                record["rejection_reason"] = (
                    "implausible_endometrioma_morphology"
                )
            elif not uterus_relation["endometrioma_uterus_relation_plausible"]:
                record["rejection_reason"] = (
                    "implausible_endometrioma_uterus_relation"
                )
            elif not selected_ovaries:
                # Ovarian segmentation failure does not stop reasoning when a
                # reliable uterus is available. This is deliberately recorded
                # as a lower-evidence pathway.
                record["reasoning_path"] = "uterus_only"
                record["anatomically_plausible"] = True
            else:
                surface_distance = float(ovary_distance[candidate_mask].min())
                record["surface_distance_to_ovary_mm"] = surface_distance
                record["joint_anchor_surface_distance_mm"] = max(
                    uterus_surface, surface_distance
                )
                if surface_distance > priors.endometrioma_ovary_surface_max_mm:
                    record["rejection_reason"] = (
                        "implausible_endometrioma_ovary_distance"
                    )
                else:
                    ovary_relation = _endometrioma_relation_to_ovaries(
                        candidate, selected_ovaries, priors
                    )
                    record.update(
                        {
                            key: value
                            for key, value in ovary_relation.items()
                            if key != "endometrioma_ovary_relation_plausible"
                        }
                    )
                    record["distance_to_matched_ovary_normalized"] = (
                        record["distance_to_matched_ovary_centroid_mm"]
                        / max(uterus_equivalent_radius_mm, 1e-6)
                    )
                    if not ovary_relation[
                        "endometrioma_ovary_relation_plausible"
                    ]:
                        record["rejection_reason"] = (
                            "implausible_endometrioma_ovary_relation"
                        )
                    else:
                        record["reasoning_path"] = "uterus_and_ovary"
                        record["anatomically_plausible"] = True

            if record["anatomically_plausible"]:
                candidate_values = _block_values(
                    rank_image,
                    candidate_mask,
                    block_size_mm,
                    voxel_spacing_mm,
                )
                distance_from_candidate = ndimage.distance_transform_edt(
                    ~candidate_mask,
                    sampling=voxel_spacing_mm,
                )
                local_ring_mask = (
                    (distance_from_candidate > 0)
                    & (distance_from_candidate <= local_ring_mm)
                )
                local_ring_values = _block_values(
                    rank_image,
                    local_ring_mask,
                    block_size_mm,
                    voxel_spacing_mm,
                )
                record["local_ring_blocks"] = len(local_ring_values)
                (
                    record["p_vs_local_ring"],
                    record["effect_vs_local_ring"],
                ) = permutation_energy_test(
                    candidate_values,
                    local_ring_values,
                    permutations,
                    rng,
                    maximum_samples,
                )
                uterus_values, uterus_erosion, uterus_blocks = (
                    adaptive_reference_values(
                        rank_image,
                        uterus_mask,
                        candidate_mask,
                        block_size_mm,
                        voxel_spacing_mm,
                    )
                )
                record["uterus_reference_erosion_iterations"] = uterus_erosion
                record["uterus_reference_blocks"] = uterus_blocks
                if selected_ovaries:
                    ovary_values, ovary_erosion, ovary_blocks = (
                        adaptive_reference_values(
                            rank_image,
                            ovary_mask,
                            candidate_mask,
                            block_size_mm,
                            voxel_spacing_mm,
                        )
                    )
                    record["ovary_reference_erosion_iterations"] = ovary_erosion
                    record["ovary_reference_blocks"] = ovary_blocks
                    record["p_vs_ovary"], record["effect_vs_ovary"] = (
                        permutation_energy_test(
                            candidate_values,
                            ovary_values,
                            permutations,
                            rng,
                            maximum_samples,
                        )
                    )
                record["p_vs_uterus"], record["effect_vs_uterus"] = (
                    permutation_energy_test(
                        candidate_values,
                        uterus_values,
                        permutations,
                        rng,
                        maximum_samples,
                    )
                )
        raw_p_values.extend(
            [
                record["p_vs_ovary"],
                record["p_vs_uterus"],
                record["p_vs_local_ring"],
            ]
        )
        if save_previews:
            preview_path = (
                output_dir
                / "dashboard_assets"
                / f"{scan_name}_candidate_{candidate['component_id']}.png"
            )
            _save_candidate_preview(
                normalized_image,
                uterus_mask,
                ovary_mask,
                candidate_mask,
                preview_path,
            )
            record["preview"] = str(preview_path.relative_to(output_dir))
        records.append(record)

    adjusted = benjamini_hochberg(raw_p_values)
    for index, record in enumerate(records):
        record["q_vs_ovary"] = adjusted[3 * index]
        record["q_vs_uterus"] = adjusted[3 * index + 1]
        record["q_vs_local_ring"] = adjusted[3 * index + 2]
        record["stage_1_anatomical_pass"] = bool(
            record["anatomically_plausible"]
        )
        if not record["anatomically_plausible"]:
            continue
        if not np.isfinite(record["q_vs_uterus"]):
            record["rejection_reason"] = "insufficient_reference_samples"
        elif record["effect_vs_uterus"] < minimum_effect_size:
            record["rejection_reason"] = "insufficient_difference_from_uterus"
        elif record["q_vs_uterus"] >= fdr_alpha:
            record["rejection_reason"] = "fdr_not_significant_vs_uterus"
        elif not np.isfinite(record["q_vs_local_ring"]):
            record["rejection_reason"] = "insufficient_local_ring_samples"
        elif record["effect_vs_local_ring"] < minimum_effect_size:
            record["rejection_reason"] = (
                "insufficient_difference_from_local_ring"
            )
        elif record["q_vs_local_ring"] >= fdr_alpha:
            record["rejection_reason"] = (
                "fdr_not_significant_vs_local_ring"
            )
        elif record["reasoning_path"] == "uterus_only":
            record["preliminary_accepted"] = True
            record["rejection_reason"] = "accepted_uterus_only"
        elif not np.isfinite(record["q_vs_ovary"]):
            record["rejection_reason"] = "insufficient_reference_samples"
        elif record["effect_vs_ovary"] < minimum_effect_size:
            record["rejection_reason"] = "insufficient_difference_from_ovary"
        elif record["q_vs_ovary"] >= fdr_alpha:
            record["rejection_reason"] = "fdr_not_significant_vs_ovary"
        else:
            record["preliminary_accepted"] = True
            record["rejection_reason"] = "accepted_uterus_and_ovary"
        record["stage_2_inter_class_pass"] = bool(
            record["preliminary_accepted"]
        )

    # Final intra-endometrioma consistency stage. The reference minimizes the
    # worst (maximum) surface distance to the available anatomical anchors;
    # every other survivor must be statistically indistinguishable after FDR.
    survivors = [record for record in records if record["preliminary_accepted"]]
    if not survivors:
        return records
    credible_references = []
    for record in survivors:
        ovarian_origin_supported = (
            record["reasoning_path"] == "uterus_only"
            or (
                np.isfinite(record["surface_distance_to_ovary_mm"])
                and record["surface_distance_to_ovary_mm"]
                <= reference_ovary_surface_max_mm
            )
        )
        record["reference_credible"] = bool(
            record["morphology_plausible"]
            and ovarian_origin_supported
            and np.isfinite(record["q_vs_local_ring"])
            and record["q_vs_local_ring"] < fdr_alpha
            and record["effect_vs_local_ring"] >= minimum_effect_size
        )
        if record["reference_credible"]:
            credible_references.append(record)
    if not credible_references:
        for record in survivors:
            record["rejection_reason"] = "no_credible_endometrioma_reference"
        return records

    def reference_rank(record: dict) -> tuple[float, float, float]:
        """Minimize the worst surface distance to the available anchors."""
        ovary_distance_value = record["surface_distance_to_ovary_mm"]
        if not np.isfinite(ovary_distance_value):
            ovary_distance_value = np.inf
        return (
            float(record["joint_anchor_surface_distance_mm"]),
            float(ovary_distance_value),
            float(record["surface_distance_to_uterus_mm"]),
        )

    reference = min(
        credible_references,
        key=reference_rank,
    )
    reference_id = reference["candidate_id"]
    reference_mask = maps["endometrioma"] == reference_id
    reference_values = _block_values(
        rank_image,
        reference_mask,
        block_size_mm,
        voxel_spacing_mm,
    )
    reference["accepted"] = True
    reference["stage_3_intra_class_pass"] = True
    reference["intra_reference_candidate_id"] = reference_id
    reference["rejection_reason"] = "accepted_intra_endometrioma_reference"

    comparison_records = [
        record for record in survivors if record["candidate_id"] != reference_id
    ]
    intra_p_values = []
    for record in comparison_records:
        candidate_values = _block_values(
            rank_image,
            maps["endometrioma"] == record["candidate_id"],
            block_size_mm,
            voxel_spacing_mm,
        )
        p_value, effect_size = permutation_energy_test(
            reference_values,
            candidate_values,
            permutations,
            rng,
            maximum_samples,
        )
        record["intra_reference_candidate_id"] = reference_id
        record["p_vs_reference_endometrioma"] = p_value
        record["effect_vs_reference_endometrioma"] = effect_size
        intra_p_values.append(p_value)

    intra_adjusted = benjamini_hochberg(intra_p_values)
    for record, q_value in zip(comparison_records, intra_adjusted):
        record["q_vs_reference_endometrioma"] = q_value
        if not np.isfinite(q_value):
            record["accepted"] = False
            record["rejection_reason"] = "insufficient_intra_endometrioma_samples"
        elif q_value < fdr_alpha:
            record["accepted"] = False
            record["rejection_reason"] = "intra_endometrioma_statistical_difference"
        elif (
            record["effect_vs_reference_endometrioma"]
            > intra_equivalence_effect_max
        ):
            record["accepted"] = False
            record["rejection_reason"] = (
                "intra_endometrioma_outside_equivalence_margin"
            )
        else:
            record["accepted"] = True
            record["stage_3_intra_class_pass"] = True
            record["rejection_reason"] = "accepted_intra_endometrioma_consistent"
    return records


def run_hierarchical_rejection(
    images_dir: Path,
    probability_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    suffix: str,
    threshold: float,
    minimum_volume_mm3: float,
    minimum_uterus_volume_mm3: float,
    local_ring_mm: float,
    reference_ovary_surface_max_mm: float,
    intra_equivalence_effect_max: float,
    endometrioma_distance_cap_mm: float,
    permutations: int,
    fdr_alpha: float,
    minimum_effect_size: float,
    block_size_mm: float,
    maximum_samples: int,
    random_seed: int,
    save_previews: bool,
) -> tuple[pd.DataFrame, SpatialPriors]:
    """Learn D1 priors and apply scan-specific hierarchical rejection."""
    priors = learn_spatial_priors(labels_dir)
    priors.endometrioma_ovary_surface_max_mm = min(
        priors.endometrioma_ovary_surface_max_mm,
        endometrioma_distance_cap_mm,
    )
    (output_dir / "spatial_priors.json").write_text(
        json.dumps(asdict(priors), indent=2), encoding="utf-8"
    )
    records = []
    probability_paths = sorted(probability_dir.glob(f"*{suffix}"))
    for index, probability_path in enumerate(probability_paths, start=1):
        scan_name = scan_name_from_probability_path(probability_path, suffix)
        image_path = images_dir / f"{scan_name}.nii.gz"
        if not image_path.exists():
            LOGGER.warning("Missing image for %s", scan_name)
            continue
        LOGGER.info("Hierarchical rejection %d/%d: %s", index, len(probability_paths), scan_name)
        records.extend(
            analyze_scan_hierarchy(
                image_path,
                probability_path,
                priors,
                output_dir,
                threshold,
                minimum_volume_mm3,
                minimum_uterus_volume_mm3,
                local_ring_mm,
                reference_ovary_surface_max_mm,
                intra_equivalence_effect_max,
                permutations,
                fdr_alpha,
                minimum_effect_size,
                block_size_mm,
                maximum_samples,
                random_seed + index,
                save_previews,
            )
        )
    results = pd.DataFrame(records)
    results.to_csv(output_dir / "candidate_rejection_results.csv", index=False)
    return results, priors


def _selected_component_volume_mm3(
    probability_path: Path,
    channel: int,
    selected_component_ids: list[int],
    threshold: float,
    minimum_volume_mm3: float,
) -> float:
    """Reconstruct retained connected-component volumes without rerunning reasoning."""
    if not selected_component_ids:
        return 0.0
    nii = nib.as_closest_canonical(nib.load(str(probability_path)))
    probability = (
        np.asanyarray(nii.dataobj[..., channel], dtype=np.float32)
        / probability_scale(nii)
    )
    labels, count = ndimage.label(
        probability >= threshold,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    voxel_volume_mm3 = float(abs(np.linalg.det(nii.affine[:3, :3])))
    component_volumes = np.bincount(labels.ravel())[1:] * voxel_volume_mm3
    retained_volumes = [
        float(volume)
        for volume in component_volumes[:count]
        if volume >= minimum_volume_mm3
    ]
    return float(
        sum(
            retained_volumes[component_id - 1]
            for component_id in selected_component_ids
            if 0 < component_id <= len(retained_volumes)
        )
    )


def collect_pre_post_reasoning_volumes(
    pre_volumes: pd.DataFrame,
    candidates: pd.DataFrame,
    threshold: float,
    minimum_volume_mm3: float,
) -> pd.DataFrame:
    """Combine saved pre volumes with final scan-level component volumes."""
    required_pre = {
        "scan_name",
        "probability_map",
        "endometrioma_label",
        "domain",
        *(
            f"{class_name}_physical_volume_mm3"
            for class_name in ANATOMY_CHANNELS
        ),
    }
    missing = required_pre.difference(pre_volumes.columns)
    if missing:
        raise ValueError(f"Pre-volume CSV is missing columns: {sorted(missing)}")

    candidate_groups = {
        scan_name: group
        for scan_name, group in candidates.groupby("scan_name", sort=False)
    }
    rows = []
    for index, pre_row in pre_volumes.iterrows():
        scan_name = pre_row["scan_name"]
        group = candidate_groups.get(scan_name)
        post_endometrioma = 0.0
        post_ovary = 0.0
        post_uterus = 0.0
        selected_ovary_ids: list[int] = []
        if group is not None and not group.empty:
            accepted = group["accepted"].fillna(False).astype(bool)
            post_endometrioma = float(
                group.loc[accepted, "candidate_volume_mm3"].sum()
            )
            first = group.iloc[0]
            if bool(first.get("uterus_detected", False)):
                post_uterus = float(first.get("uterus_volume_mm3", 0.0))
            raw_ids = first.get("selected_ovary_component_ids", "[]")
            if isinstance(raw_ids, str):
                selected_ovary_ids = [int(value) for value in json.loads(raw_ids)]
            elif isinstance(raw_ids, (list, tuple)):
                selected_ovary_ids = [int(value) for value in raw_ids]
            post_ovary = _selected_component_volume_mm3(
                Path(pre_row["probability_map"]),
                ANATOMY_CHANNELS["ovary"],
                selected_ovary_ids,
                threshold,
                minimum_volume_mm3,
            )
        rows.append(
            {
                "scan_name": scan_name,
                "domain": pre_row["domain"],
                "case_id": pre_row.get("case_id", scan_name),
                "endometrioma_label": int(pre_row["endometrioma_label"]),
                "probability_map": pre_row["probability_map"],
                "field_of_view_volume_mm3": float(
                    pre_row.get("field_of_view_volume_mm3", np.nan)
                ),
                **{
                    f"pre_{class_name}_volume_mm3": float(
                        pre_row[f"{class_name}_physical_volume_mm3"]
                    )
                    for class_name in ANATOMY_CHANNELS
                },
                "post_endometrioma_volume_mm3": post_endometrioma,
                "post_ovary_volume_mm3": post_ovary,
                "post_uterus_volume_mm3": post_uterus,
                "selected_ovary_component_ids": json.dumps(selected_ovary_ids),
            }
        )
        if (index + 1) % 25 == 0:
            LOGGER.info(
                "Post-reasoning volumes %d/%d", index + 1, len(pre_volumes)
            )
    result = pd.DataFrame(rows)
    if np.isfinite(result["field_of_view_volume_mm3"]).all():
        for class_name in ANATOMY_CHANNELS:
            result[f"pre_{class_name}_fov_normalized"] = (
                result[f"pre_{class_name}_volume_mm3"]
                / result["field_of_view_volume_mm3"]
            )
    return result


def plot_pre_post_volume_distributions(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
) -> None:
    """Plot scan-level pre/post volume distributions for the three anatomy classes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    titles = {
        "endometrioma": "Endometrioma",
        "ovary": "Ovary",
        "uterus": "Uterus",
    }
    for axis, class_name in zip(axes, ANATOMY_CHANNELS):
        pre = data[f"pre_{class_name}_volume_mm3"].to_numpy(dtype=float)
        post = data[f"post_{class_name}_volume_mm3"].to_numpy(dtype=float)
        violin = axis.violinplot(
            [pre, post],
            positions=[1, 2],
            widths=0.78,
            showmedians=True,
            showextrema=False,
            points=200,
        )
        for body, color in zip(violin["bodies"], ("#4c78a8", "#e45756")):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.68)
        violin["cmedians"].set_color("black")
        axis.set_xticks([1, 2], ["Pre", "Post"])
        axis.set_title(titles[class_name])
        axis.set_yscale("symlog", linthresh=100.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.03,
            0.97,
            f"Median pre: {np.median(pre):,.0f}\nMedian post: {np.median(post):,.0f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("Physical volume (mm³; zero-compatible log scale)")
    fig.suptitle(
        f"{domain}: volumes before and after anatomical/statistical rejection"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_pre_post_detection_curves(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Plot sensitivity and specificity separately over physical-volume thresholds."""
    y_true = data["endometrioma_label"].to_numpy(dtype=int)
    if np.unique(y_true).size != 2:
        raise ValueError(f"{domain} requires both labels for detection curves")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        sharey=True,
    )
    rows = []
    titles = {
        "endometrioma": "Endometrioma",
        "ovary": "Ovary",
        "uterus": "Uterus",
    }
    for class_index, class_name in enumerate(ANATOMY_CHANNELS):
        for stage_index, stage in enumerate(("pre", "post")):
            axis = axes[stage_index, class_index]
            column = f"{stage}_{class_name}_volume_mm3"
            scores = data[column].to_numpy(dtype=float)
            thresholds, sensitivity, specificity = sensitivity_specificity_curve(
                y_true, scores
            )
            auc = roc_auc_score(y_true, scores)
            ci_low, ci_high = bootstrap_auc_ci(
                y_true,
                scores,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed + 2 * class_index + stage_index,
            )
            case_data = (
                data.groupby(["case_id", "endometrioma_label"], as_index=False)[
                    column
                ]
                .mean()
            )
            case_y_true = case_data["endometrioma_label"].to_numpy(dtype=int)
            case_scores = case_data[column].to_numpy(dtype=float)
            case_auc = roc_auc_score(case_y_true, case_scores)
            case_ci_low, case_ci_high = bootstrap_auc_ci(
                case_y_true,
                case_scores,
                n_bootstrap=n_bootstrap,
                random_seed=(
                    random_seed + 100 + 2 * class_index + stage_index
                ),
            )
            axis.plot(
                thresholds,
                sensitivity,
                color="#4c78a8",
                linewidth=2,
                label="Sensitivity",
            )
            axis.plot(
                thresholds,
                specificity,
                color="#e45756",
                linewidth=2,
                linestyle="--",
                label="Specificity",
            )
            rows.append(
                {
                    "domain": domain,
                    "class": class_name,
                    "stage": stage,
                    "score": column,
                    "auroc": auc,
                    "ci_2.5%": ci_low,
                    "ci_97.5%": ci_high,
                    "case_auroc": case_auc,
                    "case_ci_2.5%": case_ci_low,
                    "case_ci_97.5%": case_ci_high,
                    "n_scans": len(data),
                    "n_cases": len(case_data),
                }
            )
            axis.set_xlim(thresholds.min(), thresholds.max())
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, loc="best")
            ci_text = (
                f"Scan AUROC {auc:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f})\n"
                f"Case AUROC {case_auc:.3f} "
                f"(95% CI {case_ci_low:.3f}–{case_ci_high:.3f})\n"
                "Chance = 0.500"
                if np.isfinite(ci_low)
                else f"Scan AUROC = {auc:.3f}\nChance = 0.500"
            )
            axis.text(
                0.04,
                0.08,
                ci_text,
                transform=axis.transAxes,
                verticalalignment="bottom",
                bbox={
                    "facecolor": "#eef7ee" if auc >= 0.5 else "#fff0f0",
                    "alpha": 0.9,
                    "edgecolor": "none",
                },
                fontsize=8.5,
            )
            axis.set_title(f"{stage.title()} · {titles[class_name]}")
            if class_index == 0:
                axis.set_ylabel(
                    f"{stage.title()}\nSensitivity / specificity"
                )
            axis.set_xlabel("Physical-volume threshold (mm³)")
    n_positive = int(y_true.sum())
    n_negative = int((1 - y_true).sum())
    fig.suptitle(
        f"{domain}: sensitivity and specificity versus physical-volume threshold\n"
        f"n={len(y_true)} scans ({n_positive} positive, {n_negative} negative)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_fov_detection_curves(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Plot scan- and case-level detection curves using physical FOV volume."""
    score_column = "field_of_view_volume_mm3"
    required = {"case_id", "endometrioma_label", score_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"FOV curve data is missing: {sorted(missing)}")
    scan_data = data[["endometrioma_label", score_column]].copy()
    case_data = (
        data.groupby(["case_id", "endometrioma_label"], as_index=False)[
            score_column
        ]
        .mean()
    )
    levels = (("Scan level", scan_data), ("Case level", case_data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    rows = []
    for index, (level_name, level_data) in enumerate(levels):
        axis = axes[index]
        y_true = level_data["endometrioma_label"].to_numpy(dtype=int)
        scores = level_data[score_column].to_numpy(dtype=float)
        thresholds, sensitivity, specificity = sensitivity_specificity_curve(
            y_true, scores
        )
        auc = roc_auc_score(y_true, scores)
        ci_low, ci_high = bootstrap_auc_ci(
            y_true,
            scores,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + index,
        )
        axis.plot(thresholds, sensitivity, linewidth=2, label="Sensitivity")
        axis.plot(
            thresholds,
            specificity,
            linewidth=2,
            linestyle="--",
            color="#e45756",
            label="Specificity",
        )
        axis.set_xlim(thresholds.min(), thresholds.max())
        axis.set_ylim(0.0, 1.02)
        axis.set_title(level_name)
        axis.set_xlabel("Physical FOV threshold (mm³)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="best")
        axis.text(
            0.04,
            0.08,
            f"AUROC {auc:.3f}\n95% CI {ci_low:.3f}–{ci_high:.3f}\n"
            "Chance = 0.500",
            transform=axis.transAxes,
            va="bottom",
            fontsize=9,
            bbox={
                "facecolor": "#eef7ee" if auc >= 0.5 else "#fff0f0",
                "alpha": 0.9,
                "edgecolor": "none",
            },
        )
        rows.append(
            {
                "domain": domain,
                "level": level_name.lower().replace(" ", "_"),
                "score": score_column,
                "auroc": auc,
                "ci_2.5%": ci_low,
                "ci_97.5%": ci_high,
                "n": len(level_data),
            }
        )
    axes[0].set_ylabel("Sensitivity / specificity")
    fig.suptitle(f"{domain}: detection from physical field-of-view volume")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_raw_vs_fov_normalized_curves(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compare raw and FOV-normalized pre-volume detection curves."""
    titles = {
        "endometrioma": "Endometrioma",
        "ovary": "Ovary",
        "uterus": "Uterus",
    }
    variants = (
        ("Raw pre volume", "pre_{class_name}_volume_mm3", "Threshold (mm³)"),
        (
            "FOV-normalized pre volume",
            "pre_{class_name}_fov_normalized",
            "Threshold (volume / FOV)",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    rows = []
    for class_index, class_name in enumerate(ANATOMY_CHANNELS):
        for variant_index, (variant, template, x_label) in enumerate(variants):
            axis = axes[variant_index, class_index]
            score_column = template.format(class_name=class_name)
            y_true = data["endometrioma_label"].to_numpy(dtype=int)
            scores = data[score_column].to_numpy(dtype=float)
            thresholds, sensitivity, specificity = sensitivity_specificity_curve(
                y_true, scores
            )
            scan_auc = roc_auc_score(y_true, scores)
            scan_ci = bootstrap_auc_ci(
                y_true,
                scores,
                n_bootstrap,
                random_seed + 2 * class_index + variant_index,
            )
            case_data = (
                data.groupby(["case_id", "endometrioma_label"], as_index=False)[
                    score_column
                ]
                .mean()
            )
            case_y = case_data["endometrioma_label"].to_numpy(dtype=int)
            case_scores = case_data[score_column].to_numpy(dtype=float)
            case_auc = roc_auc_score(case_y, case_scores)
            case_ci = bootstrap_auc_ci(
                case_y,
                case_scores,
                n_bootstrap,
                random_seed + 100 + 2 * class_index + variant_index,
            )
            axis.plot(thresholds, sensitivity, linewidth=2, label="Sensitivity")
            axis.plot(
                thresholds,
                specificity,
                linewidth=2,
                linestyle="--",
                color="#e45756",
                label="Specificity",
            )
            axis.set_xlim(thresholds.min(), thresholds.max())
            axis.set_ylim(0.0, 1.02)
            axis.set_title(f"{variant} · {titles[class_name]}")
            axis.set_xlabel(x_label)
            if class_index == 0:
                axis.set_ylabel(f"{variant}\nSensitivity / specificity")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, loc="best")
            axis.text(
                0.04,
                0.08,
                f"Scan AUROC {scan_auc:.3f} "
                f"({scan_ci[0]:.3f}–{scan_ci[1]:.3f})\n"
                f"Case AUROC {case_auc:.3f} "
                f"({case_ci[0]:.3f}–{case_ci[1]:.3f})\n"
                "Chance = 0.500",
                transform=axis.transAxes,
                va="bottom",
                fontsize=8.5,
                bbox={
                    "facecolor": "#eef7ee" if scan_auc >= 0.5 else "#fff0f0",
                    "alpha": 0.9,
                    "edgecolor": "none",
                },
            )
            rows.append(
                {
                    "domain": domain,
                    "class": class_name,
                    "variant": variant,
                    "score": score_column,
                    "scan_auroc": scan_auc,
                    "scan_ci_2.5%": scan_ci[0],
                    "scan_ci_97.5%": scan_ci[1],
                    "case_auroc": case_auc,
                    "case_ci_2.5%": case_ci[0],
                    "case_ci_97.5%": case_ci[1],
                    "n_scans": len(data),
                    "n_cases": len(case_data),
                }
            )
    fig.suptitle(
        f"{domain}: raw versus FOV-normalized pre-volume detection"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_modality_stratified_auroc(
    data: pd.DataFrame,
    output_path: Path,
    domain: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Plot modality-specific AUROC for raw, FOV-normalized, and post volumes."""
    data = data.copy()
    data["modality"] = data["scan_name"].str.extract(
        r"_(T1FS|T2FS|T1|T2)$", expand=False
    )
    if data["modality"].isna().any():
        raise ValueError(f"Cannot infer modality for some {domain} scans")
    variants = (
        ("Raw pre", "pre_{class_name}_volume_mm3", "#4c78a8"),
        ("FOV-normalized pre", "pre_{class_name}_fov_normalized", "#72b7b2"),
        ("Post", "post_{class_name}_volume_mm3", "#e45756"),
    )
    titles = {
        "endometrioma": "Endometrioma",
        "ovary": "Ovary",
        "uterus": "Uterus",
    }
    modalities = sorted(data["modality"].unique())
    rows = []
    for class_index, class_name in enumerate(ANATOMY_CHANNELS):
        for modality_index, modality in enumerate(modalities):
            subset = data.loc[data["modality"] == modality]
            y_true = subset["endometrioma_label"].to_numpy(dtype=int)
            if np.unique(y_true).size != 2:
                continue
            for variant_index, (variant, template, _) in enumerate(variants):
                score_column = template.format(class_name=class_name)
                scores = subset[score_column].to_numpy(dtype=float)
                auc = roc_auc_score(y_true, scores)
                ci_low, ci_high = bootstrap_auc_ci(
                    y_true,
                    scores,
                    n_bootstrap,
                    random_seed
                    + 100 * class_index
                    + 10 * modality_index
                    + variant_index,
                )
                rows.append(
                    {
                        "domain": domain,
                        "modality": modality,
                        "class": class_name,
                        "variant": variant,
                        "score": score_column,
                        "auroc": auc,
                        "ci_2.5%": ci_low,
                        "ci_97.5%": ci_high,
                        "n_scans": len(subset),
                        "n_positive": int(y_true.sum()),
                        "n_negative": int((1 - y_true).sum()),
                    }
                )
    results = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = np.arange(len(modalities), dtype=float)
    width = 0.24
    for class_index, (axis, class_name) in enumerate(
        zip(axes, ANATOMY_CHANNELS)
    ):
        for variant_index, (variant, _, color) in enumerate(variants):
            subset = (
                results.loc[
                    (results["class"] == class_name)
                    & (results["variant"] == variant)
                ]
                .set_index("modality")
                .reindex(modalities)
            )
            values = subset["auroc"].to_numpy(dtype=float)
            lower = np.maximum(
                0.0, values - subset["ci_2.5%"].to_numpy(dtype=float)
            )
            upper = np.maximum(
                0.0, subset["ci_97.5%"].to_numpy(dtype=float) - values
            )
            axis.bar(
                x + (variant_index - 1) * width,
                values,
                width,
                color=color,
                label=variant,
                yerr=np.vstack([lower, upper]),
                capsize=3,
                alpha=0.85,
            )
        axis.axhline(0.5, color="0.35", linestyle=":", linewidth=1.5)
        axis.set_xticks(x, modalities)
        axis.set_ylim(0.0, 1.02)
        axis.set_title(titles[class_name])
        axis.set_xlabel("Modality")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("AUROC (bootstrap 95% CI)")
    fig.suptitle(f"{domain}: modality-stratified detection performance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return results


def analyze_probability_channel_correlations(
    probability_dir: Path,
    csv_path: Path,
    output_dir: Path,
    suffix: str,
) -> pd.DataFrame:
    """Measure per-scan voxelwise Pearson correlation between anatomy channels."""
    presence = pd.read_csv(csv_path)
    pairs = (
        ("endometrioma", "ovary"),
        ("endometrioma", "uterus"),
        ("ovary", "uterus"),
    )
    rows = []
    for _, record in presence.iterrows():
        scan_name = str(record["scan_name"])
        probability_path = probability_dir / f"{scan_name}{suffix}"
        if not probability_path.exists():
            LOGGER.warning("Missing probability map for correlation: %s", scan_name)
            continue
        probability_nii = nib.as_closest_canonical(nib.load(str(probability_path)))
        probability = np.asanyarray(
            probability_nii.dataobj[..., :3], dtype=np.float32
        )
        probability /= probability_scale(probability_nii)
        domain = "D1" if str(record["center"]).startswith("D1") else "D2"
        modality = next(
            (
                token
                for token in reversed(scan_name.upper().split("_"))
                if token in {"T1", "T2", "T1FS", "T2FS", "CT"}
            ),
            infer_modality(scan_name),
        )
        for first, second in pairs:
            first_values = probability[
                ..., ANATOMY_CHANNELS[first]
            ].reshape(-1)
            second_values = probability[
                ..., ANATOMY_CHANNELS[second]
            ].reshape(-1)
            first_centered = first_values - first_values.mean()
            second_centered = second_values - second_values.mean()
            denominator = float(
                np.sqrt(
                    np.dot(first_centered, first_centered)
                    * np.dot(second_centered, second_centered)
                )
            )
            correlation = (
                float(np.dot(first_centered, second_centered) / denominator)
                if denominator > 0
                else np.nan
            )
            rows.append(
                {
                    "scan_name": scan_name,
                    "case_id": record.get("case_id", scan_name),
                    "domain": domain,
                    "modality": modality,
                    "first_class": first,
                    "second_class": second,
                    "pair": f"{first} vs {second}",
                    "pearson_correlation": correlation,
                    "n_voxels": int(first_values.size),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "probability_channel_correlations.csv", index=False
    )
    if results.empty:
        return results

    pair_order = [f"{first} vs {second}" for first, second in pairs]
    pair_labels = [
        "Endometrioma\nvs ovary",
        "Endometrioma\nvs uterus",
        "Ovary\nvs uterus",
    ]
    colors = {"D1": "#4c78a8", "D2": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    rng = np.random.default_rng(42)
    for axis, domain in zip(axes, ("D1", "D2")):
        domain_data = results.loc[results["domain"] == domain]
        distributions = [
            domain_data.loc[
                domain_data["pair"] == pair, "pearson_correlation"
            ].dropna().to_numpy(dtype=float)
            for pair in pair_order
        ]
        box = axis.boxplot(
            distributions,
            labels=pair_labels,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[domain])
            patch.set_alpha(0.55)
        for index, values in enumerate(distributions, start=1):
            jitter = rng.normal(0, 0.045, size=len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=11,
                color=colors[domain],
                alpha=0.32,
                edgecolors="none",
            )
            if len(values):
                axis.text(
                    index,
                    min(1.04, float(np.nanmax(values)) + 0.055),
                    f"n={len(values)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.axhline(0, color="0.35", linestyle=":", linewidth=1)
        axis.set_title(domain)
        axis.set_ylim(-0.15, 1.08)
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlabel("Probability-map pair")
    axes[0].set_ylabel("Voxelwise Pearson correlation")
    fig.suptitle(
        "Cross-channel probability-map correlation by centre\n"
        "Each point is one scan; boxes show median and interquartile range"
    )
    fig.tight_layout()
    output_path = output_dir / "probability_channel_correlations.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return results


def analyze_gt_recall_by_domain_modality(
    probability_dir: Path,
    labels_dir: Path,
    csv_path: Path,
    output_dir: Path,
    suffix: str,
    operating_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure annotated-organ voxel recall across thresholds and modalities.

    Only explicitly annotated class/scan pairs are included. A case-level
    endometrioma label is deliberately not treated as a scan-level mask.
    """
    class_flags = {"endometrioma": "em", "ovary": "ov", "uterus": "ut"}
    thresholds = np.unique(
        np.append(
            np.round(np.arange(0.05, 0.951, 0.05), 2),
            float(operating_threshold),
        )
    )
    presence = pd.read_csv(csv_path)
    presence["domain"] = presence["center"].astype(str).map(
        lambda value: "D1" if value.startswith("D1") else "D2"
    )
    presence["modality"] = presence["scan_name"].astype(str).map(
        lambda scan_name: next(
            (
                token
                for token in reversed(scan_name.upper().split("_"))
                if token in {"T1", "T2", "T1FS", "T2FS", "CT"}
            ),
            "MR",
        )
    )
    coverage_rows = []
    for class_name, flag in class_flags.items():
        for (domain, modality), group in presence.groupby(
            ["domain", "modality"]
        ):
            annotated = int(group[flag].fillna(0).astype(int).sum())
            total = int(group["scan_name"].nunique())
            coverage_rows.append(
                {
                    "class": class_name,
                    "domain": domain,
                    "modality": modality,
                    "total_scans": total,
                    "annotated_scans": annotated,
                    "unannotated_scans": total - annotated,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        output_dir / "annotation_coverage_summary.csv", index=False
    )

    modalities = ("T1", "T2", "T1FS", "T2FS")
    class_names = tuple(ANATOMY_CHANNELS)
    class_titles = {
        "endometrioma": "Endometrioma",
        "ovary": "Ovary",
        "uterus": "Uterus",
    }
    colors = {"D1": "#4c78a8", "D2": "#e45756"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    x = np.arange(len(modalities), dtype=float)
    width = 0.34
    for axis, class_name in zip(axes, class_names):
        for domain_index, domain in enumerate(("D1", "D2")):
            subset = (
                coverage.loc[
                    (coverage["class"] == class_name)
                    & (coverage["domain"] == domain)
                ]
                .set_index("modality")
                .reindex(modalities)
            )
            positions = x + (domain_index - 0.5) * width
            annotated = subset["annotated_scans"].fillna(0).to_numpy(dtype=float)
            unannotated = (
                subset["unannotated_scans"].fillna(0).to_numpy(dtype=float)
            )
            annotated_bars = axis.bar(
                positions,
                annotated,
                width,
                color=colors[domain],
                label=f"{domain}: annotated",
                alpha=0.88,
            )
            unannotated_bars = axis.bar(
                positions,
                unannotated,
                width,
                bottom=annotated,
                color=colors[domain],
                edgecolor=colors[domain],
                hatch="//",
                label=f"{domain}: unannotated",
                alpha=0.22,
            )
            for annotated_bar, unannotated_bar, (_, row) in zip(
                annotated_bars, unannotated_bars, subset.iterrows()
            ):
                if pd.isna(row["total_scans"]):
                    continue
                annotated_count = int(row["annotated_scans"])
                unannotated_count = int(row["unannotated_scans"])
                total = int(row["total_scans"])
                if annotated_count:
                    axis.text(
                        annotated_bar.get_x() + annotated_bar.get_width() / 2,
                        annotated_count / 2,
                        str(annotated_count),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                if unannotated_count:
                    axis.text(
                        unannotated_bar.get_x() + unannotated_bar.get_width() / 2,
                        annotated_count + unannotated_count / 2,
                        str(unannotated_count),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                axis.text(
                    annotated_bar.get_x() + annotated_bar.get_width() / 2,
                    total + max(0.4, total * 0.02),
                    f"total={total}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_title(class_titles[class_name])
        axis.set_xticks(x, modalities)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Scans")
    maximum_total = max(float(coverage["total_scans"].max()), 1.0)
    axes[0].set_ylim(0, maximum_total * 1.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Ground-truth annotation coverage by centre, modality, and class\n"
        "Stack height = all scans; solid = annotated; hatched = unannotated"
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    coverage_path = output_dir / "annotation_coverage_by_domain_modality.png"
    fig.savefig(coverage_path, dpi=220, bbox_inches="tight")
    fig.savefig(coverage_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    rows = []
    for _, record in presence.iterrows():
        annotated_classes = [
            class_name
            for class_name, flag in class_flags.items()
            if int(record.get(flag, 0)) == 1
        ]
        if not annotated_classes:
            continue
        scan_name = str(record["scan_name"])
        probability_path = probability_dir / f"{scan_name}{suffix}"
        label_path = labels_dir / f"{scan_name}_seg.nii.gz"
        if not probability_path.exists() or not label_path.exists():
            LOGGER.warning("Missing GT-recall input for %s", scan_name)
            continue
        probability_nii = nib.as_closest_canonical(nib.load(str(probability_path)))
        label_nii = nib.as_closest_canonical(nib.load(str(label_path)))
        probability = np.asanyarray(probability_nii.dataobj, dtype=np.float32)
        probability /= probability_scale(probability_nii)
        label = np.asanyarray(label_nii.dataobj)
        if probability.shape[:3] != label.shape:
            LOGGER.warning(
                "Skipping GT recall for %s: shape mismatch %s versus %s",
                scan_name,
                probability.shape[:3],
                label.shape,
            )
            continue
        if not np.allclose(probability_nii.affine, label_nii.affine, atol=1e-4):
            LOGGER.warning("Skipping GT recall for %s: affine mismatch", scan_name)
            continue
        domain = "D1" if str(record["center"]).startswith("D1") else "D2"
        modality = next(
            (
                token
                for token in reversed(scan_name.upper().split("_"))
                if token in {"T1", "T2", "T1FS", "T2FS", "CT"}
            ),
            infer_modality(scan_name),
        )
        for class_name in annotated_classes:
            gt_mask = label == GT_LABELS[class_name]
            if not gt_mask.any():
                LOGGER.warning(
                    "%s is flagged for %s but contains no label %d",
                    scan_name,
                    class_name,
                    GT_LABELS[class_name],
                )
                continue
            values = probability[..., ANATOMY_CHANNELS[class_name]][gt_mask]
            for threshold in thresholds:
                rows.append(
                    {
                        "scan_name": scan_name,
                        "case_id": record.get("case_id", scan_name),
                        "domain": domain,
                        "modality": modality,
                        "class": class_name,
                        "threshold": float(threshold),
                        "gt_voxels": int(gt_mask.sum()),
                        "gt_recall": float(np.mean(values >= threshold)),
                        "gt_probability_mean": float(values.mean()),
                        "gt_probability_median": float(np.median(values)),
                        "gt_zero_fraction": float(np.mean(values == 0)),
                    }
                )
    scan_results = pd.DataFrame(rows)
    if scan_results.empty:
        return scan_results, pd.DataFrame()
    summary = (
        scan_results.groupby(
            ["domain", "modality", "class", "threshold"], as_index=False
        )
        .agg(
            n_annotated_scans=("scan_name", "nunique"),
            recall_median=("gt_recall", "median"),
            recall_q25=("gt_recall", lambda values: values.quantile(0.25)),
            recall_q75=("gt_recall", lambda values: values.quantile(0.75)),
            probability_median=("gt_probability_median", "median"),
            zero_fraction_median=("gt_zero_fraction", "median"),
        )
    )
    scan_results.to_csv(output_dir / "gt_recall_scan_level.csv", index=False)
    summary.to_csv(output_dir / "gt_recall_summary.csv", index=False)

    fig, axes = plt.subplots(
        len(class_names),
        len(modalities),
        figsize=(18, 11),
        sharex=True,
        sharey=True,
    )
    for row_index, class_name in enumerate(class_names):
        for column_index, modality in enumerate(modalities):
            axis = axes[row_index, column_index]
            for domain in ("D1", "D2"):
                subset = summary.loc[
                    (summary["class"] == class_name)
                    & (summary["modality"] == modality)
                    & (summary["domain"] == domain)
                ].sort_values("threshold")
                if subset.empty:
                    continue
                x = subset["threshold"].to_numpy(dtype=float)
                y = subset["recall_median"].to_numpy(dtype=float)
                axis.plot(
                    x,
                    y,
                    color=colors[domain],
                    linewidth=2,
                    label=(
                        f"{domain} (n={int(subset['n_annotated_scans'].iloc[0])})"
                    ),
                )
                axis.fill_between(
                    x,
                    subset["recall_q25"].to_numpy(dtype=float),
                    subset["recall_q75"].to_numpy(dtype=float),
                    color=colors[domain],
                    alpha=0.16,
                )
            axis.axvline(
                operating_threshold,
                color="0.35",
                linestyle=":",
                linewidth=1,
            )
            axis.set_xlim(0.05, 0.95)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(modality)
            if column_index == 0:
                axis.set_ylabel(f"{class_titles[class_name]}\nGT recall")
            if row_index == len(class_names) - 1:
                axis.set_xlabel("Probability threshold")
            handles, labels = axis.get_legend_handles_labels()
            if labels:
                axis.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "Ground-truth voxel recall: cross-center and cross-modality robustness\n"
        "Median per scan with interquartile range; only explicitly annotated masks"
    )
    fig.tight_layout()
    curve_path = output_dir / "gt_recall_by_domain_modality.png"
    fig.savefig(curve_path, dpi=220, bbox_inches="tight")
    fig.savefig(curve_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    operating_scans = scan_results.loc[
        np.isclose(scan_results["threshold"], operating_threshold)
    ].copy()
    operating_scans["scan_recalled"] = operating_scans["gt_recall"] > 0
    at_half = (
        operating_scans.groupby(
            ["domain", "modality", "class"], as_index=False
        )
        .agg(
            annotated_scans=("scan_name", "nunique"),
            recalled_scans=("scan_recalled", "sum"),
            median_voxel_recall=("gt_recall", "median"),
        )
    )
    at_half["missed_scans"] = (
        at_half["annotated_scans"] - at_half["recalled_scans"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)
    x = np.arange(len(modalities), dtype=float)
    width = 0.34
    for axis, class_name in zip(axes, class_names):
        for domain_index, domain in enumerate(("D1", "D2")):
            subset = (
                at_half.loc[
                    (at_half["class"] == class_name)
                    & (at_half["domain"] == domain)
                ]
                .set_index("modality")
                .reindex(modalities)
            )
            positions = x + (domain_index - 0.5) * width
            recalled = subset["recalled_scans"].fillna(0).to_numpy(dtype=float)
            missed = subset["missed_scans"].fillna(0).to_numpy(dtype=float)
            recalled_bars = axis.bar(
                positions,
                recalled,
                width,
                color=colors[domain],
                label=f"{domain}: recalled",
                alpha=0.88,
            )
            missed_bars = axis.bar(
                positions,
                missed,
                width,
                bottom=recalled,
                color=colors[domain],
                edgecolor=colors[domain],
                hatch="//",
                label=f"{domain}: missed",
                alpha=0.22,
            )
            for recalled_bar, missed_bar, (_, row) in zip(
                recalled_bars, missed_bars, subset.iterrows()
            ):
                if pd.notna(row["annotated_scans"]):
                    total = int(row["annotated_scans"])
                    recalled_count = int(row["recalled_scans"])
                    missed_count = int(row["missed_scans"])
                    if recalled_count:
                        axis.text(
                            recalled_bar.get_x() + recalled_bar.get_width() / 2,
                            recalled_count / 2,
                            str(recalled_count),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
                    if missed_count:
                        axis.text(
                            missed_bar.get_x() + missed_bar.get_width() / 2,
                            recalled_count + missed_count / 2,
                            str(missed_count),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
                    axis.text(
                        recalled_bar.get_x() + recalled_bar.get_width() / 2,
                        total + max(0.35, total * 0.025),
                        (
                            f"total={total}\n"
                            f"median={row['median_voxel_recall']:.2f}"
                        ),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        axis.set_title(class_titles[class_name])
        axis.set_xticks(x, modalities)
        class_total = at_half.loc[
            at_half["class"] == class_name, "annotated_scans"
        ].max()
        axis.set_ylim(0, max(2, float(class_total) * 1.18))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Annotated scans")
    fig.suptitle(
        f"GT object recall at probability threshold {operating_threshold:g}\n"
        "Stack height = all annotated scans; solid = any GT overlap; "
        "hatched = completely missed"
    )
    fig.tight_layout()
    bar_path = output_dir / "gt_recall_at_operating_threshold.png"
    fig.savefig(bar_path, dpi=220, bbox_inches="tight")
    fig.savefig(bar_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return scan_results, summary


def build_review_dashboard(
    candidates: pd.DataFrame,
    priors: SpatialPriors,
    output_path: Path,
    gt_recall_summary: pd.DataFrame | None = None,
    operating_threshold: float = 0.5,
) -> None:
    """Build a self-contained local dashboard with scan and decision filters."""
    if candidates.empty:
        output_path.write_text(
            "<p>No endometrioma candidates were extracted.</p>", encoding="utf-8"
        )
        return
    stage_columns = [
        ("Stage 1: anatomical", "stage_1_anatomical_pass"),
        ("Stage 2: inter-class", "stage_2_inter_class_pass"),
        ("Stage 3: intra-class", "stage_3_intra_class_pass"),
    ]
    summary_rows = []
    for domain_name, domain_data in candidates.groupby("domain"):
        before = len(domain_data)
        for stage_name, pass_column in stage_columns:
            after = int(domain_data[pass_column].sum())
            summary_rows.extend(
                [
                    {
                        "domain": domain_name,
                        "stage": stage_name,
                        "checkpoint": "Before",
                        "candidates": before,
                    },
                    {
                        "domain": domain_name,
                        "stage": stage_name,
                        "checkpoint": "After",
                        "candidates": after,
                    },
                ]
            )
            before = after
    summary = pd.DataFrame(summary_rows)
    bar = px.bar(
        summary,
        x="stage",
        y="candidates",
        color="checkpoint",
        barmode="group",
        facet_col="domain",
        text="candidates",
        title="Candidate survival through the three rejection stages",
        category_orders={"checkpoint": ["Before", "After"]},
    )
    bar.update_layout(xaxis_title="Stage", yaxis_title="Candidates")
    # D1 and D2 contain different numbers of scans/candidates, so a shared
    # y-axis would compress the smaller domain and obscure its stage changes.
    bar.update_yaxes(matches=None, showticklabels=True)
    bar.update_traces(textposition="outside")
    plot_html = bar.to_html(full_html=False, include_plotlyjs="cdn")

    safe_records = candidates.replace({np.nan: None}).to_dict(orient="records")
    records_json = json.dumps(safe_records)
    priors_json = html.escape(json.dumps(asdict(priors), indent=2))
    analysis_figures = []
    for domain_name in ("D1", "D2"):
        for stem, title in (
            (
                f"pre_post_volume_distributions_{domain_name}.png",
                f"{domain_name}: pre/post physical volumes",
            ),
            (
                f"pre_post_sensitivity_specificity_{domain_name}.png",
                f"{domain_name}: pre/post sensitivity–specificity",
            ),
            (
                f"fov_sensitivity_specificity_{domain_name}.png",
                f"{domain_name}: FOV sensitivity–specificity",
            ),
            (
                f"raw_vs_fov_normalized_{domain_name}.png",
                f"{domain_name}: raw versus FOV-normalized pre volumes",
            ),
            (
                f"modality_stratified_auroc_{domain_name}.png",
                f"{domain_name}: modality-stratified AUROC",
            ),
        ):
            if (output_path.parent / stem).exists():
                analysis_figures.append(
                    f'<figure class="analysis-figure"><figcaption>{title}</figcaption>'
                    f'<img src="{stem}" alt="{title}"></figure>'
                )
    for stem, title in (
        (
            "gt_recall_by_domain_modality.png",
            "GT recall across centres, modalities, classes, and thresholds",
        ),
        (
            "gt_recall_at_operating_threshold.png",
            f"GT recall at the selected component threshold "
            f"({operating_threshold:g})",
        ),
        (
            "probability_channel_correlations.png",
            "Probability-map correlation between anatomical channels",
        ),
        (
            "annotation_coverage_by_domain_modality.png",
            "Dataset overview: scan and annotation coverage",
        ),
    ):
        if (output_path.parent / stem).exists():
            analysis_figures.insert(
                0,
                f'<figure class="analysis-figure gt-recall-figure">'
                f"<figcaption>{title}</figcaption>"
                f'<img src="{stem}" alt="{title}"></figure>',
            )
    figure_gallery = (
        '<section class="panel analysis-gallery">'
        + "".join(analysis_figures)
        + "</section>"
        if analysis_figures
        else ""
    )
    recall_table = ""
    if gt_recall_summary is not None and not gt_recall_summary.empty:
        table_thresholds = sorted({0.1, 0.5, 0.9, float(operating_threshold)})
        selected = gt_recall_summary.loc[
            gt_recall_summary["threshold"].isin(table_thresholds)
        ].copy()
        selected["recall"] = selected.apply(
            lambda row: (
                f"{row['recall_median']:.3f} "
                f"[{row['recall_q25']:.3f}, {row['recall_q75']:.3f}]"
            ),
            axis=1,
        )
        pivot = selected.pivot_table(
            index=["class", "domain", "modality", "n_annotated_scans"],
            columns="threshold",
            values="recall",
            aggfunc="first",
        ).reset_index()
        pivot = pivot.rename(
            columns={
                "class": "Class",
                "domain": "Domain",
                "modality": "Modality",
                "n_annotated_scans": "Annotated scans",
                **{
                    threshold: f"Recall @ {threshold:g}"
                    for threshold in table_thresholds
                },
            }
        )
        recall_table = (
            '<section class="panel recall-table"><h2>Ground-truth recall audit</h2>'
            f"<p><strong>Operating threshold: {operating_threshold:g}.</strong> "
            "Median [IQR] across explicitly annotated scans. Missing "
            "domain–modality combinations are not interpreted as negatives.</p>"
            + pivot.to_html(index=False, border=0, na_rep="—")
            + "</section>"
        )
    template = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Endometriosis: Do NOT fully trust FM – Question it</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f7fb;color:#172033}}
main{{max-width:1900px;margin:auto;padding:24px}} h1{{margin:0 0 8px}}
.threshold-banner{{display:inline-block;margin-top:10px;padding:8px 12px;background:#fff3cd;border:1px solid #e5c65c;border-radius:6px}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}
select{{padding:8px 10px;border:1px solid #c8d0df;border-radius:6px;background:white}}
.grid{{display:grid;grid-template-columns:minmax(900px,2fr) minmax(320px,1fr);gap:18px}}
.panel{{background:white;border:1px solid #dde3ee;border-radius:10px;padding:16px}}
section.panel{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}} th,td{{padding:7px;border-bottom:1px solid #e7ebf2;text-align:left}}
tr{{cursor:pointer}} tr:hover{{background:#eef4ff}} img{{width:100%;height:auto}}
.accepted{{color:#08783e;font-weight:600}} .rejected{{color:#a33a2b;font-weight:600}}
.stage-head{{text-align:center;background:#eaf0fa;border-left:2px solid white}} .pass{{color:#08783e;font-weight:700;font-size:16px}} .fail{{color:#b42318;font-weight:700;font-size:16px}}
.analysis-gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:18px 0}}
.gt-recall-figure{{grid-column:1/-1}} .analysis-figure{{margin:0}} .analysis-figure figcaption{{font-weight:600;margin-bottom:8px}}
.recall-table{{margin:18px 0}} .recall-table h2{{margin-top:0}}
.analysis-figure{{margin:0}} .analysis-figure figcaption{{font-weight:600;margin-bottom:8px}}
pre{{white-space:pre-wrap;font-size:12px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Endometriosis: Do NOT fully trust FM – Question it</h1>
<div>Hierarchical anatomical reasoning and statistical rejection</div>
<div class="threshold-banner">Operating probability threshold: <strong>{operating_threshold:g}</strong></div>
{plot_html}
{figure_gallery}
{recall_table}
<div class="controls">
<label>Domain <select id="domain"><option value="all">All</option><option>D1</option><option>D2</option></select></label>
<label>Decision <select id="decision"><option value="all">All</option><option value="accepted">Accepted</option><option value="rejected">Rejected</option></select></label>
<label>Scan <select id="scan"><option value="all">All scans</option></select></label>
</div>
<div class="grid"><section class="panel"><table><thead>
<tr><th rowspan="2">Scan</th><th rowspan="2">ID</th><th rowspan="2">Final</th><th rowspan="2">Reason</th><th class="stage-head" colspan="9">Stage 1 · Anatomical reasoning</th><th class="stage-head" colspan="7">Stage 2 · Inter-class rank statistics</th><th class="stage-head" colspan="6">Stage 3 · Intra-class equivalence</th></tr>
<tr><th>Pass</th><th>Path</th><th>Uterus repair</th><th>Morphology</th><th>Sphericity</th><th>Elongation</th><th>Uterus ΔS</th><th>Ovary ΔS</th><th>Ovary surface</th><th>Pass</th><th>q uterus</th><th>Effect uterus</th><th>q ovary</th><th>Effect ovary</th><th>q local</th><th>Effect local</th><th>Pass</th><th>Credible ref</th><th>Reference</th><th>Joint distance</th><th>q intra</th><th>Effect intra</th></tr>
</thead><tbody id="rows"></tbody></table></section>
<aside class="panel"><img id="preview" alt="Select a candidate to view its overlay"><pre id="detail">Select a candidate.</pre></aside></div>
<details class="panel" style="margin-top:18px"><summary>D1-derived spatial priors</summary><pre>{priors_json}</pre></details>
</main><script>
const records={records_json};
const domain=document.getElementById('domain'),decision=document.getElementById('decision'),scan=document.getElementById('scan');
[...new Set(records.map(r=>r.scan_name))].sort().forEach(s=>{{const o=document.createElement('option');o.value=o.textContent=s;scan.appendChild(o)}});
function fmt(v){{return v==null?'—':Number(v).toFixed(3)}}
function mark(v){{return `<span class="${{v?'pass':'fail'}}">${{v?'✓':'✕'}}</span>`}}
function selectRecord(r){{const image=document.getElementById('preview');image.src=r.preview?new URL(r.preview,window.location.href).href:'';document.getElementById('detail').textContent=JSON.stringify(r,null,2)}}
function render(){{const body=document.getElementById('rows');body.innerHTML='';const filtered=records.filter(r=>(domain.value==='all'||r.domain===domain.value)&&(scan.value==='all'||r.scan_name===scan.value)&&(decision.value==='all'||(decision.value==='accepted')===r.accepted));filtered.forEach(r=>{{const tr=document.createElement('tr');tr.innerHTML=`<td>${{r.scan_name}}</td><td>${{r.candidate_id}}</td><td class="${{r.accepted?'accepted':'rejected'}}">${{r.accepted?'Accepted':'Rejected'}}</td><td>${{r.rejection_reason}}</td><td>${{mark(r.stage_1_anatomical_pass)}}</td><td>${{r.reasoning_path}}</td><td>${{mark(r.uterus_refinement_accepted)}}</td><td>${{mark(r.morphology_plausible)}}</td><td>${{fmt(r.candidate_sphericity)}}</td><td>${{fmt(r.candidate_elongation)}}</td><td>${{fmt(r.endometrioma_delta_s_mm)}}</td><td>${{fmt(r.endometrioma_ovary_delta_s_mm)}}</td><td>${{fmt(r.surface_distance_to_ovary_mm)}}</td><td>${{mark(r.stage_2_inter_class_pass)}}</td><td>${{fmt(r.q_vs_uterus)}}</td><td>${{fmt(r.effect_vs_uterus)}}</td><td>${{fmt(r.q_vs_ovary)}}</td><td>${{fmt(r.effect_vs_ovary)}}</td><td>${{fmt(r.q_vs_local_ring)}}</td><td>${{fmt(r.effect_vs_local_ring)}}</td><td>${{mark(r.stage_3_intra_class_pass)}}</td><td>${{mark(r.reference_credible)}}</td><td>${{fmt(r.intra_reference_candidate_id)}}</td><td>${{fmt(r.joint_anchor_surface_distance_mm)}}</td><td>${{fmt(r.q_vs_reference_endometrioma)}}</td><td>${{fmt(r.effect_vs_reference_endometrioma)}}</td>`;tr.onclick=()=>selectRecord(r);body.appendChild(tr)}});const initial=filtered.find(r=>r.accepted)||filtered[0];if(initial)selectRecord(initial)}}
[domain,decision,scan].forEach(e=>e.addEventListener('change',render));render();
</script></body></html>"""
    output_path.write_text(template, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probability-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "segmentations" / "BiomedParse",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "class_presence.csv",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "images",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "labels",
        help="D1 annotations used only when the required masks are present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "endometriosis_detection",
    )
    parser.add_argument(
        "--suffix",
        default="_endometrioma.nii.gz",
        help="Filename suffix removed to obtain scan_name.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Target voxel spacing in millimetres (default: 1 1 1).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--component-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-component-volume-mm3", type=float, default=20.0)
    parser.add_argument(
        "--minimum-uterus-volume-mm3",
        type=float,
        default=5000.0,
        help="Reject individual uterus components smaller than this volume.",
    )
    parser.add_argument(
        "--max-endometrioma-ovary-distance-mm",
        type=float,
        default=30.0,
        help="Clinical safety cap applied after learning the D1 spatial prior.",
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-effect-size", type=float, default=0.2)
    parser.add_argument(
        "--local-ring-mm",
        type=float,
        default=5.0,
        help="Physical radius of the candidate-local comparison ring.",
    )
    parser.add_argument(
        "--reference-ovary-surface-max-mm",
        type=float,
        default=5.0,
        help="Maximum ovary surface distance for a credible reference candidate.",
    )
    parser.add_argument(
        "--intra-equivalence-effect-max",
        type=float,
        default=0.5,
        help="Maximum standardized rank-energy distance for Stage 3 equivalence.",
    )
    parser.add_argument(
        "--statistical-block-size-mm",
        type=float,
        default=3.0,
        help="Physical edge length of blocks used as permutation samples.",
    )
    parser.add_argument("--maximum-statistical-samples", type=int, default=1000)
    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save candidate overlay images used by the local dashboard.",
    )
    parser.add_argument(
        "--skip-volume-analysis",
        action="store_true",
        help="Skip volume/AUROC figures when only rejection analysis is needed.",
    )
    parser.add_argument(
        "--skip-hierarchical-rejection",
        action="store_true",
        help="Skip anatomical/statistical rejection and dashboard generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_spacing = tuple(args.target_spacing)
    if any(spacing <= 0 for spacing in target_spacing):
        raise ValueError("--target-spacing values must be positive")
    if not 0.0 < args.component_threshold < 1.0:
        raise ValueError("--component-threshold must be between 0 and 1")
    if not 0.0 < args.fdr_alpha < 1.0:
        raise ValueError("--fdr-alpha must be between 0 and 1")
    if args.permutations < 1 or args.statistical_block_size_mm <= 0:
        raise ValueError(
            "--permutations and --statistical-block-size-mm must be positive"
        )
    if args.max_endometrioma_ovary_distance_mm <= 0:
        raise ValueError("--max-endometrioma-ovary-distance-mm must be positive")
    if args.minimum_uterus_volume_mm3 <= 0:
        raise ValueError("--minimum-uterus-volume-mm3 must be positive")
    if (
        args.local_ring_mm <= 0
        or args.reference_ovary_surface_max_mm < 0
        or args.intra_equivalence_effect_max <= 0
    ):
        raise ValueError(
            "Local-ring, reference-distance, and equivalence settings are invalid"
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "component_threshold": args.component_threshold,
                "minimum_component_volume_mm3": (
                    args.minimum_component_volume_mm3
                ),
                "minimum_uterus_volume_mm3": args.minimum_uterus_volume_mm3,
                "fdr_alpha": args.fdr_alpha,
                "permutations": args.permutations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.skip_volume_analysis:
        measurements = collect_measurements(
            args.probability_dir,
            suffix=args.suffix,
            target_spacing=target_spacing,
        )
        analysis_data = join_labels(measurements, args.csv_path)
        measurements_path = args.output_dir / "segmentation_volumes.csv"
        analysis_data.to_csv(measurements_path, index=False)

        domain_results = []
        for domain in ("D1", "D2"):
            domain_data = analysis_data.loc[analysis_data["domain"] == domain].copy()
            if domain_data.empty:
                raise ValueError(f"No scans found for {domain}")
            figure_path = (
                args.output_dir / f"sensitivity_specificity_by_class_{domain}.png"
            )
            domain_results.append(
                plot_detection_curves(
                    domain_data,
                    output_path=figure_path,
                    domain=domain,
                    n_bootstrap=args.bootstrap,
                    random_seed=args.random_seed,
                )
            )
            LOGGER.info("Saved %s figure: %s", domain, figure_path)

            violin_path = (
                args.output_dir / f"volume_distributions_by_class_{domain}.png"
            )
            plot_volume_distributions(
                domain_data,
                output_path=violin_path,
                domain=domain,
            )
            LOGGER.info("Saved %s violin figure: %s", domain, violin_path)

        auc_results = pd.concat(domain_results, ignore_index=True)
        auc_path = args.output_dir / "auroc_by_class.csv"
        auc_results.to_csv(auc_path, index=False)
        LOGGER.info("Saved measurements: %s", measurements_path)
        LOGGER.info("Saved AUROC results: %s", auc_path)

    if not args.skip_hierarchical_rejection:
        candidates, priors = run_hierarchical_rejection(
            images_dir=args.images_dir,
            probability_dir=args.probability_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            suffix=args.suffix,
            threshold=args.component_threshold,
            minimum_volume_mm3=args.minimum_component_volume_mm3,
            minimum_uterus_volume_mm3=args.minimum_uterus_volume_mm3,
            local_ring_mm=args.local_ring_mm,
            reference_ovary_surface_max_mm=args.reference_ovary_surface_max_mm,
            intra_equivalence_effect_max=args.intra_equivalence_effect_max,
            endometrioma_distance_cap_mm=args.max_endometrioma_ovary_distance_mm,
            permutations=args.permutations,
            fdr_alpha=args.fdr_alpha,
            minimum_effect_size=args.minimum_effect_size,
            block_size_mm=args.statistical_block_size_mm,
            maximum_samples=args.maximum_statistical_samples,
            random_seed=args.random_seed,
            save_previews=args.save_previews,
        )
        pre_volume_path = args.output_dir / "segmentation_volumes.csv"
        if pre_volume_path.exists():
            pre_volume_data = pd.read_csv(pre_volume_path)
            pre_post_volumes = collect_pre_post_reasoning_volumes(
                pre_volume_data,
                candidates,
                threshold=args.component_threshold,
                minimum_volume_mm3=args.minimum_component_volume_mm3,
            )
            pre_post_path = args.output_dir / "pre_post_reasoning_volumes.csv"
            pre_post_volumes.to_csv(pre_post_path, index=False)
            comparison_results = []
            for domain in ("D1", "D2"):
                domain_data = pre_post_volumes.loc[
                    pre_post_volumes["domain"] == domain
                ].copy()
                if domain_data.empty:
                    continue
                plot_pre_post_volume_distributions(
                    domain_data,
                    args.output_dir
                    / f"pre_post_volume_distributions_{domain}.png",
                    domain,
                )
                comparison_results.append(
                    plot_pre_post_detection_curves(
                        domain_data,
                        args.output_dir
                        / f"pre_post_sensitivity_specificity_{domain}.png",
                        domain,
                    )
                )
            if comparison_results:
                pd.concat(comparison_results, ignore_index=True).to_csv(
                    args.output_dir / "pre_post_auroc.csv", index=False
                )
            normalization_results = []
            modality_results = []
            for domain in ("D1", "D2"):
                domain_data = pre_post_volumes.loc[
                    pre_post_volumes["domain"] == domain
                ].copy()
                if domain_data.empty:
                    continue
                normalization_results.append(
                    plot_raw_vs_fov_normalized_curves(
                        domain_data,
                        args.output_dir
                        / f"raw_vs_fov_normalized_{domain}.png",
                        domain,
                    )
                )
                modality_results.append(
                    plot_modality_stratified_auroc(
                        domain_data,
                        args.output_dir
                        / f"modality_stratified_auroc_{domain}.png",
                        domain,
                    )
                )
            if normalization_results:
                pd.concat(normalization_results, ignore_index=True).to_csv(
                    args.output_dir / "fov_normalized_pre_auroc.csv",
                    index=False,
                )
            if modality_results:
                pd.concat(modality_results, ignore_index=True).to_csv(
                    args.output_dir / "modality_stratified_auroc.csv",
                    index=False,
                )
            if "field_of_view_volume_mm3" in pre_volume_data.columns:
                fov_results = []
                for domain in ("D1", "D2"):
                    domain_data = pre_volume_data.loc[
                        pre_volume_data["domain"] == domain
                    ].copy()
                    if domain_data.empty:
                        continue
                    fov_results.append(
                        plot_fov_detection_curves(
                            domain_data,
                            args.output_dir
                            / f"fov_sensitivity_specificity_{domain}.png",
                            domain,
                        )
                    )
                if fov_results:
                    pd.concat(fov_results, ignore_index=True).to_csv(
                        args.output_dir / "fov_auroc.csv", index=False
                    )
            LOGGER.info("Saved pre/post scan volumes: %s", pre_post_path)
        else:
            LOGGER.warning(
                "Cannot create pre/post figures: %s does not exist",
                pre_volume_path,
            )
        analyze_probability_channel_correlations(
            probability_dir=args.probability_dir,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            suffix=args.suffix,
        )
        _, gt_recall_summary = analyze_gt_recall_by_domain_modality(
            probability_dir=args.probability_dir,
            labels_dir=args.labels_dir,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            suffix=args.suffix,
            operating_threshold=args.component_threshold,
        )
        dashboard_path = args.output_dir / "candidate_review_dashboard.html"
        build_review_dashboard(
            candidates,
            priors,
            dashboard_path,
            gt_recall_summary=gt_recall_summary,
            operating_threshold=args.component_threshold,
        )
        LOGGER.info("Saved candidate results: %s", args.output_dir / "candidate_rejection_results.csv")
        LOGGER.info("Saved local review dashboard: %s", dashboard_path)


if __name__ == "__main__":
    main()
