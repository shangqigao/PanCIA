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
    normalized_image = normalize_image(image, infer_modality(scan_name))
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
            "endometrioma_delta_r_mm": np.nan,
            "endometrioma_delta_a_mm": np.nan,
            "endometrioma_delta_s_mm": np.nan,
            "surface_distance_to_ovary_mm": np.nan,
            "surface_distance_to_uterus_mm": np.nan,
            "joint_anchor_surface_distance_mm": np.nan,
            "matched_ovary_component_id": np.nan,
            "distance_to_matched_ovary_centroid_mm": np.nan,
            "endometrioma_ovary_delta_r_mm": np.nan,
            "endometrioma_ovary_delta_a_mm": np.nan,
            "endometrioma_ovary_delta_s_mm": np.nan,
            "p_vs_ovary": np.nan,
            "q_vs_ovary": np.nan,
            "effect_vs_ovary": np.nan,
            "p_vs_uterus": np.nan,
            "q_vs_uterus": np.nan,
            "effect_vs_uterus": np.nan,
            "uterus_reference_erosion_iterations": np.nan,
            "uterus_reference_blocks": 0,
            "ovary_reference_erosion_iterations": np.nan,
            "ovary_reference_blocks": 0,
            "accepted": False,
            "preliminary_accepted": False,
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
            if not uterus_relation["endometrioma_uterus_relation_plausible"]:
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
                    normalized_image,
                    candidate_mask,
                    block_size_mm,
                    voxel_spacing_mm,
                )
                uterus_values, uterus_erosion, uterus_blocks = (
                    adaptive_reference_values(
                        normalized_image,
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
                            normalized_image,
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
        raw_p_values.extend([record["p_vs_ovary"], record["p_vs_uterus"]])
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
        record["q_vs_ovary"] = adjusted[2 * index]
        record["q_vs_uterus"] = adjusted[2 * index + 1]
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
        survivors,
        key=reference_rank,
    )
    reference_id = reference["candidate_id"]
    reference_mask = maps["endometrioma"] == reference_id
    reference_values = _block_values(
        normalized_image,
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
            normalized_image,
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


def build_review_dashboard(
    candidates: pd.DataFrame,
    priors: SpatialPriors,
    output_path: Path,
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
    bar.update_traces(textposition="outside")
    plot_html = bar.to_html(full_html=False, include_plotlyjs="cdn")

    safe_records = candidates.replace({np.nan: None}).to_dict(orient="records")
    records_json = json.dumps(safe_records)
    priors_json = html.escape(json.dumps(asdict(priors), indent=2))
    template = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Endometrioma candidate review</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f7fb;color:#172033}}
main{{max-width:1900px;margin:auto;padding:24px}} h1{{margin:0 0 8px}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}
select{{padding:8px 10px;border:1px solid #c8d0df;border-radius:6px;background:white}}
.grid{{display:grid;grid-template-columns:minmax(900px,2fr) minmax(320px,1fr);gap:18px}}
.panel{{background:white;border:1px solid #dde3ee;border-radius:10px;padding:16px}}
section.panel{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}} th,td{{padding:7px;border-bottom:1px solid #e7ebf2;text-align:left}}
tr{{cursor:pointer}} tr:hover{{background:#eef4ff}} img{{width:100%;height:auto}}
.accepted{{color:#08783e;font-weight:600}} .rejected{{color:#a33a2b;font-weight:600}}
.stage-head{{text-align:center;background:#eaf0fa;border-left:2px solid white}} .pass{{color:#08783e;font-weight:700;font-size:16px}} .fail{{color:#b42318;font-weight:700;font-size:16px}}
pre{{white-space:pre-wrap;font-size:12px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Endometrioma candidate review</h1>
<div>Uterus → ovary → endometrioma reasoning with per-scan permutation FDR.</div>
{plot_html}
<div class="controls">
<label>Domain <select id="domain"><option value="all">All</option><option>D1</option><option>D2</option></select></label>
<label>Decision <select id="decision"><option value="all">All</option><option value="accepted">Accepted</option><option value="rejected">Rejected</option></select></label>
<label>Scan <select id="scan"><option value="all">All scans</option></select></label>
</div>
<div class="grid"><section class="panel"><table><thead>
<tr><th rowspan="2">Scan</th><th rowspan="2">ID</th><th rowspan="2">Final</th><th rowspan="2">Reason</th><th class="stage-head" colspan="5">Stage 1 · Anatomical reasoning</th><th class="stage-head" colspan="5">Stage 2 · Inter-class statistics</th><th class="stage-head" colspan="5">Stage 3 · Intra-class statistics</th></tr>
<tr><th>Pass</th><th>Path</th><th>Uterus ΔS</th><th>Ovary ΔS</th><th>Ovary surface</th><th>Pass</th><th>q uterus</th><th>Effect uterus</th><th>q ovary</th><th>Effect ovary</th><th>Pass</th><th>Reference</th><th>Joint distance</th><th>q intra</th><th>Effect intra</th></tr>
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
function render(){{const body=document.getElementById('rows');body.innerHTML='';const filtered=records.filter(r=>(domain.value==='all'||r.domain===domain.value)&&(scan.value==='all'||r.scan_name===scan.value)&&(decision.value==='all'||(decision.value==='accepted')===r.accepted));filtered.forEach(r=>{{const tr=document.createElement('tr');tr.innerHTML=`<td>${{r.scan_name}}</td><td>${{r.candidate_id}}</td><td class="${{r.accepted?'accepted':'rejected'}}">${{r.accepted?'Accepted':'Rejected'}}</td><td>${{r.rejection_reason}}</td><td>${{mark(r.stage_1_anatomical_pass)}}</td><td>${{r.reasoning_path}}</td><td>${{fmt(r.endometrioma_delta_s_mm)}}</td><td>${{fmt(r.endometrioma_ovary_delta_s_mm)}}</td><td>${{fmt(r.surface_distance_to_ovary_mm)}}</td><td>${{mark(r.stage_2_inter_class_pass)}}</td><td>${{fmt(r.q_vs_uterus)}}</td><td>${{fmt(r.effect_vs_uterus)}}</td><td>${{fmt(r.q_vs_ovary)}}</td><td>${{fmt(r.effect_vs_ovary)}}</td><td>${{mark(r.stage_3_intra_class_pass)}}</td><td>${{fmt(r.intra_reference_candidate_id)}}</td><td>${{fmt(r.joint_anchor_surface_distance_mm)}}</td><td>${{fmt(r.q_vs_reference_endometrioma)}}</td><td>${{fmt(r.effect_vs_reference_endometrioma)}}</td>`;tr.onclick=()=>selectRecord(r);body.appendChild(tr)}});const initial=filtered.find(r=>r.accepted)||filtered[0];if(initial)selectRecord(initial)}}
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
            endometrioma_distance_cap_mm=args.max_endometrioma_ovary_distance_mm,
            permutations=args.permutations,
            fdr_alpha=args.fdr_alpha,
            minimum_effect_size=args.minimum_effect_size,
            block_size_mm=args.statistical_block_size_mm,
            maximum_samples=args.maximum_statistical_samples,
            random_seed=args.random_seed,
            save_previews=args.save_previews,
        )
        dashboard_path = args.output_dir / "candidate_review_dashboard.html"
        build_review_dashboard(candidates, priors, dashboard_path)
        LOGGER.info("Saved candidate results: %s", args.output_dir / "candidate_rejection_results.csv")
        LOGGER.info("Saved local review dashboard: %s", dashboard_path)


if __name__ == "__main__":
    main()
