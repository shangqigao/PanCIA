"""Scan-level endometrioma detection from six-channel segmentation volumes.

For every probability NIfTI, this script resamples each channel to 1 mm
isotropic resolution, integrates its probabilities to obtain a physical volume,
and evaluates that volume against ``endometrioma_label``. D1 and D2 are analysed
separately to distinguish training-domain from domain-generalization results.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to, resample_to_output
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_spacing = tuple(args.target_spacing)
    if any(spacing <= 0 for spacing in target_spacing):
        raise ValueError("--target-spacing values must be positive")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

        violin_path = args.output_dir / f"volume_distributions_by_class_{domain}.png"
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


if __name__ == "__main__":
    main()
