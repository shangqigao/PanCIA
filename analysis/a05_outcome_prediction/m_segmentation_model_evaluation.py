"""Compare six EndoMRI segmentation models at a fixed probability threshold.

The evaluation is strictly scan-level. Cohorts are the 44 training and 12
internal-test scans from EndoMRI/dataset.json plus 85 D2 scans having at least
one foreground annotation. Endometrioma is evaluated on every cohort scan and
also summarized on positive scans only. Ovary and uterus are evaluated only
when their explicit masks are present.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon


LOGGER = logging.getLogger(__name__)
DEFAULT_ALL_ROOT = Path(
    "/Users/sg2162/Datasets/CancerDatasets/Endometriosis/EndoMRI_All"
)
DEFAULT_SPLIT = Path(
    "/Users/sg2162/Datasets/CancerDatasets/Endometriosis/EndoMRI/dataset.json"
)
MODEL_DIRS = {
    "baseline": "segmentations",
    "r3": "segmentations_r3",
    "r4": "segmentations_r4",
    "r5": "segmentations_r5",
    "r6": "segmentations_r6",
    "r7": "segmentations_r7",
}
CLASS_INFO = {
    "endometrioma": {"channel": 0, "label": 1, "flag": "em"},
    "ovary": {"channel": 1, "label": 3, "flag": "ov"},
    "uterus": {"channel": 2, "label": 4, "flag": "ut"},
}
COHORT_ORDER = ["training", "internal_test", "external_test"]
MODEL_ORDER = list(MODEL_DIRS)


def probability_scale(nii: nib.spatialimages.SpatialImage) -> float:
    dtype = np.dtype(nii.get_data_dtype())
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0


def load_probabilities_on_gt_grid(
    probability_path: Path,
    gt_nii: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, bool]:
    """Load all six channels as [0, 1] probabilities on the GT spatial grid."""
    source = nib.load(str(probability_path))
    divisor = probability_scale(source)
    source = nib.as_closest_canonical(source)
    if len(source.shape) != 4 or source.shape[-1] != 6:
        raise ValueError(f"invalid_shape_{source.shape}")
    probabilities = np.asanyarray(source.dataobj, dtype=np.float32) / divisor
    same_grid = source.shape[:3] == gt_nii.shape and np.allclose(
        source.affine, gt_nii.affine, atol=1e-4
    )
    if same_grid:
        return probabilities, False
    channels = []
    for channel in range(probabilities.shape[-1]):
        channel_nii = nib.Nifti1Image(probabilities[..., channel], source.affine)
        aligned = resample_from_to(
            channel_nii, (gt_nii.shape, gt_nii.affine), order=1
        )
        channels.append(np.asanyarray(aligned.dataobj, dtype=np.float32))
    return np.stack(channels, axis=-1), True


def scan_name(entry: dict) -> str:
    return Path(entry["image"]).name.removesuffix(".nii.gz")


def modality_from_name(name: str) -> str:
    return name.rsplit("_", 1)[-1]


def build_manifest(all_root: Path, split_path: Path) -> pd.DataFrame:
    split = json.loads(split_path.read_text())
    presence = pd.read_csv(all_root / "class_presence.csv")
    flags = ["em", "cy", "ov", "ut", "cds"]
    presence["any_class_annotated"] = (
        presence[flags].fillna(0).astype(int).any(axis=1)
    )
    training = {scan_name(item) for item in split["train"]}
    internal = {scan_name(item) for item in split["test"]}

    def cohort(row: pd.Series) -> str | None:
        name = row["scan_name"]
        if name in training:
            return "training"
        if name in internal:
            return "internal_test"
        if name.startswith("D2-") and row["any_class_annotated"]:
            return "external_test"
        return None

    presence["cohort"] = presence.apply(cohort, axis=1)
    manifest = presence.loc[presence["cohort"].notna()].copy()
    manifest["modality"] = manifest["scan_name"].map(modality_from_name)
    manifest["gt_path"] = manifest["scan_name"].map(
        lambda value: str(all_root / "labels" / f"{value}_seg.nii.gz")
    )
    if manifest["scan_name"].duplicated().any():
        raise ValueError("Evaluation manifest contains duplicate scan names")
    expected = {"training": 44, "internal_test": 12, "external_test": 85}
    observed = manifest["cohort"].value_counts().to_dict()
    if any(observed.get(key, 0) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected cohort counts: {observed}; expected {expected}")
    return manifest


def surface_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    surface = mask & ~ndimage.binary_erosion(mask, structure=structure)
    points = np.argwhere(surface)
    return nib.affines.apply_affine(affine, points)


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    affine: np.ndarray,
    tolerance_mm: float,
    max_points: int,
) -> tuple[float, float, float, bool]:
    if not prediction.any() or not target.any():
        return np.nan, np.nan, np.nan, False
    pred_points = surface_points(prediction, affine)
    gt_points = surface_points(target, affine)
    sampled = len(pred_points) > max_points or len(gt_points) > max_points
    if len(pred_points) > max_points:
        pred_points = pred_points[np.linspace(
            0, len(pred_points) - 1, max_points, dtype=int
        )]
    if len(gt_points) > max_points:
        gt_points = gt_points[np.linspace(
            0, len(gt_points) - 1, max_points, dtype=int
        )]
    pred_to_gt = cKDTree(gt_points).query(pred_points, workers=-1)[0]
    gt_to_pred = cKDTree(pred_points).query(gt_points, workers=-1)[0]
    hd95 = float(np.percentile(np.concatenate([pred_to_gt, gt_to_pred]), 95))
    assd = float((pred_to_gt.sum() + gt_to_pred.sum()) / (
        len(pred_to_gt) + len(gt_to_pred)
    ))
    surface_dice = float(
        ((pred_to_gt <= tolerance_mm).sum() + (gt_to_pred <= tolerance_mm).sum())
        / (len(pred_to_gt) + len(gt_to_pred))
    )
    return hd95, assd, surface_dice, sampled


def lesion_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    structure = ndimage.generate_binary_structure(3, 3)
    pred_labels, n_pred = ndimage.label(prediction, structure=structure)
    gt_labels, n_gt = ndimage.label(target, structure=structure)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    overlap = prediction & target
    if overlap.any():
        pairs = np.unique(
            np.stack([pred_labels[overlap], gt_labels[overlap]], axis=1), axis=0
        )
        for pred_id, gt_id in pairs:
            if pred_id > 0 and gt_id > 0:
                matched_pred.add(int(pred_id))
                matched_gt.add(int(gt_id))
    return {
        "n_predicted_lesions": int(n_pred),
        "n_gt_lesions": int(n_gt),
        "detected_gt_lesions": len(matched_gt),
        "missed_gt_lesions": int(n_gt) - len(matched_gt),
        "false_positive_lesions": int(n_pred) - len(matched_pred),
        "lesion_sensitivity": len(matched_gt) / n_gt if n_gt else np.nan,
        "lesion_precision": len(matched_pred) / n_pred if n_pred else np.nan,
    }


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_volume_mm3: float,
) -> dict[str, float | str]:
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(prediction.size - tp - fp - fn)
    pred_count = tp + fp
    gt_count = tp + fn
    if pred_count == 0 and gt_count == 0:
        status, dice, iou = "both_empty", 1.0, 1.0
    elif pred_count == 0:
        status, dice, iou = "prediction_empty", 0.0, 0.0
    elif gt_count == 0:
        status, dice, iou = "gt_empty", 0.0, 0.0
    else:
        status = "both_nonempty"
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    return {
        "mask_status": status,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "predicted_volume_mm3": pred_count * voxel_volume_mm3,
        "gt_volume_mm3": gt_count * voxel_volume_mm3,
        "signed_volume_error_mm3": (pred_count - gt_count) * voxel_volume_mm3,
        "absolute_volume_error_mm3": abs(pred_count - gt_count) * voxel_volume_mm3,
        "relative_volume_error": (
            abs(pred_count - gt_count) / gt_count if gt_count else np.nan
        ),
        "false_positive_volume_mm3": fp * voxel_volume_mm3,
    }


def evaluate(
    manifest: pd.DataFrame,
    all_root: Path,
    threshold: float,
    surface_tolerance_mm: float,
    max_surface_points: int,
    probability_dirs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    lesion_rows = []
    if probability_dirs is None:
        probability_dirs = {
            model: all_root / directory / "BiomedParse"
            for model, directory in MODEL_DIRS.items()
        }
    total = len(probability_dirs) * len(manifest)
    completed = 0
    for model, probability_dir in probability_dirs.items():
        for record in manifest.itertuples(index=False):
            completed += 1
            if completed % 25 == 0 or completed == total:
                LOGGER.info("Evaluating %d/%d", completed, total)
            probability_path = probability_dir / f"{record.scan_name}_endometrioma.nii.gz"
            gt_path = Path(record.gt_path)
            base = {
                "model": model,
                "cohort": record.cohort,
                "scan_name": record.scan_name,
                "case_id": record.case_id,
                "modality": record.modality,
                "probability_path": str(probability_path),
                "gt_path": str(gt_path),
            }
            if not probability_path.exists() or not gt_path.exists():
                metric_rows.append({**base, "class": "all", "eligible": False,
                                    "validation_status": "missing_file"})
                continue
            gt_nii = nib.as_closest_canonical(nib.load(str(gt_path)))
            try:
                probabilities, resampled = load_probabilities_on_gt_grid(
                    probability_path, gt_nii
                )
            except ValueError as error:
                metric_rows.append({**base, "class": "all", "eligible": False,
                                    "validation_status": str(error)})
                continue
            gt = np.asanyarray(gt_nii.dataobj)
            voxel_volume = float(abs(np.linalg.det(gt_nii.affine[:3, :3])))
            for class_name, info in CLASS_INFO.items():
                eligible = class_name == "endometrioma" or int(
                    getattr(record, info["flag"])
                ) == 1
                row = {
                    **base,
                    "class": class_name,
                    "eligible": eligible,
                    "threshold": threshold,
                    "prediction_resampled": resampled,
                    "validation_status": "ok" if eligible else "class_not_annotated",
                }
                if not eligible:
                    metric_rows.append(row)
                    continue
                probability = probabilities[..., info["channel"]]
                row["probability_min"] = float(np.nanmin(probability))
                row["probability_max"] = float(np.nanmax(probability))
                if not np.isfinite(probability).all():
                    row["validation_status"] = "nonfinite_probability_channel"
                    metric_rows.append(row)
                    continue
                # A constant channel is still a valid prediction. In
                # particular, post-reasoning maps intentionally use an all-zero
                # channel when every candidate is rejected; that must score as
                # an empty prediction rather than be dropped from evaluation.
                prediction = probability >= threshold
                target = gt == info["label"]
                row.update(binary_metrics(prediction, target, voxel_volume))
                hd95, assd, surface_dice, surface_sampled = surface_metrics(
                    prediction, target, gt_nii.affine, surface_tolerance_mm,
                    max_surface_points,
                )
                row.update({"hd95_mm": hd95, "assd_mm": assd,
                            "surface_dice_2mm": surface_dice,
                            "surface_points_sampled": surface_sampled})
                metric_rows.append(row)
                if class_name == "endometrioma":
                    lesion_rows.append({**base, "threshold": threshold,
                                        "gt_positive": bool(target.any()),
                                        **lesion_metrics(prediction, target),
                                        "false_positive_volume_mm3": row["false_positive_volume_mm3"]})
    return pd.DataFrame(metric_rows), pd.DataFrame(lesion_rows)


def bootstrap_ci(values: pd.Series, seed: int = 42, iterations: int = 2000) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.array([
        np.mean(array[rng.integers(0, len(array), len(array))])
        for _ in range(iterations)
    ])
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    eligible = metrics.loc[metrics["eligible"] & metrics["validation_status"].eq("ok")].copy()
    eligible["analysis_subset"] = "all_eligible"
    positive = eligible.loc[eligible["gt_volume_mm3"] > 0].copy()
    positive["analysis_subset"] = "gt_positive_only"
    combined = pd.concat([eligible, positive], ignore_index=True)
    measures = ["dice", "iou", "recall", "precision", "specificity", "npv",
                "predicted_volume_mm3", "gt_volume_mm3", "absolute_volume_error_mm3",
                "relative_volume_error", "false_positive_volume_mm3", "hd95_mm",
                "assd_mm", "surface_dice_2mm"]
    rows = []
    for keys, group in combined.groupby(
        ["model", "cohort", "class", "analysis_subset"], sort=False
    ):
        for measure in measures:
            values = group[measure].dropna()
            low, high = bootstrap_ci(values)
            rows.append({
                "model": keys[0], "cohort": keys[1], "class": keys[2],
                "analysis_subset": keys[3], "metric": measure,
                "n_eligible": len(group), "n_measured": len(values),
                "mean": values.mean(), "std": values.std(ddof=1),
                "median": values.median(), "q25": values.quantile(.25),
                "q75": values.quantile(.75), "mean_ci_low": low,
                "mean_ci_high": high,
            })
    return pd.DataFrame(rows)


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    data = metrics.loc[metrics["eligible"] & metrics["validation_status"].eq("ok")].copy()
    data = data.loc[data["gt_volume_mm3"] > 0]
    rows = []
    for (cohort, class_name), group in data.groupby(["cohort", "class"]):
        for model in MODEL_ORDER[1:]:
            for metric in ("dice", "recall", "precision", "hd95_mm"):
                pivot = group.loc[group.model.isin(["baseline", model])].pivot_table(
                    index="scan_name", columns="model", values=metric, aggfunc="first"
                ).dropna()
                if "baseline" not in pivot.columns or model not in pivot.columns:
                    continue
                if len(pivot) < 2:
                    continue
                difference = pivot[model] - pivot["baseline"]
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
                low, high = bootstrap_ci(difference)
                rows.append({"cohort": cohort, "class": class_name, "model": model,
                             "reference": "baseline", "metric": metric, "n": len(pivot),
                             "mean_difference": difference.mean(), "ci_low": low,
                             "ci_high": high, "wilcoxon_p": p_value})
    columns = ["cohort", "class", "model", "reference", "metric", "n",
               "mean_difference", "ci_low", "ci_high", "wilcoxon_p", "fdr_q"]
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q"] = benjamini_hochberg(result["wilcoxon_p"])
    return result.reindex(columns=columns)


def save_figures(metrics: pd.DataFrame, lesions: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    positive = metrics.loc[
        metrics["eligible"] & metrics["validation_status"].eq("ok")
        & (metrics["gt_volume_mm3"] > 0)
    ].copy()
    colors = plt.cm.tab10(np.linspace(0, .85, len(MODEL_ORDER)))
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), sharey=True)
    for row, class_name in enumerate(CLASS_INFO):
        for col, cohort in enumerate(COHORT_ORDER):
            axis = axes[row, col]
            distributions = [positive.loc[(positive.model.eq(model)) &
                (positive.cohort.eq(cohort)) & (positive["class"].eq(class_name)), "dice"].dropna()
                for model in MODEL_ORDER]
            box = axis.boxplot(distributions, labels=MODEL_ORDER, patch_artist=True,
                               showfliers=False)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(.65)
            for index, values in enumerate(distributions, 1):
                axis.scatter(np.full(len(values), index), values, s=12, color=colors[index-1], alpha=.4)
            if row == 0: axis.set_title(cohort.replace("_", " ").title())
            if col == 0: axis.set_ylabel(f"{class_name.title()}\nDice")
            axis.set_ylim(0, 1.02); axis.grid(axis="y", alpha=.2)
            axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Positive-only segmentation Dice at probability threshold 0.5")
    fig.tight_layout()
    path = output_dir / "positive_only_dice_by_model_cohort_class.png"
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey="row")
    for column, cohort in enumerate(COHORT_ORDER):
        rate_axis = axes[0, column]
        fp_axis = axes[1, column]
        subset = lesions.loc[lesions.cohort.eq(cohort)]
        positives = subset.loc[subset.gt_positive]
        negatives = subset.loc[~subset.gt_positive]
        recall = positives.groupby("model").lesion_sensitivity.mean().reindex(MODEL_ORDER)
        miss = positives.groupby("model").missed_gt_lesions.apply(lambda x: (x > 0).mean()).reindex(MODEL_ORDER)
        fp = negatives.groupby("model").false_positive_lesions.mean().reindex(MODEL_ORDER)
        width = .36
        rate_axis.bar(x-width/2, recall, width, label="Lesion sensitivity")
        rate_axis.bar(x+width/2, miss, width, label="Scans with ≥1 missed lesion")
        rate_axis.set_xticks(x, MODEL_ORDER, rotation=30)
        rate_axis.set_title(cohort.replace("_", " ").title())
        rate_axis.set_ylim(0, 1.05)
        rate_axis.grid(axis="y", alpha=.2)
        rate_axis.legend(frameon=False, fontsize=8)

        fp_axis.bar(x, fp, width=.6, color=colors)
        fp_axis.set_xticks(x, MODEL_ORDER, rotation=30)
        fp_axis.set_ylim(bottom=0)
        fp_axis.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Proportion of GT-positive scans")
    axes[1, 0].set_ylabel("Mean FP lesions per GT-negative scan")
    fig.suptitle("Endometrioma lesion performance")
    fig.tight_layout()
    path = output_dir / "endometrioma_lesion_performance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for axis, class_name in zip(axes, CLASS_INFO):
        for model, color in zip(MODEL_ORDER, colors):
            medians = [positive.loc[(positive.model.eq(model)) &
                (positive.cohort.eq(cohort)) & (positive["class"].eq(class_name)), "dice"].median()
                for cohort in COHORT_ORDER]
            axis.plot(COHORT_ORDER, medians, marker="o", label=model, color=color)
        axis.set_title(class_name.title()); axis.set_ylim(0,1); axis.grid(alpha=.2)
        axis.tick_params(axis="x", rotation=25); axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Median positive-only Dice")
    fig.suptitle("Generalization from training to internal and external tests")
    fig.tight_layout()
    path = output_dir / "segmentation_generalization.png"
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_dashboard(
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    figure_paths: list[Path],
    output_path: Path,
) -> None:
    gallery = "".join(
        f'<figure><img src="{image_data_uri(path)}" alt="{html.escape(path.stem)}">'
        f'<figcaption>{html.escape(path.stem.replace("_", " ").title())}</figcaption></figure>'
        for path in figure_paths
    )
    template = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EndoMRI segmentation model comparison</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f6f8fb;color:#182033}}main{{max-width:1800px;margin:auto;padding:24px}}
h1{{margin-bottom:5px}}.note{{background:#fff3cd;border:1px solid #e1c45b;padding:10px;border-radius:7px;margin:14px 0}}
.gallery{{display:grid;grid-template-columns:1fr;gap:20px}}figure{{margin:0;background:white;border:1px solid #dce3ed;padding:12px;border-radius:9px}}img{{width:100%;height:auto}}figcaption{{font-weight:600;margin-top:8px}}
@media(max-width:700px){{main{{padding:10px}}}}
</style></head><body><main><h1>EndoMRI segmentation model comparison</h1>
<div>Six models • scan-level evaluation • fixed probability threshold 0.5</div>
<div class="note"><strong>Primary segmentation analysis is GT-positive only for endometrioma, ovary and uterus.</strong> Endometrioma additionally includes all 44/12/85 scans for negative-scan false-positive and lesion-detection analysis.</div>
<section class="gallery">{gallery}</section>
</main></body></html>'''
    output_path.write_text(template, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-root", type=Path, default=DEFAULT_ALL_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ALL_ROOT / "segmentation_model_evaluation",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surface-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--max-surface-points", type=int, default=50000,
                        help="Deterministic cap per mask for tractable surface metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.all_root, args.split_json)
    manifest.to_csv(args.output_dir / "segmentation_evaluation_manifest.csv", index=False)
    metrics, lesions = evaluate(
        manifest, args.all_root, args.threshold, args.surface_tolerance_mm,
        args.max_surface_points,
    )
    metrics.to_csv(args.output_dir / "segmentation_metrics_scan_level.csv", index=False)
    lesions.to_csv(args.output_dir / "endometrioma_lesion_metrics_scan_level.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(args.output_dir / "segmentation_performance_summary.csv", index=False)
    tests = paired_tests(metrics)
    tests.to_csv(args.output_dir / "segmentation_model_pairwise_tests.csv", index=False)
    figures = save_figures(metrics, lesions, args.output_dir)
    build_dashboard(manifest, metrics, summary, tests, figures,
                    args.output_dir / "segmentation_model_comparison_dashboard.html")
    LOGGER.info("Saved evaluation to %s", args.output_dir)


if __name__ == "__main__":
    main()
