"""Compare raw and post-reasoning segmentation for baseline, r6, and r7."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m_segmentation_model_evaluation import (
    CLASS_INFO,
    COHORT_ORDER,
    DEFAULT_ALL_ROOT,
    DEFAULT_SPLIT,
    bootstrap_ci,
    build_manifest,
    evaluate,
)


RAW_SUBDIRS = {
    "baseline": "segmentations",
    "r6": "segmentations_r6",
    "r7": "segmentations_r7",
}


def probability_directories(
    all_root: Path, reasoning_root: Path
) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for model, subdirectory in RAW_SUBDIRS.items():
        directories[f"{model}_pre"] = all_root / subdirectory / "BiomedParse"
        directories[f"{model}_post"] = (
            reasoning_root / model / "post_probability_maps"
        )
    return directories


def add_design_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result[["model", "stage"]] = result["model"].str.rsplit(
        "_", n=1, expand=True
    )
    return result


def positive_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep every GT-positive scan and score an empty prediction as zero overlap."""
    positive = metrics.loc[
        metrics["eligible"] & (metrics["gt_volume_mm3"] > 0)
    ].copy()
    empty_prediction = positive["validation_status"].eq("empty_prediction")
    for metric in ("dice", "iou", "recall", "precision", "surface_dice_2mm"):
        positive.loc[empty_prediction, metric] = 0.0
    return positive


def positive_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    positive = positive_metrics(metrics)
    rows = []
    for keys, group in positive.groupby(
        ["model", "stage", "cohort", "class"], sort=False
    ):
        for metric in (
            "dice", "iou", "recall", "precision", "hd95_mm", "assd_mm",
            "surface_dice_2mm", "absolute_volume_error_mm3",
        ):
            values = group[metric].dropna()
            low, high = bootstrap_ci(values)
            rows.append(
                {
                    "model": keys[0],
                    "stage": keys[1],
                    "cohort": keys[2],
                    "class": keys[3],
                    "metric": metric,
                    "n": len(group),
                    "metric_n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_changes(metrics: pd.DataFrame) -> pd.DataFrame:
    positive = positive_metrics(metrics)
    rows = []
    for (model, cohort, class_name), group in positive.groupby(
        ["model", "cohort", "class"]
    ):
        for metric in ("dice", "iou", "recall", "precision", "hd95_mm"):
            paired = group.pivot_table(
                index="scan_name", columns="stage", values=metric,
                aggfunc="first",
            ).dropna()
            if not {"pre", "post"}.issubset(paired.columns):
                continue
            change = paired["post"] - paired["pre"]
            low, high = bootstrap_ci(change)
            rows.append(
                {
                    "model": model,
                    "cohort": cohort,
                    "class": class_name,
                    "metric": metric,
                    "n": len(change),
                    "pre_mean": paired["pre"].mean(),
                    "post_mean": paired["post"].mean(),
                    "mean_change": change.mean(),
                    "change_ci_low": low,
                    "change_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_positive_dice(
    summary: pd.DataFrame, output_path: Path, threshold: float = 0.5
) -> None:
    data = summary.loc[summary["metric"].eq("dice")]
    models = list(RAW_SUBDIRS)
    colors = {"pre": "#4c78a8", "post": "#e45756"}
    x = np.arange(len(models))
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharey=True)
    for row, class_name in enumerate(CLASS_INFO):
        for column, cohort in enumerate(COHORT_ORDER):
            axis = axes[row, column]
            for offset, stage in ((-0.18, "pre"), (0.18, "post")):
                subset = data.loc[
                    data["stage"].eq(stage)
                    & data["cohort"].eq(cohort)
                    & data["class"].eq(class_name)
                ].set_index("model").reindex(models)
                mean = subset["mean"].to_numpy(dtype=float)
                lower = mean - subset["mean_ci_low"].to_numpy(dtype=float)
                upper = subset["mean_ci_high"].to_numpy(dtype=float) - mean
                axis.bar(
                    x + offset, mean, width=0.34, color=colors[stage],
                    label=stage.title(), alpha=0.85,
                )
                axis.errorbar(
                    x + offset, mean, yerr=np.vstack([lower, upper]),
                    fmt="none", ecolor="black", capsize=3, linewidth=1,
                )
            axis.set_xticks(x, models)
            axis.set_ylim(0, 1.02)
            axis.grid(axis="y", alpha=0.2)
            if row == 0:
                axis.set_title(cohort.replace("_", " ").title())
            if column == 0:
                axis.set_ylabel(f"{class_name.title()}\nMean Dice")
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    fig.suptitle(
        f"Positive-only segmentation before versus after reasoning · threshold {threshold:g}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_endometrioma_false_positives(
    metrics: pd.DataFrame, output_path: Path
) -> None:
    negative = metrics.loc[
        metrics["eligible"]
        & metrics["validation_status"].eq("ok")
        & metrics["class"].eq("endometrioma")
        & metrics["gt_volume_mm3"].eq(0)
    ]
    models = list(RAW_SUBDIRS)
    colors = {"pre": "#4c78a8", "post": "#e45756"}
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for axis, cohort in zip(axes, COHORT_ORDER):
        for offset, stage in ((-0.18, "pre"), (0.18, "post")):
            values = []
            for model in models:
                subset = negative.loc[
                    negative["model"].eq(model)
                    & negative["stage"].eq(stage)
                    & negative["cohort"].eq(cohort),
                    "false_positive_volume_mm3",
                ]
                values.append(subset.median())
            axis.bar(
                x + offset, np.asarray(values) / 1000.0, width=0.34,
                color=colors[stage], label=stage.title(), alpha=0.85,
            )
        axis.set_xticks(x, models)
        axis.set_ylim(bottom=0)
        axis.set_title(cohort.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Median FP volume on GT-negative scans (cm³)")
    fig.suptitle("Endometrioma false-positive burden before versus after reasoning")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-root", type=Path, default=DEFAULT_ALL_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--reasoning-root", type=Path,
        default=DEFAULT_ALL_ROOT / "reasoning_model_comparison",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_ALL_ROOT / "reasoning_model_comparison" / "comparison",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surface-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--max-surface-points", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = probability_directories(args.all_root, args.reasoning_root)
    missing = {
        name: directory for name, directory in directories.items()
        if not directory.exists()
    }
    if missing:
        raise FileNotFoundError(f"Missing probability directories: {missing}")
    manifest = build_manifest(args.all_root, args.split_json)
    metrics, lesions = evaluate(
        manifest, args.all_root, args.threshold, args.surface_tolerance_mm,
        args.max_surface_points, probability_dirs=directories,
    )
    metrics = add_design_columns(metrics)
    lesions = add_design_columns(lesions)
    metrics.to_csv(args.output_dir / "pre_post_segmentation_metrics.csv", index=False)
    lesions.to_csv(args.output_dir / "pre_post_endometrioma_lesion_metrics.csv", index=False)
    summary = positive_summary(metrics)
    summary.to_csv(args.output_dir / "pre_post_positive_summary.csv", index=False)
    changes = paired_changes(metrics)
    changes.to_csv(args.output_dir / "pre_post_paired_changes.csv", index=False)
    plot_positive_dice(
        summary, args.output_dir / "pre_post_positive_dice.png", args.threshold
    )
    plot_endometrioma_false_positives(
        metrics, args.output_dir / "pre_post_endometrioma_false_positives.png"
    )


if __name__ == "__main__":
    main()
