from analysis.utilities.m_metrics import evaluate_segmentation_metrics
import pandas as pd
import pathlib
import cv2
import os

root_dir = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation'
save_dir = '/home/sg2162/rds/hpc-work/Experiments/TTA'
datasets = pathlib.Path(root_dir).glob('*_Tumor+Background')
datasets = [p.name for p in datasets if p.is_dir()]
prediction = ['test_BiomedParse', 'test_BiomedParse_wo_LoRA', 'test_BiomedParse_with_LoRA'][0]

for dataset in datasets:
    df = pd.read_csv(f"{root_dir}/{dataset}/test_slices.csv")
    slice_names = df['slice_name'].to_list()

    pred_masks = pathlib.Path(f"{root_dir}/{dataset}/{prediction}").glob('*.png')
    pred_masks = {m.stem.replace(f"_{m.stem.split('_')[-1]}", '') : m for m in pred_masks}
    pred_masks = [pred_masks[k] for k in slice_names]

    gt_masks = pathlib.Path(f"{root_dir}/{dataset}/test_mask").glob('*.png')
    gt_masks = {m.stem.replace(f"_{m.stem.split('_')[-1]}", '') : m for m in gt_masks}
    gt_masks = [gt_masks[k] for k in slice_names]

    metrics = []
    for pred, gt in zip(pred_masks, gt_masks):
        prob = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        seg_metrics = evaluate_segmentation_metrics(
            predictions=prob / 255.0,
            labels=mask
        )
        metrics.append(seg_metrics)
    df_metrics = pd.DataFrame(metrics)
    df_combined = pd.concat([df, df_metrics], axis=1)

    os.makedirs(f"{save_dir}/{dataset}", exist_ok=True)
    df_combined.to_csv(f"{save_dir}/{prediction}/{prediction}_results.csv")


