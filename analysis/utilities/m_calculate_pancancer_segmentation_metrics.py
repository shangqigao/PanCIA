from m_metrics import evaluate_segmentation_metrics
import pandas as pd
import pathlib
import shutil
import cv2
import os

from joblib import Parallel, delayed
from tqdm import tqdm

root_dir = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation'
save_dir = '/home/sg2162/rds/hpc-work/Experiments/TTA'
datasets = pathlib.Path(root_dir).glob('*_Tumor+Background')
datasets = [p.name for p in datasets if p.is_dir()]
datasets = ['AMOS22CT_Abdomen', 'AMOS22MR_Abdomen', 'MMs_Heart']

# for dataset in datasets:
#     all_masks = sorted(pathlib.Path(f"{root_dir}/{dataset}/test_mask").glob('*.png'))
#     slice_names = [p.stem for p in all_masks]
#     image_names = [n.replace('_' + '_'.join(n.split('_')[-3:]), '') for n in slice_names]
#     all_masks = sorted(pathlib.Path(f"{root_dir}/{dataset}/test_mask").glob('*.png'))
#     slice_classes = [p.stem.split('_')[-1].replace('+', ' ') for p in all_masks]
#     all_slices = [{'img_name': i, 'slice_name': s, 'class': c} for i, s, c in zip(image_names, slice_names, slice_classes)]
#     df = pd.DataFrame(all_slices)
#     df = df.sort_values(by='slice_name', ascending=True).reset_index(drop=True)

#     df.to_csv(f'{root_dir}/{dataset}/test_slices.csv')
#     src = pathlib.Path(f'{root_dir}/{dataset}/test_slices.csv')
#     dst_folder = pathlib.Path(f'/home/sg2162/rds/hpc-work/Experiments/TTA/{dataset}')
#     dst_folder.mkdir(parents=True, exist_ok=True)
#     dst = dst_folder / src.name
#     shutil.copy2(src, dst)
#     print(f"File copied to: {dst}")

prediction = ['test_BiomedParse', 'test_BiomedParse_wo_LoRA', 'test_BiomedParse_with_LoRA'][0]

for dataset in datasets:
    print(f"Calculating metrics on dataset {dataset}...")
    df = pd.read_csv(f"{root_dir}/{dataset}/test_slices.csv", index_col=0)
    slice_names = df['slice_name'].to_list()
    slice_classes = df['class'].to_list()

    pred_masks = sorted(pathlib.Path(f"{root_dir}/{dataset}/{prediction}").glob('*.png'))
    # pred_masks = {m.stem.replace(f"_{m.stem.split('_')[-1]}", '') : m for m in pred_masks}
    pred_masks = {m.stem.replace(f"_{m.stem.split('_')[-1]}", f'_{c}'.replace(' ', '+')) : m for m, c in zip(pred_masks, slice_classes)}
    print(list(pred_masks.keys())[0], slice_names[0], slice_classes[0])
    pred_masks = [pred_masks[k] for k in slice_names]

    gt_masks = pathlib.Path(f"{root_dir}/{dataset}/test_mask").glob('*.png')
    # gt_masks = {m.stem.replace(f"_{m.stem.split('_')[-1]}", '') : m for m in gt_masks}
    gt_masks = {m.stem : m for m in gt_masks}
    gt_masks = [gt_masks[k] for k in slice_names]

    print(f"Found {len(gt_masks)} to process!")

    def process_pair(idx, pred, gt):
        prob = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        metrics = evaluate_segmentation_metrics(
            predictions=prob / 255.0,
            labels=mask / 255.0
        )
        return {"id": idx, "metrics": metrics}


    # Parallel processing — preserves order automatically
    results = Parallel(n_jobs=32)(
        delayed(process_pair)(i, pred, gt)
        for i, (pred, gt) in enumerate(tqdm(zip(pred_masks, gt_masks), total=len(pred_masks)))
    )
    results_sorted = sorted(results, key=lambda x: x['id'])
    metrics = [r['metrics'] for r in results_sorted]
    df_metrics = pd.DataFrame(metrics)
    df_combined = pd.concat([df.reset_index(drop=True), df_metrics.reset_index(drop=True)], axis=1)

    os.makedirs(f"{save_dir}/{dataset}", exist_ok=True)
    df_combined.to_csv(f"{save_dir}/{dataset}/{prediction}_results.csv")


