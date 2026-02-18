import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label

# =====================================================
# CONFIG (your paths preserved)
# =====================================================
ROOT_DIR = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation'
ROOT_TTA = '/home/sg2162/rds/hpc-work/Experiments/TTA'
OUT_DIR = '/home/sg2162/rds/hpc-work/Experiments/TTA'
DATASET = 'Pancreas40CT_Tumor+Background'
CSV_PATH = os.path.join(ROOT_DIR, DATASET, 'test_slices.csv')

PROB_FOLDERS = [
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse'),
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse_wo_LoRA'),
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse_with_LoRA'),
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse_TTA'),
    # os.path.join(ROOT_TTA, 'TTA3')
]

MODEL_NAMES = [
    'BiomedParse',
    'BiomedParse-FT',
    'BiomedParse-LoRA',
    'R2Seg'
]

GT_FOLDER = os.path.join(ROOT_DIR, DATASET, "test_mask")  # adjust if needed
# PROB_EXT = "_pancreas+tumor+in+pancreas+CT.png"
PROB_EXT = "__pancreas_tumor.png"
GT_EXT = "_pancreas+tumor.png"

THRESHOLDS = np.linspace(0, 1, 21)
IOU_THRESHOLD = 0.1
TOPK = 50
IMAGE_SIZE = 256

# =====================================================
# LOAD CSV
# =====================================================
df = pd.read_csv(CSV_PATH)
scan_names = df["img_name"].unique()
metrics_dir = os.path.join(OUT_DIR, DATASET, "cached_metrics")
os.makedirs(metrics_dir, exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def load_prob_slice(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )
    
    return img.astype(np.float16) / 255.0


def load_gt_slice(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    
    return (img > 0).astype(np.uint8)



from joblib import Parallel, delayed

def build_volume(scan, folder, ext, loader, n_jobs=32):
    slice_rows = df[df["img_name"] == scan]
    slice_names = slice_rows["slice_name"].tolist()
    paths = [os.path.join(folder, s + ext) for s in slice_names]

    slices = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(loader)(p) for p in paths
    )

    # Filter out None
    slices = [sl for sl in slices if sl is not None]

    if len(slices) == 0:
        return None

    return np.stack(slices, axis=0)


def extract_lesions(binary_vol, k=5):
    labeled, num = label(binary_vol)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    top_labels = np.argsort(counts)[-k:][::-1]

    return [(labeled == i) for i in top_labels if counts[i] > 0]


def compute_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0


def match_lesions(pred_lesions, gt_lesions, iou_threshold=0.1, n_jobs=-1):
    """
    Compute TP, FP, FN by matching predicted lesions to GT in parallel.
    
    Parameters:
        pred_lesions : list of np.array
        gt_lesions   : list of np.array
        iou_threshold: float
        n_jobs       : int, number of parallel jobs (-1 = all cores)
        
    Returns:
        TP, FP, FN
    """
    matched_gt = set()

    def process_pred(p):
        best_iou = 0
        best_idx = -1
        for i, g in enumerate(gt_lesions):
            if i in matched_gt:
                continue
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold:
            return True, best_idx  # TP, GT index matched
        else:
            return False, None     # FP

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_pred)(p) for p in pred_lesions
    )

    TP = 0
    FP = 0
    for is_tp, gt_idx in results:
        if is_tp:
            TP += 1
            matched_gt.add(gt_idx)
        else:
            FP += 1

    FN = len(gt_lesions) - len(matched_gt)
    return TP, FP, FN

def preload_volumes(prob_folder):
    prob_volumes = {}
    gt_volumes = {}

    for i, scan in enumerate(scan_names):
        print(f'Preloading scan [{i+1}/{len(scan_names)}]...')
        prob_vol = build_volume(scan, prob_folder, PROB_EXT, load_prob_slice)
        gt_vol = build_volume(scan, GT_FOLDER, GT_EXT, load_gt_slice)
        if prob_vol is not None and gt_vol is not None:
            prob_volumes[scan] = prob_vol
            gt_volumes[scan] = gt_vol

    return prob_volumes, gt_volumes

# =====================================================
# METRIC COMPUTATION
# =====================================================
def compute_metrics_for_model(prob_volumes, gt_volumes, n_jobs=8):
    froc_fp = []
    froc_sens = []
    pr_prec = []
    pr_rec = []

    scan_names_local = list(prob_volumes.keys())
    n_scans = len(scan_names_local)

    for i, th in enumerate(THRESHOLDS):
        print(f'Computing metrics for threhold [{i+1}/{len(THRESHOLDS)}]...')
        # Function to process one scan
        def process_scan(scan):
            prob_vol = prob_volumes[scan]
            gt_vol = gt_volumes[scan]

            pred_bin = (prob_vol > th).astype(np.uint8)

            pred_lesions = extract_lesions(pred_bin, TOPK)
            gt_lesions = extract_lesions(gt_vol, TOPK)

            TP, FP, FN = match_lesions(pred_lesions, gt_lesions)
            return TP, FP, len(gt_lesions)

        # Parallel execution over scans
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_scan)(scan) for scan in scan_names_local
        )

        # Aggregate results
        total_TP = sum(r[0] for r in results)
        total_FP = sum(r[1] for r in results)
        total_GT = sum(r[2] for r in results)

        sens = total_TP / total_GT if total_GT > 0 else 0
        prec = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 1
        fp_scan = total_FP / n_scans

        froc_sens.append(sens)
        froc_fp.append(fp_scan)
        pr_rec.append(sens)
        pr_prec.append(prec)

    return (
        np.array(froc_fp),
        np.array(froc_sens),
        np.array(pr_rec),
        np.array(pr_prec),
    )

def compute_fp_distribution(prob_volumes, gt_volumes, threshold=0.5, n_jobs=8):
    """
    Compute false positives per scan in parallel using joblib.

    Parameters:
        prob_volumes: dict of scan -> probability volume
        gt_volumes: dict of scan -> ground truth volume
        threshold: float, binarization threshold
        n_jobs: int, number of parallel jobs (-1 = all cores)
    """
    
    scan_names_local = list(prob_volumes.keys())

    def compute_fp_for_scan(scan):
        prob_vol = prob_volumes[scan]
        gt_vol = gt_volumes[scan]

        pred_bin = (prob_vol > threshold).astype(np.uint8)

        pred_lesions = extract_lesions(pred_bin, TOPK)
        gt_lesions = extract_lesions(gt_vol, TOPK)

        TP, FP, FN = match_lesions(pred_lesions, gt_lesions)
        return FP

    fp_list = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_fp_for_scan)(scan) for scan in scan_names_local
    )

    return np.array(fp_list)

# ==========================
# compute metrics
# ==========================
for i, (model_name, prob_folder) in enumerate(zip(MODEL_NAMES, PROB_FOLDERS)):
    cache_file = os.path.join(metrics_dir, f"{model_name}_lesion_metrics.npz")

    if not os.path.exists(cache_file):
        print(f"No cache found. Preloading volumes for {model_name}...")
        prob_volumes, gt_volumes = preload_volumes(prob_folder)

        print(f'Computing metrics for {model_name}...')
        froc_fp, froc_sens, pr_rec, pr_prec = compute_metrics_for_model(
            prob_volumes, gt_volumes
        )
        print(f'Computing FP distribution for {model_name}...')
        fp_dist = compute_fp_distribution(
            prob_volumes, gt_volumes, threshold=0.5
        )
        np.savez(cache_file,
                 froc_fp=froc_fp,
                 froc_sens=froc_sens,
                 pr_rec=pr_rec,
                 pr_prec=pr_prec,
                 fp_dist=fp_dist)


# ==========================
# PLOTTING PARAMETERS
# ==========================
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
markers = ['o', 's', '^', 'D', 'x']
markevery = 1  # plot markers every 5 points

# ==========================
# FROC - all models in one figure
# ==========================
plt.figure(figsize=(6, 6))

for i, model_name in enumerate(MODEL_NAMES):
    cache_file = os.path.join(metrics_dir, f"{model_name}_lesion_metrics.npz")
    data = np.load(cache_file)
    froc_fp = data["froc_fp"]
    froc_sens = data["froc_sens"]

    # Sort FP and ensure monotone sensitivity
    sort_idx = np.argsort(froc_fp)
    fp_sorted = froc_fp[sort_idx]
    sens_sorted = froc_sens[sort_idx]
    sens_monotone = np.maximum.accumulate(sens_sorted)

    plt.plot(fp_sorted, sens_monotone,
             label=model_name,
             color=colors[i % len(colors)],
             marker=markers[i % len(markers)],
             markevery=markevery,
             linestyle='-')

for x in [1, 3, 5]:
    plt.axvline(x=x, linestyle='--', linewidth=1, color='gray', alpha=0.6)
    plt.text(x+1, -0.15, f"{x} FP", rotation=90, va='bottom', ha='right')

plt.xlabel("FP per Scan")
plt.ylabel("Sensitivity")
plt.title("FROC Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, DATASET, "FROC_multi_model.png"), dpi=300)
plt.show()


# ==========================
# PR - all models in one figure
# ==========================
plt.figure(figsize=(10, 6))

for i, model_name in enumerate(MODEL_NAMES):
    data = np.load(os.path.join(metrics_dir, f"{model_name}_lesion_metrics.npz"))
    rec = data["pr_rec"]
    prec = data["pr_prec"]

    # Sort Recall for plotting (no monotone)
    sort_idx = np.argsort(rec)
    rec_sorted = rec[sort_idx]
    prec_sorted = prec[sort_idx]

    plt.plot(rec_sorted, prec_sorted,
             label=model_name,
             color=colors[i % len(colors)],
             marker=markers[i % len(markers)],
             markevery=markevery,
             linestyle='-')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, DATASET, "PR_multi_model.png"), dpi=300)
plt.show()


# ==========================
# FP DISTRIBUTION - subplots per model
# ==========================
figsize = (12, 10)
n_models = len(MODEL_NAMES)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
axes = axes.flatten()

for i, model_name in enumerate(MODEL_NAMES):
    data = np.load(os.path.join(metrics_dir, f"{model_name}_lesion_metrics.npz"))
    fp_dist = data["fp_dist"]

    axes[i].hist(fp_dist, bins=20, color='tab:green', alpha=0.7)
    axes[i].set_title(model_name)
    axes[i].set_xlabel("FP per Scan")
    axes[i].set_ylabel("Number of Scans")
    axes[i].grid(True)

    print(f"\n{model_name} FP statistics:")
    print("Median FP/scan:", np.median(fp_dist))
    print("90th percentile FP:", np.percentile(fp_dist, 90))

# Turn off empty axes if less than n_cols*n_rows
for j in range(n_models, len(axes)):
    axes[j].axis('off')

plt.suptitle("FP Distribution per Model", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, DATASET, "FP_distribution_subplots.png"), dpi=300)
plt.show()
