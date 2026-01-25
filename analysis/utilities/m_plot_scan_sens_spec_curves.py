import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import cv2

# =====================================================
# Configuration
# =====================================================
ROOT_DIR = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation'
OUT_DIR = '/home/sg2162/rds/hpc-work/Experiments/TTA'
DATASET = 'Pancreas40CT_Tumor+Background'
CSV_PATH = os.path.join(ROOT_DIR, DATASET, 'test_slices.csv')

PROB_FOLDERS = [
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse'),
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse_wo_LoRA'),
    os.path.join(ROOT_DIR, DATASET, 'test_BiomedParse_with_LoRA')
]

MODEL_NAMES = [
    'BiomedParse',
    'BiomedParse-FT',
    'BiomedParse-LoRA'
]

PROB_EXT = "_pancreas+tumor+in+pancreas+CT.png"
THRESHOLDS = np.linspace(0, 1, 201)


# =====================================================
# Helper Functions
# =====================================================
def load_scan_gt(csv_path):
    """Load scan-level ground truth from CSV."""
    df = pd.read_csv(csv_path)
    scan_gt = (
        df.groupby("img_name")["class"]
        .apply(lambda x: 1 if (x == "tumor").any() else 0)
        .to_dict()
    )
    return df, scan_gt


def load_prob_image(path):
    """Load grayscale probability image and return max pixel value."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.max()


def compute_scan_max_probs(df, scan_names, prob_folder, prob_ext):
    """Compute maximum probability per scan."""
    scan_max_prob = {}

    for i, scan in enumerate(scan_names, start=1):
        print(f"[{i}/{len(scan_names)}] Processing scan: {scan}")
        slice_rows = df[df["img_name"] == scan]
        max_prob = 0

        for _, row in slice_rows.iterrows():
            slice_name = row["slice_name"]
            prob_path = os.path.join(prob_folder, slice_name + prob_ext)
            slice_max = load_prob_image(prob_path)
            if slice_max is not None and slice_max > max_prob:
                max_prob = slice_max

        scan_max_prob[scan] = max_prob

    return scan_max_prob


def compute_metrics(scan_names, scan_gt, scan_max_prob, thresholds):
    """Compute sensitivity and specificity for all thresholds."""
    sens, spec = [], []

    for th in thresholds:
        th_scaled = th * 255  # PNG scale
        TP = FP = TN = FN = 0

        for scan in scan_names:
            gt = scan_gt[scan]
            pred = 1 if scan_max_prob[scan] > th_scaled else 0

            if gt == 1 and pred == 1:
                TP += 1
            elif gt == 1 and pred == 0:
                FN += 1
            elif gt == 0 and pred == 0:
                TN += 1
            elif gt == 0 and pred == 1:
                FP += 1

        sens.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        spec.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

    return np.array(sens), np.array(spec)


def compute_auc_ci(scan_gt_dict, scan_max_prob_dict, n_bootstrap=1000, random_seed=42):
    """Compute ROC-AUC with 95% confidence interval via bootstrap."""
    y_true = np.array([scan_gt_dict[k] for k in scan_max_prob_dict.keys()])
    y_score = np.array([scan_max_prob_dict[k] for k in scan_max_prob_dict.keys()])
    auc = roc_auc_score(y_true, y_score)

    rng = np.random.RandomState(random_seed)
    boot_aucs = []

    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[indices], y_score[indices]))

    ci_lower = np.percentile(boot_aucs, 2.5)
    ci_upper = np.percentile(boot_aucs, 97.5)

    return auc, ci_lower, ci_upper


def plot_metrics(thresholds, sens, spec, auc, ci_lower, ci_upper, model_name, color, ax):
    """Plot sensitivity, specificity and annotate AUC."""
    ax.plot(thresholds, sens, linestyle='-', color=color, label='Sensitivity')
    ax.plot(thresholds, spec, linestyle='--', color=color, label='Specificity')
    ax.set_title(model_name)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    ax.text(0.05, 0.05, f"AUC={auc:.3f}\n95% CI=({ci_lower:.3f},{ci_upper:.3f})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


# =====================================================
# Main
# =====================================================
df, scan_gt = load_scan_gt(CSV_PATH)
scan_names = list(scan_gt.keys())

metrics_dir = os.path.join(OUT_DIR, DATASET, "cached_metrics")
os.makedirs(metrics_dir, exist_ok=True)

# Plot setup
n_models = len(PROB_FOLDERS)
n_cols = 2
n_rows = math.ceil(n_models / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))
axes = axes.flatten()

# Color map
cmap = plt.cm.get_cmap("tab10", n_models)
colors = [cmap(i) for i in range(n_models)]

for i, (prob_folder, model_name) in enumerate(zip(PROB_FOLDERS, MODEL_NAMES)):
    cache_file = os.path.join(metrics_dir, f"{model_name}_scan_metrics.npz")

    if os.path.exists(cache_file):
        print(f"Loading cached metrics for {model_name}")
        data = np.load(cache_file, allow_pickle=True)
        sens = data["sens"]
        spec = data["spec"]
        scan_max_prob = data["scan_max_prob"].item()
    else:
        print(f"Computing metrics for {model_name}")
        scan_max_prob = compute_scan_max_probs(df, scan_names, prob_folder, PROB_EXT)
        sens, spec = compute_metrics(scan_names, scan_gt, scan_max_prob, THRESHOLDS)
        np.savez(cache_file, thresholds=THRESHOLDS, sens=sens, spec=spec, scan_max_prob=scan_max_prob)

    auc, ci_lower, ci_upper = compute_auc_ci(scan_gt, scan_max_prob)
    plot_metrics(THRESHOLDS, sens, spec, auc, ci_lower, ci_upper, model_name, colors[i], axes[i])

# Remove empty subplots
for j in range(n_models, len(axes)):
    fig.delaxes(axes[j])

# Save figure
os.makedirs(os.path.join(OUT_DIR, DATASET), exist_ok=True)
save_path = os.path.join(OUT_DIR, DATASET, "scan_level_sens_spec.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print("Saved plot to", save_path)
plt.show()
