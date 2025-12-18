import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIG
# ============================================================

ROOT_DIR = "/home/sg2162/rds/hpc-work/Experiments/outcomes/TCGA_multitask_noprotein/LVMMed+UNI/radiomics_pathomics_Survival_Prediction_GCNConv_SPARRA"   # directory that contains 00/, 01/, ..., 04/
LOG_NAME = "debug.log"
MODEL_NAME = "SPARRA"      # <<=== set your model name here
SAVE_DIR = "/home/sg2162/rds/hpc-work/PanCIA/figures/plots"      # directory to save figures

os.makedirs(SAVE_DIR, exist_ok=True)

# Regex to capture: "|timestamp|timestamp| [INFO] key: value"
LINE_RE = re.compile(r"\[INFO\]\s+(.+?):\s+([-+]?\d*\.\d+|\d+)")

# Metrics of interest grouped by task
TASKS = {
    "survival": ["Cindex"],
    "classification": ["F1"],
    "regression": ["R2"],
}

SPLITS = ["train", "valid-A", "valid-B"]


# ============================================================
# 2. PARSE ALL FOLDS
# ============================================================

def parse_log_file(path):
    """Parse one debug.log and return dict: metrics[epoch][metric_name] = value"""
    metrics = {}
    current_epoch = None

    with open(path, "r") as f:
        for line in f:
            if "EPOCH" in line:
                m = re.search(r"EPOCH:\s*(\d+)", line)
                if m:
                    current_epoch = int(m.group(1))
                    if current_epoch not in metrics:
                        metrics[current_epoch] = {}
                continue

            m = LINE_RE.search(line)
            if m and current_epoch is not None:
                key, value = m.group(1), float(m.group(2))
                metrics[current_epoch][key] = value

    return metrics


# Load all folds
all_folds = []
for fold_dir in sorted(glob.glob(os.path.join(ROOT_DIR, "[0-9][0-9]"))):
    log_path = os.path.join(fold_dir, LOG_NAME)
    if os.path.exists(log_path):
        all_folds.append(parse_log_file(log_path))

print(f"Loaded {len(all_folds)} folds.")


# ============================================================
# 3. AGGREGATE METRICS ACROSS FOLDS
# ============================================================

def aggregate_metric(metric_key):
    """
    Return:
        epochs: sorted list
        mean_vals: array (epochs)
        std_vals:  array (epochs)
    """
    # gather per fold
    fold_series = []
    for fold in all_folds:
        values = []
        epochs = sorted(fold.keys())
        for ep in epochs:
            values.append(fold[ep].get(metric_key, np.nan))
        fold_series.append(values)

    # Convert to array: (folds, epochs)
    arr = np.array(fold_series)
    return epochs, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ============================================================
# 4. PLOTTING FUNCTION
# ============================================================

def plot_task(task_name, metric_suffixes):
    """
    task_name: "survival", "classification", "regression"
    metric_suffixes: ["Cindex"], ["F1"], ["R2"]
    """
    plt.figure(figsize=(9, 6))
    
    for split in SPLITS:
        for m in metric_suffixes:
            key = f"infer-{split}-{m}"
            try:
                epochs, mean_vals, std_vals = aggregate_metric(key)
            except:
                continue

            label = f"{split} ({m})"
            plt.plot(epochs, mean_vals, label=label)
            plt.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

    plt.title(f"{MODEL_NAME} – {task_name.capitalize()} Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_{task_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 5. PLOT ALL TASKS
# ============================================================

for task, metrics in TASKS.items():
    plot_task(task, metrics)