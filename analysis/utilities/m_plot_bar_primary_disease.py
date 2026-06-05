import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
root_dir = Path("/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_llm/TCGA_phenotype_PrimaryDisease")
save_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Radiomics_PrimaryDisease"

cancer_types = {
    1: "Breast invasive carcinoma",
    2: "Bladder urothelial carcinoma",
    3: "Ovarian serous cystadenocarcinoma",
    4: "Lung adenocarcinoma",
    5: "Stomach adenocarcinoma",
    6: "Lung squamous cell carcinoma",
    7: "Liver hepatocellular carcinoma",
    8: "Kidney clear cell carcinoma",
    9: "Uterine corpus endometrioid carcinoma",
    10: "Cervical & endocervical cancer",
}

# Short labels for x-axis
short_labels = [
    "BRCA", "BLCA", "OV", "LUAD", "STAD",
    "LUSC", "LIHC", "KIRC", "UCEC", "CESC"
]

# =========================
# Load data
# =========================
# Store:
# model_name -> mean F1 vector (length 10)
# model_name -> std F1 vector  (length 10)
model_mean = {}
model_std = {}

for model_dir in sorted(root_dir.iterdir()):
    if not model_dir.is_dir():
        continue

    metric_files = list(model_dir.glob("*_metrics.json"))
    if not metric_files:
        continue

    metric_file = metric_files[0]

    with open(metric_file, "r") as f:
        data = json.load(f)

    # Collect F1 vectors from all folds
    fold_f1 = []
    for fold_name, fold_data in data.items():
        if "f1" in fold_data:
            fold_f1.append(np.array(fold_data["f1"], dtype=float))

    if not fold_f1:
        continue

    # Shape: (num_folds, 10)
    fold_f1 = np.vstack(fold_f1)

    # Mean and standard deviation across folds
    mean_f1 = fold_f1.mean(axis=0)
    std_f1 = fold_f1.std(axis=0, ddof=1)  # sample std

    # Use folder name as model name, remove trailing "UNI"
    model_name = model_dir.name
    if model_name.endswith("UNI"):
        model_name = model_name[:-4]

    model_mean[model_name] = mean_f1
    model_std[model_name] = std_f1

if not model_mean:
    raise ValueError(f"No valid *_metrics.json files found under {root_dir}")

# =========================
# Convert to DataFrames
# Rows = cancer types
# Columns = models
# =========================
mean_df = pd.DataFrame(model_mean, index=short_labels)
std_df = pd.DataFrame(model_std, index=short_labels)

# =========================
# Plot grouped bar chart with error bars
# =========================
num_cancers = len(mean_df.index)
num_models = len(mean_df.columns)

x = np.arange(num_cancers)
bar_width = 0.8 / num_models  # total width per group = 0.8

fig_width = max(12, num_cancers * 1.2)
fig, ax = plt.subplots(figsize=(fig_width, 6))

for i, model in enumerate(mean_df.columns):
    offset = (i - (num_models - 1) / 2) * bar_width

    ax.bar(
        x + offset,
        mean_df[model].values,
        width=bar_width,
        label=model,
        yerr=std_df[model].values,  # standard deviation error bars
        capsize=3,
        error_kw={
            "elinewidth": 1,
            "capthick": 1
        }
    )

# Formatting
ax.set_xlabel("Cancer Type")
ax.set_ylabel("Mean F1 Score ± SD")
ax.set_title("F1 Score Comparison Across Models and Cancer Types")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha="right")
ax.set_ylim(0, 1.0)

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"{save_dir}/f1_comparison_barplot_with_std.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional: print abbreviation mapping
print("Cancer type abbreviations:")
for abbr, full in zip(short_labels, cancer_types.values()):
    print(f"{abbr}: {full}")