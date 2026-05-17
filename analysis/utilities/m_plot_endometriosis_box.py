import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_dir = "/home/sg2162/rds/hpc-work/Experiments/Endometriosis"
output_dir = "/home/sg2162/rds/hpc-work/Experiments/Endometriosis"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load all CSVs
all_dfs = []

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == "test_BiomedParse_with_LoRA_results.csv":
            file_path = os.path.join(subdir, file)
            df = pd.read_csv(file_path)
            all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# Metrics to plot
metrics = [
    "accuracy", "f1_score", "sensitivity", "specificity",
    "dice", "soft_dice", "hd95", "ca"
]

sns.set_style("whitegrid")

for metric in metrics:
    plt.figure(figsize=(12, 14))  # taller figure for 24 classes
    
    order = data.groupby("class")[metric].median().sort_values().index
    
    ax = sns.boxplot(
        data=data,
        y="class",
        x=metric,
        order=order
    )
    
    # Keep labels single-line, just shrink font a bit
    ax.tick_params(axis='y', labelsize=9)
    
    plt.title(f"{metric} per class", fontsize=16)
    plt.xlabel(metric, fontsize=12)
    plt.ylabel("")
    
    # Add grid for readability
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Increase left margin to fit long names
    plt.subplots_adjust(left=0.4)
    
    save_path = os.path.join(output_dir, f"{metric}_boxplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()