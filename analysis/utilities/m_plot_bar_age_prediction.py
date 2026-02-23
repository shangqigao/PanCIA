import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ====== CHANGE THIS ======
root_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes/TCGA_signature_AGE"
output_dir = os.path.join('/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots', "AGE")
# =========================

os.makedirs(output_dir, exist_ok=True)

def parse_model_name(folder_name, file_name):
    parts = folder_name.split("+")
    radio_model = parts[0]
    patho_model = parts[1] if len(parts) > 1 else ""

    base = file_name.replace("_results.json", "")
    tokens = base.split("_")

    omics = tokens[0]  # radiomics / pathomics / radiopathomics

    radio_aggr = None
    patho_aggr = None

    for t in tokens:
        if t.startswith("radio+"):
            radio_aggr = t.split("+")[1]
        if t.startswith("patho+"):
            patho_aggr = t.split("+")[1]

    if omics == "radiomics":
        model = radio_model
        aggr = radio_aggr
        omics_type = "Radiomics"

    elif omics == "pathomics":
        model = patho_model
        aggr = patho_aggr
        omics_type = "Pathomics"

    elif omics == "radiopathomics":
        model = f"{radio_model}+{patho_model}"
        aggr = radio_aggr
        omics_type = "Radiopathomics"

    else:
        model = folder_name
        aggr = ""
        omics_type = "Other"

    if aggr == "None": aggr = "MEAN"

    name = f"{model} ({aggr})" if aggr else model
    return name, omics_type


results = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith("_results.json"):
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(os.path.dirname(file_path))

            with open(file_path, "r") as f:
                data = json.load(f)

            preds = np.array(data["pred"]).flatten() * 100
            labels = np.array(data["label"]).flatten() * 100

            r2 = r2_score(labels, preds)
            rmse = np.sqrt(mean_squared_error(labels, preds))
            mae = mean_absolute_error(labels, preds)

            model_name, omics_type = parse_model_name(folder_name, file)

            results.append({
                "Model": model_name,
                "Omics": omics_type,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae
            })
# Create Data table
df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
# Rank models (1 = best)
df["Rank_R2"] = df["R2"].rank(ascending=False, method="min").astype(int)
df["Rank_RMSE"] = df["RMSE"].rank(ascending=True, method="min").astype(int)
df["Rank_MAE"] = df["MAE"].rank(ascending=True, method="min").astype(int)

# Save table
csv_path = os.path.join(output_dir, "model_performance.csv")
df.to_csv(csv_path, index=False)

print(f"\nSaved summary table to:\n{csv_path}")

# ========================
# ====== BAR PLOTS ======
# ========================
omics_colors = {
    "Radiomics": "#1f77b4",        # blue
    "Pathomics": "#ff7f0e",        # orange
    "Radiopathomics": "#2ca02c",   # green
    "Other": "#7f7f7f"
}

import matplotlib.patches as mpatches

def plot_ranked_vertical(df, metric, output_dir):
    ascending = metric in ["RMSE", "MAE"]
    df_sorted = df.sort_values(metric, ascending=ascending).reset_index(drop=True)

    values = df_sorted[metric]
    labels = [f"{i+1}\n{m}" for i, m in enumerate(df_sorted["Model"])]

    colors = [omics_colors[o] for o in df_sorted["Omics"]]

    plt.figure(figsize=(1.2 * len(labels), 10))
    bars = plt.bar(labels, values, color=colors)

    # Axis scaling for RMSE & MAE
    if metric in ["RMSE", "MAE"]:
        min_val = values.min()
        max_val = values.max()
        margin = (max_val - min_val) * 0.15
        plt.ylim(min_val - margin, max_val + margin)

    plt.ylabel(metric)
    plt.title(f"{metric} (Ranked)")

    plt.xticks(rotation=45, ha="right", fontsize=12)

    # value labels
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"{v:.2f}",
                 ha='center', va='bottom', fontsize=12)

    # legend
    legend_handles = [
        mpatches.Patch(color=color, label=omics)
        for omics, color in omics_colors.items()
        if omics in df_sorted["Omics"].values
    ]
    plt.legend(handles=legend_handles, title="Omics Type")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_ranked.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {metric} plot → {save_path}")

plot_ranked_vertical(df, "R2", output_dir)
plot_ranked_vertical(df, "RMSE", output_dir)
plot_ranked_vertical(df, "MAE", output_dir)

print("\n✅ Done.")