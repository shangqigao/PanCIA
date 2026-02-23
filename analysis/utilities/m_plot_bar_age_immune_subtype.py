import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.patches as mpatches
from collections import Counter

# ====== CHANGE THESE ======
root_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes/TCGA_signature_AGE"
immune_csv = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/TCGA_Pan-Cancer_outcomes/phenotypes/immune_subtype/immune_subtype.csv"  # CSV with SampleID, Subtype_Immune_Model_Based
output_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/AGE_Radiopathomics_ImmuneSubtype"
MIN_SAMPLES = 20
# =========================

os.makedirs(output_dir, exist_ok=True)

# ====== Load immune subtypes ======
immune_df = pd.read_csv(immune_csv)
immune_df["ID3"] = immune_df["SampleID"].apply(lambda x: "-".join(x.split("-")[:3]))
immune_df["Subtype"] = immune_df["Subtype_Immune_Model_Based"].str.replace(r"\s*\(.*\)", "", regex=True)

# ====== Model parser ======
def parse_model_name(folder_name, file_name):
    parts = folder_name.split("+")
    radio_model = parts[0]
    patho_model = parts[1] if len(parts) > 1 else ""
    base = file_name.replace("_results.json", "")
    tokens = base.split("_")
    omics = tokens[0]

    radio_aggr = None
    patho_aggr = None
    for t in tokens:
        if t.startswith("radio+"): radio_aggr = t.split("+")[1]
        if t.startswith("patho+"): patho_aggr = t.split("+")[1]

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

# ====== Process all JSON files ======
subtype_results = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if not file.endswith("_results.json"):
            continue
        file_path = os.path.join(root, file)
        folder_name = os.path.basename(os.path.dirname(file_path))
        with open(file_path, "r") as f:
            data = json.load(f)
        model_name, omics_type = parse_model_name(folder_name, file)

        subjects = data["subject"]
        preds_all = np.array(data["pred"]).flatten() * 100
        labels_all = np.array(data["label"]).flatten() * 100

        # Match immune subtypes
        subtypes = []
        valid_indices = []
        for i, s in enumerate(subjects):
            id3 = "-".join(s.split("-")[:3])
            row = immune_df.loc[immune_df["ID3"] == id3]
            if len(row) == 0:
                continue
            subtypes.append(row.iloc[0]["Subtype"])
            valid_indices.append(i)

        if len(valid_indices) == 0:
            continue  # skip if no matched samples

        preds = preds_all[valid_indices]
        labels = labels_all[valid_indices]

        counts = Counter(subtypes)
        for subtype in counts:
            idx = [i for i, st in enumerate(subtypes) if st == subtype]
            if len(idx) < MIN_SAMPLES:
                continue
            sub_preds = preds[idx]
            sub_labels = labels[idx]
            subtype_results.append({
                "Model": model_name,
                "Omics": omics_type,
                "Subtype": subtype,
                "R2": r2_score(sub_labels, sub_preds),
                "RMSE": np.sqrt(mean_squared_error(sub_labels, sub_preds)),
                "MAE": mean_absolute_error(sub_labels, sub_preds),
                "N": len(idx)
            })

# ====== Create DataFrame ======
sub_df = pd.DataFrame(subtype_results)
csv_path = os.path.join(output_dir, "Radiopathomics_by_subtype.csv")
sub_df.to_csv(csv_path, index=False)
print(f"Saved table → {csv_path}")

# ====== Plot grouped bar for radiopathomics ======
def plot_radiopathomics_grouped(sub_df, metric, output_dir, sort_subtype="IFN-gamma Dominant"):
    df = sub_df[sub_df["Omics"] == "Radiopathomics"]
    if df.empty:
        print("No radiopathomics data found.")
        return
    
    # Filter for that subtype
    df_sub = df[df["Subtype"] == sort_subtype]

    # Determine ascending / descending based on metric type
    if metric == "R2":
        ascending = False  # higher R2 better
    else:  # MAE, RMSE
        ascending = True  # lower error better

    # Sort models by performance in this subtype
    model_order = df_sub.sort_values(metric, ascending=ascending)["Model"].tolist()

    # If some models are missing in this subtype, append them at the end
    missing_models = [m for m in df["Model"].unique() if m not in model_order]
    model_order.extend(missing_models)

    # All subtypes for plotting
    subtypes = df["Subtype"].unique()
    n_models = len(model_order)
    n_subtypes = len(subtypes)

    # Colors per subtype
    cmap = plt.get_cmap("tab10").colors
    subtype_colors = {s: cmap[i % len(cmap)] for i, s in enumerate(subtypes)}

    total_width = 0.8
    bar_width = total_width / n_subtypes
    x = np.arange(n_models)

    plt.figure(figsize=(1.5*n_models, 10))

    all_vals = []

    for i, subtype in enumerate(subtypes):
        vals = [df[(df["Model"] == m) & (df["Subtype"] == subtype)][metric].values[0]
                if len(df[(df["Model"] == m) & (df["Subtype"] == subtype)]) > 0 else np.nan
                for m in model_order]
        all_vals.extend([v for v in vals if not np.isnan(v)])

        plt.bar(x + i*bar_width, vals, width=bar_width, color=subtype_colors[subtype], label=subtype)

        # Add value labels
        for xi, v in zip(x + i*bar_width, vals):
            if not np.isnan(v):
                plt.text(xi, v, f"{v:.2f}", ha='center', va='bottom', fontsize=12, rotation=90)

    plt.xticks(x + total_width/2 - bar_width/2, model_order, rotation=45, ha="right", fontsize=10)
    plt.ylabel(metric)
    plt.title(f"Radiopathomics models - {metric} by immune subtype")

    # ====== Axis scaling for RMSE & MAE ======
    if metric in ["RMSE", "MAE"]:
        min_val = min(all_vals)
        max_val = max(all_vals)
        margin = (max_val - min_val) * 0.15
        plt.ylim(min_val - margin, max_val + margin)  # scaled, does NOT start from 0

    plt.tight_layout()

    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in subtype_colors.items()]
    plt.legend(handles=legend_handles, title="Immune Subtype", bbox_to_anchor=(1.05,1), loc='upper left')

    save_path = os.path.join(output_dir, f"Radiopathomics_{metric}_grouped.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved grouped bar plot → {save_path}")

# ====== Generate plots ======
for metric in ["MAE", "RMSE", "R2"]:
    plot_radiopathomics_grouped(sub_df, metric, output_dir)

print("\n✅ Done.")