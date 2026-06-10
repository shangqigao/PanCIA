# =============================================================================
# Survival performance by immune subtype
# =============================================================================
# This script is the survival counterpart of your AGE-by-immune-subtype code.
# It:
#   1. Loads immune subtype annotations
#   2. Reads all survival result JSON files
#   3. Matches subjects to immune subtypes using TCGA ID3
#   4. Computes subtype-specific C-index
#   5. Saves a CSV table
#   6. Plots grouped bar charts for Radiopathomics models by immune subtype
#
# Required JSON keys:
#   - subject
#   - risk
#   - event
#   - duration
#
# =============================================================================

import os
import json
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines.utils import concordance_index

# ====== CHANGE THESE ==========================================================
root_parent = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_slice+tumor"
immune_csv = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/TCGA_Pan-Cancer_outcomes/phenotypes/immune_subtype/immune_subtype.csv"
output_root = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Survival_ImmuneSubtype"

MIN_SAMPLES = 20          # minimum samples per subtype
MIN_EVENTS = 5            # minimum number of observed events per subtype
# ==============================================================================

os.makedirs(output_root, exist_ok=True)

# ==============================================================================
# Load immune subtype annotations
# ==============================================================================
immune_df = pd.read_csv(immune_csv)
immune_df["ID3"] = immune_df["SampleID"].apply(
    lambda x: "-".join(str(x).split("-")[:3])
)
immune_df["Subtype"] = immune_df["Subtype_Immune_Model_Based"].str.replace(
    r"\s*\(.*\)", "", regex=True
)

# ==============================================================================
# Model parser (same as your original code)
# ==============================================================================
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

    if aggr == "None":
        aggr = "MEAN"

    name = f"{model} ({aggr})" if aggr else model
    return name, omics_type

def bootstrap_cindex(
    risk, event, duration,
    n_boot=1000,
    seed=42
):
    """
    Bootstrap C-index with patient resampling.
    Returns mean and CI.
    """

    rng = np.random.default_rng(seed)

    n = len(risk)
    scores = []

    risk = np.array(risk)
    event = np.array(event)
    duration = np.array(duration)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        try:
            c = concordance_index(
                event_times=duration[idx],
                predicted_scores=-risk[idx],
                event_observed=event[idx]
            )
            scores.append(c)
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    scores = np.array(scores)

    return (
        np.mean(scores),
        np.percentile(scores, 2.5),
        np.percentile(scores, 97.5)
    )

# ==============================================================================
# Compute subtype-specific survival C-index
# ==============================================================================
def compute_subtype_cindex(data, immune_df, model_name, omics_type,
                           min_samples=20, min_events=5, n_boot=1000):
    """
    Compute C-index separately for each immune subtype.

    Parameters
    ----------
    data : dict
        Loaded JSON data with keys:
        subject, risk, event, duration
    immune_df : DataFrame
        Contains columns:
        ID3, Subtype
    model_name : str
    omics_type : str
    min_samples : int
    min_events : int

    Returns
    -------
    list of dict
    """

    # Check required keys
    if not all(k in data for k in ["subject", "risk", "event", "duration"]):
        return []

    subjects = data["subject"]
    risk_all = np.array(data["risk"]).flatten()
    event_all = np.array(data["event"]).flatten().astype(int)
    duration_all = np.array(data["duration"]).flatten()

    # Match subjects to immune subtype
    subtype_list = []
    valid_indices = []

    for i, id3 in enumerate(subjects):
        row = immune_df.loc[immune_df["ID3"] == id3]

        if len(row) == 0:
            continue

        subtype_list.append(row.iloc[0]["Subtype"])
        valid_indices.append(i)

    if len(valid_indices) == 0:
        return []

    # Keep matched samples only
    risk = risk_all[valid_indices]
    event = event_all[valid_indices]
    duration = duration_all[valid_indices]

    # Remove NaNs
    valid = ~(
        np.isnan(risk) |
        np.isnan(event) |
        np.isnan(duration)
    )

    risk = risk[valid]
    event = event[valid]
    duration = duration[valid]
    subtype_list = np.array(subtype_list)[valid]

    if len(risk) == 0:
        return []

    # Count samples per subtype
    counts = Counter(subtype_list)

    subtype_results = []

    for subtype in counts:
        idx = np.where(subtype_list == subtype)[0]

        # Minimum sample threshold
        if len(idx) < min_samples:
            continue

        sub_risk = risk[idx]
        sub_event = event[idx]
        sub_duration = duration[idx]

        n_events = int(np.sum(sub_event))

        # Need enough observed events
        if n_events < min_events:
            continue

        # Compute subtype-specific C-index
        try:
            mean_c, low_c, high_c = bootstrap_cindex(
                sub_risk,
                sub_event,
                sub_duration,
                n_boot=n_boot
            )
        except Exception:
            continue

        subtype_results.append({
            "Model": model_name,
            "Omics": omics_type,
            "Subtype": subtype,
            "CIndex_mean": mean_c,
            "CIndex_low": low_c,
            "CIndex_high": high_c,
            "N": len(idx),
            "Events": n_events
        })

    return subtype_results

def plot_best_omics_by_subtype_with_ci(df, metric, output_dir):
    """
    For each subtype:
        show best Radiomics / Pathomics / Radiopathomics
        with bootstrap CI error bars
    """

    if df.empty:
        print("Input dataframe is empty.")
        return

    # ---------------------------------------------------------------------
    # Determine whether larger or smaller values are better
    # ---------------------------------------------------------------------
    if metric in ["MAE", "RMSE"]:
        ascending = True      # smaller is better
    else:
        ascending = False     # larger is better (CIndex, R2)

    # ---------------------------------------------------------------------
    # Omics categories
    # ---------------------------------------------------------------------
    omics_types = ["Radiomics", "Pathomics", "Radiopathomics"]

    df = df[df["Omics"].isin(omics_types)].copy()

    if df.empty:
        print("No Radiomics/Pathomics/Radiopathomics data found.")
        return

    # ---------------------------------------------------------------------
    # Select best model for each subtype and omics
    # ---------------------------------------------------------------------
    best_rows = []

    for subtype in sorted(df["Subtype"].unique()):
        for omics in omics_types:
            sub = df[
                (df["Subtype"] == subtype) &
                (df["Omics"] == omics)
            ]

            if sub.empty:
                continue

            best = (
                sub.sort_values(f"{metric}_mean", ascending=ascending)
                   .iloc[0]
            )

            best_rows.append(best)

    if len(best_rows) == 0:
        print("No valid results found.")
        return

    best_df = pd.DataFrame(best_rows)

    # Save table of selected best models
    csv_path = os.path.join(
        output_dir,
        f"Best_Models_by_Subtype_{metric}.csv"
    )
    best_df.to_csv(csv_path, index=False)
    print(f"Saved best-model table → {csv_path}")

    subtypes = sorted(df["Subtype"].unique())

    color_map = {
        "Radiomics": "#0072B2",
        "Pathomics": "#009E73",
        "Radiopathomics": "#D55E00",
    }

    x = np.arange(len(subtypes))
    bar_width = 0.25

    plt.figure(figsize=(max(10, len(subtypes)*1.5), 8))

    for i, omics in enumerate(omics_types):

        means, lows, highs = [], [], []

        for subtype in subtypes:
            row = best_df[
                (best_df["Subtype"] == subtype) &
                (best_df["Omics"] == omics)
            ]

            if row.empty:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
            else:
                means.append(row.iloc[0][f"{metric}_mean"])
                lows.append(row.iloc[0][f"{metric}_low"])
                highs.append(row.iloc[0][f"{metric}_high"])

        means = np.array(means)
        lows = np.array(lows)
        highs = np.array(highs)

        yerr = np.vstack([
            means - lows,
            highs - means
        ])

        plt.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            color=color_map[omics],
            label=omics,
            yerr=yerr,
            capsize=4
        )

    plt.xticks(x, subtypes, rotation=30, ha="right")
    plt.ylabel(f"{metric} (bootstrap mean ± 95% CI)")
    plt.axhline(0.5, linestyle="--", alpha=0.6)
    plt.title("Best Models by Immune Subtype (Bootstrap CI)")
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(output_dir, "Subtype_Best_models_bootstrap_CI.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved → {save_path}")

def compute_overall_bootstrap(df_raw, n_boot=1000):
    """
    Pool all patients across subtypes.
    Compute model-level bootstrap C-index.
    """

    results = []

    for (model, omics), sub in df_raw.groupby(["Model", "Omics"]):

        mean_c, low_c, high_c = bootstrap_cindex(
            sub["risk"].values,
            sub["event"].values,
            sub["duration"].values,
            n_boot=n_boot
        )

        results.append({
            "Model": model,
            "Omics": omics,
            "CIndex_mean": mean_c,
            "CIndex_low": low_c,
            "CIndex_high": high_c,
            "N": len(sub)
        })

    return pd.DataFrame(results)

def plot_best_omics_overall_with_ci(df, metric, output_dir):

    if df.empty:
        print("Input dataframe is empty.")
        return

    # ---------------------------------------------------------------------
    # Determine whether larger or smaller values are better
    # ---------------------------------------------------------------------
    if metric in ["MAE", "RMSE"]:
        ascending = True      # smaller is better
    else:
        ascending = False     # larger is better (CIndex, R2)

    # ---------------------------------------------------------------------
    # Omics categories
    # ---------------------------------------------------------------------
    omics_types = ["Radiomics", "Pathomics", "Radiopathomics"]

    df = df[df["Omics"].isin(omics_types)].copy()

    if df.empty:
        print("No Radiomics/Pathomics/Radiopathomics data found.")
        return

    # ---------------------------------------------------------------------
    # Select best model for each omics
    # ---------------------------------------------------------------------
    best_rows = []

    for omics in omics_types:
        sub = df[df["Omics"] == omics]

        if sub.empty:
            continue

        best = sub.sort_values(f"{metric}_mean", ascending=ascending).iloc[0]

        best_rows.append(best)

    if len(best_rows) == 0:
        print("No valid results found.")
        return

    best_df = pd.DataFrame(best_rows)

    # Save table of selected best models
    csv_path = os.path.join(
        output_dir,
        f"Best_Models_{metric}.csv"
    )
    best_df.to_csv(csv_path, index=False)
    print(f"Saved best-model table → {csv_path}")

    omics_types = ["Radiomics", "Pathomics", "Radiopathomics"]

    color_map = {
        "Radiomics": "#0072B2",
        "Pathomics": "#009E73",
        "Radiopathomics": "#D55E00",
    }

    n_omics = len(omics_types)
    bar_width = 1

    plt.figure(figsize=(max(4, n_omics*1.5), 8))

    x = np.arange(n_omics)

    means, lows, highs = [], [], []
    colors, labels, models = [], [], []
    
    for omics in omics_types:
        sub = best_df[best_df["Omics"] == omics]

        if not sub.empty:
            means.append(sub.iloc[0][f"{metric}_mean"])
            lows.append(sub.iloc[0][f"{metric}_low"])
            highs.append(sub.iloc[0][f"{metric}_high"])
            models.append(sub.iloc[0]["Model"])
            colors.append(color_map[omics])
            labels.append(omics)


    means = np.array(means)
    lows = np.array(lows)
    highs = np.array(highs)

    yerr = np.vstack([means - lows, highs - means])

    plt.bar(
        x,
        means,
        width=bar_width,
        color=colors,
        label=labels,
        yerr=yerr,
        capsize=3
    )

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel(f"{metric} (bootstrap mean ± 95% CI)")
    plt.axhline(0.5, linestyle="--", alpha=0.6)
    plt.title("Overall Model Performance (Bootstrap)")

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "Overall_best_models_bootstrap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved → {save_path}")

# ==============================================================================
# Find all survival endpoints
# ==============================================================================
survival_dirs = sorted(
    glob.glob(os.path.join(root_parent, "TCGA_survival_*"))
)

print(f"Found {len(survival_dirs)} survival folders.")


# ==============================================================================
# Process each survival endpoint
# ==============================================================================
for survival_dir in survival_dirs:

    survival_name = os.path.basename(survival_dir)
    print(f"\nProcessing {survival_name}")

    output_dir = os.path.join(output_root, survival_name)
    os.makedirs(output_dir, exist_ok=True)

    subtype_results = []
    overall_rows = []

    # Walk all result files
    for root, dirs, files in os.walk(survival_dir):
        for file in files:
            if not file.endswith("_results.json"):
                continue

            file_path = os.path.join(root, file)
            folder_name = os.path.basename(
                os.path.dirname(file_path)
            )

            # Load JSON
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

            # Parse model name
            model_name, omics_type = parse_model_name(
                folder_name,
                file
            )

            # Compute subtype-specific C-index
            results = compute_subtype_cindex(
                data=data,
                immune_df=immune_df,
                model_name=model_name,
                omics_type=omics_type,
                min_samples=MIN_SAMPLES,
                min_events=MIN_EVENTS
            )

            subtype_results.extend(results)

            for i in range(len(data["subject"])):
                overall_rows.append({
                    "Model": model_name,
                    "Omics": omics_type,
                    "risk": float(np.array(data["risk"]).flatten()[i]),
                    "event": int(np.array(data["event"]).flatten()[i]),
                    "duration": float(np.array(data["duration"]).flatten()[i]),
                })

    # Skip if nothing valid
    if len(subtype_results) == 0:
        print(f"No valid subtype results for {survival_name}")
        continue

    # Create DataFrame
    sub_df = pd.DataFrame(subtype_results)

    # Save table
    csv_path = os.path.join(
        output_dir,
        f"{survival_name}_CIndex_by_subtype.csv"
    )
    sub_df.to_csv(csv_path, index=False)
    print(f"Saved table → {csv_path}")

    # plot best subtype omics chart with CI
    plot_best_omics_by_subtype_with_ci(
        sub_df,
        metric="CIndex",
        output_dir=output_dir
    )

    overall_df_raw = pd.DataFrame(overall_rows)
    overall_df = compute_overall_bootstrap(overall_df_raw, n_boot=1000)
    
    # Save table
    csv_path = os.path.join(
        output_dir,
        f"{survival_name}_CIndex_overall.csv"
    )
    overall_df.to_csv(csv_path, index=False)
    print(f"Saved table → {csv_path}")

    # plot best overall omics chart with CI
    plot_best_omics_overall_with_ci(
        overall_df,
        metric="CIndex",
        output_dir=output_dir
    )

print("\n✅ Done.")