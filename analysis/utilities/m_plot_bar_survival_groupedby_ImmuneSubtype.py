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


# ==============================================================================
# Compute subtype-specific survival C-index
# ==============================================================================
def compute_subtype_cindex(data, immune_df, model_name, omics_type,
                           min_samples=20, min_events=5):
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
            cindex = concordance_index(
                event_times=sub_duration,
                predicted_scores=-sub_risk,   # higher risk = shorter survival
                event_observed=sub_event
            )
        except Exception:
            continue

        subtype_results.append({
            "Model": model_name,
            "Omics": omics_type,
            "Subtype": subtype,
            "CIndex": cindex,
            "N": len(idx),
            "Events": n_events
        })

    return subtype_results


# ==============================================================================
# Plot grouped bar chart for Radiopathomics models
# ==============================================================================
def plot_omics_grouped(
    df,
    metric,
    output_dir,
    sort_subtype="IFN-gamma Dominant",
    omics="Radiomics"
):
    """
    Plot grouped bar chart for Radiopathomics models across immune subtypes.
    Models are ordered according to performance in sort_subtype.
    """

    df = df[df["Omics"] == omics]

    if df.empty:
        print("No Radiopathomics data found.")
        return

    # Subset used for sorting
    df_sub = df[df["Subtype"] == sort_subtype]

    # For C-index, larger is better
    ascending = False

    # Sort models by performance in the chosen subtype
    model_order = (
        df_sub.sort_values(metric, ascending=ascending)["Model"]
        .tolist()
    )

    # Append models missing from sort_subtype
    missing_models = [
        m for m in df["Model"].unique()
        if m not in model_order
    ]
    model_order.extend(missing_models)

    # Preserve subtype order as they appear
    subtypes = list(df["Subtype"].unique())

    n_models = len(model_order)
    n_subtypes = len(subtypes)

    # Colors by subtype
    cmap = plt.get_cmap("tab10").colors
    subtype_colors = {
        subtype: cmap[i % len(cmap)]
        for i, subtype in enumerate(subtypes)
    }

    total_width = 0.8
    bar_width = total_width / n_subtypes
    x = np.arange(n_models)

    plt.figure(figsize=(max(12, 1.5 * n_models), 10))

    all_vals = []

    for i, subtype in enumerate(subtypes):
        vals = []

        for model in model_order:
            row = df[
                (df["Model"] == model) &
                (df["Subtype"] == subtype)
            ]

            if len(row) > 0:
                vals.append(row[metric].values[0])
            else:
                vals.append(np.nan)

        all_vals.extend([v for v in vals if not np.isnan(v)])

        plt.bar(
            x + i * bar_width,
            vals,
            width=bar_width,
            color=subtype_colors[subtype],
            label=subtype
        )

        # Value labels
        for xi, v in zip(x + i * bar_width, vals):
            if not np.isnan(v):
                plt.text(
                    xi,
                    v,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    # X axis
    plt.xticks(
        x + total_width / 2 - bar_width / 2,
        model_order,
        rotation=45,
        ha="right",
        fontsize=10
    )

    # Labels
    plt.ylabel("Concordance Index (C-index)")
    plt.title(f"{omics} Models by Immune Subtype")

    # Reference line
    plt.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)

    # Dynamic y-axis
    if len(all_vals) > 0:
        ymin = min(min(all_vals), 0.4)
        ymax = max(all_vals)
        padding = max(0.02, 0.05 * (ymax - ymin + 1e-6))
        plt.ylim(ymin - padding, min(1.0, ymax + padding))

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=subtype)
        for subtype, color in subtype_colors.items()
    ]

    plt.legend(
        handles=legend_handles,
        title="Immune Subtype",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"{omics}_{metric}_grouped.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved grouped bar plot → {save_path}")

def plot_best_omics_by_subtype(df, metric, output_dir):
    """
    For each immune subtype:
        - Select the best Radiomics model
        - Select the best Pathomics model
        - Select the best Radiopathomics model

    Plot grouped bars where:
        - X axis = immune subtypes
        - Bars = Best Radiomics / Best Pathomics / Best Radiopathomics
        - Legend = omics type
        - Each bar is annotated with:
            1. metric value
            2. best model name

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
            ["Model", "Omics", "Subtype", metric]
    metric : str
        Example: "CIndex", "R2", "MAE", "RMSE"
    output_dir : str
        Directory to save figure
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
                sub.sort_values(metric, ascending=ascending)
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

    # ---------------------------------------------------------------------
    # Plot configuration
    # ---------------------------------------------------------------------
    subtypes = sorted(best_df["Subtype"].unique())
    n_subtypes = len(subtypes)
    n_omics = len(omics_types)

    # Nature-style, colorblind-friendly palette (inspired by common Nature figures)
    color_map = {
        "Radiomics": "#0072B2",       # deep blue
        "Pathomics": "#009E73",       # bluish green
        "Radiopathomics": "#D55E00",  # vermillion / orange-red
    }

    total_width = 0.8
    bar_width = total_width / n_omics
    x = np.arange(n_subtypes)

    # Increase figure height to accommodate model names
    plt.figure(figsize=(max(10, 1.8 * n_subtypes), 10))

    all_vals = []

    # ---------------------------------------------------------------------
    # Draw bars
    # ---------------------------------------------------------------------
    for i, omics in enumerate(omics_types):
        vals = []
        model_names = []

        for subtype in subtypes:
            row = best_df[
                (best_df["Subtype"] == subtype) &
                (best_df["Omics"] == omics)
            ]

            if row.empty:
                vals.append(np.nan)
                model_names.append("")
            else:
                vals.append(row.iloc[0][metric])
                model_names.append(row.iloc[0]["Model"])

        all_vals.extend([v for v in vals if not np.isnan(v)])

        bars = plt.bar(
            x + i * bar_width,
            vals,
            width=bar_width,
            color=color_map[omics],
            label=f"Best {omics}"
        )

        # Annotate each bar with value + model name
        for bar, val, model_name in zip(bars, vals, model_names):
            if np.isnan(val):
                continue

            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()

            # Optional shortening of very long model names
            display_name = str(model_name)
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."

            plt.text(
                x_pos,
                y_pos,
                f"{val:.3f}\n{display_name}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

    # ---------------------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------------------
    plt.xticks(
        x + total_width / 2 - bar_width / 2,
        subtypes,
        rotation=30,
        ha="right",
        fontsize=10
    )

    if metric == "CIndex":
        plt.ylabel("Concordance Index (C-index)")
        plt.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)
    else:
        plt.ylabel(metric)

    plt.title(f"Best Model Performance by Immune Subtype ({metric})")

    # ---------------------------------------------------------------------
    # Dynamic y-axis limits with extra headroom for annotations
    # ---------------------------------------------------------------------
    if len(all_vals) > 0:
        ymin = min(all_vals)
        ymax = max(all_vals)

        if metric == "CIndex":
            ymin = min(ymin, 0.4)

        data_range = ymax - ymin
        padding = max(0.02, 0.05 * (data_range + 1e-6))

        # Extra top room for model-name annotations
        top_padding = max(0.08, 0.20 * (data_range + 1e-6))

        if metric in ["MAE", "RMSE"]:
            plt.ylim(ymin - padding, ymax + top_padding)
        else:
            upper = ymax + top_padding
            if metric in ["CIndex", "R2"]:
                upper = min(1.0, upper)
            plt.ylim(ymin - padding, upper)

    # ---------------------------------------------------------------------
    # Legend
    # ---------------------------------------------------------------------
    plt.legend(
        title="Best Model Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------------------
    save_path = os.path.join(
        output_dir,
        f"Best_Omics_by_Subtype_{metric}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved grouped bar plot → {save_path}")

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

    # Plot grouped Radiopathomics chart
    plot_omics_grouped(
        sub_df,
        metric="CIndex",
        output_dir=output_dir,
        sort_subtype="IFN-gamma Dominant",
        omics="Radiopathomics"
    )

    # plot best omics chart
    plot_best_omics_by_subtype(
        sub_df,
        metric="CIndex",
        output_dir=output_dir
    )

print("\n✅ Done.")