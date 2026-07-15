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
root_parent = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_strategies"
immune_csv = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/TCGA_Pan-Cancer_outcomes/phenotypes/immune_subtype/immune_subtype.csv"
output_root = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Survival_ImmuneSubtype"

MIN_SAMPLES = 20          # minimum samples per subtype
MIN_EVENTS = 5            # minimum number of observed events per subtype
Y_AXIS_MIN = 0.4          # start bar plot y-axis at 0.4
N_BOOT_DIFF = 1000        # bootstrap iterations for model-difference CI
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

def bootstrap_cindex_difference(
    df_raw,
    model_a,
    model_b,
    subtype=None,
    n_boot=1000,
    seed=42
):
    """
    Paired bootstrap CI for C-index difference between two models.
    Difference is model_b - model_a, using subjects shared by both models.
    """

    cols = ["subject", "risk", "event", "duration"]
    sub = df_raw.copy()

    if subtype is not None:
        sub = sub[sub["Subtype"] == subtype]

    a = sub[sub["Model"] == model_a][cols].rename(columns={"risk": "risk_a"})
    b = sub[sub["Model"] == model_b][cols].rename(columns={"risk": "risk_b"})

    paired = a.merge(
        b[["subject", "event", "duration", "risk_b"]],
        on=["subject", "event", "duration"],
        how="inner"
    ).dropna(subset=["risk_a", "risk_b", "event", "duration"])

    if len(paired) < MIN_SAMPLES or int(paired["event"].sum()) < MIN_EVENTS:
        return np.nan, np.nan, np.nan, len(paired)

    rng = np.random.default_rng(seed)
    diffs = []

    risk_a = paired["risk_a"].values
    risk_b = paired["risk_b"].values
    event = paired["event"].values.astype(int)
    duration = paired["duration"].values
    n = len(paired)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        try:
            c_a = concordance_index(
                event_times=duration[idx],
                predicted_scores=-risk_a[idx],
                event_observed=event[idx]
            )
            c_b = concordance_index(
                event_times=duration[idx],
                predicted_scores=-risk_b[idx],
                event_observed=event[idx]
            )
            diffs.append(c_b - c_a)
        except Exception:
            continue

    if len(diffs) == 0:
        return np.nan, np.nan, np.nan, len(paired)

    diffs = np.array(diffs)

    return (
        np.mean(diffs),
        np.percentile(diffs, 2.5),
        np.percentile(diffs, 97.5),
        len(paired)
    )

def add_difference_bracket(ax, x1, x2, y, diff_mean, diff_low, diff_high,
                           label_prefix, line_height=0.012):
    """Draw a model-difference CI bracket above two bars."""

    if x1 is None or x2 is None:
        return

    if np.isnan(diff_mean) or np.isnan(diff_low) or np.isnan(diff_high):
        return

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + line_height, y + line_height, y],
        color="black",
        linewidth=1.2,
        clip_on=False
    )
    ax.text(
        (x1 + x2) / 2,
        y + line_height,
        f"Δ={diff_mean:.3f} [{diff_low:.3f}, {diff_high:.3f}]",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        clip_on=False
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

def plot_best_omics_by_subtype_with_ci(df, metric, output_dir, df_raw=None):
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

    fig, ax = plt.subplots(figsize=(max(10, len(subtypes)*1.5), 9))

    bar_positions = {}
    bar_tops = {}

    for i, omics in enumerate(omics_types):

        means, lows, highs, models = [], [], [], []

        for subtype in subtypes:
            row = best_df[
                (best_df["Subtype"] == subtype) &
                (best_df["Omics"] == omics)
            ]

            if row.empty:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                models.append("")
            else:
                means.append(row.iloc[0][f"{metric}_mean"])
                lows.append(row.iloc[0][f"{metric}_low"])
                highs.append(row.iloc[0][f"{metric}_high"])
                models.append(row.iloc[0]["Model"])

        means = np.array(means)
        lows = np.array(lows)
        highs = np.array(highs)

        yerr = np.vstack([
            means - lows,
            highs - means
        ])

        positions = x + i * bar_width

        bars = ax.bar(
            positions,
            means,
            width=bar_width,
            color=color_map[omics],
            label=omics,
            yerr=yerr,
            capsize=4
        )

        # Add mean+CI and model name on top of each bar (two columns, vertical text, bottom-aligned)
        for j, (bar, mean_val, low_val, high_val, model_name) in enumerate(zip(bars, means, lows, highs, models)):
            if not np.isnan(mean_val):
                height = bar.get_height()
                bar_center_x = bar.get_x() + bar.get_width() / 2
                
                # Left column: Mean + CI (vertical)
                ci_text = f'Mean: {mean_val:.3f}; CI: [{low_val:.3f}, {high_val:.3f}]\n {model_name}'
                ax.text(
                    bar_center_x,  # Left side of bar
                    Y_AXIS_MIN + 0.01,  # Bottom of visible bar region
                    ci_text,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black',
                    rotation=90  # Vertical text
                )

                bar_positions[(subtypes[j], omics)] = bar_center_x
                bar_tops[(subtypes[j], omics)] = high_val

    if df_raw is not None and not df_raw.empty:
        for j, subtype in enumerate(subtypes):
            selected = {
                row["Omics"]: row["Model"]
                for _, row in best_df[best_df["Subtype"] == subtype].iterrows()
            }

            comparisons = [
                ("Radiomics", "Pathomics", "Pathomics - Radiomics"),
                ("Pathomics", "Radiopathomics", "Radiopathomics - Pathomics"),
            ]
            subtype_tops = [
                bar_tops.get((subtype, omics), np.nan)
                for omics in omics_types
            ]
            base_y = np.nanmax(subtype_tops) if not np.all(np.isnan(subtype_tops)) else Y_AXIS_MIN

            for k, (omics_a, omics_b, label) in enumerate(comparisons):
                if omics_a not in selected or omics_b not in selected:
                    continue

                diff_mean, diff_low, diff_high, _ = bootstrap_cindex_difference(
                    df_raw=df_raw,
                    model_a=selected[omics_a],
                    model_b=selected[omics_b],
                    subtype=subtype,
                    n_boot=N_BOOT_DIFF,
                    seed=42 + j * 10 + k
                )

                add_difference_bracket(
                    ax=ax,
                    x1=bar_positions.get((subtype, omics_a)),
                    x2=bar_positions.get((subtype, omics_b)),
                    y=base_y + 0.01 + k * 0.07,
                    diff_mean=diff_mean,
                    diff_low=diff_low,
                    diff_high=diff_high,
                    label_prefix=label
                )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(subtypes, rotation=30, ha="right")
    ax.set_ylabel(f"{metric} (bootstrap mean ± 95% CI)")
    ax.axhline(0.5, linestyle="--", alpha=0.6)
    ax.set_title("Best Models by Immune Subtype (Bootstrap CI)")
    ax.legend()

    max_top = max(
        [v for v in bar_tops.values() if not np.isnan(v)] + [Y_AXIS_MIN + 0.1]
    )
    ax.set_ylim(Y_AXIS_MIN, min(1.15, max_top + 0.22))

    fig.tight_layout()

    save_path = os.path.join(output_dir, "Subtype_Best_models_bootstrap_CI.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

def plot_best_omics_overall_with_ci(df, metric, output_dir, df_raw=None):

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
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(max(5, n_omics*1.8), 9))

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

    plot_x = x[:len(means)]

    bars = ax.bar(
        plot_x,
        means,
        width=bar_width,
        color=colors,
        label=labels,
        yerr=yerr,
        capsize=3
    )

    # Add mean and CI on top of each bar
    for bar, mean_val, low_val, high_val in zip(bars, means, lows, highs):
        if not np.isnan(mean_val):
            height = bar.get_height()
            bar_center_x = bar.get_x() + bar.get_width() / 2
            
            # Format the text with mean and CI on two lines
            text = f'Mean: {mean_val:.3f}\n CI: [{low_val:.3f}, {high_val:.3f}]'
            
            # Add text at the top of the bar
            ax.text(
                bar_center_x,
                Y_AXIS_MIN + 0.01,  # Small offset above visible baseline
                text,
                ha='center',
                va='bottom',
                fontsize=10,
                color='black',
                rotation=90
            )

    if df_raw is not None and not df_raw.empty:
        selected = {
            row["Omics"]: row["Model"]
            for _, row in best_df.iterrows()
        }
        position_by_omics = dict(zip(labels, plot_x))

        comparisons = [
            ("Radiomics", "Pathomics", "Pathomics - Radiomics"),
            ("Pathomics", "Radiopathomics", "Radiopathomics - Pathomics"),
        ]
        base_y = np.nanmax(highs) if len(highs) > 0 and not np.all(np.isnan(highs)) else Y_AXIS_MIN

        for k, (omics_a, omics_b, label) in enumerate(comparisons):
            if omics_a not in selected or omics_b not in selected:
                continue

            diff_mean, diff_low, diff_high, _ = bootstrap_cindex_difference(
                df_raw=df_raw,
                model_a=selected[omics_a],
                model_b=selected[omics_b],
                subtype=None,
                n_boot=N_BOOT_DIFF,
                seed=100 + k
            )

            add_difference_bracket(
                ax=ax,
                x1=position_by_omics.get(omics_a),
                x2=position_by_omics.get(omics_b),
                y=base_y + 0.035 + k * 0.07,
                diff_mean=diff_mean,
                diff_low=diff_low,
                diff_high=diff_high,
                label_prefix=label
            )

    ax.set_xticks(plot_x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(f"{metric} (bootstrap mean ± 95% CI)")
    ax.axhline(0.5, linestyle="--", alpha=0.6, color='gray')
    ax.set_title("Overall Model Performance (Bootstrap)")
    max_top = np.nanmax(highs) if len(highs) > 0 and not np.all(np.isnan(highs)) else Y_AXIS_MIN + 0.1
    ax.set_ylim(Y_AXIS_MIN, min(1.15, max_top + 0.22))

    ax.legend(loc='upper left')
    fig.tight_layout()

    save_path = os.path.join(output_dir, "Overall_best_models_bootstrap.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved → {save_path}")

    return models

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

            if file.startswith("radiomics_") and "_Strategy1_" not in str(file):
                continue

            if file.startswith("pathomics_") and "_Strategy1_" not in str(file):
                continue
            
            if file.startswith("radiopathomics_") and "_Strategy1_" not in str(file):
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
                subject = data["subject"][i]
                subtype_match = immune_df.loc[immune_df["ID3"] == subject]
                subtype = (
                    subtype_match.iloc[0]["Subtype"]
                    if len(subtype_match) > 0
                    else np.nan
                )

                overall_rows.append({
                    "Model": model_name,
                    "Omics": omics_type,
                    "subject": subject,
                    "Subtype": subtype,
                    "risk": float(np.array(data["risk"]).flatten()[i]),
                    "event": int(np.array(data["event"]).flatten()[i]),
                    "duration": float(np.array(data["duration"]).flatten()[i]),
                })

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
    best_models = plot_best_omics_overall_with_ci(
        overall_df,
        metric="CIndex",
        output_dir=output_dir,
        df_raw=overall_df_raw
    )

    # Skip if nothing valid
    if len(subtype_results) == 0:
        print(f"No valid subtype results for {survival_name}")
        continue

    # Create DataFrame
    sub_df = pd.DataFrame(subtype_results)

    # Keep best models
    sub_df = sub_df[sub_df["Model"].isin(best_models)]

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
        output_dir=output_dir,
        df_raw=overall_df_raw
    )

print("\n✅ Done.")
