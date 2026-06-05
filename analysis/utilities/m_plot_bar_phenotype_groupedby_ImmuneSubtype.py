import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import textwrap

# ====== CHANGE THESE ======
root_parent = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_slice+tumor"
immune_csv = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/TCGA_Pan-Cancer_outcomes/phenotypes/immune_subtype/immune_subtype.csv"
output_root = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Phenotype_ImmuneSubtype"
MIN_SAMPLES = 20
THRESHOLD = 0.5
phenotype_classes = {
    "TCGA_phenotype_ImmuneSubtype": [
        "IFN-gamma Dominant (Immune C2)", 
        "Inflammatory (Immune C3)", 
        "Wound Healing (Immune C1)", 
        "Lymphocyte Depleted (Immune C4)"
    ],
    "TCGA_phenotype_MolecularSubtype": [
        "BRCA.LumA",
        "BRCA.LumB",
        "BRCA.Basal",
        "BRCA.Normal",
        "GI.CIN",
        "KIRC.1",
        "KIRC.2",
        "KIRC.3",
        "KIRC.4",
        "LIHC.iCluster:3",
        "LIHC.iCluster:1",
        "LIHC.iCluster:2",
        "OVCA.Proliferative",
        "OVCA.Differentiated",
        "OVCA.Mesenchymal",
        "UCEC.CN_HIGH"
    ],
    "TCGA_phenotype_PrimaryDisease": [
        "breast invasive carcinoma",
        "bladder urothelial carcinoma",
        "ovarian serous cystadenocarcinoma",
        "lung adenocarcinoma",
        "stomach adenocarcinoma",
        "lung squamous cell carcinoma",
        "liver hepatocellular carcinoma",
        "kidney clear cell carcinoma",
        "uterine corpus endometrioid carcinoma",
        "cervical & endocervical cancer"
    ]
}
# =========================

os.makedirs(output_root, exist_ok=True)

# ====== Load immune subtypes ======
immune_df = pd.read_csv(immune_csv)
immune_df["ID3"] = immune_df["SampleID"].apply(lambda x: "-".join(x.split("-")[:3]))
immune_df["Subtype"] = immune_df["Subtype_Immune_Model_Based"].str.replace(
    r"\s*\(.*\)", "", regex=True
)

# ====== Model parser ======
def parse_model_name(folder_name, file_name):
    """
    Same parser as your original code.
    """
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


# ====== Plot grouped bar ======
def plot_omics_grouped(
    df,
    metric="F1",
    output_dir=None,
    sort_subtype="IFN-gamma Dominant",
    omics="Radiomics",
    phenotype_name="Phenotype"
):
    """
    Plot grouped bar chart for omics models across immune subtypes.
    Models are ordered according to performance in sort_subtype.
    """

    df = df[df["Omics"] == omics].copy()

    if df.empty:
        print(f"No {omics} data found.")
        return

    # Subset used for sorting
    df_sub = df[df["Subtype"] == sort_subtype]

    if df_sub.empty:
        print(f"No data for sorting subtype: {sort_subtype}")
        return

    # Sort models by performance in chosen subtype
    model_order = (
        df_sub.sort_values(metric, ascending=False)["Model"]
        .tolist()
    )

    # Append missing models
    missing_models = [
        m for m in df["Model"].unique()
        if m not in model_order
    ]
    model_order.extend(missing_models)

    # Preserve subtype order as appearance in data (more stable than sorted)
    subtypes = list(df["Subtype"].unique())

    n_models = len(model_order)
    n_subtypes = len(subtypes)

    # Colors
    cmap = plt.get_cmap("tab10").colors
    subtype_colors = {
        subtype: cmap[i % len(cmap)]
        for i, subtype in enumerate(subtypes)
    }

    total_width = 0.8
    bar_width = total_width / n_subtypes
    x = np.arange(n_models)

    plt.figure(figsize=(max(12, 1.5 * n_models), 8))

    all_vals = []

    # Pre-index dataframe for faster lookup
    grouped = df.set_index(["Model", "Subtype"])[metric]

    for i, subtype in enumerate(subtypes):
        vals = []

        for model in model_order:
            try:
                v = grouped.loc[(model, subtype)]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                vals.append(v)
            except KeyError:
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
                    f"{v:.2f}",
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

    plt.ylabel(metric)
    plt.title(f"{phenotype_name} ({omics}) — {metric} by Immune Subtype")

    # Reference line (only if metric is probability-like)
    if metric.lower() in ["f1", "f1_score"]:
        plt.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)

    # Dynamic y-limits (better than fixed 0–1)
    if len(all_vals) > 0:
        ymin = min(all_vals)
        ymax = max(all_vals)
        padding = max(0.05, 0.1 * (ymax - ymin + 1e-6))
        plt.ylim(max(0, ymin - padding), min(1.0, ymax + padding))

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

    if output_dir is not None:
        save_path = os.path.join(
            output_dir,
            f"{omics}_{metric}_{phenotype_name}_grouped.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved grouped bar plot → {save_path}")
    else:
        plt.show()

def plot_best_omics_by_subtype(df, metric, output_dir):
    """
    For each immune subtype:
        - Select best F1 model for Radiomics
        - Select best F1 model for Pathomics
        - Select best F1 model for Radiopathomics

    Plot grouped bars:
        X axis = immune subtypes
        Bars = best F1 per omics type
        Annotation = F1 score + model name
    """

    if df.empty:
        print("Input dataframe is empty.")
        return

    # ------------------------------------------------------------
    # Omics types
    # ------------------------------------------------------------
    omics_types = ["Radiomics", "Pathomics", "Radiopathomics"]

    df = df[df["Omics"].isin(omics_types)].copy()

    if df.empty:
        print("No relevant omics data found.")
        return

    # ------------------------------------------------------------
    # Select best model per (Subtype, Omics)
    # ------------------------------------------------------------
    best_rows = []

    for subtype in sorted(df["Subtype"].unique()):
        for omics in omics_types:

            sub = df[
                (df["Subtype"] == subtype) &
                (df["Omics"] == omics)
            ]

            if sub.empty:
                continue

            best = sub.sort_values(metric, ascending=False).iloc[0]
            best_rows.append(best)

    if not best_rows:
        print("No valid best-model results found.")
        return

    best_df = pd.DataFrame(best_rows)

    # Save table
    csv_path = os.path.join(
        output_dir,
        f"Best_{metric}_by_Subtype.csv"
    )
    best_df.to_csv(csv_path, index=False)
    print(f"Saved best-model table → {csv_path}")

    # ------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------
    subtypes = sorted(best_df["Subtype"].unique())
    omics_types = ["Radiomics", "Pathomics", "Radiopathomics"]

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

    plt.figure(figsize=(max(10, 1.8 * n_subtypes), 10))

    all_vals = []

    # ------------------------------------------------------------
    # Plot bars
    # ------------------------------------------------------------
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

        # --------------------------------------------------------
        # Annotations: F1 + model name
        # --------------------------------------------------------
        for bar, val, model_name in zip(bars, vals, model_names):

            if np.isnan(val):
                continue

            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()

            display_name = str(model_name)
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."

            plt.text(
                x_pos,
                y_pos,
                f"{val:.2f}\n{display_name}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    plt.xticks(
        x + total_width / 2 - bar_width / 2,
        subtypes,
        rotation=30,
        ha="right",
        fontsize=10
    )

    plt.ylabel(f"{metric} Score")
    plt.title(f"Best {metric} Model Performance by Immune Subtype")

    plt.ylim(0, 1)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    plt.legend(
        title="Best Model Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    save_path = os.path.join(
        output_dir,
        f"Best_{metric}_by_Subtype.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved grouped bar plot → {save_path}")


def _wrap(labels, width=22):
    return ["\n".join(textwrap.wrap(str(x), width)) for x in labels]


def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    save_path,
    normalize=False,
    class_names=None
):
    """
    Clean Nature-style confusion matrix with:
    - fixed alignment
    - x labels rotated 90°
    - no display shift issues
    """

    # -----------------------------
    # CLASS ORDER (FIXED)
    # -----------------------------
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)

    # -----------------------------
    # LABELS
    # -----------------------------
    if class_names is None:
        labels = [f"Class {c}" for c in classes]
    elif isinstance(class_names, list):
        labels = class_names
    else:
        labels = [class_names.get(c, str(c)) for c in classes]

    labels = _wrap(labels, width=22)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype(float)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-8)

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7 + n * 0.3, 7 + n * 0.3))

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -----------------------------
    # TICKS (NO SHIFT)
    # -----------------------------
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # ✔ FIX: x-axis rotated 90° as requested
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", va="center")

    # -----------------------------
    # ANNOTATION
    # -----------------------------
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"

            ax.text(
                j, i, text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val > cm.max() / 2 else "black"
            )

    # -----------------------------
    # LABELS
    # -----------------------------
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    # keep matrix properly aligned
    ax.set_ylim(n - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ====== Find all phenotype folders ======
phenotype_dirs = sorted(
    glob.glob(os.path.join(root_parent, "TCGA_phenotype_*"))
)

print(f"Found {len(phenotype_dirs)} phenotype folders.")

# ====== Process each phenotype ======
for phenotype_dir in phenotype_dirs:
    phenotype_name = os.path.basename(phenotype_dir)
    print(f"\nProcessing {phenotype_name}")

    output_dir = os.path.join(output_root, phenotype_name)
    os.makedirs(output_dir, exist_ok=True)

    subtype_results = []

    # Walk through all result files
    for root, dirs, files in os.walk(phenotype_dir):
        for file in files:
            if not file.endswith("_results.json"):
                continue

            file_path = os.path.join(root, file)
            folder_name = os.path.basename(os.path.dirname(file_path))

            # Load JSON
            with open(file_path, "r") as f:
                data = json.load(f)

            # Skip if required keys missing
            if not all(k in data for k in ["subject", "prob", "label"]):
                continue

            model_name, omics_type = parse_model_name(folder_name, file)

            subjects = data["subject"]
            # ------------------------------------------------------------
            # Convert probabilities and labels for multiclass classification
            # ------------------------------------------------------------
            probs_all = np.array(data["prob"])
            labels_all = np.array(data["label"])

            # labels may be one-hot encoded
            if labels_all.ndim > 1:
                labels_all = np.argmax(labels_all, axis=1)
            else:
                labels_all = labels_all.astype(int).flatten()

            # Predictions: class with highest probability
            if probs_all.ndim == 1:
                # Binary classification fallback
                preds_all = (probs_all >= THRESHOLD).astype(int)
            else:
                # Multiclass classification
                preds_all = np.argmax(probs_all, axis=1)

            # Match immune subtype
            subtypes = []
            valid_indices = []

            for i, id3 in enumerate(subjects):
                row = immune_df.loc[immune_df["ID3"] == id3]
                if len(row) == 0:
                    continue

                subtypes.append(row.iloc[0]["Subtype"])
                valid_indices.append(i)

            if len(valid_indices) == 0:
                continue

            preds = preds_all[valid_indices]
            labels = labels_all[valid_indices]

            # Group by subtype
            counts = Counter(subtypes)

            for subtype in counts:
                idx = [i for i, st in enumerate(subtypes) if st == subtype]

                if len(idx) < MIN_SAMPLES:
                    continue

                sub_preds = preds[idx]
                sub_labels = labels[idx]

                if "_ImmuneSubtype" in phenotype_name:
                    score_name = "ACC"
                    score = np.mean(sub_preds == sub_labels)
                else:
                    score_name = "F1"
                    score = f1_score(
                        sub_labels,
                        sub_preds,
                        average="macro"
                    )

                subtype_results.append({
                    "Model": model_name,
                    "Omics": omics_type,
                    "Subtype": subtype,
                    score_name: score,
                    "N": len(idx),
                })

    # ====== Save results ======
    if len(subtype_results) == 0:
        print(f"No valid subtype results for {phenotype_name}")
        continue

    sub_df = pd.DataFrame(subtype_results)

    csv_path = os.path.join(
        output_dir,
        f"{phenotype_name}_by_subtype_{score_name}.csv"
    )
    sub_df.to_csv(csv_path, index=False)
    print(f"Saved table → {csv_path}")

    # ====== Plot ======
    plot_omics_grouped(
        sub_df, 
        metric=score_name,
        output_dir=output_dir,
        sort_subtype="IFN-gamma Dominant",
        omics="Radiopathomics",
        phenotype_name=phenotype_name
    )

    # plot best omics chart
    plot_best_omics_by_subtype(
        sub_df,
        metric=score_name,
        output_dir=output_dir
    )

    # ============================================================
    # CONFUSION MATRIX FOR BEST MODEL PER OMICS
    # ============================================================
    # Compute best model per omics (global, not subtype-based)
    best_models = (
        sub_df.groupby(["Model", "Omics"])[score_name]
        .mean()
        .reset_index()
        .sort_values(score_name, ascending=False)
        .groupby("Omics")
        .head(1)
    )

    for _, row in best_models.iterrows():

        model_name = row["Model"]
        omics_type = row["Omics"]
        score_val = row[score_name]

        # Reconstruct predictions for this model
        preds_all_model = []
        labels_all_model = []

        for root, dirs, files in os.walk(phenotype_dir):
            for file in files:
                if not file.endswith("_results.json"):
                    continue

                file_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(file_path))

                m_name, omics = parse_model_name(folder_name, file)

                if m_name != model_name or omics != omics_type:
                    continue

                with open(file_path, "r") as f:
                    data = json.load(f)

                subjects = data["subject"]
                probs_all = np.array(data["prob"])
                labels_all = np.array(data["label"])

                if labels_all.ndim > 1:
                    labels_all = np.argmax(labels_all, axis=1)
                else:
                    labels_all = labels_all.astype(int).flatten()

                if probs_all.ndim == 1:
                    preds_all = (probs_all >= THRESHOLD).astype(int)
                else:
                    preds_all = np.argmax(probs_all, axis=1)

                # match immune subtypes
                valid_preds = []
                valid_labels = []

                for i, sid in enumerate(subjects):
                    row = immune_df.loc[immune_df["ID3"] == sid]
                    if len(row) == 0:
                        continue

                    valid_preds.append(preds_all[i])
                    valid_labels.append(labels_all[i])

                preds_all_model.extend(valid_preds)
                labels_all_model.extend(valid_labels)

        if len(labels_all_model) == 0:
            continue

        y_true = np.array(labels_all_model)
        y_pred = np.array(preds_all_model)

        safe_model = model_name.replace("/", "_")
        safe_omics = omics_type.replace("/", "_")

        class_names = phenotype_classes[phenotype_name]

        # RAW CM
        plot_confusion_matrix(
            y_true,
            y_pred,
            title=f"{phenotype_name} | {omics_type} | {model_name}",
            save_path=os.path.join(
                output_dir,
                f"{safe_omics}_{safe_model}_cm.png"
            ),
            normalize=False,
            class_names=class_names
        )

        # NORMALIZED CM
        plot_confusion_matrix(
            y_true,
            y_pred,
            title=f"{phenotype_name} | {omics_type} | {model_name} (Normalized)",
            save_path=os.path.join(
                output_dir,
                f"{safe_omics}_{safe_model}_cm_norm.png"
            ),
            normalize=True,
            class_names=class_names
        )

        print(f"Saved CM for BEST {omics_type}: {model_name}")

print("\n✅ Done.")

