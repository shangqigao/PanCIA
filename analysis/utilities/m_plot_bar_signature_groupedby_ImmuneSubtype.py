import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.metrics import r2_score
import textwrap

# ====== CHANGE THESE ======
root_parent = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_slice+tumor"

immune_csv = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/TCGA_Pan-Cancer_outcomes/phenotypes/immune_subtype/immune_subtype.csv"

output_root = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Signature_ImmuneSubtype"

MIN_SAMPLES = 20
signature_labels = {
    "TCGA_signature_AGE": ["Age"],
    "TCGA_signature_GeneProgrames": [
        'GP1_Proliferation/DNA_repair',
        'GP2_Immune-Tcell/Bcell',
        'GP3_Tumor_suppressing_miRNA_targets',
        'GP4_MES/ECM',
        'GP5_MYC_targets/TERT',
        'GP6_Squamous_differentiation/development',
        'GP7_Estrogen_signaling',
        'GP8_FOXO/stemness',
        'GP9_Cell-cell_adhesion',
        'GP10_Fatty_acid_oxidation',
        'GP11_Immune-IFN_GP12_Hypoxia/glycolosis',
        'GP13_Neural_signailng',
        'GP14_Plasma_membrane_cell-cell_signaling',
        'GP15_EGF_signailng',
        'GP16_Protein_kinase_signailng_(MAPKs)',
        'GP17_Basal_signaling',
        'GP18_Vesicle/EPR_membrane_coat',
        'GP19_1Q_amplicon',
        'GP20_TAL1-Leukemia/erythropoiesis',
        'GP21_Anti-apoptosis/DNA_stability',
        'GP22_16Q22-24_amplicon',
        'AKT_PATHWAY',
        'ALK_PATHWAY',
        'BRCA_ATR_PATHWAY',
        'CASPASE_CASCADE_(APOPTOSIS)',
        'CTLA4_PATHWAY',
        'HDAC_TARGETS_DN',
        'HER2_AMPLIFIED',
        'IGF1R_PATHWAY',
        'MTOR_PATHWAY',
        'MYC_amplified',
        'PD1_SIGNALING',
        'PI3K_CASCADE',
        'PTEN_PATHWAY',
        'RAS_PATHWAY',
        'RB_PATHWAY',
        'RESPONSE_TO_ANDROGEN',
        'RETINOL_METABOLISM',
        'VEGF_PATHWAY'
    ],
    "TCGA_signature_HRDscore": [
        'ai1',
        'lst1',
        'hrd-loh',
        'HRD'
    ],
    "TCGA_signature_ImmuneSignatureScore": [
        'ICS5_score',
        'LIexpression_score',
        'Chemokine12_score',
        'NHI_5gene_score',
        'CD68',
        'CD8A',
        'PD1_data',
        'PDL1_data',
        'PD1_PDL1_score',
        'CTLA4_data',
        'Bcell_mg_IGJ',
        'Bcell_receptors_score',
        'STAT1_score',
        'CSF1_response',
        'TcClassII_score',
        'IL12_score_21050467',
        'IL4_score_21050467',
        'IL2_score_21050467',
        'IL13_score_21050467',
        'IFNG_score_21050467',
        'TGFB_score_21050467',
        'TREM1_data',
        'DAP12_data',
        'Tcell_receptors_score',
        'IL8_21978456',
        'IFN_21978456',
        'MHC1_21978456',
        'MHC2_21978456',
        'Bcell_21978456',
        'Tcell_21978456',
        'CD103pos_mean_25446897',
        'CD103neg_mean_25446897',
        'IgG_19272155',
        'Interferon_19272155',
        'LCK_19272155',
        'MHC.I_19272155',
        'MHC.II_19272155',
        'STAT1_19272155',
        'Troester_WoundSig_19887484',
        'MDACC.FNA.1_20805453',
        'IGG_Cluster_21214954',
        'Minterferon_Cluster_21214954',
        'Immune_cell_Cluster_21214954',
        'MCD3_CD8_21214954',
        'Interferon_Cluster_21214954',
        'B_cell_PCA_16704732',
        'CD8_PCA_16704732',
        'GRANS_PCA_16704732',
        'LYMPHS_PCA_16704732',
        'T_cell_PCA_16704732',
        'TGFB_PCA_17349583',
        'Rotterdam_ERneg_PCA_15721472',
        'HER2_Immune_PCA_18006808',
        'IR7_score',
        'Buck14_score',
        'TAMsurr_score',
        'Immune_NSCLC_score',
        'Module3_IFN_score',
        'Module4_TcellBcell_score',
        'Module5_TcellBcell_score',
        'Module11_Prolif_score',
        'GP11_Immune_IFN',
        'GP2_ImmuneTcellBcell_score',
        'CD8_CD68_ratio',
        'TAMsurr_TcClassII_ratio',
        'CHANG_CORE_SERUM_RESPONSE_UP',
        'CSR_Activated_15701700',
        'CD103pos_CD103neg_ratio_25446897'
    ],
    "TCGA_signature_StemnessScoreDNA": [
        'DNAss',
        'EREG-METHss',
        'DMPss',
        'ENHss'
    ],
    "TCGA_signature_StemScoreRNA": [
        'RNAss',
        'EREG.EXPss'
    ]
}
# =========================

os.makedirs(output_root, exist_ok=True)

# ============================================================
# LOAD IMMUNE SUBTYPE TABLE
# ============================================================
immune_df = pd.read_csv(immune_csv)

immune_df["ID3"] = immune_df["SampleID"].apply(
    lambda x: "-".join(x.split("-")[:3])
)

immune_df["Subtype"] = immune_df[
    "Subtype_Immune_Model_Based"
].str.replace(r"\s*\(.*\)", "", regex=True)


# ============================================================
# MODEL PARSER
# ============================================================
def parse_model_name(folder_name, file_name):
    """
    Same parser as classification code.
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


# ============================================================
# GROUPED BAR PLOT
# ============================================================
def plot_omics_grouped(
    df,
    metric="R2",
    output_dir=None,
    sort_subtype="IFN-gamma Dominant",
    omics="Radiopathomics",
    task_name="Signature",
    outcome_name="Outcome"
):
    """
    Plot grouped bar chart:
        x-axis = models
        grouped bars = immune subtypes
        values = R²
    """

    df = df[df["Omics"] == omics].copy()

    if df.empty:
        print(f"No {omics} data found.")
        return

    # --------------------------------------------------------
    # sort models using chosen subtype
    # --------------------------------------------------------
    df_sub = df[df["Subtype"] == sort_subtype]

    if df_sub.empty:
        print(f"No sorting subtype found: {sort_subtype}")
        return

    model_order = (
        df_sub.sort_values(metric, ascending=False)["Model"]
        .tolist()
    )

    # append missing models
    missing_models = [
        m for m in df["Model"].unique()
        if m not in model_order
    ]

    model_order.extend(missing_models)

    subtypes = list(df["Subtype"].unique())

    n_models = len(model_order)
    n_subtypes = len(subtypes)

    # --------------------------------------------------------
    # colors
    # --------------------------------------------------------
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

    grouped = df.set_index(["Model", "Subtype"])[metric]

    # --------------------------------------------------------
    # bars
    # --------------------------------------------------------
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

        # value labels
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

    # --------------------------------------------------------
    # axis
    # --------------------------------------------------------
    plt.xticks(
        x + total_width / 2 - bar_width / 2,
        model_order,
        rotation=45,
        ha="right",
        fontsize=10
    )

    plt.ylabel(metric)

    plt.title(
        f"{task_name} | {outcome_name}\n"
        f"{omics} — {metric} by Immune Subtype"
    )

    # dynamic limits
    if len(all_vals) > 0:

        ymin = min(all_vals)
        ymax = max(all_vals)

        padding = max(
            0.05,
            0.1 * (ymax - ymin + 1e-6)
        )

        plt.ylim(
            ymin - padding,
            ymax + padding
        )

    # legend
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
            f"{outcome_name}_{omics}_{metric}_grouped.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved grouped plot → {save_path}")

    else:
        plt.show()


# ============================================================
# BEST OMICS PLOT
# ============================================================
def plot_best_omics_by_subtype(
    df,
    metric,
    output_dir,
    task_name,
    outcome_name
):
    """
    For each subtype:
        select best Radiomics
        select best Pathomics
        select best Radiopathomics
    """

    if df.empty:
        return

    omics_types = [
        "Radiomics",
        "Pathomics",
        "Radiopathomics"
    ]

    df = df[df["Omics"].isin(omics_types)].copy()

    best_rows = []

    for subtype in sorted(df["Subtype"].unique()):

        for omics in omics_types:

            sub = df[
                (df["Subtype"] == subtype) &
                (df["Omics"] == omics)
            ]

            if sub.empty:
                continue

            best = sub.sort_values(
                metric,
                ascending=False
            ).iloc[0]

            best_rows.append(best)

    if len(best_rows) == 0:
        return

    best_df = pd.DataFrame(best_rows)

    # --------------------------------------------------------
    # save csv
    # --------------------------------------------------------
    csv_path = os.path.join(
        output_dir,
        f"{outcome_name}_Best_{metric}_by_Subtype.csv"
    )

    best_df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # plotting
    # --------------------------------------------------------
    subtypes = sorted(best_df["Subtype"].unique())

    color_map = {
        "Radiomics": "#0072B2",
        "Pathomics": "#009E73",
        "Radiopathomics": "#D55E00",
    }

    total_width = 0.8
    bar_width = total_width / len(omics_types)

    x = np.arange(len(subtypes))

    plt.figure(figsize=(max(10, 1.8 * len(subtypes)), 10))

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

        bars = plt.bar(
            x + i * bar_width,
            vals,
            width=bar_width,
            color=color_map[omics],
            label=omics
        )

        # annotations
        for bar, val, model_name in zip(
            bars,
            vals,
            model_names
        ):

            if np.isnan(val):
                continue

            display_name = str(model_name)

            if len(display_name) > 30:
                display_name = display_name[:27] + "..."

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.2f}\n{display_name}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

    plt.xticks(
        x + total_width / 2 - bar_width / 2,
        subtypes,
        rotation=30,
        ha="right"
    )

    plt.ylabel(metric)

    plt.title(
        f"{task_name} | {outcome_name}\n"
        f"Best {metric} by Immune Subtype"
    )

    plt.legend(
        title="Best Omics",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"{outcome_name}_Best_{metric}_by_Subtype.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved best plot → {save_path}")


# ============================================================
# FIND SIGNATURE TASKS
# ============================================================
signature_dirs = sorted(
    glob.glob(
        os.path.join(root_parent, "TCGA_signature_*")
    )
)

print(f"Found {len(signature_dirs)} signature folders.")


# ============================================================
# PROCESS EACH TASK
# ============================================================
for signature_dir in signature_dirs:

    signature_name = os.path.basename(signature_dir)

    print(f"\nProcessing {signature_name}")

    # --------------------------------------------------------
    # outcome-wise results
    # --------------------------------------------------------
    outcome_results = {}

    # --------------------------------------------------------
    # walk files
    # --------------------------------------------------------
    for root, dirs, files in os.walk(signature_dir):

        for file in files:

            if not file.endswith("_results.json"):
                continue

            file_path = os.path.join(root, file)

            folder_name = os.path.basename(
                os.path.dirname(file_path)
            )

            # ------------------------------------------------
            # load json
            # ------------------------------------------------
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

            # ------------------------------------------------
            # required keys
            # ------------------------------------------------
            if not all(
                k in data
                for k in ["subject", "pred", "label"]
            ):
                continue

            subjects = data["subject"]

            preds_all = np.array(data["pred"])
            labels_all = np.array(data["label"])

            # ------------------------------------------------
            # ensure 2D
            # shape = [N, num_outcomes]
            # ------------------------------------------------
            if preds_all.ndim == 1:
                preds_all = preds_all[:, None]

            if labels_all.ndim == 1:
                labels_all = labels_all[:, None]

            num_outcomes = preds_all.shape[1]

            model_name, omics_type = parse_model_name(
                folder_name,
                file
            )

            # ------------------------------------------------
            # subtype matching
            # ------------------------------------------------
            sample_subtypes = []
            valid_indices = []

            for i, sid in enumerate(subjects):

                row = immune_df.loc[
                    immune_df["ID3"] == sid
                ]

                if len(row) == 0:
                    continue

                sample_subtypes.append(
                    row.iloc[0]["Subtype"]
                )

                valid_indices.append(i)

            if len(valid_indices) == 0:
                continue

            preds_all = preds_all[valid_indices]
            labels_all = labels_all[valid_indices]

            # ------------------------------------------------
            # process EACH outcome separately
            # ------------------------------------------------
            for outcome_idx in range(num_outcomes):

                outcome_name = signature_labels[signature_name][outcome_idx]

                if outcome_name not in outcome_results:
                    outcome_results[outcome_name] = []

                preds = preds_all[:, outcome_idx]
                labels = labels_all[:, outcome_idx]

                # remove NaNs
                valid = ~(
                    np.isnan(preds) |
                    np.isnan(labels)
                )

                preds = preds[valid]
                labels = labels[valid]

                valid_subtypes = np.array(
                    sample_subtypes
                )[valid]

                if len(preds) < 2:
                    continue

                # ------------------------------------------------
                # subgroup analysis
                # ------------------------------------------------
                counts = Counter(valid_subtypes)

                for subtype in counts:

                    idx = [
                        i
                        for i, st in enumerate(valid_subtypes)
                        if st == subtype
                    ]

                    if len(idx) < MIN_SAMPLES:
                        continue

                    sub_preds = preds[idx]
                    sub_labels = labels[idx]

                    try:
                        r2 = r2_score(
                            sub_labels,
                            sub_preds
                        )

                    except Exception:
                        continue

                    outcome_results[outcome_name].append({
                        "Model": model_name,
                        "Omics": omics_type,
                        "Subtype": subtype,
                        "R2": r2,
                        "N": len(idx),
                    })

    # ========================================================
    # SAVE + PLOT EACH OUTCOME
    # ========================================================
    for outcome_name, results in outcome_results.items():

        if len(results) == 0:
            continue

        result_df = pd.DataFrame(results)

        outcome_name = outcome_name.replace("/", "-")
        
        output_dir = os.path.join(
            output_root,
            signature_name,
            outcome_name
        )

        os.makedirs(output_dir, exist_ok=True)

        # ----------------------------------------------------
        # save csv
        # ----------------------------------------------------
        csv_path = os.path.join(
            output_dir,
            f"{outcome_name}_R2_by_subtype.csv"
        )

        result_df.to_csv(csv_path, index=False)

        print(f"Saved table → {csv_path}")

        # ----------------------------------------------------
        # grouped plot
        # ----------------------------------------------------
        plot_omics_grouped(
            result_df,
            metric="R2",
            output_dir=output_dir,
            sort_subtype="IFN-gamma Dominant",
            omics="Radiopathomics",
            task_name=signature_name,
            outcome_name=outcome_name
        )

        # ----------------------------------------------------
        # best plot
        # ----------------------------------------------------
        plot_best_omics_by_subtype(
            result_df,
            metric="R2",
            output_dir=output_dir,
            task_name=signature_name,
            outcome_name=outcome_name
        )

print("\n✅ Done.")