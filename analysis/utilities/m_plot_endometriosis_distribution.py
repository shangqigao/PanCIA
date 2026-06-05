import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/Endometriosis"
output_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/Endometriosis"


df = pd.read_csv(f'{root_dir}/EndoMRI/test_BiomedParse_with_LoRA_results.csv')

# =========================
# Settings
# =========================

metric = "ca"  # change to accuracy, dice, precision, recall, etc.

target_classes = [
    "endometrioma in pelvis",
    "uterus",
    "ovary",
]

# =========================
# Get images containing endometrioma
# =========================

endo_imgs = df.loc[
    df["class"] == "endometrioma in pelvis",
    "img_name"
].unique()

# Keep only target classes on those images
df_plot = df[
    df["img_name"].isin(endo_imgs)
    & df["class"].isin(target_classes)
].copy()

# =========================
# Build image × class matrix
# =========================

heatmap_df = (
    df_plot
    .pivot_table(
        index="img_name",
        columns="class",
        values=metric,
        aggfunc="mean"  # change if needed
    )
    .reindex(columns=target_classes)
)

# Sort images by endometrioma score
heatmap_df = heatmap_df.sort_values(
    by="endometrioma in pelvis",
    ascending=False,
    na_position="last"
)


# =====================================
# Prepare matrix for visualization
# =====================================

# classes on y-axis, images on x-axis
heatmap_plot = heatmap_df.T

# color map
cmap = sns.color_palette("viridis", as_cmap=True)
cmap.set_bad("lightgray")

# =====================================
# 1. Regular heatmap
# =====================================

fig_width = max(16, len(heatmap_plot.columns) * 0.15)

plt.figure(figsize=(fig_width, 4))

sns.heatmap(
    heatmap_plot,
    cmap=cmap,
    vmin=0,
    vmax=1,
    linewidths=0.1,
    linecolor="white",
    cbar_kws={"label": metric}
)

plt.title(
    f"{metric} heatmap\n"
    "Gray = class absent"
)

plt.xlabel(f"Images (n={heatmap_plot.shape[1]})")
plt.ylabel("Class")

# hide image labels if many images
if heatmap_plot.shape[1] > 30:
    plt.xticks([])

plt.tight_layout()

plt.savefig(
    f"{output_dir}/{metric}_heatmap_horizontal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================
# 2. Clustered heatmap
# =====================================

# clustermap cannot handle NaN directly
cluster_data = heatmap_plot.fillna(-0.1)

cluster_cmap = sns.color_palette("viridis", as_cmap=True)
cluster_cmap.set_under("lightgray")

g = sns.clustermap(
    cluster_data,
    cmap=cluster_cmap,
    vmin=0,
    vmax=1,

    # cluster images only
    col_cluster=True,
    row_cluster=False,

    linewidths=0.05,

    figsize=(fig_width, 4),

    xticklabels=False,

    cbar_kws={
        "label": metric
    }
)

g.ax_heatmap.set_xlabel(
    f"Images (clustered, n={heatmap_plot.shape[1]})"
)

g.ax_heatmap.set_ylabel("Class")

g.ax_heatmap.set_title(
    f"{metric} clustered heatmap\n"
    "Gray = class absent"
)

g.savefig(
    f"{output_dir}/{metric}_clustermap_horizontal.png",
    dpi=300
)

plt.close()
