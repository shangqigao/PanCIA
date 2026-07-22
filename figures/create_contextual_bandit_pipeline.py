"""Create a widescreen, PowerPoint-ready ContextualBandit pipeline figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT = Path(__file__).with_name("contextual_bandit_pipeline.png")

COLORS = {
    "ink": "#17233A",
    "muted": "#4D5B72",
    "line": "#65738A",
    "blue": "#E8F1FF",
    "blue_edge": "#4F78BE",
    "green": "#E6F6ED",
    "green_edge": "#3C8968",
    "purple": "#EFEAFF",
    "purple_edge": "#765CBA",
    "orange": "#FFF0D9",
    "orange_edge": "#B9792C",
    "teal": "#E5F7F5",
    "teal_edge": "#31867F",
}


def box(ax, xy, width, height, title, lines=(), fill="blue", fontsize=15):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=2,
        edgecolor=COLORS[f"{fill}_edge"],
        facecolor=COLORS[fill],
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height * 0.70, title,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold", color=COLORS["ink"],
    )
    if lines:
        ax.text(
            x + width / 2, y + height * 0.34, "\n".join(lines),
            ha="center", va="center", fontsize=fontsize - 2,
            color=COLORS["muted"], linespacing=1.35,
        )
    return patch


def arrow(ax, start, end, color=None, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=19,
        linewidth=2.3, color=color or COLORS["line"],
        connectionstyle=connectionstyle,
    ))


def phase(ax, x, number, title, color):
    ax.text(
        x - 0.09, 0.885, str(number), ha="center", va="center",
        fontsize=15, fontweight="bold", color="white",
        bbox=dict(boxstyle="circle,pad=0.36", facecolor=color, edgecolor="none"),
    )
    ax.text(
        x + 0.02, 0.885, title, ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLORS["ink"],
    )


fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5, 0.965,
    "ContextualBandit: patient-specific hard selection of survival experts",
    ha="center", va="center", fontsize=25, fontweight="bold", color=COLORS["ink"],
)

phase(ax, 0.125, 1, "Experts", COLORS["blue_edge"])
phase(ax, 0.362, 2, "Policy state", COLORS["purple_edge"])
phase(ax, 0.64, 3, "EM policy learning", COLORS["orange_edge"])
phase(ax, 0.892, 4, "Hard deployment", COLORS["teal_edge"])

# Stage 1
box(ax, (0.025, 0.675), 0.20, 0.13, "Training data",
    ("Radiomics Xᴿ  ·  Pathomics Xᴾ", "Time T  ·  Event E"), "blue")
arrow(ax, (0.125, 0.675), (0.125, 0.625))

expert_y = 0.48
for x, name, subtitle in [
    (0.025, "R", "Radiomic Cox"),
    (0.096, "P", "Pathomic Cox"),
    (0.167, "RP", "Fusion Cox"),
]:
    box(ax, (x, expert_y), 0.058, 0.12, name, tuple(subtitle.split()), "green", fontsize=14)

arrow(ax, (0.125, 0.48), (0.125, 0.425))
box(ax, (0.025, 0.29), 0.20, 0.12, "Expert predictions",
    ("Full-fit risks", "Out-of-fold risks"), "green")
box(ax, (0.025, 0.115), 0.20, 0.11, "Training-only references",
    ("Expert-risk calibration",), "blue", fontsize=14)

# Stage 2
state = FancyBboxPatch(
    (0.26, 0.20), 0.205, 0.605,
    boxstyle="round,pad=0.015,rounding_size=0.02",
    linewidth=2.2, edgecolor=COLORS["purple_edge"], facecolor=COLORS["purple"],
)
ax.add_patch(state)
ax.text(0.3625, 0.755, "Compact four-dimensional state", ha="center", va="center",
        fontsize=16, fontweight="bold", color=COLORS["ink"])
ax.plot([0.285, 0.44], [0.715, 0.715], color="#B9ACD8", linewidth=1.8)
ax.text(0.285, 0.665, "Expert risks", fontsize=15, fontweight="bold", color=COLORS["ink"])
ax.text(0.285, 0.625, "R,  P,  RP", fontsize=15, color=COLORS["muted"])
ax.text(0.285, 0.55, "Signed contrast", fontsize=15, fontweight="bold", color=COLORS["ink"])
ax.text(0.285, 0.50, "R − P", fontsize=15, color=COLORS["muted"])
ax.text(0.285, 0.405, "Version B", fontsize=15, fontweight="bold", color=COLORS["ink"])
ax.text(0.285, 0.36, "R,  P,  RP,  R − P", fontsize=15, color=COLORS["muted"])
ax.text(0.3625, 0.235, "Same state at deployment", ha="center", fontsize=12,
        color=COLORS["purple_edge"], fontweight="bold")

arrow(ax, (0.225, 0.35), (0.26, 0.50))
arrow(ax, (0.225, 0.17), (0.26, 0.31))
arrow(ax, (0.465, 0.50), (0.495, 0.50))

# Stage 3: EM band
em = FancyBboxPatch(
    (0.495, 0.12), 0.29, 0.685,
    boxstyle="round,pad=0.012,rounding_size=0.02",
    linewidth=1.8, linestyle="--", edgecolor="#AFB8C7", facecolor="#FAFBFD",
)
ax.add_patch(em)
ax.text(0.64, 0.77, "EM optimization", ha="center", fontsize=16,
        fontweight="bold", color=COLORS["ink"])
box(ax, (0.525, 0.635), 0.23, 0.10, "Policy network",
    ("4-D state → R / P / RP logits",), "purple", fontsize=15)
arrow(ax, (0.64, 0.635), (0.64, 0.59))
box(ax, (0.525, 0.48), 0.23, 0.105, "ST Gumbel-Softmax",
    ("Forward: one-hot action", "Backward: soft gradient"), "purple", fontsize=14)
arrow(ax, (0.64, 0.48), (0.64, 0.435))
box(ax, (0.525, 0.315), 0.23, 0.115, "Policy loss",
    ("Hard-action Cox likelihood", "+ RP cost + soft regularization"), "orange", fontsize=15)
arrow(ax, (0.64, 0.315), (0.64, 0.275))
box(ax, (0.515, 0.16), 0.115, 0.105, "Validate",
    ("Argmax + OOF risks",), "blue", fontsize=14)
box(ax, (0.655, 0.16), 0.115, 0.105, "M-step",
    ("Refit weighted Cox",), "green", fontsize=14)
arrow(ax, (0.63, 0.212), (0.655, 0.212))
ax.plot([0.77, 0.779, 0.779], [0.212, 0.212, 0.685],
        color=COLORS["purple_edge"], linewidth=2.3)
arrow(ax, (0.779, 0.685), (0.752, 0.685), COLORS["purple_edge"])
ax.text(0.773, 0.47, "repeat", rotation=90, ha="center", fontsize=12,
        fontweight="bold", color=COLORS["purple_edge"])
ax.text(0.64, 0.128, "Restore best synchronized checkpoint", ha="center",
        fontsize=11, color=COLORS["muted"])

arrow(ax, (0.785, 0.50), (0.81, 0.50))

# Stage 4
box(ax, (0.81, 0.675), 0.165, 0.13, "New patient",
    ("Xᴿ + Xᴾ", "three expert risks"), "teal")
arrow(ax, (0.8925, 0.675), (0.8925, 0.62))
box(ax, (0.81, 0.50), 0.165, 0.11, "Same 4-D state",
    ("Policy network → logits",), "purple", fontsize=15)
arrow(ax, (0.8925, 0.50), (0.8925, 0.445))
diamond = Polygon(
    [[0.8925, 0.445], [0.97, 0.355], [0.8925, 0.265], [0.815, 0.355]],
    closed=True, linewidth=2.2, edgecolor=COLORS["teal_edge"], facecolor=COLORS["teal"],
)
ax.add_patch(diamond)
ax.text(0.8925, 0.37, "ARGMAX", ha="center", va="center", fontsize=17,
        fontweight="bold", color=COLORS["ink"])
ax.text(0.8925, 0.335, "hard selection", ha="center", va="center", fontsize=13,
        color=COLORS["muted"])
arrow(ax, (0.8925, 0.265), (0.8925, 0.21))
box(ax, (0.81, 0.08), 0.165, 0.12, "Final survival risk",
    ("exactly one of  R · P · RP",), "teal")

ax.text(
    0.5, 0.035,
    "Aligned hard expert choice during policy optimization, validation, and deployment",
    ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["ink"],
)

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches=None, pad_inches=0)
plt.close(fig)
print(OUT)
