# ==============================
# Bubble Heatmap for Multiple Survival Tasks
# ==============================
install.packages(c(
  "ggplot2",
  "dplyr",
  "purrr",
  "stringr",
  "jsonlite",
  "scales",
  "tidyr"
))

library(ggplot2)
library(dplyr)
library(purrr)
library(stringr)
library(jsonlite)
library(scales)
library(tidyr)

# ------------------------------
# USER SETTINGS
# ------------------------------
base_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes"
out_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots"
tasks <- c("OS", "DSS", "DFI", "PFI")
metric_name <- "C-index"

agg_levels <- c("None", "MEAN", "ABMIL", "SPARRA")
omics_levels <- c("radiomics", "pathomics", "radiopathomics")

task_colors <- c(
  OS  = "#1B9E77",
  DSS = "#D95F02",
  DFI = "#7570B3",
  PFI = "#E7298A"
)

row_order <- c("SPARRA", "ABMIL", "MEAN")

radiomics_order <- c("pyradiomics", "BiomedParse", "LVMMed")
pathomics_order <- c("CHIEF", "CONCH", "UNI")
omics_type_order <- c("radiomics", "pathomics", "radiopathomics")

# ------------------------------
# 1. Recursively get JSON files
# ------------------------------
json_files <- map_dfr(tasks, function(task) {

  task_dir <- file.path(base_dir, paste0("TCGA_survival_", task))

  files <- list.files(
    path = task_dir,
    pattern = "_metrics\\.json$",
    recursive = TRUE,
    full.names = TRUE
  )

  tibble(
    file = files,
    task = task
  )
})

# ------------------------------
# 2. Parse filename + load JSON
# ------------------------------
parse_one <- function(file, task) {
  fname <- basename(file)

  # ---- parse omics type + aggregation from filename ----
  m <- str_match(
    fname,
    "^(radiomics|pathomics|radiopathomics)_radio\\+([^_]+)_patho\\+([^_]+)_"
  )
  if (any(is.na(m))) return(NULL)

  omics_type <- m[2]
  radio_agg  <- m[3]
  patho_agg  <- m[4]

  # ---- parse radiomics/pathomics MODE from parent dir ----
  parent_dir <- basename(dirname(file))
  m2 <- str_match(parent_dir, "^([^+]+)\\+([^+]+)$")
  if (any(is.na(m2))) return(NULL)

  radio_mode <- m2[2]
  patho_mode <- m2[3]

  # ---- enforce pyradiomics aggregation rule ----
  if (radio_mode == "pyradiomics") {
    radio_agg <- "MEAN"
  }

  # ---- aggregation mode (single for radiopathomics) ----
  agg_mode <- case_when(
    omics_type == "radiomics"      ~ radio_agg,
    omics_type == "pathomics"      ~ patho_agg,
    omics_type == "radiopathomics" ~ radio_agg
  )

  # ---- model name ----
  model <- case_when(
    omics_type == "radiomics"      ~ radio_mode,
    omics_type == "pathomics"      ~ patho_mode,
    omics_type == "radiopathomics" ~ paste(radio_mode, patho_mode, sep = "+")
  )

  # ---- read json ----
  js <- fromJSON(file)
  folds <- js[str_detect(names(js), "^Fold")]
  metric_vals <- map_dbl(folds, ~ .x[[metric_name]])

  tibble(
    task        = task,
    omics_type  = omics_type,
    agg_mode    = agg_mode,
    model       = model,
    mean_metric = mean(metric_vals, na.rm = TRUE),
    log_p       = -log10(js[["p-value"]])
  )
}


df <- pmap_dfr(json_files, parse_one)

# ------------------------------
# Generate model order for radiopathomics
# ------------------------------
radiopathomics_order <- c()
for (r in radiomics_order) {
  for (p in pathomics_order) {
    radiopathomics_order <- c(radiopathomics_order, paste(r, p, sep = "+"))
  }
}

# Combine into a list
model_order_list <- list(
  radiomics      = radiomics_order,
  pathomics      = pathomics_order,
  radiopathomics = radiopathomics_order
)

# ------------------------------
# Set row order
# ------------------------------
df <- df %>%
  mutate(
    row_id = paste(task, agg_mode, sep = " | "),
    agg_mode = factor(agg_mode, levels = row_order)
  )

# Order rows by task + agg_mode
row_levels <- df %>%
  distinct(task, agg_mode) %>%
  arrange(task, factor(agg_mode, levels = row_order)) %>%
  mutate(row_id = paste(task, agg_mode, sep = " | ")) %>%
  pull(row_id)

df <- df %>%
  mutate(row_id = factor(row_id, levels = row_levels))

# ------------------------------
# Set column order per omics_type
# ------------------------------
df <- df %>%
  group_by(omics_type) %>%
  mutate(
    model = factor(model, levels = model_order_list[[unique(omics_type)]])
  ) %>%
  ungroup()

df <- df %>%
  mutate(
    omics_type = factor(omics_type, levels = omics_type_order)
  )

# Use df directly
df_complete <- df

# Global min and max for metric color scale
metric_min <- min(df_complete$mean_metric, na.rm = TRUE)
metric_max <- max(df_complete$mean_metric, na.rm = TRUE)

# ------------------------------
# Create bubble heatmap
# ------------------------------
p <- ggplot(df_complete, aes(x = model, y = row_id)) +

  # Bubble points
  geom_point(
    aes(size = log_p, color = mean_metric),
    alpha = 0.85,
    na.rm = TRUE
  ) +

  # Task row color bar
  geom_tile(
    data = df_complete %>% distinct(row_id, task),
    aes(x = -0.5, y = row_id, fill = task),
    width = 0.4,
    inherit.aes = FALSE
  ) +

  # Facet by omics_type
  facet_grid(. ~ omics_type, scales = "free_x", space = "free_x") +

  # Color scale for metric
  scale_color_viridis_c(
    option = "inferno",
    name = paste0("Mean ", metric_name),
    limits = c(metric_min, metric_max),
    oob = scales::squish
  ) +

  # Bubble size scale
  scale_size(
    range = c(2, 6),
    name = expression(-log[10]~"(p-value)")
  ) +

  # Task fill colors
  scale_fill_manual(
    values = task_colors,
    name = "Task"
  ) +

  coord_cartesian(clip = "off") +

  labs(x = NULL, y = NULL) +

  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 13),
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(t = 15, r = 15, b = 5, l = 5)
  )

# ------------------------------
# 6. Save heatmap
# ------------------------------
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_rows <- length(unique(df_complete$row_id))

ggsave(
  file.path(out_dir, "survival_bubble_heatmap.png"),
  plot  = p,
  width = 180,
  height = n_rows * 10,   # ~6 mm per row
  units = "mm",
  dpi   = 300
)

