# ==============================
# Bubble Heatmap for Multiple Survival Tasks
# ==============================

library(jsonlite)
library(tidyverse)
library(stringr)
library(ggplot2)
library(scales)

# ------------------------------
# USER SETTINGS
# ------------------------------
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

# ------------------------------
# 1. Recursively get JSON files
# ------------------------------
json_files <- map_dfr(tasks, function(task) {
  files <- list.files(
    path = task,
    pattern = "_metrics\\.json$",
    recursive = TRUE,
    full.names = TRUE
  )
  tibble(file = files, task = task)
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

  # ---- read json ----
  js <- fromJSON(file)

  folds <- js[str_detect(names(js), "^Fold")]
  metric_vals <- map_dbl(folds, ~ .x[[metric_name]])

  tibble(
    task = task,
    omics_type = omics_type,

    radio_agg = radio_agg,
    patho_agg = patho_agg,

    agg_mode = case_when(
      omics_type == "radiomics"      ~ radio_agg,
      omics_type == "pathomics"      ~ patho_agg,
      omics_type == "radiopathomics" ~ paste0("R:", radio_agg, " | P:", patho_agg)
    ),

    # ✅ column name logic (this is the key change)
    model = case_when(
      omics_type == "radiomics"      ~ radio_mode,
      omics_type == "pathomics"      ~ patho_mode,
      omics_type == "radiopathomics" ~ paste(radio_mode, patho_mode, sep = "+")
    ),

    mean_metric = mean(metric_vals, na.rm = TRUE),
    log_p = -log2(js[["p-value"]])
  )
}


df <- pmap_dfr(json_files, parse_one)

# ------------------------------
# 3. Factor handling & layout
# ------------------------------
df <- df %>%
  mutate(
    task = factor(task, levels = tasks),
    omics_type = factor(omics_type, levels = omics_levels),
    agg_mode = factor(agg_mode, levels = agg_levels),
    model = factor(model)
  )

# Create combined row variable
df <- df %>%
  mutate(row_id = interaction(task, agg_mode, sep = " | "))

# Ensure missing cells stay empty
df_complete <- df %>%
  complete(
    row_id,
    model,
    omics_type,
    fill = list(mean_metric = NA, log_p = NA)
  )

# ------------------------------
# 4. Bubble heatmap
# ------------------------------
p <- ggplot(
  df_complete,
  aes(x = model, y = row_id)
) +
  geom_point(
    aes(size = log_p, color = mean_metric),
    alpha = 0.85,
    na.rm = TRUE
  ) +
  facet_grid(
    . ~ omics_type,
    scales = "free_x",
    space = "free_x"
  ) +
  scale_color_viridis_c(
    option = "magma",
    name = paste0("Mean ", metric_name),
    limits = c(0.5, 0.75),
    oob = squish
  ) +
  scale_size(
    range = c(2, 9),
    name = expression(-log[2]~"(p-value)")
  ) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 10),
    strip.text = element_text(face = "bold", size = 13),
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA)
  )

# ------------------------------
# 5. Add task row color bar
# ------------------------------
p <- p +
  geom_tile(
    data = df_complete %>% distinct(row_id, task),
    aes(x = -0.5, y = row_id, fill = task),
    width = 0.4,
    inherit.aes = FALSE
  ) +
  scale_fill_manual(
    values = task_colors,
    name = "Task"
  ) +
  coord_cartesian(clip = "off") +
  theme(
    plot.margin = margin(5.5, 5.5, 5.5, 40)
  )

print(p)

# ------------------------------
# 6. save heatmap
# ------------------------------
out_dir <- "/home/sg2162/rds/hpc-work/PanCIA/figures/plots"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ggsave(
  file.path(out_dir, "survival_bubble_heatmap.png"),
  plot = p,
  width = 180,        # mm (Nature column width)
  height = 100,
  units = "mm",
  dpi = 300
)
