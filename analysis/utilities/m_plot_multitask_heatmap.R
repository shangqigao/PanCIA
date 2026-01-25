# ==============================
# Heatmap for Multiple Tasks
# ==============================
install.packages(c(
  "ggplot2",
  "dplyr",
  "purrr",
  "stringr",
  "jsonlite",
  "scales",
  "tidyr",
  "patchwork",
  "purrr"
))

library(ggplot2)
library(dplyr)
library(purrr)
library(stringr)
library(jsonlite)
library(tidyr)
library(scales)
library(patchwork)
library(purrr)

base_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes"
out_dir  <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots"

big_tasks <- list(
  survival   = list(subtasks = c("OS", "DSS", "DFI", "PFI"), metric = "C-index"),
  phenotype  = list(subtasks = c("ImmuneSubtype", "MolecularSubtype", "PrimaryDisease"), metric = "f1"),
  signature  = list(subtasks = c("GeneProgrames", "HRDscore", "ImmuneSignatureScore", "StemnessScoreDNA", "StemScoreRNA"), metric = "r2")
)

agg_levels <- c("MEAN", "ABMIL", "SPARRA")
omics_levels <- c("radiomics", "pathomics", "radiopathomics")

row_order <- c("MEAN", "ABMIL", "SPARRA")
omics_type_order <- c("radiomics", "pathomics", "radiopathomics")
radiomics_order  <- c("pyradiomics", "BiomedParse", "LVMMed")
pathomics_order  <- c("CHIEF", "CONCH", "UNI")

task_palettes <- list(
  survival  = viridis_pal(option = "inferno"),
  phenotype = viridis_pal(option = "magma"),
  signature = viridis_pal(option = "plasma")
)

parse_one <- function(file, big_task, metric_name) {

  fname <- basename(file)

  m <- str_match(
    fname,
    "^(radiomics|pathomics|radiopathomics)_radio\\+([^_]+)_patho\\+([^_]+)_"
  )
  if (any(is.na(m))) return(NULL)

  omics_type <- m[2]
  radio_agg  <- m[3]
  patho_agg  <- m[4]

  parent_dir <- basename(dirname(file))
  m2 <- str_match(parent_dir, "^([^+]+)\\+([^+]+)$")
  if (any(is.na(m2))) return(NULL)

  radio_mode <- m2[2]
  patho_mode <- m2[3]

  if (radio_mode == "pyradiomics") radio_agg <- "MEAN"

  agg_mode <- case_when(
    omics_type == "radiomics"      ~ radio_agg,
    omics_type == "pathomics"      ~ patho_agg,
    omics_type == "radiopathomics" ~ radio_agg
  )

  model <- case_when(
    omics_type == "radiomics"      ~ radio_mode,
    omics_type == "pathomics"      ~ patho_mode,
    omics_type == "radiopathomics" ~ paste(radio_mode, patho_mode, sep = "+")
  )

  js <- fromJSON(file)

  metric_vals <- js[names(js) != "p-value"] |>
    map_dbl(~ mean(as.numeric(.x[[metric_name]]), na.rm = TRUE))
  
  tibble(
    big_task   = big_task,
    omics_type = omics_type,
    agg_mode   = agg_mode,
    model      = model,
    metric     = mean(metric_vals, na.rm = TRUE)
  )
}


df_all <- imap_dfr(big_tasks, function(cfg, big_task) {

  task_dirs <- list.dirs(
    path = base_dir,
    recursive = FALSE,
    full.names = TRUE
  ) %>% keep(~ str_detect(.x, paste0("TCGA_", big_task)))

  json_files <- list.files(
    path = task_dirs,
    pattern = "_metrics\\.json$",
    recursive = TRUE,
    full.names = TRUE
  )

  map_dfr(json_files, parse_one,
          big_task = big_task,
          metric_name = cfg$metric)
})

# ------------------------------
# Explicit row and column order
# ------------------------------
radiopathomics_order <- c()
for (r in radiomics_order) {
  for (p in pathomics_order) {
    radiopathomics_order <- c(radiopathomics_order, paste(r, p, sep = "+"))
  }
}

df_plot <- df_all %>%
  group_by(big_task, omics_type, agg_mode, model) %>%
  summarise(mean_metric = mean(metric, na.rm = TRUE), .groups = "drop") %>%
  # reorder agg_mode per task
  mutate(
    agg_mode = factor(agg_mode, levels = row_order),
    big_task = factor(big_task, levels = names(big_tasks)),
    omics_type = factor(omics_type, levels = omics_type_order)
  ) %>%
  # create row_id using ordered agg_mode
  arrange(big_task, agg_mode) %>%
  mutate(
    row_id = paste(big_task, agg_mode, sep = " | "),
    row_id = factor(row_id, levels = paste(rep(names(big_tasks), each = length(row_order)),
                                           rep(row_order, times = length(names(big_tasks))),
                                           sep = " | "))
  )

# ------------------------------
# Column (model) order per omics_type
# ------------------------------
model_order_list <- list(
  radiomics      = radiomics_order,
  pathomics      = pathomics_order,
  radiopathomics = radiopathomics_order
)

df_plot <- df_plot %>%
  group_by(omics_type) %>%
  mutate(model = factor(model, levels = model_order_list[[unique(omics_type)]])) %>%
  ungroup()

# ------------------------------
# Function to plot one big_task
# ------------------------------
plot_one_task <- function(task_name) {
  cfg <- big_tasks[[task_name]]
  d <- df_plot %>% filter(big_task == task_name)
  if(nrow(d) == 0) return(NULL)

  d <- d %>% mutate(row_id = factor(row_id, levels = rev(unique(row_id))))

  palette_option <- switch(task_name,
                           survival  = "inferno",
                           phenotype = "magma",
                           signature = "plasma")

  ggplot(d, aes(x=model, y=row_id, fill=mean_metric)) +
    geom_tile(color="white", linewidth=0.3) +
    facet_grid(. ~ omics_type, scales="free_x", space="free_x") +
    scale_fill_viridis_c(option=palette_option, name=cfg$metric) +
    labs(title=str_to_title(task_name), x=NULL, y=NULL) +
    theme_minimal(base_size=13) +
    theme(
      axis.text.x = element_text(angle=45, hjust=1),
      strip.text = element_text(face="bold"),
      panel.grid = element_blank(),
      legend.title = element_text(face="bold")
    )
}

# ------------------------------
# Generate plots for each task
# ------------------------------
task_plots <- map(names(big_tasks), plot_one_task) %>% compact()

# Combine vertically
combined_plot <- wrap_plots(task_plots, ncol=1) & theme(legend.position="right")

# ------------------------------
# Save
# ------------------------------
n_rows <- df_plot %>% distinct(big_task, row_id) %>% count(big_task) %>% pull(n)

ggsave(file.path(out_dir, "multi_task_heatmap.png"),
       combined_plot,
       width=180,
       height=sum(n_rows)*30,
       units="mm",
       dpi=300)
