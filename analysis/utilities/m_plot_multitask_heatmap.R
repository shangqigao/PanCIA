library(ggplot2)
library(dplyr)
library(purrr)
library(stringr)
library(jsonlite)
library(tidyr)
library(scales)

base_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes"
out_dir  <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots"

big_tasks <- list(
  survival   = list(subtasks = c("OS", "DSS", "DFI", "PFI"), metric = "C-index"),
  phenotype  = list(subtasks = c("ImmuneSubtype", "MolecularSubtype", "PrimaryDisease"), metric = "F1"),
  signature  = list(subtasks = c("GeneProgrames", "HRDscore", "ImmuneSignatureScore", "StemnessScoreDNA", "StemScoreRNA"), metric = "R2")
)

agg_levels <- c("MEAN", "ABMIL", "SPARRA")

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

  metric_vals <- unlist(js[names(js) != "p-value"])[metric_name]
  metric_vals <- metric_vals[!is.na(metric_vals)]

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

df_plot <- df_all %>%
  group_by(big_task, omics_type, agg_mode, model) %>%
  summarise(mean_metric = mean(metric, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    row_id = paste(big_task, agg_mode, sep = " | "),
    agg_mode = factor(agg_mode, levels = agg_levels),
    big_task = factor(big_task, levels = names(big_tasks)),
    omics_type = factor(omics_type, levels = omics_type_order)
  )

radiopathomics_order <- as.vector(outer(radiomics_order, pathomics_order, paste, sep = "+"))

model_order_list <- list(
  radiomics      = radiomics_order,
  pathomics      = pathomics_order,
  radiopathomics = radiopathomics_order
)

df_plot <- df_plot %>%
  group_by(omics_type) %>%
  mutate(model = factor(model, levels = model_order_list[[unique(omics_type)]])) %>%
  ungroup()

p <- ggplot(df_plot, aes(x = model, y = row_id, fill = mean_metric)) +
  geom_tile(color = "white", linewidth = 0.3) +

  facet_grid(
    big_task ~ omics_type,
    scales = "free_x",
    space  = "free_x"
  ) +

  scale_fill_gradientn(
    colours = task_palettes,
    name = "Mean metric"
  ) +

  labs(x = NULL, y = NULL) +

  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    panel.grid = element_blank(),
    legend.position = "right"
  )

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ggsave(
  file.path(out_dir, "multi_task_heatmap.png"),
  p,
  width  = 180,
  height = length(unique(df_plot$row_id)) * 8,
  units  = "mm",
  dpi    = 300
)