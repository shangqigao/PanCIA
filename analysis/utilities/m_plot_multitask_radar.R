# Install and load packages
install.packages(c(
  "ggplot2", "dplyr", "purrr", "stringr", "jsonlite",
  "tidyr", "scales", "patchwork", "cowplot"
))
install.packages("remotes")
remotes::install_github("ricardo-bion/ggradar")

library(ggplot2)
library(dplyr)
library(purrr)
library(stringr)
library(jsonlite)
library(tidyr)
library(scales)
library(patchwork)
library(cowplot)
library(ggradar)

# Set directories
base_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes"
out_dir  <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots"

# Create output directory if it doesn't exist
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# Define tasks and metrics
big_tasks <- list(
  survival   = list(metric = "C-index"),
  phenotype  = list(metric = "f1"),
  signature  = list(metric = "r2")
)

radiomics_order  <- c("pyradiomics", "BiomedParse", "LVMMed")
pathomics_order  <- c("CHIEF", "CONCH", "UNI")

radiopathomics_order <- c()
for (r in radiomics_order) {
  for (p in pathomics_order) {
    radiopathomics_order <- c(radiopathomics_order, paste(r, p, sep = "+"))
  }
}

# Define colors and styles
omics_colors <- c(
  radiomics      = "#1f78b4",
  pathomics      = "#33a02c",
  radiopathomics = "#6a3d9a"
)

agg_colors <- c(
  MEAN   = "#e41a1c",
  ABMIL  = "#377eb8", 
  SPARRA = "#4daf4a"
)

agg_linetypes <- c(
  MEAN   = "solid",
  ABMIL  = "dashed",
  SPARRA = "dotted"
)

metric_ranges <- list(
  survival  = c(0.6, 0.8),
  phenotype = c(0.4, 0.7),
  signature = c(-0.5, 0.6)
)

# Parse JSON files
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

# Load all data
df_all <- imap_dfr(big_tasks, function(cfg, big_task) {
  
  task_dirs <- list.dirs(base_dir, recursive = FALSE) |>
    keep(~ str_detect(.x, paste0("TCGA_", big_task)))
  
  json_files <- list.files(
    task_dirs,
    pattern = "_metrics\\.json$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  map_dfr(
    json_files,
    parse_one,
    big_task = big_task,
    metric_name = cfg$metric
  )
})

# Summarize data
df_plot <- df_all |>
  group_by(big_task, omics_type, agg_mode, model) |>
  summarise(mean_metric = mean(metric, na.rm = TRUE), .groups = "drop")

# Prepare radar data
prepare_radar_data <- function(task_name) {
  
  d <- df_plot |>
    filter(big_task == task_name)
  
  d$model <- factor(
    d$model,
    levels = c(radiomics_order, pathomics_order, radiopathomics_order)
  )
  
  d_wide <- d |>
    select(agg_mode, model, mean_metric) |>
    pivot_wider(names_from = model, values_from = mean_metric) |>
    arrange(agg_mode)
  
  colnames(d_wide)[1] <- "group"
  
  ## identify metric columns
  metric_cols <- setdiff(colnames(d_wide), "group")
  
  ## 1️⃣ collapse list-columns to single numeric per cell
  d_wide[metric_cols] <- lapply(
    d_wide[metric_cols],
    function(col) {
      vapply(
        col,
        function(x) mean(as.numeric(x), na.rm = TRUE),
        numeric(1)
      )
    }
  )
  
  range <- metric_ranges[[task_name]]
  
  d_wide[metric_cols] <- lapply(
    d_wide[metric_cols],
    function(x) {
      x[is.na(x)] <- mean(x, na.rm = TRUE)
      x <- pmin(pmax(x, range[1]), range[2])  # clamp
      scales::rescale(x, to = c(0, 1), from = range)
    }
  )
  
  d_wide
}

# Plotting function with ALL axis labels
plot_one_task_radar <- function(task_name) {
  
  d_radar <- prepare_radar_data(task_name)
  if (nrow(d_radar) == 0) return(NULL)
  
  # Create base radar plot WITHOUT ggradar's built-in colors
  base_plot <- ggradar(
    d_radar,
    grid.min = 0, grid.mid = 0.5, grid.max = 1,
    axis.label.size = 0,  # We'll add custom labels
    group.colours = agg_colors[d_radar$group],
    group.line.width = 1.1,
    fill = TRUE,
    fill.alpha = 0.08,
    background.circle.colour = "white",
    grid.label.size = 3,
    plot.legend = FALSE,  # We'll create custom legend
    legend.position = "none"  # Disable ggradar legend
  )
  
  # Get axis information
  axes <- colnames(d_radar)[-1]
  n_axes <- length(axes)
  
  # Calculate angles for each axis
  angles <- ((seq_len(n_axes) - 1) / n_axes) * 2 * pi
  
  # Create data for ALL axis labels
  create_all_axis_labels <- function() {
    tibble(
      axis = axes,
      omics = case_when(
        axes %in% radiomics_order ~ "radiomics",
        axes %in% pathomics_order ~ "pathomics",
        TRUE ~ "radiopathomics"
      ),
      angle_rad = angles,
      # Position labels at 1.1 radius
      x = 1.1 * sin(angles),
      y = 1.1 * cos(angles),
      # Calculate label angles in degrees for rotation
      label_angle = ifelse(angles > pi/2 & angles < 3*pi/2,
                          angles * 180/pi + 180,
                          angles * 180/pi),
      # Adjust horizontal justification
      hjust = ifelse(angles > pi/2 & angles < 3*pi/2, 1, 0),
      vjust = 0.5
    )
  }
  
  # Create data for group separators
  create_group_separators <- function() {
    counts <- c(
      sum(axes %in% radiomics_order),
      sum(axes %in% pathomics_order),
      sum(axes %in% radiopathomics_order)
    )
    
    cumulative <- cumsum(counts)
    # Start positions (between groups)
    start_idx <- c(1, cumulative[-length(cumulative)] + 1)
    
    # Convert to angles (use mid-point between last of previous and first of next)
    sep_angles <- ((start_idx - 1.5) / n_axes) * 2 * pi
    
    tibble(
      angle = sep_angles,
      x = 0,
      y = 0,
      xend = 1.25 * sin(sep_angles),
      yend = 1.25 * cos(sep_angles),
      group = c("Radiomics", "Pathomics", "Radiopathomics")
    )
  }
  
  # Create data for group labels (positioned further out)
  create_group_labels <- function() {
    counts <- c(
      sum(axes %in% radiomics_order),
      sum(axes %in% pathomics_order),
      sum(axes %in% radiopathomics_order)
    )
    
    # Calculate middle index for each group
    cumulative <- cumsum(counts)
    mid_idx <- cumulative - counts/2
    
    # Convert to angles
    angles_mid <- ((mid_idx - 1) / n_axes) * 2 * pi
    
    tibble(
      label = c("RADIOMICS", "PATHOMICS", "RADIOPATHOMICS"),
      x = 1.5 * sin(angles_mid),
      y = 1.5 * cos(angles_mid),
      omics = names(omics_colors),
      angle = ifelse(angles_mid > pi/2 & angles_mid < 3*pi/2,
                     angles_mid * 180/pi - 90,
                     angles_mid * 180/pi + 90),
      hjust = 0.5,
      vjust = 0.5
    )
  }
  
  # Build the final plot
  final_plot <- base_plot +
    # Add group separators (subtle dashed lines)
    geom_segment(
      data = create_group_separators(),
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      linetype = "dashed",
      color = "grey50",
      alpha = 0.6,
      linewidth = 0.5
    ) +
    # Add ALL axis labels with rotation
    geom_text(
      data = create_all_axis_labels(),
      aes(x = x, y = y, label = axis, color = omics, angle = label_angle, hjust = hjust),
      inherit.aes = FALSE,
      size = 2.8,
      fontface = "plain",
      show.legend = FALSE
    ) +
    # Add group category labels
    geom_text(
      data = create_group_labels(),
      aes(x = x, y = y, label = label, color = omics, angle = angle),
      inherit.aes = FALSE,
      fontface = "bold",
      size = 3.5,
      hjust = 0.5,
      vjust = 0.5,
      show.legend = FALSE
    ) +
    # Add title and subtitle
    labs(
      title = str_to_title(task_name),
      subtitle = paste("Performance Metric:", big_tasks[[task_name]]$metric)
    ) +
    # Customize theme
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, margin = margin(b = 10)),
      plot.margin = margin(20, 60, 20, 20)  # Extra right margin for labels
    )
  
  # Add color scale at the end to override ggradar's defaults
  final_plot <- final_plot +
    scale_color_manual(
      name = "Omics Categories",
      values = omics_colors,
      labels = c("Radiomics", "Pathomics", "Radiopathomics"),
      guide = guide_legend(
        override.aes = list(size = 3, linetype = 0)
      )
    )
  
  return(final_plot)
}

# Create a simple legend plot for omics categories
create_omics_legend_plot <- function() {
  # Create data for the legend
  legend_data <- tibble(
    category = c("Radiomics", "Pathomics", "Radiopathomics"),
    color = omics_colors,
    x = 1:3,
    y = 1
  )
  
  p <- ggplot(legend_data, aes(x = x, y = y, color = category)) +
    geom_point(size = 5) +
    scale_color_manual(
      name = "Omics Categories",
      values = omics_colors,
      labels = c("Radiomics", "Pathomics", "Radiopathomics")
    ) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 9),
      legend.text = element_text(size = 8),
      legend.key = element_rect(fill = "white", color = "white"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )
  
  return(cowplot::get_legend(p))
}

# Create a simple legend plot for aggregation methods
create_aggregation_legend_plot <- function() {
  # Create data for the legend
  legend_data <- tibble(
    method = factor(names(agg_linetypes), levels = names(agg_linetypes)),
    color = agg_colors[names(agg_linetypes)],
    linetype = agg_linetypes,
    x = 1:3,
    y = 1
  )
  
  p <- ggplot(legend_data, aes(x = x, y = y, color = method, linetype = method, group = method)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(
      name = "Aggregation",
      values = agg_colors,
      labels = names(agg_linetypes)
    ) +
    scale_linetype_manual(
      name = "Aggregation",
      values = agg_linetypes,
      labels = names(agg_linetypes)
    ) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 9),
      legend.text = element_text(size = 8),
      legend.key.width = unit(1.2, "cm"),
      legend.key = element_rect(fill = "white", color = "white"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )
  
  return(cowplot::get_legend(p))
}

# Generate radar plots for all tasks
cat("Generating radar plots...\n")
radar_plots <- list()
for (task_name in names(big_tasks)) {
  cat("  Creating radar plot for:", task_name, "\n")
  radar_plots[[task_name]] <- plot_one_task_radar(task_name)
}

# Create legends
cat("Creating legends...\n")
omics_legend <- create_omics_legend_plot()
agg_legend <- create_aggregation_legend_plot()

# Combine legends into one plot
cat("Combining legends...\n")
combined_legend_plot <- plot_grid(
  omics_legend,
  agg_legend,
  ncol = 1,
  align = "v",
  rel_heights = c(1, 1.2)
)

# Save individual plots
cat("\nSaving individual plots...\n")
for (task_name in names(radar_plots)) {
  p <- radar_plots[[task_name]]
  if (!is.null(p)) {
    # Save with built-in legend
    ggsave(
      filename = file.path(out_dir, paste0("Radar_", task_name, "_with_legend.png")),
      plot = p,
      width = 180,
      height = 160,
      units = "mm",
      dpi = 600,
      bg = "white"
    )
    cat("  Saved:", task_name, "with legend\n")
    
    # Save without legend
    plot_no_legend <- p + theme(legend.position = "none")
    ggsave(
      filename = file.path(out_dir, paste0("Radar_", task_name, "_no_legend.png")),
      plot = plot_no_legend,
      width = 160,
      height = 160,
      units = "mm",
      dpi = 600,
      bg = "white"
    )
    cat("  Saved:", task_name, "without legend\n")
    
    # Save with separate legends using patchwork
    plot_with_separate_legends <- plot_no_legend + combined_legend_plot +
      plot_layout(widths = c(3, 1))
    
    ggsave(
      filename = file.path(out_dir, paste0("Radar_", task_name, "_all_labels.png")),
      plot = plot_with_separate_legends,
      width = 200,
      height = 160,
      units = "mm",
      dpi = 600,
      bg = "white"
    )
    cat("  Saved:", task_name, "with separate legends\n")
  }
}

# Create a combined figure with all three tasks
cat("\nCreating combined figure...\n")
if (all(names(big_tasks) %in% names(radar_plots))) {
  # Create plots without legends
  survival_plot <- radar_plots$survival + 
    theme(legend.position = "none") +
    labs(title = "Survival\n(C-index)")
  
  phenotype_plot <- radar_plots$phenotype + 
    theme(legend.position = "none") +
    labs(title = "Phenotype\n(F1 score)")
  
  signature_plot <- radar_plots$signature + 
    theme(legend.position = "none") +
    labs(title = "Signature\n(R²)")
  
  # Create the combined plot
  combined_plot <- (survival_plot | phenotype_plot | signature_plot | combined_legend_plot) +
    plot_layout(widths = c(1, 1, 1, 0.7)) +
    plot_annotation(
      title = "Multi-Omics Model Performance Across Different Tasks",
      theme = theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.margin = margin(20, 20, 20, 20)
      )
    )
  
  # Save combined figure
  ggsave(
    filename = file.path(out_dir, "Radar_All_Tasks_Combined.png"),
    plot = combined_plot,
    width = 320,
    height = 120,
    units = "mm",
    dpi = 600,
    bg = "white"
  )
  
  cat("  Saved combined figure\n")
}

# Save the legends separately
cat("\nSaving legends separately...\n")
ggsave(
  filename = file.path(out_dir, "Radar_Legend_Omics.png"),
  plot = omics_legend,
  width = 80,
  height = 60,
  units = "mm",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = file.path(out_dir, "Radar_Legend_Aggregation.png"),
  plot = agg_legend,
  width = 80,
  height = 60,
  units = "mm",
  dpi = 600,
  bg = "white"
)

cat("\n=== Summary ===\n")
cat("Output directory:", out_dir, "\n")
cat("Plots created for each task:\n")
cat("  - With legend (built-in)\n")
cat("  - Without legend\n")
cat("  - With separate legends\n")
cat("Combined figure with all tasks\n")
cat("Individual legend files\n")
cat("Total files:", length(names(big_tasks)) * 3 + 3, "\n")
cat("\nDone!\n")