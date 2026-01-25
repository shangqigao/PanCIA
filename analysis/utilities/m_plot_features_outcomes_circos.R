library(jsonlite)
library(dplyr)
library(circlize)

# -------- 1. Load all fold JSON files --------
root_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments"

files <- file.path(
  root_dir,
  sprintf("%02d", 0:4),
  "all_predictor_weights.json"
)

fold_data <- lapply(files, fromJSON)

# -------- 2. Average weights across folds --------
average_folds <- function(fold_list) {
  
  avg_data <- list()
  
  modalities <- names(fold_list[[1]])
  
  for (mod in modalities) {
    
    outcomes <- names(fold_list[[1]][[mod]])
    modality_list <- list()
    
    for (out in outcomes) {
      
      features <- names(fold_list[[1]][[mod]][[out]])
      
      # Collect weights across folds
      weight_matrix <- sapply(fold_list, function(fold) {
        as.numeric(fold[[mod]][[out]][features])
      })
      
      # Average
      mean_weights <- rowMeans(weight_matrix)
      names(mean_weights) <- features
      
      modality_list[[out]] <- mean_weights
    }
    
    avg_data[[mod]] <- modality_list
  }
  
  return(avg_data)
}

avg_data <- average_folds(fold_data)

# -------- 3. Circos plotting function --------
plot_modality_circos <- function(modality_name, modality_data, top_k = 20) {
  
  # Convert to dataframe
  df_list <- list()
  
  for (outcome in names(modality_data)) {
    
    weights <- modality_data[[outcome]]
    
    temp_df <- data.frame(
      feature = names(weights),
      weight = as.numeric(weights),
      outcome = outcome,
      stringsAsFactors = FALSE
    )
    
    # Top-k by absolute weight
    temp_df <- temp_df %>%
      arrange(desc(abs(weight))) %>%
      slice_head(n = top_k)
    
    df_list[[length(df_list) + 1]] <- temp_df
  }
  
  df <- bind_rows(df_list)
  df$abs_weight <- abs(df$weight)
  
  circos.clear()
  circos.par(gap.degree = 4)
  
  features <- unique(df$feature)
  outcomes <- unique(df$outcome)
  sectors <- c(features, outcomes)
  
  circos.initialize(factors = sectors, xlim = c(0, 1))
  
  circos.trackPlotRegion(
    ylim = c(0, 1),
    panel.fun = function(x, y) {
      sector.name <- get.cell.meta.data("sector.index")
      circos.text(
        0.5, 0.5, sector.name,
        facing = "clockwise",
        niceFacing = TRUE,
        cex = 0.5
      )
    },
    bg.border = NA
  )
  
  max_weight <- max(df$abs_weight)
  
  for (i in 1:nrow(df)) {
    
    link_color <- ifelse(df$weight[i] > 0,
                         "#E41A1C80",   # red positive
                         "#377EB880")   # blue negative
    
    circos.link(
      df$feature[i], 0.5,
      df$outcome[i], 0.5,
      col = link_color,
      lwd = 4 * df$abs_weight[i] / max_weight
    )
  }
  
  title(paste("Averaged Circos:", modality_name))
}

# -------- 4. Plot each modality --------
for (modality_name in names(avg_data)) {
  
  png(paste0("circos_", modality_name, ".png"),
      width = 3000,
      height = 3000,
      res = 300)
  
  plot_modality_circos(
    modality_name,
    avg_data[[modality_name]],
    top_k = 30
  )
  
  dev.off()
}

