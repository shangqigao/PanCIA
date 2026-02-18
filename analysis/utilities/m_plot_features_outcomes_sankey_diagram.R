install.packages("ggalluvial")

library(jsonlite)
library(ggplot2)
library(ggalluvial)
library(dplyr)

# -------- 1. Load all fold JSON files --------
root_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/radiomics_pathomics_GCNConv_SPARRA"
out_dir  <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/multitask"

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

library(dplyr)
library(circlize)
library(RColorBrewer)

# -------- 整理后的Circos绘图函数 --------
plot_modality_circos_debug <- function(modality_name, modality_data, 
                                      top_k_per_outcome = 10,
                                      top_features = 30,
                                      top_outcomes = 20,
                                      min_abs_weight = 0.01) {
  
  cat(paste("=== Starting circos for", modality_name, "===\n"))
  
  # 1. 数据准备
  df_list <- list()
  for (outcome in names(modality_data)) {
    weights <- modality_data[[outcome]]
    
    if (length(weights) == 0) {
      cat(paste("  Warning: Empty weights for outcome", outcome, "\n"))
      next
    }
    
    df_temp <- data.frame(
      feature = names(weights),
      outcome = outcome,
      weight = as.numeric(weights),
      stringsAsFactors = FALSE
    )
    
    df_temp <- df_temp %>%
      arrange(desc(abs(weight))) %>%
      slice_head(n = min(top_k_per_outcome, nrow(df_temp)))
    
    df_list[[outcome]] <- df_temp
  }
  
  if (length(df_list) == 0) {
    warning(paste("No data for modality:", modality_name))
    return(FALSE)
  }
  
  df <- bind_rows(df_list)
  
  # 2. 选择最重要的特征
  feature_summary <- df %>%
    group_by(feature) %>%
    summarise(total_weight = sum(abs(weight))) %>%
    arrange(desc(total_weight))
  
  top_features_selected <- feature_summary %>%
    slice_head(n = min(top_features, nrow(feature_summary))) %>%
    pull(feature)
  
  # 3. 选择最重要的结果
  outcome_summary <- df %>%
    group_by(outcome) %>%
    summarise(total_weight = sum(abs(weight))) %>%
    arrange(desc(total_weight))
  
  top_outcomes_selected <- outcome_summary %>%
    slice_head(n = min(top_outcomes, nrow(outcome_summary))) %>%
    pull(outcome)
  
  # 4. 过滤数据
  df_filtered <- df %>%
    filter(feature %in% top_features_selected,
           outcome %in% top_outcomes_selected,
           abs(weight) >= min_abs_weight)
  
  if (nrow(df_filtered) == 0) {
    warning(paste("No data after filtering for:", modality_name))
    return(FALSE)
  }
  
  # 5. 准备Circos数据
  chord_data <- df_filtered %>%
    mutate(
      from = feature,
      to = outcome,
      value = abs(weight),
      direction = ifelse(weight > 0, "positive", "negative")
    ) %>%
    select(from, to, value, direction)
  
  chord_data$width <- sqrt(chord_data$value) * 2
  
  # 6. 创建Circos图
  tryCatch({
    circos.clear()
    
    n_features <- length(top_features_selected)
    n_outcomes <- length(top_outcomes_selected)
    
    # 设置颜色
    feature_colors <- colorRampPalette(c("#2E86AB", "#A23B72"))(n_features)
    outcome_colors <- colorRampPalette(c("#F18F01", "#C73E1D"))(n_outcomes)
    
    names(feature_colors) <- top_features_selected
    names(outcome_colors) <- top_outcomes_selected
    
    # 设置图形参数
    par(mar = c(1, 1, 3, 1), cex = 0.8)
    
    # 初始化Circos
    circos.par(
      start.degree = 90,
      gap.degree = 1,
      track.margin = c(0.01, 0.01),
      points.overflow.warning = FALSE
    )
    
    # 绘制Chord图
    chordDiagram(
      chord_data[, c("from", "to", "value")],
      grid.col = c(feature_colors, outcome_colors),
      col = ifelse(chord_data$direction == "positive", "#E41A1C80", "#377EB880"),
      transparency = 0.3,
      directional = 0,
      link.sort = TRUE,
      link.decreasing = TRUE,
      link.lwd = chord_data$width,
      link.lty = 1,
      link.border = ifelse(chord_data$direction == "positive", "#E41A1C", "#377EB8"),
      annotationTrack = "grid",
      preAllocateTracks = list(track.height = 0.1)
    )
    
    # 添加标签
    circos.trackPlotRegion(
      track.index = 1,
      panel.fun = function(x, y) {
        sector.index = CELL_META$sector.index
        xlim = CELL_META$xlim
        ylim = CELL_META$ylim
        
        label <- sector.index
        if (nchar(label) > 12) {
          label <- paste0(substr(label, 1, 10), "..")
        }
        
        circos.text(
          mean(xlim), 
          ylim[1] + 0.05, 
          label, 
          facing = "clockwise",
          niceFacing = TRUE,
          adj = c(0, 0.5),
          cex = 0.6
        )
      },
      bg.border = NA
    )
    
    # 添加标题和图例
    title(main = paste("Circos:", modality_name),
          sub = paste(nrow(chord_data), "connections"),
          cex.main = 1.2, cex.sub = 0.8)
    
    legend("bottomleft",
           legend = c(paste("Pos:", sum(chord_data$direction == "positive")),
                     paste("Neg:", sum(chord_data$direction == "negative"))),
           fill = c("#E41A1C80", "#377EB880"),
           border = c("#E41A1C", "#377EB8"),
           title = "Direction",
           cex = 0.8,
           bty = "n")
    
    cat(paste("  ✓ Successfully created circos\n"))
    return(TRUE)
    
  }, error = function(e) {
    cat(paste("  ✗ Error in circos:", e$message, "\n"))
    return(FALSE)
  })
}

# -------- 整理后的生成循环 --------
library(circlize)
library(dplyr)

# 创建输出目录（如果不存在）
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

cat("\n=== Generating Circos Plots ===\n")

# 生成所有模态的Circos图
for (modality_name in names(avg_data)) {
  
  cat(paste("\nProcessing:", modality_name, "\n"))
  
  # 定义文件路径
  file_path_pdf <- file.path(out_dir, paste0("circos_", modality_name, ".pdf"))
  file_path_png <- file.path(out_dir, paste0("circos_", modality_name, ".png"))
  
  # 生成PDF版本
  pdf(file_path_pdf, width = 10, height = 10)
  pdf_result <- tryCatch({
    result <- plot_modality_circos_debug(
      modality_name = modality_name,
      modality_data = avg_data[[modality_name]],
      top_k_per_outcome = 3,
      top_features = 10,
      top_outcomes = 149,
      min_abs_weight = 0.001
    )
    result
  }, error = function(e) {
    cat(paste("  ✗ PDF generation failed:", e$message, "\n"))
    FALSE
  })
  dev.off()
  
  # 如果PDF生成成功，生成PNG版本
  if (pdf_result == TRUE) {
    cat(paste0("  ✓ Generated PDF for: ", modality_name, "\n"))
    
    # 生成PNG版本
    png(file_path_png, width = 1200, height = 1200, res = 150)
    png_result <- tryCatch({
      result <- plot_modality_circos_debug(
        modality_name = modality_name,
        modality_data = avg_data[[modality_name]],
        top_k_per_outcome = 3,
        top_features = 10,
        top_outcomes = 149,
        min_abs_weight = 0.001
      )
      result
    }, error = function(e) {
      cat(paste("  ✗ PNG generation failed:", e$message, "\n"))
      FALSE
    })
    dev.off()
    
    if (png_result == TRUE) {
      cat(paste0("  ✓ Generated PNG for: ", modality_name, "\n"))
    } else {
      cat(paste0("  ✗ Failed to generate PNG for: ", modality_name, "\n"))
    }
  } else {
    cat(paste0("  ✗ Failed to generate PDF for: ", modality_name, "\n"))
    
    # 尝试使用简化版本
    cat("  Trying simplified version...\n")
    
    # 生成简化版PDF
    pdf(file.path(out_dir, paste0("circos_simple_", modality_name, ".pdf")), 
        width = 10, height = 10)
    plot_simple_alternative(modality_name, avg_data[[modality_name]])
    dev.off()
    
    # 生成简化版PNG
    png(file.path(out_dir, paste0("circos_simple_", modality_name, ".png")),
        width = 1200, height = 1200, res = 150)
    plot_simple_alternative(modality_name, avg_data[[modality_name]])
    dev.off()
    
    cat(paste0("  ✓ Generated simplified versions for: ", modality_name, "\n"))
  }
}

# 简单的替代图函数
plot_simple_alternative <- function(modality_name, modality_data) {
  # 提取简化数据
  df_list <- list()
  for (outcome in names(modality_data)) {
    weights <- modality_data[[outcome]]
    if (length(weights) == 0) next
    
    df_temp <- data.frame(
      feature = names(weights)[1:min(5, length(weights))],
      outcome = outcome,
      value = abs(as.numeric(weights[1:min(5, length(weights))]))
    )
    df_list[[outcome]] <- df_temp
  }
  
  df <- bind_rows(df_list)
  
  if (is.null(df) || nrow(df) == 0) {
    # 显示错误信息
    par(mar = c(5, 5, 4, 2))
    plot(1, 1, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
         axes = FALSE, xlab = "", ylab = "")
    text(0.5, 0.5, paste("No valid data for:", modality_name), cex = 1.5, col = "red")
    return()
  }
  
  df <- df[complete.cases(df), ]
  
  # 绘制简化版Circos
  circos.clear()
  par(mar = c(1, 1, 3, 1))
  
  chordDiagram(df[, c("feature", "outcome", "value")],
               grid.col = "grey",
               transparency = 0.5,
               annotationTrack = "grid")
  
  # 添加标签
  circos.trackPlotRegion(
    track.index = 1,
    panel.fun = function(x, y) {
      sector.index = CELL_META$sector.index
      xlim = CELL_META$xlim
      ylim = CELL_META$ylim
      
      label <- sector.index
      if (nchar(label) > 15) {
        label <- paste0(substr(label, 1, 13), "..")
      }
      
      circos.text(
        mean(xlim), 
        ylim[1] + 0.05, 
        label, 
        facing = "clockwise",
        niceFacing = TRUE,
        adj = c(0, 0.5),
        cex = 0.7
      )
    },
    bg.border = NA
  )
  
  title(paste("Simple Circos:", modality_name),
        sub = paste(nrow(df), "connections"),
        cex.main = 1.2, cex.sub = 0.8)
  
  circos.clear()
}

cat("\n=== Generation complete ===\n")
cat(paste("PDF and PNG files saved to:", out_dir, "\n"))