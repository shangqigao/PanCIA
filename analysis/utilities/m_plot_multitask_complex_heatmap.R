# ==============================
# Revised Hierarchical Heatmap for New Path Structure
# ==============================
install.packages(c(
  "ggplot2", "dplyr", "purrr", "stringr", "jsonlite",
  "tidyr", "scales", "patchwork", "circlize", "pheatmap",
  "RColorBrewer", "viridisLite", "pheatmap", "viridis"
))

# 加载所有库
library(ggplot2)
library(dplyr)
library(purrr)
library(stringr)
library(jsonlite)
library(tidyr)
library(scales)
library(patchwork)
library(circlize)
library(RColorBrewer)
library(viridisLite)
library(pheatmap)
library(viridis)

# 设置目录
base_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes"
out_dir  <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/multitask"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# 定义任务及其指标
big_tasks <- list(
  survival = list(
    metric = "C-index",
    subtask_types = c("OS", "DSS", "DFI", "PFI")
  ),
  phenotype = list(
    metric = "f1",
    subtask_types = c("ImmuneSubtype", "MolecularSubtype", "PrimaryDisease")
  ),
  signature = list(
    metric = "r2",
    subtask_types = c("GeneProgrames", "HRDscore", "ImmuneSignatureScore", 
                     "StemnessScoreDNA", "StemScoreRNA")
  )
)

# 定义顺序和级别
agg_levels <- c("MEAN", "ABMIL", "SPARRA")
omics_levels <- c("radiomics", "pathomics", "radiopathomics")
radiomics_order <- c("pyradiomics", "FMCIB", "BiomedParse", "LVMMed")
pathomics_order <- c("CHIEF", "CONCH", "UNI")

# 定义颜色
agg_colors <- c(
  MEAN = "#e41a1c",
  ABMIL = "#377eb8",
  SPARRA = "#4daf4a"
)

omics_colors <- c(
  radiomics = "#1f78b4",
  pathomics = "#33a02c",
  radiopathomics = "#6a3d9a"
)

big_task_colors <- c(
  survival = "#1b9e77",
  phenotype = "#d95f02",
  signature = "#7570b3"
)

# 生成 radiopathomics 组合
radiopathomics_order <- c()
for (r in radiomics_order) {
  for (p in pathomics_order) {
    radiopathomics_order <- c(radiopathomics_order, paste(r, p, sep = "+"))
  }
}

# 按组学类型排序模型
model_order_list <- list(
  radiomics = radiomics_order,
  pathomics = pathomics_order,
  radiopathomics = radiopathomics_order
)

# ==============================
# 更新后的 JSON 解析函数（针对新路径结构）
# ==============================
parse_json_with_path_structure <- function(file_path) {
  # 从文件路径提取信息
  path_parts <- str_split(file_path, "/", simplify = TRUE)
  
  # 找到 TCGA_ 开始的索引
  tcga_idx <- which(str_detect(path_parts, "^TCGA_"))[1]
  
  if (is.na(tcga_idx) || tcga_idx > length(path_parts) - 2) {
    warning("Could not parse path structure:", file_path)
    return(NULL)
  }
  
  # 提取大任务和子任务
  task_dir <- path_parts[tcga_idx]
  task_parts <- str_split(task_dir, "_", simplify = TRUE)
  
  if (length(task_parts) < 3) {
    warning("Unexpected task directory format:", task_dir)
    return(NULL)
  }
  
  big_task <- task_parts[2]
  subtask <- task_parts[3]
  
  # 提取模型组合
  model_dir <- path_parts[tcga_idx + 1]
  model_parts <- str_split(model_dir, "\\+", simplify = TRUE)
  
  if (length(model_parts) < 2) {
    warning("Unexpected model directory format:", model_dir)
    return(NULL)
  }
  
  radio_model <- model_parts[1]
  patho_model <- model_parts[2]
  
  # 从文件名提取配置
  file_name <- basename(file_path)
  filename_parts <- str_split(file_name, "_", simplify = TRUE)
  
  if (length(filename_parts) < 5) {
    warning("Unexpected filename format:", file_name)
    return(NULL)
  }
  
  omics_type <- filename_parts[1]
  
  # 解析聚合方法
  radio_part <- filename_parts[2]
  patho_part <- filename_parts[3]
  
  radio_agg <- str_split(radio_part, "\\+", simplify = TRUE)[, 2]
  patho_agg <- str_split(patho_part, "\\+", simplify = TRUE)[, 2]
  
  # pyradiomics 特殊处理
  if (radio_model %in% c("pyradiomics", "FMCIB")) {
  radio_agg <- "MEAN"
}
  
  # 根据组学类型确定聚合方法
  agg_mode <- case_when(
    omics_type == "radiomics" ~ radio_agg,
    omics_type == "pathomics" ~ patho_agg,
    omics_type == "radiopathomics" ~ radio_agg,
    TRUE ~ NA_character_
  )
  
  # 根据组学类型确定模型名称
  model <- case_when(
    omics_type == "radiomics" ~ radio_model,
    omics_type == "pathomics" ~ patho_model,
    omics_type == "radiopathomics" ~ paste(radio_model, patho_model, sep = "+"),
    TRUE ~ NA_character_
  )
  
  # 解析其他信息
  model_type <- ifelse(length(filename_parts) > 3, filename_parts[4], NA)
  scorer <- ifelse(length(filename_parts) > 4, filename_parts[5], NA)
  
  scorer_type <- ifelse(!is.na(scorer), 
                       str_split(scorer, "\\+", simplify = TRUE)[, 2], 
                       big_tasks[[big_task]]$metric)
  
  # 加载 JSON 数据
  json_data <- fromJSON(file_path)
  
  # 获取该任务的指标
  target_metric <- big_tasks[[big_task]]$metric
  
  # 初始化结果列表
  results <- list()
  
  # 处理每个 fold
  for (fold_name in names(json_data)) {
    if (!str_detect(fold_name, "^Fold")) {
        next
    }

    fold_data <- json_data[[fold_name]]
    
    metric_values <- pmax(fold_data[[target_metric]], -1)
    
    if (is.null(metric_values)) {
      metric_values <- fold_data[[scorer_type]]
    }
    
    if (is.null(metric_values)) {
      next
    }
    
    # 处理不同的指标结构
    if (is.numeric(metric_values) && length(metric_values) > 0) {
      for (i in seq_along(metric_values)) {
        results[[length(results) + 1]] <- tibble(
          big_task = big_task,
          subtask = subtask,
          omics_type = omics_type,
          agg_mode = agg_mode,
          model = model,
          radio_model = radio_model,
          patho_model = patho_model,
          model_type = model_type,
          scorer = scorer_type,
          metric_name = target_metric,
          class_index = ifelse(length(metric_values) > 1, i, 1),
          class_label = ifelse(length(metric_values) > 1, 
                              paste("Class", i), 
                              "Overall"),
          fold = as.numeric(str_extract(fold_name, "\\d+")),
          value = metric_values[i],
          file_path = file_path
        )
      }
    } else if (is.list(metric_values) && length(metric_values) > 0) {
      for (i in seq_along(metric_values)) {
        if (is.numeric(metric_values[[i]])) {
          class_value <- metric_values[[i]]
          results[[length(results) + 1]] <- tibble(
            big_task = big_task,
            subtask = subtask,
            omics_type = omics_type,
            agg_mode = agg_mode,
            model = model,
            radio_model = radio_model,
            patho_model = patho_model,
            model_type = model_type,
            scorer = scorer_type,
            metric_name = target_metric,
            class_index = i,
            class_label = paste("Class", i),
            fold = as.numeric(str_extract(fold_name, "\\d+")),
            value = class_value,
            file_path = file_path
          )
        }
      }
    }
  }
  
  if (length(results) == 0) {
    return(NULL)
  }
  
  # 转换为数据框
  df <- bind_rows(results)
  
  return(df)
}

# ==============================
# 从 JSON 文件加载所有数据
# ==============================
load_all_data <- function() {
  cat("Loading data from JSON files...\n")
  
  all_data <- list()
  
  # 递归查找所有 JSON 文件
  json_files <- list.files(
    path = base_dir,
    pattern = "_metrics\\.json$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  cat("Found", length(json_files), "JSON files\n")
  
  # 处理每个文件
  for (i in seq_along(json_files)) {
    if (i %% 100 == 0) {
      cat("  Processed", i, "files\n")
    }
    
    file_path <- json_files[i]

    # if (str_detect(file_path, "pyradiomics")) {
    #   next
    # }

    result <- tryCatch({
      parse_json_with_path_structure(file_path)
    }, error = function(e) {
      cat("  Error parsing", basename(file_path), ":", e$message, "\n")
      NULL
    })
    
    if (!is.null(result) && nrow(result) > 0) {
      all_data[[length(all_data) + 1]] <- result
    }
  }
  
  # 合并所有数据
  if (length(all_data) == 0) {
    stop("No data was successfully parsed. Check file paths and formats.")
  }
  
  df_all <- bind_rows(all_data)
  
  # 打印摘要
  cat("\nData summary:\n")
  cat("Total data points:", nrow(df_all), "\n")
  cat("Big tasks:", paste(unique(df_all$big_task), collapse = ", "), "\n")
  cat("Subtasks:", paste(unique(df_all$subtask), collapse = ", "), "\n")
  cat("Omics types:", paste(unique(df_all$omics_type), collapse = ", "), "\n")
  cat("Models:", length(unique(df_all$model)), "\n")
  cat("Aggregation methods:", paste(unique(df_all$agg_mode), collapse = ", "), "\n")
  cat("Max classes:", max(df_all$class_index, na.rm = TRUE), "\n")
  
  return(df_all)
}

# ==============================
# 准备热图数据
# ==============================
normalize_rows_minmax <- function(mat) {
  t(apply(mat, 1, function(x) {
    rng <- range(x, na.rm = TRUE)
    if (diff(rng) == 0) return(rep(0.5, length(x)))  # constant row safeguard
    (x - rng[1]) / diff(rng)
  }))
}

normalize_rows_top3 <- function(mat) {
  t(apply(mat, 1, function(x) {
    # Initialize all values to 0
    res <- rep(0, length(x))
    
    # Find indices of top 3 values
    top_idx <- order(x, decreasing = TRUE)[1:min(3, length(x))]
    
    # Assign scores
    scores <- c(1, 0.7, 0.4)
    res[top_idx] <- scores[1:length(top_idx)]
    
    res
  }))
}


prepare_heatmap_data <- function(df) {
  cat("\nPreparing heatmap data...\n")
  
  # 跨 folds 汇总
  df_summary <- df %>%
    group_by(big_task, subtask, omics_type, agg_mode, model, metric_name, class_index, class_label) %>%
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sd_value = sd(value, na.rm = TRUE),
      n_folds = n(),
      .groups = "drop"
    ) %>%
    filter(!is.na(mean_value))
  
  # 创建唯一标识符
  df_summary <- df_summary %>%
    mutate(
      row_group = paste(big_task, subtask, sep = " | "),
      row_subgroup = class_label,
      row_id = paste(row_group, row_subgroup, sep = " | "),
      
      col_group = paste(omics_type, agg_mode, sep = " | "),
      col_id = paste(col_group, model, sep = " | "),
      
      big_task = factor(big_task, levels = names(big_tasks)),
      subtask = factor(subtask),
      omics_type = factor(omics_type, levels = omics_levels),
      agg_mode = factor(agg_mode, levels = agg_levels),
      model = factor(model, levels = unlist(model_order_list)),
      
      sort_key = paste(
        sprintf("%02d", as.numeric(big_task)),
        sprintf("%02d", as.numeric(subtask)),
        sprintf("%03d", class_index),
        sep = "_"
      )
    ) %>%
    arrange(sort_key, omics_type, agg_mode, model)
  
  # 创建热图矩阵（均值）
  cat("Creating heatmap matrix...\n")
  hm_matrix <- df_summary %>%
    select(row_id, col_id, mean_value) %>%
    pivot_wider(
      names_from = col_id,
      values_from = mean_value,
      values_fill = NA
    ) %>%
    as.data.frame()
  
  rownames(hm_matrix) <- hm_matrix$row_id
  hm_matrix <- as.matrix(hm_matrix[, -1])

  # hm_matrix <- normalize_rows_minmax(hm_matrix)

  # 更简洁的版本
  hm_matrix_scored <- t(apply(hm_matrix, 1, function(x) {
    result <- rep(0, length(x))
    non_na_idx <- which(!is.na(x))
    non_na_count <- length(non_na_idx)
    
    if(non_na_count > 0) {
      # 获取排序后的索引
      sorted_idx <- non_na_idx[order(x[non_na_idx], decreasing = TRUE)]
      
      # 赋值向量
      values <- c(1, 0.8, 0.6, 0.4, 0.2)
      
      # 取前min(5, non_na_count)个进行赋值
      n <- min(5, non_na_count)
      for(i in 1:n) {
        result[sorted_idx[i]] <- values[i]
      }
    }
    
    return(result)
  }))

  colnames(hm_matrix_scored) <- colnames(hm_matrix)
  rownames(hm_matrix_scored) <- rownames(hm_matrix)

  # 保持列名
  colnames(hm_matrix_scored) <- colnames(hm_matrix)
  rownames(hm_matrix_scored) <- rownames(hm_matrix)
  
  # 创建 SD 矩阵
  sd_matrix <- df_summary %>%
    select(row_id, col_id, sd_value) %>%
    pivot_wider(
      names_from = col_id,
      values_from = sd_value,
      values_fill = NA
    ) %>%
    as.data.frame()
  
  rownames(sd_matrix) <- sd_matrix$row_id
  sd_matrix <- as.matrix(sd_matrix[, -1])
  
  # 确保矩阵维度一致
  sd_matrix <- sd_matrix[rownames(hm_matrix), colnames(hm_matrix)]
  
  # 创建行注释数据
  row_anno <- df_summary %>%
    distinct(row_id, big_task, subtask, class_label, class_index) %>%
    arrange(match(row_id, rownames(hm_matrix))) %>%
    mutate(
      row_id = factor(row_id, levels = rownames(hm_matrix)),
      big_task = factor(big_task, levels = names(big_tasks)),
      subtask = factor(subtask),
      class_label = factor(class_label)
    )
  
  # 创建列注释数据
  col_anno <- df_summary %>%
    distinct(col_id, omics_type, agg_mode, model) %>%
    arrange(match(col_id, colnames(hm_matrix))) %>%
    mutate(
      col_id = factor(col_id, levels = colnames(hm_matrix)),
      omics_type = factor(omics_type, levels = omics_levels),
      agg_mode = factor(agg_mode, levels = agg_levels),
      model = factor(model, levels = unlist(model_order_list))
    )
  
  return(list(
    matrix = hm_matrix_scored,
    sd_matrix = sd_matrix,
    row_anno = row_anno,
    col_anno = col_anno,
    raw_data = df_summary,
    full_data = df
  ))
}

# ==============================
# 1. Clean Heatmap Function
# ==============================

create_clean_heatmap <- function(heatmap_data, plot_title = "Multi-Omics Model Performance", 
                                output_file = "clean_heatmap.png") {
  
  hm_matrix <- heatmap_data$matrix
  row_anno <- heatmap_data$row_anno
  col_anno <- heatmap_data$col_anno
  
  cat("Creating clean heatmap...\n")
  cat("Dimensions:", dim(hm_matrix), "\n")
  
  # 简化行名（移除冗余信息）
  simplified_row_names <- sapply(strsplit(rownames(hm_matrix), " \\| "), function(x) {
    if (length(x) >= 3) {
      # 格式: big_task | subtask | class_label
      # 只显示 subtask 和 class_label
      paste(x[2], x[3], sep = " | ")
    } else {
      rownames(hm_matrix)
    }
  })
  
  # 简化列名
  simplified_col_names <- sapply(strsplit(colnames(hm_matrix), " \\| "), function(x) {
    if (length(x) >= 3) {
      # 格式: omics_type | agg_mode | model
      # 只显示 model 和 agg_mode
      paste(x[3], "(", x[2], ")", sep = "")
    } else {
      colnames(hm_matrix)
    }
  })
  
  # 更新矩阵的行列名
  rownames(hm_matrix) <- simplified_row_names
  colnames(hm_matrix) <- simplified_col_names
  
  # 准备行注释数据
  row_annotation <- data.frame(
    Task = row_anno$big_task,
    Subtask = row_anno$subtask,
    row.names = simplified_row_names
  )
  
  # 准备列注释数据
  col_annotation <- data.frame(
    Omics = col_anno$omics_type,
    Aggregation = col_anno$agg_mode,
    row.names = simplified_col_names
  )
  
  # 创建注释颜色
  annotation_colors <- list(
    Task = big_task_colors,
    Omics = omics_colors,
    Aggregation = agg_colors,
    Subtask = colorRampPalette(brewer.pal(8, "Set2"))(length(unique(row_anno$subtask)))
  )
  names(annotation_colors$Subtask) <- unique(row_anno$subtask)
  
  n_rows <- nrow(hm_matrix)
  n_cols <- ncol(hm_matrix)

  cellheight <- 7
  cellwidth  <- 14

  plot_width  <- n_cols * cellwidth / 72 + 6
  plot_height <- n_rows * cellheight / 72 + 6

  plot_width  <- max(5, plot_width)
  plot_height <- max(5, plot_height)

  # 创建热图
  pheatmap(
    hm_matrix,
    
    # 注释
    annotation_row = row_annotation,
    annotation_col = col_annotation,
    annotation_colors = annotation_colors,
    
    # 显示设置
    show_rownames = TRUE,
    show_colnames = TRUE,
    fontsize_row = 6,
    fontsize_col = 9,
    fontsize = 10,
    
    # 聚类
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    
    # 颜色
    color = viridis(100),
    
    # 边框
    border_color = NA,
    cellwidth = cellwidth,
    cellheight = cellheight,
    
    # 图例
    legend = TRUE,
    annotation_legend = TRUE,
    
    # 标题
    main = plot_title,
    
    # 文件输出
    filename = file.path(out_dir, output_file),
    width = plot_width,
    height = plot_height
  )
  
  cat("Saved:", output_file, "\n")
}

# ==============================
# 2. 按任务分开展示的热图
# ==============================

create_faceted_heatmaps <- function(heatmap_data) {
  
  for (task in names(big_tasks)) {
    # 筛选该任务的数据
    task_rows <- which(heatmap_data$row_anno$big_task == task)
    
    if (length(task_rows) > 0) {
      # 提取子矩阵
      task_matrix <- heatmap_data$matrix[task_rows, , drop = FALSE]
      task_row_anno <- heatmap_data$row_anno[task_rows, , drop = FALSE]
      
      # 简化行名（只显示 subtask 和 class_label）
      simplified_row_names <- sapply(strsplit(rownames(task_matrix), " \\| "), function(x) {
        if (length(x) >= 3) {
          paste(x[2], x[3], sep = " | ")
        } else {
          rownames(task_matrix)
        }
      })
      rownames(task_matrix) <- simplified_row_names
      
      # 简化列名
      simplified_col_names <- sapply(strsplit(colnames(task_matrix), " \\| "), function(x) {
        if (length(x) >= 3) {
          paste(x[3], "(", x[2], ")", sep = "")
        } else {
          colnames(task_matrix)
        }
      })
      colnames(task_matrix) <- simplified_col_names
      
      # 准备注释
      row_annotation <- data.frame(
        Subtask = task_row_anno$subtask,
        Class = task_row_anno$class_label,
        row.names = simplified_row_names
      )
      
      col_annotation <- data.frame(
        Omics = heatmap_data$col_anno$omics_type,
        Aggregation = heatmap_data$col_anno$agg_mode,
        row.names = simplified_col_names
      )
      
      # 创建颜色方案
      annotation_colors <- list(
        Omics = omics_colors,
        Aggregation = agg_colors,
        Subtask = colorRampPalette(brewer.pal(8, "Set2"))(length(unique(task_row_anno$subtask))),
        Class = colorRampPalette(brewer.pal(12, "Set3"))(length(unique(task_row_anno$class_label)))
      )
      names(annotation_colors$Subtask) <- unique(task_row_anno$subtask)
      names(annotation_colors$Class) <- unique(task_row_anno$class_label)
      
      # 根据任务类型选择颜色方案
      if (task == "survival") {
        color_palette <- viridis(100, option = "inferno")
      } else if (task == "phenotype") {
        color_palette <- viridis(100, option = "magma")
      } else {
        color_palette <- viridis(100, option = "plasma")
      }
      
      n_rows <- nrow(task_matrix)
      n_cols <- ncol(task_matrix)

      cellheight <- 10
      cellwidth  <- 18

      plot_width  <- n_cols * cellwidth / 72 + 6
      plot_height <- n_rows * cellheight / 72 + 6

      plot_width  <- max(5, plot_width)
      plot_height <- max(5, plot_height)
      
      # 创建热图
      pheatmap(
        task_matrix,
        main = paste(str_to_title(task), "Analysis"),
        
        annotation_row = row_annotation,
        annotation_col = col_annotation,
        annotation_colors = annotation_colors,
        
        show_rownames = TRUE,
        show_colnames = TRUE,
        fontsize_row = 9,
        fontsize_col = 10,
        
        cluster_rows = FALSE,
        cluster_cols = FALSE,
        
        color = color_palette,
        breaks = seq(min(task_matrix, na.rm = TRUE), max(task_matrix, na.rm = TRUE), length = 100),
        
        border_color = "white",
        cellwidth = cellwidth,
        cellheight = cellheight,
        
        filename = file.path(out_dir, paste0("faceted_heatmap_", task, ".png")),
        width = plot_width,
        height = plot_height
      )
      
      cat("Saved: faceted_heatmap_", task, ".png\n", sep = "")
    }
  }
}

# ==============================
# 3. 汇总热图（按组平均）
# ==============================

create_summary_heatmap <- function(heatmap_data) {
  
  hm_matrix <- heatmap_data$matrix
  row_anno <- heatmap_data$row_anno
  col_anno <- heatmap_data$col_anno
  
  # 按 Task 和 Subtask 分组平均
  summary_by_task <- heatmap_data$raw_data %>%
    group_by(big_task, omics_type, agg_mode) %>%
    summarise(
      mean_performance = mean(mean_value, na.rm = TRUE),
      sd_performance = sd(mean_value, na.rm = TRUE),
      n_models = n(),
      .groups = "drop"
    )
  
  # 创建矩阵
  summary_matrix <- summary_by_task %>%
    mutate(
      row_id = paste(big_task, sep = " | "),
      col_id = paste(omics_type, agg_mode, sep = " | ")
    ) %>%
    select(row_id, col_id, mean_performance) %>%
    pivot_wider(names_from = col_id, values_from = mean_performance) %>%
    as.data.frame()
  
  rownames(summary_matrix) <- summary_matrix$row_id
  summary_matrix <- as.matrix(summary_matrix[, -1])
  
  # 准备注释
  row_annotation <- data.frame(
    Task = str_extract(rownames(summary_matrix), "^[^|]+"),
    row.names = rownames(summary_matrix)
  )
  
  col_annotation <- data.frame(
    Omics = str_extract(colnames(summary_matrix), "^[^|]+"),
    Aggregation = str_extract(colnames(summary_matrix), "[^|]+$"),
    row.names = colnames(summary_matrix)
  )

  n_rows <- nrow(summary_matrix)
  n_cols <- ncol(summary_matrix)

  cellheight <- 25
  cellwidth  <- 25

  plot_width  <- n_cols * cellwidth / 72 + 6
  plot_height <- n_rows * cellheight / 72 + 6

  plot_width  <- max(5, plot_width)
  plot_height <- max(5, plot_height)
  
  # 创建热图
  pheatmap(
    summary_matrix,
    main = "Summary: Average Performance by Configuration",
    
    annotation_row = row_annotation,
    annotation_col = col_annotation,
    annotation_colors = list(
      Task = big_task_colors,
      Omics = omics_colors,
      Aggregation = agg_colors
    ),
    
    show_rownames = TRUE,
    show_colnames = TRUE,
    fontsize = 11,
    
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    
    color = viridis(100),
    border_color = "white",
    cellwidth = cellwidth,
    cellheight = cellheight,
    
    display_numbers = TRUE,
    number_format = "%.3f",
    number_color = "black",
    fontsize_number = 9,
    
    filename = file.path(out_dir, "summary_heatmap.png"),
    width = plot_width,
    height = plot_height
  )
  
  cat("Saved: summary_heatmap.png\n")
}

# ==============================
# 4. 紧凑型热图（优化显示）
# ==============================

create_compact_heatmap <- function(heatmap_data) {
  
  hm_matrix <- heatmap_data$matrix
  
  # 如果矩阵太大，进行采样或聚合
  if (nrow(hm_matrix) > 50 || ncol(hm_matrix) > 50) {
    cat("Matrix is large, creating compact version...\n")
    
    # 按任务聚合行
    row_means <- heatmap_data$raw_data %>%
      group_by(big_task, omics_type, agg_mode) %>%
      summarise(
        mean_value = mean(mean_value, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        row_id = big_task,
        col_id = paste(omics_type, agg_mode, sep = " | ")
      ) %>%
      select(row_id, col_id, mean_value) %>%
      pivot_wider(names_from = col_id, values_from = mean_value) %>%
      as.data.frame()
    
    rownames(row_means) <- row_means$row_id
    hm_matrix_compact <- as.matrix(row_means[, -1])
    
  } else {
    hm_matrix_compact <- hm_matrix
  }

  n_rows <- nrow(hm_matrix_compact)
  n_cols <- ncol(hm_matrix_compact)

  cellheight <- 20
  cellwidth  <- 20

  plot_width  <- n_cols * cellwidth / 72 + 6
  plot_height <- n_rows * cellheight / 72 + 6

  plot_width  <- max(5, plot_width)
  plot_height <- max(5, plot_height)
  
  # 创建热图
  pheatmap(
    hm_matrix_compact,
    main = "Compact Performance Overview",
    
    show_rownames = TRUE,
    show_colnames = TRUE,
    fontsize_row = 10,
    fontsize_col = 10,
    
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    
    color = colorRampPalette(brewer.pal(9, "YlOrRd"))(100),
    
    border_color = "white",
    cellwidth = cellwidth,
    cellheight = cellheight,
    
    display_numbers = ifelse(nrow(hm_matrix_compact) <= 20, TRUE, FALSE),
    number_format = "%.2f",
    
    filename = file.path(out_dir, "compact_heatmap.png"),
    width = plot_width,
    height = plot_height
  )
  
  cat("Saved: compact_heatmap.png\n")
}

# ==============================
# 修复的汇总热图函数
# ==============================
create_summary_heatmap <- function(heatmap_data) {
  
  hm_matrix <- heatmap_data$matrix
  row_anno <- heatmap_data$row_anno
  col_anno <- heatmap_data$col_anno
  
  cat("Creating summary heatmap...\n")
  
  # 按 Task 和 Omics/Aggregation 分组平均
  summary_by_task <- heatmap_data$raw_data %>%
    group_by(big_task, omics_type, agg_mode) %>%
    summarise(
      mean_performance = mean(mean_value, na.rm = TRUE),
      sd_performance = sd(mean_value, na.rm = TRUE),
      n_models = n(),
      .groups = "drop"
    )
  
  # 确保有数据
  if (nrow(summary_by_task) == 0) {
    cat("No data for summary heatmap.\n")
    return(NULL)
  }
  
  # 创建矩阵
  summary_matrix <- summary_by_task %>%
    mutate(
      row_id = big_task,
      col_id = paste(omics_type, agg_mode, sep = " | ")
    ) %>%
    select(row_id, col_id, mean_performance) %>%
    pivot_wider(
      names_from = col_id, 
      values_from = mean_performance,
      values_fill = NA
    ) %>%
    as.data.frame()
  
  rownames(summary_matrix) <- summary_matrix$row_id
  summary_matrix <- as.matrix(summary_matrix[, -1])
  
  cat("Summary matrix dimensions:", dim(summary_matrix), "\n")
  
  # 准备行注释
  row_annotation <- data.frame(
    Task = rownames(summary_matrix),
    row.names = rownames(summary_matrix)
  )
  
  # 准备列注释
  col_names <- colnames(summary_matrix)
  col_omics <- sapply(strsplit(col_names, " \\| "), function(x) x[1])
  col_agg <- sapply(strsplit(col_names, " \\| "), function(x) x[2])
  
  col_annotation <- data.frame(
    Omics = col_omics,
    Aggregation = col_agg,
    row.names = col_names
  )
  
  # 创建注释颜色
  annotation_colors <- list()
  
  # Task 颜色
  if (all(unique(row_annotation$Task) %in% names(big_task_colors))) {
    annotation_colors$Task <- big_task_colors[names(big_task_colors) %in% unique(row_annotation$Task)]
  } else {
    task_colors <- colorRampPalette(brewer.pal(8, "Set2"))(length(unique(row_annotation$Task)))
    names(task_colors) <- unique(row_annotation$Task)
    annotation_colors$Task <- task_colors
  }
  
  # Omics 颜色
  existing_omics <- intersect(unique(col_annotation$Omics), names(omics_colors))
  if (length(existing_omics) > 0) {
    annotation_colors$Omics <- omics_colors[existing_omics]
  } else {
    omics_colors_custom <- colorRampPalette(brewer.pal(8, "Set1"))(length(unique(col_annotation$Omics)))
    names(omics_colors_custom) <- unique(col_annotation$Omics)
    annotation_colors$Omics <- omics_colors_custom
  }
  
  # Aggregation 颜色
  existing_agg <- intersect(unique(col_annotation$Aggregation), names(agg_colors))
  if (length(existing_agg) > 0) {
    annotation_colors$Aggregation <- agg_colors[existing_agg]
  } else {
    agg_colors_custom <- colorRampPalette(brewer.pal(8, "Set3"))(length(unique(col_annotation$Aggregation)))
    names(agg_colors_custom) <- unique(col_annotation$Aggregation)
    annotation_colors$Aggregation <- agg_colors_custom
  }
  
  # 检查矩阵是否有有效数据
  if (all(is.na(summary_matrix))) {
    cat("Warning: Summary matrix contains only NA values\n")
    return(NULL)
  }

  n_rows <- nrow(summary_matrix)
  n_cols <- ncol(summary_matrix)

  cellheight <- 25
  cellwidth  <- 25

  plot_width  <- n_cols * cellwidth / 72 + 6
  plot_height <- n_rows * cellheight / 72 + 6

  plot_width  <- max(5, plot_width)
  plot_height <- max(5, plot_height)
  
  # 创建热图
  tryCatch({
    pheatmap(
      summary_matrix,
      main = "Summary: Average Performance by Configuration",
      
      annotation_row = row_annotation,
      annotation_col = col_annotation,
      annotation_colors = annotation_colors,
      
      show_rownames = TRUE,
      show_colnames = TRUE,
      fontsize = 11,
      
      cluster_rows = FALSE,
      cluster_cols = FALSE,
      
      color = viridis(100),
      na_col = "gray90",
      
      border_color = "white",
      cellwidth = cellwidth,
      cellheight = cellheight,
      
      display_numbers = TRUE,
      number_format = "%.3f",
      number_color = "black",
      fontsize_number = 9,
      
      filename = file.path(out_dir, "summary_heatmap.png"),
      width = plot_width,
      height = plot_height,
      silent = FALSE
    )
    
    cat("Saved: summary_heatmap_fixed.png\n")
    
  }, error = function(e) {
    cat("Error creating summary heatmap:", e$message, "\n")
    
    # 尝试简化版本
    tryCatch({
      pheatmap(
        summary_matrix,
        main = "Summary Performance",
        show_rownames = TRUE,
        show_colnames = TRUE,
        cluster_rows = FALSE,
        cluster_cols = FALSE,
        color = viridis(100),
        display_numbers = TRUE,
        number_format = "%.3f",
        filename = file.path(out_dir, "summary_heatmap_simple.png"),
        width = 12,
        height = 8
      )
      cat("Saved simplified version: summary_heatmap_simple.png\n")
    }, error = function(e2) {
      cat("Simplified version also failed:", e2$message, "\n")
    })
  })
}

# ==============================
# 增强的汇总热图（显示均值和标准差）
# ==============================
create_enhanced_summary_heatmap <- function(heatmap_data) {
  
  cat("Creating enhanced summary heatmap...\n")
  
  # 计算均值和标准差
  summary_stats <- heatmap_data$raw_data %>%
    group_by(big_task, omics_type, agg_mode) %>%
    summarise(
      mean_value = mean(mean_value, na.rm = TRUE),
      sd_value = sd(mean_value, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(
      display_text = sprintf("%.3f\n(±%.3f)", mean_value, sd_value)
    )
  
  # 创建均值矩阵
  mean_matrix <- summary_stats %>%
    mutate(
      row_id = big_task,
      col_id = paste(omics_type, agg_mode, sep = " | ")
    ) %>%
    select(row_id, col_id, mean_value) %>%
    pivot_wider(
      names_from = col_id,
      values_from = mean_value
    ) %>%
    as.data.frame()
  
  rownames(mean_matrix) <- mean_matrix$row_id
  mean_matrix <- as.matrix(mean_matrix[, -1])
  
  # 创建显示文本矩阵
  text_matrix <- summary_stats %>%
    mutate(
      row_id = big_task,
      col_id = paste(omics_type, agg_mode, sep = " | ")
    ) %>%
    select(row_id, col_id, display_text) %>%
    pivot_wider(
      names_from = col_id,
      values_from = display_text
    ) %>%
    as.data.frame()
  
  rownames(text_matrix) <- text_matrix$row_id
  text_matrix <- as.matrix(text_matrix[, -1])
  
  # 准备注释
  col_names <- colnames(mean_matrix)
  col_omics <- sapply(strsplit(col_names, " \\| "), function(x) x[1])
  col_agg <- sapply(strsplit(col_names, " \\| "), function(x) x[2])
  
  col_annotation <- data.frame(
    Omics = col_omics,
    Aggregation = col_agg,
    row.names = col_names
  )
  
  # 创建注释颜色
  annotation_colors <- list(
    Task = big_task_colors[names(big_task_colors) %in% rownames(mean_matrix)],
    Omics = omics_colors[names(omics_colors) %in% unique(col_omics)],
    Aggregation = agg_colors[names(agg_colors) %in% unique(col_agg)]
  )
  
  n_rows <- nrow(mean_matrix)
  n_cols <- ncol(mean_matrix)

  cellheight <- 30
  cellwidth  <- 30

  plot_width  <- n_cols * cellwidth / 72 + 6
  plot_height <- n_rows * cellheight / 72 + 6

  plot_width  <- max(5, plot_width)
  plot_height <- max(5, plot_height)

  # 创建热图
  pheatmap(
    mean_matrix,
    main = "Enhanced Summary: Mean ± SD",
    
    annotation_row = data.frame(Task = rownames(mean_matrix), row.names = rownames(mean_matrix)),
    annotation_col = col_annotation,
    annotation_colors = annotation_colors,
    
    show_rownames = TRUE,
    show_colnames = TRUE,
    fontsize_row = 12,
    fontsize_col = 11,
    
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    
    color = colorRampPalette(brewer.pal(9, "YlOrRd"))(100),
    
    display_numbers = text_matrix,
    number_format = "%s",
    number_color = "black",
    fontsize_number = 8,
    
    border_color = "white",
    cellwidth = cellwidth,
    cellheight = 30,
    
    filename = file.path(out_dir, "enhanced_summary_heatmap.png"),
    width = plot_width,
    height = plot_height
  )
  
  cat("Saved: enhanced_summary_heatmap.png\n")
}

# ==============================
# 按任务类型分开的汇总热图
# ==============================
create_taskwise_summary_heatmaps <- function(heatmap_data) {
  
  for (task in names(big_tasks)) {
    cat("\nCreating summary for:", task, "\n")
    
    # 筛选该任务的数据
    task_data <- heatmap_data$raw_data %>%
      filter(big_task == task)
    
    if (nrow(task_data) == 0) {
      cat("  No data for", task, "\n")
      next
    }
    
    # 按 omics_type 和 agg_mode 汇总
    task_summary <- task_data %>%
      group_by(omics_type, agg_mode) %>%
      summarise(
        mean_value = mean(mean_value, na.rm = TRUE),
        sd_value = sd(mean_value, na.rm = TRUE),
        n = n(),
        .groups = "drop"
      ) %>%
      mutate(
        display_text = sprintf("%.3f\n(±%.3f)", mean_value, sd_value)
      )
    
    # 创建矩阵
    mean_matrix <- task_summary %>%
      mutate(col_id = paste(omics_type, agg_mode, sep = " | ")) %>%
      select(col_id, mean_value) %>%
      pivot_wider(
        names_from = col_id,
        values_from = mean_value,
        values_fill = NA
      )
    
    if (ncol(mean_matrix) == 0) {
      cat("  Cannot create matrix for", task, "\n")
      next
    }
    
    # 转换为矩阵
    mean_matrix <- as.matrix(mean_matrix)
    
    # 准备列注释
    col_names <- colnames(mean_matrix)
    col_omics <- sapply(strsplit(col_names, " \\| "), function(x) x[1])
    col_agg <- sapply(strsplit(col_names, " \\| "), function(x) x[2])
    
    col_annotation <- data.frame(
      Omics = col_omics,
      Aggregation = col_agg,
      row.names = col_names
    )
    
    n_rows <- nrow(mean_matrix)
    n_cols <- ncol(mean_matrix)

    cellheight <- 20
    cellwidth  <- 25

    plot_width  <- n_cols * cellwidth / 72 + 6
    plot_height <- n_rows * cellheight / 72 + 6

    plot_width  <- max(5, plot_width)
    plot_height <- max(5, plot_height)

    # 创建热图
    tryCatch({
      pheatmap(
        mean_matrix,
        main = paste(str_to_title(task), "Summary"),
        
        annotation_col = col_annotation,
        annotation_colors = list(
          Omics = omics_colors[names(omics_colors) %in% unique(col_omics)],
          Aggregation = agg_colors[names(agg_colors) %in% unique(col_agg)]
        ),
        
        show_rownames = FALSE,
        show_colnames = TRUE,
        fontsize_col = 11,
        
        cluster_rows = FALSE,
        cluster_cols = FALSE,
        
        color = if (task == "survival") {
          viridis(100, option = "inferno")
        } else if (task == "phenotype") {
          viridis(100, option = "magma")
        } else {
          viridis(100, option = "plasma")
        },
        
        display_numbers = TRUE,
        number_format = "%.3f",
        fontsize_number = 10,
        
        border_color = "white",
        cellwidth = cellwidth,
        cellheight = cellheight,
        
        filename = file.path(out_dir, paste0("task_summary_", task, ".png")),
        width = plot_width,
        height = plot_height
      )
      
      cat("  Saved: task_summary_", task, ".png\n", sep = "")
      
    }, error = function(e) {
      cat("  Error:", e$message, "\n")
    })
  }
}

# ==============================
# 5. 主执行函数
# ==============================

main_pheatmap <- function() {
  # 加载数据（使用之前的数据加载函数）
  df_all <- load_all_data()
  heatmap_data <- prepare_heatmap_data(df_all)
  
  cat("\n=== Creating Professional Heatmaps with pheatmap ===\n")
  
  # 1. 创建整洁热图
  cat("\n1. Creating clean heatmap...\n")
  tryCatch({
    create_clean_heatmap(heatmap_data, 
                        "Multi-Omics Model Performance",
                        "clean_heatmap.png")
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })
  
  # 2. 创建分面热图
  cat("\n2. Creating faceted heatmaps...\n")
  tryCatch({
    create_faceted_heatmaps(heatmap_data)
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })
  
  # 3. 创建汇总热图
  cat("\n3. Creating summary heatmap...\n")
  tryCatch({
    create_summary_heatmap(heatmap_data)
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })
  
  # 4. 创建紧凑热图
  cat("\n4. Creating compact heatmap...\n")
  tryCatch({
    create_compact_heatmap(heatmap_data)
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })

  # 5. 增强的汇总热图（显示均值和标准差）
  cat("\n5. Creating enhanced summary heatmap...\n")
  tryCatch({
    create_enhanced_summary_heatmap(heatmap_data)
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })

  # 6. 按任务的汇总热图
  cat("\n6. Creating task-wise summary heatmaps...\n")
  tryCatch({
    create_taskwise_summary_heatmaps(heatmap_data)
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })
  
  cat("\n=== All heatmaps created successfully! ===\n")
  cat("Files saved to:", out_dir, "\n")
}

# 运行
main_pheatmap()