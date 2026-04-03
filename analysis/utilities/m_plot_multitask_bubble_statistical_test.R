# ==============================
# Revised Hierarchical Heatmap for New Path Structure
# ==============================
install.packages(c(
  "ggplot2", "dplyr", "purrr", "stringr", "jsonlite",
  "tidyr", "scales", "patchwork", "circlize", "pheatmap",
  "RColorBrewer", "viridisLite", "pheatmap", "viridis", "scico"
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
library(scico)

# 设置目录
base_dir <- "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_slice+tumor"
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

# visualize topk or not
topk <- TRUE

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
# Helper: select best model per omics type for each outcome
# ==============================
select_best_models <- function(df) {
  # df: full_data from heatmap_data (fold‑level)
  df %>%
    group_by(big_task, subtask, class_index, class_label, omics_type, model, agg_mode) %>%
    summarise(
      mean_perf = mean(value, na.rm = TRUE),
      fold_values = list(value),          # store the 5 values for later testing
      .groups = "drop"
    ) %>%
    group_by(big_task, subtask, class_index, class_label, omics_type) %>%
    slice_max(mean_perf, n = 1, with_ties = FALSE) %>%   # best model per omics
    ungroup()
}

# ==============================
# Perform pairwise statistical tests
# ==============================
compare_best_models <- function(best_models_df, full_df) {
  # Join best models with full fold data
  selected <- best_models_df %>%
    select(big_task, subtask, class_index, class_label, omics_type, model, agg_mode) %>%
    left_join(full_df, by = c("big_task", "subtask", "class_index", "class_label",
                               "omics_type", "model", "agg_mode"))

  # Identify unique outcomes
  outcomes <- selected %>%
    distinct(big_task, subtask, class_index, class_label) %>%
    mutate(outcome_id = paste(big_task, subtask, class_label, sep = " | "))

  results <- list()

  for (i in seq_len(nrow(outcomes))) {
    out <- outcomes[i, ]
    out_data <- selected %>%
      filter(big_task == out$big_task,
             subtask == out$subtask,
             class_index == out$class_index)

    # Ensure we have all three omics types
    omics_present <- unique(out_data$omics_type)
    if (!all(c("radiomics", "pathomics", "radiopathomics") %in% omics_present)) {
      next
    }

    # Pivot to wide format: one column per omics type, rows = folds
    paired_wide <- out_data %>%
      select(fold, omics_type, value) %>%
      pivot_wider(names_from = omics_type, values_from = value) %>%
      drop_na()  # keep only folds present in all three

    # Should have exactly 5 rows if all folds are complete
    if (nrow(paired_wide) < 5) {
      warning(sprintf("Outcome %s has only %d complete folds", out$outcome_id, nrow(paired_wide)))
      next
    }

    # Extract vectors
    rad    <- paired_wide$radiomics
    path   <- paired_wide$pathomics
    radpath <- paired_wide$radiopathomics

    # Paired t‑tests (or wilcox.test)
    test_rad_path   <- t.test(rad, path, paired = TRUE)
    test_rad_radpath <- t.test(rad, radpath, paired = TRUE)
    test_path_radpath <- t.test(path, radpath, paired = TRUE)

    results[[length(results) + 1]] <- tibble(
      outcome_id = out$outcome_id,
      big_task = out$big_task,
      subtask = out$subtask,
      class_label = out$class_label,
      comparison = c("radiomics vs pathomics",
                     "radiomics vs radiopathomics",
                     "pathomics vs radiopathomics"),
      p_value = c(test_rad_path$p.value,
                  test_rad_radpath$p.value,
                  test_path_radpath$p.value),
      estimate = c(test_rad_path$estimate,
                   test_rad_radpath$estimate,
                   test_path_radpath$estimate),
      conf_low = c(test_rad_path$conf.int[1],
                   test_rad_radpath$conf.int[1],
                   test_path_radpath$conf.int[1]),
      conf_high = c(test_rad_path$conf.int[2],
                    test_rad_radpath$conf.int[2],
                    test_path_radpath$conf.int[2]),
      statistic = c(test_rad_path$statistic,
                    test_rad_radpath$statistic,
                    test_path_radpath$statistic),
      parameter = c(test_rad_path$parameter,
                    test_rad_radpath$parameter,
                    test_path_radpath$parameter),
      method = c(test_rad_path$method,
                 test_rad_radpath$method,
                 test_path_radpath$method)
    )
  }

  bind_rows(results) %>%
    mutate(
      log2_p = -log2(p_value),
      log2_p = ifelse(is.infinite(log2_p), -log2(.Machine$double.xmin), log2_p)
    )
}

# ==============================
# Plot -log2(p-value) with reference lines
# ==============================
# ==============================
# Updated plot_comparisons (bar plot, if still used)
# ==============================
plot_comparisons <- function(comp_results, out_dir) {
  if (nrow(comp_results) == 0) {
    cat("No comparison results to plot.\n")
    return(NULL)
  }

  comp_results <- comp_results %>%
    mutate(outcome_id = factor(outcome_id,
                               levels = unique(outcome_id[order(big_task, subtask, class_label)])))

  # Define thresholds
  thresholds <- c(0.05, 0.01, 0.001)
  threshold_vals <- -log2(thresholds)

  # Compute y limits with padding
  y_limits <- range(c(comp_results$log2_p, threshold_vals), na.rm = TRUE)
  y_limits <- y_limits + c(-0.05, 0.1) * diff(y_limits)

  p <- ggplot(comp_results, aes(x = outcome_id, y = log2_p, fill = comparison)) +
    geom_col(position = position_dodge(0.8), width = 0.7) +

    # 🔥 Explicit threshold lines (robust)
    geom_hline(yintercept = -log2(0.05), linetype = "dashed", color = "red", alpha = 0.7) +
    geom_hline(yintercept = -log2(0.01), linetype = "dotted", color = "red", alpha = 0.7) +
    geom_hline(yintercept = -log2(0.001), linetype = "dotdash", color = "red", alpha = 0.7) +

    # 🔥 Annotate thresholds directly (no legend needed)
    annotate("text", x = Inf, y = -log2(0.05), label = "p = 0.05",
             hjust = 1.1, vjust = -0.3, color = "red", size = 3) +
    annotate("text", x = Inf, y = -log2(0.01), label = "p = 0.01",
             hjust = 1.1, vjust = -0.3, color = "red", size = 3) +
    annotate("text", x = Inf, y = -log2(0.001), label = "p = 0.001",
             hjust = 1.1, vjust = -0.3, color = "red", size = 3) +

    coord_cartesian(ylim = y_limits) +

    labs(
      title = "Pairwise Model Comparisons (-log₂ p-value)",
      x = "Outcome (Task | Subtask | Class)",
      y = expression(-log[2](p))
    ) +

    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +

    scale_fill_brewer(palette = "Set1") +
    guides(fill = guide_legend(title = "Comparison"))

  ggsave(file.path(out_dir, "model_comparisons.png"),
         p, width = 14, height = 8, dpi = 300)

  cat("Saved: model_comparisons.png\n")
  return(p)
}

plot_comparisons_by_task <- function(comp_results, out_dir) {
  if (nrow(comp_results) == 0) {
    cat("No comparison results to plot.\n")
    return(NULL)
  }

  comp_results <- comp_results %>%
    mutate(
      short_outcome = paste(subtask, class_label, sep = " | "),
      outcome_short = factor(short_outcome,
                             levels = unique(short_outcome[order(big_task, subtask, class_label)]))
    )

  for (task in unique(comp_results$big_task)) {
    task_data <- comp_results %>% filter(big_task == task)
    if (nrow(task_data) == 0) next

    # y limits
    threshold_vals <- -log2(c(0.05, 0.01, 0.001))
    y_limits <- range(c(task_data$log2_p, threshold_vals), na.rm = TRUE)
    y_limits <- y_limits + c(-0.05, 0.1) * diff(y_limits)

    n_outcomes <- length(unique(task_data$outcome_short))
    plot_width <- max(8, n_outcomes * 0.25)

    p <- ggplot(task_data, aes(x = outcome_short, y = log2_p, color = comparison)) +
      geom_point(position = position_dodge(width = 0.6), size = 2.5, alpha = 0.85) +

      # Threshold lines
      geom_hline(yintercept = -log2(0.05), linetype = "dashed", color = "red", alpha = 0.7) +
      geom_hline(yintercept = -log2(0.01), linetype = "dotted", color = "red", alpha = 0.7) +
      geom_hline(yintercept = -log2(0.001), linetype = "dotdash", color = "red", alpha = 0.7) +

      # Labels
      annotate("text", x = Inf, y = -log2(0.05), label = "p = 0.05",
               hjust = 1.1, vjust = -0.3, color = "red", size = 3) +
      annotate("text", x = Inf, y = -log2(0.01), label = "p = 0.01",
               hjust = 1.1, vjust = -0.3, color = "red", size = 3) +
      annotate("text", x = Inf, y = -log2(0.001), label = "p = 0.001",
               hjust = 1.1, vjust = -0.3, color = "red", size = 3) +

      coord_cartesian(ylim = y_limits) +

      labs(
        title = paste(str_to_title(task), "Model Comparisons"),
        x = "Outcome (Subtask | Class)",
        y = expression(-log[2](p))
      ) +

      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major.x = element_blank()
      ) +

      scale_color_brewer(palette = "Set1") +
      guides(color = guide_legend(title = "Comparison", nrow = 1))

    filename <- file.path(out_dir, paste0(task, "_comparisons.png"))
    ggsave(filename, p, width = plot_width, height = 6, dpi = 300, limitsize = FALSE)

    cat("Saved:", filename, "\n")
  }
}

# ==============================
# Plot effect sizes (mean differences) with confidence intervals
# ==============================
plot_effect_sizes <- function(comp_results, out_dir) {

  if (nrow(comp_results) == 0) return(NULL)

  comp_results <- comp_results %>%
    filter(!is.na(conf_low), !is.na(conf_high), is.finite(estimate)) %>%
    mutate(
      outcome = paste(subtask, class_label, sep = " | "),
      outcome = factor(outcome, levels = unique(outcome[order(big_task, subtask, class_label)])),
      comparison = factor(comparison, levels = c(
        "radiomics vs pathomics",
        "radiomics vs radiopathomics",
        "pathomics vs radiopathomics"
      ))
    )

  for (task in unique(comp_results$big_task)) {

    df <- comp_results %>% filter(big_task == task)
    if (nrow(df) == 0) next

    n_outcomes <- length(unique(df$outcome))
    plot_width <- max(10, n_outcomes * 0.35)

    # x limits
    x_limits <- range(c(df$conf_low, df$conf_high), na.rm = TRUE)
    x_limits <- x_limits + c(-0.05, 0.05) * diff(x_limits)

    p <- ggplot(df, aes(x = outcome, y = estimate, color = comparison)) +

      # zero line
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +

      # error bars
      geom_errorbar(aes(ymin = conf_low, ymax = conf_high),
                    width = 0.2,
                    position = position_dodge(width = 0.6),
                    size = 0.4) +

      # points
      geom_point(position = position_dodge(width = 0.6),
                 size = 1.8, alpha = 0.9) +

      coord_cartesian(ylim = x_limits) +

      facet_wrap(~ comparison, nrow = 1) +   # 🔥 key improvement

      labs(
        title = paste(str_to_title(task), "Effect Sizes"),
        x = "Outcome",
        y = "Mean Difference"
      ) +

      theme_minimal(base_size = 11) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        axis.text.y = element_text(size = 8),
        legend.position = "none",   # 🔥 redundant due to facet
        plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid.major.x = element_blank(),
        strip.text = element_text(size = 9, face = "bold")
      ) +

      scale_color_brewer(palette = "Set1")

    ggsave(
      file.path(out_dir, paste0(task, "_effectsize_compact.png")),
      p,
      width = plot_width,
      height = 5,
      dpi = 300,
      limitsize = FALSE
    )

    cat("Saved:", paste0(task, "_effectsize_compact.png\n"))
  }
}

library(patchwork)

plot_composite <- function(comp_results, out_dir) {

  if (nrow(comp_results) == 0) return(NULL)

  df <- comp_results %>%
    filter(
      !is.na(conf_low), !is.na(conf_high),
      is.finite(log2_p), is.finite(estimate)
    ) %>%
    mutate(
      outcome = paste(subtask, class_label, sep = " | "),
      outcome = factor(outcome, levels = unique(outcome[order(big_task, subtask, class_label)]))
    )

  for (task in unique(df$big_task)) {

    task_data <- df %>% filter(big_task == task)
    if (nrow(task_data) == 0) next

    for (comp in unique(task_data$comparison)) {

      d <- task_data %>% filter(comparison == comp)
      if (nrow(d) == 0) next

      # ======================
      # Axis limits
      # ======================
      y_p <- range(d$log2_p, -log2(c(0.05, 0.01, 0.001)), na.rm = TRUE)
      y_p <- y_p + c(-0.05, 0.1) * diff(y_p)

      y_eff <- range(c(d$conf_low, d$conf_high), na.rm = TRUE)
      y_eff <- y_eff + c(-0.05, 0.05) * diff(y_eff)

      n_outcomes <- length(unique(d$outcome))
      plot_width <- max(8, n_outcomes * 0.35)

      # ======================
      # P-VALUE PANEL
      # ======================
      p_top <- ggplot(d, aes(x = outcome, y = log2_p)) +
        geom_point(size = 2, color = "#d95f02") +

        geom_hline(yintercept = -log2(0.05), linetype = "dashed", color = "red") +
        geom_hline(yintercept = -log2(0.01), linetype = "dotted", color = "red") +
        geom_hline(yintercept = -log2(0.001), linetype = "dotdash", color = "red") +

        coord_cartesian(ylim = y_p) +

        labs(
          title = NULL,
          y = expression(-log[2](p)),
          x = NULL
        ) +

        theme_minimal(base_size = 11) +
        theme(
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          panel.grid.major.x = element_blank()
        )

      # ======================
      # EFFECT SIZE PANEL
      # ======================
      p_bottom <- ggplot(d, aes(x = outcome, y = estimate)) +

        geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +

        geom_errorbar(aes(ymin = conf_low, ymax = conf_high),
                      width = 0.2, size = 0.4, color = "#1f78b4") +

        geom_point(
          aes(shape = conf_low * conf_high > 0),  # significance
          size = 2, color = "#1f78b4"
        ) +

        scale_shape_manual(
          values = c(1, 16),
          labels = c("Not significant", "Significant"),
          name = "Effect CI"
        ) +

        coord_cartesian(ylim = y_eff) +

        labs(
          x = "Outcome",
          y = "Mean Difference"
        ) +

        theme_minimal(base_size = 11) +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          panel.grid.major.x = element_blank(),
          legend.position = "bottom"
        )

      # ======================
      # Combine
      # ======================
      comp_title <- case_when(
        comp == "radiomics vs pathomics" ~ "Radiomics vs Pathomics",
        comp == "radiomics vs radiopathomics" ~ "Radiomics vs Radiopathomics",
        comp == "pathomics vs radiopathomics" ~ "Pathomics vs Radiopathomics",
        TRUE ~ comp
      )

      combined <- p_top / p_bottom +
        plot_annotation(
          title = paste(str_to_title(task), "-", comp_title),
          theme = theme(plot.title = element_text(hjust = 0.5, face = "bold"))
        )

      filename <- file.path(
        out_dir,
        paste0(task, "_", gsub(" ", "_", comp), "_composite.png")
      )

      ggsave(
        filename,
        combined,
        width = plot_width,
        height = 7,
        dpi = 300,
        limitsize = FALSE
      )

      cat("Saved:", filename, "\n")
    }
  }
}

# ==============================
# Add to main execution
# ==============================
main_comparisons <- function() {
  cat("\n=== Performing model comparisons ===\n")

  # Load data (reuse from earlier, but we can call load_all_data again if needed)
  # We assume df_all is already available from main_pheatmap. For standalone, we'd reload.
  # Here we'll recreate it if not present.
  if (!exists("df_all")) {
    df_all <- load_all_data()
  }

  # Select best models
  best_models <- select_best_models(df_all)
  cat("Best models selected:\n")
  print(best_models %>% count(big_task, omics_type))

  # Perform comparisons
  comp_results <- compare_best_models(best_models, df_all)
  cat("Number of comparisons:", nrow(comp_results), "\n")

  if (nrow(comp_results) > 0) {
    # Write results to CSV
    write.csv(comp_results, file.path(out_dir, "model_comparisons.csv"), row.names = FALSE)

    # Plot
    plot_comparisons_by_task(comp_results, out_dir)

    # Plot effect sizes
    # plot_effect_sizes(comp_results, out_dir)

    plot_composite(comp_results, out_dir)

  } else {
    cat("No valid comparisons (missing omics types for some outcomes).\n")
  }
}

main_comparisons()