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

# visualize topk, ranking or raw score
topk <- FALSE
ranking <- TRUE

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

class_name_map <- list(
  GeneProgrames = c(
    `0` = "GP1_Proliferation/DNA_repair",
    `1` = "GP2_Immune-Tcell/Bcell",
    `2` = "GP3_Tumor_suppressing_miRNA_targets",
    `3` = "GP4_MES/ECM",
    `4` = "GP5_MYC_targets/TERT",
    `5` = "GP6_Squamous_differentiation/development",
    `6` = "GP7_Estrogen_signaling",
    `7` = "GP8_FOXO/stemness",
    `8` = "GP9_Cell-cell_adhesion",
    `9` = "GP10_Fatty_acid_oxidation",
    `10` = "GP11_Immune-IFN_GP12_Hypoxia/glycolosis",
    `11` = "GP13_Neural_signailng",
    `12` = "GP14_Plasma_membrane_cell-cell_signaling",
    `13` = "GP15_EGF_signailng",
    `14` = "GP16_Protein_kinase_signailng_(MAPKs)",
    `15` = "GP17_Basal_signaling",
    `16` = "GP18_Vesicle/EPR_membrane_coat",
    `17` = "GP19_1Q_amplicon",
    `18` = "GP20_TAL1-Leukemia/erythropoiesis",
    `19` = "GP21_Anti-apoptosis/DNA_stability",
    `20` = "GP22_16Q22-24_amplicon",
    `21` = "AKT_PATHWAY",
    `22` = "ALK_PATHWAY",
    `23` = "BRCA_ATR_PATHWAY",
    `24` = "CASPASE_CASCADE_(APOPTOSIS)",
    `25` = "CTLA4_PATHWAY",
    `26` = "HDAC_TARGETS_DN",
    `27` = "HER2_AMPLIFIED",
    `28` = "IGF1R_PATHWAY",
    `29` = "MTOR_PATHWAY",
    `30` = "MYC_amplified",
    `31` = "PD1_SIGNALING",
    `32` = "PI3K_CASCADE",
    `33` = "PTEN_PATHWAY",
    `34` = "RAS_PATHWAY",
    `35` = "RB_PATHWAY",
    `36` = "RESPONSE_TO_ANDROGEN",
    `37` = "RETINOL_METABOLISM",
    `38` = "VEGF_PATHWAY"
  ),
  HRDscore = c(
    `0` = "ai1",
    `1` = "lst1",
    `2` = "hrd-loh",
    `3` = "HRD"
  ),
  ImmuneSignatureScore = c(
    `0` = "ICS5_score",
    `1` = "LIexpression_score",
    `2` = "Chemokine12_score",
    `3` = "NHI_5gene_score",
    `4` = "CD68",
    `5` = "CD8A",
    `6` = "PD1_data",
    `7` = "PDL1_data",
    `8` = "PD1_PDL1_score",
    `9` = "CTLA4_data",
    `10` = "Bcell_mg_IGJ",
    `11` = "Bcell_receptors_score",
    `12` = "STAT1_score",
    `13` = "CSF1_response",
    `14` = "TcClassII_score",
    `15` = "IL12_score_21050467",
    `16` = "IL4_score_21050467",
    `17` = "IL2_score_21050467",
    `18` = "IL13_score_21050467",
    `19` = "IFNG_score_21050467",
    `20` = "TGFB_score_21050467",
    `21` = "TREM1_data",
    `22` = "DAP12_data",
    `23` = "Tcell_receptors_score",
    `24` = "IL8_21978456",
    `25` = "IFN_21978456",
    `26` = "MHC1_21978456",
    `27` = "MHC2_21978456",
    `28` = "Bcell_21978456",
    `29` = "Tcell_21978456",
    `30` = "CD103pos_mean_25446897",
    `31` = "CD103neg_mean_25446897",
    `32` = "IgG_19272155",
    `33` = "Interferon_19272155",
    `34` = "LCK_19272155",
    `35` = "MHC.I_19272155",
    `36` = "MHC.II_19272155",
    `37` = "STAT1_19272155",
    `38` = "Troester_WoundSig_19887484",
    `39` = "MDACC.FNA.1_20805453",
    `40` = "IGG_Cluster_21214954",
    `41` = "Minterferon_Cluster_21214954",
    `42` = "Immune_cell_Cluster_21214954",
    `43` = "MCD3_CD8_21214954",
    `44` = "Interferon_Cluster_21214954",
    `45` = "B_cell_PCA_16704732",
    `46` = "CD8_PCA_16704732",
    `47` = "GRANS_PCA_16704732",
    `48` = "LYMPHS_PCA_16704732",
    `49` = "T_cell_PCA_16704732",
    `50` = "TGFB_PCA_17349583",
    `51` = "Rotterdam_ERneg_PCA_15721472",
    `52` = "HER2_Immune_PCA_18006808",
    `53` = "IR7_score",
    `54` = "Buck14_score",
    `55` = "TAMsurr_score",
    `56` = "Immune_NSCLC_score",
    `57` = "Module3_IFN_score",
    `58` = "Module4_TcellBcell_score",
    `59` = "Module5_TcellBcell_score",
    `60` = "Module11_Prolif_score",
    `61` = "GP11_Immune_IFN",
    `62` = "GP2_ImmuneTcellBcell_score",
    `63` = "CD8_CD68_ratio",
    `64` = "TAMsurr_TcClassII_ratio",
    `65` = "CHANG_CORE_SERUM_RESPONSE_UP",
    `66` = "CSR_Activated_15701700",
    `67` = "CD103pos_CD103neg_ratio_25446897"
  ),
  ImmuneSubtype = c(
    `0` = "IFN-gamma Dominant (Immune C2)",
    `1` = "Inflammatory (Immune C3)",
    `2` = "Wound Healing (Immune C1)",
    `3` = "Lymphocyte Depleted (Immune C4)"
  ),
  MolecularSubtype = c(
    `0` = "BRCA.LumA",
    `1` = "BRCA.LumB",
    `2` = "BRCA.Basal",
    `3` = "BRCA.Normal",
    `4` = "GI.CIN",
    `5` = "KIRC.1",
    `6` = "KIRC.2",
    `7` = "KIRC.3",
    `8` = "KIRC.4",
    `9` = "LIHC.iCluster:3",
    `10` = "LIHC.iCluster:1",
    `11` = "LIHC.iCluster:2",
    `12` = "OVCA.Proliferative",
    `13` = "OVCA.Differentiated",
    `14` = "OVCA.Mesenchymal",
    `15` = "UCEC.CN_HIGH"
  ),
  PrimaryDisease = c(
    `0` = "breast invasive carcinoma",
    `1` = "bladder urothelial carcinoma",
    `2` = "ovarian serous cystadenocarcinoma",
    `3` = "lung adenocarcinoma",
    `4` = "stomach adenocarcinoma",
    `5` = "lung squamous cell carcinoma",
    `6` = "liver hepatocellular carcinoma",
    `7` = "kidney clear cell carcinoma",
    `8` = "uterine corpus endometrioid carcinoma",
    `9` = "cervical & endocervical cancer"
  ),
  StemnessScoreDNA = c(
    `0` = "DNAss",
    `1` = "EREG-METHss",
    `2` = "DMPss",
    `3` = "ENHss"
  ),
  StemnessScoreRNA = c(
    `0` = "RNAss",
    `1` = "EREG.EXPss"
  )
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

  # 根据参数选择矩阵处理方式
  if (ranking) {
    # 排名分数：归一化到 [0, 1]，higher score = better performance
    cat("Computing normalized ranking scores [0, 1]...\n")
    hm_matrix_scored <- t(apply(hm_matrix, 1, function(x) {
      result <- rep(0, length(x))
      non_na_idx <- which(!is.na(x))
      non_na_count <- length(non_na_idx)
      
      if (non_na_count > 0) {
        # 按值降序排序
        sorted_idx <- non_na_idx[order(x[non_na_idx], decreasing = TRUE)]
        
        # 排名分数：最高值 = non_na_count，最低值 = 1
        raw_scores <- rep(0, length(x))
        for (i in 1:non_na_count) {
          raw_scores[sorted_idx[i]] <- non_na_count - i + 1
        }
        
        # 归一化到 [0, 1]
        if (non_na_count > 1) {
          result <- (raw_scores - 1) / (non_na_count - 1)
        } else {
          # 只有一个非NA值时，设为1
          result[non_na_idx] <- 1
        }
      }
      
      return(result)
    })) 
  } else if (topk) {
    # 原始 topk 逻辑：只给前5名赋值 1, 0.8, 0.6, 0.4, 0.2
    cat("Computing top-k scores (top 5 models)...\n")
    hm_matrix_scored <- t(apply(hm_matrix, 1, function(x) {
      result <- rep(0, length(x))
      non_na_idx <- which(!is.na(x))
      non_na_count <- length(non_na_idx)
      
      if (non_na_count > 0) {
        # 获取排序后的索引
        sorted_idx <- non_na_idx[order(x[non_na_idx], decreasing = TRUE)]
        
        # 赋值向量
        values <- c(1, 0.8, 0.6, 0.4, 0.2)
        
        # 取前 min(5, non_na_count) 个进行赋值
        n <- min(5, non_na_count)
        for (i in 1:n) {
          result[sorted_idx[i]] <- values[i]
        }
      }
      
      return(result)
    }))
  } else {
    # 默认：使用原始均值
    hm_matrix_scored <- hm_matrix
    cat("Using raw mean values...\n")
  }

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
        color_palette <- scico(100, palette = "vik")
      }

      if (task == "survival") {
        breaks = seq(0.5, 0.9, length = 100)
      } else if (task == "phenotype") {
        breaks = seq(0, 1, length = 100)
      } else {
        breaks = seq(-1, 1, length = 100)
      }

      if (topk || ranking) {
          breaks = seq(0, 1, length = 100)
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
        breaks = breaks,
        
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
create_summary_boxplots <- function(heatmap_data) {
  
  cat("\nCreating summary box plots...\n")
  
  hm_matrix <- heatmap_data$matrix
  
  if (is.null(hm_matrix) || length(hm_matrix) == 0) {
    cat("No data for summary box plots.\n")
    return(NULL)
  }
  
  if (!exists("out_dir")) {
    out_dir <- "."
  }
  
  row_id_order <- rownames(heatmap_data$matrix)
  col_id_order <- colnames(heatmap_data$matrix)
  
  plot_data <- as.data.frame(hm_matrix) %>%
    tibble::rownames_to_column("row_id") %>%
    pivot_longer(
      cols = -row_id,
      names_to = "col_id",
      values_to = "value"
    ) %>%
    filter(!is.na(value))
  
  if (!is.null(heatmap_data$row_anno)) {
    row_anno <- heatmap_data$row_anno %>%
      mutate(row_id = as.character(row_id))
    
    plot_data <- plot_data %>%
      left_join(row_anno, by = "row_id")
  }
  
  if (!is.null(heatmap_data$col_anno)) {
    col_anno <- heatmap_data$col_anno %>%
      mutate(col_id = as.character(col_id))
    
    plot_data <- plot_data %>%
      left_join(col_anno, by = "col_id")
  }
  
  if (nrow(plot_data) == 0) {
    cat("No non-missing values for summary box plots.\n")
    return(NULL)
  }
  
  plot_data <- plot_data %>%
    mutate(
      row_id = factor(row_id, levels = row_id_order),
      col_id = factor(col_id, levels = col_id_order)
    )
  
  if ("omics_type" %in% colnames(plot_data)) {
    plot_data$omics_type <- factor(plot_data$omics_type, levels = omics_levels)
  }
  
  if ("agg_mode" %in% colnames(plot_data)) {
    plot_data$agg_mode <- factor(plot_data$agg_mode, levels = agg_levels)
  }
  
  if (!"big_task" %in% colnames(plot_data)) {
    plot_data$big_task <- NA_character_
  }
  
  if (!"subtask" %in% colnames(plot_data)) {
    plot_data$subtask <- NA_character_
  }
  
  if (!"class_index" %in% colnames(plot_data)) {
    plot_data$class_index <- NA_integer_
  }
  
  if (!"class_label" %in% colnames(plot_data)) {
    plot_data$class_label <- NA_character_
  }
  
  mapped_class_label <- mapply(
    function(subtask, class_index, class_label) {
      subtask <- as.character(subtask)
      class_key <- as.character(as.integer(class_index) - 1)
      
      if (!is.na(subtask) &&
          subtask %in% names(class_name_map) &&
          !is.na(class_key) &&
          class_key %in% names(class_name_map[[subtask]])) {
        return(class_name_map[[subtask]][[class_key]])
      }
      
      return(as.character(class_label))
    },
    plot_data$subtask,
    plot_data$class_index,
    plot_data$class_label,
    USE.NAMES = FALSE
  )
  
  mapped_row_id <- ifelse(
    as.character(plot_data$subtask) %in% names(class_name_map),
    paste(plot_data$big_task, plot_data$subtask, mapped_class_label, sep = " | "),
    as.character(plot_data$row_id)
  )
  
  plot_data$row_id_display <- mapped_row_id
  row_id_display_order <- plot_data %>%
    distinct(row_id, row_id_display) %>%
    arrange(match(as.character(row_id), row_id_order)) %>%
    pull(row_id_display)
  
  plot_data$row_id_display <- factor(plot_data$row_id_display, levels = row_id_display_order)
  
  value_label <- case_when(
    exists("ranking") && isTRUE(ranking) ~ "Ranking Score",
    exists("topk") && isTRUE(topk) ~ "Top-k Score",
    TRUE ~ "Performance Value"
  )
  
  omics_plot_data <- plot_data %>%
    filter(as.character(agg_mode) == "SPARRA")
  
  if (nrow(omics_plot_data) == 0) {
    cat("No SPARRA values found for omics box plots.\n")
  }
  
  cat("Creating box plot by row_id across all models...\n")
  
  rowid_plot <- ggplot(plot_data, aes(x = row_id_display, y = value, fill = row_id_display)) +
    geom_boxplot(alpha = 0.7, outlier.size = 1.5, outlier.alpha = 0.5) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3, 
                 fill = "white", color = "black", stroke = 1.2) +
    labs(
      title = "Score Distribution by Subtask",
      subtitle = "Across all models, omics types, and aggregation modes",
      x = "Subtask",
      y = value_label
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
      legend.position = "none",
      panel.grid.minor = element_blank(),
      plot.margin = margin(t = 15, r = 20, b = 120, l = 140)
    ) +
    scale_fill_viridis_d() +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.1))) +
    coord_cartesian(clip = "off")
  
  n_rows <- length(unique(plot_data$row_id_display))
  plot_width <- max(10, n_rows * 0.4)
  
  ggsave(
    filename = file.path(out_dir, "summary_boxplot_by_rowid.png"),
    plot = rowid_plot,
    width = plot_width,
    height = 10,
    dpi = 300,
    limitsize = FALSE
  )
  cat("Saved: summary_boxplot_by_rowid.png\n")
  
  cat("Creating box plot by Omics Type using SPARRA values only...\n")
  
  if (nrow(omics_plot_data) > 0) {
    omics_plot <- ggplot(omics_plot_data, aes(x = omics_type, y = value, fill = omics_type)) +
      geom_boxplot(alpha = 0.7, outlier.size = 1.5, outlier.alpha = 0.5) +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 3, 
                   fill = "white", color = "black", stroke = 1.2) +
      labs(
        title = "Score Distribution by Omics Type",
        subtitle = "SPARRA aggregation only, across all models",
        x = "Omics Type",
        y = value_label
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 10),
        axis.title = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
        legend.position = "none",
        panel.grid.minor = element_blank()
      ) +
      scale_fill_viridis_d() +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))
    
    ggsave(
      filename = file.path(out_dir, "summary_boxplot_by_omics.png"),
      plot = omics_plot,
      width = 10,
      height = 8,
      dpi = 300
    )
    cat("Saved: summary_boxplot_by_omics.png\n")
  }
  
  cat("Creating box plot by Aggregation Mode across models and omics types...\n")
  
  agg_plot <- ggplot(plot_data, aes(x = agg_mode, y = value, fill = agg_mode)) +
    geom_boxplot(alpha = 0.7, outlier.size = 1.5, outlier.alpha = 0.5) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3, 
                 fill = "white", color = "black", stroke = 1.2) +
    labs(
      title = "Score Distribution by Aggregation Mode",
      subtitle = "Across all models and omics types",
      x = "Aggregation Mode",
      y = value_label
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
      legend.position = "none",
      panel.grid.minor = element_blank()
    ) +
    scale_fill_viridis_d() +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))
  
  ggsave(
    filename = file.path(out_dir, "summary_boxplot_by_agg.png"),
    plot = agg_plot,
    width = 10,
    height = 8,
    dpi = 300
  )
  cat("Saved: summary_boxplot_by_agg.png\n")
  
  task_order <- if (exists("big_tasks")) names(big_tasks) else unique(as.character(plot_data$big_task))
  task_order <- task_order[task_order %in% unique(as.character(plot_data$big_task))]
  
  for (task in task_order) {
    task_data <- plot_data %>%
      filter(as.character(big_task) == task)
    
    if (nrow(task_data) == 0) {
      next
    }
    
    task_label <- str_to_title(task)
    task_file <- str_replace_all(task, "[^A-Za-z0-9]+", "_")
    task_row_levels <- row_id_order[row_id_order %in% as.character(task_data$row_id)]
    task_row_display_levels <- plot_data %>%
      filter(as.character(row_id) %in% task_row_levels) %>%
      distinct(row_id, row_id_display) %>%
      arrange(match(as.character(row_id), task_row_levels)) %>%
      pull(row_id_display)
    
    task_data <- task_data %>%
      mutate(
        row_id = factor(as.character(row_id), levels = task_row_levels),
        row_id_display = factor(as.character(row_id_display), levels = task_row_display_levels)
      )
    
    cat("Creating task-specific box plots for ", task, "...\n", sep = "")
    
    task_rowid_plot <- ggplot(task_data, aes(x = row_id_display, y = value, fill = row_id_display)) +
      geom_boxplot(alpha = 0.7, outlier.size = 1.5, outlier.alpha = 0.5) +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
                   fill = "white", color = "black", stroke = 1.2) +
      labs(
        title = paste(task_label, "Score Distribution by Subtask"),
        subtitle = "Across all models, omics types, and aggregation modes",
        x = "Subtask",
        y = value_label
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
        axis.text.y = element_text(size = 10),
        axis.title = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
        legend.position = "none",
        panel.grid.minor = element_blank(),
        plot.margin = margin(t = 15, r = 20, b = 120, l = 140)
      ) +
      scale_fill_viridis_d() +
      scale_y_continuous(
        breaks = seq(
          floor(min(task_data$value, na.rm = TRUE) / 0.15) * 0.15,
          ceiling(max(task_data$value, na.rm = TRUE) / 0.15) * 0.15,
          by = 0.15
        ),
        expand = expansion(mult = c(0.05, 0.1))
      ) +
      coord_cartesian(clip = "off")
    
    task_n_rows <- length(unique(task_data$row_id_display))
    task_plot_width <- max(10, task_n_rows * 0.4)
    
    ggsave(
      filename = file.path(out_dir, paste0("summary_boxplot_", task_file, "_by_rowid.png")),
      plot = task_rowid_plot,
      width = task_plot_width,
      height = 10,
      dpi = 300,
      limitsize = FALSE
    )
    cat("Saved: summary_boxplot_", task_file, "_by_rowid.png\n", sep = "")
    
    task_omics_data <- task_data %>%
      filter(as.character(agg_mode) == "SPARRA")
    
    if (nrow(task_omics_data) > 0) {
      task_omics_plot <- ggplot(task_omics_data, aes(x = omics_type, y = value, fill = omics_type)) +
        geom_boxplot(alpha = 0.7, outlier.size = 1.5, outlier.alpha = 0.5) +
        stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
                     fill = "white", color = "black", stroke = 1.2) +
        labs(
          title = paste(task_label, "Score Distribution by Omics Type"),
          subtitle = "SPARRA aggregation only, across all models",
          x = "Omics Type",
          y = value_label
        ) +
        theme_minimal() +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          axis.text.y = element_text(size = 10),
          axis.title = element_text(size = 12, face = "bold"),
          plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
          legend.position = "none",
          panel.grid.minor = element_blank()
        ) +
        scale_fill_viridis_d() +
        scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))
      
      ggsave(
        filename = file.path(out_dir, paste0("summary_boxplot_", task_file, "_by_omics.png")),
        plot = task_omics_plot,
        width = 10,
        height = 8,
        dpi = 300
      )
      cat("Saved: summary_boxplot_", task_file, "_by_omics.png\n", sep = "")
    } else {
      cat("No SPARRA values for ", task, " omics box plot.\n", sep = "")
    }
    
    task_agg_plot <- ggplot(task_data, aes(x = agg_mode, y = value, fill = agg_mode)) +
      geom_boxplot(alpha = 0.7, outlier.size = 1.5, outlier.alpha = 0.5) +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
                   fill = "white", color = "black", stroke = 1.2) +
      labs(
        title = paste(task_label, "Score Distribution by Aggregation Mode"),
        subtitle = "Across all models and omics types",
        x = "Aggregation Mode",
        y = value_label
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 10),
        axis.title = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
        legend.position = "none",
        panel.grid.minor = element_blank()
      ) +
      scale_fill_viridis_d() +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))
    
    ggsave(
      filename = file.path(out_dir, paste0("summary_boxplot_", task_file, "_by_agg.png")),
      plot = task_agg_plot,
      width = 10,
      height = 8,
      dpi = 300
    )
    cat("Saved: summary_boxplot_", task_file, "_by_agg.png\n", sep = "")
  }
  
  summary_stats <- list(
    by_rowid = plot_data %>%
      group_by(row_id) %>%
      summarise(
        big_task = first(big_task),
        subtask = first(subtask),
        mean = mean(value, na.rm = TRUE),
        median = median(value, na.rm = TRUE),
        sd = sd(value, na.rm = TRUE),
        min = min(value, na.rm = TRUE),
        max = max(value, na.rm = TRUE),
        n = n(),
        .groups = "drop"
      ) %>%
      arrange(factor(row_id, levels = row_id_order)),
    by_omics = omics_plot_data %>%
      group_by(omics_type) %>%
      summarise(
        mean = mean(value, na.rm = TRUE),
        median = median(value, na.rm = TRUE),
        sd = sd(value, na.rm = TRUE),
        min = min(value, na.rm = TRUE),
        max = max(value, na.rm = TRUE),
        n = n(),
        .groups = "drop"
      ),
    by_agg = plot_data %>%
      group_by(agg_mode) %>%
      summarise(
        mean = mean(value, na.rm = TRUE),
        median = median(value, na.rm = TRUE),
        sd = sd(value, na.rm = TRUE),
        min = min(value, na.rm = TRUE),
        max = max(value, na.rm = TRUE),
        n = n(),
        .groups = "drop"
      )
  )
  
  cat("\nSummary box plots completed successfully!\n")
  cat("Generated overall plot files:\n")
  cat("  - summary_boxplot_by_rowid.png\n")
  cat("  - summary_boxplot_by_omics.png\n")
  cat("  - summary_boxplot_by_agg.png\n")
  cat("Generated task-specific plot files for:", paste(task_order, collapse = ", "), "\n")
  
  return(summary_stats)
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
  
  # 3. 创建汇总箱线图
  cat("\n3. Creating summary box plots...\n")
  tryCatch({
    create_summary_boxplots(heatmap_data)
  }, error = function(e) {
    cat("   Error:", e$message, "\n")
  })
  
  cat("\n=== All heatmaps created successfully! ===\n")
  cat("Files saved to:", out_dir, "\n")
}

# 运行
main_pheatmap()
