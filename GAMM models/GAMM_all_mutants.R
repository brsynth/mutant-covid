library(readxl)
library(writexl)
library(tidyr)
library(stringr)
library(mgcv)
library(tibble)
library(purrr)
library(gtools)
library(MASS)
library(dplyr)

# ============================================================
# Settings
# ============================================================
data_folder <- "time_series_pools"
input_file <- file.path(data_folder, "Summary - Pools.xlsx")
output_folder <- file.path(data_folder, "bam_selected_model")
dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)

replicate_sheets <- c(
  "Curves - Replicate 1",
  "Curves - Replicate 2",
  "Curves - Replicate 3"
)

analysis_modes <- c("MS", "NP", "NS", "MN", "three_groups")

alpha <- 0.1
nsim_simultaneous <- 10000
random_seed <- 123
rho_value <- 0

pairwise_correction_requested <- "BH"
# allowed: "none", "holm", "bonferroni", "BH"

overall_correction_requested <- "BH"
# allowed: "none", "holm", "bonferroni", "BH"

fallback_degree_global <- 20
fallback_degree_fs <- 20

# ============================================================
# Existing AIC workbooks = source of truth for model selection
# ============================================================
aic_workbook_paths <- list(
  MS = file.path("time_series_pools", "bam_model_selection", "BAM_MS_all_strains_model_comparison.xlsx"),
  NP = file.path("time_series_pools", "bam_model_selection", "BAM_NP_all_strains_model_comparison.xlsx"),
  NS = file.path("time_series_pools", "bam_model_selection", "BAM_NS_all_strains_model_comparison.xlsx"),
  MN = file.path("time_series_pools", "bam_model_selection", "BAM_MN_all_strains_model_comparison.xlsx"),
  three_groups = file.path("time_series_pools", "bam_model_selection", "BAM_three_groups_all_strains_model_comparison.xlsx")
)

# ============================================================
# Helpers
# ============================================================
text_sheet <- function(x) {
  data.frame(output = capture.output(x), stringsAsFactors = FALSE)
}

safe_bind_rows <- function(x) {
  x <- x[vapply(x, function(z) !is.null(z), logical(1))]
  x <- x[vapply(x, function(z) nrow(z) > 0, logical(1))]
  if (length(x) == 0) {
    return(data.frame())
  }
  dplyr::bind_rows(x)
}

empty_sig_windows <- function() {
  data.frame(
    strain = character(0),
    analysis_mode = character(0),
    selected_model = character(0),
    prediction_type = character(0),
    comparison = character(0),
    block_id = numeric(0),
    start_time = numeric(0),
    end_time = numeric(0),
    duration = numeric(0),
    mean_diff = numeric(0),
    max_abs_diff = numeric(0),
    stringsAsFactors = FALSE
  )
}

empty_all_groups_windows <- function() {
  data.frame(
    strain = character(0),
    analysis_mode = character(0),
    selected_model = character(0),
    prediction_type = character(0),
    interpretation = character(0),
    block_id = numeric(0),
    start_time = numeric(0),
    end_time = numeric(0),
    duration = numeric(0),
    stringsAsFactors = FALSE
  )
}

# ============================================================
# Generic multiplicity helpers
# ============================================================
adjust_pvalues_vector <- function(p, method) {
  if (!method %in% c("none", "holm", "bonferroni", "BH")) {
    stop("Correction method must be one of: 'none', 'holm', 'bonferroni', 'BH'")
  }
  
  p <- as.numeric(p)
  
  if (method == "none") {
    return(p)
  }
  
  stats::p.adjust(p, method = method)
}

apply_workbook_level_overall_correction <- function(overall_global_test,
                                                    run_info,
                                                    alpha,
                                                    method) {
  if (nrow(overall_global_test) == 0) {
    return(list(
      Overall_Global_Test = overall_global_test,
      Run_Info = run_info
    ))
  }
  
  pcol <- intersect(c("p-value", "p.value", "p_value"), names(overall_global_test))
  if (length(pcol) == 0) {
    stop("No p-value column found in Overall_Global_Test.")
  }
  
  overall_tbl <- overall_global_test %>%
    dplyr::mutate(
      overall_p_raw = as.numeric(.data[[pcol[1]]]),
      overall_p_adjusted = adjust_pvalues_vector(overall_p_raw, method = method),
      overall_correction_used = method,
      overall_passes = is.finite(overall_p_adjusted) &
        !is.na(overall_p_adjusted) &
        overall_p_adjusted <= alpha,
      overall_interpretation_adjusted = dplyr::case_when(
        overall_passes ~ "Overall test passes workbook-level multiplicity correction across all strains",
        TRUE ~ "Overall test does not pass workbook-level multiplicity correction across all strains"
      )
    )
  
  run_info_tbl <- run_info %>%
    dplyr::select(
      -dplyr::any_of(c(
        "overall_p_raw", "overall_p_adjusted",
        "overall_correction_used", "overall_passes",
        "overall_interpretation_adjusted"
      ))
    ) %>%
    dplyr::left_join(
      overall_tbl %>%
        dplyr::select(
          strain, analysis_mode, selected_model,
          overall_p_raw, overall_p_adjusted,
          overall_correction_used, overall_passes,
          overall_interpretation_adjusted
        ) %>%
        dplyr::distinct(),
      by = c("strain", "analysis_mode", "selected_model")
    )
  
  list(
    Overall_Global_Test = overall_tbl,
    Run_Info = run_info_tbl
  )
}

apply_workbook_level_pairwise_global_correction <- function(pairwise_global_tests,
                                                            alpha,
                                                            method) {
  if (nrow(pairwise_global_tests) == 0) {
    return(pairwise_global_tests)
  }
  
  out <- pairwise_global_tests %>%
    dplyr::mutate(
      pairwise_global_p_raw = as.numeric(global_p),
      pairwise_global_p_adjusted = adjust_pvalues_vector(pairwise_global_p_raw, method = method),
      pairwise_correction_used = method,
      passes_global = is.finite(pairwise_global_p_adjusted) &
        !is.na(pairwise_global_p_adjusted) &
        pairwise_global_p_adjusted <= alpha,
      final_pairwise_interpretation = dplyr::case_when(
        passes_global ~ "Pairwise global contrast test passes workbook-level multiplicity correction",
        TRUE ~ "Pairwise global contrast test does not pass workbook-level multiplicity correction"
      )
    )
  
  out
}

apply_workbook_level_pairwise_mgcv_correction <- function(pairwise_global_tests_mgcv,
                                                          alpha,
                                                          method) {
  if (nrow(pairwise_global_tests_mgcv) == 0) {
    return(pairwise_global_tests_mgcv)
  }
  
  pcol <- intersect(c("p-value", "p.value", "p_value"), names(pairwise_global_tests_mgcv))
  if (length(pcol) == 0) {
    stop("No p-value column found in Pairwise_Global_Tests_mgcv.")
  }
  
  pairwise_global_tests_mgcv %>%
    dplyr::mutate(
      pairwise_mgcv_p_raw = as.numeric(.data[[pcol[1]]]),
      pairwise_mgcv_p_adjusted = adjust_pvalues_vector(pairwise_mgcv_p_raw, method = method),
      pairwise_mgcv_correction_used = method,
      pairwise_mgcv_passes = is.finite(pairwise_mgcv_p_adjusted) &
        !is.na(pairwise_mgcv_p_adjusted) &
        pairwise_mgcv_p_adjusted <= alpha,
      pairwise_mgcv_interpretation_adjusted = dplyr::case_when(
        pairwise_mgcv_passes ~ "Pairwise mgcv smooth-term test passes workbook-level multiplicity correction",
        TRUE ~ "Pairwise mgcv smooth-term test does not pass workbook-level multiplicity correction"
      )
    )
}

apply_final_significance_logic <- function(pairwise_differences,
                                           overall_global_test,
                                           pairwise_global_tests,
                                           pairwise_global_tests_mgcv,
                                           alpha) {
  if (nrow(pairwise_differences) == 0) {
    return(pairwise_differences)
  }
  
  overall_flags <- overall_global_test %>%
    dplyr::select(
      strain, analysis_mode, selected_model,
      overall_p_raw, overall_p_adjusted,
      overall_correction_used, overall_passes
    ) %>%
    dplyr::distinct()
  
  pairwise_flags <- pairwise_global_tests %>%
    dplyr::select(
      strain, analysis_mode, selected_model, prediction_type,
      comparison, pairwise_global_p_raw, pairwise_global_p_adjusted,
      pairwise_correction_used, passes_global, final_pairwise_interpretation
    ) %>%
    dplyr::distinct()
  
  if (nrow(pairwise_global_tests_mgcv) > 0) {
    mgcv_flags <- pairwise_global_tests_mgcv %>%
      dplyr::rename(pairwise_selected_model = selected_model) %>%
      dplyr::select(
        strain, analysis_mode, comparison,
        pairwise_selected_model,
        pairwise_mgcv_p_raw, pairwise_mgcv_p_adjusted,
        pairwise_mgcv_correction_used, pairwise_mgcv_passes,
        pairwise_mgcv_interpretation_adjusted
      ) %>%
      dplyr::distinct()
  } else {
    mgcv_flags <- pairwise_differences %>%
      dplyr::distinct(strain, analysis_mode, comparison) %>%
      dplyr::mutate(
        pairwise_selected_model = NA_character_,
        pairwise_mgcv_p_raw = NA_real_,
        pairwise_mgcv_p_adjusted = NA_real_,
        pairwise_mgcv_correction_used = "none",
        pairwise_mgcv_passes = TRUE,
        pairwise_mgcv_interpretation_adjusted = "No pairwise mgcv multiplicity correction needed for this mode"
      )
  }
  
  pairwise_differences %>%
    dplyr::select(
      -dplyr::any_of(c(
        "overall_p_raw", "overall_p_adjusted", "overall_correction_used", "overall_passes",
        "pairwise_global_p_raw", "pairwise_global_p_adjusted", "pairwise_correction_used",
        "passes_global", "final_pairwise_interpretation",
        "pairwise_mgcv_p_raw", "pairwise_mgcv_p_adjusted", "pairwise_mgcv_correction_used",
        "pairwise_mgcv_passes", "pairwise_selected_model",
        "significant", "show_band", "diff_plot", "lower_simul_plot", "upper_simul_plot"
      ))
    ) %>%
    dplyr::left_join(
      overall_flags,
      by = c("strain", "analysis_mode", "selected_model")
    ) %>%
    dplyr::left_join(
      pairwise_flags,
      by = c("strain", "analysis_mode", "selected_model", "prediction_type", "comparison")
    ) %>%
    dplyr::left_join(
      mgcv_flags,
      by = c("strain", "analysis_mode", "comparison")
    ) %>%
    dplyr::mutate(
      significant = overall_passes & pairwise_mgcv_passes & passes_global & significant_raw,
      show_band = significant,
      diff_plot = ifelse(show_band, diff, NA_real_),
      lower_simul_plot = ifelse(show_band, lower_simul, NA_real_),
      upper_simul_plot = ifelse(show_band, upper_simul, NA_real_),
      alpha = alpha
    ) %>%
    dplyr::select(
      strain, analysis_mode, selected_model, prediction_type,
      time, group_1, group_2, comparison,
      diff, se, crit_simul, lower_simul, upper_simul,
      diff_plot, lower_simul_plot, upper_simul_plot,
      global_p, pairwise_global_p_raw, pairwise_global_p_adjusted,
      pairwise_correction_used, passes_global,
      pairwise_selected_model,
      pairwise_mgcv_p_raw, pairwise_mgcv_p_adjusted,
      pairwise_mgcv_correction_used, pairwise_mgcv_passes,
      overall_p_raw, overall_p_adjusted,
      overall_correction_used, overall_passes,
      final_pairwise_interpretation,
      significant_raw, significant, show_band, alpha
    )
}

# ============================================================
# Load external model-selection results
# ============================================================
load_precomputed_selection <- function(aic_workbook_paths, strain_levels) {
  out <- lapply(names(aic_workbook_paths), function(mode) {
    path <- aic_workbook_paths[[mode]]
    
    if (!file.exists(path)) {
      stop("Model-selection workbook not found: ", path)
    }
    
    aic_tbl <- readxl::read_excel(path, sheet = "AIC_Table") %>%
      dplyr::mutate(
        strain = trimws(as.character(strain)),
        analysis_mode = trimws(as.character(analysis_mode)),
        model = trimws(as.character(model)),
        AIC = as.numeric(AIC),
        degree_global = as.numeric(degree_global),
        degree_fs = as.numeric(degree_fs)
      ) %>%
      dplyr::mutate(
        strain = factor(strain, levels = strain_levels)
      )
    
    selected_tbl <- readxl::read_excel(path, sheet = "Selected_Parameters") %>%
      dplyr::mutate(
        strain = trimws(as.character(strain)),
        analysis_mode = trimws(as.character(analysis_mode)),
        model = trimws(as.character(model)),
        degree_global = as.numeric(degree_global),
        degree_fs = as.numeric(degree_fs),
        selection_status = as.character(selection_status),
        degree_global_selection_status = as.character(degree_global_selection_status),
        degree_fs_selection_status = as.character(degree_fs_selection_status)
      ) %>%
      dplyr::mutate(
        strain = factor(strain, levels = strain_levels)
      )
    
    best_models <- aic_tbl %>%
      dplyr::group_by(strain, analysis_mode) %>%
      dplyr::arrange(AIC, .by_group = TRUE) %>%
      dplyr::slice(1) %>%
      dplyr::ungroup() %>%
      dplyr::rename(
        selected_model = model,
        selected_model_AIC = AIC
      )
    
    best_models_with_k <- best_models %>%
      dplyr::left_join(
        selected_tbl %>%
          dplyr::rename(selected_model = model),
        by = c("strain", "analysis_mode", "selected_model"),
        suffix = c("_from_aic", "_from_selected")
      ) %>%
      dplyr::mutate(
        selected_degree_global = dplyr::coalesce(degree_global_from_selected, degree_global_from_aic),
        selected_degree_fs = dplyr::coalesce(degree_fs_from_selected, degree_fs_from_aic)
      ) %>%
      dplyr::select(
        strain, analysis_mode, selected_model, selected_model_AIC,
        selected_degree_global, selected_degree_fs,
        selection_status, degree_global_selection_status, degree_fs_selection_status
      )
    
    list(
      AIC_Table = aic_tbl,
      Selected_Parameters = selected_tbl,
      Best_Model = best_models_with_k
    )
  })
  
  names(out) <- names(aic_workbook_paths)
  out
}

get_preselected_model_info <- function(strain_name,
                                       analysis_mode,
                                       precomputed_selection,
                                       fallback_degree_global = 12,
                                       fallback_degree_fs = 6) {
  hit <- precomputed_selection[[analysis_mode]]$Best_Model %>%
    dplyr::filter(as.character(strain) == trimws(as.character(strain_name))) %>%
    dplyr::slice(1)
  
  if (nrow(hit) == 0) {
    stop("No precomputed model-selection row found for strain ", strain_name, " in mode ", analysis_mode)
  }
  
  selected_model <- hit$selected_model[[1]]
  selected_degree_global <- hit$selected_degree_global[[1]]
  selected_degree_fs <- hit$selected_degree_fs[[1]]
  
  if (is.na(selected_degree_global)) {
    selected_degree_global <- fallback_degree_global
  }
  
  if (selected_model == "model3") {
    if (is.na(selected_degree_fs)) {
      selected_degree_fs <- fallback_degree_fs
    }
  } else {
    selected_degree_fs <- NA_real_
  }
  
  list(
    selected_model = selected_model,
    selected_model_AIC = hit$selected_model_AIC[[1]],
    degree_global = selected_degree_global,
    degree_fs = selected_degree_fs,
    selection_status = hit$selection_status[[1]],
    degree_global_selection_status = hit$degree_global_selection_status[[1]],
    degree_fs_selection_status = hit$degree_fs_selection_status[[1]]
  )
}

smooth_table_from_bam <- function(model_obj, strain_name, analysis_mode, selected_model_name) {
  sm <- summary(model_obj)$s.table
  if (is.null(sm)) return(data.frame())
  
  out <- as.data.frame(sm)
  out$term <- rownames(sm)
  rownames(out) <- NULL
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  out$selected_model <- selected_model_name
  
  out %>%
    dplyr::select(strain, analysis_mode, selected_model, term, dplyr::everything())
}

param_table_from_bam <- function(model_obj, strain_name, analysis_mode, selected_model_name) {
  pt <- summary(model_obj)$p.table
  if (is.null(pt)) return(data.frame())
  
  out <- as.data.frame(pt)
  out$term <- rownames(pt)
  rownames(out) <- NULL
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  out$selected_model <- selected_model_name
  
  out %>%
    dplyr::filter(!grepl("^Group", term)) %>%
    dplyr::select(strain, analysis_mode, selected_model, term, dplyr::everything())
}

diagnostic_summary_from_bam <- function(model_obj, strain_name, analysis_mode, selected_model_name) {
  res <- residuals(model_obj, type = "deviance")
  fit <- fitted(model_obj)
  
  data.frame(
    strain = strain_name,
    analysis_mode = analysis_mode,
    selected_model = selected_model_name,
    metric = c(
      "n_observations",
      "mean_residual",
      "sd_residual",
      "min_residual",
      "max_residual",
      "cor_fitted_abs_residual"
    ),
    value = c(
      length(res),
      mean(res, na.rm = TRUE),
      sd(res, na.rm = TRUE),
      min(res, na.rm = TRUE),
      max(res, na.rm = TRUE),
      cor(fit, abs(res), use = "complete.obs")
    ),
    stringsAsFactors = FALSE
  )
}

# ============================================================
# FIXED ACF: compute within each curve, not on concatenated series
# ============================================================
acf_table_from_bam <- function(model_obj,
                               df_used,
                               strain_name,
                               analysis_mode,
                               selected_model_name,
                               lag.max = 25,
                               residual_type = "pearson") {
  res <- try(residuals(model_obj, type = residual_type), silent = TRUE)
  
  if (inherits(res, "try-error") || length(res) != nrow(df_used)) {
    return(data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      curve = NA_character_,
      replicate = NA_character_,
      Group = NA_character_,
      lag = NA_real_,
      acf = NA_real_,
      n_points = NA_integer_,
      residual_type = residual_type,
      note = "Residuals could not be aligned with df_used",
      stringsAsFactors = FALSE
    ))
  }
  
  tmp <- df_used %>%
    dplyr::mutate(
      residual_value = as.numeric(res),
      curve = as.character(curve),
      replicate = as.character(replicate),
      Group = as.character(Group)
    ) %>%
    dplyr::arrange(curve, time)
  
  split_list <- tmp %>%
    dplyr::group_split(curve)
  
  acf_list <- purrr::map(split_list, function(d) {
    d <- d %>% dplyr::arrange(time)
    
    n_valid <- sum(!is.na(d$residual_value))
    if (n_valid < 2) return(NULL)
    
    lag_use <- min(lag.max, n_valid - 1)
    if (lag_use < 1) return(NULL)
    
    acf_obj <- try(
      stats::acf(d$residual_value, plot = FALSE, lag.max = lag_use, na.action = na.pass),
      silent = TRUE
    )
    
    if (inherits(acf_obj, "try-error")) {
      return(NULL)
    }
    
    data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      curve = d$curve[1],
      replicate = d$replicate[1],
      Group = d$Group[1],
      lag = as.numeric(acf_obj$lag),
      acf = as.numeric(acf_obj$acf),
      n_points = nrow(d),
      residual_type = residual_type,
      note = NA_character_,
      stringsAsFactors = FALSE
    )
  })
  
  out <- dplyr::bind_rows(acf_list)
  
  if (nrow(out) == 0) {
    return(data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      curve = NA_character_,
      replicate = NA_character_,
      Group = NA_character_,
      lag = NA_real_,
      acf = NA_real_,
      n_points = NA_integer_,
      residual_type = residual_type,
      note = "ACF could not be computed for any curve",
      stringsAsFactors = FALSE
    ))
  }
  
  out
}

gam_vcomp_table <- function(model_obj, strain_name, analysis_mode, selected_model_name) {
  vc <- try(mgcv::gam.vcomp(model_obj), silent = TRUE)
  
  if (inherits(vc, "try-error")) {
    return(data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      note = "gam.vcomp() could not be computed",
      stringsAsFactors = FALSE
    ))
  }
  
  out <- as.data.frame(vc)
  out$component <- rownames(out)
  rownames(out) <- NULL
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  out$selected_model <- selected_model_name
  
  out %>%
    dplyr::select(strain, analysis_mode, selected_model, component, dplyr::everything())
}

k_check_table_from_bam <- function(model_obj, strain_name, analysis_mode, selected_model_name) {
  kc <- try(mgcv::k.check(model_obj), silent = TRUE)
  
  if (inherits(kc, "try-error") || is.null(kc)) {
    return(data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      note = "k.check() could not be computed",
      stringsAsFactors = FALSE
    ))
  }
  
  out <- as.data.frame(kc)
  out$term <- rownames(kc)
  rownames(out) <- NULL
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  out$selected_model <- selected_model_name
  
  out %>%
    dplyr::select(strain, analysis_mode, selected_model, term, dplyr::everything())
}

# ============================================================
# Read workbook and build one long dataset
# ============================================================
read_one_replicate_sheet <- function(file_path, sheet_name) {
  replicate_id <- stringr::str_extract(sheet_name, "\\d+$")
  
  readxl::read_excel(file_path, sheet = sheet_name) %>%
    dplyr::rename(time = 1) %>%
    tidyr::pivot_longer(
      cols = -time,
      names_to = "raw_name",
      values_to = "OD"
    ) %>%
    dplyr::mutate(
      replicate = factor(paste0("Rep", replicate_id), levels = c("Rep1", "Rep2", "Rep3")),
      strain = stringr::str_trim(stringr::str_extract(raw_name, "^[^-]+")),
      original_group = stringr::str_trim(stringr::str_extract(raw_name, "(?<=-).+$"))
    ) %>%
    dplyr::filter(!is.na(OD)) %>%
    dplyr::mutate(
      time = as.numeric(time),
      OD = as.numeric(OD),
      original_group = factor(original_group, levels = c("Negative", "Mild", "Severe"))
    )
}

all_data <- purrr::map_dfr(replicate_sheets, ~ read_one_replicate_sheet(input_file, .x))

strain_levels <- gtools::mixedsort(unique(all_data$strain))

all_data <- all_data %>%
  dplyr::mutate(
    strain = factor(strain, levels = strain_levels)
  )

# ============================================================
# Simultaneous band helpers
# ============================================================
get_simultaneous_band <- function(Xd, beta, V, alpha = 0.05, nsim = 10000, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  fit <- as.numeric(Xd %*% beta)
  se  <- sqrt(pmax(rowSums((Xd %*% V) * Xd), 0))
  se_safe <- pmax(se, .Machine$double.eps)
  
  delta_beta <- MASS::mvrnorm(
    n = nsim,
    mu = rep(0, length(beta)),
    Sigma = V
  )
  
  sim_proc <- Xd %*% t(delta_beta)
  sim_std  <- sweep(sim_proc, 1, se_safe, FUN = "/")
  sim_sup  <- apply(abs(sim_std), 2, max)
  
  crit <- as.numeric(
    stats::quantile(sim_sup, probs = 1 - alpha, names = FALSE, type = 8)
  )
  
  obs_sup <- max(abs(fit / se_safe))
  global_p <- (sum(sim_sup >= obs_sup) + 1) / (length(sim_sup) + 1)
  
  list(fit = fit, se = se, crit = crit, global_p = global_p)
}

make_contrast_simultaneous <- function(group_a, group_b, grid_avg, Xavg, V, beta,
                                       alpha = 0.05, nsim = 10000, seed = NULL) {
  ia <- which(grid_avg$Group == group_a)
  ib <- which(grid_avg$Group == group_b)
  
  if (length(ia) != length(ib)) {
    stop("Contrast construction failed: groups do not have matching time-grid lengths.")
  }
  
  if (!isTRUE(all.equal(grid_avg$time[ia], grid_avg$time[ib], tolerance = 1e-12))) {
    stop("Contrast construction failed: time grids do not match between groups.")
  }
  
  Xa <- Xavg[ia, , drop = FALSE]
  Xb <- Xavg[ib, , drop = FALSE]
  Xd <- Xa - Xb
  
  sb <- get_simultaneous_band(
    Xd = Xd,
    beta = beta,
    V = V,
    alpha = alpha,
    nsim = nsim,
    seed = seed
  )
  
  out <- data.frame(
    time        = grid_avg$time[ia],
    group_1     = group_a,
    group_2     = group_b,
    comparison  = paste0(group_a, "-", group_b),
    diff        = sb$fit,
    se          = sb$se,
    crit_simul  = sb$crit,
    lower_simul = sb$fit - sb$crit * sb$se,
    upper_simul = sb$fit + sb$crit * sb$se,
    global_p    = sb$global_p,
    stringsAsFactors = FALSE
  )
  
  out$significant_raw <- with(out, lower_simul > 0 | upper_simul < 0)
  out
}

# ============================================================
# Mode-specific recoding
# ============================================================
recode_mode_data <- function(df, analysis_mode) {
  if (analysis_mode == "MS") {
    return(
      df %>%
        dplyr::filter(original_group %in% c("Mild", "Severe")) %>%
        dplyr::mutate(Group = factor(original_group, levels = c("Mild", "Severe")))
    )
  }
  
  if (analysis_mode == "NP") {
    return(
      df %>%
        dplyr::filter(original_group %in% c("Negative", "Mild", "Severe")) %>%
        dplyr::mutate(
          Group = dplyr::case_when(
            original_group == "Negative" ~ "Negative",
            original_group %in% c("Mild", "Severe") ~ "P"
          ),
          Group = factor(Group, levels = c("Negative", "P"))
        )
    )
  }
  
  if (analysis_mode == "NS") {
    return(
      df %>%
        dplyr::filter(original_group %in% c("Negative", "Severe")) %>%
        dplyr::mutate(Group = factor(original_group, levels = c("Negative", "Severe")))
    )
  }
  
  if (analysis_mode == "MN") {
    return(
      df %>%
        dplyr::filter(original_group %in% c("Negative", "Mild")) %>%
        dplyr::mutate(Group = factor(original_group, levels = c("Negative", "Mild")))
    )
  }
  
  if (analysis_mode == "three_groups") {
    return(
      df %>%
        dplyr::filter(original_group %in% c("Negative", "Mild", "Severe")) %>%
        dplyr::mutate(Group = factor(original_group, levels = c("Negative", "Mild", "Severe")))
    )
  }
  
  stop("Unknown analysis_mode: ", analysis_mode)
}

# ============================================================
# Fit selected model using externally selected k
# ============================================================
fit_selected_model <- function(df, selected_model_name, degree_global, degree_fs, rho_value) {
  if (selected_model_name == "model1") {
    return(
      mgcv::bam(
        OD ~ Group +
          s(time, k = degree_global) +
          s(replicate, bs = "re"),
        data = df,
        method = "fREML",
        rho = rho_value,
        AR.start = df$ar_start
      )
    )
  }
  
  if (selected_model_name == "model2") {
    return(
      mgcv::bam(
        OD ~ Group +
          s(time, k = degree_global) +
          s(time, Group, bs = "sz", k = degree_global) +
          s(replicate, bs = "re"),
        data = df,
        method = "fREML",
        rho = rho_value,
        AR.start = df$ar_start
      )
    )
  }
  
  if (selected_model_name == "model3") {
    return(
      mgcv::bam(
        OD ~ Group +
          s(time, k = degree_global) +
          s(time, Group, bs = "sz", k = degree_global) +
          s(replicate, bs = "re") +
          s(curve, bs = "re") +
          s(time, curve, bs = "fs", k = degree_fs, m = 1),
        data = df,
        method = "fREML",
        rho = rho_value,
        AR.start = df$ar_start
      )
    )
  }
  
  stop("Unknown selected_model_name: ", selected_model_name)
}

# ============================================================
# Population-averaged predictions with SD + CI
# ============================================================
build_population_prediction_grid <- function(df, time_grid) {
  unit_df <- df %>%
    dplyr::distinct(Group, replicate, curve) %>%
    dplyr::arrange(Group, replicate, curve)
  
  pred_grid_full <- tidyr::expand_grid(
    time = time_grid,
    unit_row = seq_len(nrow(unit_df))
  ) %>%
    dplyr::left_join(
      unit_df %>% dplyr::mutate(unit_row = seq_len(dplyr::n())),
      by = "unit_row"
    ) %>%
    dplyr::select(time, Group, replicate, curve)
  
  pred_grid_full$Group <- factor(pred_grid_full$Group, levels = levels(df$Group))
  pred_grid_full$replicate <- factor(pred_grid_full$replicate, levels = levels(df$replicate))
  pred_grid_full$curve <- factor(pred_grid_full$curve, levels = levels(df$curve))
  
  pred_grid_full
}

aggregate_lpmatrix_by_group_time <- function(pred_grid_full, Xfull) {
  key_df <- pred_grid_full %>%
    dplyr::mutate(row_id = seq_len(n())) %>%
    dplyr::group_by(Group, time) %>%
    dplyr::summarise(
      row_ids = list(row_id),
      n_units = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::arrange(Group, time)
  
  Xavg <- t(vapply(
    key_df$row_ids,
    function(idx) colMeans(Xfull[idx, , drop = FALSE]),
    numeric(ncol(Xfull))
  ))
  
  grid_avg <- key_df %>%
    dplyr::select(time, Group, n_units)
  
  list(grid_avg = grid_avg, Xavg = Xavg)
}

get_population_average_predictions <- function(model_obj, df, time_grid) {
  pred_grid_full <- build_population_prediction_grid(df = df, time_grid = time_grid)
  
  Xfull <- predict(
    model_obj,
    newdata = pred_grid_full,
    type = "lpmatrix"
  )
  
  beta <- stats::coef(model_obj)
  Vp   <- stats::vcov(model_obj)
  
  fit_full <- as.numeric(Xfull %*% beta)
  pred_grid_full$fit_full <- fit_full
  
  pred_summary <- pred_grid_full %>%
    dplyr::group_by(time, Group) %>%
    dplyr::summarise(
      n_units = dplyr::n(),
      fit_mean = mean(fit_full, na.rm = TRUE),
      fit_sd = stats::sd(fit_full, na.rm = TRUE),
      fit_min = min(fit_full, na.rm = TRUE),
      fit_max = max(fit_full, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(Group, time)
  
  agg <- aggregate_lpmatrix_by_group_time(
    pred_grid_full = pred_grid_full,
    Xfull = Xfull
  )
  
  grid_avg <- agg$grid_avg
  Xavg <- agg$Xavg
  
  fit_avg <- as.numeric(Xavg %*% beta)
  se_avg  <- sqrt(pmax(rowSums((Xavg %*% Vp) * Xavg), 0))
  
  pred_curves <- pred_summary %>%
    dplyr::mutate(
      fit = fit_avg,
      se = se_avg,
      lower_ci = fit - 1.96 * se,
      upper_ci = fit + 1.96 * se,
      lower_sd = fit_mean - fit_sd,
      upper_sd = fit_mean + fit_sd
    ) %>%
    dplyr::select(
      time, Group, n_units,
      fit, se, lower_ci, upper_ci,
      fit_mean, fit_sd, lower_sd, upper_sd,
      fit_min, fit_max
    )
  
  list(
    pred_grid_full = pred_grid_full,
    Xfull = Xfull,
    grid_avg = grid_avg,
    Xavg = Xavg,
    beta = beta,
    Vp = Vp,
    pred_curves = pred_curves
  )
}

# ============================================================
# Pairwise mgcv smooth-term tests
# For three_groups only: each pair uses its own preselected model
# ============================================================
pair_to_analysis_mode <- function(group_1, group_2) {
  pair_sorted <- sort(c(as.character(group_1), as.character(group_2)))
  
  if (identical(pair_sorted, c("Mild", "Severe"))) {
    return("MS")
  }
  
  if (identical(pair_sorted, c("Negative", "Severe"))) {
    return("NS")
  }
  
  if (identical(pair_sorted, c("Mild", "Negative"))) {
    return("MN")
  }
  
  stop("No pairwise analysis mode defined for pair: ", paste(pair_sorted, collapse = " vs "))
}

pairwise_mgcv_smooth_tests <- function(df_strain_original,
                                       strain_name,
                                       precomputed_selection,
                                       rho_value,
                                       fallback_degree_global = 12,
                                       fallback_degree_fs = 6) {
  available_groups <- sort(unique(as.character(stats::na.omit(df_strain_original$original_group))))
  
  if (!all(c("Negative", "Mild", "Severe") %in% available_groups)) {
    return(data.frame())
  }
  
  group_pairs <- list(
    c("Mild", "Severe"),
    c("Negative", "Severe"),
    c("Negative", "Mild")
  )
  
  out_list <- lapply(group_pairs, function(pair) {
    pair_mode <- pair_to_analysis_mode(pair[1], pair[2])
    
    pair_df <- df_strain_original %>%
      recode_mode_data(analysis_mode = pair_mode) %>%
      dplyr::filter(!is.na(OD), !is.na(time), !is.na(Group), !is.na(replicate)) %>%
      dplyr::mutate(
        Group = droplevels(Group),
        replicate = droplevels(replicate),
        curve = interaction(replicate, Group, drop = TRUE)
      ) %>%
      dplyr::arrange(curve, time)
    
    if (nrow(pair_df) == 0 || nlevels(pair_df$Group) < 2) {
      return(NULL)
    }
    
    common_max_time <- pair_df %>%
      dplyr::group_by(curve) %>%
      dplyr::summarise(max_time = max(time, na.rm = TRUE), .groups = "drop") %>%
      dplyr::summarise(common_end = min(max_time)) %>%
      dplyr::pull(common_end)
    
    pair_df <- pair_df %>%
      dplyr::filter(time <= common_max_time) %>%
      dplyr::arrange(curve, time) %>%
      dplyr::group_by(curve) %>%
      dplyr::mutate(ar_start = dplyr::row_number() == 1) %>%
      dplyr::ungroup() %>%
      dplyr::mutate(
        Group = droplevels(Group),
        replicate = droplevels(replicate),
        curve = droplevels(curve)
      )
    
    selected_info_pair <- get_preselected_model_info(
      strain_name = strain_name,
      analysis_mode = pair_mode,
      precomputed_selection = precomputed_selection,
      fallback_degree_global = fallback_degree_global,
      fallback_degree_fs = fallback_degree_fs
    )
    
    pair_model <- fit_selected_model(
      df = pair_df,
      selected_model_name = selected_info_pair$selected_model,
      degree_global = selected_info_pair$degree_global,
      degree_fs = selected_info_pair$degree_fs,
      rho_value = rho_value
    )
    
    sm <- smooth_table_from_bam(
      model_obj = pair_model,
      strain_name = strain_name,
      analysis_mode = "three_groups",
      selected_model_name = selected_info_pair$selected_model
    )
    
    lv <- levels(pair_df$Group)
    
    sm %>%
      dplyr::filter(grepl("^s\\(time,Group\\)", term)) %>%
      dplyr::mutate(
        pairwise_mode = pair_mode,
        group_1 = lv[1],
        group_2 = lv[2],
        comparison = paste0(lv[1], "-", lv[2]),
        pair_selected_model_AIC = selected_info_pair$selected_model_AIC,
        pair_degree_global = selected_info_pair$degree_global,
        pair_degree_fs = selected_info_pair$degree_fs,
        pair_selection_status = selected_info_pair$selection_status,
        pair_degree_global_selection_status = selected_info_pair$degree_global_selection_status,
        pair_degree_fs_selection_status = selected_info_pair$degree_fs_selection_status,
        interpretation = "Pairwise BAM smooth-term test fitted with the pair-specific preselected model from the corresponding BAM_<mode>_all_strains_model_comparison workbook"
      ) %>%
      dplyr::select(
        strain, analysis_mode, selected_model,
        pairwise_mode, comparison, group_1, group_2,
        pair_selected_model_AIC,
        pair_degree_global, pair_degree_fs,
        pair_selection_status,
        pair_degree_global_selection_status,
        pair_degree_fs_selection_status,
        interpretation,
        dplyr::everything()
      )
  })
  
  safe_bind_rows(out_list)
}

# ============================================================
# Build final significance windows
# ============================================================
build_significant_windows_from_pairwise <- function(pairwise_diffs) {
  if (nrow(pairwise_diffs) == 0 || !"significant" %in% names(pairwise_diffs)) {
    return(empty_sig_windows())
  }
  
  sig_rows <- pairwise_diffs %>%
    dplyr::filter(significant)
  
  if (nrow(sig_rows) == 0) {
    return(empty_sig_windows())
  }
  
  median_time_step <- pairwise_diffs %>%
    dplyr::distinct(time) %>%
    dplyr::arrange(time) %>%
    dplyr::pull(time) %>%
    diff() %>%
    stats::median(na.rm = TRUE)
  
  if (!is.finite(median_time_step)) median_time_step <- 0
  
  sig_source <- pairwise_diffs %>%
    dplyr::arrange(strain, analysis_mode, selected_model, prediction_type, comparison, time) %>%
    dplyr::group_by(strain, analysis_mode, selected_model, prediction_type, comparison) %>%
    dplyr::mutate(
      dt = time - dplyr::lag(time),
      dt = ifelse(is.na(dt), median_time_step, dt),
      new_block = dplyr::case_when(
        !significant ~ 0,
        is.na(dplyr::lag(significant)) ~ 1,
        !dplyr::lag(significant) & significant ~ 1,
        significant & (time - dplyr::lag(time) > 1.5 * median_time_step) ~ 1,
        TRUE ~ 0
      ),
      block_id = cumsum(new_block * significant)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(significant)
  
  if (nrow(sig_source) == 0) {
    return(empty_sig_windows())
  }
  
  sig_source %>%
    dplyr::group_by(strain, analysis_mode, selected_model, prediction_type, comparison, block_id) %>%
    dplyr::summarise(
      start_time = min(time),
      end_time = max(time),
      duration = end_time - start_time,
      mean_diff = mean(diff, na.rm = TRUE),
      max_abs_diff = max(abs(diff), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(strain, comparison, start_time)
}

build_all_groups_different_windows <- function(pairwise_diffs) {
  if (nrow(pairwise_diffs) == 0 || !"significant" %in% names(pairwise_diffs)) {
    return(empty_all_groups_windows())
  }
  
  out_list <- list()
  
  split_keys <- pairwise_diffs %>%
    dplyr::distinct(strain, analysis_mode, selected_model, prediction_type)
  
  for (i in seq_len(nrow(split_keys))) {
    key <- split_keys[i, ]
    
    df_sub <- pairwise_diffs %>%
      dplyr::filter(
        strain == key$strain,
        analysis_mode == key$analysis_mode,
        selected_model == key$selected_model,
        prediction_type == key$prediction_type
      )
    
    if (unique(as.character(df_sub$analysis_mode)) != "three_groups") next
    
    n_pairs <- df_sub %>% dplyr::distinct(comparison) %>% nrow()
    if (n_pairs != 3) next
    
    pairwise_signif_wide <- df_sub %>%
      dplyr::select(time, comparison, significant) %>%
      dplyr::distinct() %>%
      tidyr::pivot_wider(
        names_from = comparison,
        values_from = significant,
        values_fill = FALSE
      ) %>%
      dplyr::arrange(time)
    
    pairwise_cols <- setdiff(names(pairwise_signif_wide), "time")
    if (length(pairwise_cols) != 3) next
    
    all_groups_different_df <- pairwise_signif_wide %>%
      dplyr::mutate(
        all_pairwise_significant = apply(
          dplyr::select(., dplyr::all_of(pairwise_cols)),
          1,
          all
        )
      )
    
    if (!any(all_groups_different_df$all_pairwise_significant)) next
    
    median_time_step <- stats::median(diff(sort(unique(all_groups_different_df$time))), na.rm = TRUE)
    if (!is.finite(median_time_step)) median_time_step <- 0
    
    all_groups_source <- all_groups_different_df %>%
      dplyr::arrange(time) %>%
      dplyr::mutate(
        dt = time - dplyr::lag(time),
        dt = ifelse(is.na(dt), median_time_step, dt),
        new_block = dplyr::case_when(
          !all_pairwise_significant ~ 0,
          is.na(dplyr::lag(all_pairwise_significant)) ~ 1,
          !dplyr::lag(all_pairwise_significant) & all_pairwise_significant ~ 1,
          all_pairwise_significant & (time - dplyr::lag(time) > 1.5 * median_time_step) ~ 1,
          TRUE ~ 0
        ),
        block_id = cumsum(new_block * all_pairwise_significant)
      ) %>%
      dplyr::filter(all_pairwise_significant)
    
    if (nrow(all_groups_source) == 0) next
    
    out_list[[length(out_list) + 1]] <- all_groups_source %>%
      dplyr::group_by(block_id) %>%
      dplyr::summarise(
        start_time = min(time),
        end_time = max(time),
        duration = end_time - start_time,
        .groups = "drop"
      ) %>%
      dplyr::mutate(
        strain = as.character(key$strain),
        analysis_mode = as.character(key$analysis_mode),
        selected_model = as.character(key$selected_model),
        prediction_type = as.character(key$prediction_type),
        interpretation = "Time window where all three pairwise contrasts are significant simultaneously after workbook-level multiplicity correction"
      ) %>%
      dplyr::select(
        strain, analysis_mode, selected_model, prediction_type,
        interpretation, block_id, start_time, end_time, duration
      )
  }
  
  safe_bind_rows(out_list)
}

# ============================================================
# Run one strain + one mode
# ============================================================
run_one_strain_analysis <- function(df_strain,
                                    strain_name,
                                    analysis_mode,
                                    precomputed_selection,
                                    alpha = 0.05,
                                    nsim_simultaneous = 10000,
                                    random_seed = 123,
                                    pairwise_correction_requested = "BH",
                                    rho_value = 0,
                                    fallback_degree_global = 12,
                                    fallback_degree_fs = 6) {
  
  if (!pairwise_correction_requested %in% c("none", "holm", "bonferroni", "BH")) {
    stop("pairwise_correction_requested must be one of: 'none', 'holm', 'bonferroni', 'BH'")
  }
  
  df <- df_strain %>%
    recode_mode_data(analysis_mode = analysis_mode) %>%
    dplyr::filter(!is.na(OD), !is.na(time), !is.na(Group), !is.na(replicate)) %>%
    dplyr::mutate(
      Group = droplevels(Group),
      replicate = droplevels(replicate),
      curve = interaction(replicate, Group, drop = TRUE)
    ) %>%
    dplyr::arrange(curve, time)
  
  if (nrow(df) == 0) stop("No usable rows after filtering.")
  if (dplyr::n_distinct(df$Group) < 2) stop("Fewer than 2 groups present.")
  if (dplyr::n_distinct(df$replicate) < 2) stop("Fewer than 2 replicates present.")
  if (dplyr::n_distinct(df$curve) < 2) stop("Fewer than 2 curves present.")
  
  common_max_time <- df %>%
    dplyr::group_by(curve) %>%
    dplyr::summarise(max_time = max(time, na.rm = TRUE), .groups = "drop") %>%
    dplyr::summarise(common_end = min(max_time)) %>%
    dplyr::pull(common_end)
  
  df <- df %>%
    dplyr::filter(time <= common_max_time) %>%
    dplyr::arrange(curve, time) %>%
    dplyr::group_by(curve) %>%
    dplyr::mutate(ar_start = dplyr::row_number() == 1) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      Group = droplevels(Group),
      replicate = droplevels(replicate),
      curve = droplevels(curve)
    )
  
  selected_info <- get_preselected_model_info(
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    precomputed_selection = precomputed_selection,
    fallback_degree_global = fallback_degree_global,
    fallback_degree_fs = fallback_degree_fs
  )
  
  selected_model_name <- selected_info$selected_model
  degree_global <- selected_info$degree_global
  degree_fs <- selected_info$degree_fs
  
  selected_model <- fit_selected_model(
    df = df,
    selected_model_name = selected_model_name,
    degree_global = degree_global,
    degree_fs = degree_fs,
    rho_value = rho_value
  )
  
  aic_table <- precomputed_selection[[analysis_mode]]$AIC_Table %>%
    dplyr::filter(as.character(strain) == as.character(strain_name)) %>%
    dplyr::mutate(
      analysis_mode = analysis_mode,
      selected_from_external_table = model == selected_model_name
    ) %>%
    dplyr::select(strain, analysis_mode, dplyr::everything())
  
  selected_parametric_terms <- param_table_from_bam(
    model_obj = selected_model,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    selected_model_name = selected_model_name
  )
  
  selected_smooth_terms <- smooth_table_from_bam(
    model_obj = selected_model,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    selected_model_name = selected_model_name
  )
  
  overall_global_test <- selected_smooth_terms %>%
    dplyr::filter(grepl("^s\\(time,Group\\)", term)) %>%
    dplyr::mutate(
      interpretation = "Overall test of whether group trajectories differ over time"
    ) %>%
    dplyr::select(strain, analysis_mode, selected_model, interpretation, term, dplyr::everything())
  
  overall_p_value <- NA_real_
  pcol <- intersect(c("p-value", "p.value", "p_value"), names(overall_global_test))
  
  if (nrow(overall_global_test) > 0 && length(pcol) > 0) {
    overall_p_value <- as.numeric(overall_global_test[[pcol[1]]][1])
  }
  
  time_grid <- seq(min(df$time), max(df$time), length.out = 300)
  
  pop_pred <- get_population_average_predictions(
    model_obj = selected_model,
    df = df,
    time_grid = time_grid
  )
  
  pred_curves <- pop_pred$pred_curves %>%
    dplyr::mutate(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      prediction_type = "population_averaged"
    ) %>%
    dplyr::select(
      strain, analysis_mode, selected_model, prediction_type,
      time, Group, n_units,
      fit, se, lower_ci, upper_ci,
      fit_mean, fit_sd, lower_sd, upper_sd,
      fit_min, fit_max
    )
  
  beta <- pop_pred$beta
  Vp   <- pop_pred$Vp
  grid_avg <- pop_pred$grid_avg
  Xavg <- pop_pred$Xavg
  
  group_levels <- levels(df$Group)
  group_pairs  <- combn(group_levels, 2, simplify = FALSE)
  
  pairwise_list <- lapply(
    seq_along(group_pairs),
    function(i) {
      pair <- group_pairs[[i]]
      make_contrast_simultaneous(
        group_a = pair[1],
        group_b = pair[2],
        grid_avg = grid_avg,
        Xavg = Xavg,
        V = Vp,
        beta = beta,
        alpha = alpha,
        nsim = nsim_simultaneous,
        seed = random_seed + i
      )
    }
  )
  
  pairwise_diffs <- dplyr::bind_rows(pairwise_list) %>%
    dplyr::mutate(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      prediction_type = "population_averaged"
    ) %>%
    dplyr::select(strain, analysis_mode, selected_model, prediction_type, dplyr::everything())
  
  pairwise_global_tests <- pairwise_diffs %>%
    dplyr::distinct(
      strain, analysis_mode, selected_model, prediction_type,
      comparison, group_1, group_2, global_p
    ) %>%
    dplyr::arrange(comparison) %>%
    dplyr::mutate(
      interpretation = "Pairwise global test from simultaneous-band contrast inference",
      pairwise_global_p_raw = global_p,
      pairwise_global_p_adjusted = NA_real_,
      pairwise_correction_used = NA_character_,
      passes_global = NA,
      final_pairwise_interpretation = NA_character_
    ) %>%
    dplyr::select(
      strain, analysis_mode, selected_model, prediction_type,
      comparison, group_1, group_2, interpretation,
      global_p, pairwise_global_p_raw, pairwise_global_p_adjusted,
      pairwise_correction_used, passes_global, final_pairwise_interpretation
    )
  
  if (analysis_mode == "three_groups" && length(group_levels) == 3) {
    pairwise_global_tests_mgcv <- pairwise_mgcv_smooth_tests(
      df_strain_original = df_strain,
      strain_name = strain_name,
      precomputed_selection = precomputed_selection,
      rho_value = rho_value,
      fallback_degree_global = fallback_degree_global,
      fallback_degree_fs = fallback_degree_fs
    ) %>%
      dplyr::mutate(
        pairwise_mgcv_p_raw = NA_real_,
        pairwise_mgcv_p_adjusted = NA_real_,
        pairwise_mgcv_correction_used = NA_character_,
        pairwise_mgcv_passes = NA,
        pairwise_mgcv_interpretation_adjusted = NA_character_
      )
  } else {
    pairwise_global_tests_mgcv <- data.frame()
  }
  
  pairwise_diffs <- pairwise_diffs %>%
    dplyr::mutate(
      overall_p_raw = overall_p_value,
      overall_p_adjusted = NA_real_,
      overall_correction_used = NA_character_,
      overall_passes = NA,
      pairwise_global_p_raw = global_p,
      pairwise_global_p_adjusted = NA_real_,
      pairwise_correction_used = NA_character_,
      passes_global = NA,
      pairwise_mgcv_p_raw = NA_real_,
      pairwise_mgcv_p_adjusted = NA_real_,
      pairwise_mgcv_correction_used = NA_character_,
      pairwise_mgcv_passes = NA,
      final_pairwise_interpretation = NA_character_,
      significant = NA,
      show_band = NA,
      diff_plot = NA_real_,
      lower_simul_plot = NA_real_,
      upper_simul_plot = NA_real_
    ) %>%
    dplyr::select(
      strain, analysis_mode, selected_model, prediction_type,
      time, group_1, group_2, comparison,
      diff, se, crit_simul, lower_simul, upper_simul,
      diff_plot, lower_simul_plot, upper_simul_plot,
      global_p, pairwise_global_p_raw, pairwise_global_p_adjusted,
      pairwise_correction_used, passes_global,
      pairwise_mgcv_p_raw, pairwise_mgcv_p_adjusted,
      pairwise_mgcv_correction_used, pairwise_mgcv_passes,
      overall_p_raw, overall_p_adjusted,
      overall_correction_used, overall_passes,
      final_pairwise_interpretation,
      significant_raw, significant, show_band
    )
  
  selected_gam_summary_text <- text_sheet(summary(selected_model))
  
  diagnostics_gam_check <- try(text_sheet(mgcv::gam.check(selected_model)), silent = TRUE)
  if (inherits(diagnostics_gam_check, "try-error")) {
    diagnostics_gam_check <- data.frame(
      output = "gam.check() could not be computed",
      stringsAsFactors = FALSE
    )
  }
  
  selected_k_check <- k_check_table_from_bam(
    model_obj = selected_model,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    selected_model_name = selected_model_name
  )
  
  concurvity_obj <- try(mgcv::concurvity(selected_model, full = TRUE), silent = TRUE)
  if (inherits(concurvity_obj, "try-error")) {
    diagnostics_concurvity <- data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      selected_model = selected_model_name,
      note = "concurvity() could not be computed",
      stringsAsFactors = FALSE
    )
  } else {
    diagnostics_concurvity <- as.data.frame(concurvity_obj)
    diagnostics_concurvity$measure <- rownames(diagnostics_concurvity)
    rownames(diagnostics_concurvity) <- NULL
    diagnostics_concurvity$strain <- strain_name
    diagnostics_concurvity$analysis_mode <- analysis_mode
    diagnostics_concurvity$selected_model <- selected_model_name
    diagnostics_concurvity <- diagnostics_concurvity %>%
      dplyr::select(strain, analysis_mode, selected_model, measure, dplyr::everything())
  }
  
  diagnostics_summary <- diagnostic_summary_from_bam(
    model_obj = selected_model,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    selected_model_name = selected_model_name
  )
  
  residual_acf <- acf_table_from_bam(
    model_obj = selected_model,
    df_used = df,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    selected_model_name = selected_model_name,
    lag.max = 25,
    residual_type = "pearson"
  )
  
  variance_components <- gam_vcomp_table(
    model_obj = selected_model,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    selected_model_name = selected_model_name
  )
  
  run_info <- data.frame(
    strain = strain_name,
    analysis_mode = analysis_mode,
    selected_model = selected_model_name,
    prediction_type = "population_averaged",
    selected_model_AIC = selected_info$selected_model_AIC,
    degree_global = degree_global,
    degree_fs = degree_fs,
    selection_status = selected_info$selection_status,
    degree_global_selection_status = selected_info$degree_global_selection_status,
    degree_fs_selection_status = selected_info$degree_fs_selection_status,
    alpha = alpha,
    nsim_simultaneous = nsim_simultaneous,
    overall_correction_requested = overall_correction_requested,
    pairwise_correction_requested = pairwise_correction_requested,
    rho_value = rho_value,
    common_max_time = common_max_time,
    overall_p_raw = overall_p_value,
    overall_p_adjusted = NA_real_,
    overall_correction_used = NA_character_,
    overall_passes = NA,
    n_rows = nrow(df),
    n_groups = nlevels(df$Group),
    groups = paste(levels(df$Group), collapse = ", "),
    n_replicates = dplyr::n_distinct(df$replicate),
    n_curves = dplyr::n_distinct(df$curve),
    stringsAsFactors = FALSE
  )
  
  list(
    Run_Info = run_info,
    AIC_Table = aic_table,
    Overall_Global_Test = overall_global_test,
    Selected_Parametric_Terms = selected_parametric_terms,
    Selected_Smooth_Terms = selected_smooth_terms,
    Predicted_Curves = pred_curves,
    Pairwise_Global_Tests = pairwise_global_tests,
    Pairwise_Global_Tests_mgcv = pairwise_global_tests_mgcv,
    Pairwise_Differences = pairwise_diffs,
    Significant_Windows = empty_sig_windows(),
    All_Groups_Different_Windows = empty_all_groups_windows(),
    Diagnostics_Summary = diagnostics_summary,
    Residual_ACF = residual_acf,
    Diagnostics_GAM_Check = diagnostics_gam_check,
    Diagnostics_Concurvity = diagnostics_concurvity,
    Selected_GAM_Summary = selected_gam_summary_text,
    Selected_Variance_Components = variance_components,
    Selected_K_Check = selected_k_check
  )
}

safe_run_one_strain_analysis <- function(...) {
  tryCatch(
    {
      out <- run_one_strain_analysis(...)
      list(success = TRUE, result = out, error_message = NA_character_)
    },
    error = function(e) {
      list(success = FALSE, result = NULL, error_message = conditionMessage(e))
    }
  )
}

# ============================================================
# Build workbook for one mode
# ============================================================
build_mode_workbook <- function(all_data,
                                analysis_mode,
                                precomputed_selection,
                                output_folder,
                                alpha = 0.05,
                                nsim_simultaneous = 10000,
                                random_seed = 123,
                                pairwise_correction_requested = "BH",
                                overall_correction_requested = "BH",
                                rho_value = 0,
                                fallback_degree_global = 12,
                                fallback_degree_fs = 6) {
  
  all_results <- list(
    Run_Info = list(),
    AIC_Table = list(),
    Overall_Global_Test = list(),
    Selected_Parametric_Terms = list(),
    Selected_Smooth_Terms = list(),
    Predicted_Curves = list(),
    Pairwise_Global_Tests = list(),
    Pairwise_Global_Tests_mgcv = list(),
    Pairwise_Differences = list(),
    Significant_Windows = list(),
    All_Groups_Different_Windows = list(),
    Diagnostics_Summary = list(),
    Residual_ACF = list(),
    Diagnostics_GAM_Check = list(),
    Diagnostics_Concurvity = list(),
    Selected_GAM_Summary = list(),
    Selected_Variance_Components = list(),
    Selected_K_Check = list()
  )
  
  run_log <- list()
  strain_list <- levels(all_data$strain)
  
  for (i in seq_along(strain_list)) {
    strain_i <- strain_list[i]
    message("Running strain ", i, " of ", length(strain_list), ": ", strain_i, " | mode = ", analysis_mode)
    
    df_strain <- all_data %>%
      dplyr::filter(strain == strain_i)
    
    res <- safe_run_one_strain_analysis(
      df_strain = df_strain,
      strain_name = strain_i,
      analysis_mode = analysis_mode,
      precomputed_selection = precomputed_selection,
      alpha = alpha,
      nsim_simultaneous = nsim_simultaneous,
      random_seed = random_seed,
      pairwise_correction_requested = pairwise_correction_requested,
      rho_value = rho_value,
      fallback_degree_global = fallback_degree_global,
      fallback_degree_fs = fallback_degree_fs
    )
    
    run_log[[i]] <- data.frame(
      strain = strain_i,
      analysis_mode = analysis_mode,
      status = ifelse(res$success, "success", "error"),
      error_message = res$error_message,
      stringsAsFactors = FALSE
    )
    
    if (res$success) {
      for (nm in names(all_results)) {
        all_results[[nm]][[length(all_results[[nm]]) + 1]] <- res$result[[nm]]
      }
    }
  }
  
  final_sheets <- lapply(all_results, safe_bind_rows)
  
  # ------------------------------------------------------------
  # Workbook-level multiplicity correction across all strains
  # For three_groups pairwise tables, this spans all strains
  # and all three pairwise modes together within that workbook.
  # ------------------------------------------------------------
  overall_corrected <- apply_workbook_level_overall_correction(
    overall_global_test = final_sheets$Overall_Global_Test,
    run_info = final_sheets$Run_Info,
    alpha = alpha,
    method = overall_correction_requested
  )
  
  final_sheets$Overall_Global_Test <- overall_corrected$Overall_Global_Test
  final_sheets$Run_Info <- overall_corrected$Run_Info
  
  final_sheets$Pairwise_Global_Tests <- apply_workbook_level_pairwise_global_correction(
    pairwise_global_tests = final_sheets$Pairwise_Global_Tests,
    alpha = alpha,
    method = pairwise_correction_requested
  )
  
  final_sheets$Pairwise_Global_Tests_mgcv <- apply_workbook_level_pairwise_mgcv_correction(
    pairwise_global_tests_mgcv = final_sheets$Pairwise_Global_Tests_mgcv,
    alpha = alpha,
    method = pairwise_correction_requested
  )
  
  final_sheets$Pairwise_Differences <- apply_final_significance_logic(
    pairwise_differences = final_sheets$Pairwise_Differences,
    overall_global_test = final_sheets$Overall_Global_Test,
    pairwise_global_tests = final_sheets$Pairwise_Global_Tests,
    pairwise_global_tests_mgcv = final_sheets$Pairwise_Global_Tests_mgcv,
    alpha = alpha
  )
  
  if ("Pairwise_Differences" %in% names(final_sheets) &&
      nrow(final_sheets$Pairwise_Differences) > 0) {
    
    final_sheets$Significant_Windows <- build_significant_windows_from_pairwise(
      final_sheets$Pairwise_Differences
    )
    
    final_sheets$All_Groups_Different_Windows <- build_all_groups_different_windows(
      final_sheets$Pairwise_Differences
    )
  } else {
    final_sheets$Significant_Windows <- empty_sig_windows()
    final_sheets$All_Groups_Different_Windows <- empty_all_groups_windows()
  }
  
  final_sheets <- lapply(final_sheets, function(df) {
    if ("strain" %in% names(df)) {
      df %>%
        dplyr::mutate(strain = factor(as.character(strain), levels = levels(all_data$strain))) %>%
        dplyr::arrange(strain)
    } else {
      df
    }
  })
  
  run_log_df <- dplyr::bind_rows(run_log) %>%
    dplyr::mutate(strain = factor(as.character(strain), levels = levels(all_data$strain))) %>%
    dplyr::arrange(strain)
  
  about_sheet <- data.frame(
    sheet_name = c(
      "About_Sheets",
      "Run_Info",
      "AIC_Table",
      "Overall_Global_Test",
      "Selected_Parametric_Terms",
      "Selected_Smooth_Terms",
      "Predicted_Curves",
      "Pairwise_Global_Tests",
      "Pairwise_Global_Tests_mgcv",
      "Pairwise_Differences",
      "Significant_Windows",
      "All_Groups_Different_Windows",
      "Diagnostics_Summary",
      "Residual_ACF",
      "Diagnostics_GAM_Check",
      "Diagnostics_Concurvity",
      "Selected_GAM_Summary",
      "Selected_Variance_Components",
      "Selected_K_Check",
      "Run_Log"
    ),
    description = c(
      paste("Workbook for selected BAM inference by strain. Mode:", analysis_mode),
      "Run settings and externally selected best model for each strain; includes workbook-level corrected overall p-values across all strains in this workbook.",
      "External AIC rows used for each strain, annotated with the selected model.",
      "Overall test of whether group trajectories differ over time in the selected model. Workbook-level multiplicity correction is applied across all strains in this workbook.",
      "Selected-model parametric terms after removing Group rows.",
      "Selected-model smooth terms.",
      "Population-averaged group curves obtained from full-model predictions averaged over observed replicate/curve units within each group; includes CI and SD bands.",
      "One global test per pairwise contrast from simultaneous-band inference. Workbook-level multiplicity correction is applied within this workbook only. In three_groups, this spans all strains and all three pairwise modes together within this workbook.",
      "Pairwise BAM smooth-term tests. For three_groups, each pair uses its own preselected model from the corresponding pairwise BAM_<mode>_all_strains_model_comparison workbook. Workbook-level multiplicity correction is also applied within this workbook only.",
      "Pairwise difference curves with simultaneous confidence bands, based on population-averaged predictions. Final significance requires: overall_passes & pairwise_mgcv_passes & passes_global & significant_raw. All decision thresholds use <= alpha.",
      "Collapsed significant time windows from final simultaneous-band inference after workbook-level multiplicity correction.",
      "Time windows where all three pairwise contrasts are significant simultaneously after workbook-level multiplicity correction; relevant for three_groups only.",
      "Compact residual diagnostics for the selected model.",
      "Residual ACF computed separately within each curve for the selected model.",
      "Plain-text gam.check() for the selected model.",
      "Concurvity diagnostics for the selected model.",
      "Plain-text summary() for the selected model.",
      "Variance components from gam.vcomp() for the selected model.",
      "k.check() table for the selected model; use this to assess whether any smooth basis dimension k looks too small.",
      "Success/error log for strain-wise processing."
    ),
    stringsAsFactors = FALSE
  )
  
  workbook <- c(
    list(About_Sheets = about_sheet),
    final_sheets,
    list(Run_Log = run_log_df)
  )
  
  output_file <- file.path(
    output_folder,
    paste0("BAM_", analysis_mode, "_all_strains.xlsx")
  )
  
  writexl::write_xlsx(workbook, path = output_file)
  output_file
}

# ============================================================
# Load precomputed model-selection tables
# ============================================================
precomputed_selection <- load_precomputed_selection(
  aic_workbook_paths = aic_workbook_paths,
  strain_levels = levels(all_data$strain)
)

# ============================================================
# Run all modes
# ============================================================
output_files <- lapply(
  analysis_modes,
  function(mode) {
    build_mode_workbook(
      all_data = all_data,
      analysis_mode = mode,
      precomputed_selection = precomputed_selection,
      output_folder = output_folder,
      alpha = alpha,
      nsim_simultaneous = nsim_simultaneous,
      random_seed = random_seed,
      pairwise_correction_requested = pairwise_correction_requested,
      overall_correction_requested = overall_correction_requested,
      rho_value = rho_value,
      fallback_degree_global = fallback_degree_global,
      fallback_degree_fs = fallback_degree_fs
    )
  }
)

print(output_files)
cat("Done.\n")