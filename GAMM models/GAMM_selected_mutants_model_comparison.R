library(readxl)
library(writexl)
library(dplyr)
library(tidyr)
library(stringr)
library(mgcv)
library(tibble)
library(purrr)
library(gtools)
library(future)
library(future.apply)
library(progressr)

# Keep inner math libraries single-threaded
# so the outer future parallelism is the main parallel layer.
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(1)
  RhpcBLASctl::omp_set_num_threads(1)
}

# ============================================================
# Settings
# ============================================================
data_folder <- "time_series_selected"
input_file <- file.path(data_folder, "Summary - Pools.xlsx")
output_folder <- file.path(data_folder, "bam_model_selection")
dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)

analysis_modes <- c("MS", "NS", "MN", "NP", "three_groups")

# Parallel settings
workers <- 6

# Progress bar
progressr::handlers(global = TRUE)
progressr::handlers("progress")

# Candidate ranges
degree_global_candidates <- 3:20
degree_fs_patient_candidates <- 3:20
degree_fs_curve_candidates <- 3:20

# Constraints: fs k cannot exceed 150% of global k
fs_patient_to_global_ratio <- 1.5
fs_curve_to_global_ratio   <- 1.5

# k.check acceptance rule for GLOBAL smooths only
k_index_threshold <- 0.90
p_value_threshold <- 0.05

# Fallbacks
fallback_degree_global <- 20
fallback_degree_fs_patient <- 20
fallback_degree_fs_curve <- 20

# Additional rules for fs tuning
fs_aic_tolerance <- 2
fs_edf_ratio_threshold <- 0.90

rho_value <- 0

# Screening k values used while selecting global k
degree_fs_patient_for_model2_global_screening <- 6
degree_fs_patient_for_model3_global_screening <- 6
degree_fs_curve_for_model3_global_screening <- 6

# If TRUE, prints k.check row names/columns during selection
verbose_k_debug <- FALSE

# ============================================================
# Helpers
# ============================================================
text_sheet <- function(x) {
  data.frame(output = capture.output(x), stringsAsFactors = FALSE)
}

safe_bind_rows <- function(x) {
  x <- x[purrr::map_lgl(x, ~ !is.null(.x) && nrow(.x) > 0)]
  if (length(x) == 0) return(data.frame())
  dplyr::bind_rows(x)
}

smooth_table_from_bam <- function(model_obj, model_name, file_name, analysis_mode) {
  sm <- summary(model_obj)$s.table
  if (is.null(sm)) return(data.frame())
  
  out <- as.data.frame(sm)
  out$term <- rownames(sm)
  rownames(out) <- NULL
  out$model <- model_name
  out$file_name <- file_name
  out$analysis_mode <- analysis_mode
  
  out %>%
    dplyr::select(file_name, analysis_mode, model, term, dplyr::everything())
}

param_table_from_bam <- function(model_obj, model_name, file_name, analysis_mode) {
  pt <- summary(model_obj)$p.table
  if (is.null(pt)) return(data.frame())
  
  out <- as.data.frame(pt)
  out$term <- rownames(pt)
  rownames(out) <- NULL
  out$model <- model_name
  out$file_name <- file_name
  out$analysis_mode <- analysis_mode
  
  out %>%
    dplyr::select(file_name, analysis_mode, model, term, dplyr::everything())
}

diagnostic_summary_from_bam <- function(model_obj, model_name, file_name, analysis_mode) {
  res <- residuals(model_obj, type = "deviance")
  fit <- fitted(model_obj)
  
  data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = model_name,
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
# FIXED ACF: compute within each curve, not on concatenated residuals
# ============================================================
acf_table_from_bam <- function(model_obj,
                               df_used,
                               model_name,
                               file_name,
                               analysis_mode,
                               lag.max = 10,
                               residual_type = "pearson") {
  res <- try(residuals(model_obj, type = residual_type), silent = TRUE)
  
  if (inherits(res, "try-error") || length(res) != nrow(df_used)) {
    return(data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = model_name,
      curve = NA_character_,
      patient = NA_character_,
      repetition = NA_character_,
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
      patient = as.character(patient),
      repetition = as.character(repetition),
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
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = model_name,
      curve = d$curve[1],
      patient = d$patient[1],
      repetition = d$repetition[1],
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
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = model_name,
      curve = NA_character_,
      patient = NA_character_,
      repetition = NA_character_,
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

k_check_table_from_bam <- function(model_obj, model_name, file_name, analysis_mode) {
  kc <- try(mgcv::k.check(model_obj), silent = TRUE)
  
  if (inherits(kc, "try-error") || is.null(kc)) {
    return(data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = model_name,
      note = "k.check() could not be computed",
      stringsAsFactors = FALSE
    ))
  }
  
  out <- as.data.frame(kc)
  out$term <- rownames(kc)
  rownames(out) <- NULL
  out$model <- model_name
  out$file_name <- file_name
  out$analysis_mode <- analysis_mode
  
  out %>%
    dplyr::select(file_name, analysis_mode, model, term, dplyr::everything())
}

# ============================================================
# Robust parsing helpers
# ============================================================
standardize_colnames <- function(x) {
  x <- trimws(x)
  x <- gsub("\\.+", "_", x)
  x <- gsub("-", "_", x)
  x
}

standardize_kcheck_df <- function(kc_obj, verbose = FALSE) {
  if (inherits(kc_obj, "try-error") || is.null(kc_obj)) return(NULL)
  
  out <- as.data.frame(kc_obj)
  out$term <- rownames(kc_obj)
  rownames(out) <- NULL
  
  names(out) <- standardize_colnames(names(out))
  
  if (verbose) {
    message("[DEBUG k.check] columns: ", paste(names(out), collapse = ", "))
    message("[DEBUG k.check] terms: ", paste(out$term, collapse = " | "))
  }
  
  nm_low <- tolower(names(out))
  
  k_index_col <- names(out)[nm_low %in% c("k_index", "k.index")]
  p_value_col <- names(out)[nm_low %in% c("p_value", "p.value")]
  edf_col     <- names(out)[nm_low %in% c("edf")]
  kprime_col  <- names(out)[nm_low %in% c("k", "k_", "k.", "kprime", "k_prime")]
  
  if (length(kprime_col) == 0) {
    possible_k <- names(out)[grepl("^k", nm_low)]
    if (length(possible_k) > 0) kprime_col <- possible_k[1]
  }
  
  if (length(k_index_col) == 0 || length(p_value_col) == 0) {
    return(NULL)
  }
  
  out <- out %>%
    dplyr::mutate(
      k_index = suppressWarnings(as.numeric(.data[[k_index_col[1]]])),
      p_value = suppressWarnings(as.numeric(.data[[p_value_col[1]]])),
      edf_std = if (length(edf_col) > 0) suppressWarnings(as.numeric(.data[[edf_col[1]]])) else NA_real_,
      kprime_std = if (length(kprime_col) > 0) suppressWarnings(as.numeric(.data[[kprime_col[1]]])) else NA_real_
    )
  
  out
}

extract_smooth_summary_table <- function(model_obj) {
  sm <- summary(model_obj)$s.table
  if (is.null(sm)) return(NULL)
  
  out <- as.data.frame(sm)
  out$term <- rownames(sm)
  rownames(out) <- NULL
  
  names(out) <- standardize_colnames(names(out))
  nm_low <- tolower(names(out))
  
  edf_col <- names(out)[nm_low %in% c("edf")]
  p_col   <- names(out)[nm_low %in% c("p_value", "p.value")]
  
  out <- out %>%
    dplyr::mutate(
      edf_std = if (length(edf_col) > 0) suppressWarnings(as.numeric(.data[[edf_col[1]]])) else NA_real_,
      p_value_std = if (length(p_col) > 0) suppressWarnings(as.numeric(.data[[p_col[1]]])) else NA_real_
    )
  
  out
}

# ============================================================
# Identify smooths robustly
# ============================================================
normalize_term <- function(term_name) {
  x <- as.character(term_name)
  x <- gsub("\\s+", "", x)
  tolower(x)
}

is_global_kcheck_term <- function(term_name) {
  x <- normalize_term(term_name)
  
  has_time <- grepl("time", x)
  has_patient <- grepl("patient", x)
  has_curve <- grepl("curve", x)
  
  has_time && !has_patient && !has_curve
}

is_patient_fs_smooth_term <- function(term_name) {
  x <- normalize_term(term_name)
  grepl("time", x) && grepl("patient", x)
}

is_curve_fs_smooth_term <- function(term_name) {
  x <- normalize_term(term_name)
  grepl("time", x) && grepl("curve", x)
}

is_group_time_smooth_term <- function(term_name) {
  x <- normalize_term(term_name)
  grepl("time", x) && grepl("group", x) && !grepl("patient", x) && !grepl("curve", x)
}

kcheck_model_pass_global_only <- function(model_obj,
                                          k_index_threshold = 0.90,
                                          p_value_threshold = 0.05,
                                          verbose = FALSE) {
  kc <- try(mgcv::k.check(model_obj), silent = TRUE)
  kc_df <- standardize_kcheck_df(kc, verbose = verbose)
  
  if (is.null(kc_df) || nrow(kc_df) == 0) {
    return(list(
      pass = NA,
      status = "k.check_parse_failed",
      details = data.frame(
        term = NA_character_,
        k_index = NA_real_,
        p_value = NA_real_,
        edf_std = NA_real_,
        kprime_std = NA_real_,
        problematic = NA,
        used_for_selection = NA,
        stringsAsFactors = FALSE
      )
    ))
  }
  
  kc_df <- kc_df %>%
    dplyr::mutate(
      used_for_selection = vapply(term, is_global_kcheck_term, logical(1)),
      problematic = used_for_selection &
        !is.na(k_index) & !is.na(p_value) &
        (k_index < k_index_threshold) &
        (p_value < p_value_threshold)
    )
  
  used_rows <- kc_df %>% dplyr::filter(used_for_selection)
  
  if (verbose) {
    message("[DEBUG k.check] used rows: ", paste(used_rows$term, collapse = " | "))
  }
  
  if (nrow(used_rows) == 0) {
    return(list(
      pass = NA,
      status = "no_global_terms_identified",
      details = kc_df %>%
        dplyr::select(term, k_index, p_value, edf_std, kprime_std, problematic, used_for_selection)
    ))
  }
  
  list(
    pass = !any(used_rows$problematic, na.rm = TRUE),
    status = ifelse(!any(used_rows$problematic, na.rm = TRUE), "pass", "fail"),
    details = kc_df %>%
      dplyr::select(term, k_index, p_value, edf_std, kprime_std, problematic, used_for_selection)
  )
}

patient_fs_term_diagnostics <- function(model_obj, degree_fs_patient) {
  sm_df <- extract_smooth_summary_table(model_obj)
  
  if (is.null(sm_df) || nrow(sm_df) == 0) {
    return(data.frame(
      term = NA_character_,
      edf_std = NA_real_,
      fs_k_candidate = degree_fs_patient,
      edf_over_k_candidate_minus1 = NA_real_,
      fs_significant = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  fs_row <- sm_df %>%
    dplyr::filter(vapply(term, is_patient_fs_smooth_term, logical(1)))
  
  if (nrow(fs_row) == 0) {
    return(data.frame(
      term = NA_character_,
      edf_std = NA_real_,
      fs_k_candidate = degree_fs_patient,
      edf_over_k_candidate_minus1 = NA_real_,
      fs_significant = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  fs_row <- fs_row %>% dplyr::slice(1)
  denom <- max(degree_fs_patient - 1, 1)
  
  data.frame(
    term = fs_row$term,
    edf_std = fs_row$edf_std,
    fs_k_candidate = degree_fs_patient,
    edf_over_k_candidate_minus1 = fs_row$edf_std / denom,
    fs_significant = ifelse(is.na(fs_row$p_value_std), NA, fs_row$p_value_std < 0.05),
    stringsAsFactors = FALSE
  )
}

curve_fs_term_diagnostics <- function(model_obj, degree_fs_curve) {
  sm_df <- extract_smooth_summary_table(model_obj)
  
  if (is.null(sm_df) || nrow(sm_df) == 0) {
    return(data.frame(
      term = NA_character_,
      edf_std = NA_real_,
      fs_k_candidate = degree_fs_curve,
      edf_over_k_candidate_minus1 = NA_real_,
      fs_significant = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  fs_row <- sm_df %>%
    dplyr::filter(vapply(term, is_curve_fs_smooth_term, logical(1)))
  
  if (nrow(fs_row) == 0) {
    return(data.frame(
      term = NA_character_,
      edf_std = NA_real_,
      fs_k_candidate = degree_fs_curve,
      edf_over_k_candidate_minus1 = NA_real_,
      fs_significant = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  fs_row <- fs_row %>% dplyr::slice(1)
  denom <- max(degree_fs_curve - 1, 1)
  
  data.frame(
    term = fs_row$term,
    edf_std = fs_row$edf_std,
    fs_k_candidate = degree_fs_curve,
    edf_over_k_candidate_minus1 = fs_row$edf_std / denom,
    fs_significant = ifelse(is.na(fs_row$p_value_std), NA, fs_row$p_value_std < 0.05),
    stringsAsFactors = FALSE
  )
}

# ============================================================
# Read/prepare one file + one mode
# ============================================================
prepare_bam_data_one_file <- function(file_path, analysis_mode) {
  file_name <- basename(file_path)
  file_stub <- tools::file_path_sans_ext(file_name)
  
  raw_df <- readxl::read_excel(file_path) %>%
    dplyr::rename(time = 1)
  
  long_df <- raw_df %>%
    tidyr::pivot_longer(
      cols = -time,
      names_to = "raw_name",
      values_to = "OD"
    )
  
  df <- long_df %>%
    dplyr::mutate(
      original_group = stringr::str_extract(raw_name, "^[NMS]"),
      patient        = stringr::str_extract(raw_name, "^[NMS]\\d+"),
      repetition     = stringr::str_extract(raw_name, "(?<=Replicate\\s)\\d+")
    ) %>%
    dplyr::filter(!is.na(OD)) %>%
    dplyr::mutate(
      time       = as.numeric(time),
      OD         = as.numeric(OD),
      patient    = factor(patient),
      repetition = factor(repetition)
    )
  
  if (analysis_mode == "MS") {
    df <- df %>%
      dplyr::filter(original_group %in% c("M", "S")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("M", "S")))
  }
  
  if (analysis_mode == "NS") {
    df <- df %>%
      dplyr::filter(original_group %in% c("N", "S")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("N", "S")))
  }
  
  if (analysis_mode == "MN") {
    df <- df %>%
      dplyr::filter(original_group %in% c("M", "N")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("M", "N")))
  }
  
  if (analysis_mode == "NP") {
    df <- df %>%
      dplyr::filter(original_group %in% c("N", "M", "S")) %>%
      dplyr::mutate(
        Group = dplyr::case_when(
          original_group == "N" ~ "N",
          original_group %in% c("M", "S") ~ "P"
        ),
        Group = factor(Group, levels = c("N", "P"))
      )
  }
  
  if (analysis_mode == "three_groups") {
    df <- df %>%
      dplyr::filter(original_group %in% c("N", "M", "S")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("N", "M", "S")))
  }
  
  df <- df %>%
    dplyr::filter(!is.na(OD), !is.na(time), !is.na(Group), !is.na(patient), !is.na(repetition)) %>%
    dplyr::mutate(
      Group = droplevels(Group),
      patient = droplevels(patient),
      repetition = droplevels(repetition),
      curve = interaction(patient, repetition, drop = TRUE)
    ) %>%
    dplyr::arrange(curve, time)
  
  if (nrow(df) == 0) stop("No usable rows after filtering/re-coding.")
  if (dplyr::n_distinct(df$Group) < 2) stop("Fewer than 2 groups present after filtering.")
  if (dplyr::n_distinct(df$patient) < 2) stop("Fewer than 2 patients present after filtering.")
  if (dplyr::n_distinct(df$curve) < 2) stop("Fewer than 2 replicate curves present after filtering.")
  
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
    dplyr::ungroup()
  
  dataset_info <- data.frame(
    file_name = file_name,
    file_stub = file_stub,
    analysis_mode = analysis_mode,
    common_max_time = common_max_time,
    n_rows = nrow(df),
    n_groups = dplyr::n_distinct(df$Group),
    n_patients = dplyr::n_distinct(df$patient),
    n_repetitions = dplyr::n_distinct(df$repetition),
    n_curves = dplyr::n_distinct(df$curve),
    min_time = min(df$time, na.rm = TRUE),
    max_time = max(df$time, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  
  group_counts <- df %>%
    dplyr::count(Group, name = "n_rows") %>%
    dplyr::mutate(file_name = file_name, analysis_mode = analysis_mode) %>%
    dplyr::select(file_name, analysis_mode, dplyr::everything())
  
  curve_counts <- df %>%
    dplyr::distinct(Group, patient, repetition, curve) %>%
    dplyr::count(Group, name = "n_curves") %>%
    dplyr::mutate(file_name = file_name, analysis_mode = analysis_mode) %>%
    dplyr::select(file_name, analysis_mode, dplyr::everything())
  
  list(
    data = df,
    dataset_info = dataset_info,
    group_counts = group_counts,
    curve_counts = curve_counts
  )
}

# ============================================================
# Model-specific fitters
# ============================================================
fit_model1 <- function(df, degree_global, rho_value = 0) {
  mgcv::bam(
    OD ~ Group +
      s(time, k = degree_global),
    data = df,
    method = "fREML",
    rho = rho_value,
    AR.start = df$ar_start
  )
}

fit_model2 <- function(df, degree_global, degree_fs_patient, rho_value = 0) {
  mgcv::bam(
    OD ~ Group +
      s(time, k = degree_global) +
      s(time, Group, bs = "sz", k = degree_global) +
      s(patient, bs = "re") +
      s(curve, bs = "re") +
      s(time, patient, bs = "fs", k = degree_fs_patient, m = 1),
    data = df,
    method = "fREML",
    rho = rho_value,
    AR.start = df$ar_start
  )
}

fit_model3 <- function(df, degree_global, degree_fs_patient, degree_fs_curve, rho_value = 0) {
  mgcv::bam(
    OD ~ Group +
      s(time, k = degree_global) +
      s(time, Group, bs = "sz", k = degree_global) +
      s(patient, bs = "re") +
      s(curve, bs = "re") +
      s(time, patient, bs = "fs", k = degree_fs_patient, m = 1) +
      s(time, curve, bs = "fs", k = degree_fs_curve, m = 1),
    data = df,
    method = "fREML",
    rho = rho_value,
    AR.start = df$ar_start
  )
}

# ============================================================
# Selection helpers
# ============================================================
selection_note_from_check <- function(chk) {
  if (isTRUE(chk$pass)) return("Accepted by k.check on global smooths")
  if (identical(chk$pass, FALSE)) return("Rejected by k.check on global smooths")
  paste("k.check inconclusive:", chk$status)
}

final_global_selection_status <- function(tried_df, prefix = NULL) {
  out <- if (any(tried_df$pass %in% TRUE, na.rm = TRUE)) {
    "Accepted by k.check on global smooths"
  } else if (any(is.na(tried_df$pass))) {
    "Fallback used after inconclusive k.check"
  } else {
    "Fallback used"
  }
  
  if (!is.null(prefix)) out <- paste0(out, " in ", prefix)
  out
}

# ============================================================
# Model1: select global k
# ============================================================
select_degree_global_model1_one_file <- function(df_prepared,
                                                 file_name,
                                                 analysis_mode,
                                                 degree_global_candidates = 3:20,
                                                 rho_value = 0,
                                                 k_index_threshold = 0.90,
                                                 p_value_threshold = 0.05,
                                                 fallback_degree_global = 20,
                                                 verbose_k_debug = FALSE) {
  tried_rows <- list()
  counter <- 0
  selected_degree_global <- NULL
  
  for (kg in degree_global_candidates) {
    counter <- counter + 1
    
    message("[SELECT model1 degree_global] File: ", file_name,
            " | Mode: ", analysis_mode,
            " | Trying degree_global=", kg)
    
    fit <- try(
      fit_model1(df_prepared, degree_global = kg, rho_value = rho_value),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model1",
        degree_global = kg,
        degree_fs_patient = NA_integer_,
        degree_fs_curve = NA_integer_,
        pass = NA,
        selected = FALSE,
        note = "Model fit failed during degree_global screening",
        stringsAsFactors = FALSE
      )
      next
    }
    
    chk <- kcheck_model_pass_global_only(
      fit,
      k_index_threshold = k_index_threshold,
      p_value_threshold = p_value_threshold,
      verbose = verbose_k_debug
    )
    
    tried_rows[[counter]] <- data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = "model1",
      degree_global = kg,
      degree_fs_patient = NA_integer_,
      degree_fs_curve = NA_integer_,
      pass = chk$pass,
      selected = FALSE,
      note = selection_note_from_check(chk),
      stringsAsFactors = FALSE
    )
    
    if (isTRUE(chk$pass) && is.null(selected_degree_global)) {
      selected_degree_global <- kg
      tried_rows[[counter]]$selected <- TRUE
      break
    }
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (is.null(selected_degree_global)) {
    selected_degree_global <- fallback_degree_global
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model1",
        degree_global = fallback_degree_global,
        degree_fs_patient = NA_integer_,
        degree_fs_curve = NA_integer_,
        pass = NA,
        selected = TRUE,
        note = if (any(is.na(tried_df$pass))) {
          "Fallback degree_global used because k.check was inconclusive for all candidates"
        } else {
          "Fallback degree_global used because no candidate passed"
        },
        stringsAsFactors = FALSE
      )
    )
  }
  
  selected_df <- data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = "model1",
    degree_global = selected_degree_global,
    degree_fs_patient = NA_integer_,
    degree_fs_curve = NA_integer_,
    degree_global_selection_status = final_global_selection_status(tried_df),
    degree_fs_patient_selection_status = NA_character_,
    degree_fs_curve_selection_status = NA_character_,
    selection_status = final_global_selection_status(tried_df),
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

# ============================================================
# Model2: select global k
# ============================================================
select_degree_global_model2_one_file <- function(df_prepared,
                                                 file_name,
                                                 analysis_mode,
                                                 degree_global_candidates = 3:20,
                                                 degree_fs_patient_for_screening = 6,
                                                 rho_value = 0,
                                                 k_index_threshold = 0.90,
                                                 p_value_threshold = 0.05,
                                                 fallback_degree_global = 20,
                                                 verbose_k_debug = FALSE) {
  tried_rows <- list()
  counter <- 0
  selected_degree_global <- NULL
  
  for (kg in degree_global_candidates) {
    counter <- counter + 1
    
    message("[SELECT model2 degree_global] File: ", file_name,
            " | Mode: ", analysis_mode,
            " | Trying degree_global=", kg,
            " | Screening degree_fs_patient=", degree_fs_patient_for_screening)
    
    fit <- try(
      fit_model2(
        df_prepared,
        degree_global = kg,
        degree_fs_patient = degree_fs_patient_for_screening,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model2",
        degree_global = kg,
        degree_fs_patient = degree_fs_patient_for_screening,
        degree_fs_curve = NA_integer_,
        pass = NA,
        selected = FALSE,
        note = "Model2 fit failed during degree_global screening",
        stringsAsFactors = FALSE
      )
      next
    }
    
    chk <- kcheck_model_pass_global_only(
      fit,
      k_index_threshold = k_index_threshold,
      p_value_threshold = p_value_threshold,
      verbose = verbose_k_debug
    )
    
    tried_rows[[counter]] <- data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = "model2",
      degree_global = kg,
      degree_fs_patient = degree_fs_patient_for_screening,
      degree_fs_curve = NA_integer_,
      pass = chk$pass,
      selected = FALSE,
      note = selection_note_from_check(chk),
      stringsAsFactors = FALSE
    )
    
    if (isTRUE(chk$pass) && is.null(selected_degree_global)) {
      selected_degree_global <- kg
      tried_rows[[counter]]$selected <- TRUE
      break
    }
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (is.null(selected_degree_global)) {
    selected_degree_global <- fallback_degree_global
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model2",
        degree_global = fallback_degree_global,
        degree_fs_patient = degree_fs_patient_for_screening,
        degree_fs_curve = NA_integer_,
        pass = NA,
        selected = TRUE,
        note = if (any(is.na(tried_df$pass))) {
          "Fallback degree_global used because k.check was inconclusive for all model2 candidates"
        } else {
          "Fallback degree_global used because no candidate passed in model2"
        },
        stringsAsFactors = FALSE
      )
    )
  }
  
  selected_df <- data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = "model2",
    degree_global = selected_degree_global,
    degree_fs_patient = NA_integer_,
    degree_fs_curve = NA_integer_,
    degree_global_selection_status = final_global_selection_status(tried_df, prefix = "model2"),
    degree_fs_patient_selection_status = NA_character_,
    degree_fs_curve_selection_status = NA_character_,
    selection_status = final_global_selection_status(tried_df, prefix = "model2"),
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

# ============================================================
# Model2: select patient fs k
# ============================================================
select_degree_fs_patient_model2_one_file <- function(df_prepared,
                                                     file_name,
                                                     analysis_mode,
                                                     degree_global,
                                                     degree_fs_patient_candidates = 3:20,
                                                     fs_patient_to_global_ratio = 1.5,
                                                     rho_value = 0,
                                                     fallback_degree_fs_patient = 20,
                                                     fs_aic_tolerance = 2,
                                                     fs_edf_ratio_threshold = 0.90) {
  tried_rows <- list()
  counter <- 0
  
  for (kf in degree_fs_patient_candidates) {
    if (kf > floor(fs_patient_to_global_ratio * degree_global)) next
    
    counter <- counter + 1
    
    message("[SELECT model2 degree_fs_patient] File: ", file_name,
            " | Mode: ", analysis_mode,
            " | degree_global=", degree_global,
            " | Trying degree_fs_patient=", kf)
    
    fit2 <- try(
      fit_model2(
        df_prepared,
        degree_global = degree_global,
        degree_fs_patient = kf,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit2, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model2",
        degree_global = degree_global,
        degree_fs_patient = kf,
        degree_fs_curve = NA_integer_,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = FALSE,
        note = "Model2 fit failed",
        stringsAsFactors = FALSE
      )
      next
    }
    
    fs_diag <- patient_fs_term_diagnostics(fit2, degree_fs_patient = kf)
    
    aic_val <- try(AIC(fit2), silent = TRUE)
    if (inherits(aic_val, "try-error")) aic_val <- NA_real_
    
    fs_ratio_ok <- !is.na(fs_diag$edf_over_k_candidate_minus1[1]) &&
      fs_diag$edf_over_k_candidate_minus1[1] < fs_edf_ratio_threshold
    
    tried_rows[[counter]] <- data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = "model2",
      degree_global = degree_global,
      degree_fs_patient = kf,
      degree_fs_curve = NA_integer_,
      AIC = as.numeric(aic_val),
      fs_term = fs_diag$term[1],
      fs_edf = fs_diag$edf_std[1],
      fs_edf_ratio = fs_diag$edf_over_k_candidate_minus1[1],
      fs_ratio_ok = fs_ratio_ok,
      selected = FALSE,
      note = if (is.na(fs_diag$term[1])) {
        "Candidate evaluated; patient fs term could not be identified robustly"
      } else {
        "Candidate evaluated"
      },
      stringsAsFactors = FALSE
    )
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (nrow(tried_df) == 0 || all(is.na(tried_df$AIC))) {
    selected_degree_fs_patient <- fallback_degree_fs_patient
    
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model2",
        degree_global = degree_global,
        degree_fs_patient = fallback_degree_fs_patient,
        degree_fs_curve = NA_integer_,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = TRUE,
        note = "Fallback degree_fs_patient used because no candidate could be evaluated",
        stringsAsFactors = FALSE
      )
    )
    
    selected_status <- "Fallback used"
  } else {
    best_aic <- min(tried_df$AIC, na.rm = TRUE)
    
    within_tol <- tried_df %>%
      dplyr::filter(!is.na(AIC)) %>%
      dplyr::filter(AIC <= best_aic + fs_aic_tolerance)
    
    preferred_pool <- within_tol %>%
      dplyr::filter(fs_ratio_ok %in% TRUE)
    
    if (nrow(preferred_pool) == 0) {
      preferred_pool <- within_tol
    }
    
    preferred_pool <- preferred_pool %>%
      dplyr::arrange(degree_fs_patient, AIC)
    
    selected_degree_fs_patient <- preferred_pool$degree_fs_patient[1]
    
    tried_df <- tried_df %>%
      dplyr::mutate(selected = degree_fs_patient == selected_degree_fs_patient)
    
    selected_status <- if (any(tried_df$fs_ratio_ok %in% TRUE, na.rm = TRUE)) {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, ") with patient-fs edf-ratio preference")
    } else {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, "); no candidate met patient-fs edf-ratio preference")
    }
  }
  
  selected_df <- data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = "model2",
    degree_global = degree_global,
    degree_fs_patient = selected_degree_fs_patient,
    degree_fs_curve = NA_integer_,
    degree_global_selection_status = NA_character_,
    degree_fs_patient_selection_status = selected_status,
    degree_fs_curve_selection_status = NA_character_,
    selection_status = selected_status,
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

# ============================================================
# Model3: select global k
# ============================================================
select_degree_global_model3_one_file <- function(df_prepared,
                                                 file_name,
                                                 analysis_mode,
                                                 degree_global_candidates = 3:20,
                                                 degree_fs_patient_for_screening = 6,
                                                 degree_fs_curve_for_screening = 6,
                                                 rho_value = 0,
                                                 k_index_threshold = 0.90,
                                                 p_value_threshold = 0.05,
                                                 fallback_degree_global = 20,
                                                 verbose_k_debug = FALSE) {
  tried_rows <- list()
  counter <- 0
  selected_degree_global <- NULL
  
  for (kg in degree_global_candidates) {
    counter <- counter + 1
    
    message("[SELECT model3 degree_global] File: ", file_name,
            " | Mode: ", analysis_mode,
            " | Trying degree_global=", kg,
            " | Screening degree_fs_patient=", degree_fs_patient_for_screening,
            " | Screening degree_fs_curve=", degree_fs_curve_for_screening)
    
    fit <- try(
      fit_model3(
        df_prepared,
        degree_global = kg,
        degree_fs_patient = degree_fs_patient_for_screening,
        degree_fs_curve = degree_fs_curve_for_screening,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model3",
        degree_global = kg,
        degree_fs_patient = degree_fs_patient_for_screening,
        degree_fs_curve = degree_fs_curve_for_screening,
        pass = NA,
        selected = FALSE,
        note = "Model3 fit failed during degree_global screening",
        stringsAsFactors = FALSE
      )
      next
    }
    
    chk <- kcheck_model_pass_global_only(
      fit,
      k_index_threshold = k_index_threshold,
      p_value_threshold = p_value_threshold,
      verbose = verbose_k_debug
    )
    
    tried_rows[[counter]] <- data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = "model3",
      degree_global = kg,
      degree_fs_patient = degree_fs_patient_for_screening,
      degree_fs_curve = degree_fs_curve_for_screening,
      pass = chk$pass,
      selected = FALSE,
      note = if (isTRUE(chk$pass)) {
        "Accepted by k.check on global smooths in model3"
      } else if (identical(chk$pass, FALSE)) {
        "Rejected by k.check on global smooths in model3"
      } else {
        paste("k.check inconclusive in model3:", chk$status)
      },
      stringsAsFactors = FALSE
    )
    
    if (isTRUE(chk$pass) && is.null(selected_degree_global)) {
      selected_degree_global <- kg
      tried_rows[[counter]]$selected <- TRUE
      break
    }
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (is.null(selected_degree_global)) {
    selected_degree_global <- fallback_degree_global
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model3",
        degree_global = fallback_degree_global,
        degree_fs_patient = degree_fs_patient_for_screening,
        degree_fs_curve = degree_fs_curve_for_screening,
        pass = NA,
        selected = TRUE,
        note = if (any(is.na(tried_df$pass))) {
          "Fallback degree_global used because k.check was inconclusive for all model3 candidates"
        } else {
          "Fallback degree_global used because no candidate passed in model3"
        },
        stringsAsFactors = FALSE
      )
    )
  }
  
  selected_df <- data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = "model3",
    degree_global = selected_degree_global,
    degree_fs_patient = NA_integer_,
    degree_fs_curve = NA_integer_,
    degree_global_selection_status = final_global_selection_status(tried_df, prefix = "model3"),
    degree_fs_patient_selection_status = NA_character_,
    degree_fs_curve_selection_status = NA_character_,
    selection_status = final_global_selection_status(tried_df, prefix = "model3"),
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

# ============================================================
# Model3: select patient fs k
# ============================================================
select_degree_fs_patient_model3_one_file <- function(df_prepared,
                                                     file_name,
                                                     analysis_mode,
                                                     degree_global,
                                                     degree_fs_curve_fixed = 6,
                                                     degree_fs_patient_candidates = 3:20,
                                                     fs_patient_to_global_ratio = 1.5,
                                                     rho_value = 0,
                                                     fallback_degree_fs_patient = 20,
                                                     fs_aic_tolerance = 2,
                                                     fs_edf_ratio_threshold = 0.90) {
  tried_rows <- list()
  counter <- 0
  
  for (kf in degree_fs_patient_candidates) {
    if (kf > floor(fs_patient_to_global_ratio * degree_global)) next
    
    counter <- counter + 1
    
    message("[SELECT model3 degree_fs_patient] File: ", file_name,
            " | Mode: ", analysis_mode,
            " | degree_global=", degree_global,
            " | degree_fs_curve_fixed=", degree_fs_curve_fixed,
            " | Trying degree_fs_patient=", kf)
    
    fit3 <- try(
      fit_model3(
        df_prepared,
        degree_global = degree_global,
        degree_fs_patient = kf,
        degree_fs_curve = degree_fs_curve_fixed,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit3, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model3_patient_fs",
        degree_global = degree_global,
        degree_fs_patient = kf,
        degree_fs_curve = degree_fs_curve_fixed,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = FALSE,
        note = "Model3 fit failed during patient fs screening",
        stringsAsFactors = FALSE
      )
      next
    }
    
    fs_diag <- patient_fs_term_diagnostics(fit3, degree_fs_patient = kf)
    
    aic_val <- try(AIC(fit3), silent = TRUE)
    if (inherits(aic_val, "try-error")) aic_val <- NA_real_
    
    fs_ratio_ok <- !is.na(fs_diag$edf_over_k_candidate_minus1[1]) &&
      fs_diag$edf_over_k_candidate_minus1[1] < fs_edf_ratio_threshold
    
    tried_rows[[counter]] <- data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = "model3_patient_fs",
      degree_global = degree_global,
      degree_fs_patient = kf,
      degree_fs_curve = degree_fs_curve_fixed,
      AIC = as.numeric(aic_val),
      fs_term = fs_diag$term[1],
      fs_edf = fs_diag$edf_std[1],
      fs_edf_ratio = fs_diag$edf_over_k_candidate_minus1[1],
      fs_ratio_ok = fs_ratio_ok,
      selected = FALSE,
      note = if (is.na(fs_diag$term[1])) {
        "Candidate evaluated; patient fs term could not be identified robustly"
      } else {
        "Candidate evaluated"
      },
      stringsAsFactors = FALSE
    )
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (nrow(tried_df) == 0 || all(is.na(tried_df$AIC))) {
    selected_degree_fs_patient <- fallback_degree_fs_patient
    
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model3_patient_fs",
        degree_global = degree_global,
        degree_fs_patient = fallback_degree_fs_patient,
        degree_fs_curve = degree_fs_curve_fixed,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = TRUE,
        note = "Fallback degree_fs_patient used because no candidate could be evaluated",
        stringsAsFactors = FALSE
      )
    )
    
    selected_status <- "Fallback used"
  } else {
    best_aic <- min(tried_df$AIC, na.rm = TRUE)
    
    within_tol <- tried_df %>%
      dplyr::filter(!is.na(AIC)) %>%
      dplyr::filter(AIC <= best_aic + fs_aic_tolerance)
    
    preferred_pool <- within_tol %>%
      dplyr::filter(fs_ratio_ok %in% TRUE)
    
    if (nrow(preferred_pool) == 0) {
      preferred_pool <- within_tol
    }
    
    preferred_pool <- preferred_pool %>%
      dplyr::arrange(degree_fs_patient, AIC)
    
    selected_degree_fs_patient <- preferred_pool$degree_fs_patient[1]
    
    tried_df <- tried_df %>%
      dplyr::mutate(selected = degree_fs_patient == selected_degree_fs_patient)
    
    selected_status <- if (any(tried_df$fs_ratio_ok %in% TRUE, na.rm = TRUE)) {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, ") with patient-fs edf-ratio preference")
    } else {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, "); no candidate met patient-fs edf-ratio preference")
    }
  }
  
  selected_df <- data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = "model3",
    degree_global = degree_global,
    degree_fs_patient = selected_degree_fs_patient,
    degree_fs_curve = NA_integer_,
    degree_global_selection_status = NA_character_,
    degree_fs_patient_selection_status = selected_status,
    degree_fs_curve_selection_status = NA_character_,
    selection_status = selected_status,
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

# ============================================================
# Model3: select curve fs k
# ============================================================
select_degree_fs_curve_model3_one_file <- function(df_prepared,
                                                   file_name,
                                                   analysis_mode,
                                                   degree_global,
                                                   degree_fs_patient_fixed,
                                                   degree_fs_curve_candidates = 3:20,
                                                   fs_curve_to_global_ratio = 1.5,
                                                   rho_value = 0,
                                                   fallback_degree_fs_curve = 20,
                                                   fs_aic_tolerance = 2,
                                                   fs_edf_ratio_threshold = 0.90) {
  tried_rows <- list()
  counter <- 0
  
  for (kf in degree_fs_curve_candidates) {
    if (kf > floor(fs_curve_to_global_ratio * degree_global)) next
    
    counter <- counter + 1
    
    message("[SELECT model3 degree_fs_curve] File: ", file_name,
            " | Mode: ", analysis_mode,
            " | degree_global=", degree_global,
            " | degree_fs_patient_fixed=", degree_fs_patient_fixed,
            " | Trying degree_fs_curve=", kf)
    
    fit3 <- try(
      fit_model3(
        df_prepared,
        degree_global = degree_global,
        degree_fs_patient = degree_fs_patient_fixed,
        degree_fs_curve = kf,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit3, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model3_curve_fs",
        degree_global = degree_global,
        degree_fs_patient = degree_fs_patient_fixed,
        degree_fs_curve = kf,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = FALSE,
        note = "Model3 fit failed during curve fs screening",
        stringsAsFactors = FALSE
      )
      next
    }
    
    fs_diag <- curve_fs_term_diagnostics(fit3, degree_fs_curve = kf)
    
    aic_val <- try(AIC(fit3), silent = TRUE)
    if (inherits(aic_val, "try-error")) aic_val <- NA_real_
    
    fs_ratio_ok <- !is.na(fs_diag$edf_over_k_candidate_minus1[1]) &&
      fs_diag$edf_over_k_candidate_minus1[1] < fs_edf_ratio_threshold
    
    tried_rows[[counter]] <- data.frame(
      file_name = file_name,
      analysis_mode = analysis_mode,
      model = "model3_curve_fs",
      degree_global = degree_global,
      degree_fs_patient = degree_fs_patient_fixed,
      degree_fs_curve = kf,
      AIC = as.numeric(aic_val),
      fs_term = fs_diag$term[1],
      fs_edf = fs_diag$edf_std[1],
      fs_edf_ratio = fs_diag$edf_over_k_candidate_minus1[1],
      fs_ratio_ok = fs_ratio_ok,
      selected = FALSE,
      note = if (is.na(fs_diag$term[1])) {
        "Candidate evaluated; curve fs term could not be identified robustly"
      } else {
        "Candidate evaluated"
      },
      stringsAsFactors = FALSE
    )
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (nrow(tried_df) == 0 || all(is.na(tried_df$AIC))) {
    selected_degree_fs_curve <- fallback_degree_fs_curve
    
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        file_name = file_name,
        analysis_mode = analysis_mode,
        model = "model3_curve_fs",
        degree_global = degree_global,
        degree_fs_patient = degree_fs_patient_fixed,
        degree_fs_curve = fallback_degree_fs_curve,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = TRUE,
        note = "Fallback degree_fs_curve used because no candidate could be evaluated",
        stringsAsFactors = FALSE
      )
    )
    
    selected_status <- "Fallback used"
  } else {
    best_aic <- min(tried_df$AIC, na.rm = TRUE)
    
    within_tol <- tried_df %>%
      dplyr::filter(!is.na(AIC)) %>%
      dplyr::filter(AIC <= best_aic + fs_aic_tolerance)
    
    preferred_pool <- within_tol %>%
      dplyr::filter(fs_ratio_ok %in% TRUE)
    
    if (nrow(preferred_pool) == 0) {
      preferred_pool <- within_tol
    }
    
    preferred_pool <- preferred_pool %>%
      dplyr::arrange(degree_fs_curve, AIC)
    
    selected_degree_fs_curve <- preferred_pool$degree_fs_curve[1]
    
    tried_df <- tried_df %>%
      dplyr::mutate(selected = degree_fs_curve == selected_degree_fs_curve)
    
    selected_status <- if (any(tried_df$fs_ratio_ok %in% TRUE, na.rm = TRUE)) {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, ") with curve-fs edf-ratio preference")
    } else {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, "); no candidate met curve-fs edf-ratio preference")
    }
  }
  
  selected_df <- data.frame(
    file_name = file_name,
    analysis_mode = analysis_mode,
    model = "model3",
    degree_global = degree_global,
    degree_fs_patient = degree_fs_patient_fixed,
    degree_fs_curve = selected_degree_fs_curve,
    degree_global_selection_status = NA_character_,
    degree_fs_patient_selection_status = NA_character_,
    degree_fs_curve_selection_status = selected_status,
    selection_status = selected_status,
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

# ============================================================
# Select k for all models for one file
# ============================================================
select_k_all_models_one_file <- function(df_prepared,
                                         file_name,
                                         analysis_mode,
                                         degree_global_candidates = 3:20,
                                         degree_fs_patient_candidates = 3:20,
                                         degree_fs_curve_candidates = 3:20,
                                         fs_patient_to_global_ratio = 1.5,
                                         fs_curve_to_global_ratio = 1.5,
                                         rho_value = 0,
                                         k_index_threshold = 0.90,
                                         p_value_threshold = 0.05,
                                         fallback_degree_global = 20,
                                         fallback_degree_fs_patient = 20,
                                         fallback_degree_fs_curve = 20,
                                         fs_aic_tolerance = 2,
                                         fs_edf_ratio_threshold = 0.90,
                                         degree_fs_patient_for_model2_global_screening = 6,
                                         degree_fs_patient_for_model3_global_screening = 6,
                                         degree_fs_curve_for_model3_global_screening = 6,
                                         verbose_k_debug = FALSE) {
  sel_m1_global <- select_degree_global_model1_one_file(
    df_prepared = df_prepared,
    file_name = file_name,
    analysis_mode = analysis_mode,
    degree_global_candidates = degree_global_candidates,
    rho_value = rho_value,
    k_index_threshold = k_index_threshold,
    p_value_threshold = p_value_threshold,
    fallback_degree_global = fallback_degree_global,
    verbose_k_debug = verbose_k_debug
  )
  
  sel_m2_global <- select_degree_global_model2_one_file(
    df_prepared = df_prepared,
    file_name = file_name,
    analysis_mode = analysis_mode,
    degree_global_candidates = degree_global_candidates,
    degree_fs_patient_for_screening = degree_fs_patient_for_model2_global_screening,
    rho_value = rho_value,
    k_index_threshold = k_index_threshold,
    p_value_threshold = p_value_threshold,
    fallback_degree_global = fallback_degree_global,
    verbose_k_debug = verbose_k_debug
  )
  
  selected_degree_global_model2 <- sel_m2_global$selected_parameters$degree_global[1]
  
  sel_m2_patient_fs <- select_degree_fs_patient_model2_one_file(
    df_prepared = df_prepared,
    file_name = file_name,
    analysis_mode = analysis_mode,
    degree_global = selected_degree_global_model2,
    degree_fs_patient_candidates = degree_fs_patient_candidates,
    fs_patient_to_global_ratio = fs_patient_to_global_ratio,
    rho_value = rho_value,
    fallback_degree_fs_patient = fallback_degree_fs_patient,
    fs_aic_tolerance = fs_aic_tolerance,
    fs_edf_ratio_threshold = fs_edf_ratio_threshold
  )
  
  selected_model2 <- sel_m2_global$selected_parameters %>%
    dplyr::select(file_name, analysis_mode, model, degree_global, degree_global_selection_status) %>%
    dplyr::left_join(
      sel_m2_patient_fs$selected_parameters %>%
        dplyr::select(file_name, analysis_mode, model, degree_fs_patient, degree_fs_patient_selection_status),
      by = c("file_name", "analysis_mode", "model")
    ) %>%
    dplyr::mutate(
      degree_fs_curve = NA_integer_,
      degree_fs_curve_selection_status = NA_character_,
      selection_status = paste(
        degree_global_selection_status,
        "|",
        degree_fs_patient_selection_status
      )
    )
  
  sel_m3_global <- select_degree_global_model3_one_file(
    df_prepared = df_prepared,
    file_name = file_name,
    analysis_mode = analysis_mode,
    degree_global_candidates = degree_global_candidates,
    degree_fs_patient_for_screening = degree_fs_patient_for_model3_global_screening,
    degree_fs_curve_for_screening = degree_fs_curve_for_model3_global_screening,
    rho_value = rho_value,
    k_index_threshold = k_index_threshold,
    p_value_threshold = p_value_threshold,
    fallback_degree_global = fallback_degree_global,
    verbose_k_debug = verbose_k_debug
  )
  
  selected_degree_global_model3 <- sel_m3_global$selected_parameters$degree_global[1]
  
  sel_m3_patient_fs <- select_degree_fs_patient_model3_one_file(
    df_prepared = df_prepared,
    file_name = file_name,
    analysis_mode = analysis_mode,
    degree_global = selected_degree_global_model3,
    degree_fs_curve_fixed = degree_fs_curve_for_model3_global_screening,
    degree_fs_patient_candidates = degree_fs_patient_candidates,
    fs_patient_to_global_ratio = fs_patient_to_global_ratio,
    rho_value = rho_value,
    fallback_degree_fs_patient = fallback_degree_fs_patient,
    fs_aic_tolerance = fs_aic_tolerance,
    fs_edf_ratio_threshold = fs_edf_ratio_threshold
  )
  
  selected_degree_fs_patient_model3 <- sel_m3_patient_fs$selected_parameters$degree_fs_patient[1]
  
  sel_m3_curve_fs <- select_degree_fs_curve_model3_one_file(
    df_prepared = df_prepared,
    file_name = file_name,
    analysis_mode = analysis_mode,
    degree_global = selected_degree_global_model3,
    degree_fs_patient_fixed = selected_degree_fs_patient_model3,
    degree_fs_curve_candidates = degree_fs_curve_candidates,
    fs_curve_to_global_ratio = fs_curve_to_global_ratio,
    rho_value = rho_value,
    fallback_degree_fs_curve = fallback_degree_fs_curve,
    fs_aic_tolerance = fs_aic_tolerance,
    fs_edf_ratio_threshold = fs_edf_ratio_threshold
  )
  
  selected_model3 <- sel_m3_global$selected_parameters %>%
    dplyr::select(file_name, analysis_mode, model, degree_global, degree_global_selection_status) %>%
    dplyr::left_join(
      sel_m3_patient_fs$selected_parameters %>%
        dplyr::select(file_name, analysis_mode, model, degree_fs_patient, degree_fs_patient_selection_status),
      by = c("file_name", "analysis_mode", "model")
    ) %>%
    dplyr::left_join(
      sel_m3_curve_fs$selected_parameters %>%
        dplyr::select(file_name, analysis_mode, model, degree_fs_curve, degree_fs_curve_selection_status),
      by = c("file_name", "analysis_mode", "model")
    ) %>%
    dplyr::mutate(
      selection_status = paste(
        degree_global_selection_status,
        "|",
        degree_fs_patient_selection_status,
        "|",
        degree_fs_curve_selection_status
      )
    )
  
  selected_all <- dplyr::bind_rows(
    sel_m1_global$selected_parameters,
    selected_model2,
    selected_model3
  )
  
  grid_all <- dplyr::bind_rows(
    sel_m1_global$selection_grid,
    sel_m2_global$selection_grid,
    sel_m2_patient_fs$selection_grid,
    sel_m3_global$selection_grid,
    sel_m3_patient_fs$selection_grid,
    sel_m3_curve_fs$selection_grid
  )
  
  list(
    selection_grid = grid_all,
    selected_parameters = selected_all
  )
}

# ============================================================
# Final fitting with model-specific selected k
# ============================================================
run_bam_one_file <- function(file_path,
                             analysis_mode,
                             degree_global_model1 = 12,
                             degree_global_model2 = 12,
                             degree_fs_patient_model2 = 6,
                             degree_global_model3 = 12,
                             degree_fs_patient_model3 = 6,
                             degree_fs_curve_model3 = 6,
                             rho_value = 0) {
  file_name <- basename(file_path)
  file_stub <- tools::file_path_sans_ext(file_name)
  
  message("\n====================================================")
  message("[START FINAL FIT] File: ", file_name, " | Mode: ", analysis_mode)
  message("[START FINAL FIT] model1 degree_global = ", degree_global_model1)
  message("[START FINAL FIT] model2 degree_global = ", degree_global_model2,
          " | degree_fs_patient = ", degree_fs_patient_model2)
  message("[START FINAL FIT] model3 degree_global = ", degree_global_model3,
          " | degree_fs_patient = ", degree_fs_patient_model3,
          " | degree_fs_curve = ", degree_fs_curve_model3)
  message("====================================================")
  
  prep <- prepare_bam_data_one_file(file_path, analysis_mode)
  df <- prep$data
  
  dataset_info <- prep$dataset_info %>%
    dplyr::mutate(
      degree_global_model1 = degree_global_model1,
      degree_global_model2 = degree_global_model2,
      degree_fs_patient_model2 = degree_fs_patient_model2,
      degree_global_model3 = degree_global_model3,
      degree_fs_patient_model3 = degree_fs_patient_model3,
      degree_fs_curve_model3 = degree_fs_curve_model3,
      rho_value = rho_value
    ) %>%
    dplyr::select(
      file_name, file_stub, analysis_mode,
      degree_global_model1, degree_global_model2, degree_fs_patient_model2,
      degree_global_model3, degree_fs_patient_model3, degree_fs_curve_model3, rho_value,
      dplyr::everything()
    )
  
  group_counts <- prep$group_counts
  curve_counts <- prep$curve_counts
  
  model1 <- fit_model1(
    df = df,
    degree_global = degree_global_model1,
    rho_value = rho_value
  )
  
  model2 <- fit_model2(
    df = df,
    degree_global = degree_global_model2,
    degree_fs_patient = degree_fs_patient_model2,
    rho_value = rho_value
  )
  
  model3 <- fit_model3(
    df = df,
    degree_global = degree_global_model3,
    degree_fs_patient = degree_fs_patient_model3,
    degree_fs_curve = degree_fs_curve_model3,
    rho_value = rho_value
  )
  
  aic_table <- AIC(model1, model2, model3) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("model") %>%
    dplyr::mutate(
      file_name = file_name,
      analysis_mode = analysis_mode,
      degree_global = dplyr::case_when(
        model == "model1" ~ degree_global_model1,
        model == "model2" ~ degree_global_model2,
        model == "model3" ~ degree_global_model3,
        TRUE ~ NA_real_
      ),
      degree_fs_patient = dplyr::case_when(
        model == "model2" ~ degree_fs_patient_model2,
        model == "model3" ~ degree_fs_patient_model3,
        TRUE ~ NA_real_
      ),
      degree_fs_curve = dplyr::case_when(
        model == "model3" ~ degree_fs_curve_model3,
        TRUE ~ NA_real_
      )
    ) %>%
    dplyr::select(file_name, analysis_mode, model, degree_global, degree_fs_patient, degree_fs_curve, dplyr::everything())
  
  param_tables <- dplyr::bind_rows(
    param_table_from_bam(model1, "model1", file_name, analysis_mode),
    param_table_from_bam(model2, "model2", file_name, analysis_mode),
    param_table_from_bam(model3, "model3", file_name, analysis_mode)
  )
  
  smooth_tables <- dplyr::bind_rows(
    smooth_table_from_bam(model1, "model1", file_name, analysis_mode),
    smooth_table_from_bam(model2, "model2", file_name, analysis_mode),
    smooth_table_from_bam(model3, "model3", file_name, analysis_mode)
  )
  
  diagnostics_summary <- dplyr::bind_rows(
    diagnostic_summary_from_bam(model1, "model1", file_name, analysis_mode),
    diagnostic_summary_from_bam(model2, "model2", file_name, analysis_mode),
    diagnostic_summary_from_bam(model3, "model3", file_name, analysis_mode)
  )
  
  acf_tables <- dplyr::bind_rows(
    acf_table_from_bam(
      model_obj = model1,
      df_used = df,
      model_name = "model1",
      file_name = file_name,
      analysis_mode = analysis_mode,
      lag.max = 10,
      residual_type = "pearson"
    ),
    acf_table_from_bam(
      model_obj = model2,
      df_used = df,
      model_name = "model2",
      file_name = file_name,
      analysis_mode = analysis_mode,
      lag.max = 10,
      residual_type = "pearson"
    ),
    acf_table_from_bam(
      model_obj = model3,
      df_used = df,
      model_name = "model3",
      file_name = file_name,
      analysis_mode = analysis_mode,
      lag.max = 10,
      residual_type = "pearson"
    )
  )
  
  k_check_tables <- dplyr::bind_rows(
    k_check_table_from_bam(model1, "model1", file_name, analysis_mode),
    k_check_table_from_bam(model2, "model2", file_name, analysis_mode),
    k_check_table_from_bam(model3, "model3", file_name, analysis_mode)
  )
  
  model1_summary <- text_sheet(summary(model1))
  model2_summary <- text_sheet(summary(model2))
  model3_summary <- text_sheet(summary(model3))
  
  model1_gam_check <- text_sheet(mgcv::gam.check(model1))
  model2_gam_check <- text_sheet(mgcv::gam.check(model2))
  model3_gam_check <- text_sheet(mgcv::gam.check(model3))
  
  smooth_tables_std <- smooth_tables
  names(smooth_tables_std) <- standardize_colnames(names(smooth_tables_std))
  
  pcol <- names(smooth_tables_std)[tolower(names(smooth_tables_std)) %in% c("p_value", "p.value")]
  
  if (length(pcol) > 0) {
    group_difference_flag <- smooth_tables_std %>%
      dplyr::filter(model %in% c("model2", "model3")) %>%
      dplyr::filter(vapply(term, is_group_time_smooth_term, logical(1))) %>%
      dplyr::mutate(significant = suppressWarnings(as.numeric(.data[[pcol[1]]])) < 0.05)
  } else {
    group_difference_flag <- smooth_tables_std %>%
      dplyr::filter(model %in% c("model2", "model3")) %>%
      dplyr::filter(vapply(term, is_group_time_smooth_term, logical(1))) %>%
      dplyr::mutate(significant = NA)
  }
  
  list(
    Dataset_Info         = dataset_info,
    Group_Counts         = group_counts,
    Curve_Counts         = curve_counts,
    AIC_Table            = aic_table,
    Parametric_Terms     = param_tables,
    Smooth_Terms         = smooth_tables,
    K_Check              = k_check_tables,
    Diagnostics_Summary  = diagnostics_summary,
    Residual_ACF         = acf_tables,
    Group_Difference     = group_difference_flag,
    Model1_Summary       = model1_summary,
    Model2_Summary       = model2_summary,
    Model3_Summary       = model3_summary,
    Model1_Gam_Check     = model1_gam_check,
    Model2_Gam_Check     = model2_gam_check,
    Model3_Gam_Check     = model3_gam_check
  )
}

safe_run_bam_one_file <- function(file_path,
                                  analysis_mode,
                                  degree_global_model1 = 12,
                                  degree_global_model2 = 12,
                                  degree_fs_patient_model2 = 6,
                                  degree_global_model3 = 12,
                                  degree_fs_patient_model3 = 6,
                                  degree_fs_curve_model3 = 6,
                                  rho_value = 0) {
  tryCatch(
    {
      out <- run_bam_one_file(
        file_path = file_path,
        analysis_mode = analysis_mode,
        degree_global_model1 = degree_global_model1,
        degree_global_model2 = degree_global_model2,
        degree_fs_patient_model2 = degree_fs_patient_model2,
        degree_global_model3 = degree_global_model3,
        degree_fs_patient_model3 = degree_fs_patient_model3,
        degree_fs_curve_model3 = degree_fs_curve_model3,
        rho_value = rho_value
      )
      
      list(
        success = TRUE,
        file_name = basename(file_path),
        analysis_mode = analysis_mode,
        result = out,
        error_message = NA_character_
      )
    },
    error = function(e) {
      message("[ERROR] File: ", basename(file_path), " | Mode: ", analysis_mode)
      message("[ERROR] ", conditionMessage(e))
      
      list(
        success = FALSE,
        file_name = basename(file_path),
        analysis_mode = analysis_mode,
        result = NULL,
        error_message = conditionMessage(e)
      )
    }
  )
}

# ============================================================
# One worker job = one (file, mode) pair
# ============================================================
process_one_file_one_mode <- function(file_path,
                                      analysis_mode,
                                      degree_global_candidates = 3:20,
                                      degree_fs_patient_candidates = 3:20,
                                      degree_fs_curve_candidates = 3:20,
                                      fs_patient_to_global_ratio = 1.5,
                                      fs_curve_to_global_ratio = 1.5,
                                      rho_value = 0,
                                      k_index_threshold = 0.90,
                                      p_value_threshold = 0.05,
                                      fallback_degree_global = 20,
                                      fallback_degree_fs_patient = 20,
                                      fallback_degree_fs_curve = 20,
                                      fs_aic_tolerance = 2,
                                      fs_edf_ratio_threshold = 0.90,
                                      degree_fs_patient_for_model2_global_screening = 6,
                                      degree_fs_patient_for_model3_global_screening = 6,
                                      degree_fs_curve_for_model3_global_screening = 6,
                                      verbose_k_debug = FALSE) {
  file_name <- basename(file_path)
  
  selection_res <- tryCatch(
    {
      prep <- prepare_bam_data_one_file(file_path, analysis_mode)
      
      select_k_all_models_one_file(
        df_prepared = prep$data,
        file_name = file_name,
        analysis_mode = analysis_mode,
        degree_global_candidates = degree_global_candidates,
        degree_fs_patient_candidates = degree_fs_patient_candidates,
        degree_fs_curve_candidates = degree_fs_curve_candidates,
        fs_patient_to_global_ratio = fs_patient_to_global_ratio,
        fs_curve_to_global_ratio = fs_curve_to_global_ratio,
        rho_value = rho_value,
        k_index_threshold = k_index_threshold,
        p_value_threshold = p_value_threshold,
        fallback_degree_global = fallback_degree_global,
        fallback_degree_fs_patient = fallback_degree_fs_patient,
        fallback_degree_fs_curve = fallback_degree_fs_curve,
        fs_aic_tolerance = fs_aic_tolerance,
        fs_edf_ratio_threshold = fs_edf_ratio_threshold,
        degree_fs_patient_for_model2_global_screening = degree_fs_patient_for_model2_global_screening,
        degree_fs_patient_for_model3_global_screening = degree_fs_patient_for_model3_global_screening,
        degree_fs_curve_for_model3_global_screening = degree_fs_curve_for_model3_global_screening,
        verbose_k_debug = verbose_k_debug
      )
    },
    error = function(e) {
      list(
        selection_grid = data.frame(
          file_name = file_name,
          analysis_mode = analysis_mode,
          model = NA_character_,
          degree_global = NA_integer_,
          degree_fs_patient = NA_integer_,
          degree_fs_curve = NA_integer_,
          note = paste("Selection step failed:", conditionMessage(e)),
          stringsAsFactors = FALSE
        ),
        selected_parameters = data.frame(
          file_name = file_name,
          analysis_mode = analysis_mode,
          model = c("model1", "model2", "model3"),
          degree_global = c(fallback_degree_global, fallback_degree_global, fallback_degree_global),
          degree_fs_patient = c(NA_integer_, fallback_degree_fs_patient, fallback_degree_fs_patient),
          degree_fs_curve = c(NA_integer_, NA_integer_, fallback_degree_fs_curve),
          degree_global_selection_status = paste("Fallback used after selection error:", conditionMessage(e)),
          degree_fs_patient_selection_status = c(
            NA_character_,
            paste("Fallback used after selection error:", conditionMessage(e)),
            paste("Fallback used after selection error:", conditionMessage(e))
          ),
          degree_fs_curve_selection_status = c(
            NA_character_,
            NA_character_,
            paste("Fallback used after selection error:", conditionMessage(e))
          ),
          selection_status = paste("Fallback used after selection error:", conditionMessage(e)),
          stringsAsFactors = FALSE
        )
      )
    }
  )
  
  selected_params <- selection_res$selected_parameters
  
  sel_m1 <- selected_params %>% dplyr::filter(model == "model1") %>% dplyr::slice(1)
  sel_m2 <- selected_params %>% dplyr::filter(model == "model2") %>% dplyr::slice(1)
  sel_m3 <- selected_params %>% dplyr::filter(model == "model3") %>% dplyr::slice(1)
  
  fit_res <- safe_run_bam_one_file(
    file_path = file_path,
    analysis_mode = analysis_mode,
    degree_global_model1 = sel_m1$degree_global[[1]],
    degree_global_model2 = sel_m2$degree_global[[1]],
    degree_fs_patient_model2 = sel_m2$degree_fs_patient[[1]],
    degree_global_model3 = sel_m3$degree_global[[1]],
    degree_fs_patient_model3 = sel_m3$degree_fs_patient[[1]],
    degree_fs_curve_model3 = sel_m3$degree_fs_curve[[1]],
    rho_value = rho_value
  )
  
  run_log <- data.frame(
    file_name = fit_res$file_name,
    analysis_mode = fit_res$analysis_mode,
    degree_global_model1 = sel_m1$degree_global[[1]],
    degree_global_model2 = sel_m2$degree_global[[1]],
    degree_fs_patient_model2 = sel_m2$degree_fs_patient[[1]],
    degree_global_model3 = sel_m3$degree_global[[1]],
    degree_fs_patient_model3 = sel_m3$degree_fs_patient[[1]],
    degree_fs_curve_model3 = sel_m3$degree_fs_curve[[1]],
    model1_selection_status = sel_m1$selection_status[[1]],
    model2_selection_status = sel_m2$selection_status[[1]],
    model3_selection_status = sel_m3$selection_status[[1]],
    status = ifelse(fit_res$success, "success", "error"),
    error_message = fit_res$error_message,
    stringsAsFactors = FALSE
  )
  
  list(
    file_name = file_name,
    analysis_mode = analysis_mode,
    selection_grid = selection_res$selection_grid,
    selected_parameters = selection_res$selected_parameters,
    fit_success = fit_res$success,
    fit_result = fit_res$result,
    run_log = run_log
  )
}

# ============================================================
# Result container helpers
# ============================================================
init_results_container <- function() {
  list(
    Selected_Parameters = list(),
    K_Selection_Grid    = list(),
    Dataset_Info        = list(),
    Group_Counts        = list(),
    Curve_Counts        = list(),
    AIC_Table           = list(),
    Parametric_Terms    = list(),
    Smooth_Terms        = list(),
    K_Check             = list(),
    Diagnostics_Summary = list(),
    Residual_ACF        = list(),
    Group_Difference    = list(),
    Model1_Summary      = list(),
    Model2_Summary      = list(),
    Model3_Summary      = list(),
    Model1_Gam_Check    = list(),
    Model2_Gam_Check    = list(),
    Model3_Gam_Check    = list()
  )
}

# ============================================================
# Build one workbook for one mode from already-computed job results
# ============================================================
write_mode_workbook_from_results <- function(per_job_results,
                                             analysis_mode,
                                             output_folder = ".") {
  all_results <- init_results_container()
  
  message("\n[WORKBOOK] Writing mode: ", analysis_mode,
          " | completed jobs = ", length(per_job_results))
  
  for (res in per_job_results) {
    all_results$K_Selection_Grid[[length(all_results$K_Selection_Grid) + 1]] <- res$selection_grid
    all_results$Selected_Parameters[[length(all_results$Selected_Parameters) + 1]] <- res$selected_parameters
    
    if (isTRUE(res$fit_success)) {
      for (nm in setdiff(names(all_results), c("Selected_Parameters", "K_Selection_Grid"))) {
        all_results[[nm]][[length(all_results[[nm]]) + 1]] <- res$fit_result[[nm]]
      }
    }
  }
  
  final_sheets <- lapply(all_results, safe_bind_rows)
  run_log_df <- safe_bind_rows(lapply(per_job_results, function(x) x$run_log))
  
  about_sheet <- data.frame(
    sheet_name = c(
      "About_Sheets",
      "Selected_Parameters",
      "K_Selection_Grid",
      "Dataset_Info",
      "Group_Counts",
      "Curve_Counts",
      "AIC_Table",
      "Parametric_Terms",
      "Smooth_Terms",
      "K_Check",
      "Diagnostics_Summary",
      "Residual_ACF",
      "Group_Difference",
      "Model1_Summary",
      "Model2_Summary",
      "Model3_Summary",
      "Model1_Gam_Check",
      "Model2_Gam_Check",
      "Model3_Gam_Check",
      "Run_Log"
    ),
    description = c(
      paste("Workbook with BAM analyses per file. Mode:", analysis_mode),
      "Final selected k values for each file and each model. Model1 has its own degree_global. Model2 has its own degree_global and degree_fs_patient. Model3 has its own degree_global, degree_fs_patient, and degree_fs_curve.",
      "Selection grid for each file and each model. Includes pass/fail/inconclusive notes for global k selection and AIC/fs diagnostics for patient and curve fs selections.",
      "Basic metadata for each file using final selected k values.",
      "Number of rows per group.",
      "Number of curves per group.",
      "AIC comparison of final models using model-specific selected k values.",
      "Parametric terms from summary().",
      "Smooth terms from summary().",
      "k.check() output for final models.",
      "Compact diagnostics by file and model.",
      "Residual ACF computed separately within each curve.",
      "Quick significance table for group-specific smooths.",
      "Plain-text summary(model1).",
      "Plain-text summary(model2).",
      "Plain-text summary(model3).",
      "Plain-text gam.check(model1).",
      "Plain-text gam.check(model2).",
      "Plain-text gam.check(model3).",
      "Success/error log with selected k values."
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
    paste0("BAM_", analysis_mode, "_selected_mutants_model_comparison.xlsx")
  )
  
  writexl::write_xlsx(workbook, path = output_file)
  output_file
}

# ============================================================
# Run all (file, mode) jobs in one parallel layer, then write
# one workbook per mode
# ============================================================
build_all_workbooks_parallel <- function(files,
                                         analysis_modes,
                                         degree_global_candidates = 3:20,
                                         degree_fs_patient_candidates = 3:20,
                                         degree_fs_curve_candidates = 3:20,
                                         fs_patient_to_global_ratio = 1.5,
                                         fs_curve_to_global_ratio = 1.5,
                                         rho_value = 0,
                                         k_index_threshold = 0.90,
                                         p_value_threshold = 0.05,
                                         fallback_degree_global = 20,
                                         fallback_degree_fs_patient = 20,
                                         fallback_degree_fs_curve = 20,
                                         fs_aic_tolerance = 2,
                                         fs_edf_ratio_threshold = 0.90,
                                         degree_fs_patient_for_model2_global_screening = 6,
                                         degree_fs_patient_for_model3_global_screening = 6,
                                         degree_fs_curve_for_model3_global_screening = 6,
                                         verbose_k_debug = FALSE,
                                         output_folder = ".",
                                         workers = 6) {
  jobs <- tidyr::expand_grid(
    file_path = files,
    analysis_mode = analysis_modes
  ) %>%
    dplyr::slice_sample(prop = 1)
  
  message("\n[PARALLEL] Starting combined file x mode processing")
  message("[PARALLEL] total files = ", length(files))
  message("[PARALLEL] total modes = ", length(analysis_modes))
  message("[PARALLEL] total jobs = ", nrow(jobs))
  message("[PARALLEL] workers = ", workers)
  
  old_plan <- future::plan()
  on.exit(future::plan(old_plan), add = TRUE)
  
  future::plan(future::multisession, workers = workers)
  
  job_results <- progressr::with_progress({
    p <- progressr::progressor(steps = nrow(jobs))
    
    future.apply::future_lapply(
      seq_len(nrow(jobs)),
      function(i) {
        res <- process_one_file_one_mode(
          file_path = jobs$file_path[[i]],
          analysis_mode = jobs$analysis_mode[[i]],
          degree_global_candidates = degree_global_candidates,
          degree_fs_patient_candidates = degree_fs_patient_candidates,
          degree_fs_curve_candidates = degree_fs_curve_candidates,
          fs_patient_to_global_ratio = fs_patient_to_global_ratio,
          fs_curve_to_global_ratio = fs_curve_to_global_ratio,
          rho_value = rho_value,
          k_index_threshold = k_index_threshold,
          p_value_threshold = p_value_threshold,
          fallback_degree_global = fallback_degree_global,
          fallback_degree_fs_patient = fallback_degree_fs_patient,
          fallback_degree_fs_curve = fallback_degree_fs_curve,
          fs_aic_tolerance = fs_aic_tolerance,
          fs_edf_ratio_threshold = fs_edf_ratio_threshold,
          degree_fs_patient_for_model2_global_screening = degree_fs_patient_for_model2_global_screening,
          degree_fs_patient_for_model3_global_screening = degree_fs_patient_for_model3_global_screening,
          degree_fs_curve_for_model3_global_screening = degree_fs_curve_for_model3_global_screening,
          verbose_k_debug = verbose_k_debug
        )
        
        p(sprintf("%s | %s", basename(jobs$file_path[[i]]), jobs$analysis_mode[[i]]))
        res
      },
      future.seed = TRUE,
      future.scheduling = Inf,
      future.packages = c(
        "readxl", "writexl", "dplyr", "tidyr", "stringr",
        "mgcv", "tibble", "purrr", "gtools"
      )
    )
  })
  
  output_files <- lapply(
    analysis_modes,
    function(mode) {
      mode_results <- job_results[vapply(job_results, function(x) identical(x$analysis_mode, mode), logical(1))]
      
      write_mode_workbook_from_results(
        per_job_results = mode_results,
        analysis_mode = mode,
        output_folder = output_folder
      )
    }
  )
  
  output_files
}

# ============================================================
# Find Excel files in natural order
# ============================================================
excel_files <- list.files(
  path = input_folder,
  pattern = "\\.(xlsx|xls)$",
  full.names = TRUE,
  ignore.case = TRUE
)

excel_files <- excel_files[!grepl("^~\\$", basename(excel_files))]
excel_files <- excel_files[mixedorder(basename(excel_files))]

if (length(excel_files) == 0) {
  stop("No Excel files found in input_folder.")
}

# ============================================================
# Run all (file, mode) jobs in one parallel layer
# ============================================================
output_files <- build_all_workbooks_parallel(
  files = excel_files,
  analysis_modes = analysis_modes,
  degree_global_candidates = degree_global_candidates,
  degree_fs_patient_candidates = degree_fs_patient_candidates,
  degree_fs_curve_candidates = degree_fs_curve_candidates,
  fs_patient_to_global_ratio = fs_patient_to_global_ratio,
  fs_curve_to_global_ratio = fs_curve_to_global_ratio,
  rho_value = rho_value,
  k_index_threshold = k_index_threshold,
  p_value_threshold = p_value_threshold,
  fallback_degree_global = fallback_degree_global,
  fallback_degree_fs_patient = fallback_degree_fs_patient,
  fallback_degree_fs_curve = fallback_degree_fs_curve,
  fs_aic_tolerance = fs_aic_tolerance,
  fs_edf_ratio_threshold = fs_edf_ratio_threshold,
  degree_fs_patient_for_model2_global_screening = degree_fs_patient_for_model2_global_screening,
  degree_fs_patient_for_model3_global_screening = degree_fs_patient_for_model3_global_screening,
  degree_fs_curve_for_model3_global_screening = degree_fs_curve_for_model3_global_screening,
  verbose_k_debug = verbose_k_debug,
  output_folder = output_folder,
  workers = workers
)

print(output_files)
cat("Done.\n")