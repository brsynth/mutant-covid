library(readxl)
library(writexl)
library(dplyr)
library(tidyr)
library(stringr)
library(mgcv)
library(tibble)
library(purrr)
library(gtools)

# ============================================================
# Settings
# ============================================================
data_folder <- "time_series_pools"
input_file <- file.path(data_folder, "Summary - Pools.xlsx")
output_folder <- file.path(data_folder, "bam_model_selection")
dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)

replicate_sheets <- c(
  "Curves - Replicate 1",
  "Curves - Replicate 2",
  "Curves - Replicate 3"
)

analysis_modes <- c("MS", "NP", "NS", "MN", "three_groups")

# Search ranges for k
degree_global_candidates <- 3:20
degree_fs_candidates <- 3:20

# Constraint: degree_fs cannot be > 150% of degree_global
fs_to_global_ratio <- 1.5

# k.check acceptance rule for GLOBAL smooths only
k_index_threshold <- 0.90
p_value_threshold <- 0.05

# Fallbacks
fallback_degree_global <- 20
fallback_degree_fs <- 20

# Additional rule for fs tuning
fs_aic_tolerance <- 2
fs_edf_ratio_threshold <- 0.90

rho_value <- 0

# Used only while screening model3 degree_global
degree_fs_for_model3_global_screening <- 6

# If TRUE, prints k.check row names/columns during selection
verbose_k_debug <- FALSE

# Residual type for ACF diagnostics
acf_residual_type <- "pearson"

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

smooth_table_from_bam <- function(model_obj, model_name, strain_name, analysis_mode) {
  sm <- summary(model_obj)$s.table
  if (is.null(sm)) return(data.frame())
  
  out <- as.data.frame(sm)
  out$term <- rownames(sm)
  rownames(out) <- NULL
  out$model <- model_name
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  
  out %>%
    dplyr::select(strain, analysis_mode, model, term, dplyr::everything())
}

param_table_from_bam <- function(model_obj, model_name, strain_name, analysis_mode) {
  pt <- summary(model_obj)$p.table
  if (is.null(pt)) return(data.frame())
  
  out <- as.data.frame(pt)
  out$term <- rownames(pt)
  rownames(out) <- NULL
  out$model <- model_name
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  
  out %>%
    dplyr::select(strain, analysis_mode, model, term, dplyr::everything())
}

diagnostic_summary_from_bam <- function(model_obj, model_name, strain_name, analysis_mode) {
  res <- residuals(model_obj, type = "deviance")
  fit <- fitted(model_obj)
  
  data.frame(
    strain = strain_name,
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
# FIXED ACF FUNCTIONS
# ============================================================
acf_table_from_bam <- function(model_obj,
                               df_used,
                               model_name,
                               strain_name,
                               analysis_mode,
                               lag.max = 10,
                               residual_type = "pearson") {
  res <- try(residuals(model_obj, type = residual_type), silent = TRUE)
  
  if (inherits(res, "try-error") || length(res) != nrow(df_used)) {
    return(data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      model = model_name,
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
  
  split_list <- tmp %>% dplyr::group_split(curve)
  
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
    
    if (inherits(acf_obj, "try-error")) return(NULL)
    
    data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      model = model_name,
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
      model = model_name,
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

acf_summary_from_curve_table <- function(acf_curve_table) {
  required_cols <- c("strain", "analysis_mode", "model", "lag", "acf", "curve", "residual_type")
  
  if (!all(required_cols %in% names(acf_curve_table))) {
    return(data.frame(
      strain = NA_character_,
      analysis_mode = NA_character_,
      model = NA_character_,
      lag = NA_real_,
      mean_acf = NA_real_,
      median_acf = NA_real_,
      sd_acf = NA_real_,
      min_acf = NA_real_,
      max_acf = NA_real_,
      n_curves = NA_integer_,
      residual_type = NA_character_,
      note = "Input table missing required columns",
      stringsAsFactors = FALSE
    ))
  }
  
  valid_rows <- acf_curve_table %>%
    dplyr::filter(!is.na(lag), !is.na(acf), !is.na(curve))
  
  if (nrow(valid_rows) == 0) {
    return(data.frame(
      strain = unique(acf_curve_table$strain)[1],
      analysis_mode = unique(acf_curve_table$analysis_mode)[1],
      model = unique(acf_curve_table$model)[1],
      lag = NA_real_,
      mean_acf = NA_real_,
      median_acf = NA_real_,
      sd_acf = NA_real_,
      min_acf = NA_real_,
      max_acf = NA_real_,
      n_curves = NA_integer_,
      residual_type = unique(acf_curve_table$residual_type)[1],
      note = "No valid per-curve ACF values available",
      stringsAsFactors = FALSE
    ))
  }
  
  valid_rows %>%
    dplyr::group_by(strain, analysis_mode, model, lag, residual_type) %>%
    dplyr::summarise(
      mean_acf = mean(acf, na.rm = TRUE),
      median_acf = median(acf, na.rm = TRUE),
      sd_acf = sd(acf, na.rm = TRUE),
      min_acf = min(acf, na.rm = TRUE),
      max_acf = max(acf, na.rm = TRUE),
      n_curves = dplyr::n_distinct(curve),
      note = NA_character_,
      .groups = "drop"
    )
}

k_check_table_from_bam <- function(model_obj, model_name, strain_name, analysis_mode) {
  kc <- try(mgcv::k.check(model_obj), silent = TRUE)
  
  if (inherits(kc, "try-error") || is.null(kc)) {
    return(data.frame(
      strain = strain_name,
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
  out$strain <- strain_name
  out$analysis_mode <- analysis_mode
  
  out %>%
    dplyr::select(strain, analysis_mode, model, term, dplyr::everything())
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
  
  # Exclude obvious random/curve-specific terms
  has_replicate <- grepl("replicate", x)
  has_curve <- grepl("curve", x)
  
  has_time && !has_replicate && !has_curve
}

is_fs_smooth_term <- function(term_name) {
  x <- normalize_term(term_name)
  grepl("time", x) && grepl("curve", x)
}

is_group_time_smooth_term <- function(term_name) {
  x <- normalize_term(term_name)
  grepl("time", x) && grepl("group", x) && !grepl("curve", x)
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

fs_term_diagnostics_from_model3 <- function(model3, degree_fs) {
  sm_df <- extract_smooth_summary_table(model3)
  
  if (is.null(sm_df) || nrow(sm_df) == 0) {
    return(data.frame(
      term = NA_character_,
      edf_std = NA_real_,
      fs_k_candidate = degree_fs,
      edf_over_k_candidate_minus1 = NA_real_,
      fs_significant = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  fs_row <- sm_df %>%
    dplyr::filter(vapply(term, is_fs_smooth_term, logical(1)))
  
  if (nrow(fs_row) == 0) {
    return(data.frame(
      term = NA_character_,
      edf_std = NA_real_,
      fs_k_candidate = degree_fs,
      edf_over_k_candidate_minus1 = NA_real_,
      fs_significant = NA,
      stringsAsFactors = FALSE
    ))
  }
  
  fs_row <- fs_row %>% dplyr::slice(1)
  denom <- max(degree_fs - 1, 1)
  
  data.frame(
    term = fs_row$term,
    edf_std = fs_row$edf_std,
    fs_k_candidate = degree_fs,
    edf_over_k_candidate_minus1 = fs_row$edf_std / denom,
    fs_significant = ifelse(is.na(fs_row$p_value_std), NA, fs_row$p_value_std < 0.05),
    stringsAsFactors = FALSE
  )
}

# ============================================================
# Read workbook and build one long dataset
# ============================================================
read_one_replicate_sheet <- function(file_path, sheet_name) {
  replicate_id <- str_extract(sheet_name, "\\d+$")
  
  df <- readxl::read_excel(file_path, sheet = sheet_name) %>%
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
  
  df
}

all_data <- purrr::map_dfr(replicate_sheets, ~ read_one_replicate_sheet(input_file, .x))

strain_levels <- gtools::mixedsort(unique(all_data$strain))

all_data <- all_data %>%
  dplyr::mutate(
    strain = factor(strain, levels = strain_levels)
  )

print(levels(all_data$strain))
print(levels(droplevels(all_data$original_group)))
print(table(all_data$replicate, useNA = "ifany"))

# ============================================================
# Prepare one strain + one mode dataset once
# ============================================================
prepare_bam_data_one_strain <- function(df_strain, strain_name, analysis_mode) {
  df <- df_strain %>%
    dplyr::filter(!is.na(OD), !is.na(time), !is.na(original_group), !is.na(replicate))
  
  if (analysis_mode == "MS") {
    df <- df %>%
      dplyr::filter(original_group %in% c("Mild", "Severe")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("Mild", "Severe")))
  }
  
  if (analysis_mode == "NP") {
    df <- df %>%
      dplyr::filter(original_group %in% c("Negative", "Mild", "Severe")) %>%
      dplyr::mutate(
        Group = dplyr::case_when(
          original_group == "Negative" ~ "Negative",
          original_group %in% c("Mild", "Severe") ~ "P"
        ),
        Group = factor(Group, levels = c("Negative", "P"))
      )
  }
  
  if (analysis_mode == "NS") {
    df <- df %>%
      dplyr::filter(original_group %in% c("Negative", "Severe")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("Negative", "Severe")))
  }
  
  if (analysis_mode == "MN") {
    df <- df %>%
      dplyr::filter(original_group %in% c("Mild", "Negative")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("Mild", "Negative")))
  }
  
  if (analysis_mode == "three_groups") {
    df <- df %>%
      dplyr::filter(original_group %in% c("Negative", "Mild", "Severe")) %>%
      dplyr::mutate(Group = factor(original_group, levels = c("Negative", "Mild", "Severe")))
  }
  
  df <- df %>%
    dplyr::mutate(
      Group = droplevels(Group),
      replicate = droplevels(replicate),
      curve = interaction(replicate, Group, drop = TRUE)
    ) %>%
    dplyr::arrange(curve, time)
  
  if (nrow(df) == 0) stop("No usable rows.")
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
    dplyr::ungroup()
  
  dataset_info <- data.frame(
    strain = strain_name,
    analysis_mode = analysis_mode,
    common_max_time = common_max_time,
    n_rows = nrow(df),
    n_groups = dplyr::n_distinct(df$Group),
    n_replicates = dplyr::n_distinct(df$replicate),
    n_curves = dplyr::n_distinct(df$curve),
    min_time = min(df$time, na.rm = TRUE),
    max_time = max(df$time, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  
  group_counts <- df %>%
    dplyr::count(Group, name = "n_rows") %>%
    dplyr::mutate(strain = strain_name, analysis_mode = analysis_mode) %>%
    dplyr::select(strain, analysis_mode, dplyr::everything())
  
  curve_counts <- df %>%
    dplyr::distinct(Group, replicate, curve) %>%
    dplyr::count(Group, name = "n_curves") %>%
    dplyr::mutate(strain = strain_name, analysis_mode = analysis_mode) %>%
    dplyr::select(strain, analysis_mode, dplyr::everything())
  
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
      s(time, k = degree_global) +
      s(replicate, bs = "re"),
    data = df,
    method = "fREML",
    rho = rho_value,
    AR.start = df$ar_start
  )
}

fit_model2 <- function(df, degree_global, rho_value = 0) {
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
}

fit_model3 <- function(df, degree_global, degree_fs, rho_value = 0) {
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
}

# ============================================================
# Selection helpers
# ============================================================
selection_note_from_check <- function(chk) {
  if (isTRUE(chk$pass)) return("Accepted by k.check on global smooths")
  if (identical(chk$pass, FALSE)) return("Rejected by k.check on global smooths")
  paste("k.check inconclusive:", chk$status)
}

first_nonmissing_pass_degree <- function(tried_df, fallback_degree_global) {
  pass_rows <- tried_df %>% dplyr::filter(pass %in% TRUE)
  if (nrow(pass_rows) > 0) return(pass_rows$degree_global[1])
  fallback_degree_global
}

final_global_selection_status <- function(tried_df) {
  if (any(tried_df$pass %in% TRUE, na.rm = TRUE)) {
    "Accepted by k.check on global smooths"
  } else if (any(is.na(tried_df$pass))) {
    "Fallback used after inconclusive k.check"
  } else {
    "Fallback used"
  }
}

# ============================================================
# Model-specific k selection
# ============================================================
select_degree_global_model1_one_strain <- function(df_prepared,
                                                   strain_name,
                                                   analysis_mode,
                                                   degree_global_candidates = 6:20,
                                                   rho_value = 0,
                                                   k_index_threshold = 0.90,
                                                   p_value_threshold = 0.05,
                                                   fallback_degree_global = 12,
                                                   verbose_k_debug = FALSE) {
  tried_rows <- list()
  counter <- 0
  selected_degree_global <- NULL
  
  for (kg in degree_global_candidates) {
    counter <- counter + 1
    
    message("[SELECT model1 degree_global] Strain: ", strain_name,
            " | Mode: ", analysis_mode,
            " | Trying degree_global=", kg)
    
    fit <- try(
      fit_model1(df_prepared, degree_global = kg, rho_value = rho_value),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model1",
        degree_global = kg,
        degree_fs = NA_integer_,
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
      strain = strain_name,
      analysis_mode = analysis_mode,
      model = "model1",
      degree_global = kg,
      degree_fs = NA_integer_,
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
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model1",
        degree_global = fallback_degree_global,
        degree_fs = NA_integer_,
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
    strain = strain_name,
    analysis_mode = analysis_mode,
    model = "model1",
    degree_global = selected_degree_global,
    degree_fs = NA_integer_,
    degree_global_selection_status = final_global_selection_status(tried_df),
    degree_fs_selection_status = NA_character_,
    selection_status = final_global_selection_status(tried_df),
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

select_degree_global_model2_one_strain <- function(df_prepared,
                                                   strain_name,
                                                   analysis_mode,
                                                   degree_global_candidates = 6:20,
                                                   rho_value = 0,
                                                   k_index_threshold = 0.90,
                                                   p_value_threshold = 0.05,
                                                   fallback_degree_global = 12,
                                                   verbose_k_debug = FALSE) {
  tried_rows <- list()
  counter <- 0
  selected_degree_global <- NULL
  
  for (kg in degree_global_candidates) {
    counter <- counter + 1
    
    message("[SELECT model2 degree_global] Strain: ", strain_name,
            " | Mode: ", analysis_mode,
            " | Trying degree_global=", kg)
    
    fit <- try(
      fit_model2(df_prepared, degree_global = kg, rho_value = rho_value),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model2",
        degree_global = kg,
        degree_fs = NA_integer_,
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
      strain = strain_name,
      analysis_mode = analysis_mode,
      model = "model2",
      degree_global = kg,
      degree_fs = NA_integer_,
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
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model2",
        degree_global = fallback_degree_global,
        degree_fs = NA_integer_,
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
    strain = strain_name,
    analysis_mode = analysis_mode,
    model = "model2",
    degree_global = selected_degree_global,
    degree_fs = NA_integer_,
    degree_global_selection_status = final_global_selection_status(tried_df),
    degree_fs_selection_status = NA_character_,
    selection_status = final_global_selection_status(tried_df),
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

select_degree_global_model3_one_strain <- function(df_prepared,
                                                   strain_name,
                                                   analysis_mode,
                                                   degree_global_candidates = 6:20,
                                                   degree_fs_for_screening = 6,
                                                   rho_value = 0,
                                                   k_index_threshold = 0.90,
                                                   p_value_threshold = 0.05,
                                                   fallback_degree_global = 12,
                                                   verbose_k_debug = FALSE) {
  tried_rows <- list()
  counter <- 0
  selected_degree_global <- NULL
  
  for (kg in degree_global_candidates) {
    counter <- counter + 1
    
    message("[SELECT model3 degree_global] Strain: ", strain_name,
            " | Mode: ", analysis_mode,
            " | Trying degree_global=", kg,
            " | Screening degree_fs=", degree_fs_for_screening)
    
    fit <- try(
      fit_model3(
        df_prepared,
        degree_global = kg,
        degree_fs = degree_fs_for_screening,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model3",
        degree_global = kg,
        degree_fs = degree_fs_for_screening,
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
      strain = strain_name,
      analysis_mode = analysis_mode,
      model = "model3",
      degree_global = kg,
      degree_fs = degree_fs_for_screening,
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
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model3",
        degree_global = fallback_degree_global,
        degree_fs = degree_fs_for_screening,
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
    strain = strain_name,
    analysis_mode = analysis_mode,
    model = "model3",
    degree_global = selected_degree_global,
    degree_fs = NA_integer_,
    degree_global_selection_status = if (any(tried_df$pass %in% TRUE, na.rm = TRUE)) {
      "Accepted by k.check on global smooths in model3"
    } else if (any(is.na(tried_df$pass))) {
      "Fallback used after inconclusive k.check in model3"
    } else {
      "Fallback used"
    },
    degree_fs_selection_status = NA_character_,
    selection_status = if (any(tried_df$pass %in% TRUE, na.rm = TRUE)) {
      "Accepted by k.check on global smooths in model3"
    } else if (any(is.na(tried_df$pass))) {
      "Fallback used after inconclusive k.check in model3"
    } else {
      "Fallback used"
    },
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

select_degree_fs_model3_one_strain <- function(df_prepared,
                                               strain_name,
                                               analysis_mode,
                                               degree_global,
                                               degree_fs_candidates = 3:12,
                                               fs_to_global_ratio = 1.5,
                                               rho_value = 0,
                                               fallback_degree_fs = 6,
                                               fs_aic_tolerance = 2,
                                               fs_edf_ratio_threshold = 0.90) {
  tried_rows <- list()
  counter <- 0
  
  for (kf in degree_fs_candidates) {
    if (kf > floor(fs_to_global_ratio * degree_global)) next
    
    counter <- counter + 1
    
    message("[SELECT model3 degree_fs] Strain: ", strain_name,
            " | Mode: ", analysis_mode,
            " | degree_global=", degree_global,
            " | Trying degree_fs=", kf)
    
    fit3 <- try(
      fit_model3(
        df_prepared,
        degree_global = degree_global,
        degree_fs = kf,
        rho_value = rho_value
      ),
      silent = TRUE
    )
    
    if (inherits(fit3, "try-error")) {
      tried_rows[[counter]] <- data.frame(
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model3",
        degree_global = degree_global,
        degree_fs = kf,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = FALSE,
        note = "Model3 fit failed",
        stringsAsFactors = FALSE
      )
      next
    }
    
    fs_diag <- fs_term_diagnostics_from_model3(fit3, degree_fs = kf)
    
    aic_val <- try(AIC(fit3), silent = TRUE)
    if (inherits(aic_val, "try-error")) aic_val <- NA_real_
    
    fs_ratio_ok <- !is.na(fs_diag$edf_over_k_candidate_minus1[1]) &&
      fs_diag$edf_over_k_candidate_minus1[1] < fs_edf_ratio_threshold
    
    tried_rows[[counter]] <- data.frame(
      strain = strain_name,
      analysis_mode = analysis_mode,
      model = "model3",
      degree_global = degree_global,
      degree_fs = kf,
      AIC = as.numeric(aic_val),
      fs_term = fs_diag$term[1],
      fs_edf = fs_diag$edf_std[1],
      fs_edf_ratio = fs_diag$edf_over_k_candidate_minus1[1],
      fs_ratio_ok = fs_ratio_ok,
      selected = FALSE,
      note = if (is.na(fs_diag$term[1])) {
        "Candidate evaluated; fs term could not be identified robustly"
      } else {
        "Candidate evaluated"
      },
      stringsAsFactors = FALSE
    )
  }
  
  tried_df <- dplyr::bind_rows(tried_rows)
  
  if (nrow(tried_df) == 0 || all(is.na(tried_df$AIC))) {
    selected_degree_fs <- fallback_degree_fs
    
    tried_df <- dplyr::bind_rows(
      tried_df,
      data.frame(
        strain = strain_name,
        analysis_mode = analysis_mode,
        model = "model3",
        degree_global = degree_global,
        degree_fs = fallback_degree_fs,
        AIC = NA_real_,
        fs_term = NA_character_,
        fs_edf = NA_real_,
        fs_edf_ratio = NA_real_,
        fs_ratio_ok = NA,
        selected = TRUE,
        note = "Fallback degree_fs used because no candidate could be evaluated",
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
      dplyr::arrange(degree_fs, AIC)
    
    selected_degree_fs <- preferred_pool$degree_fs[1]
    
    tried_df <- tried_df %>%
      dplyr::mutate(selected = degree_fs == selected_degree_fs)
    
    selected_status <- if (any(tried_df$fs_ratio_ok %in% TRUE, na.rm = TRUE)) {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, ") with fs edf-ratio preference")
    } else {
      paste0("Selected by AIC tolerance (", fs_aic_tolerance, "); no candidate met fs edf-ratio preference")
    }
  }
  
  selected_df <- data.frame(
    strain = strain_name,
    analysis_mode = analysis_mode,
    model = "model3",
    degree_global = degree_global,
    degree_fs = selected_degree_fs,
    degree_global_selection_status = NA_character_,
    degree_fs_selection_status = selected_status,
    selection_status = selected_status,
    stringsAsFactors = FALSE
  )
  
  list(
    selection_grid = tried_df,
    selected_parameters = selected_df
  )
}

select_k_all_models_one_strain <- function(df_prepared,
                                           strain_name,
                                           analysis_mode,
                                           degree_global_candidates = 6:20,
                                           degree_fs_candidates = 3:12,
                                           fs_to_global_ratio = 1.5,
                                           rho_value = 0,
                                           k_index_threshold = 0.90,
                                           p_value_threshold = 0.05,
                                           fallback_degree_global = 12,
                                           fallback_degree_fs = 6,
                                           fs_aic_tolerance = 2,
                                           fs_edf_ratio_threshold = 0.90,
                                           degree_fs_for_model3_global_screening = 6,
                                           verbose_k_debug = FALSE) {
  sel_m1 <- select_degree_global_model1_one_strain(
    df_prepared = df_prepared,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    degree_global_candidates = degree_global_candidates,
    rho_value = rho_value,
    k_index_threshold = k_index_threshold,
    p_value_threshold = p_value_threshold,
    fallback_degree_global = fallback_degree_global,
    verbose_k_debug = verbose_k_debug
  )
  
  sel_m2 <- select_degree_global_model2_one_strain(
    df_prepared = df_prepared,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    degree_global_candidates = degree_global_candidates,
    rho_value = rho_value,
    k_index_threshold = k_index_threshold,
    p_value_threshold = p_value_threshold,
    fallback_degree_global = fallback_degree_global,
    verbose_k_debug = verbose_k_debug
  )
  
  sel_m3_global <- select_degree_global_model3_one_strain(
    df_prepared = df_prepared,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    degree_global_candidates = degree_global_candidates,
    degree_fs_for_screening = degree_fs_for_model3_global_screening,
    rho_value = rho_value,
    k_index_threshold = k_index_threshold,
    p_value_threshold = p_value_threshold,
    fallback_degree_global = fallback_degree_global,
    verbose_k_debug = verbose_k_debug
  )
  
  selected_degree_global_model3 <- sel_m3_global$selected_parameters$degree_global[1]
  
  sel_m3_fs <- select_degree_fs_model3_one_strain(
    df_prepared = df_prepared,
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    degree_global = selected_degree_global_model3,
    degree_fs_candidates = degree_fs_candidates,
    fs_to_global_ratio = fs_to_global_ratio,
    rho_value = rho_value,
    fallback_degree_fs = fallback_degree_fs,
    fs_aic_tolerance = fs_aic_tolerance,
    fs_edf_ratio_threshold = fs_edf_ratio_threshold
  )
  
  selected_model3 <- sel_m3_global$selected_parameters %>%
    dplyr::select(strain, analysis_mode, model, degree_global, degree_global_selection_status) %>%
    dplyr::left_join(
      sel_m3_fs$selected_parameters %>%
        dplyr::select(strain, analysis_mode, model, degree_fs, degree_fs_selection_status),
      by = c("strain", "analysis_mode", "model")
    ) %>%
    dplyr::mutate(
      selection_status = paste(
        degree_global_selection_status,
        "|",
        degree_fs_selection_status
      )
    )
  
  selected_all <- dplyr::bind_rows(
    sel_m1$selected_parameters,
    sel_m2$selected_parameters,
    selected_model3
  )
  
  grid_all <- dplyr::bind_rows(
    sel_m1$selection_grid,
    sel_m2$selection_grid,
    sel_m3_global$selection_grid,
    sel_m3_fs$selection_grid
  )
  
  list(
    selection_grid = grid_all,
    selected_parameters = selected_all
  )
}

# ============================================================
# Final fitting with model-specific selected k
# ============================================================
run_bam_one_strain <- function(df_strain,
                               strain_name,
                               analysis_mode,
                               degree_global_model1 = 12,
                               degree_global_model2 = 12,
                               degree_global_model3 = 12,
                               degree_fs_model3 = 6,
                               rho_value = 0,
                               acf_residual_type = "pearson") {
  message("\n====================================================")
  message("[START FINAL FIT] Strain: ", strain_name, " | Mode: ", analysis_mode)
  message("[START FINAL FIT] model1 degree_global = ", degree_global_model1)
  message("[START FINAL FIT] model2 degree_global = ", degree_global_model2)
  message("[START FINAL FIT] model3 degree_global = ", degree_global_model3,
          " | model3 degree_fs = ", degree_fs_model3)
  message("====================================================")
  
  prep <- prepare_bam_data_one_strain(df_strain, strain_name, analysis_mode)
  df <- prep$data
  
  dataset_info <- prep$dataset_info %>%
    dplyr::mutate(
      degree_global_model1 = degree_global_model1,
      degree_global_model2 = degree_global_model2,
      degree_global_model3 = degree_global_model3,
      degree_fs_model3 = degree_fs_model3,
      rho_value = rho_value,
      acf_residual_type = acf_residual_type
    ) %>%
    dplyr::select(
      strain, analysis_mode,
      degree_global_model1, degree_global_model2, degree_global_model3, degree_fs_model3,
      rho_value, acf_residual_type,
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
    rho_value = rho_value
  )
  
  model3 <- fit_model3(
    df = df,
    degree_global = degree_global_model3,
    degree_fs = degree_fs_model3,
    rho_value = rho_value
  )
  
  aic_table <- AIC(model1, model2, model3) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("model") %>%
    dplyr::mutate(
      strain = strain_name,
      analysis_mode = analysis_mode,
      degree_global = dplyr::case_when(
        model == "model1" ~ degree_global_model1,
        model == "model2" ~ degree_global_model2,
        model == "model3" ~ degree_global_model3,
        TRUE ~ NA_real_
      ),
      degree_fs = dplyr::case_when(
        model == "model3" ~ degree_fs_model3,
        TRUE ~ NA_real_
      )
    ) %>%
    dplyr::select(strain, analysis_mode, model, degree_global, degree_fs, dplyr::everything())
  
  param_tables <- dplyr::bind_rows(
    param_table_from_bam(model1, "model1", strain_name, analysis_mode),
    param_table_from_bam(model2, "model2", strain_name, analysis_mode),
    param_table_from_bam(model3, "model3", strain_name, analysis_mode)
  )
  
  smooth_tables <- dplyr::bind_rows(
    smooth_table_from_bam(model1, "model1", strain_name, analysis_mode),
    smooth_table_from_bam(model2, "model2", strain_name, analysis_mode),
    smooth_table_from_bam(model3, "model3", strain_name, analysis_mode)
  )
  
  diagnostics_summary <- dplyr::bind_rows(
    diagnostic_summary_from_bam(model1, "model1", strain_name, analysis_mode),
    diagnostic_summary_from_bam(model2, "model2", strain_name, analysis_mode),
    diagnostic_summary_from_bam(model3, "model3", strain_name, analysis_mode)
  )
  
  acf_curve_model1 <- acf_table_from_bam(
    model_obj = model1,
    df_used = df,
    model_name = "model1",
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    lag.max = 10,
    residual_type = acf_residual_type
  )
  
  acf_curve_model2 <- acf_table_from_bam(
    model_obj = model2,
    df_used = df,
    model_name = "model2",
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    lag.max = 10,
    residual_type = acf_residual_type
  )
  
  acf_curve_model3 <- acf_table_from_bam(
    model_obj = model3,
    df_used = df,
    model_name = "model3",
    strain_name = strain_name,
    analysis_mode = analysis_mode,
    lag.max = 10,
    residual_type = acf_residual_type
  )
  
  acf_tables_by_curve <- dplyr::bind_rows(
    acf_curve_model1,
    acf_curve_model2,
    acf_curve_model3
  )
  
  acf_tables_summary <- dplyr::bind_rows(
    acf_summary_from_curve_table(acf_curve_model1),
    acf_summary_from_curve_table(acf_curve_model2),
    acf_summary_from_curve_table(acf_curve_model3)
  )
  
  k_check_tables <- dplyr::bind_rows(
    k_check_table_from_bam(model1, "model1", strain_name, analysis_mode),
    k_check_table_from_bam(model2, "model2", strain_name, analysis_mode),
    k_check_table_from_bam(model3, "model3", strain_name, analysis_mode)
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
    Dataset_Info              = dataset_info,
    Group_Counts              = group_counts,
    Curve_Counts              = curve_counts,
    AIC_Table                 = aic_table,
    Parametric_Terms          = param_tables,
    Smooth_Terms              = smooth_tables,
    K_Check                   = k_check_tables,
    Diagnostics_Summary       = diagnostics_summary,
    Residual_ACF_By_Curve     = acf_tables_by_curve,
    Residual_ACF_Summary      = acf_tables_summary,
    Group_Difference          = group_difference_flag,
    Model1_Summary            = model1_summary,
    Model2_Summary            = model2_summary,
    Model3_Summary            = model3_summary,
    Model1_Gam_Check          = model1_gam_check,
    Model2_Gam_Check          = model2_gam_check,
    Model3_Gam_Check          = model3_gam_check
  )
}

safe_run_bam_one_strain <- function(df_strain,
                                    strain_name,
                                    analysis_mode,
                                    degree_global_model1 = 12,
                                    degree_global_model2 = 12,
                                    degree_global_model3 = 12,
                                    degree_fs_model3 = 6,
                                    rho_value = 0,
                                    acf_residual_type = "pearson") {
  tryCatch(
    {
      out <- run_bam_one_strain(
        df_strain = df_strain,
        strain_name = strain_name,
        analysis_mode = analysis_mode,
        degree_global_model1 = degree_global_model1,
        degree_global_model2 = degree_global_model2,
        degree_global_model3 = degree_global_model3,
        degree_fs_model3 = degree_fs_model3,
        rho_value = rho_value,
        acf_residual_type = acf_residual_type
      )
      
      list(
        success = TRUE,
        strain = strain_name,
        analysis_mode = analysis_mode,
        result = out,
        error_message = NA_character_
      )
    },
    error = function(e) {
      message("[ERROR] Strain: ", strain_name, " | Mode: ", analysis_mode)
      message("[ERROR] ", conditionMessage(e))
      
      list(
        success = FALSE,
        strain = strain_name,
        analysis_mode = analysis_mode,
        result = NULL,
        error_message = conditionMessage(e)
      )
    }
  )
}

# ============================================================
# Build one workbook per mode
# ============================================================
build_mode_workbook <- function(all_data,
                                analysis_mode,
                                degree_global_candidates = 6:20,
                                degree_fs_candidates = 3:12,
                                fs_to_global_ratio = 1.5,
                                rho_value = 0,
                                k_index_threshold = 0.90,
                                p_value_threshold = 0.05,
                                fallback_degree_global = 12,
                                fallback_degree_fs = 6,
                                fs_aic_tolerance = 2,
                                fs_edf_ratio_threshold = 0.90,
                                degree_fs_for_model3_global_screening = 6,
                                verbose_k_debug = FALSE,
                                output_folder = ".",
                                acf_residual_type = "pearson") {
  all_results <- list(
    Selected_Parameters      = list(),
    K_Selection_Grid         = list(),
    Dataset_Info             = list(),
    Group_Counts             = list(),
    Curve_Counts             = list(),
    AIC_Table                = list(),
    Parametric_Terms         = list(),
    Smooth_Terms             = list(),
    K_Check                  = list(),
    Diagnostics_Summary      = list(),
    Residual_ACF_By_Curve    = list(),
    Residual_ACF_Summary     = list(),
    Group_Difference         = list(),
    Model1_Summary           = list(),
    Model2_Summary           = list(),
    Model3_Summary           = list(),
    Model1_Gam_Check         = list(),
    Model2_Gam_Check         = list(),
    Model3_Gam_Check         = list()
  )
  
  run_log <- list()
  strain_list <- levels(all_data$strain)
  
  for (i in seq_along(strain_list)) {
    strain_i <- strain_list[i]
    
    message("\n[WORKBOOK] Processing strain ", i, " of ", length(strain_list),
            ": ", strain_i, " | mode = ", analysis_mode)
    
    df_strain <- all_data %>% dplyr::filter(strain == strain_i)
    
    selection_res <- tryCatch(
      {
        prep <- prepare_bam_data_one_strain(df_strain, strain_i, analysis_mode)
        
        select_k_all_models_one_strain(
          df_prepared = prep$data,
          strain_name = strain_i,
          analysis_mode = analysis_mode,
          degree_global_candidates = degree_global_candidates,
          degree_fs_candidates = degree_fs_candidates,
          fs_to_global_ratio = fs_to_global_ratio,
          rho_value = rho_value,
          k_index_threshold = k_index_threshold,
          p_value_threshold = p_value_threshold,
          fallback_degree_global = fallback_degree_global,
          fallback_degree_fs = fallback_degree_fs,
          fs_aic_tolerance = fs_aic_tolerance,
          fs_edf_ratio_threshold = fs_edf_ratio_threshold,
          degree_fs_for_model3_global_screening = degree_fs_for_model3_global_screening,
          verbose_k_debug = verbose_k_debug
        )
      },
      error = function(e) {
        list(
          selection_grid = data.frame(
            strain = strain_i,
            analysis_mode = analysis_mode,
            model = NA_character_,
            degree_global = NA_integer_,
            degree_fs = NA_integer_,
            note = paste("Selection step failed:", conditionMessage(e)),
            stringsAsFactors = FALSE
          ),
          selected_parameters = data.frame(
            strain = strain_i,
            analysis_mode = analysis_mode,
            model = c("model1", "model2", "model3"),
            degree_global = c(fallback_degree_global, fallback_degree_global, fallback_degree_global),
            degree_fs = c(NA_integer_, NA_integer_, fallback_degree_fs),
            degree_global_selection_status = paste("Fallback used after selection error:", conditionMessage(e)),
            degree_fs_selection_status = c(NA_character_, NA_character_, paste("Fallback used after selection error:", conditionMessage(e))),
            selection_status = paste("Fallback used after selection error:", conditionMessage(e)),
            stringsAsFactors = FALSE
          )
        )
      }
    )
    
    all_results$K_Selection_Grid[[length(all_results$K_Selection_Grid) + 1]] <- selection_res$selection_grid
    all_results$Selected_Parameters[[length(all_results$Selected_Parameters) + 1]] <- selection_res$selected_parameters
    
    selected_params <- selection_res$selected_parameters
    
    sel_m1 <- selected_params %>% dplyr::filter(model == "model1") %>% dplyr::slice(1)
    sel_m2 <- selected_params %>% dplyr::filter(model == "model2") %>% dplyr::slice(1)
    sel_m3 <- selected_params %>% dplyr::filter(model == "model3") %>% dplyr::slice(1)
    
    res <- safe_run_bam_one_strain(
      df_strain = df_strain,
      strain_name = strain_i,
      analysis_mode = analysis_mode,
      degree_global_model1 = sel_m1$degree_global[[1]],
      degree_global_model2 = sel_m2$degree_global[[1]],
      degree_global_model3 = sel_m3$degree_global[[1]],
      degree_fs_model3 = sel_m3$degree_fs[[1]],
      rho_value = rho_value,
      acf_residual_type = acf_residual_type
    )
    
    run_log[[i]] <- data.frame(
      strain = res$strain,
      analysis_mode = res$analysis_mode,
      degree_global_model1 = sel_m1$degree_global[[1]],
      degree_global_model2 = sel_m2$degree_global[[1]],
      degree_global_model3 = sel_m3$degree_global[[1]],
      degree_fs_model3 = sel_m3$degree_fs[[1]],
      model1_selection_status = sel_m1$selection_status[[1]],
      model2_selection_status = sel_m2$selection_status[[1]],
      model3_selection_status = sel_m3$selection_status[[1]],
      status = ifelse(res$success, "success", "error"),
      error_message = res$error_message,
      stringsAsFactors = FALSE
    )
    
    if (res$success) {
      for (nm in setdiff(names(all_results), c("Selected_Parameters", "K_Selection_Grid"))) {
        all_results[[nm]][[length(all_results[[nm]]) + 1]] <- res$result[[nm]]
      }
    }
  }
  
  final_sheets <- lapply(all_results, safe_bind_rows)
  
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
      "Residual_ACF_By_Curve",
      "Residual_ACF_Summary",
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
      paste("Workbook with BAM analyses per strain. Mode:", analysis_mode),
      "Final selected k values for each strain and each model. Model1 and Model2 have their own degree_global. Model3 has its own degree_global and degree_fs.",
      "Selection grid for each strain and each model. Includes pass/fail/inconclusive notes for global k selection and AIC/fs diagnostics for model3 fs selection.",
      "Basic metadata for each strain using final selected k values.",
      "Number of rows per group.",
      "Number of replicate-curves per group.",
      "AIC comparison of final models using model-specific selected k values.",
      "Parametric terms from summary().",
      "Smooth terms from summary().",
      "k.check() output for final models.",
      "Compact diagnostics by strain and model.",
      "Residual ACF computed separately within each curve.",
      "Summary of per-curve ACF values by lag.",
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
    paste0("BAM_", analysis_mode, "_all_strains_model_comparison.xlsx")
  )
  
  writexl::write_xlsx(workbook, path = output_file)
  output_file
}

# ============================================================
# Run all modes
# ============================================================
output_files <- lapply(
  analysis_modes,
  function(mode) {
    build_mode_workbook(
      all_data = all_data,
      analysis_mode = mode,
      degree_global_candidates = degree_global_candidates,
      degree_fs_candidates = degree_fs_candidates,
      fs_to_global_ratio = fs_to_global_ratio,
      rho_value = rho_value,
      k_index_threshold = k_index_threshold,
      p_value_threshold = p_value_threshold,
      fallback_degree_global = fallback_degree_global,
      fallback_degree_fs = fallback_degree_fs,
      fs_aic_tolerance = fs_aic_tolerance,
      fs_edf_ratio_threshold = fs_edf_ratio_threshold,
      degree_fs_for_model3_global_screening = degree_fs_for_model3_global_screening,
      verbose_k_debug = verbose_k_debug,
      output_folder = output_folder,
      acf_residual_type = acf_residual_type
    )
  }
)

print(output_files)
cat("Done.\n")