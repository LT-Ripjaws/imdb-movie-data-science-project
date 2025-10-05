# ===========================
# helper_functions.R
# Utility functions for IMDB Movie Analysis
# ===========================

# a func to get mode of a vector
# We input a vector and get the most frequent value in return
get_mode <- function(v) {
  v <- v[!is.na(v)]
  if (length(v) == 0) return(NA)
  uniq <- unique(v)
  uniq[which.max(tabulate(match(v, uniq)))]
}


#' we detect outliers using IQR method
#'
#' where x is Numeric vector
#' k is the Multiplier for IQR (default 1.5)
#' and it returns indices of outliers
detect_outliers_iqr <- function(x, k = 1.5) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - k * iqr
  upper <- q3 + k * iqr
  which(x < lower | x > upper)
}


#' Min-Max normalization (0-1 scaling)
#'
#' where x is numeric vector
#' and this returns a normalized vector between 0 and 1
normalize_01 <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}


#' to calc descriptive statistics for a numeric vector
#'
#' x = Numeric vector
#' name = Variable name for display
#' This returns a data frame with statistics
get_descriptive_stats <- function(x, name = "Variable") {
  data.frame(
    Variable = name,
    Mean = round(mean(x, na.rm = TRUE), 2),
    Median = round(median(x, na.rm = TRUE), 2),
    SD = round(sd(x, na.rm = TRUE), 2),
    IQR = round(IQR(x, na.rm = TRUE), 2),
    Min = round(min(x, na.rm = TRUE), 2),
    Max = round(max(x, na.rm = TRUE), 2),
    N_Missing = sum(is.na(x)),
    stringsAsFactors = FALSE
  )
}


#' Print data quality report
#'
#' we input: df Data frame
#' and we input the stage: Character string indicating cleaning stage
print_data_quality <- function(df, stage = "Current") {
  cat("\n========================================\n")
  cat(paste0("Data Quality Report: ", stage, "\n"))
  cat("========================================\n")
  cat("Dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
  cat("Duplicates:", sum(duplicated(df)), "\n")
  cat("Missing values by column:\n")
  
  na_summary <- colSums(is.na(df))
  na_pct <- round(100 * na_summary / nrow(df), 2)
  na_df <- data.frame(
    Column = names(na_summary),
    Missing = na_summary,
    Percent = na_pct,
    stringsAsFactors = FALSE
  )
  na_df <- na_df[na_df$Missing > 0, ]
  
  if (nrow(na_df) > 0) {
    print(na_df[order(-na_df$Missing), ], row.names = FALSE)
  } else {
    cat("  No missing values!\n")
  }
  cat("========================================\n\n")
}


#' Save data with timestamp
#'
#' input: Data frame to save
#' input: Base filename
#' input: Directory path
save_with_timestamp <- function(df, filename, path = "data/processed") {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  full_name <- file.path(path, paste0(filename, "_", timestamp, ".csv"))
  write.csv(df, full_name, row.names = FALSE)
  cat("Data saved to:", full_name, "\n")
  
  # Also save without timestamp for easy access
  latest_name <- file.path(path, paste0(filename, "_latest.csv"))
  write.csv(df, latest_name, row.names = FALSE)
  cat("Latest version saved to:", latest_name, "\n")
}


#' Extract primary genre from pipe-separated genres
#'
#' we input Character string with genres separated by |
#' and the function returns first genre
extract_primary_genre <- function(genre_string) {
  sapply(strsplit(as.character(genre_string), "\\|"), `[`, 1)
}


#' Create train-test split (for test data and training data of the model)
#'
#' input: Data frame
#' input: train_ratio which is the proportion for training (default 0.8)
#' input: Random seed for reproducibility
#' This returns a list with train and test data frames
train_test_split <- function(df, train_ratio = 0.8, seed = 123) {
  set.seed(seed)
  train_idx <- sample(seq_len(nrow(df)), size = floor(train_ratio * nrow(df)))
  
  list(
    train = df[train_idx, ],
    test = df[-train_idx, ]
  )
}


#' Calculate regression metrics
#'
#' we give in actual values
#' as well as Predicted values
#' and in return we get a list with RMSE, MAE, and R-squared
calc_regression_metrics <- function(actual, predicted) {
  residuals <- actual - predicted
  
  list(
    RMSE = sqrt(mean(residuals^2, na.rm = TRUE)),
    MAE = mean(abs(residuals), na.rm = TRUE),
    R_squared = 1 - (sum(residuals^2) / sum((actual - mean(actual))^2))
  )
}


#' Print model performance metrics
#'
#' input: List of metrics from calc_regression_metrics
#' input: Name of the model
print_metrics <- function(metrics, model_name = "Model") {
  cat("\n========================================\n")
  cat(paste0(model_name, " Performance Metrics\n"))
  cat("========================================\n")
  cat(sprintf("RMSE:      %.4f\n", metrics$RMSE))
  cat(sprintf("MAE:       %.4f\n", metrics$MAE))
  cat(sprintf("R-squared: %.4f\n", metrics$R_squared))
  cat("========================================\n\n")
}