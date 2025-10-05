# ===========================
# 02_data_cleaning.R
# Clean and preprocess our IMDB dataset
# ===========================

# helper functions
source("utils/helper_functions.R")

# Required libraries
library(dplyr)

# Remove duplicate data in our dafa-set
#
# input: Data frame
# returns a cleaned data frame
remove_duplicates <- function(data) {
  cat("Removing duplicates...\n")
  
  n_before <- nrow(data)
  
  # Remove complete duplicates
  data <- data %>% distinct()
  
  # Remove duplicates based on movie title and year
  data <- data %>%
    distinct(movie_title, title_year, .keep_all = TRUE)
  
  n_after <- nrow(data)
  n_removed <- n_before - n_after
  
  cat(sprintf("  Removed %d duplicate rows (%.2f%%)\n", 
              n_removed, 100 * n_removed / n_before))
  
  return(data)
}


# Filter out rows with critical missing values
#
# input: Data frame
# input: critical_cols = column names
# returns a filtered data frame
filter_critical_missing <- function(data, 
                                    critical_cols = c("budget", "gross", "imdb_score")) {
  cat("Filtering rows with missing critical values...\n")
  
  n_before <- nrow(data)
  
  # Remove rows where any critical column is NA
  for (col in critical_cols) {
    if (col %in% names(data)) {
      data <- data %>% filter(!is.na(.data[[col]]))
    }
  }
  
  n_after <- nrow(data)
  n_removed <- n_before - n_after
  
  cat(sprintf("  Removed %d rows (%.2f%%) with missing: %s\n", 
              n_removed, 100 * n_removed / n_before,
              paste(critical_cols, collapse = ", ")))
  
  return(data)
}


# Clean text columns
#
# input: data = Data frame
# input: text_cols = column names to clean
# returns a data frame with cleaned text columns
clean_text_columns <- function(data, text_cols = c("movie_title", "director_name")) {
  cat("Cleaning text columns...\n")
  
  for (col in text_cols) {
    if (col %in% names(data)) {
      # Trim whitespace and convert to character
      data[[col]] <- trimws(as.character(data[[col]]))
    }
  }
  
  cat("  Text columns cleaned\n")
  return(data)
}


# ~Feature engineering~
#
# input: data frame 
# returns a data frame with new features :D
engineer_features <- function(data) {
  cat("Engineering new features...\n")
  
  data <- data %>%
    mutate(
      # Financial metrics
      profit = gross - budget,
      roi = ifelse(!is.na(budget) & budget > 0, profit / budget, NA_real_),
      
      # Extract primary genre
      primary_genre = extract_primary_genre(genres),
      
      # Convert title_year to numeric
      title_year = as.numeric(as.character(title_year)),
      
      # Budget categories for classification
      budget_category = case_when(
        budget < 10000000 ~ "Low",
        budget >= 10000000 & budget < 50000000 ~ "Medium",
        budget >= 50000000 ~ "High",
        TRUE ~ NA_character_
      ),
      
      # Success indicator (for classification tasks)
      is_profitable = ifelse(profit > 0, 1, 0),
      
      # IMDb rating categories
      rating_category = case_when(
        imdb_score >= 7.5 ~ "Excellent",
        imdb_score >= 6.5 ~ "Good",
        imdb_score >= 5.5 ~ "Average",
        TRUE ~ "Poor"
      )
    )
  
  # Remove rows with NA in engineered features
  data <- data %>% filter(!is.na(roi))
  
  cat("  Created: profit, roi, primary_genre, budget_category, is_profitable, rating_category\n")
  
  return(data)
}


# Select relevant columns for analysis
#
# input: Data frame
# returns a data frame with selected columns
select_relevant_columns <- function(data) {
  cat("Selecting relevant columns...\n")
  
  relevant_cols <- c(
    # Identifiers
    "movie_title", "director_name", "title_year",
    
    # Target variables
    "imdb_score", "rating_category",
    
    # Financial
    "budget", "gross", "profit", "roi", "budget_category", "is_profitable",
    
    # Movie characteristics
    "duration", "content_rating", "primary_genre", "language", "country",
    
    # Social metrics
    "num_voted_users", "num_critic_for_reviews", 
    "num_user_for_reviews", "movie_facebook_likes"
  )
  
  # Only select columns that exist
  available_cols <- relevant_cols[relevant_cols %in% names(data)]
  data <- data %>% select(all_of(available_cols))
  
  cat(sprintf("  Selected %d columns\n", ncol(data)))
  
  return(data)
}


# function to flag the outliers in our data set
#
# input: Data frame
# input: numeric_cols = Columns to check for outliers
# returns a data frame with outlier flags
flag_outliers <- function(data, 
                          numeric_cols = c("budget", "gross", "profit", "roi", "imdb_score")) {
  cat("Flagging outliers...\n")
  
  for (col in numeric_cols) {
    if (col %in% names(data) && is.numeric(data[[col]])) {
      outlier_idx <- detect_outliers_iqr(data[[col]])
      flag_col <- paste0(col, "_outlier")
      data[[flag_col]] <- FALSE
      data[[flag_col]][outlier_idx] <- TRUE
      
      n_outliers <- length(outlier_idx)
      cat(sprintf("  %s: %d outliers (%.2f%%)\n", 
                  col, n_outliers, 100 * n_outliers / nrow(data)))
    }
  }
  
  return(data)
}


# Main cleaning pipeline
#
# input: Raw data frame
# input: remove_outliers = whether to remove outliers or na
# returns a cleaned data frame
clean_imdb_data <- function(data, remove_outliers = FALSE) {
  cat("\n========================================\n")
  cat("Data Cleaning Pipeline\n")
  cat("========================================\n")
  
  print_data_quality(data, "Before Cleaning")
  
  # Step 1: Remove duplicates
  data <- remove_duplicates(data)
  
  # Step 2: Filter critical missing values
  data <- filter_critical_missing(data)
  
  # Step 3: Clean text columns
  data <- clean_text_columns(data)
  
  # Step 4: Feature engineering
  data <- engineer_features(data)
  
  # Step 5: Select relevant columns
  data <- select_relevant_columns(data)
  
  # Step 6: Flag outliers
  data <- flag_outliers(data)
  
  # Optional: Remove outliers (if want to remove, set to TRUE in func)
  if (remove_outliers) {
    cat("Removing flagged outliers...\n")
    outlier_cols <- grep("_outlier$", names(data), value = TRUE)
    n_before <- nrow(data)
    for (col in outlier_cols) {
      data <- data[!data[[col]], ]
    }
    n_removed <- n_before - nrow(data)
    cat(sprintf("  Removed %d outlier rows\n", n_removed))
    
    # Remove outlier flag columns
    data <- data %>% select(-ends_with("_outlier"))
  }
  
  print_data_quality(data, "After Cleaning")
  
  cat("Cleaning complete!\n")
  cat("========================================\n\n")
  
  return(data)
}


# ===========================
# Main Execution
# ===========================

if (!interactive()) {
  # Load raw data
  imdb_raw <- read.csv("data/raw/imdb_raw_latest.csv", stringsAsFactors = FALSE)
  
  # Clean data
  imdb_clean <- clean_imdb_data(imdb_raw, remove_outliers = FALSE)
  
  # Save cleaned data
  save_with_timestamp(imdb_clean, "imdb_cleaned", path = "data/processed")
  
  cat("\nCleaned data ready for analysis and modeling!\n")
}