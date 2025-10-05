# ===========================
# 01_data_loading.R
# Load and perform initial inspection of IMDB movie dataset
# ===========================

# ref the helper functions file we created
source("utils/helper_functions.R")

# Required libraries
library(dplyr)
library(janitor)

# Load IMDB movie dataset
#
# input: URL to the dataset
# input: Path to save/load cached data
# returns a Cleaned data frame
load_imdb_data <- function(
    url = "https://raw.githubusercontent.com/LT-Ripjaws/imdb-movie-data-science-project/refs/heads/main/movie-metadata/movie_metadata.csv",
    cache_path = "data/raw/imdb_raw.csv") {
  
  cat("========================================\n")
  cat("Loading IMDB Movie Dataset\n")
  cat("========================================\n")
  
  # Create directory if it doesn't exist
  if (!dir.exists(dirname(cache_path))) {
    dir.create(dirname(cache_path), recursive = TRUE)
    cat("Created directory:", dirname(cache_path), "\n")
  }
  
  # Check if cached file exists
  if (file.exists(cache_path)) {
    cat("Loading from cache:", cache_path, "\n")
    data <- read.csv(cache_path, stringsAsFactors = FALSE)
  } else {
    cat("Downloading from URL...\n")
    data <- tryCatch({
      read.csv(url, stringsAsFactors = FALSE)
    }, error = function(e) {
      stop("Failed to download data. Error: ", e$message)
    })
    
    # Save to cache
    write.csv(data, cache_path, row.names = FALSE)
    cat("Data cached to:", cache_path, "\n")
  }
  
  # Clean column names
  data <- janitor::clean_names(data)
  
  cat("Successfully loaded!\n")
  cat("Dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
  cat("========================================\n\n")
  
  return(data)
}


# We perform initial data exploration
#
# input: Data frame
initial_exploration <- function(data) {
  cat("========================================\n")
  cat("Initial Data Exploration\n")
  cat("========================================\n\n")
  
  # Structure
  cat("--- Data Structure ---\n")
  str(data)
  
  cat("\n--- Column Names ---\n")
  print(names(data))
  
  cat("\n--- Summary Statistics ---\n")
  print(summary(data))
  
  cat("\n--- First 5 Rows ---\n")
  print(head(data, 5))
  
  # Data quality before cleaning
  print_data_quality(data, stage = "Before Cleaning")
  
  # Missing values percentage
  cat("--- Missing Values (Top 10) ---\n")
  na_pct <- sort(100 * colSums(is.na(data)) / nrow(data), decreasing = TRUE)
  print(head(na_pct, 10))
  
  # Distinct value counts
  cat("\n--- Columns with Fewest Unique Values (Top 10) ---\n")
  distinct_counts <- sapply(data, function(x) length(unique(x)))
  print(head(sort(distinct_counts), 10))
  
  cat("\n========================================\n\n")
}


# ===========================
# Execution of our funcs
# ===========================

if (!interactive()) {
  # Load data
  imdb_raw <- load_imdb_data()
  
  # Initial exploration
  initial_exploration(imdb_raw)
  
  # Save to processed folde
  save_with_timestamp(imdb_raw, "imdb_raw", path = "data/raw")
}