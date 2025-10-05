# ===========================
# main.R
# Master script for IMDB Movie Analysis & Prediction
# ===========================

# Clear environment
rm(list = ls())

# Set working directory to project root (adjust it or it aint working)
setwd("path/to/imdb-movie-data-science-project")

cat("========================================\n")
cat("IMDB Movie Analysis & Prediction\n")
cat("========================================\n\n")

# ===========================
# 0. Install & Load Required Packages
# ===========================

cat("Checking required packages...\n")

required_packages <- c("dplyr", "janitor", "ggplot2", "reshape2", 
                       "caret", "randomForest")

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(new_packages) > 0) {
  cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages)
}

# Load packages
invisible(lapply(required_packages, library, character.only = TRUE))

cat("All packages loaded successfully!\n\n")

# ===========================
# Configuration
# ===========================

# Dataset URL
DATA_URL <- "https://raw.githubusercontent.com/LT-Ripjaws/imdb-movie-data-science-project/refs/heads/main/movie-metadata/movie_metadata.csv"

# Modeling parameters
TRAIN_RATIO <- 0.8
TEST_RATIO <- 0.2
RANDOM_SEED <- 125

# Output options
SAVE_FIGURES <- TRUE
SAVE_MODELS <- TRUE
REMOVE_OUTLIERS <- FALSE  # Set to TRUE if we want to remove outliers


# ===========================
# 1. Data Loading
# ===========================

cat("\n========================================\n")
cat("STEP 1: Loading our movie Data >:D\n")
cat("========================================\n")

source("R/01_data_loading.R")

imdb_raw <- load_imdb_data(url = DATA_URL, 
                           cache_path = "data/raw/imdb_raw.csv")

initial_exploration(imdb_raw)

# ===========================
# 2. Data Cleaning
# ===========================

cat("\n========================================\n")
cat("STEP 2: Data Cleaning\n")
cat("========================================\n")

source("R/02_data_cleaning.R")

imdb_clean <- clean_imdb_data(imdb_raw, remove_outliers = REMOVE_OUTLIERS)

# Save cleaned data
save_with_timestamp(imdb_clean, "imdb_cleaned", path = "data/processed")

cat("\nCleaned data shape:", dim(imdb_clean), "\n")
cat("Columns:", ncol(imdb_clean), "\n")
cat("Rows:", nrow(imdb_clean), "\n\n")

# ===========================
# 3. Exploratory Data Analysis
# ===========================

cat("\n========================================\n")
cat("STEP 3: Exploratory Data Analysis\n")
cat("========================================\n")

source("R/03_eda.R")

eda_results <- perform_eda(imdb_clean, save_figures = SAVE_FIGURES)

cat("\nKey Insights:\n")
cat("  Top rated movie:", eda_results$top_imdb$movie_title[1], 
    "- Score:", eda_results$top_imdb$imdb_score[1], "\n")
cat("  Highest ROI movie:", eda_results$top_roi$movie_title[1], 
    "- ROI:", round(eda_results$top_roi$roi[1], 2), "\n")
cat("  Most profitable director:", eda_results$top_directors$director_name[1], "\n\n")

# ===========================
# 4. Machine Learning Models
# ===========================

cat("\n========================================\n")
cat("STEP 4: Machine Learning Modeling\n")
cat("========================================\n")

source("R/04_reg_modeling.R")
source("R/05_classification.R")

# Regression: Predict IMDb Score
cat("\n--- REGRESSION: Predicting IMDb Scores ---\n")
modeling_results <- run_modeling_pipeline(
  data = imdb_clean,
  test_ratio = TEST_RATIO,
  save_outputs = SAVE_MODELS
)

# Classification: Predict Profitability
cat("\n--- CLASSIFICATION: Predicting Profitability ---\n")
classification_results <- run_classification_pipeline(
  data = imdb_clean,
  test_ratio = TEST_RATIO,
  save_outputs = SAVE_MODELS)
  
# ===========================
# 5. Final Summary
# ===========================

cat("\n========================================\n")
cat("FINAL SUMMARY\n")
cat("========================================\n\n")

cat("Data Processing:\n")
cat("  Raw data rows:", nrow(imdb_raw), "\n")
cat("  Cleaned data rows:", nrow(imdb_clean), "\n")
cat("  Rows removed:", nrow(imdb_raw) - nrow(imdb_clean), 
    sprintf("(%.2f%%)\n", 100 * (nrow(imdb_raw) - nrow(imdb_clean)) / nrow(imdb_raw)))

cat("\nModel Performance:\n")
cat("\nREGRESSION (IMDb Score Prediction):\n")
print(modeling_results$comparison)

cat("\nCLASSIFICATION (Profitability Prediction):\n")
print(classification_results$comparison)

cat("\nOutput Files Generated:\n")
if (SAVE_FIGURES) {
  cat("  Figures: output/figures/\n")
  cat("    - Distribution plots\n")
  cat("    - Scatter plots\n")
  cat("    - Correlation heatmap\n")
  cat("    - Time trends\n")
  cat("    - Model evaluation plots\n")
}
if (SAVE_MODELS) {
  cat("  Models: output/models/\n")
  cat("    - best_model.rds\n")
}
cat("  Data: data/processed/\n")
cat("    - imdb_cleaned_latest.csv\n")

cat("\n========================================\n")
cat("PIPELINE COMPLETE!\n")
cat("========================================\n\n")

cat("Analysis complete! ! YAY\n")
cat("Check the output/ directory for all results.\n\n")