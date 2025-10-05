# ===========================
# 04_reg_modeling.R
# Machine Learning Models for IMDb Score Prediction
# ===========================

# Source helper functions
source("utils/helper_functions.R")

# Required libraries
library(dplyr)
library(caret)
library(randomForest)

# Prepare data for modeling
#
# input: Data frame
# returns a prepared data frame with only modeling features
prepare_modeling_data <- function(data) {
  cat("Preparing data for modeling...\n")
  
  # Select features for modeling
  modeling_data <- data %>%
    select(
      # Target variable
      imdb_score,
      
      # Numeric features
      budget, duration, num_voted_users, 
      num_critic_for_reviews, num_user_for_reviews,
      movie_facebook_likes,
      
      # Categorical features
      primary_genre, content_rating, language, country
    ) %>%
    # Remove rows with any missing values
    na.omit()
  
  cat(sprintf("  Original data: %d rows\n", nrow(data)))
  cat(sprintf("  After removing NAs: %d rows\n", nrow(modeling_data)))
  cat(sprintf("  Features selected: %d\n", ncol(modeling_data) - 1))
  
  return(modeling_data)
}


# Encode categorical variables
#
# input: Training data
# input: Test data
# input: Categorical variable names
# returns a List with encoded train and test data
encode_categorical <- function(train_data, test_data, cat_vars) {
  cat("Encoding categorical variables...\n")
  
  for (var in cat_vars) {
    if (var %in% names(train_data)) {
      # Convert to factor
      train_data[[var]] <- as.factor(train_data[[var]])
      test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data[[var]]))
      
      cat(sprintf("  %s: %d levels\n", var, nlevels(train_data[[var]])))
    }
  }
  
  return(list(train = train_data, test = test_data))
}


# Build Linear Regression model
#
# input: Training data
# input: Target variable name
# returns a Trained model
build_linear_model <- function(train_data, target = "imdb_score") {
  cat("\n--- Building Linear Regression Model ---\n")
  
  # Create formula
  features <- setdiff(names(train_data), target)
  formula_str <- paste(target, "~", paste(features, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  # Train model
  model <- lm(model_formula, data = train_data)
  
  cat("Model trained successfully!\n")
  cat("\nModel Summary:\n")
  print(summary(model))
  
  return(model)
}


# Build Random Forest model
#
# input: Training data
# input: Target variable name
# input: ntree = Number of trees
# input: mtry = Number of variables at each split
# returns a Trained model
build_random_forest <- function(train_data, target = "imdb_score", 
                                ntree = 500, mtry = NULL) {
  cat("\n--- Building Random Forest Model ---\n")
  
  # Prepare formula
  features <- setdiff(names(train_data), target)
  formula_str <- paste(target, "~", paste(features, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  # Default mtry (sqrt of number of features)
  if (is.null(mtry)) {
    mtry <- floor(sqrt(length(features)))
  }
  
  cat(sprintf("Training with ntree=%d, mtry=%d...\n", ntree, mtry))
  
  # Train model
  set.seed(123)
  model <- randomForest(model_formula, data = train_data, 
                        ntree = ntree, mtry = mtry, importance = TRUE)
  
  cat("Model trained successfully!\n")
  print(model)
  
  return(model)
}


# Evaluate model performance
#
# input: Trained model
# input: Test data
# input: Target variable name
# input:  Model name for display
# returns a List with predictions and metrics
evaluate_model <- function(model, test_data, target = "imdb_score", 
                           model_name = "Model") {
  
  cat(sprintf("\n--- Evaluating %s ---\n", model_name))
  
  # Make predictions
  predictions <- predict(model, newdata = test_data)
  actual <- test_data[[target]]
  
  # Calculate metrics
  metrics <- calc_regression_metrics(actual, predictions)
  
  # Print metrics
  print_metrics(metrics, model_name)
  
  # Create results data frame
  results <- data.frame(
    Actual = actual,
    Predicted = predictions,
    Residual = actual - predictions
  )
  
  return(list(
    predictions = predictions,
    metrics = metrics,
    results = results
  ))
}


# Plot actual vs predicted
#
# input: Results data frame from evaluate_model
# input: Model name
# input: Path to save plot
plot_actual_vs_predicted <- function(results, model_name = "Model", 
                                     save_path = NULL) {
  
  p <- ggplot(results, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.5, color = "darkblue") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = paste(model_name, "- Actual vs Predicted"),
         x = "Actual IMDb Score",
         y = "Predicted IMDb Score") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Plot saved:", save_path, "\n")
  }
  
  print(p)
  return(p)
}


# Plot residuals
#
# input: Results data frame from evaluate_model
# input:  Model name
# input:  Path to save plot
plot_residuals <- function(results, model_name = "Model", save_path = NULL) {
  
  p <- ggplot(results, aes(x = Predicted, y = Residual)) +
    geom_point(alpha = 0.5, color = "darkgreen") +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title = paste(model_name, "- Residual Plot"),
         x = "Predicted IMDb Score",
         y = "Residuals") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Plot saved:", save_path, "\n")
  }
  
  print(p)
  return(p)
}


# Plot feature importance for Random Forest
#
# input: Random Forest model
# input: top_n Number of top features to show
# input: Path to save plot
plot_feature_importance <- function(model, top_n = 10, save_path = NULL) {
  
  if (!"randomForest" %in% class(model)) {
    cat("Feature importance only available for Random Forest models.\n")
    return(NULL)
  }
  
  # Extract importance
  importance_df <- data.frame(
    Feature = rownames(importance(model)),
    Importance = importance(model)[, "%IncMSE"]
  )
  
  # Sort and take top N
  importance_df <- importance_df[order(-importance_df$Importance), ]
  importance_df <- head(importance_df, top_n)
  
  p <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    coord_flip() +
    labs(title = "Feature Importance (Top 10)",
         x = "Feature",
         y = "Importance (% Increase MSE)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Plot saved:", save_path, "\n")
  }
  
  print(p)
  return(p)
}


# Compare multiple models
#
# input: List of model evaluation results
# returns a Comparison data frame
compare_models <- function(model_results) {
  
  cat("\n========================================\n")
  cat("Model Comparison\n")
  cat("========================================\n")
  
  comparison <- data.frame(
    Model = names(model_results),
    RMSE = sapply(model_results, function(x) x$metrics$RMSE),
    MAE = sapply(model_results, function(x) x$metrics$MAE),
    R_squared = sapply(model_results, function(x) x$metrics$R_squared),
    stringsAsFactors = FALSE
  )
  
  comparison <- comparison[order(comparison$RMSE), ]
  
  print(comparison)
  
  cat("\nBest Model:", comparison$Model[1], "\n")
  cat("========================================\n\n")
  
  return(comparison)
}


# Save model to disk (BASIC VERSION - no metadata)
#
# input:  Trained model
# input:  Model name
# input:  Directory to save model
save_model <- function(model, model_name, path = "output/models") {
  
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  
  filename <- file.path(path, paste0(model_name, ".rds"))
  saveRDS(model, filename)
  
  cat("Model saved to:", filename, "\n")
}


# Save model with metadata
#
# input: Trained model
# input: Training data used to train the model
# input:  Model name
# input: Directory to save model
save_model_with_metadata <- function(model, train_data, model_name, 
                                     path = "output/models") {
  
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  
  # Save model
  model_file <- file.path(path, paste0(model_name, ".rds"))
  saveRDS(model, model_file)
  cat("Model saved to:", model_file, "\n")
  
  # Save factor levels for categorical variables
  factor_levels <- list()
  categorical_vars <- c("primary_genre", "content_rating", "language", "country")
  
  for (var in categorical_vars) {
    if (var %in% names(train_data) && is.factor(train_data[[var]])) {
      factor_levels[[var]] <- levels(train_data[[var]])
    }
  }
  
  # Save factor levels metadata
  if (length(factor_levels) > 0) {
    levels_file <- file.path(path, paste0(model_name, "_factor_levels.rds"))
    saveRDS(factor_levels, levels_file)
    cat("Factor levels saved to:", levels_file, "\n")
  }
  
  # Save feature names
  if ("randomForest" %in% class(model)) {
    feature_names <- names(model$forest$xlevels)
  } else if ("lm" %in% class(model)) {
    feature_names <- names(coef(model))[-1]  # Exclude intercept
  } else {
    feature_names <- setdiff(names(train_data), "imdb_score")
  }
  
  features_file <- file.path(path, paste0(model_name, "_features.rds"))
  saveRDS(feature_names, features_file)
  cat("Feature names saved to:", features_file, "\n")
  
  # Save training data summary
  training_summary <- list(
    n_rows = nrow(train_data),
    n_features = ncol(train_data) - 1,
    target = "imdb_score",
    categorical_vars = categorical_vars,
    numeric_vars = setdiff(names(train_data), c("imdb_score", categorical_vars))
  )
  
  summary_file <- file.path(path, paste0(model_name, "_training_info.rds"))
  saveRDS(training_summary, summary_file)
  cat("Training info saved to:", summary_file, "\n")
}


# Load model with metadata
#
# input:  Model name
# input:  Directory containing model files
# returns a List with model and metadata
load_model_with_metadata <- function(model_name, path = "output/models") {
  
  model_file <- file.path(path, paste0(model_name, ".rds"))
  levels_file <- file.path(path, paste0(model_name, "_factor_levels.rds"))
  features_file <- file.path(path, paste0(model_name, "_features.rds"))
  summary_file <- file.path(path, paste0(model_name, "_training_info.rds"))
  
  if (!file.exists(model_file)) {
    stop("Model file not found: ", model_file)
  }
  
  model <- readRDS(model_file)
  cat("Model loaded from:", model_file, "\n")
  
  metadata <- list(model = model)
  
  # Load factor levels if available
  if (file.exists(levels_file)) {
    metadata$factor_levels <- readRDS(levels_file)
    cat("Factor levels loaded\n")
  }
  
  # Load feature names if available
  if (file.exists(features_file)) {
    metadata$feature_names <- readRDS(features_file)
    cat("Feature names loaded\n")
  }
  
  # Load training info if available
  if (file.exists(summary_file)) {
    metadata$training_info <- readRDS(summary_file)
    cat("Training info loaded\n")
  }
  
  return(metadata)
}


# Complete modeling pipeline
#
# input: Cleaned data frame
# input: Proportion for test set
# input: Save models and plots
# returns a List with models and results
run_modeling_pipeline <- function(data, test_ratio = 0.2, save_outputs = TRUE) {
  
  cat("\n========================================\n")
  cat("Machine Learning Pipeline\n")
  cat("========================================\n\n")
  
  # 1. Prepare data
  modeling_data <- prepare_modeling_data(data)
  
  # 2. Train-test split
  cat("\nSplitting data into train and test sets...\n")
  split <- train_test_split(modeling_data, train_ratio = 1 - test_ratio, seed = 123)
  train_data <- split$train
  test_data <- split$test
  
  cat(sprintf("  Training set: %d rows (%.0f%%)\n", 
              nrow(train_data), 100 * (1 - test_ratio)))
  cat(sprintf("  Test set: %d rows (%.0f%%)\n", 
              nrow(test_data), 100 * test_ratio))
  
  # 3. Encode categorical variables
  cat_vars <- c("primary_genre", "content_rating", "language", "country")
  encoded <- encode_categorical(train_data, test_data, cat_vars)
  train_data <- encoded$train
  test_data <- encoded$test
  
  # 4. Build models
  cat("\n========================================\n")
  cat("Training Models\n")
  cat("========================================\n")
  
  # Linear Regression
  lm_model <- build_linear_model(train_data)
  
  # Random Forest
  rf_model <- build_random_forest(train_data, ntree = 500)
  
  # 5. Evaluate models
  cat("\n========================================\n")
  cat("Model Evaluation\n")
  cat("========================================\n")
  
  lm_results <- evaluate_model(lm_model, test_data, 
                               model_name = "Linear Regression")
  
  rf_results <- evaluate_model(rf_model, test_data, 
                               model_name = "Random Forest")
  
  # 6. Compare models
  model_results <- list(
    "Linear Regression" = lm_results,
    "Random Forest" = rf_results
  )
  
  comparison <- compare_models(model_results)
  
  # 7. Visualizations
  if (save_outputs) {
    cat("\nGenerating visualizations...\n")
    
    if (!dir.exists("output/figures")) {
      dir.create("output/figures", recursive = TRUE)
    }
    
    # Actual vs Predicted plots
    plot_actual_vs_predicted(lm_results$results, "Linear Regression",
                             "output/figures/lm_actual_vs_pred.png")
    
    plot_actual_vs_predicted(rf_results$results, "Random Forest",
                             "output/figures/rf_actual_vs_pred.png")
    
    # Residual plots
    plot_residuals(lm_results$results, "Linear Regression",
                   "output/figures/lm_residuals.png")
    
    plot_residuals(rf_results$results, "Random Forest",
                   "output/figures/rf_residuals.png")
    
    # Feature importance
    plot_feature_importance(rf_model, top_n = 10,
                            "output/figures/rf_feature_importance.png")
  }
  
  # 8. Save best model WITH METADATA
  if (save_outputs) {
    best_model_name <- comparison$Model[1]
    best_model <- if (best_model_name == "Linear Regression") lm_model else rf_model
    
    # Use the save function with metadata
    save_model_with_metadata(best_model, train_data, "best_model")
    
    # Also save individual models with metadata
    save_model_with_metadata(lm_model, train_data, "linear_regression_model")
    save_model_with_metadata(rf_model, train_data, "random_forest_model")
  }
  
  cat("\n========================================\n")
  cat("Modeling Pipeline Complete!\n")
  cat("========================================\n\n")
  
  return(list(
    models = list(lm = lm_model, rf = rf_model),
    results = model_results,
    comparison = comparison,
    train_data = train_data,
    test_data = test_data
  ))
}


# ===========================
# Main Execution
# ===========================

if (!interactive()) {
  # Load cleaned data
  imdb_clean <- read.csv("data/processed/imdb_cleaned_latest.csv", 
                         stringsAsFactors = FALSE)
  
  # Run modeling pipeline
  pipeline_results <- run_modeling_pipeline(imdb_clean, 
                                            test_ratio = 0.2, 
                                            save_outputs = TRUE)
  
  cat("\nModeling complete! Check output/ for models and visualizations.\n")
  cat("Models saved with metadata for easy predictions.\n")
}