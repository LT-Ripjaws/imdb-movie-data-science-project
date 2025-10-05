# ===========================
# 05_classification.R
# Classification Models for Movie Profitability Prediction
# Task: Predict if a movie will be profitable (Binary Classification)
# ===========================

# Source helper functions
source("utils/helper_functions.R")

# Required libraries
library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)

#' Prepare data for classification
#'
#' we input a data frame
#' we get a prepared data frame with binary target
prepare_classification_data <- function(data) {
  cat("Preparing data for classification...\n")
  
  # Create binary target: 1 = Profitable, 0 = Not Profitable
  classification_data <- data %>%
    mutate(
      is_profitable = ifelse(profit > 0, 1, 0),
      is_profitable = factor(is_profitable, levels = c(0, 1), 
                             labels = c("Not_Profitable", "Profitable"))
    ) %>%
    select(
      # Target variable
      is_profitable,
      
      # Numeric features
      budget, duration, imdb_score,
      num_voted_users, num_critic_for_reviews, 
      num_user_for_reviews, movie_facebook_likes,
      
      # Categorical features
      primary_genre, content_rating, language, country
    ) %>%
    # Remove rows with any missing values
    na.omit()
  
  cat(sprintf("  Original data: %d rows\n", nrow(data)))
  cat(sprintf("  After removing NAs: %d rows\n", nrow(classification_data)))
  cat(sprintf("  Features selected: %d\n", ncol(classification_data) - 1))
  
  # Show class distribution
  cat("\nClass Distribution:\n")
  print(table(classification_data$is_profitable))
  cat(sprintf("  Profitable: %.1f%%\n", 
              100 * sum(classification_data$is_profitable == "Profitable") / 
                nrow(classification_data)))
  
  return(classification_data)
}


#' Build Logistic Regression model
#'
#' we input training data
#' as well as target variable name
#' and it returns a trained model
build_logistic_model <- function(train_data, target = "is_profitable") {
  cat("\n--- Building Logistic Regression Model ---\n")
  
  # Create formula
  features <- setdiff(names(train_data), target)
  formula_str <- paste(target, "~", paste(features, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  # Train model
  model <- glm(model_formula, data = train_data, family = binomial(link = "logit"))
  
  cat("Model trained successfully!\n")
  cat("\nModel Summary:\n")
  print(summary(model))
  
  return(model)
}


#' Build Random Forest Classifier
#'
#' input: Training data
#' input: Target variable name
#' input: ntree = Number of trees
#' input: mtry = Number of variables at each split
#' returns Trained model
build_rf_classifier <- function(train_data, target = "is_profitable", 
                                ntree = 500, mtry = NULL) {
  cat("\n--- Building Random Forest Classifier ---\n")
  
  # Prepare formula
  features <- setdiff(names(train_data), target)
  formula_str <- paste(target, "~", paste(features, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  # Default mtry
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


#' Calculate classification metrics
#'
#' input: Actual class labels
#' input: Predicted class labels
#' input: Predicted probabilities (optional tho)
#' returns List with metrics
calc_classification_metrics <- function(actual, predicted, predicted_probs = NULL) {
  
  # Confusion matrix
  cm <- table(Predicted = predicted, Actual = actual)
  
  # Extract values
  tn <- cm[1, 1]  # True Negative
  fp <- cm[1, 2]  # False Positive
  fn <- cm[2, 1]  # False Negative
  tp <- cm[2, 2]  # True Positive
  
  
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  specificity <- tn / (tn + fp)
  
  metrics <- list(
    confusion_matrix = cm,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    specificity = specificity,
    tp = tp, tn = tn, fp = fp, fn = fn
  )
  
  # AUC if probabilities provided
  if (!is.null(predicted_probs)) {
    #  AUC calculation
    metrics$auc <- tryCatch({
      roc_obj <- pROC::roc(as.numeric(actual) - 1, predicted_probs)
      as.numeric(roc_obj$auc)
    }, error = function(e) {
      NA
    })
  }
  
  return(metrics)
}


#' Print classification metrics
#'
#' input: Metrics from calc_classification_metrics
#' input: Model name
print_classification_metrics <- function(metrics, model_name = "Model") {
  cat("\n========================================\n")
  cat(paste0(model_name, " Performance Metrics\n"))
  cat("========================================\n")
  
  cat("\nConfusion Matrix:\n")
  print(metrics$confusion_matrix)
  
  cat("\nMetrics:\n")
  cat(sprintf("Accuracy:    %.4f (%.2f%%)\n", metrics$accuracy, metrics$accuracy * 100))
  cat(sprintf("Precision:   %.4f\n", metrics$precision))
  cat(sprintf("Recall:      %.4f\n", metrics$recall))
  cat(sprintf("F1-Score:    %.4f\n", metrics$f1_score))
  cat(sprintf("Specificity: %.4f\n", metrics$specificity))
  
  if (!is.null(metrics$auc) && !is.na(metrics$auc)) {
    cat(sprintf("AUC:         %.4f\n", metrics$auc))
  }
  
  cat("\nClassification Report:\n")
  cat(sprintf("  True Positives:  %d\n", metrics$tp))
  cat(sprintf("  True Negatives:  %d\n", metrics$tn))
  cat(sprintf("  False Positives: %d\n", metrics$fp))
  cat(sprintf("  False Negatives: %d\n", metrics$fn))
  
  cat("========================================\n\n")
}


#' Evaluate classification model
#'
#' input: Trained model
#' input: Test data
#' input: Target variable name
#' input: Model name for display
#' returns List with predictions and metrics
evaluate_classifier <- function(model, test_data, target = "is_profitable", 
                                model_name = "Model") {
  
  cat(sprintf("\n--- Evaluating %s ---\n", model_name))
  
  # Make predictions
  if ("glm" %in% class(model)) {
    # Logistic regression - get probabilities and convert to class
    predicted_probs <- predict(model, newdata = test_data, type = "response")
    predictions <- ifelse(predicted_probs > 0.5, "Profitable", "Not_Profitable")
    predictions <- factor(predictions, levels = c("Not_Profitable", "Profitable"))
  } else {
    # Random Forest or other
    predictions <- predict(model, newdata = test_data)
    predicted_probs <- predict(model, newdata = test_data, type = "prob")[, 2]
  }
  
  actual <- test_data[[target]]
  
  # Calculate metrics
  metrics <- calc_classification_metrics(actual, predictions, predicted_probs)
  
  # Print metrics
  print_classification_metrics(metrics, model_name)
  
  # Create results data frame
  results <- data.frame(
    Actual = actual,
    Predicted = predictions,
    Probability = predicted_probs,
    Correct = (actual == predictions)
  )
  
  return(list(
    predictions = predictions,
    probabilities = predicted_probs,
    metrics = metrics,
    results = results
  ))
}


#' Plot confusion matrix
#'
#' input: Metrics containing confusion matrix
#' input: Model name
#' input:Path to save plot
plot_confusion_matrix <- function(metrics, model_name = "Model", save_path = NULL) {
  
  # Convert to data frame for ggplot
  cm_df <- as.data.frame(metrics$confusion_matrix)
  colnames(cm_df) <- c("Predicted", "Actual", "Count")
  
  p <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), size = 10, color = "white", fontface = "bold") +
    scale_fill_gradient(low = "steelblue", high = "darkblue") +
    labs(title = paste(model_name, "- Confusion Matrix"),
         x = "Actual Class",
         y = "Predicted Class") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          axis.text = element_text(size = 12),
          legend.position = "right")
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Plot saved:", save_path, "\n")
  }
  
  print(p)
  return(p)
}


#' Plot ROC curve
#'
#' input: Results from evaluate_classifier
#' input: Model name
#' input: Path to save plot
plot_roc_curve <- function(results, model_name = "Model", save_path = NULL) {
  
  # Try to load pROC, if not available, skip
  if (!requireNamespace("pROC", quietly = TRUE)) {
    cat("pROC package not installed. Skipping ROC curve.\n")
    cat("Install with: install.packages('pROC')\n")
    return(NULL)
  }
  
  library(pROC)
  
  # Calculate ROC
  actual_numeric <- as.numeric(results$Actual) - 1
  roc_obj <- roc(actual_numeric, results$Probability, quiet = TRUE)
  
  # Create data for plotting
  roc_data <- data.frame(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )
  
  auc_value <- as.numeric(roc_obj$auc)
  
  p <- ggplot(roc_data, aes(x = FPR, y = TPR)) +
    geom_line(color = "darkblue", size = 1.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    annotate("text", x = 0.7, y = 0.3, 
             label = paste0("AUC = ", round(auc_value, 3)),
             size = 6, fontface = "bold") +
    labs(title = paste(model_name, "- ROC Curve"),
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Plot saved:", save_path, "\n")
  }
  
  print(p)
  return(p)
}


#' Compare classification models
#'
#' input: List of model evaluation results
#' returns Comparison data frame
compare_classifiers <- function(model_results) {
  
  cat("\n========================================\n")
  cat("Classification Model Comparison\n")
  cat("========================================\n")
  
  comparison <- data.frame(
    Model = names(model_results),
    Accuracy = sapply(model_results, function(x) x$metrics$accuracy),
    Precision = sapply(model_results, function(x) x$metrics$precision),
    Recall = sapply(model_results, function(x) x$metrics$recall),
    F1_Score = sapply(model_results, function(x) x$metrics$f1_score),
    stringsAsFactors = FALSE
  )
  
  comparison <- comparison[order(-comparison$F1_Score), ]
  
  print(comparison)
  
  cat("\nBest Model (by F1-Score):", comparison$Model[1], "\n")
  cat("========================================\n\n")
  
  return(comparison)
}


#' Complete classification pipeline
#'
#' input: Cleaned data frame
#' input: Proportion for test set
#' input: Save models and plots
#' returns List with models and results
run_classification_pipeline <- function(data, test_ratio = 0.2, save_outputs = TRUE) {
  
  cat("\n========================================\n")
  cat("Classification Pipeline\n")
  cat("Task: Predict Movie Profitability\n")
  cat("========================================\n\n")
  
  # 1. Prepare data
  classification_data <- prepare_classification_data(data)
  
  # 2. Train-test split
  cat("\nSplitting data into train and test sets...\n")
  split <- train_test_split(classification_data, train_ratio = 1 - test_ratio, seed = 123)
  train_data <- split$train
  test_data <- split$test
  
  cat(sprintf("  Training set: %d rows (%.0f%%)\n", 
              nrow(train_data), 100 * (1 - test_ratio)))
  cat(sprintf("  Test set: %d rows (%.0f%%)\n", 
              nrow(test_data), 100 * test_ratio))
  
  # 3. Encode categorical variables
  cat_vars <- c("primary_genre", "content_rating", "language", "country")
  
  for (var in cat_vars) {
    if (var %in% names(train_data)) {
      train_data[[var]] <- as.factor(train_data[[var]])
      test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data[[var]]))
    }
  }
  
  # 4. Build models
  cat("\n========================================\n")
  cat("Training Classification Models\n")
  cat("========================================\n")
  
  # Logistic Regression
  logistic_model <- build_logistic_model(train_data)
  
  # Random Forest Classifier
  rf_classifier <- build_rf_classifier(train_data, ntree = 500)
  
  # 5. Evaluate models
  cat("\n========================================\n")
  cat("Model Evaluation\n")
  cat("========================================\n")
  
  logistic_results <- evaluate_classifier(logistic_model, test_data, 
                                          model_name = "Logistic Regression")
  
  rf_results <- evaluate_classifier(rf_classifier, test_data, 
                                    model_name = "Random Forest Classifier")
  
  # 6. Compare models
  model_results <- list(
    "Logistic Regression" = logistic_results,
    "Random Forest" = rf_results
  )
  
  comparison <- compare_classifiers(model_results)
  
  # 7. Visualizations
  if (save_outputs) {
    cat("\nGenerating visualizations...\n")
    
    if (!dir.exists("output/figures")) {
      dir.create("output/figures", recursive = TRUE)
    }
    
    # Confusion matrices
    plot_confusion_matrix(logistic_results$metrics, "Logistic Regression",
                          "output/figures/logistic_confusion_matrix.png")
    
    plot_confusion_matrix(rf_results$metrics, "Random Forest",
                          "output/figures/rf_classifier_confusion_matrix.png")
    
    # ROC curves
    plot_roc_curve(logistic_results$results, "Logistic Regression",
                   "output/figures/logistic_roc_curve.png")
    
    plot_roc_curve(rf_results$results, "Random Forest",
                   "output/figures/rf_classifier_roc_curve.png")
    
    # Feature importance for RF
    if ("randomForest" %in% class(rf_classifier)) {
      importance_df <- data.frame(
        Feature = rownames(importance(rf_classifier)),
        Importance = importance(rf_classifier)[, "MeanDecreaseGini"]
      )
      importance_df <- importance_df[order(-importance_df$Importance), ]
      importance_df <- head(importance_df, 10)
      
      p <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
        geom_col(fill = "steelblue", alpha = 0.7) +
        coord_flip() +
        labs(title = "Feature Importance for Profitability Prediction",
             x = "Feature", y = "Importance (Mean Decrease Gini)") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, face = "bold"))
      
      ggsave("output/figures/profitability_feature_importance.png", p, 
             width = 10, height = 6)
    }
  }
  
  # 8. Save best model
  if (save_outputs) {
    best_model_name <- comparison$Model[1]
    best_model <- if (best_model_name == "Logistic Regression") logistic_model else rf_classifier
    
    source("R/04_reg_modeling.R")  # Load save function
    save_model_with_metadata(best_model, train_data, "best_classifier")
    save_model_with_metadata(logistic_model, train_data, "logistic_classifier")
    save_model_with_metadata(rf_classifier, train_data, "rf_classifier")
  }
  
  cat("\n========================================\n")
  cat("Classification Pipeline Complete!\n")
  cat("========================================\n\n")
  
  return(list(
    models = list(logistic = logistic_model, rf = rf_classifier),
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
  
  # Run classification pipeline
  classification_results <- run_classification_pipeline(imdb_clean, 
                                                        test_ratio = 0.2, 
                                                        save_outputs = TRUE)
  
  cat("\nClassification complete! Check output/ for models and visualizations.\n")
}