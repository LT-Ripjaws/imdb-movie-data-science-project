# ===========================
# EXAMPLE_USAGE.R
# Usage examples for IMDB Movie Analysis project
# ===========================

# Load required libraries and helper functions
source("utils/helper_functions.R")
library(dplyr)
library(ggplot2)

# ===========================
# Quick Analysis
# ===========================

quick_analysis <- function() {
  cat("Running quick analysis...\n\n")
  
  # Load cleaned data
  data <- read.csv("data/processed/imdb_cleaned_latest.csv")
  
  # Quick stats
  cat("Dataset Overview:\n")
  cat("  Rows:", nrow(data), "\n")
  cat("  Columns:", ncol(data), "\n")
  cat("  Average IMDb Score:", round(mean(data$imdb_score, na.rm = TRUE), 2), "\n")
  cat("  Average Budget:", scales::dollar(mean(data$budget, na.rm = TRUE)), "\n\n")
  
  # Top 5 movies
  cat("Top 5 Highest Rated Movies:\n")
  top5 <- head(data[order(-data$imdb_score), c("movie_title", "imdb_score", "title_year")], 5)
  print(top5)
  
  return(data)
}

# usage
 data <- quick_analysis()


# ===========================
# Custom Filtering & Analysis
# ===========================

analyze_by_genre <- function(data, genre_name) {
  cat(paste("\nAnalyzing", genre_name, "movies...\n\n"))
  
  # Filter by genre
  genre_data <- data %>%
    filter(primary_genre == genre_name)
  
  # Statistics
  stats <- data.frame(
    Metric = c("Count", "Avg IMDb Score", "Avg Budget", "Avg Gross", "Avg ROI"),
    Value = c(
      nrow(genre_data),
      round(mean(genre_data$imdb_score, na.rm = TRUE), 2),
      paste0("$", round(mean(genre_data$budget, na.rm = TRUE) / 1e6, 1), "M"),
      paste0("$", round(mean(genre_data$gross, na.rm = TRUE) / 1e6, 1), "M"),
      round(mean(genre_data$roi, na.rm = TRUE), 2)
    )
  )
  
  print(stats)
  
  # Top 3 movies
  cat("\nTop 3", genre_name, "Movies:\n")
  top3 <- head(genre_data[order(-genre_data$imdb_score), c("movie_title", "imdb_score")], 3)
  print(top3)
  
  return(genre_data)
}

# usage:
 data <- read.csv("data/processed/imdb_cleaned_latest.csv")
 action_movies <- analyze_by_genre(data, "Action")
 drama_movies <- analyze_by_genre(data, "Drama")


# ===========================
# Comparison of Two Directors
# ===========================

compare_directors <- function(data, director1, director2) {
  cat(paste("\nComparing:", director1, "vs", director2, "\n\n"))
  
  # Filter data
  d1_movies <- data %>% filter(director_name == director1)
  d2_movies <- data %>% filter(director_name == director2)
  
  # Comparison table
  comparison <- data.frame(
    Director = c(director1, director2),
    Movies = c(nrow(d1_movies), nrow(d2_movies)),
    Avg_IMDb = c(
      round(mean(d1_movies$imdb_score, na.rm = TRUE), 2),
      round(mean(d2_movies$imdb_score, na.rm = TRUE), 2)
    ),
    Total_Profit = c(
      paste0("$", round(sum(d1_movies$profit, na.rm = TRUE) / 1e9, 2), "B"),
      paste0("$", round(sum(d2_movies$profit, na.rm = TRUE) / 1e9, 2), "B")
    ),
    Avg_ROI = c(
      round(mean(d1_movies$roi, na.rm = TRUE), 2),
      round(mean(d2_movies$roi, na.rm = TRUE), 2)
    )
  )
  
  print(comparison)
  
  # Visualization
  combined <- rbind(
    d1_movies %>% mutate(Director = director1),
    d2_movies %>% mutate(Director = director2)
  )
  
  p <- ggplot(combined, aes(x = Director, y = imdb_score, fill = Director)) +
    geom_boxplot(alpha = 0.7) +
    labs(title = paste(director1, "vs", director2),
         y = "IMDb Score") +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(p)
  
  return(comparison)
}

# usage
 data <- read.csv("data/processed/imdb_cleaned_latest.csv")
 compare_directors(data, "Christopher Nolan", "Steven Spielberg")


# ===========================
# Find Similar Movies
# ===========================

find_similar_movies <- function(data, movie_title, n = 5) {
  cat(paste("\nFinding movies similar to:", movie_title, "\n\n"))
  
  # Get target movie
  target <- data %>% filter(movie_title == movie_title)
  
  if (nrow(target) == 0) {
    cat("Movie not found!\n")
    return(NULL)
  }
  
  # Calculate similarity score (we did it based on genre, year, budget)
  data <- data %>%
    mutate(
      similarity_score = (
        (primary_genre == target$primary_genre) * 3 +  # Genre match is important otherwise it aint similar is it?
          (1 - abs(title_year - target$title_year) / 100) +  # Year proximity
          (1 - abs(log(budget) - log(target$budget)) / 10)   # Budget similarity
      )
    ) %>%
    filter(movie_title != movie_title) %>%  # Exclude the target movie itself
    arrange(desc(similarity_score))
  
  similar <- head(data[, c("movie_title", "primary_genre", "title_year", 
                           "imdb_score", "similarity_score")], n)
  
  cat("Target Movie:\n")
  print(target[, c("movie_title", "primary_genre", "title_year", "imdb_score")])
  
  cat("\nSimilar Movies:\n")
  print(similar)
  
  return(similar)
}

# usage
 data <- read.csv("data/processed/imdb_cleaned_latest.csv")
 find_similar_movies(data, "The Dark Knight", n = 5)


# ===========================
# Predict Score for Hypothetical Movie
# ===========================

 predict_movie_score <- function(budget, duration, genre, content_rating = "PG-13",
                                 num_votes = 50000, 
                                 model_name = "best_model",
                                 model_path = "output/models") {
   
   cat("\nPredicting IMDb score for hypothetical movie...\n\n")
   
   # Load model with metadata
   model_file <- file.path(model_path, paste0(model_name, ".rds"))
   levels_file <- file.path(model_path, paste0(model_name, "_factor_levels.rds"))
   
   if (!file.exists(model_file)) {
     cat("Model not found! Please run the modeling pipeline first.\n")
     cat("Expected location:", model_file, "\n")
     return(NULL)
   }
   
   model <- readRDS(model_file)
   
   # Load factor levels if available
   factor_levels <- NULL
   if (file.exists(levels_file)) {
     factor_levels <- readRDS(levels_file)
     cat("Loaded factor levels metadata\n")
   } else {
     cat("Warning: Factor levels metadata not found. Using training data instead.\n")
     # Fallback: load training data
     training_data_path <- "data/processed/imdb_cleaned_latest.csv"
     if (file.exists(training_data_path)) {
       training_data <- read.csv(training_data_path, stringsAsFactors = FALSE)
     } else {
       cat("Error: Cannot find training data or factor levels!\n")
       return(NULL)
     }
   }
   
   # Create new data
   new_movie <- data.frame(
     budget = budget,
     duration = duration,
     primary_genre = genre,
     content_rating = content_rating,
     num_voted_users = num_votes,
     num_critic_for_reviews = 200,
     num_user_for_reviews = 400,
     movie_facebook_likes = 5000,
     language = "English",
     country = "USA",
     stringsAsFactors = FALSE
   )
   
   # Convert categorical variables to factors with correct levels
   categorical_vars <- c("primary_genre", "content_rating", "language", "country")
   
   for (var in categorical_vars) {
     if (var %in% names(new_movie)) {
       
       # Get levels from metadata or training data
       if (!is.null(factor_levels) && var %in% names(factor_levels)) {
         valid_levels <- factor_levels[[var]]
       } else if (exists("training_data")) {
         valid_levels <- unique(training_data[[var]])
       } else {
         cat("Warning: Cannot determine valid levels for", var, "\n")
         next
       }
       
       # Check if value is valid
       if (!(new_movie[[var]] %in% valid_levels)) {
         cat(sprintf("Warning: '%s' = '%s' not found in training data.\n", 
                     var, new_movie[[var]]))
         cat(sprintf("Available %s: %s\n", 
                     var, paste(head(sort(valid_levels), 5), collapse = ", ")))
         
         # Use most common value
         if (!is.null(factor_levels)) {
           new_movie[[var]] <- valid_levels[1]
         } else {
           new_movie[[var]] <- names(sort(table(training_data[[var]]), 
                                          decreasing = TRUE))[1]
         }
         cat(sprintf("Using '%s' instead.\n\n", new_movie[[var]]))
       }
       
       # Convert to factor
       new_movie[[var]] <- factor(new_movie[[var]], levels = valid_levels)
     }
   }
   
   # Make prediction
   tryCatch({
     prediction <- predict(model, newdata = new_movie)
     
     cat("Movie Specifications:\n")
     cat("  Budget:", scales::dollar(budget), "\n")
     cat("  Duration:", duration, "minutes\n")
     cat("  Genre:", as.character(new_movie$primary_genre), "\n")
     cat("  Content Rating:", as.character(new_movie$content_rating), "\n\n")
     
     cat("Predicted IMDb Score:", round(prediction, 2), "/10\n")
     
     # Rating interpretation
     if (prediction >= 7.5) {
       cat("Rating: Excellent! This should be a hit!\n")
     } else if (prediction >= 6.5) {
       cat("Rating: Good. Solid movie.\n")
     } else if (prediction >= 5.5) {
       cat("Rating: Average. Mixed reviews expected.\n")
     } else {
       cat("Rating: Below average. May struggle.\n")
     }
     
     return(prediction)
     
   }, error = function(e) {
     cat("Prediction failed with error:\n")
     cat(e$message, "\n\n")
     cat("Debugging info:\n")
     cat("Model class:", class(model), "\n")
     cat("New data structure:\n")
     str(new_movie)
     return(NULL)
   })
 }
 
 
 # Show available prediction options
 #
 #param model_name Model name
 #param model_path Path to models
 show_prediction_options <- function(model_name = "best_model", 
                                     model_path = "output/models") {
   
   levels_file <- file.path(model_path, paste0(model_name, "_factor_levels.rds"))
   
   if (file.exists(levels_file)) {
     factor_levels <- readRDS(levels_file)
     
     cat("========================================\n")
     cat("Available Prediction Options\n")
     cat("========================================\n\n")
     
     for (var in names(factor_levels)) {
       cat(sprintf("%s:\n", var))
       levels_to_show <- head(factor_levels[[var]], 10)
       for (level in levels_to_show) {
         cat(sprintf("  - %s\n", level))
       }
       if (length(factor_levels[[var]]) > 10) {
         cat(sprintf("  ... and %d more\n", length(factor_levels[[var]]) - 10))
       }
       cat("\n")
     }
     
   } else {
     cat("Factor levels metadata not found. Using training data...\n")
     show_available_options()
   }
 }
 
 
 # usage
 show_prediction_options()
 predict_movie_score(budget = 140000000, duration = 140, genre = "Action")
 predict_movie_score(budget = 20000000, duration = 110, genre = "Drama")
 


# ===========================
# Analyze Budget vs Success
# ===========================

analyze_budget_efficiency <- function(data) {
  cat("\nAnalyzing budget efficiency...\n\n")
  
  # Create budget categories
  data <- data %>%
    mutate(
      budget_tier = case_when(
        budget < 10000000 ~ "Low (<$10M)",
        budget < 50000000 ~ "Medium ($10M-$50M)",
        budget < 100000000 ~ "High ($50M-$100M)",
        TRUE ~ "Blockbuster (>$100M)"
      )
    )
  
  # Summary by budget tier
  summary_stats <- data %>%
    group_by(budget_tier) %>%
    summarise(
      Count = n(),
      Avg_IMDb = round(mean(imdb_score, na.rm = TRUE), 2),
      Avg_ROI = round(mean(roi, na.rm = TRUE), 2),
      Success_Rate = round(100 * mean(is_profitable == 1, na.rm = TRUE), 1),
      .groups = "drop"
    )
  
  cat("Budget Tier Analysis:\n")
  print(summary_stats)
  
  # Visualization
  p <- ggplot(data, aes(x = budget_tier, y = roi, fill = budget_tier)) +
    geom_boxplot(alpha = 0.7) +
    scale_y_continuous(limits = c(-1, 5)) +  # Focus on reasonable ROI range
    labs(title = "ROI by Budget Tier",
         x = "Budget Tier",
         y = "Return on Investment") +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
  
  cat("\nKey Insight:\n")
  best_roi_tier <- summary_stats$budget_tier[which.max(summary_stats$Avg_ROI)]
  cat("  Best average ROI:", best_roi_tier, "\n")
  
  return(summary_stats)
}

# usage
 data <- read.csv("data/processed/imdb_cleaned_latest.csv")
 budget_analysis <- analyze_budget_efficiency(data)


# ===========================
# Time Series Analysis
# ===========================

analyze_trends_over_time <- function(data, start_year = 2000) {
  cat(paste("\nAnalyzing trends from", start_year, "onwards...\n\n"))
  
  # Filter recent years
  recent_data <- data %>%
    filter(title_year >= start_year) %>%
    group_by(title_year) %>%
    summarise(
      Movies = n(),
      Avg_Score = mean(imdb_score, na.rm = TRUE),
      Avg_Budget = mean(budget, na.rm = TRUE),
      Avg_ROI = mean(roi, na.rm = TRUE),
      .groups = "drop"
    )
  
  cat("Summary Statistics:\n")
  cat("  Years analyzed:", nrow(recent_data), "\n")
  cat("  Total movies:", sum(recent_data$Movies), "\n")
  cat("  Score trend:", 
      ifelse(tail(recent_data$Avg_Score, 1) > head(recent_data$Avg_Score, 1), 
             "Increasing ↑", "Decreasing ↓"), "\n\n")
  
  # Plot trends
  p1 <- ggplot(recent_data, aes(x = title_year, y = Avg_Score)) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "red") +
    geom_smooth(method = "lm", se = FALSE, color = "darkgreen", linetype = "dashed") +
    labs(title = "Average IMDb Score Trend",
         x = "Year", y = "Average Score") +
    theme_minimal()
  
  print(p1)
  
  return(recent_data)
}

# usage
 data <- read.csv("data/processed/imdb_cleaned_latest.csv")
 trends <- analyze_trends_over_time(data, start_year = 2000)


# ===========================
# Create Custom Report
# ===========================

generate_custom_report <- function(data, output_file = "movie_report.txt") {
  
  sink(output_file)  # Redirect output to file
  
  cat("========================================\n")
  cat("IMDB MOVIE ANALYSIS REPORT\n")
  cat("========================================\n\n")
  
  # Overview
  cat("DATASET OVERVIEW\n")
  cat("----------------------------------------\n")
  cat("Total Movies:", nrow(data), "\n")
  cat("Time Period:", min(data$title_year, na.rm = TRUE), "-", 
      max(data$title_year, na.rm = TRUE), "\n")
  cat("Unique Directors:", length(unique(data$director_name)), "\n")
  cat("Unique Genres:", length(unique(data$primary_genre)), "\n\n")
  
  # Key Statistics
  cat("KEY STATISTICS\n")
  cat("----------------------------------------\n")
  cat("Average IMDb Score:", round(mean(data$imdb_score, na.rm = TRUE), 2), "\n")
  cat("Average Budget:", scales::dollar(mean(data$budget, na.rm = TRUE)), "\n")
  cat("Average Gross:", scales::dollar(mean(data$gross, na.rm = TRUE)), "\n")
  cat("Average ROI:", round(mean(data$roi, na.rm = TRUE), 2), "\n")
  cat("Success Rate:", paste0(round(100 * mean(data$is_profitable == 1, na.rm = TRUE), 1), "%"), "\n\n")
  
  # Top Lists
  cat("TOP 10 HIGHEST RATED MOVIES\n")
  cat("----------------------------------------\n")
  top_rated <- head(data[order(-data$imdb_score), c("movie_title", "imdb_score", "title_year")], 10)
  print(top_rated)
  
  cat("\n\nTOP 10 MOST PROFITABLE MOVIES\n")
  cat("----------------------------------------\n")
  top_profit <- head(data[order(-data$profit), c("movie_title", "profit", "roi")], 10)
  print(top_profit)
  
  cat("\n\nTOP 10 DIRECTORS (by avg IMDb score, min 5 films)\n")
  cat("----------------------------------------\n")
  top_directors <- data %>%
    group_by(director_name) %>%
    summarise(Movies = n(), Avg_Score = mean(imdb_score, na.rm = TRUE)) %>%
    filter(Movies >= 5) %>%
    arrange(desc(Avg_Score)) %>%
    head(10)
  print(top_directors)
  
  cat("\n========================================\n")
  cat("END OF REPORT\n")
  cat("========================================\n")
  
  sink()  # Stop redirecting output
  
  cat("\nReport saved to:", output_file, "\n")
}

# usage
 data <- read.csv("data/processed/imdb_cleaned_latest.csv")
 generate_custom_report(data, "my_movie_report.txt")

 
 
 # ===========================
 # Predict Movie Profitability (Classification)
 # ===========================
 
 predict_profitability <- function(budget, duration, genre, imdb_score_estimate = 6.5,
                                   content_rating = "PG-13", num_votes = 50000,
                                   model_name = "best_classifier",
                                   model_path = "output/models") {
   
   cat("\nPredicting movie profitability...\n\n")
   
   # Load model with metadata
   model_file <- file.path(model_path, paste0(model_name, ".rds"))
   levels_file <- file.path(model_path, paste0(model_name, "_factor_levels.rds"))
   
   if (!file.exists(model_file)) {
     cat("Classifier model not found! Please run classification pipeline first.\n")
     cat("Run: source('R/06_classification.R')\n")
     return(NULL)
   }
   
   model <- readRDS(model_file)
   
   # Load factor levels if available
   factor_levels <- NULL
   if (file.exists(levels_file)) {
     factor_levels <- readRDS(levels_file)
   }
   
   # Create new data
   new_movie <- data.frame(
     budget = budget,
     duration = duration,
     imdb_score = imdb_score_estimate,
     primary_genre = genre,
     content_rating = content_rating,
     num_voted_users = num_votes,
     num_critic_for_reviews = 200,
     num_user_for_reviews = 400,
     movie_facebook_likes = 5000,
     language = "English",
     country = "USA",
     stringsAsFactors = FALSE
   )
   
   # Convert categorical variables to factors
   categorical_vars <- c("primary_genre", "content_rating", "language", "country")
   
   for (var in categorical_vars) {
     if (var %in% names(new_movie)) {
       if (!is.null(factor_levels) && var %in% names(factor_levels)) {
         valid_levels <- factor_levels[[var]]
       } else {
         # Fallback to training data
         training_data <- read.csv("data/processed/imdb_cleaned_latest.csv", 
                                   stringsAsFactors = FALSE)
         valid_levels <- unique(training_data[[var]])
       }
       
       if (!(new_movie[[var]] %in% valid_levels)) {
         cat(sprintf("Warning: '%s' not in training data. Using most common value.\n", 
                     new_movie[[var]]))
         new_movie[[var]] <- valid_levels[1]
       }
       
       new_movie[[var]] <- factor(new_movie[[var]], levels = valid_levels)
     }
   }
   
   # Make prediction
   tryCatch({
     if ("glm" %in% class(model)) {
       # Logistic regression
       prob <- predict(model, newdata = new_movie, type = "response")
       prediction <- ifelse(prob > 0.5, "Profitable", "Not Profitable")
     } else {
       # Random Forest
       prediction <- as.character(predict(model, newdata = new_movie))
       prob <- predict(model, newdata = new_movie, type = "prob")[, "Profitable"]
     }
     
     cat("Movie Specifications:\n")
     cat("  Budget:", scales::dollar(budget), "\n")
     cat("  Duration:", duration, "minutes\n")
     cat("  Genre:", as.character(new_movie$primary_genre), "\n")
     cat("  Estimated IMDb Score:", imdb_score_estimate, "\n\n")
     
     cat("Prediction:", prediction, "\n")
     cat("Confidence:", sprintf("%.1f%%", prob * 100), "\n\n")
     
     # Interpretation
     if (prob > 0.75) {
       cat(" High confidence this movie will be profitable!\n")
       cat("   Strong box office potential.\n")
     } else if (prob > 0.5) {
       cat(" Likely to be profitable, but not guaranteed.\n")
       cat("   Moderate risk investment.\n")
     } else if (prob > 0.25) {
       cat(" Unlikely to be profitable.\n")
       cat("   High risk investment.\n")
     } else {
       cat(" Very low chance of profitability.\n")
       cat("   Reconsider this project.\n")
     }
     
     # ROI estimate
     if (prob > 0.5) {
       expected_roi <- (prob - 0.5) * 2  # Simple estimate
       cat(sprintf("\nEstimated ROI if profitable: %.0f%%\n", expected_roi * 100))
     }
     
     return(list(prediction = prediction, probability = prob))
     
   }, error = function(e) {
     cat("Prediction failed:", e$message, "\n")
     return(NULL)
   })
 }
 
 
 # usage
predict_profitability(budget = 100000000, duration = 140, genre = "Action", imdb_score_estimate = 7.5)
predict_profitability(budget = 20000000, duration = 110, genre = "Drama", imdb_score_estimate = 6.0)
 
 
 # ===========================
 # Investment Decision Tool
 # ===========================
 
 investment_decision <- function(budget, duration, genre, imdb_score_estimate, content_rating = "PG-13") {
   
   cat("\n========================================\n")
   cat("MOVIE INVESTMENT DECISION TOOL\n")
   cat("========================================\n\n")
   
   # Predict IMDb score
   cat("1. Predicting IMDb Score...\n")
   score_pred <- predict_movie_score(budget, duration, genre, content_rating)
   
   # Use predicted or estimated score
   if (!is.null(score_pred)) {
     final_score <- score_pred
   } else {
     final_score <- imdb_score_estimate
   }
   
   cat("\n")
   
   # Predict profitability
   cat("2. Predicting Profitability...\n")
   profit_pred <- predict_profitability(budget, duration, genre, final_score, content_rating)
   
   # Investment recommendation
   cat("\n========================================\n")
   cat("INVESTMENT RECOMMENDATION\n")
   cat("========================================\n\n")
   
   if (!is.null(profit_pred)) {
     score_rating <- ifelse(final_score >= 7.0, "Good", ifelse(final_score >= 6.0, "Average", "Poor"))
     profit_rating <- ifelse(profit_pred$probability > 0.7, "High", 
                             ifelse(profit_pred$probability > 0.5, "Medium", "Low"))
     
     cat("Expected IMDb Score:", round(final_score, 2), sprintf("(%s)\n", score_rating))
     cat("Profitability Confidence:", sprintf("%.1f%%", profit_pred$probability * 100), 
         sprintf("(%s)\n\n", profit_rating))
     
     # Overall recommendation
     if (profit_pred$probability > 0.7 && final_score >= 6.5) {
       cat("RECOMMENDED: Strong investment opportunity!\n")
       cat("   Good balance of quality and profit potential.\n")
     } else if (profit_pred$probability > 0.6 && final_score >= 6.0) {
       cat("CONSIDER: Moderate investment with decent potential.\n")
       cat("   Watch budget carefully.\n")
     } else if (profit_pred$probability > 0.5) {
       cat("RISKY: Profitable but quality concerns.\n")
       cat("   May hurt brand reputation.\n")
     } else {
       cat("NOT RECOMMENDED: High risk of loss.\n")
       cat("   Consider alternative projects.\n")
     }
   }
   
   cat("\n========================================\n\n")
 }
 
 # usage
investment_decision(budget = 100000000, duration = 140, genre = "Action", imdb_score_estimate = 7.5)
investment_decision(budget = 20000000, duration = 110, genre = "Drama", imdb_score_estimate = 6.5)
 
 
 # ===========================
 # Example 12: Batch Investment Analysis
 # ===========================
 
 analyze_multiple_investments <- function(movie_portfolio) {
   
   cat("\n========================================\n")
   cat("PORTFOLIO INVESTMENT ANALYSIS\n")
   cat("========================================\n\n")
   
   results <- data.frame()
   
   for (i in 1:nrow(movie_portfolio)) {
     cat(sprintf("\n--- Movie %d: %s ---\n", i, movie_portfolio$title[i]))
     
     # Predict score
     score <- predict_movie_score(
       budget = movie_portfolio$budget[i],
       duration = movie_portfolio$duration[i],
       genre = movie_portfolio$genre[i],
       content_rating = movie_portfolio$content_rating[i]
     )
     
     # Predict profitability
     profit <- predict_profitability(
       budget = movie_portfolio$budget[i],
       duration = movie_portfolio$duration[i],
       genre = movie_portfolio$genre[i],
       imdb_score_estimate = score,
       content_rating = movie_portfolio$content_rating[i]
     )
     
     if (!is.null(score) && !is.null(profit)) {
       results <- rbind(results, data.frame(
         Title = movie_portfolio$title[i],
         Budget = movie_portfolio$budget[i],
         Genre = movie_portfolio$genre[i],
         Predicted_Score = round(score, 2),
         Profit_Probability = round(profit$probability * 100, 1),
         Recommendation = ifelse(profit$probability > 0.6 && score >= 6.5, 
                                 "Approve", 
                                 ifelse(profit$probability > 0.5, " Review", "Reject"))
       ))
     }
   }
   
   cat("\n========================================\n")
   cat("PORTFOLIO SUMMARY\n")
   cat("========================================\n")
   print(results)
   
   # Summary statistics
   cat("\n")
   cat("Total Projects:", nrow(results), "\n")
   cat("Recommended:", sum(grepl("Approve", results$Recommendation)), "\n")
   cat("Need Review:", sum(grepl("Review", results$Recommendation)), "\n")
   cat("Rejected:", sum(grepl("Reject", results$Recommendation)), "\n")
   cat("Total Budget:", scales::dollar(sum(results$Budget)), "\n")
   cat("Average Predicted Score:", round(mean(results$Predicted_Score), 2), "\n")
   cat("Average Success Probability:", 
       sprintf("%.1f%%", mean(results$Profit_Probability)), "\n")
   
   return(results)
 }
 
 # usage:
 portfolio <- data.frame(
    title = c("Action Blockbuster", "Indie Drama", "Family Comedy", "Horror Thriller"),
    budget = c(150000000, 15000000, 40000000, 8000000),
    duration = c(150, 105, 95, 90),
    genre = c("Action", "Drama", "Comedy", "Horror"),
    content_rating = c("PG-13", "R", "PG", "R")
  )
 results <- analyze_multiple_investments(portfolio)
 
 
 # ===========================
 # Example 13: Risk Assessment
 # ===========================
 
 assess_investment_risk <- function(budget, duration, genre, imdb_score_estimate) {
   
   cat("\n========================================\n")
   cat("INVESTMENT RISK ASSESSMENT\n")
   cat("========================================\n\n")
   
   # Get profitability prediction
   profit_pred <- predict_profitability(budget, duration, genre, imdb_score_estimate)
   
   if (is.null(profit_pred)) {
     cat("Could not assess risk. Check inputs.\n")
     return(NULL)
   }
   
   prob <- profit_pred$probability
   
   # Calculate risk metrics
   expected_return <- prob  # Simplified
   risk_score <- 1 - prob
   
   # Risk categories
   cat("\nRisk Analysis:\n")
   cat("----------------------------------------\n")
   cat(sprintf("Probability of Success: %.1f%%\n", prob * 100))
   cat(sprintf("Risk Score: %.1f/10\n", risk_score * 10))
   
   if (risk_score < 0.3) {
     risk_level <- "LOW RISK"
     color <- "🟢"
   } else if (risk_score < 0.5) {
     risk_level <- "MODERATE RISK"
     color <- "🟡"
   } else if (risk_score < 0.7) {
     risk_level <- "HIGH RISK"
     color <- "🟠"
   } else {
     risk_level <- "VERY HIGH RISK"
     color <- "🔴"
   }
   
   cat(sprintf("\nRisk Level: %s %s\n", color, risk_level))
   
   # Risk factors
   cat("\nRisk Factors:\n")
   
   # Budget risk
   if (budget > 100000000) {
     cat("   Very high budget - significant financial exposure\n")
   } else if (budget > 50000000) {
     cat("  High budget - moderate financial risk\n")
   } else {
     cat("  Manageable budget - lower financial risk\n")
   }
   
   # Duration risk
   if (duration > 150) {
     cat("    Very long duration - audience fatigue risk\n")
   } else if (duration < 90) {
     cat("    Short duration - may feel incomplete\n")
   } else {
     cat("  Standard duration - optimal range\n")
   }
   
   # Genre risk
   high_risk_genres <- c("Horror", "Musical", "Western")
   if (genre %in% high_risk_genres) {
     cat(sprintf("    Genre '%s' is historically higher risk\n", genre))
   }
   
   # Score risk
   if (imdb_score_estimate < 6.0) {
     cat("    Low predicted score - poor word-of-mouth risk\n")
   } else if (imdb_score_estimate >= 7.5) {
     cat("  High predicted score - strong word-of-mouth potential\n")
   }
   
   # Recommendations
   cat("\nMitigation Strategies:\n")
   if (budget > 100000000) {
     cat("  • Consider reducing budget to minimize exposure\n")
     cat("  • Secure pre-sales and distribution deals\n")
     cat("  • Build strong marketing campaign\n")
   }
   if (prob < 0.5) {
     cat("  • Review script and creative elements\n")
     cat("  • Consider A-list talent to boost appeal\n")
     cat("  • Focus on international markets\n")
   }
   if (imdb_score_estimate < 6.5) {
     cat("  • Invest in script development\n")
     cat("  • Consider experienced director\n")
     cat("  • Allow adequate pre-production time\n")
   }
   
   cat("\n========================================\n\n")
   
   return(list(
     probability = prob,
     risk_score = risk_score,
     risk_level = risk_level
   ))
 }
 
 # usage:
 assess_investment_risk(budget = 150000000, duration = 150, genre = "Action", imdb_score_estimate = 7.0)
assess_investment_risk(budget = 10000000, duration = 95, genre = "Horror", imdb_score_estimate = 5.5)
 
 

# ===========================
#  Interactive Menu
# ===========================

interactive_analysis <- function() {
  
  # Load data
  if (!file.exists("data/processed/imdb_cleaned_latest.csv")) {
    cat("Please run main.R first to generate cleaned data!\n")
    return()
  }
  
  data <- read.csv("data/processed/imdb_cleaned_latest.csv")
  
  repeat {
    cat("\n========================================\n")
    cat("IMDB MOVIE ANALYSIS - INTERACTIVE MODE\n")
    cat("========================================\n")
    cat("1. Show dataset overview\n")
    cat("2. Analyze by genre\n")
    cat("3. Compare directors\n")
    cat("4. Find similar movies\n")
    cat("5. Predict movie IMDb score\n")
    cat("6. Predict movie profitability\n")
    cat("7. Investment decision tool\n")
    cat("8. Risk assessment\n")
    cat("9. Budget efficiency analysis\n")
    cat("10. Generate custom report\n")
    cat("11. Exit\n")
    cat("----------------------------------------\n")
    
    choice <- readline("Enter your choice (1-11): ")
    
    if (choice == "1") {
      quick_analysis()
      
    } else if (choice == "2") {
      genre <- readline("Enter genre (e.g., Action, Drama): ")
      analyze_by_genre(data, genre)
      
    } else if (choice == "3") {
      dir1 <- readline("Enter first director name: ")
      dir2 <- readline("Enter second director name: ")
      compare_directors(data, dir1, dir2)
      
    } else if (choice == "4") {
      movie <- readline("Enter movie title: ")
      find_similar_movies(data, movie)
      
    } else if (choice == "5") {
      budget <- as.numeric(readline("Enter budget (USD): "))
      duration <- as.numeric(readline("Enter duration (minutes): "))
      genre <- readline("Enter genre: ")
      predict_movie_score(budget, duration, genre)
      
    } else if (choice == "6") {
      budget <- as.numeric(readline("Enter budget (USD): "))
      duration <- as.numeric(readline("Enter duration (minutes): "))
      genre <- readline("Enter genre: ")
      score_est <- as.numeric(readline("Enter estimated IMDb score (6-8): "))
      predict_profitability(budget, duration, genre, score_est)
      
    } else if (choice == "7") {
      budget <- as.numeric(readline("Enter budget (USD): "))
      duration <- as.numeric(readline("Enter duration (minutes): "))
      genre <- readline("Enter genre: ")
      score_est <- as.numeric(readline("Enter estimated IMDb score: "))
      investment_decision(budget, duration, genre, score_est)
      
    } else if (choice == "8") {
      budget <- as.numeric(readline("Enter budget (USD): "))
      duration <- as.numeric(readline("Enter duration (minutes): "))
      genre <- readline("Enter genre: ")
      score_est <- as.numeric(readline("Enter estimated IMDb score: "))
      assess_investment_risk(budget, duration, genre, score_est)
      
    } else if (choice == "9") {
      analyze_budget_efficiency(data)
      
    } else if (choice == "10") {
      filename <- readline("Enter output filename (e.g., report.txt): ")
      generate_custom_report(data, filename)
      
    } else if (choice == "11") {
      cat("\nGoodbye! \n")
      break
      
    } else {
      cat("\nInvalid choice. Please try again.\n")
    }
    
    readline("\nPress Enter to continue...")
  }
}

# Run interactive mode
interactive_analysis()
