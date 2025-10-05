# ===========================
# 03_eda.R
# Exploratory Data Analysis
# ===========================

# helper functions
source("utils/helper_functions.R")

# Required libraries
library(dplyr)
library(ggplot2)
library(reshape2)

# Generate descriptive statistics table
#
# input: Data frame
# input: Vector of numeric column names
# returns a data frame with statistics
generate_stats_table <- function(data, numeric_vars) {
  cat("Generating descriptive statistics...\n")
  
  stats_list <- lapply(numeric_vars, function(var) {
    if (var %in% names(data)) {
      get_descriptive_stats(data[[var]], var)
    }
  })
  
  stats_df <- do.call(rbind, stats_list)
  return(stats_df)
}


# Plot distribution of a numeric variable
#
# input: Data frame
# input: Variable name
# input: Plot title
# input: Number of bins for histogram
# input: Use log scale for x-axis
# input: Path to save plot
plot_distribution <- function(data, var, title, bins = 30, 
                              log_scale = FALSE, save_path = NULL) {
  
  p <- ggplot(data, aes_string(x = var)) +
    geom_histogram(bins = bins, fill = "skyblue", color = "black", alpha = 0.7) +
    labs(title = title, x = var, y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (log_scale) {
    p <- p + scale_x_log10()
  }
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Saved:", save_path, "\n")
  }
  
  return(p)
}


# Plot scatter plot with regression line
#
# input: Data frame
# input: X variable name
# input: Y variable name
# input: Plot title
# input: Use log scale for x-axis
# input: Use log scale for y-axis
# input: Path to save plot
plot_scatter <- function(data, x_var, y_var, title, 
                         log_x = FALSE, log_y = FALSE, save_path = NULL) {
  
  p <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.5, color = "darkblue") +
    geom_smooth(method = "lm", color = "red", se = FALSE) +
    labs(title = title, x = x_var, y = y_var) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (log_x) p <- p + scale_x_log10()
  if (log_y) p <- p + scale_y_log10()
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Saved:", save_path, "\n")
  }
  
  return(p)
}


# Plot boxplot by categorical variable
#
# input: Data frame
# input: Categorical variable
# input: Numeric variable
# input: Plot title
# input: Flip coordinates
# input: Path to save plot
plot_boxplot_by_category <- function(data, cat_var, num_var, title, 
                                     flip = TRUE, save_path = NULL) {
  
  p <- ggplot(data, aes_string(x = cat_var, y = num_var)) +
    geom_boxplot(fill = "lightblue", alpha = 0.7) +
    labs(title = title, x = cat_var, y = num_var) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (flip) {
    p <- p + coord_flip()
  }
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 6)
    cat("  Saved:", save_path, "\n")
  }
  
  return(p)
}


# Plot bar chart for categorical variable
#
# input: Data frame
# input: Variable name
# input: Plot title
# input: Number of top categories to show
# input: Path to save plot
plot_bar_chart <- function(data, var, title, top_n = 10, save_path = NULL) {
  
  freq_table <- sort(table(data[[var]]), decreasing = TRUE)
  freq_df <- data.frame(
    Category = names(freq_table),
    Count = as.numeric(freq_table),
    stringsAsFactors = FALSE
  )
  
  plot_data <- head(freq_df, top_n)
  
  p <- ggplot(plot_data, aes(x = reorder(Category, Count), y = Count)) +
    geom_col(fill = "coral", alpha = 0.7) +
    coord_flip() +
    labs(title = title, x = var, y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 8, height = 6)
    cat("  Saved:", save_path, "\n")
  }
  
  return(p)
}


# Plot time series trend
#
# input: Data frame
# input: Time variable
# input: Value variable
# input: Plot title
# input: Path to save plot
plot_time_trend <- function(data, time_var, value_var, title, save_path = NULL) {
  
  trend_data <- data %>%
    group_by(.data[[time_var]]) %>%
    summarise(avg_value = mean(.data[[value_var]], na.rm = TRUE), .groups = "drop")
  
  p <- ggplot(trend_data, aes_string(x = time_var, y = "avg_value")) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "red", size = 2) +
    labs(title = title, x = time_var, y = paste("Average", value_var)) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 6)
    cat("  Saved:", save_path, "\n")
  }
  
  return(p)
}


# Create correlation heatmap
#
# input: Data frame
# input: Vector of numeric column names
# input: Path to save plot
plot_correlation_heatmap <- function(data, numeric_vars, save_path = NULL) {
  
  # Select numeric columns that exist
  available_vars <- numeric_vars[numeric_vars %in% names(data)]
  num_df <- data[, available_vars]
  
  # Calculate correlation matrix
  cor_mat <- cor(num_df, use = "pairwise.complete.obs")
  cor_melt <- reshape2::melt(cor_mat)
  
  p <- ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 3) +
    scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                         midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold")) +
    labs(title = "Correlation Heatmap", fill = "Correlation")
  
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 12, height = 10)
    cat("  Saved:", save_path, "\n")
  }
  
  return(p)
}


# Generate top rankings
#
# input: Data frame
# input: Variable to rank by
# input: Variables to display
# input: n Number of top items
# input: Sort ascending if TRUE
# returns a Data frame with top rankings
get_top_rankings <- function(data, rank_var, display_vars, n = 10, ascending = FALSE) {
  
  if (ascending) {
    top_data <- head(data[order(data[[rank_var]]), display_vars], n)
  } else {
    top_data <- head(data[order(-data[[rank_var]]), display_vars], n)
  }
  
  return(top_data)
}


# EDA pipeline
#
# input: Cleaned data frame
# input: save figures to output folder
# returns a List of plots and statistics
perform_eda <- function(data, save_figures = TRUE) {
  
  cat("\n========================================\n")
  cat("Exploratory Data Analysis\n")
  cat("========================================\n\n")
  
  # Create output directory
  if (save_figures && !dir.exists("output/figures")) {
    dir.create("output/figures", recursive = TRUE)
  }
  
  # Define save path
  get_save_path <- function(filename) {
    if (save_figures) file.path("output/figures", filename) else NULL
  }
  
  # 1. Descriptive Statistics
  cat("1. Generating descriptive statistics...\n")
  numeric_vars <- c("budget", "gross", "profit", "roi", "imdb_score", 
                    "duration", "num_voted_users")
  stats_table <- generate_stats_table(data, numeric_vars)
  print(stats_table)
  
  # 2. Distribution Plots
  cat("\n2. Creating distribution plots...\n")
  plot_distribution(data, "imdb_score", "Distribution of IMDb Scores", 
                    save_path = get_save_path("dist_imdb_score.png"))
  
  plot_distribution(data, "budget", "Distribution of Budget (Log Scale)", 
                    log_scale = TRUE, save_path = get_save_path("dist_budget.png"))
  
  plot_distribution(data, "roi", "Distribution of ROI", 
                    save_path = get_save_path("dist_roi.png"))
  
  # 3. Scatter Plots (Relationships)
  cat("\n3. Creating scatter plots...\n")
  plot_scatter(data, "budget", "gross", "Budget vs Gross Revenue", 
               log_x = TRUE, log_y = TRUE, 
               save_path = get_save_path("scatter_budget_gross.png"))
  
  plot_scatter(data, "imdb_score", "profit", "IMDb Score vs Profit", 
               save_path = get_save_path("scatter_imdb_profit.png"))
  
  plot_scatter(data, "roi", "imdb_score", "ROI vs IMDb Score", 
               save_path = get_save_path("scatter_roi_imdb.png"))
  
  # 4. Categorical Analysis
  cat("\n4. Creating categorical plots...\n")
  
  # IMDb by Genre
  genre_data <- data %>%
    group_by(primary_genre) %>%
    mutate(median_score = median(imdb_score, na.rm = TRUE)) %>%
    ungroup()
  
  ggplot(genre_data, aes(x = reorder(primary_genre, imdb_score, median), 
                         y = imdb_score)) +
    geom_boxplot(fill = "lightyellow", alpha = 0.7) +
    coord_flip() +
    labs(title = "IMDb Scores by Genre", x = "Genre", y = "IMDb Score") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  if (save_figures) ggsave(get_save_path("boxplot_genre_imdb.png"), width = 10, height = 8)
  
  # Most Frequent Genres
  plot_bar_chart(data, "primary_genre", "Most Frequent Genres (Top 10)", 
                 save_path = get_save_path("bar_genres.png"))
  
  # IMDb by Content Rating
  if ("content_rating" %in% names(data)) {
    plot_boxplot_by_category(data, "content_rating", "imdb_score", 
                             "IMDb Scores by Content Rating",
                             flip = FALSE,
                             save_path = get_save_path("boxplot_rating_imdb.png"))
  }
  
  # 5. Time Trends
  cat("\n5. Creating time trend plots...\n")
  plot_time_trend(data, "title_year", "imdb_score", 
                  "Average IMDb Score by Year",
                  save_path = get_save_path("trend_imdb_year.png"))
  
  plot_time_trend(data, "title_year", "profit", 
                  "Average Profit by Year",
                  save_path = get_save_path("trend_profit_year.png"))
  
  plot_time_trend(data, "title_year", "roi", 
                  "Average ROI by Year",
                  save_path = get_save_path("trend_roi_year.png"))
  
  # 6. Correlation Heatmap
  cat("\n6. Creating correlation heatmap...\n")
  cor_vars <- c("budget", "gross", "profit", "roi", "imdb_score", 
                "duration", "num_voted_users", "num_critic_for_reviews",
                "num_user_for_reviews", "movie_facebook_likes")
  plot_correlation_heatmap(data, cor_vars, 
                           save_path = get_save_path("heatmap_correlation.png"))
  
  # 7. Rankings & Leaderboards
  cat("\n7. Generating rankings...\n\n")
  
  cat("Top 10 Movies by IMDb Score:\n")
  top_imdb <- get_top_rankings(data, "imdb_score", 
                               c("movie_title", "imdb_score", "title_year", "roi"))
  print(top_imdb)
  
  cat("\nTop 10 Movies by ROI:\n")
  top_roi <- get_top_rankings(data, "roi", 
                              c("movie_title", "roi", "title_year", "imdb_score"))
  print(top_roi)
  
  cat("\nTop 10 Most Profitable Directors:\n")
  top_directors <- data %>%
    group_by(director_name) %>%
    summarise(total_profit = sum(profit, na.rm = TRUE),
              n_movies = n()) %>%
    arrange(desc(total_profit)) %>%
    head(10)
  print(top_directors)
  
  cat("\nTop 10 Directors by Average IMDb Score (≥5 films):\n")
  top_dir_imdb <- data %>%
    group_by(director_name) %>%
    summarise(avg_imdb = mean(imdb_score, na.rm = TRUE),
              n_movies = n()) %>%
    filter(n_movies >= 5) %>%
    arrange(desc(avg_imdb)) %>%
    head(10)
  print(top_dir_imdb)
  
  cat("\n========================================\n")
  cat("EDA Complete!\n")
  if (save_figures) {
    cat("All figures saved to: output/figures/\n")
  }
  cat("========================================\n\n")
  
  return(list(
    stats = stats_table,
    top_imdb = top_imdb,
    top_roi = top_roi,
    top_directors = top_directors,
    top_dir_imdb = top_dir_imdb
  ))
}


# ===========================
# Main Execution
# ===========================

if (!interactive()) {
  # Load cleaned data
  imdb_clean <- read.csv("data/processed/imdb_cleaned_latest.csv", 
                         stringsAsFactors = FALSE)
  
  # Perform EDA
  eda_results <- perform_eda(imdb_clean, save_figures = TRUE)
  
  cat("\nEDA complete! Check output/figures/ for visualizations.\n")
}