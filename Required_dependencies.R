# ===========================
# Required_dependencies.R
# the required dependencies script for IMDB Movie Analysis project
# ===========================

cat("========================================\n")
cat("IMDB Movie Analysis - Setup\n")
cat("========================================\n\n")

# ===========================
# 1. Check R Version
# ===========================

cat("Checking R version...\n")
r_version <- R.version.string
cat("  ", r_version, "\n")

if (getRversion() < "4.0.0") {
  cat("  WARNING: R version 4.0.0 or higher is recommended\n")
} else {
  cat("  R version is compatible\n")
}

# ===========================
# 2. Create Directory Structure
# ===========================

cat("\nCreating project directories...\n")

directories <- c(
  "data/raw",
  "data/processed",
  "R",
  "utils",
  "output/figures",
  "output/models"
)

for (dir in directories) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    cat("  Created:", dir, "\n")
  } else {
    cat("  Exists:", dir, "\n")
  }
}

# ===========================
# 3. Install Required Packages
# ===========================

cat("\n========================================\n")
cat("Installing Required Packages\n")
cat("========================================\n\n")

# List of required packages
required_packages <- c(
  "dplyr",           # Data manipulation
  "janitor",         # Data cleaning
  "ggplot2",         # Visualizations
  "reshape2",        # Data reshaping
  "caret",           # Machine learning framework
  "randomForest"     # Random Forest algorithm
)

cat("Required packages:\n")
for (pkg in required_packages) {
  cat("  -", pkg, "\n")
}

# Check which packages are missing
installed <- installed.packages()[, "Package"]
missing <- required_packages[!(required_packages %in% installed)]

if (length(missing) > 0) {
  cat("\nInstalling missing packages:", paste(missing, collapse = ", "), "\n\n")
  
  for (pkg in missing) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, dependencies = TRUE)
  }
  
  cat("\n All packages installed successfully!\n")
} else {
  cat("\n All required packages are already installed!\n")
}

# ===========================
# 4. Verify Installation
# ===========================

cat("\n========================================\n")
cat("Verifying Installation\n")
cat("========================================\n\n")

all_installed <- TRUE

for (pkg in required_packages) {
  result <- suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))
  
  if (result) {
    pkg_version <- packageVersion(pkg)
    cat("  Done", pkg, "-", as.character(pkg_version), "\n")
  } else {
    cat("  Not Done", pkg, "- FAILED TO LOAD\n")
    all_installed <- FALSE
  }
}


# ===========================
# 5. Final Setup Check
# ===========================

cat("\n========================================\n")
cat("Setup Complete!\n")
cat("========================================\n\n")

if (all_installed) {
  cat(" All packages installed and working\n")
  cat(" Directory structure created\n")
  cat(" Ready to run analysis!\n\n")
  
} else {
  cat("⚠ Some packages failed to install\n")
  cat("  Please install them manually and run this script again\n\n")
}