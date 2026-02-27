# ============================================================================
# NEURAL NETWORK MODEL WITH K-FOLD CROSS-VALIDATION AND AUC ANALYSIS (R)
# ============================================================================
# This section builds a fully connected neural network (FCNN/MLP),
# performs k-fold cross-validation, and calculates AUC with confidence intervals
# to compare performance against the logistic regression model

# ============================================================================
# INSTALL AND LOAD REQUIRED LIBRARIES
# ============================================================================
# Install packages if not already installed (uncomment to run)
# install.packages("keras")
# install.packages("tensorflow")
# install.packages("caret")
# install.packages("pROC")
# install.packages("ggplot2")

library(keras)          # Deep learning with Keras/TensorFlow
library(tensorflow)     # TensorFlow backend
library(caret)          # Machine learning tools including createFolds
library(pROC)           # ROC curve analysis and AUC calculation
library(ggplot2)        # Data visualization
library(tidyverse)      # Data manipulation

# ============================================================================
# PREPARE DATA FOR TRAINING
# ============================================================================
# Extract features (X) from the dataframe
# Select columns 2-6 (columns 1-5 in 0-indexed Python, but 1-indexed in R)
# These are the 5 signature genes
X_train <- as.matrix(myseeddf[, 2:6])

# Extract target variable (y) and convert to binary (0 or 1)
# Convert factor levels to numeric codes
y_train <- as.numeric(myseeddf$Status) - 1  # Subtract 1 to get 0 and 1

# Verify data dimensions
cat("X_train dimensions:", nrow(X_train), "x", ncol(X_train), "\n")
cat("y_train dimensions:", length(y_train), "\n")

# ============================================================================
# SETUP K-FOLD CROSS-VALIDATION
# ============================================================================
# Create a 5-fold cross-validation splitter
# number=5: divide data into 5 equal folds
# returnTrain=TRUE: return training indices in addition to test indices
set.seed(42)  # Set seed for reproducibility
kfold_indices <- createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)

# ============================================================================
# INITIALIZE LISTS TO STORE RESULTS FROM EACH FOLD
# ============================================================================
# Lists to store ROC curve data (FPR and TPR pairs)
train_roc_list <- list()      # Store FPR/TPR pairs for each training fold
val_roc_list <- list()        # Store FPR/TPR pairs for each validation fold

# Lists to store AUC scores
train_auc_list <- numeric()   # Store AUC scores for training sets
val_auc_list <- numeric()     # Store AUC scores for validation sets

# Lists to store 95% confidence intervals for AUC
train_ci_list <- list()       # Store 95% CI for training AUC
val_ci_list <- list()         # Store 95% CI for validation AUC

# ============================================================================
# FUNCTION TO CALCULATE AUC CONFIDENCE INTERVALS
# ============================================================================
bootstrap_auc_ci <- function(y_true, y_pred, n_bootstraps = 1000, alpha = 0.95) {
  """
  Calculate confidence interval for AUC using bootstrap resampling.
  
  This method repeatedly samples from the data (with replacement) and
  calculates AUC for each sample to estimate the distribution of AUC values.
  
  Parameters:
  -----------
  y_true : numeric vector
      True binary class labels (0 or 1)
  y_pred : numeric vector
      Predicted probabilities from the model (0 to 1)
  n_bootstraps : integer
      Number of bootstrap samples to generate (default: 1000)
      More samples = more accurate CI but slower computation
  alpha : numeric
      Confidence level (default: 0.95 for 95% confidence interval)
  
  Returns:
  --------
  list : (lower_bound, upper_bound) of the confidence interval
  """
  
  # Initialize vector to store AUC scores from each bootstrap sample
  bootstrapped_scores <- numeric(n_bootstraps)
  
  # Set seed for reproducibility
  set.seed(42)
  
  # Generate n_bootstraps samples by random sampling with replacement
  for (i in 1:n_bootstraps) {
    # Randomly select indices with replacement
    # This creates a bootstrap sample the same size as the original data
    indices <- sample(1:length(y_pred), size = length(y_pred), replace = TRUE)
    
    # Skip if bootstrap sample doesn't contain both classes (0 and 1)
    # AUC cannot be calculated with only one class
    if (length(unique(y_true[indices])) < 2) {
      next
    }
    
    # Calculate AUC for this bootstrap sample using pROC package
    # Calculate ROC curve and extract AUC
    roc_obj <- roc(y_true[indices], y_pred[indices], quiet = TRUE)
    bootstrapped_scores[i] <- as.numeric(roc_obj$auc)
  }
  
  # Remove any NA values (from skipped iterations)
  bootstrapped_scores <- bootstrapped_scores[!is.na(bootstrapped_scores)]
  
  # Sort bootstrap AUC scores in ascending order
  sorted_scores <- sort(bootstrapped_scores)
  
  # Calculate confidence interval bounds using percentile method
  # For 95% CI: use 2.5th percentile (lower) and 97.5th percentile (upper)
  # Formula: percentile index = (1 - alpha) / 2 or (1 + alpha) / 2
  ci_lower_idx <- ceiling((1 - alpha) / 2 * length(sorted_scores))
  ci_upper_idx <- ceiling((1 + alpha) / 2 * length(sorted_scores))
  
  ci_lower <- sorted_scores[ci_lower_idx]
  ci_upper <- sorted_scores[ci_upper_idx]
  
  return(list(lower = ci_lower, upper = ci_upper))
}

# ============================================================================
# K-FOLD CROSS-VALIDATION TRAINING LOOP
# ============================================================================
# Iterate through each fold in the cross-validation split
# Each iteration: 4 folds used for training, 1 fold used for validation

for (fold_num in 1:5) {
  cat("\n========================================\n")
  cat("Processing Fold", fold_num, "of 5\n")
  cat("========================================\n")
  
  # ====================================================================
  # SPLIT DATA INTO TRAINING AND VALIDATION SETS
  # ====================================================================
  # Get the indices for training and validation sets for this fold
  train_idx <- kfold_indices[[fold_num]]  # Training indices (80%)
  val_idx <- setdiff(1:nrow(X_train), train_idx)  # Validation indices (20%)
  
  # Extract training data for this fold
  X_fold_train <- X_train[train_idx, ]
  X_fold_val <- X_train[val_idx, ]
  
  # Extract corresponding labels
  y_fold_train <- y_train[train_idx]
  y_fold_val <- y_train[val_idx]
  
  cat("Training set size:", length(y_fold_train), "\n")
  cat("Validation set size:", length(y_fold_val), "\n")
  
  # ====================================================================
  # BUILD NEURAL NETWORK ARCHITECTURE
  # ====================================================================
  # Sequential model: stack layers linearly (input -> hidden -> output)
  
  model <- keras_model_sequential() %>%
    # Input layer + first hidden layer
    # units=64: number of neurons (learning capacity)
    # activation='relu': Rectified Linear Unit (standard for hidden layers)
    #   ReLU = max(0, x), introduces non-linearity
    # input_shape=c(5): expecting 5 input features (the 5 signature genes)
    # kernel_regularizer=regularizer_l2(0.001): L2 regularization penalty
    #   Adds penalty for large weights to prevent overfitting
    layer_dense(units = 64, 
                activation = 'relu', 
                input_shape = c(5),
                kernel_regularizer = regularizer_l2(0.001)) %>%
    
    # Dropout layer: randomly disable neurons during training
    # rate=0.5: disable 50% of neurons randomly in each training iteration
    # Purpose: prevents co-adaptation and overfitting
    #   Forces network to learn robust features
    layer_dropout(rate = 0.5) %>%
    
    # Output layer: single neuron for binary classification
    # units=1: single output
    # activation='sigmoid': sigmoid function outputs probability (0 to 1)
    #   sigmoid(x) = 1 / (1 + e^(-x))
    layer_dense(units = 1, activation = 'sigmoid')
  
  # ====================================================================
  # COMPILE MODEL
  # ====================================================================
  # Configure model for training
  model %>% compile(
    # optimizer=optimizer_adam(learning_rate=0.01): adaptive learning rate optimizer
    #   Adam: combines benefits of momentum and RMSprop
    #   learning_rate=0.01: controls step size for weight updates
    #     Higher = faster learning but risk of overshooting
    #     Lower = slower but more stable convergence
    optimizer = optimizer_adam(learning_rate = 0.01),
    
    # loss='binary_crossentropy': loss function for binary classification
    #   Measures difference between predicted and actual probabilities
    #   Formula: -(y*log(p) + (1-y)*log(1-p))
    loss = 'binary_crossentropy',
    
    # metrics=c('accuracy'): metric to monitor during training
    #   Percentage of correct predictions
    metrics = c('accuracy')
  )
  
  # ====================================================================
  # TRAIN MODEL
  # ====================================================================
  # Train the neural network on the training fold
  history <- model %>% fit(
    X_fold_train, y_fold_train,  # Training data
    epochs = 100,                 # Maximum number of iterations over entire dataset
    batch_size = 32,              # Process 32 samples per weight update
    # Smaller batch = noisier gradient but faster
    # Larger batch = cleaner gradient but slower
    validation_data = list(X_fold_val, y_fold_val),  # Validation data for early stopping
    callbacks = list(
      callback_early_stopping(
        monitor = 'val_loss',
        patience = 10,
        restore_best_weights = TRUE,
        verbose = 1
      )
    ),
    verbose = 1  # Print progress after each epoch (0=silent, 1=progress bar, 2=one line per epoch)
  )
  
  # ====================================================================
  # GENERATE PREDICTIONS
  # ====================================================================
  # Generate probability predictions on training set
  # model %>% predict() returns array of shape (n_samples, 1)
  # as.vector() converts to 1D vector
  y_fold_train_pred <- model %>% predict(X_fold_train, verbose = 0) %>% as.vector()
  
  # Generate probability predictions on validation set
  y_fold_val_pred <- model %>% predict(X_fold_val, verbose = 0) %>% as.vector()
  
  # ====================================================================
  # CALCULATE ROC CURVES
  # ====================================================================
  # ROC (Receiver Operating Characteristic) curve:
  #   Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
  #   at different classification thresholds
  #   Shows tradeoff between sensitivity and specificity
  
  # Calculate ROC curve for training set
  roc_train <- roc(y_fold_train, y_fold_train_pred, quiet = TRUE)
  fpr_train <- 1 - roc_train$specificities  # False Positive Rate
  tpr_train <- roc_train$sensitivities     # True Positive Rate
  
  # Calculate ROC curve for validation set
  roc_val <- roc(y_fold_val, y_fold_val_pred, quiet = TRUE)
  fpr_val <- 1 - roc_val$specificities
  tpr_val <- roc_val$sensitivities
  
  # ====================================================================
  # CALCULATE AUC SCORES
  # ====================================================================
  # AUC (Area Under the Curve): area under the ROC curve
  # Ranges from 0.5 (random classifier) to 1.0 (perfect classifier)
  # Interpretation:
  #   0.5 = no discrimination (random)
  #   0.6-0.7 = poor to acceptable
  #   0.7-0.8 = acceptable to good
  #   0.8-0.9 = good to excellent
  #   > 0.9 = excellent
  
  # Calculate AUC for training set
  train_auc <- as.numeric(roc_train$auc)
  
  # Calculate AUC for validation set
  val_auc <- as.numeric(roc_val$auc)
  
  # Store ROC curve data (FPR and TPR) for later plotting
  train_roc_list[[fold_num]] <- data.frame(fpr = fpr_train, tpr = tpr_train)
  val_roc_list[[fold_num]] <- data.frame(fpr = fpr_val, tpr = tpr_val)
  
  # Store AUC scores for comparison across folds
  train_auc_list[fold_num] <- train_auc
  val_auc_list[fold_num] <- val_auc
  
  # ====================================================================
  # CALCULATE 95% CONFIDENCE INTERVALS FOR AUC
  # ====================================================================
  # Use bootstrap resampling to estimate uncertainty in AUC scores
  # This shows the range of plausible AUC values
  
  # Calculate CI for training set AUC
  ci_train <- bootstrap_auc_ci(y_fold_train, y_fold_train_pred)
  
  # Calculate CI for validation set AUC
  ci_val <- bootstrap_auc_ci(y_fold_val, y_fold_val_pred)
  
  # Store confidence interval bounds for each fold
  train_ci_list[[fold_num]] <- ci_train
  val_ci_list[[fold_num]] <- ci_val
  
  # ====================================================================
  # PRINT FOLD RESULTS
  # ====================================================================
  # Print results for this fold for monitoring and reporting
  cat(sprintf("Fold %d Train AUC: %.4f, 95%% CI: %.4f - %.4f\n", 
              fold_num, train_auc, ci_train$lower, ci_train$upper))
  cat(sprintf("Fold %d Validation AUC: %.4f, 95%% CI: %.4f - %.4f\n", 
              fold_num, val_auc, ci_val$lower, ci_val$upper))
  
  # Clear model to free memory
  rm(model)
  keras_backend()$clear_session()
}

# ============================================================================
# PREPARE DATA FOR PLOTTING
# ============================================================================
# Create a data frame for training ROC curves
train_plot_data <- data.frame()
for (i in 1:5) {
  fold_data <- train_roc_list[[i]]
  fold_data$Fold <- paste("Fold", i)
  fold_data$AUC <- train_auc_list[i]
  fold_data$CI_lower <- train_ci_list[[i]]$lower
  fold_data$CI_upper <- train_ci_list[[i]]$upper
  train_plot_data <- rbind(train_plot_data, fold_data)
}

# Create a data frame for validation ROC curves
val_plot_data <- data.frame()
for (i in 1:5) {
  fold_data <- val_roc_list[[i]]
  fold_data$Fold <- paste("Fold", i)
  fold_data$AUC <- val_auc_list[i]
  fold_data$CI_lower <- val_ci_list[[i]]$lower
  fold_data$CI_upper <- val_ci_list[[i]]$upper
  val_plot_data <- rbind(val_plot_data, fold_data)
}

# ============================================================================
# PLOT TRAINING SET ROC CURVES
# ============================================================================
# Create a figure to display ROC curves for all training folds
png(file = "training_set_roc_curves.png", width = 800, height = 600)

# Create plot using ggplot2
train_plot <- ggplot(train_plot_data, aes(x = fpr, y = tpr, color = Fold)) +
  # Plot ROC curves for each fold
  geom_line(linewidth = 1) +
  
  # Add reference line for random classifier (AUC = 0.5)
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "black", linewidth = 1.2) +
  
  # Configure plot labels and title
  labs(
    title = "Training Set ROC Curves with AUC and 95% CI",
    x = "False Positive Rate",  # X-axis: 1 - Specificity
    y = "True Positive Rate"    # Y-axis: Sensitivity
  ) +
  
  # Set axis limits
  xlim(0, 1) +
  ylim(0, 1) +
  
  # Customize theme
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12),
    legend.position = "lower right",
    panel.grid.major = element_line(color = "gray90")
  )

# Add custom legend with AUC and CI information
for (i in 1:5) {
  ci_text <- sprintf("Fold %d AUC = %.4f (95%% CI: %.4f - %.4f)",
                     i, train_auc_list[i], train_ci_list[[i]]$lower, train_ci_list[[i]]$upper)
}

print(train_plot)
dev.off()

cat("Training ROC plot saved as 'training_set_roc_curves.png'\n")

# ============================================================================
# PLOT VALIDATION SET ROC CURVES
# ============================================================================
# Create a figure to display ROC curves for all validation folds
png(file = "test_set_roc_curves.png", width = 800, height = 600)

# Create plot using ggplot2
val_plot <- ggplot(val_plot_data, aes(x = fpr, y = tpr, color = Fold)) +
  # Plot ROC curves for each fold
  geom_line(linewidth = 1) +
  
  # Add reference line for random classifier
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "black", linewidth = 1.2) +
  
  # Configure plot
  labs(
    title = "Test Set ROC Curves with AUC and 95% CI",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  
  # Set axis limits
  xlim(0, 1) +
  ylim(0, 1) +
  
  # Customize theme
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12),
    legend.position = "lower right",
    panel.grid.major = element_line(color = "gray90")
  )

print(val_plot)
dev.off()

cat("Test ROC plot saved as 'test_set_roc_curves.png'\n")

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================
cat("\n========================================\n")
cat("CROSS-VALIDATION SUMMARY\n")
cat("========================================\n")

cat("\nTraining Set Results:\n")
cat("Mean AUC:", mean(train_auc_list), "\n")
cat("SD AUC:", sd(train_auc_list), "\n")
cat("Min AUC:", min(train_auc_list), "\n")
cat("Max AUC:", max(train_auc_list), "\n")

cat("\nValidation Set Results:\n")
cat("Mean AUC:", mean(val_auc_list), "\n")
cat("SD AUC:", sd(val_auc_list), "\n")
cat("Min AUC:", min(val_auc_list), "\n")
cat("Max AUC:", max(val_auc_list), "\n")