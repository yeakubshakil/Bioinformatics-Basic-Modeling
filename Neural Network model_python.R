# ============================================================================
# NEURAL NETWORK MODEL WITH K-FOLD CROSS-VALIDATION AND AUC ANALYSIS (Python)
# ============================================================================
# This section builds a fully connected neural network (FCNN/MLP),
# performs k-fold cross-validation, and calculates AUC with confidence intervals
# to compare performance against the logistic regression model

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================
import numpy as np                              # Numerical computations
import matplotlib.pyplot as plt                 # Data visualization
from sklearn.metrics import roc_curve, auc, roc_auc_score  # ROC/AUC metrics
from sklearn.utils import resample              # Bootstrap resampling
from sklearn.model_selection import KFold       # K-fold cross-validation
from tensorflow.keras.models import Sequential  # Neural network model
from tensorflow.keras.layers import Dense, Dropout  # Layer types
from tensorflow.keras.optimizers import Adam    # Optimizer
from tensorflow.keras.regularizers import l2    # L2 regularization
from tensorflow.keras.callbacks import EarlyStopping  # Early stopping callback

# ============================================================================
# PREPARE DATA FOR TRAINING
# ============================================================================
# Extract features (X) from the dataframe
# iloc[:, 1:6] selects all rows and columns 1-5 (0-indexed, so 5 genes)
# values: convert to numpy array for compatibility with Keras
X_train = myseeddf.iloc[:, 1:6].values

# Extract target variable (y)
# factorize()[0]: converts categorical labels to binary codes (0 or 1)
# For example: ['Control', 'Disease'] becomes [0, 1]
y_train = myseeddf['Status'].factorize()[0]

# ============================================================================
# SETUP K-FOLD CROSS-VALIDATION
# ============================================================================
# Create a 5-fold cross-validation splitter
# n_splits=5: divide data into 5 equal folds
# shuffle=True: randomly shuffle data before splitting (improves robustness)
# random_state=42: set seed for reproducibility (same splits every run)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# INITIALIZE LISTS TO STORE RESULTS FROM EACH FOLD
# ============================================================================
train_roc_list = []      # Store FPR/TPR pairs for each training fold
val_roc_list = []        # Store FPR/TPR pairs for each validation fold
train_auc_list = []      # Store AUC scores for training sets
val_auc_list = []        # Store AUC scores for validation sets

train_ci_list = []       # Store 95% confidence intervals for training AUC
val_ci_list = []         # Store 95% confidence intervals for validation AUC

# ============================================================================
# DEFINE EARLY STOPPING CALLBACK
# ============================================================================
# Early stopping prevents overfitting by halting training when validation loss stops improving
# monitor='val_loss': monitor validation loss during training
# patience=10: wait 10 epochs without improvement before stopping
# restore_best_weights=True: revert to weights from best epoch after stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ============================================================================
# FUNCTION TO CALCULATE AUC CONFIDENCE INTERVALS
# ============================================================================
def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
  """
    Calculate confidence interval for AUC using bootstrap resampling.
    
    This method repeatedly samples from the data (with replacement) and
    calculates AUC for each sample to estimate the distribution of AUC values.
    
    Parameters:
    -----------
    y_true : array
        True binary class labels (0 or 1)
    y_pred : array
        Predicted probabilities from the model (0 to 1)
    n_bootstraps : int
        Number of bootstrap samples to generate (default: 1000)
        More samples = more accurate CI but slower computation
    alpha : float
        Confidence level (default: 0.95 for 95% confidence interval)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound) of the confidence interval
    """
# Initialize list to store AUC scores from each bootstrap sample
bootstrapped_scores = []

# Create random number generator with fixed seed for reproducibility
rng = np.random.RandomState(42)

# Generate n_bootstrap samples by random sampling with replacement
for _ in range(n_bootstraps):
  # Randomly select indices with replacement
  # This creates a bootstrap sample the same size as the original data
  indices = rng.randint(0, len(y_pred), len(y_pred))

# Skip if bootstrap sample doesn't contain both classes (0 and 1)
# AUC cannot be calculated with only one class
if len(np.unique(y_true[indices])) < 2:
  continue

# Calculate AUC for this bootstrap sample
score = roc_auc_score(y_true[indices], y_pred[indices])
bootstrapped_scores.append(score)

# Sort bootstrap AUC scores in ascending order
sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# Calculate confidence interval bounds using percentile method
# For 95% CI: use 2.5th percentile (lower) and 97.5th percentile (upper)
# Formula: percentile index = (1 - alpha) / 2 or (1 + alpha) / 2
ci_lower = sorted_scores[int((1 - alpha) / 2 * len(sorted_scores))]
ci_upper = sorted_scores[int((1 + alpha) / 2 * len(sorted_scores))]

return ci_lower, ci_upper

# ============================================================================
# K-FOLD CROSS-VALIDATION TRAINING LOOP
# ============================================================================
# Iterate through each fold in the cross-validation split
# Each iteration: 4 folds used for training, 1 fold used for validation
for train_index, val_index in kf.split(X_train):
  # ====================================================================
# SPLIT DATA INTO TRAINING AND VALIDATION SETS
# ====================================================================
# Extract training data for this fold
X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]

# Extract corresponding labels
y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

# ====================================================================
# BUILD NEURAL NETWORK ARCHITECTURE
# ====================================================================
# Sequential model: stack layers linearly (input -> hidden -> output)
model = Sequential([
  # Input layer + first hidden layer
  # Dense: fully connected layer
  # 64: number of neurons (learning capacity)
  # activation='relu': Rectified Linear Unit (standard for hidden layers)
  #   ReLU = max(0, x), introduces non-linearity
  # input_shape=(5,): expecting 5 input features (the 5 signature genes)
  # kernel_regularizer=l2(0.001): L2 regularization penalty
  #   Adds penalty for large weights to prevent overfitting
  Dense(64, activation='relu', input_shape=(5,), kernel_regularizer=l2(0.001)),
  
  # Dropout layer: randomly disable neurons during training
  # 0.5: disable 50% of neurons randomly in each training iteration
  # Purpose: prevents co-adaptation and overfitting
  #   Forces network to learn robust features
  Dropout(0.5),
  
  # Output layer: single neuron for binary classification
  # Dense(1): single output
  # activation='sigmoid': sigmoid function outputs probability (0 to 1)
  #   sigmoid(x) = 1 / (1 + e^(-x))
  Dense(1, activation='sigmoid')
])

# ====================================================================
# COMPILE MODEL
# ====================================================================
# Configure model for training
model.compile(
  # optimizer=Adam(learning_rate=0.01): adaptive learning rate optimizer
  #   Adam: combines benefits of momentum and RMSprop
  #   learning_rate=0.01: controls step size for weight updates
  #     Higher = faster learning but risk of overshooting
  #     Lower = slower but more stable convergence
  optimizer=Adam(learning_rate=0.01),
  
  # loss='binary_crossentropy': loss function for binary classification
  #   Measures difference between predicted and actual probabilities
  #   Formula: -(y*log(p) + (1-y)*log(1-p))
  loss='binary_crossentropy',
  
  # metrics=['accuracy']: metric to monitor during training
  #   Percentage of correct predictions
  metrics=['accuracy']
)

# ====================================================================
# TRAIN MODEL
# ====================================================================
# Train the neural network on the training fold
model.fit(
  X_fold_train, y_fold_train,  # Training data
  epochs=100,                   # Maximum number of iterations over entire dataset
  batch_size=32,                # Process 32 samples per weight update
  # Smaller batch = noisier gradient but faster
  # Larger batch = cleaner gradient but slower
  validation_data=(X_fold_val, y_fold_val),  # Validation data for early stopping
  callbacks=[early_stop],        # Apply early stopping callback
  verbose=2                      # Print progress after each epoch
)

# ====================================================================
# GENERATE PREDICTIONS
# ====================================================================
# Generate probability predictions on training set
# model.predict() returns array of shape (n_samples, 1)
# ravel() flattens 2D array to 1D for easier use
y_fold_train_pred = model.predict(X_fold_train).ravel()

# Generate probability predictions on validation set
y_fold_val_pred = model.predict(X_fold_val).ravel()

# ====================================================================
# CALCULATE ROC CURVES
# ====================================================================
# ROC (Receiver Operating Characteristic) curve:
#   Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
#   at different classification thresholds
#   Shows tradeoff between sensitivity and specificity

# Calculate ROC curve for training set
# Returns: false positive rates, true positive rates, thresholds
fpr_train, tpr_train, _ = roc_curve(y_fold_train, y_fold_train_pred)

# Calculate ROC curve for validation set
fpr_val, tpr_val, _ = roc_curve(y_fold_val, y_fold_val_pred)

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
train_auc = auc(fpr_train, tpr_train)

# Calculate AUC for validation set
val_auc = auc(fpr_val, tpr_val)

# Store ROC curve data (FPR and TPR) for later plotting
train_roc_list.append((fpr_train, tpr_train))
val_roc_list.append((fpr_val, tpr_val))

# Store AUC scores for comparison across folds
train_auc_list.append(train_auc)
val_auc_list.append(val_auc)

# ====================================================================
# CALCULATE 95% CONFIDENCE INTERVALS FOR AUC
# ====================================================================
# Use bootstrap resampling to estimate uncertainty in AUC scores
# This shows the range of plausible AUC values

# Calculate CI for training set AUC
ci_train_lower, ci_train_upper = bootstrap_auc_ci(y_fold_train, y_fold_train_pred)

# Calculate CI for validation set AUC
ci_val_lower, ci_val_upper = bootstrap_auc_ci(y_fold_val, y_fold_val_pred)

# Store confidence interval bounds for each fold
train_ci_list.append((ci_train_lower, ci_train_upper))
val_ci_list.append((ci_val_lower, ci_val_upper))

# ====================================================================
# PRINT FOLD RESULTS
# ====================================================================
# Print results for this fold for monitoring and reporting
print(f"Fold Train AUC: {train_auc:.4f}, 95% CI: {ci_train_lower:.4f} - {ci_train_upper:.4f}")
print(f"Fold Validation AUC: {val_auc:.4f}, 95% CI: {ci_val_lower:.4f} - {ci_val_upper:.4f}")

# ============================================================================
# PLOT TRAINING SET ROC CURVES
# ============================================================================
# Create a figure to display ROC curves for all training folds
plt.figure(figsize=(10, 6))

# Plot ROC curve for each fold
for i, (fpr, tpr) in enumerate(train_roc_list):
  # Get corresponding AUC and confidence interval for this fold
  ci_lower, ci_upper = train_ci_list[i]

# Plot ROC curve with label showing fold number, AUC, and 95% CI
# Fold number starts from 1 (not 0)
plt.plot(fpr, tpr, label=f"Fold {i+1} AUC = {train_auc_list[i]:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")

# Add diagonal reference line representing random classifier (AUC = 0.5)
# Random guessing would produce a straight diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Configure plot axes and labels
plt.xlabel('False Positive Rate')  # X-axis: 1 - Specificity
plt.ylabel('True Positive Rate')   # Y-axis: Sensitivity
plt.title('Training Set ROC Curves with AUC and 95% CI')
plt.legend(loc="lower right")      # Position legend
plt.grid(True)                      # Add gridlines for readability

# Save figure to file
plt.savefig("training_set_roc_curves.png")

# Display figure
plt.show()

# ============================================================================
# PLOT VALIDATION SET ROC CURVES
# ============================================================================
# Create a figure to display ROC curves for all validation folds
plt.figure(figsize=(10, 6))

# Plot ROC curve for each fold's validation set
for i, (fpr, tpr) in enumerate(val_roc_list):
  # Get corresponding AUC and confidence interval
  ci_lower, ci_upper = val_ci_list[i]

# Plot ROC curve
plt.plot(fpr, tpr, label=f"Fold {i+1} AUC = {val_auc_list[i]:.4f} (95% CI: {ci_lower:.4f} - {ci_val_upper:.4f})")

# Add reference line for random classifier
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Configure plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set ROC Curves with AUC and 95% CI')
plt.legend(loc="lower right")
plt.grid(True)

# Save and display figure
plt.savefig("Test_set_roc_curves.png")
plt.show()