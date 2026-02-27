# ============================================================================
# LOGISTIC REGRESSION MODEL WITH NOMOGRAM AND CALIBRATION ANALYSIS (R)
# ============================================================================
# This section builds a logistic regression model using the rms package,
# creates a nomogram for visual prediction, and performs calibration validation

library(rms)

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Define signature genes to be used as features in the model
sig_gene <- c("CGB5", "LEP", "PAPPA2", "LRRC1", "SLC20A1")

# Transpose the expression set (exprSet) so genes are columns and samples are rows
# This converts the data from gene-expression matrix format to analysis format
x <- t(exprSet)

# Select only the signature genes from the transposed expression data
x <- x[, sig_gene]

# Convert to data frame for compatibility with lrm() function
x <- as.data.frame(x)

# Add the outcome variable (group status/classification) to the data frame
# This is the variable we want to predict
x$Status <- talgroup$group

# ============================================================================
# INITIALIZE RMS PACKAGE CONFIGURATION
# ============================================================================
# Create a data distribution object required by the rms package
# This stores information about variable distributions needed for predictions
# and model validation
ddist <- datadist(x)

# Set the default datadist object globally for all subsequent rms functions
options(datadist = 'ddist')

# ============================================================================
# BUILD LOGISTIC REGRESSION MODEL
# ============================================================================
# Fit logistic regression model using all features to predict Status
# Formula: Status ~ . (predict Status from all other variables)
# x=TRUE: store the design matrix (needed for validation)
# y=TRUE: store the response vector (needed for validation)
model <- lrm(Status ~ ., data = x, x = TRUE, y = TRUE)

# ============================================================================
# CREATE NOMOGRAM FOR VISUAL RISK PREDICTION
# ============================================================================
# Generate a nomogram - a graphical tool for making individual predictions
# by reading values from aligned scales

# fun = function(x)1/(1+exp(-x)): applies inverse logit transformation
#   converts linear predictor to probability scale (0 to 1)
# funlabel="Risk of Event": label for the probability axis
# conf.int=F: exclude confidence intervals for cleaner visualization
# lp=F: exclude linear predictor scale from the plot
# fun.at: specify which probability values to display on the axis
nomogram <- nomogram(model, 
                     fun = function(x) 1 / (1 + exp(-x)),
                     funlabel = "Risk of Event",
                     conf.int = F,
                     lp = F,
                     fun.at = c(.001, .01, .05, seq(.1, .9, by = .2), .95, .99, .999))

# ============================================================================
# SAVE NOMOGRAM TO FILE
# ============================================================================
# Create a new empty graphics page
grid.newpage()

# Open PNG device to save the nomogram at high resolution
# width=800, height=700: dimensions in pixels for clear, printable image
png(file = "./nomogram.tif", width = 800, height = 700)

# Plot the nomogram
plot(nomogram)

# Close the PNG device and save the file
dev.off()

# ============================================================================
# CALIBRATION ANALYSIS - BOOTSTRAP METHOD
# ============================================================================
# Assess how well the model's predicted probabilities match actual outcomes
# This evaluates if predicted risk matches observed event rates

# Perform bootstrap calibration analysis:
# method="boot": use bootstrap resampling for internal validation
# B=200: perform 200 bootstrap iterations (samples with replacement)
cal <- rms::calibrate(model, method = "boot", B = 200)

# ============================================================================
# PLOT CALIBRATION CURVES
# ============================================================================
# Create calibration plot comparing predicted vs observed probabilities
plot(cal,
     xlim = c(0, 1),           # X-axis range: 0 to 1 (predicted probability)
     xlab = "Predicted Probability",
     ylab = "Observed Probability",
     legend = FALSE,           # Don't show default legend
     subtitles = FALSE)        # Don't show subtitles

# Add ideal calibration line (perfect agreement: predicted = observed)
# slope=1, intercept=0: perfect predictions
# lty=2: dashed line style
# col="black": black color
# lwd=2: line width of 2
abline(0, 1, col = "black", lty = 2, lwd = 2)

# Add apparent (unadjusted) calibration line
# Shows actual model performance without bias correction
# predy: predicted probabilities
# calibrated.orig: observed probabilities (original, no correction)
# col="red": red line
lines(cal[, c("predy", "calibrated.orig")],
      type = "l", lwd = 2, col = "red", pch = 16)

# Add bias-corrected calibration line
# Adjusts for overfitting using bootstrap-corrected estimates
# calibrated.corrected: bias-corrected observed probabilities
# col="blue": blue line
lines(cal[, c("predy", "calibrated.corrected")],
      type = "l", lwd = 2, col = "blue", pch = 16)

# Add legend to identify the three calibration lines
# Position: (0.65, 0.55) on the plot
legend(0.65, 0.55,
       c("Ideal", "Apparent", "Bias-corrected"),  # Legend labels
       lty = c(2, 1, 1),                          # Line types (dashed, solid, solid)
       lwd = c(2, 1, 1),                          # Line widths
       col = c("black", "red", "blue"),           # Line colors
       bty = "n")                                 # No box around legend

# ============================================================================
# VALIDATION ANALYSIS - BOOTSTRAP METHOD
# ============================================================================
# Perform comprehensive bootstrap validation to assess model performance
# This evaluates discrimination ability (how well model separates events from non-events)

# Parameters:
# method="boot": use bootstrap resampling
# B=1000: perform 1000 bootstrap iterations (more iterations for stability)
# dxy=T: calculate Somers' Dxy statistic (measure of discrimination)
v <- validate(model, method = "boot", B = 1000, dxy = T)

# ============================================================================
# EXTRACT AND CALCULATE C-INDEX (CONCORDANCE INDEX)
# ============================================================================
# Extract the bias-corrected Dxy statistic (discrimination index)
# Dxy ranges from -1 to 1 (0 = no discrimination, 1 = perfect discrimination)
Dxy <- v[rownames(v) == "Dxy", colnames(v) == "index.corrected"]

# Extract the original (unadjusted) Dxy statistic before bias correction
orig_Dxy <- v[rownames(v) == "Dxy", colnames(v) == "index.orig"]

# Convert Dxy to c-index (Concordance index)
# Formula: c-index = abs(Dxy)/2 + 0.5
# c-index ranges from 0.5 (no discrimination) to 1.0 (perfect discrimination)
# 0.5 = random guessing, 0.7-0.8 = acceptable, >0.8 = excellent
bias_corrected_c_index <- abs(Dxy) / 2 + 0.5

# Calculate original c-index (before bias correction for overfitting)
orig_c_index <- abs(orig_Dxy) / 2 + 0.5