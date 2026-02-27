# ============================================================================
# Machine-learning-based feature selection models (R)
#   - LASSO (glmnet)
#   - SVM-RFE (caret::rfe)
#   - Random Forest (randomForest + rfcv)
# ============================================================================
# NOTE:
#   - This script assumes:
#       exprSet: gene expression matrix (rows = genes, cols = samples)
#       talgroup: data.frame with sample IDs and group labels
#       WGCNA2_DEGs (or similar): data.frame containing SYMBOL column
#       WGCNA_DEGs exists (used only to rename col 1 below)
# ============================================================================

library(tidyverse)
library(glmnet)
library(VennDiagram)
library(sigFeature)
library(e1071)
library(caret)
library(kernlab)
library(randomForest)

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Ensure the gene symbol column is named consistently (only used if you export/merge later)
colnames(WGCNA_DEGs)[1] <- "SYMBOL"

# Define hub genes from WGCNA DEGs (expects a SYMBOL column)
hubgenes <- c(WGCNA2_DEGs$SYMBOL)

# Extract expression for hub genes from exprSet by matching gene symbols to rownames(exprSet)
hubgenes_expression <- exprSet[match(hubgenes, rownames(exprSet)), ]

# Convert to matrix and transpose so rows = samples and columns = genes
hubgenes_selected <- as.matrix(hubgenes_expression[, 1:ncol(hubgenes_expression)])
hubgenes_selected <- t(hubgenes_selected)

# Make sample IDs an explicit column for merging
hubgenes_selected <- data.frame(X = rownames(hubgenes_selected), hubgenes_selected)

# Merge group labels with expression data by sample ID column "X"
merged_data <- merge(talgroup, hubgenes_selected, by = "X")
rownames(merged_data) <- merged_data$X
merged_data$X <- NULL

# Final training dataframe:
#   - first column is expected to be group label (train$group)
#   - remaining columns are gene features
train <- merged_data

# ----------------------------------------------------------------------------
# Extract features (x2) and labels (y2)
# ----------------------------------------------------------------------------
# x2: numeric matrix of predictors (exclude group column)
x2 <- as.matrix(train[, -1])

# y2: outcome labels
y2 <- train$group


# ============================================================================
# 1) LASSO FEATURE SELECTION (glmnet)
# ============================================================================
set.seed(10)

# Fit LASSO logistic regression
# family="binomial" for binary classification
fit2 <- glmnet(x2, y2, family = "binomial", maxit = 1000)

# Plot coefficient paths vs lambda
plot(fit2, xvar = "lambda", label = TRUE)

# ---- FIX 1: typo in family argument in cv.glmnet ----
# Your code had family="binomia" which will error.
cvfit2 <- cv.glmnet(x2, y2, family = "binomial", maxit = 1000)

# Plot cross-validation curve
plot(cvfit2)

# Extract coefficients at best lambda (lambda.min)
coef2 <- coef(fit2, s = cvfit2$lambda.min)

# Identify non-zero coefficients (selected features)
index <- which(coef2 != 0)
actCoef2 <- coef2[index]
lassoGene2 <- rownames(coef2)[index]

# Combine gene + coefficient table
geneCoef2 <- cbind(Gene = lassoGene2, Coef = actCoef2)

# Remove intercept from selected genes/coefficients
lassoGene2 <- lassoGene2[-1]
actCoef2 <- actCoef2[-1]

# Save selected genes
write.csv(lassoGene2, "WGCNA_DEGs_feature_lasso.csv", row.names = FALSE)


# ============================================================================
# 2) SVM-RFE FEATURE SELECTION (caret::rfe with SVM radial)
# ============================================================================
# Convert labels to factor for classification
# NOTE: caret usually prefers factor labels for classification.
y2_factor <- as.factor(y2)

# ---- FIX 2: Your original code did factor -> numeric.
# That turns classes into numbers (e.g., 1/2) which can force regression-like behavior.
# Keep as factor for classification RFE.
# -------------------------------------------------------------------------

# Define RFE control:
# - caretFuncs: default feature selection functions
# - method="cv": cross-validation
# - number=10: 10-fold CV inside RFE
ctrl <- rfeControl(functions = caretFuncs, method = "cv", number = 10)

# Run RFE using SVM Radial Basis Function kernel
# sizes: candidate subset sizes to test
Profile <- rfe(
  x = x2,
  y = y2_factor,
  sizes = c(2, 4, 6, 8, seq(10, 40, by = 3)),
  rfeControl = ctrl,
  method = "svmRadial"
)

# Extract optimal variables (selected genes)
featureGenes <- Profile$optVariables

# Plot RMSE vs number of variables
# NOTE: For classification, caret often reports Accuracy/Kappa rather than RMSE.
# If RMSE appears, it usually means regression mode was triggered.
# Keeping y as factor helps ensure classification.
variable <- Profile$results$Variables
rmse <- Profile$results$RMSE

plot(variable, rmse,
     xlab = "Variables",
     ylab = "RMSE (10-fold Cross-Validation)",
     col = "darkgreen")
lines(variable, rmse, col = "darkgreen")

# Mark minimum RMSE point
wmin <- which.min(rmse)
wmin.x <- variable[wmin]
wmin.y <- rmse[wmin]
points(wmin.x, wmin.y, col = "blue", pch = 16)
text(wmin.x, wmin.y, paste0("N=", wmin.x), pos = 2, col = 3)


# ============================================================================
# 3) RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
# ---- FIX 3: randomForest() does NOT support splitrule='logrank' ----
# splitrule/logrank is for randomForestSRC or ranger survival setups.
# Here we use randomForest defaults.
# -------------------------------------------------------------------

set.seed(1234)

rf <- randomForest(
  group ~ .,
  data = train,
  ntree = 1000,
  importance = TRUE,
  proximity = TRUE
)

# Visualize error vs number of trees
plot(rf)

# Variable importance table
importance_df <- as.data.frame(rf$importance)

# Sort by MeanDecreaseGini (larger = more important)
importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]

# Save importance values
write.table(
  importance_df,
  file = "importance_class.txt",
  quote = FALSE,
  sep = "\t",
  row.names = TRUE,
  col.names = TRUE
)

# Plot top 30 variables by importance
varImpPlot(
  rf,
  n.var = min(30, nrow(rf$importance)),
  main = "Top30 - variable importance"
)

# ============================================================================
# OPTIONAL: RF CV ERROR CURVE USING rfcv()
# ============================================================================
# rfcv requires numeric y (0/1). Convert class label safely:
# Example: treat "PE" as 1 and others as 0.
set.seed(315)
train$group_bin <- ifelse(train$group == "PE", 1, 0)

# Compute cross-validated error for decreasing subsets of variables
train.cv <- rfcv(train[, -1, drop = FALSE], train$group_bin, cv.fold = 10, step = 1.5)

# View CV errors
train.cv$error.cv

# ============================================================================
# BARPLOT FOR TOP 30 IMPORTANT GENES (from saved importance file)
# ============================================================================
imp <- read.table("importance_class.txt", header = TRUE, row.names = 1, sep = "\t")
imp$gene <- rownames(imp)

# Keep top 30 most important by MeanDecreaseGini
imp <- imp[order(imp$MeanDecreaseGini, decreasing = TRUE), ]
imp <- head(imp, n = 30)

# Fix factor levels so plot keeps the sorted order
imp$gene <- factor(imp$gene, levels = rev(imp$gene))

# Plot horizontal bar chart
p <- ggplot(imp, aes(x = gene, y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x = NULL, y = "Mean Decrease Gini", title = "Top 30 genes by RF importance")

print(p)