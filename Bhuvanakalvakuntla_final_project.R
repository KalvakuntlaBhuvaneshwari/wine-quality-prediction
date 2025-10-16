# Wine Quality Prediction - Final Project Script
# Author: Bhuvaneshwari kalvakuntla 
# INFO6105 - Northeastern University


# Open the dataset folder and set it as working directory 
#run the following command to know if its set as working directory 
#  Plots include correlation heatmap, boxplots, feature importance, gbm boosting graph , clustering, and accuracy comparison, PCA.
# please execute every line step by step so that all plots will be dislayed.
getwd()
wine <- read.csv("winequality-red.csv", sep = ";")
head(wine)
str(wine)

# 1. Load required libraries (Install if needed)
packages <- c("tidyverse","dplyr" ,"caret", "gbm", "xgboost", "randomForest", "corrplot", "factoextra")
lapply(packages, require, character.only = TRUE)

# 2. Load the dataset
wine <- read.csv("winequality-red.csv", sep = ";")
wine$category <- ifelse(wine$quality >= 7, "High", "Low")
wine$category <- as.factor(wine$category)

# 3. Feature scaling
# Install if not already installed
#install.packages("dplyr")
# Then load it
library(dplyr)
# Scale all numeric predictors (excluding target and quality)
wine_scaled <- wine %>%
  select(-quality, -category) %>%   # Remove target columns
  scale() %>%
  as.data.frame()
# Add target column back (binary classification: High/Low)
wine_scaled$category <- wine$category  # Add the target back
wine_scaled$category_num <- ifelse(wine_scaled$category == "High", 1, 0)
# 4. Correlation heatmap
library(corrplot)# install if its not there 
cor_matrix <- cor(wine[, 1:11])
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, number.cex = 0.7,
         addCoef.col = "black",
         main = "Correlation Heatmap of Chemical Attributes")


#  5. Alcohol vs Quality Boxplot
ggplot(wine, aes(x = factor(quality), y = alcohol)) +
  geom_boxplot(fill = "green") +
  labs(title = "Alcohol Content by Wine Quality",
       x = "Quality Score",
       y = "Alcohol (%)") +
  theme_minimal()


# Distribution of 'category'
library(ggplot2) #load the library
ggplot(wine, aes(x = category)) +
       geom_bar(fill = "#66c2a5", width = 0.6) +
     labs(title = "Distribution of Wine Quality Categories",
                      x = "Wine Category",
                     y = "Count") +
      theme_minimal()

#  6. Train/Test Split
# install and load caret
#install.packages("caret")     # Only once
library(caret)
#  Train/Test split using caret
set.seed(42)
split <- createDataPartition(wine_scaled$category_num, p = 0.75, list = FALSE)
train <- wine_scaled[split, ]
test  <- wine_scaled[-split, ]


# 7. Model 1: GBM (Gradient Boosting)
library(gbm)
library(caret)
# Recode the target variable as 0 and 1
train$category_num <- ifelse(train$category == "High", 1, 0)
test$category_num <- ifelse(test$category == "High", 1, 0)
set.seed(123)

gbm_model <- gbm(category_num ~ . -category,  # exclude the original factor
                 data = train,
                 distribution = "bernoulli",
                 n.trees = 100,
                 interaction.depth = 3,
                 shrinkage = 0.1,
                 cv.folds = 5,
                 verbose = FALSE)

best_iter <- gbm.perf(gbm_model, method = "cv")
# Predict probabilities
gbm_probs <- predict(gbm_model, newdata = test, n.trees = best_iter, type = "response")
# Convert probabilities to class labels
gbm_pred <- ifelse(gbm_probs > 0.5, 1, 0)
# Convert test target back to factor for confusionMatrix
test$category_factor <- factor(test$category_num, labels = c("Low", "High"))
pred_factor <- factor(gbm_pred, labels = c("Low", "High"))

# Evaluate
library(caret)
confusionMatrix(pred_factor, test$category_factor)
summary(gbm_model)  # Displays variable importance
# 1. Add numeric label (0 = Low, 1 = High)
wine_scaled$category_num <- ifelse(wine_scaled$category == "High", 1, 0)
# 2. Create one-hot encoded data
library(caret)
set.seed(42)
split <- createDataPartition(wine_scaled$category_num, p = 0.75, list = FALSE)
train <- wine_scaled[split, ]
test <- wine_scaled[-split, ]
# 3. Remove target columns for feature matrix
train_x <- model.matrix(~ . - category - category_num, data = train)[, -1]
test_x <- model.matrix(~ . - category - category_num, data = test)[, -1]
train_y <- train$category_num
test_y <- test$category_num


#8. Model 2: XGBoost

#install.packages("xgboost")#if its not installed 
library(xgboost) #load the library 
# XGBoost Training
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)
set.seed(123)
xgb_model <- xgboost(data = dtrain,
                     objective = "binary:logistic",
                     eval_metric = "error",
                     nrounds = 100,
                     max_depth = 4,
                     eta = 0.1,
                     verbose = 0)
# Predict & Evaluate
xgb_probs <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_probs > 0.5, 1, 0)
library(caret) #load if needed 
xgb_factor <- factor(xgb_pred, labels = c("Low", "High"))
actual_factor <- factor(test_y, labels = c("Low", "High"))
xgb_eval <- confusionMatrix(xgb_factor, actual_factor)
print(xgb_eval)
#Feature Importance 
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance, top_n = 10, main = "XGBoost Feature Importance")

#9. Model 3: Random Forest
#install.packages("randomForest")  # Run if not installed
library(randomForest) # load the library 
# Train Random Forest model
set.seed(123)
rf_model <- randomForest(category ~ . -category_num, data = train, ntree = 100)
# Predict and evaluate
rf_pred <- predict(rf_model, test)
rf_eval <- confusionMatrix(rf_pred, test$category)
print(rf_eval)
# Feature importance plot
varImpPlot(rf_model, main = "Random Forest - Feature Importance")



# 10. Model Accuracy Comparison
# Replace with your real values from confusionMatrix$overall["Accuracy"]
accuracies <- c(
  GBM = 0.87,
  XGBoost = xgb_eval$overall["Accuracy"],
  RandomForest = rf_eval$overall["Accuracy"]
)

acc_df <- data.frame(Model = names(accuracies), Accuracy = as.numeric(accuracies))

ggplot(acc_df, aes(x = reorder(Model, -Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Accuracy)), vjust = -0.5, size = 5) +
  scale_y_continuous(limits = c(0.7, 1.0), expand = c(0, 0)) +
  labs(title = "Model Accuracy Comparison",
       x = "Model",
       y = "Accuracy") +
  theme_minimal() +
  theme(legend.position = "none")

# 11: KMeans Clustering + Elbow Method
# install.packages("factoextra")if needed 
library(ggplot2)
library(factoextra)
# Use only the scaled chemical attributes (numeric)
cluster_data <- wine_scaled[, 1:11]  # Exclude category/target columns
#  Elbow Method to find optimal number of clusters (k)
wss <- sapply(1:10, function(k){
  kmeans(cluster_data, centers = k, nstart = 10)$tot.withinss
})

# Plot WSS vs K
plot(1:10, wss, type = "b", pch = 19, col = "blue",
     xlab = "Number of Clusters (k)", ylab = "Total Within-Cluster SS",
     main = "Elbow Method for Determining Optimal Clusters")


# 12: KMeans with PCA Visualization
# run KMeans with chosen number of clusters (e.g., k = 4)
set.seed(42)
kmeans_model <- kmeans(cluster_data, centers = 4, nstart = 25)
# Visualize using PCA
fviz_cluster(kmeans_model, data = cluster_data,
             palette = "jco", ellipse.type = "norm",
             main = "Wine Clustering using KMeans + PCA",
             geom = "point", show.clust.cent = TRUE)

