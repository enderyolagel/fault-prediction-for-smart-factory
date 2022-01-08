# 1. Packages and datasets -------------------------------------------------

## Check and install only missing packages 
team_packages <- c("tidyverse", "caret", "mlbench", "e1071", "MASS", "ISLR", 
                   "pROC", "MLmetrics", "gbm", "kernlab", "nnet", "klaR", 
                   "mice", "doParallel")
new_packages <- team_packages[!(team_packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) {install.packages(new_packages)}

## Load all packages at once
lapply(team_packages, require, character.only = TRUE)

## Train set
secom <- read_csv("data/secom_smote.csv", col_types = cols(class = "f"))
secom$class <- factor(secom$class, levels = c("X1", "X0"))
table(secom$class)

# Test set with fixed names and order of levels
test <- read_csv("data/secom_test.csv", col_types = cols(class = "f"))
test$class <- factor(test$class, labels = make.names(levels(test$class)))
test$class <- factor(test$class, levels = c("X1", "X0"))
table(test$class)


# 2. Outlier treatment and MV imputation on test set ------------------------

## 2.1 Outlier treatment
treatment_3s_na <- function(x) {
  
  mean <- mean(x, na.rm = T)  
  sd <- sd(x, na.rm = T)
  
  limits <- c(mean - 3 * sd, mean + 3 * sd)
  
  x <- ifelse(x < limits[1], NA, x)
  x <- ifelse(x > limits[2], NA, x)
  return(x)
}

for (i in 4:ncol(test)) {
  test[[i]] <- treatment_3s_na(test[[i]])
}


## 2.2 MICE Imputation 
pred_mat <- quickpred(test, mincor = 0.165, 
                      include = "class", 
                      exclude = c("ID", "timestamp"))

system.time({ # ~3.5 mins
  mice <- mice(test, m = 5, maxit = 5, method = "pmm", seed = 500, 
               predictorMatrix = pred_mat, print = FALSE) })

x <- data.frame(mice[["loggedEvents"]])
x %>% filter(meth == "constant") %>% count()
x %>% filter(meth == "collinear") %>% count()
rn(x)

test <- complete(mice, 5)


# 3. Prepare Models ---------------------------------------------------------

## 3.1 Control object
set.seed(5627)
seeds <- vector(mode = "list", length = 301) 
for(i in 1:300) seeds[[i]] <- sample.int(n = 1000, 301) 
seeds[[301]] <- sample.int(1000, 1) 

repeated_cv_folds <- createMultiFolds(secom$class, k = 10, times = 3)

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     seeds = seeds,
                     summaryFunction = twoClassSummary,
                     index = repeated_cv_folds,
                     classProbs = TRUE, 
                     savePredictions = TRUE,
                     verboseIter = TRUE,
                     allowParallel = TRUE)


## 3.2 Random forest 
cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # ~25 seconds
  model_forest <- train(class ~ ., 
                        data = secom, 
                        method = "rf", 
                        metric = "ROC",
                        trControl = ctrl, 
                        tuneGrid = expand.grid(.mtry = 2), 
                        mtree = 128) })
stopCluster(cl)
print(model_forest)

confusionMatrix(model_forest$pred$pred, model_forest$pred$obs, positive = "X1")
F1_Score(model_forest$pred$pred, model_forest$pred$obs, positive = "X1")

pred_model_forest <- predict(model_forest, secom)
confusionMatrix(pred_model_forest, secom$class, positive = "X1")
F1_Score(pred_model_forest, secom$class, positive = "X1")


## 3.3. Gradient Boosting Machines 
cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # ~40 seconds
gbm <- train(class ~ ., data = secom, 
             method = "gbm", 
             metric = "ROC",
             trControl = ctrl,
             verbose = 0,
             tuneGrid = expand.grid(n.trees = c(500), 
                                    interaction.depth = c(10), 
                                    shrinkage = c(0.1), 
                                    n.minobsinnode = c(10)
                                    )
             )
})
stopCluster(cl)
print(gbm)

confusionMatrix(gbm$pred$pred, gbm$pred$obs, positive = "X1")
F1_Score(gbm$pred$pred, gbm$pred$obs, positive = "X1")

pred_gbm <- predict(gbm, secom)
confusionMatrix(pred_gbm, secom$class, positive = "X1")
F1_Score(pred_gbm, secom$class, positive = "X1")


## 3.4 Support Vector Machines
cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # ~7 seconds
  svm <- train(class ~., data = secom, 
               method = "svmRadial", metric = "ROC", 
               preProc = c("center", "scale"), trControl = ctrl,
               tuneGrid = expand.grid(sigma = 0.04003563, C = 8)) })
stopCluster(cl)
print(svm)

## Confusion Matrix and F1 score = 0.9886 (it was 0.8962) 
confusionMatrix(svm$pred$pred, svm$pred$obs, positive = "X1")
F1_Score(svm$pred$pred, svm$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_svm <- predict(svm, secom)
confusionMatrix(pred_svm, secom$class, positive = "X1")
F1_Score(pred_svm, secom$class, positive = "X1")


# 4. Predict on test set -----------------------------------------------------

## 4.1 Random Forest sens = ~0.16, F1 = 0.1667
pred_rf <- predict(model_forest, test[-2])
table(pred_rf)
confusionMatrix(pred_rf, test$class, positive = "X1")
F1_Score(pred_rf, test$class, positive = "X1")

## 4.2. Gradient Boosting Machines sens = ~0.10, F1 = 0.0952
pred_gbm <- predict(gbm, test[-2])
table(pred_gbm)
confusionMatrix(pred_gbm, test$class, positive = "X1")
F1_Score(pred_gbm, test$class, positive = "X1")

## 4.3. Support Vector Machines, sens = ~0.05, F1 = 0.0625
pred_svm <- predict(svm, test[-2])
table(pred_svm)
confusionMatrix(pred_svm, test$class, positive = "X1")
F1_Score(pred_svm, test$class, positive = "X1")