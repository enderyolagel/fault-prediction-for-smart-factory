# 1. Packages and Dataset -------------------------------------------------------

## Check and install only missing packages
packages <- c("foreign", "dlookr", "VIM", "mice", "missForest", "randomForest",
              "MLmetrics", "DMwR", "ROSE",  "Boruta", "pROC", "doParallel", 
              "mlbench", "e1071", "MASS", "ISLR", "caret", "tidyverse")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) {install.packages(new_packages)}

## Load all packages at once
lapply(packages, require, character.only = TRUE)

## Import the data, fix the factor levels and its names of the target feature
rawdata <- read.spss("data/secom_mod.SAV", to.data.frame = TRUE)
rawdata$class <- factor(rawdata$class) 
rawdata$class <- factor(rawdata$class, 
                        labels = make.names(levels(rawdata$class)))
rawdata$class <- factor(rawdata$class, levels = c("X1", "X0"))
table(rawdata$class)

## Create train and test sets with stratified random sampling 
set.seed(3456)
train_index <- createDataPartition(rawdata$class, p = 0.8, list = F, times = 1)
secom_train <- rawdata[train_index, ]
secom_test <- rawdata[-train_index, ]

## Target class proportions are the same in both sets
secom_train %>% 
  group_by(class) %>% 
  summarize(n = n()) %>% 
  mutate(proportion_train = n / sum(n))

secom_test %>% 
  group_by(class) %>% 
  summarize(n = n()) %>% 
  mutate(proportion_test = n / sum(n))


# 2. Feature Reduction (NAs & Zero Variance) ---------------------------------

## Import train dataset and keep only columns with less than 55% NAs
secom_train <- secom_train[colSums(is.na(secom_train))/nrow(secom_train) < .55]
dim(secom_train) ## 569 columns

## Remove 116 constant features, 453 columns left.
variance <- apply(secom_train[, 4:ncol(secom_train)], 2, var, na.rm = TRUE) != 0
secom_train <- secom_train[, c(T, T, T, variance)]
dim(secom_train) 

## Copy secom_train into secom for simplicity and clean the environment
secom <- secom_train
rm(list = ls()[!(ls() %in% c("secom", "secom_test"))])


# 3. Outlier Treatment (3s with NA) ---------------------------------------

## Function to replace values beyond -/+ 3s with NAs without scaling
treatment_3s_na <- function(x) {
  mean <- mean(x, na.rm = T)  
  sd <- sd(x, na.rm = T)
  limits <- c(mean - 3 * sd, mean + 3 * sd)
  x <- ifelse(x < limits[1], NA, x)
  x <- ifelse(x > limits[2], NA, x)
  return(x)
}

## Iterate over all features
for (i in 4:ncol(secom)) {
  secom[[i]] <- treatment_3s_na(secom[[i]])
}

## Clean the environment
rm(list = ls()[!(ls() %in% c("secom", "secom_test", "treatment_3s_na"))])


# 4. Missing Value Imputation (MICE) -------------------------------------

## MICE imputation ~1.5 mins quickpred(), ~12.5 mins mice()
system.time({ #
pred_mat <- quickpred(secom, 
                      mincor = 0.15,
                      include = "class", 
                      exclude = c("ID", "timestamp"),
                      method = "spearman") })
mean(rowSums(pred_mat)) # 21 features on average

system.time({ mice <- mice(secom, 
                           m = 5, 
                           maxit = 5, 
                           method = "cart", 
                           seed = 500, 
                           predictorMatrix = pred_mat, 
                           print = FALSE) })

## To save time, you can load the object and inspect logged events
load("data/mice.RData")
events <- data.frame(mice[["loggedEvents"]])

## Collinear and constant features that are dropped from the imputation model
to_be_deleted <- events %>%
  filter(meth == "collinear" | meth == "constant") %>% dplyr::select(out) %>% 
  unlist() %>% as.character() %>% sort()

## Extract imputed dataset from the model
secom <- complete(mice, 5)

## Drop collinear and constant features on train set detected by the model
secom <- secom %>% dplyr::select(!all_of(to_be_deleted))

## Use pre-defined 427 variables to be used in imputation model for test set
test_vector <- names(secom)
secom_test <- secom_test %>% dplyr::select(all_of(test_vector))

## Clean the environment
rm(list = ls()[!(ls() %in% c("secom", "secom_test", 
                             "treatment_3s_na", "test_vector"))])


# 5. Feature Selection (BORUTA) ---------------------------------------------

## Register backend for parallel processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

## ~8.5 mins
set.seed(789)
boruta <- Boruta(class ~ ., 
                 data = secom[, -c(1, 3)], 
                 maxRuns = 250, 
                 doTrace = 2, 
                 getImp = getImpRfZ, 
                 num.threads = 3)
boruta$timeTaken 
stopCluster(cl)

## To save time, you can load the object and inspect results
load("data/boruta.RData")
print(boruta)

## 13 features are selected with 99% CI (2 features are tentative) 
(boruta_selected <- getSelectedAttributes(boruta, withTentative = F))
secom <- secom %>% dplyr::select(class, all_of(boruta_selected))

rm(list = ls()[!(ls() %in% c("secom", "secom_test", "treatment_3s_na",
                             "test_vector"))])


# 6. Subsampling (SMOTE) ---------------------------------------------------

## SMOTE subsampling
set.seed(9560)
secom <- SMOTE(class ~ ., secom, perc.over = 1300, perc.under = 120)

## New proportion of target feature: 47.3% X1 and 52.7% X0
secom %>% group_by(class) %>% summarize(n = n()) %>% mutate(n / sum(n))


# 7. Prepare the model (Random Forest) --------------------------------------

# 7.1 Control object

## List of seed values to be stored to allow parallel processing
set.seed(5627)
seeds <- vector(mode = "list", length = 21) 
for(i in 1:20) seeds[[i]] <- sample.int(n = 1000, 21) 
seeds[[21]] <- sample.int(1000, 1) 

## 20 bootstrap folds to be included as indices in model training
bootstrap_folds <- createResample(secom$class, times = 20)

## Control object for apple to apple comparisons
ctrl <- trainControl(method = "boot", 
                     seeds = seeds,
                     summaryFunction = twoClassSummary,
                     index = bootstrap_folds,
                     classProbs = TRUE, 
                     savePredictions = "final",
                     allowParallel = TRUE,
                     verboseIter = TRUE)

## Register backend for parallel processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)

## Random forest 
system.time({ # ~30 seconds
  model_forest <- train(class ~ ., 
                        data = secom, 
                        method = "rf", 
                        metric = "Sens",
                        trControl = ctrl, 
                        tuneGrid = expand.grid(.mtry = 2), 
                        mtree = 250) })
stopCluster(cl)

## Model Results
print(model_forest)
confusionMatrix(model_forest$pred$pred, model_forest$pred$obs, positive = "X1")
F1_Score(model_forest$pred$pred, model_forest$pred$obs, positive = "X1")

## Clean the environment
rm(list = ls()[!(ls() %in% c("secom", "secom_test", "test_vector",
                             "treatment_3s_na", "model_forest"))])


# 8. Pre-process of test set -------------------------------------------------

## 8.1 Outlier treatment
for (i in 4:ncol(secom_test)) {
  secom_test[[i]] <- treatment_3s_na(secom_test[[i]])
}

## 8.2 MICE Imputation ~1.5 mins for quickpred(), ~6 mins for mice()
system.time({ 
  pred_mat <- quickpred(secom_test, 
                        mincor = 0.15, 
                        method = "spearman",
                        include = "class", 
                        exclude = c("ID", "timestamp")) })
mean(rowSums(pred_mat)) 

system.time({ 
  mice_test <- mice(secom_test, 
                    m = 5, 
                    maxit = 5, 
                    method = "cart", 
                    seed = 500, 
                    predictorMatrix = pred_mat, 
                    print = F) })
load("data/mice_test.RData")

## Secom test is the input of imputation. Now extracted set is "test"
test <- complete(mice_test, 5)
rm(secom_test)


# 9. Prediction on test set -------------------------------------------------

# 9.1 Default cut-off needs to be adjusted
## Sens = ~0.21, Spec = 0.95, Accuracy = 0.90, AUC = 0.58, F1 = 0.2162
pred_rf <- predict(model_forest, test[-2])
confusionMatrix(pred_rf, test$class, positive = "X1")
F1_Score(pred_rf, test$class, positive = "X1")


# 9.2 Adjusted Cutoffs to lower costs of FNs
## Sens = 0.57, Spec = 0.86, Accuracy = 0.84, AUC = 0.72, F1 = 0.3142 

## Create ROC object
pred_rf <- predict(model_forest, test[-2], type = "prob")[, "X0"]
roc_rf <- roc(test$class, pred_rf, levels = rev(levels(test$class)))

## Define the best threshold 
(rf_threshold <- coords(roc_rf, x = "best", 
                        best.method = "closest.topleft", 
                        transpose = TRUE))

## Reclassify the predictions
new_rf_pred <- factor(ifelse(pred_rf > rf_threshold[1], "X0", "X1"),
                      levels = levels(test$class))

## Results of alternative cut-offs
(results <- confusionMatrix(new_rf_pred, test$class, positive = "X1"))
F1_Score(new_rf_pred, test$class, positive = "X1")

## Cost = (FN * 15) + (FP * 1)
(cost <- results$table[2] * 15 + results$table[3] * 1)

## Clean the environment
rm(list = ls()[!(ls() %in% c("test_vector", "treatment_3s_na", "model_forest"))])


# 10. Predictions on 500 Bootstraps ---------------------------------------

# 10.1 Import whole data, fix the factor levels and its names of the target
rawdata <- read.spss("data/secom_mod.SAV", to.data.frame = TRUE)
rawdata$class <- factor(rawdata$class) 
rawdata$class <- factor(rawdata$class, 
                        labels = make.names(levels(rawdata$class)))
rawdata$class <- factor(rawdata$class, levels = c("X1", "X0"))
table(rawdata$class)

## Outlier treatment
for (i in 4:ncol(rawdata)) {rawdata[[i]] <- treatment_3s_na(rawdata[[i]])}

## Same MICE imputation procedure
## Use pre-defined test vector to subset 427 variables to be used in imputation
rawdata <- rawdata %>% dplyr::select(all_of(test_vector))

## ~1.5 mins quickpred()
pred_mat <- quickpred(rawdata, 
                      mincor = 0.15, 
                      method = "spearman",
                      include = "class", 
                      exclude = c("ID", "timestamp"))
mean(rowSums(pred_mat)) # 20

## ~13.5 mins mice()
mice_rawdata <- mice(rawdata, 
                     m = 5, 
                     maxit = 5, 
                     method = "cart", 
                     seed = 500, 
                     predictorMatrix = pred_mat, 
                     print = F)

## To save time, you can load the object and inspect results
load("data/mice_rawdata.RData")
rawdata <- complete(mice_rawdata, 5)


# 10.2 Design the cost function
model_cost <- function(model, data, fn_cost = 15, fp_cost = 1) {
  
  ## Prediction values
  pred <- predict(object = model, 
                  newdata = data[-2], 
                  type = "prob")[, "X0"]
  
  ## ROC object with 95 CI
  roc <- roc(response = data$class, 
             predictor = pred,
             levels = rev(levels(data$class)))
  
  ## The best threshold 
  threshold <- coords(roc, x = "best", 
                      best.method = "closest.topleft", 
                      transpose = TRUE)
  
  ## Reclassify with the new cut-off
  new_pred <- factor(ifelse(pred > threshold[1], "X0", "X1"),
                     levels = levels(data$class))
  
  ## Get the results and calculate the cost
  results <- confusionMatrix(new_pred, data$class, positive = "X1")
  cost_value <- results$table[2] * fn_cost + results$table[3] * fp_cost
}


# 10.3 Predictions over 500 bootstrapped datasets

## List of indices for 500 bootstrapped samples  
boot_folds <- createResample(rawdata$class, 500)

## Store cost values as vector
cost <- vector(mode = "double", length = 500)

## Predict over 500 different datasets ~3 mins
system.time({ 
  for (i in 1:500) {
    boot_index <- boot_folds[[i]]
    boot_data <- rawdata[boot_index, ]
    cost_value <- model_cost(model_forest, boot_data)
    cost[i] <- cost_value
    cat("Cost of prediction", i, "is", cost_value, "€", "\n")
  }
})

## 95% Confidence Interval
quantile(cost, probs = c(0.025, 0.975))


# 10.4 Inspect the curve

## Histogram overlaid with kernel density curve
ggplot(as.data.frame(cost), aes(x = cost)) + 
  geom_histogram(aes(y = ..density..),
                 color = "black", 
                 fill = "lightblue",
                 binwidth = 20) +
  geom_density(alpha = 1) +
  labs(title = "Histogram of Prediction Costs",
       subtitle = "Overlaid with Kernel Density Curve",
       x = "Prediction Costs Across 500 Bootstraps",
       y = "Density") + 
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))


## Histogram of frequencies
ggplot(as.data.frame(cost)) + 
  geom_histogram(aes(x = cost), 
                 color = "black", 
                 fill = "lightblue", 
                 binwidth = 20) +
  labs(title = "Histogram of Prediction Costs",
       x = "Prediction Costs Across 500 Bootstraps",
       y = "Frequency") + 
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))
