# 1. Packages and dataset ----------------------------------------------------

## Check and install only missing packages
team_packages <- c("tidyverse", "caret", "mlbench", "e1071", "MASS", "ISLR", 
                   "pROC", "MLmetrics", "gbm", "kernlab", "nnet", "klaR",
                   "doParallel")
new_packages <- team_packages[!(team_packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) {install.packages(new_packages)}

## Load all packages at once
lapply(team_packages, require, character.only = TRUE)

## Import train data, convert 'class' into factor and drop IDs
secom <- read_csv("data/secom_smote.csv", col_types = cols(class = "f"))
secom$class <- factor(secom$class, levels = c("X1", "X0"))
table(secom$class)


# 2. Control object ----------------------------------------------------------

## Create a seed list for each fold for reproducibility in parallel mode
set.seed(5627)
seeds <- vector(mode = "list", length = 21) 
for(i in 1:20) seeds[[i]] <- sample.int(n = 1000, 21) 
seeds[[21]] <- sample.int(1000, 1) 

## Exact same repeated CV folds for each model for a fair comparison!
# repeated_cv_folds <- createMultiFolds(secom$class, k = 10, times = 3)
bootstrap_folds <- createResample(secom$class, times = 20)

## Control parameters. 
ctrl <- trainControl(method = "boot",
                     #method = "repeatedcv", 
                     #number = 10, 
                     #repeats = 3,
                     seeds = seeds,
                     summaryFunction = twoClassSummary,
                     #index = repeated_cv_folds,
                     index = bootstrap_folds,
                     classProbs = TRUE, 
                     savePredictions = TRUE,
                     verboseIter = TRUE,
                     allowParallel = TRUE)

# 3. Random forest ----------------------------------------------------------

## Run model in parallel  ~4 mins
cl <- makeCluster(detectCores())
registerDoParallel(cl)

## Fit a model with a deeper tuning grid (default = 3)
system.time({ 
model_forest <- train(class ~ ., 
                      data = secom, 
                      method = "rf", 
                      trControl = ctrl, 
                      metric = "ROC", 
                      tuneLength = 10) })

print(model_forest)
plot(model_forest)

## Manually tuning the model to find the best ntree
# Create tunegrid. Best ntree equals to 125
tunegrid <- expand.grid(.mtry = c(sqrt(ncol(secom))))
modellist <- list()

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Train with different ntree parameters
for (ntree in c(125, 250, 500, 1000, 1250)){
  fit <- train(class~.,
               data = secom,
               method = 'rf',
               metric = "ROC",
               tuneGrid = tunegrid,
               trControl = ctrl,
               ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
stopCluster(cl)

## Compare results
results <- resamples(modellist)
summary(results)
dotplot(results)

## Fit a random forest with the best mtry/tuneGrid = 2, mtree = 128
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
print(model_forest)

## Create confusion matrix and F1
confusionMatrix(model_forest$pred$pred, model_forest$pred$obs, positive = "X1")
F1_Score(model_forest$pred$pred, model_forest$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_model_forest <- predict(model_forest, secom)
confusionMatrix(pred_model_forest, secom$class, positive = "X1")
F1_Score(pred_model_forest, secom$class, positive = "X1")


# 4. Gradient Boosting Machines ----------------------------------------------

## Fit the model to find the best parameters
modelLookup("gbm")
gbm <- train(class ~ ., data = secom, method = "gbm", tuneLength = 10,
             trControl = ctrl, metric = "ROC", verbose = 0)

print(gbm)
plot(gbm)


## Fit the model with best hyperparameters
## Ntree = 500, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10

cl <- makeCluster(detectCores())
registerDoParallel(cl)
grid <- expand.grid(n.trees = c(500), interaction.depth = c(10), 
                    shrinkage = c(0.1), n.minobsinnode = c(10))

gbm <- train(class ~ ., data = secom, method = "gbm", 
             trControl = ctrl, metric = "ROC", verbose = 0,
             tuneGrid = grid)
print(gbm)

## Confusion matrix and F1 
confusionMatrix(gbm$pred$pred, gbm$pred$obs, positive = "X1")
F1_Score(gbm$pred$pred, gbm$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_gbm <- predict(gbm, secom)
confusionMatrix(pred_gbm, secom$class, positive = "X1")
F1_Score(pred_gbm, secom$class, positive = "X1")


# 5. Support Vector Machines -------------------------------------------------

modelLookup("svmRadial")

cl <- makeCluster(detectCores())
registerDoParallel(cl)
system.time({ # ~3 mins
svm <- train(class ~., data = secom, 
             method = "svmRadial", metric = "ROC", 
             preProc = c("center", "scale"), 
             trControl = ctrl, tuneLength = 20) })
print(svm) 
plot(svm)

## With best parameters: sigma = 0.03963949, best C = 8
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


# 6. k-Nearest Neighbors ---------------------------------------------------

modelLookup("knn")

cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # ~19 seconds
knn <- train(class ~., 
             data = secom, 
             method = "knn", 
             metric = "ROC", 
             preProc = c("center", "scale"),
             tuneLength = 20, 
             trControl = ctrl) })
knn
plot(knn)

## Fit the model with best hyperparameter (k = 9)
knn <- train(class ~., 
             data = secom, 
             method = "knn", 
             metric = "ROC", 
             preProc = c("center", "scale"),
             tuneGrid = expand.grid(k = 9), 
             trControl = ctrl)
knn
stopCluster(cl)

## Confusion matrix and F1 Score
confusionMatrix(knn$pred$pred, knn$pred$obs, positive = "X1")
F1_Score(knn$pred$pred, knn$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_knn <- predict(knn, secom)
confusionMatrix(pred_knn, secom$class, positive = "X1")
F1_Score(pred_knn, secom$class, positive = "X1")


# 7. Neural Network -------------------------------------------------------

modelLookup("nnet")

## size : hidden units ; decay : weight decay
nnet_grid <- expand.grid(.size = 1:10, .decay = c(0, .1, .5, 1, 2)) 
max_size <- max(nnet_grid$.size)
num_wts <- 1 * (max_size * (length(secom) + 1) + max_size + 1)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # ~5.5 mins
  nnet <- train(class ~.,
                data = secom, 
                method = "nnet", 
                metric = "ROC", 
                preProc = c("center", "scale", "spatialSign"), 
                tuneGrid = nnet_grid, # size = 10, decay = 0.1
                trace = FALSE, 
                maxit = 2000, 
                MaxNWts = num_wts,
                trControl = ctrl) })
stopCluster(cl)
nnet
plot(nnet)


## Fitting model with tuned hyperparameters
nnet_grid <- expand.grid(.size = 10, .decay = .1) 
max_size <- max(nnet_grid$.size)
num_wts <- 1 * (max_size * (length(secom) + 1) + max_size + 1)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
system.time({ # ~9 seconds 
  nnet <- train(class ~., 
                data = secom, 
                method = "nnet", 
                metric = "ROC", 
                preProc = c("center", "scale", "spatialSign"), 
                tuneGrid = nnet_grid, # size = 10, decay = 0.1
                trace = FALSE, 
                maxit = 2000, 
                MaxNWts = num_wts, 
                trControl = ctrl) })
stopCluster(cl)
print(nnet)
save(nnet, file = "nnet.RData")
load("nnet.RData")

## Confusion Matrix and F1 score
confusionMatrix(nnet$pred$pred, nnet$pred$obs, positive = "X1")
F1_Score(nnet$pred$pred, nnet$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_nnet <- predict(nnet, secom)
confusionMatrix(pred_nnet, secom$class, positive = "X1")
F1_Score(pred_nnet, secom$class, positive = "X1")


# 8. Naive Bayes -----------------------------------------------------------

modelLookup("nb")

# Tune hyperparameters 
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE), # KDE vs gaussian density estimates
  fL = 0:5, # allows us to incorporate the Laplace smoother
  adjust = seq(0, 10, by = 1) # to adjust the bandwidth of the kernel density 
  # (larger the number, more flexible density estimate)
)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # From 100 runs in 10 mins to 300 runs in 16 mins
  naive_bayes <- train(class ~ .,
                       data = secom,
                       method = "nb",
                       trControl = ctrl,
                       tuneGrid = search_grid,
                       preProc = c("BoxCox", "center", "scale", "pca"),
                       tuneLength = 10,
                       metric = "ROC") })
stopCluster(cl)
naive_bayes
save(naive_bayes, file = "naive_bayes.RData")
load("naive_bayes.RData")

## Top 5 model 
naive_bayes$results %>% top_n(5, wt = ROC) %>% arrange(desc(ROC))
plot(naive_bayes)
naive_bayes$finalModel

# Fit model with best hyperparameters (fL = 0, usekernel = T, adjust = 1)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
system.time({ # ~ 15 seconds
  naive_bayes <- train(class ~ .,
                       data = secom,
                       method = "nb",
                       trControl = ctrl,
                       tuneGrid = expand.grid(fL = 0, usekernel = T, adjust = 1),
                       preProc = c("BoxCox", "center", "scale", "pca"),
                       metric = "ROC") })
stopCluster(cl)  

## Confusion Matrix and F1 score
confusionMatrix(naive_bayes$pred$pred, naive_bayes$pred$obs, positive = "X1")
F1_Score(naive_bayes$pred$pred, naive_bayes$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_naive_bayes <- predict(naive_bayes, secom)
confusionMatrix(pred_naive_bayes, secom$class, positive = "X1")
F1_Score(pred_naive_bayes, secom$class, positive = "X1")


# 9. GLM -------------------------------------------------------------------

cl <- makeCluster(detectCores())
registerDoParallel(cl)

system.time({ # ~ 8 seconds
  glm <- train(class ~ ., data = secom, method = "glm", 
               metric = "ROC", trControl = ctrl) })
stopCluster(cl)

## Confusion Matrix and F1 score
confusionMatrix(glm$pred$pred, glm$pred$obs, positive = "X1")
F1_Score(glm$pred$pred, glm$pred$obs, positive = "X1")

## Predicted classes without class probabilities
pred_glm <- predict(glm, secom)
confusionMatrix(pred_glm, secom$class, positive = "X1")
F1_Score(pred_glm, secom$class, positive = "X1")


# 10. Model Evaluation -------------------------------------------------------

models <- list(random_forest = model_forest, 
               GBM = gbm, 
               SVM = svm, 
               kNN = knn, 
               neural_network = nnet, 
               naive_bayes = naive_bayes, 
               GLM = glm)
save(models, file = "models.RData")
load("models.RData")
                 
results <- resamples(models)        

# The Three Musketeers: RF, GBM, SVM
summary(results)
dotplot(results, metric = "ROC")
bwplot(results)


