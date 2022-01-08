# Packages and Dataset -------------------------------------------------------

## Check and install only missing packages
packages <- c("tidyverse", "caret", "foreign", "dlookr", "VIM", "simputation", 
              "naniar", "mice", "missForest", "doParallel", "Boruta", "corrr",
              "rFerns", "randomForest", "glmnet")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) {install.packages(new_packages)}

## Load all packages at once
lapply(packages, require, character.only = TRUE)

## Import the data, convert the target feature into factor
rawdata <- read.spss("secom_mod.SAV", to.data.frame = TRUE)
rawdata$class <- factor(rawdata$class)


# C1. Data Split ------------------------------------------------------------

## Create a balanced split that preserves the overall class distribution
set.seed(3456)
train_index <- createDataPartition(rawdata$class, p = 0.8, list = F, times = 1)

## Splitting the dataset into train and test sets 
secom_train <- rawdata[train_index, ]
secom_test <- rawdata[-train_index, ]

## Both sets have the same target class proportion
secom_train %>% 
  group_by(class) %>% summarize(n = n()) %>% mutate(prop_train = n / sum(n))

secom_test %>% 
  group_by(class) %>% summarize(n = n()) %>% mutate(prop_test = n / sum(n))

## Write train and tests sets into csv files
write_csv(secom_train, "secom_train.csv")
write_csv(secom_test, "secom_test.csv")

## Clean the environment
rm(list = ls())


# C2. Feature Reduction (NAs, Variance) ------------------------------------

## Import train dataset and keep only columns with less than 55% NAs
secom <- read_csv("secom_train.csv")
secom <- secom[colSums(is.na(secom))/nrow(secom) < .55]
dim(secom) ## 569 columns

## Remove constant values
variance <- apply(secom[, 4:ncol(secom)], 2, var, na.rm = TRUE) != 0
secom <- secom[, c(T, T, T, variance)]

## 116 columns removed, 453 columns left.
dim(secom) 
table(variance)["FALSE"]

## Remove 12 columns that have near zero variance
nzv_indices <- nearZeroVar(secom)
secom_without_nzv <- secom[, -nzv_indices] ## 441 columns left.

## Manual inspection on 12 columns deleted
secom_with_nzv <- secom[, nzv_indices]
summary(secom_with_nzv)

## Write the data into csv files
write_csv(secom_without_nzv, "secom_C2a.csv")

## Remove all environment variables to avoid confusions
rm(list = ls())


# C3. Outlier Treatment (3s with NA) ---------------------------------------

## Import the data
secom <- read_csv("secom_C2a.csv")

## Inspection of some variables with outliers before treatment
summary(secom$feature060)
summary(secom$feature336)
summary(secom$feature252)

## treatment_3s function
treatment_3s_na <- function(x) {
  
  ## Calculate the mean and the SD of variable x
  mean <- mean(x, na.rm = T)  
  sd <- sd(x, na.rm = T)
  
  ## Setting -/+ 3s limits 
  limits <- c(mean - 3 * sd, mean + 3 * sd)
  
  ## Replace outliers below and above -/+3s limits with NAs 
  x <- ifelse(x < limits[1], NA, x)
  x <- ifelse(x > limits[2], NA, x)
  return(x)
}

## Iteration over the columns of df
for (i in 4:ncol(secom)) {
  secom[[i]] <- treatment_3s_na(secom[[i]])
}

## Inspection of the same variables after the treatment
summary(secom$feature060)
summary(secom$feature336)
summary(secom$feature252)

## Backup the data and clean the environment
write_csv(secom, "secom_C3c.csv")
rm(list = ls())


# C4. Missing Value Analysis -------------------------------------------------
secom <- read_csv("secom_C3c.csv")
secom$class <- as.factor(secom$class)

## No completely missing rows
index <- rowSums(is.na(secom)) == ncol(secom)
table(index)["FALSE"]
names(secom)[index]

## In which combinations of variables with NA > .10 are missing?
test_10 <- secom[colSums(is.na(secom))/nrow(secom) > .10] 
test_10 %>% aggr(combined = TRUE, numbers = TRUE, 
                 sortVars = TRUE, ylab = "Missingness Pattern")

## How often features 073,074,346,347,113,248,386,520 are missing together?
test_20 <- secom[colSums(is.na(secom))/nrow(secom) > .20]
test_20 %>% aggr(combined = TRUE, numbers = TRUE, 
                 sortVars = TRUE, ylab = "Missingness Pattern")


## C4a. Missing Value Imputations (kNN) --------------------------------------

## Get variable names sorted increasingly by # of NAs to calculate each 
## distance on as much observed data and as little imputed data as possible
vars_by_NAs <- secom %>% 
  is.na() %>% colSums() %>% sort(decreasing = FALSE) %>% names()
head(vars_by_NAs, n = 20)

## Sort and feed secom variables into kNN imputation. Runtime ~7 mins.
system.time({secom_knn_imp <- secom %>% 
  select(all_of(vars_by_NAs)) %>% # to sort variables according to missingness
  kNN(k = 21) }) # k = sqrt of 441
save(secom_knn_imp, file = "secom_knn_imp.RData")

## Load the object to skip runtime
load("secom_knn_imp.RData")

## Evaluating imputations of features with more than 20% NAs 
secom_knn_imp %>% select(feature248, feature520, feature520_imp) %>% 
  marginplot(delimiter = "imp") ## Use 658x630 ratio to export plots

secom_knn_imp %>% select(feature346, feature386, feature386_imp) %>% 
  marginplot(delimiter = "imp")

secom_knn_imp %>% select(feature248, feature073, feature073_imp) %>% 
  marginplot(delimiter = "imp")

secom_knn_imp %>% select(feature073, feature113, feature113_imp) %>% 
  marginplot(delimiter = "imp")

## Evaluating imputations of features with 4-5% NAs 
na_prop <- colSums(is.na(secom))/nrow(secom)
names(secom[na_prop > 0.04 & na_prop < 0.05])

secom_knn_imp %>% select(feature039, feature091, feature091_imp) %>% 
  marginplot(delimiter = "imp")

secom_knn_imp %>% select(feature225, feature275, feature275_imp) %>% 
  marginplot(delimiter = "imp")

## Clean the environment except original and knn imputed datasets
rm(list = ls()[!(ls() %in% c("secom","secom_knn_imp"))])
write_csv(secom_knn_imp, "secom_knn.csv")


## C4b. Missing Value Imputations (MICE) -------------------------------------
secom <- read_csv("secom_C3c.csv")
secom$class <- as.factor(secom$class)

# 1. Imputation of the missing data m times

## 441 x 441 Predictor matrix to prevent "blind imputation" and include  
## target feature as covariate in each imputation models.  
## A value of 1 indicates the column was used to impute the row. 
pred_mat <- quickpred(secom, mincor = 0.165, include = "class", 
                      exclude = c("ID", "timestamp"))

# 15-25 predictors on average are used as suggested by Buuren. In our case 17.
mean(rowSums(pred_mat))
rownames(pred_mat)[rowSums(pred_mat) == 0]

## MICE imputation (~3 mins runtime)
load("mice.RData")

system.time({ mice <- mice(secom, m = 5, maxit = 5, method = "pmm", seed = 500, 
                           predictorMatrix = pred_mat, print = FALSE) })
save(mice, file = "mice.RData")
load("mice.RData")

## Inspect problems identified and corrective actions taken by the model
## Mice detects multicollinearity and solves it by removing them from the model
events <- data.frame(mice[["loggedEvents"]])

## As vector, collinear features that are dropped from the imputation model
(collinear <- events %>%
    filter(meth == "collinear") %>% dplyr::select(out) %>% 
    unlist() %>% as.character() %>% sort())

## Exctracting imputations from the model
secom_mice_imp <- complete(mice, 5)
save(secom_mice_imp, file = "secom_mice_imp.RData")
load("secom_mice_imp.RData")

## Check if collinear features are not imputed by the model
not_imputed <- colnames(secom_mice_imp)[colSums(is.na(secom_mice_imp)) > 0]
identical(collinear, sort(not_imputed))

## The correspondence of observed and imputed data across 5 imputations
stripplot(mice, feature074 ~ feature347 | .imp, pch = 20, cex = 0.75)
stripplot(mice, feature346 ~ feature386 | .imp, pch = 20, cex = 0.75)
stripplot(mice, feature520 ~ feature347 | .imp, pch = 20, cex = 0.75)
stripplot(mice, feature073 ~ feature113 | .imp, pch = 20, cex = 0.75)
stripplot(mice, feature039 ~ feature091 | .imp, pch = 20, cex = 0.75)
stripplot(mice, feature225 ~ feature275 | .imp, pch = 20, cex = 0.75)


# 2. Analysis of the 5 times imputed datasets

## Skipping stepwise variable selection as discussed "Model Optimism"
## Instead using heavily imputed features as predictors
test_20 <- secom[colSums(is.na(secom))/nrow(secom) > .20]
(predictors <- names(test_20)[!names(test_20) %in% collinear])

## Fit 5 glm models across 5 imputed datasets
glm_mice <- with(mice, 
                glm(class ~ feature073 + feature074 + feature113 + feature346 + 
                    feature347 + feature386 + feature520, family = "binomial"))


# 3.Pooling of the parameters across 5 analyses

## Pool regression results
glm_pooled <- pool(glm_mice)

## However accounting for imputation uncertainty with 95% confidence, 
## we are never sure of these effects as the lower bounds are negative! 
glm_summary <- as.data.frame(
  summary(glm_pooled, conf.int = TRUE, conf.level = 0.95))

## Clean the environment
rm(list = ls()[!(ls() %in% c("secom", "secom_knn_imp", "secom_mice_imp", 
                             "collinear", "predictors"))])


## C4c. Missing Value Imputations (Random Forest) ----------------------------

## Import the data in data.frame class for the package 
secom <- as.data.frame(secom)

## Parallel Processing to reduce computation from 4 hours to 61 mins
## Used n-2 cores, it can be safely used as n-1 on both MAC OS and Windows
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl) # Register the parallel backend to run with foreach

## Random Forest Imputation (61 mins runtime with NRMSE 0.1835756)
system.time({ imp <- missForest(secom[, -c(1:3)], ntree = 20, mtry = 128,
                                parallelize = "variables") })

## Stop the registered cluster and backup the imputation object
stopCluster(cl) 
save(imp, file = "secom_forest_imp.RData")

## Retrive kNN "imp" object and get imputation error
load("secom_forest_imp.RData")
imp$OOBerror

## Extract imputed data and combine with first 3 variables 
secom_forest_imp <- cbind(secom[, c(1:3)], imp$ximp)
sum(is.na(secom_forest_imp)) # No missing values

# Clean the environment
rm(list = ls()[!(ls() %in% c("secom","collinear", "predictors"))])


## C4d. Comparison and Evaluation of Imputation Models ----------------------

## Datasets imputed by each model
load("secom_mice_imp.RData")
load("secom_knn_imp.RData")
load("secom_forest_imp.RData")

## Extract combine RF dataset and omit shadow matrix from knn imputed dataset   
secom_forest_imp <- cbind(secom[, c(1:3)], imp$ximp)
secom_knn_imp <- secom_knn_imp[, 1:441]

secom_knn_imp$class <- as.factor(secom_knn_imp$class)
secom_mice_imp$class <- as.factor(secom_mice_imp$class) 

## Define outcome and predictors and create fully parameterized GLM formula
outcome <- "class"

# Pick heavily imputed features that have NAs more than 20% then subsetting 
test_20 <- secom[colSums(is.na(secom))/nrow(secom) > .20]
(predictors <- names(test_20)[!names(test_20) %in% collinear])

(f <- as.formula(
  paste(outcome, paste(predictors, collapse = " + "), sep = " ~ ")))


## Creating a giant table by combining all datasets rowwise and 
## putting an primary id, that takes values from the names of datasets
bound_models <- dplyr::bind_rows(knn = secom_knn_imp, 
                                 mice = secom_mice_imp,
                                 forest = secom_forest_imp, 
                                 .id = "imp_model")

## Fit GLM models to each imputed datasets, get model statistics (Many Models)
## List-columns to store arbitrary data structures (incl models) in a df. 
model_summary <- bound_models %>%
  group_by(imp_model) %>% # rest of the operations will be separated by imp_model
  nest() %>% # Nesting dataframes into a list-column 
  mutate(mod = map(data, ~glm(formula = f, family = "binomial", data = .)),
         res = map(mod, residuals),
         pred = map(mod, ~predict(type = "response", object = .)),
         tidy = map(mod, broom::tidy)) # Tidy components of models (coef, CI)
print(model_summary)

## Exploring coefficients of multiple models
stats <- model_summary %>% dplyr::select(imp_model, tidy) %>% unnest()

## As expected p-values of kNN & RF are more significant than MICE's
sorted_pvalues <- stats %>% dplyr::arrange(desc(p.value))

## Explore residuals of multiple models. Not much difference.
model_summary %>% dplyr::select(imp_model, res) %>% unnest() %>%
  ggplot(aes(x = res, fill = imp_model)) + 
  geom_histogram(position = "dodge", binwidth = 0.25) +
  labs(title = "Histogram of Residuals Per Models", 
       x = "Residuals", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))

## Explore predictions of multiple models. Not much difference.
model_summary %>% dplyr::select(imp_model, pred) %>% unnest() %>%
  ggplot(aes(x = pred, fill = imp_model)) + 
  geom_histogram(position = "dodge") +
  labs(title = "Histogram of Predictions Per Models", 
       x = "Predictions", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))

## Proceed with mice and omit collinear variables detected by the model 
secom <- secom_mice_imp %>% dplyr::select(-contains(collinear))
any_na(secom)

## Write csv and clean the environment
write_csv(secom, "secom_C4d.csv")
rm(list = ls()[!(ls() %in% c("secom"))])


# C5. Feature Selection -----------------------------------------------------
secom <- read_csv("secom_C4d.csv")
secom$class <- as.factor(secom$class)

## C5.1 Wrapper Methods (Boruta vs Traditional RFE) -------------------------

#### C5.1.1 BORUTA
load("boruta.RData")

## Importance source: Random Forest classifier, based on 250 iterations
set.seed(789)
boruta <- Boruta(class ~ ., data = secom[, -c(1, 3)], maxRuns = 250, 
                 doTrace = 2, getImp = getImpRfZ, num.threads = 3)
boruta$timeTaken ## ~7 mins

## Save Boruta class object and examine the summary
save(boruta, file = "boruta.RData")
print(boruta)

## Alternative importance source, rFerns: purely stochastic ensemble classifier
set.seed(789)
boruta_Ferns <- Boruta(class ~ ., data = secom[, -c(1, 3)], maxRuns = 250, 
                       doTrace = 2, getImp = getImpFerns, threads = 3)
boruta_Ferns$timeTaken ## ~15 seconds
print(boruta_Ferns)

## 11 features are selected (no tentatives)
(Ferns_selected <- getSelectedAttributes(boruta_Ferns, withTentative = F))

## 17 features are selected with 99% CI (2 features are tentative) 
(boruta_selected <- getSelectedAttributes(boruta, withTentative = F))
(boruta_all <- getSelectedAttributes(boruta, withTentative = T))

## Examine importance scores of Boruta
boruta_imp <- boruta %>% 
  attStats() %>%
  rownames_to_column("features") %>%
  dplyr::select(features, meanImp, decision) %>%
  arrange(desc(meanImp))

## Boruta variable importance chart
plot(boruta, cex.axis =.7, las=2, xlab="", main = "Variable Importance")

## Spearman to captures all types of (+) or (-) relationships (exp, log)
corr_selected <- cor(secom[, boruta_selected], method = "spearman")
corr_all <- cor(secom[, boruta_all], method = "spearman")

## Correlation Plot of selected features clustered by correlations
corrplot::corrplot(corr_selected, tl.col = "black", tl.cex = 0.75, 
                   order = "hclust", hclust.method = "average", addrect = 5)

## Correlation Plot of selected + tentative features clustered by correlations
corrplot::corrplot(corr_all, tl.col = "black", tl.cex = 0.75, 
                   order = "hclust", hclust.method = "average", addrect = 5)


#### C5.1.2 TRADITIONAL 
## Backwards Selection aka Recursive Feature Elimination
load("rfe.RData")

## Make a cluster for parallel processing in RFE.
cl <- makeCluster(detectCores() - 1, outfile = "")
registerDoParallel(cl) 

## Generate the control object to specify the details of the feature selection 
## rfFuncs: random forest selection (the underlying classifier in Boruta)
set.seed(789)
options(warn = -1)
ctrl <- rfeControl(functions = rfFuncs, method = "cv",
                   verbose = FALSE, saveDetails = TRUE)

## Implement the RFE algorithm, target should be a vector, ~30 mins elapsed
system.time({ rfe <- rfe(as.matrix(secom[, -c(1:3)]), secom[[2]], 
                         sizes = 1:100, rfeControl = ctrl) })
stopCluster(cl)

# Backup the object
save(rfe, file = "rfe.RData")
print(rfe)

## Examine the features
rfe_imp <- rfe$fit$importance %>% 
  as.data.frame() %>%
  rownames_to_column("features") %>%
  arrange(desc(MeanDecreaseAccuracy))

## 8 features are similar including top 5 most important. 
boruta_selected[boruta_selected %in% rfe$optVariables]

## While Boruta brings 9 additional features, RFE brings only additional 4. 
boruta_selected[!boruta_selected %in% rfe$optVariables]
rfe$optVariables[!rfe$optVariables %in% boruta_selected]

## Clean the environment
rm(list = ls()[!(ls() %in% c("secom"))])


## C5.2 Embedded Methods (Lasso) --------------------------------------------

## Import and convert data & define predictors and the target
secom <- read_csv("secom_C4d.csv")
secom <- as.data.frame(secom)
x <- as.matrix(secom[, c(4:420)])
y <- as.factor(secom[, 2])

# Lasso doesn't work well with imbalanced datasets, with low EPV
# EPV: n(obs)/n(predictors), expected by lasso EPV >= 25, preferable is >= 50
nrow(secom)/ncol(secom[, 4:420]) # ours is just 3.

## Iterate over cv.glmnet 5 times and take 'cvm': mean cross-validated error
## Best possible lambda that can produce the lowest error = 0.016
load("cv.RData")
load("cvm.RData")

set.seed(100)
cvm <- NULL
system.time({ # Run time = ~ 1 min
for (i in 1:2){ 
  cv <- cv.glmnet(x, y, family = "binomial", alpha = 1, standardize = TRUE)  
  cvm <- cbind(cvm, cv$cvm)
  } 
})

# Calculate the means of each row in CVM, and find the lowest error 
# to get minimum lambda from the iteration
cvm <- as.data.frame(cvm)
cvm$row_mean <- rowMeans(cvm)
cvm$lambda <- cv$lambda
lambda.min <- as.numeric(cvm$lambda[which.min(cvm$row_mean)])

save(cv, file = "cv.RData")
save(cvm, file = "cvm.RData")
plot(cv)

## Rebuilding the model with best lambda identified
lasso_best <- glmnet(x, y, family = "binomial", alpha = 1, 
                     lambda = lambda.min)

## Lasso model doesn't work well with imbalanced sets and predicts only 
## majority class "0", due to the penaly threshold is too high 
pred <- predict(lasso_best, newx = x, s = lambda.min, type = "class")
head(pred, n = 10)

## Extract important features highest coefficient
coef_list <- coef(lasso_best, s = lambda.min)
coef_list <- data.frame(coef_list@Dimnames[[1]][coef_list@i+1], coef_list@x)
names(coef_list) <- c('features', 'coefficient')

## 11 features are selected
coef_list %>% arrange(-abs(coefficient))

## Clean the environment
rm(list = ls())

# Deploy The Data ---------------------------------------------------

## Retrieve the necessary objects
load("boruta.RData")
boruta_imp <- boruta %>% 
  attStats() %>%
  dplyr::rownames_to_column("features") %>%
  dplyr::select(features, meanImp, decision) %>%
  dplyr::arrange(desc(meanImp))

## Select features, subset and export the data
features <- sort(boruta_imp[1:18, 1])
secom <- secom %>% select(ID, class, all_of(features))
write_csv(secom, "secom_13062020.csv")
