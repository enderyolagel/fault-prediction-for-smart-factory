# 1. PACKAGES & DATASET -------------------------------------------------------
start.time <- Sys.time() # Placeholder for measuring running time of whole script

## Check and install only missing packages
team_packages <- c("tidyverse", "broom", "party", "rpart", "foreign", "Hmisc", 
                   "finalfit", "outliers", "dlookr", "ggpubr", "rstatix")
new_packages <- team_packages[!(team_packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) {install.packages(new_packages)}

## Load all packages at once
lapply(team_packages, require, character.only = TRUE)

## Import the data
rawdata <- read.spss("data/secom_mod.SAV", to.data.frame = TRUE)

# 2. UNIVARIATE ANALYSIS -----------------------------------------------------

dim(rawdata) ## There are 1567 rows, 593 columns in the dataset
glimpse(rawdata) ## Features & Classes: ID (dbl), class (dbl), timestamp (factor), features001-590 (dbl)
head(rawdata) ## doublecheck with head()
str(rawdata, list.len = 10) ## 'class' has to be changed as factor, 'timestamp' needs to be changed as.POSIXct
rawdata %>% count(class) ## 1472 pass (0) (93.93746%) and 95 fail (1) (6.06254%)
nrow(rawdata) - nrow(unique(rawdata)) # No duplicated observations = 0


## 2.1 DESCRIPTIVE STATISTICS  

### Summary Statistics in Dlookr Package 
summary_statistics_dlookr <- describe(rawdata)
head(summary_statistics_dlookr)

### 538 Features that have missing values
summary_statistics_dlookr %>% filter(na > 0) %>% count()

### 28 features that have missing values more than 50%.
summary_statistics_dlookr %>% 
  mutate(complete_rate = na / nrow(rawdata)) %>% 
  filter(complete_rate > 0.5) %>% count()

### Top 20 Features that have largest skewness
summary_statistics_dlookr %>%
  select(variable, skewness, kurtosis, mean, sd, p25, p50, p75) %>% 
  filter(!is.na(skewness)) %>% 
  arrange(desc(abs(skewness))) %>%
  top_n(., n = 20)

### Dataset that includes only features
features_rawdata <- rawdata[, -c(1:3)]

### Detecting columns without zero variance
variance <- apply(features_rawdata, 2, var, na.rm = TRUE) != 0

### 116 features with constant values
table(variance)["FALSE"] 

### Dataset that includes only features without constants
features_with_variance <- features_rawdata[, variance]
dim(features_with_variance)


## 2.2 CHECKING NORMALITY OF DATA BY VISUAL AND BY SIGNIFICANCE TESTS

### Shapiro-Wilk normality test for all features 
normality_table <- normality(features_with_variance)

### P-value of 473 out of 474 features are smaller than alpha level (0.05)   
normality_table %>% filter(p_value <= 0.05) %>% arrange(p_value) %>% nrow()

### Only feature084 is normally distributed 
normality_table %>% filter(p_value > 0.05) %>% arrange(p_value)

### Bottom 10 features with the lowest p-value and test statistic
(normality_bot10 <- normality_table %>% arrange(p_value) %>% head(n = 10))

### Histogram and QQ plot of original bottom 10 features
### Histogram of log and square root transformed bottom 10 features
plot_normality(features_with_variance, normality_bot10$vars)

### Middle 10 features in terms of sorted p_values
### Slice() choose rows by their ordinal position in the tbl
(normality_mid10 <- normality_table %>% arrange(p_value) %>% slice(., 291:300))

### Histogram and QQ plot of original bottom 10 features
### Histogram of log and square root transformed bottom 10 features
plot_normality(features_with_variance, normality_mid10$vars)


# 3. MULTIVARIATE ANALYSIS ---------------------------------------------------

### Pearson Correlation Coefficients Table for each variable combinations
correlation_table <- correlate(features_with_variance)
head(correlation_table)
dim(correlation_table)

ggplot(correlation_table) + 
  geom_histogram(aes(x = coef_corr), 
                 color = "black", fill = "lightblue", bins = 50) +
  labs(title = "Histogram of Pairwise Correlations of Features",
       x = "Pearson Correlation Coefficients",
       y = "Frequency") + 
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))

# 4. TARGET FEATURE ANALYSIS --------------------------------------------------

rawdata$class <- factor(rawdata$class)
class(rawdata$class)
levels(rawdata$class)

## Create a target_by class on target feature with an object inheriting grouped df 
target_categ <- target_by(rawdata, class)
class(target_categ)


### Show relationship between Mid 10 features and target feature that correspond to 
### the combination of descriptive stats for each levels and total observation.
cat_num_mid1 <- relate(target_categ, feature121)
cat_num_mid2 <- relate(target_categ, feature183)
cat_num_mid3 <- relate(target_categ, feature527)
cat_num_mid4 <- relate(target_categ, feature106)
cat_num_mid5 <- relate(target_categ, feature567)
cat_num_mid6 <- relate(target_categ, feature240)
cat_num_mid7 <- relate(target_categ, feature169)
cat_num_mid8 <- relate(target_categ, feature094)
cat_num_mid9 <- relate(target_categ, feature565)
cat_num_mid10 <- relate(target_categ, feature487)


### Density plots (PDF) of Mid 10 in terms of sorted P-value of SW test 
### Non-normally distributed Mid features and Target Variable
### These plot functions use plot.relate() method for "relate" objects
plot(cat_num_mid1)
plot(cat_num_mid2)
plot(cat_num_mid3) 
plot(cat_num_mid4)
plot(cat_num_mid5)
plot(cat_num_mid6)
plot(cat_num_mid7)
plot(cat_num_mid8)
plot(cat_num_mid8)
plot(cat_num_mid9)
plot(cat_num_mid10)

# Manual doublecheck for the plot "cat_num_mid10" of feature487
plot(density(rawdata$feature487, na.rm = T), xlim = c(0, 1000))


# 5. DATA QUALITY ------------------------------------------------------------

## 5.1 OUTLIER ANALYSIS

### Features sorted according to number of outliers
outlier_table <- diagnose_outlier(features_with_variance) %>% 
  arrange(desc(outliers_cnt))
head(outlier_table)

### Features with average of outliers greater than the average of all observations
features_larger_outliers <- outlier_table %>%
  filter(outliers_ratio > 10) %>% # Features that have outliers more than 10%
  mutate(rate = outliers_mean / with_mean) %>% 
  filter(rate > 1) %>%
  arrange(desc(rate)) %>% 
  select(-outliers_cnt)
features_larger_outliers

### Outlier Diagnosis Plots that includes the boxplots and histograms 
### with and without (before and after removing) outliers for each 18 features
features_with_variance %>%
  plot_outlier(features_larger_outliers %>% select(variables) %>% unlist())


## 5.2 MISSING VALUE ANALYSIS

sum(is.na(rawdata)) # total number of NAs: 41951
colSums(as.matrix(summary_statistics_dlookr$na)) # Cross-check

# Proportion of NAs to whole: 4.5%
sum(is.na(rawdata))/(nrow(rawdata)*ncol(rawdata)) 

# Missing Values Table
na_table <- diagnose(rawdata)
na_table %>% filter(missing_percent > 10) %>% count()

# Histogram of Number of missing values per all features
ggplot(na_table) + 
  geom_histogram(aes(x = missing_count), 
                 color = "black", fill = "lightblue", bins = 20) +
  labs(title = "Histogram of Number of NAs Per Feature",
       x = "Number of Missing Values Per All Features",
       y = "Frequency") + 
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))


# Histogram of only features that have NAs less than 100
ggplot(na_table %>% filter(missing_count < 100)) + 
  geom_histogram(aes(x = missing_count), 
                 color = "black", fill = "lightblue", bins = 50) +
  labs(title = "Histogram of Features with Less Than 100 NAs",
       x = "Number of Missing Values Per Feature",
       y = "Frequency") + 
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(5, 10, 5, 10), units = "mm"))


# 28 features that have more than 50% missing values
features_na_50 <- na_table %>% 
  filter(missing_percent > 50) %>% 
  select(-types) %>%
  arrange(desc(missing_percent))
print(features_na_50, n = 28)

### Manual crosscheck - Table shows percentage of NAs per column
na_per_feature <- data.frame(
  "column" = names(rawdata), 
  "na_percent" = round(colMeans(is.na(rawdata)), 4)
)

### Number of Features with more than 10%, 40%, 50% missing values
na_per_feature %>% filter(na_percent > 0.1) %>% count() # 52 features
na_per_feature %>% filter(na_percent > 0.4) %>% count() # 32 features
na_per_feature %>% filter(na_percent > 0.5) %>% count() # 28 features
na_per_feature %>% filter(na_percent > 0.55) %>% count() # 24 features


### Table shows percentage of NAs per case
na_per_case <- data.frame(
  "ID" = rawdata[, 1], "class" = rawdata[, 2], "timestamp" = rawdata[, 3], 
  "na_percent" = round(rowMeans(is.na(rawdata)), 2)
)
head(na_per_case, n = 10)

# Performance of the script ---------------------------------------------------

end.time <- Sys.time() # Last placeholder
time.taken <- round(end.time - start.time, 2) ## Total running time of whole script 
time.taken # 44.21 seconds after all packages installed
# To measure total runtime of whole script please run this script from this line 

  