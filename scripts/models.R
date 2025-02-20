# install and load packages
install.packages("doParallel")
install.packages("klaR")
library(klaR)
library(doParallel)
library(mlbench)
library(lattice)
library(plyr)
library(Rmisc)
library(Hmisc)
library(caret)
library(corrplot)
library(e1071)
library(AppliedPredictiveModeling)
library(ggplot2)
library(dplyr)
library(pROC)
library(sparseLDA)
library(MLmetrics)
library(MASS)
library(mda)

# Load dataset
data = read.csv('dataset/data.csv')
attach(data)
str(data)

############################
# General data preprocessing
###########################
# check missing values
colSums(is.na(data))
anyNA(data)
sum(is.na(data))
# convert bankrupt to levels/factor
data$Bankrupt.=ifelse(data$Bankrupt.== 1, 'yes', 'no')
data$Bankrupt. = as.factor(data$Bankrupt.)
data$Liability.Assets.Flag = as.integer(Liability.Assets.Flag)
data$Net.Income.Flag = as.integer(Net.Income.Flag)
# check near zero variance predictors
preds_rmv = nearZeroVar(subset(data,select = -Bankrupt.))
colnames(data)[preds_rmv]
data=data[,-preds_rmv]
data=subset(data,select = -Net.Income.Flag)
# check distributions and apply box-cox transformation
data_nums = subset(data,select=-c(Liability.Assets.Flag,Bankrupt.))
data_cat = subset(data,select= c(Liability.Assets.Flag,Bankrupt.))
#ditribution for countinous variables
par(mfrow = c(3,3))
for (i in 1:9) {
  hist(data_nums[, i],main = colnames(data_nums)[i],
       breaks=30,xlab = '', border = "black")
}
# Reset the plot layout
par(mfrow = c(1, 1))
# check skewness
skew_vals = apply(data_nums[,1:9], 2, skewness)
skew_vals
# apply box cox
# Add constants before box cox transformation
add_constant=function(col){
  col = col + abs(min(col)) + 1
  return(col)
}
data_pre_boxcox = as.data.frame(lapply(data_nums, add_constant))
data_pre = preProcess(data_pre_boxcox, method = c("BoxCox"))
#check lambdas
data_pre
# Apply
data_boxcox_nums_trans = predict(data_pre, data_pre_boxcox)

skew_vals = apply(data_boxcox_nums_trans[,1:9], 2, skewness)
skew_vals

# check distributions after transformation
par(mfrow = c(3,3))
for (i in 1:9) {
  hist(data_boxcox_nums_trans[, i],main = colnames(data_boxcox_nums_trans)[i],
       breaks=30,xlab = '', border = "black")
}
# Reset the plot layout
par(mfrow = c(1, 1))
# final dataset after general preprocess
data_fin = cbind(data_boxcox_nums_trans,data_cat)
detach(data)
attach(data_fin)
#######################################################################
# MODEL BUILDING: Linear Classification Models
#######################################################################
#####################
# Logistic Regression
#####################
predictors = subset(data_fin,select=-Bankrupt.)
# remove highly correlated predictors
correlations = cor(predictors)
# corrplot(correlations, order = "hclust", type = 'upper',tl.pos='n')
corrplot(correlations, order = "hclust", tl.pos='n')


highCorr = findCorrelation(correlations, cutoff = .75)
length(highCorr)
predictors_corr_filter = predictors[,-highCorr]
# response varible dstribution
table(Bankrupt.)
barplot(table(Bankrupt.),
        main = "Response variable",
        xlab = "(0) not bankrupt (1) bankrupt",
        ylab = "Frequency",
        border = "black")
# Data splitting
trainingRows = createDataPartition(Bankrupt., p = .80, list= FALSE)

# Subset the data into objects for training using
# integer sub-setting
trainPredictors = predictors_corr_filter[trainingRows, ]
trainClasses = Bankrupt.[trainingRows]

# Do the same for the test set using negative integers.
testPredictors =predictors_corr_filter[-trainingRows, ]
testClasses = Bankrupt.[-trainingRows]

table(trainClasses)
table(testClasses)
# model building
# resampling
# Define the trainControl with Kappa and class probabilities
set.seed(100)
ctrl = trainControl(method = "CV",
                    number = 5,
                    summaryFunction = defaultSummary,
                    classProbs = TRUE,
                    savePredictions = TRUE)

log_fit = train(trainPredictors, trainClasses,
                method = "glm",
                metric = "Kappa",  
                trControl = ctrl)
log_fit
# test
log_pred = predict(log_fit, newdata = testPredictors)
## perforamnce values
postResample(pred = log_pred, obs = testClasses)

confusionMatrix(data = log_pred,
                reference = testClasses)

####################################
# Linear Discriminant Analysis (LDA)
####################################

# model building
lda_fit = train(trainPredictors, trainClasses,
                 method = "lda",
                 preProc = c("center", "scale"),
                 metric = "Kappa",
                 trControl = ctrl)

lda_fit

# test
lda_pred = predict(lda_fit, newdata = testPredictors)
## perforamnce values
postResample(pred = lda_pred, obs = testClasses)

confusionMatrix(data = lda_pred,
                reference = testClasses)
#####################################################
# Partial Least Squares Discriminant Analysis (PLSDA)
#####################################################
set.seed(100)
# Data splitting
trainingRows = createDataPartition(Bankrupt., p = .80, list= FALSE)
# Subset the data into objects for training using
# integer sub-setting
trainPredictors = predictors[trainingRows, ]
trainClasses = Bankrupt.[trainingRows]
# Do the same for the test set using negative integers.
testPredictors =predictors[-trainingRows, ]
testClasses = Bankrupt.[-trainingRows]

# model building
plsda_fit = train(trainPredictors, trainClasses,
                  method = "pls",
                  tuneLength = 6,
                  preProc = c("center","scale"),
                  metric = "Kappa",
                  trControl = ctrl)
plsda_fit
plot(plsda_fit)

# test
plsda_pred = predict(plsda_fit, newdata = testPredictors)
## perforamnce values
postResample(pred = plsda_pred, obs = testClasses)

confusionMatrix(data = plsda_pred,
                reference = testClasses)

######################
# Penalized models
#####################

set.seed(100)
pen_fit = train(trainPredictors, trainClasses,
                 method = "glmnet",
                 tuneLength = 4,
                 preProc = c("center", "scale"),
                 metric = "Kappa",
                 trControl = ctrl)
pen_fit
## The heat map in the top panel of Fig. 12.16 was produced using the code
plot(pen_fit)
# plot(pen_fit, plotType = "level")

# test
pen_pred = predict(pen_fit, newdata = testPredictors)
## perforamnce values
postResample(pred =pen_pred, obs = testClasses)

confusionMatrix(data = pen_pred,
                reference = testClasses)

#######################################################################
# MODEL BUILDING: Non-Linear Classification Models
#######################################################################

#########################################
# Mixture Discriminant Analysis (MDA)
########################################
# remove highly correlated predictors
correlations = cor(predictors)
highCorr = findCorrelation(correlations, cutoff = .75)
length(highCorr)
predictors_corr_filter = predictors[,-highCorr]
# Data splitting
trainingRows = createDataPartition(Bankrupt., p = .80, list= FALSE)

# Subset the data into objects for training using
# integer sub-setting
trainPredictors = predictors_corr_filter[trainingRows, ]
trainClasses = Bankrupt.[trainingRows]

# Do the same for the test set using negative integers.
testPredictors =predictors_corr_filter[-trainingRows, ]
testClasses = Bankrupt.[-trainingRows]

# model building
mda_fit = train(trainPredictors, trainClasses,
                method = "mda",
                preProc = c("center", "scale"),
                tuneLength = 4,
                metric = "Kappa",
                trControl = ctrl)
mda_fit
plot(mda_fit)

# test
mda_pred = predict(mda_fit, newdata = testPredictors)
## perforamnce values
postResample(pred = mda_pred, obs = testClasses)

confusionMatrix(data = mda_pred,
                reference = testClasses)
#########################################
# Regularized Discriminant Analysis (RDA)
########################################
# remove highly correlated predictors
set.seed(100)
correlations = cor(predictors)
highCorr = findCorrelation(correlations, cutoff = .65)
length(highCorr)
predictors_corr_filter = predictors[,-highCorr]
# Data splitting
trainingRows = createDataPartition(Bankrupt., p = .80, list= FALSE)

# Subset the data into objects for training using
# integer sub-setting
trainPredictors = predictors_corr_filter[trainingRows, ]
trainClasses = Bankrupt.[trainingRows]

# Do the same for the test set using negative integers.
testPredictors =predictors_corr_filter[-trainingRows, ]
testClasses = Bankrupt.[-trainingRows]
# model building
rda_fit = train(trainPredictors, trainClasses,
                method = "rda",
                preProc = c("center", "scale"),
                tuneLength = 4,
                metric = "Kappa",
                trControl = ctrl)
rda_fit
plot(rda_fit)

# test
rda_pred = predict(rda_fit, newdata = testPredictors)
## perforamnce values
postResample(pred = rda_pred, obs = testClasses)

confusionMatrix(data = rda_pred,
                reference = testClasses)

#########################################
# Quadratic Discriminant Analysis (QDA)
########################################
# remove highly correlated predictors
correlations = cor(predictors)
highCorr = findCorrelation(correlations, cutoff = .75)
length(highCorr)
predictors_corr_filter = predictors[,-highCorr]
# Data splitting
trainingRows = createDataPartition(Bankrupt., p = .80, list= FALSE)

# Subset the data into objects for training using
# integer sub-setting
trainPredictors = predictors_corr_filter[trainingRows, ]
trainClasses = Bankrupt.[trainingRows]

# Do the same for the test set using negative integers.
testPredictors =predictors_corr_filter[-trainingRows, ]
testClasses = Bankrupt.[-trainingRows]

# model building
qda_fit = train(trainPredictors, trainClasses,
                method = "qda",
                preProc = c("center", "scale"),
                metric = "Kappa",
                trControl = ctrl)
qda_fit

# test
qda_pred = predict(qda_fit, newdata = testPredictors)
## perforamnce values
postResample(pred = qda_pred, obs = testClasses)

confusionMatrix(data = qda_pred,
                reference = testClasses)
####################################
# Flexible Discriminant Analysis (FDA)
####################################
marsGrid = expand.grid(.degree = 1:2, .nprune = 2:38)
## train model
fda_fit = train(trainPredictors, trainClasses,
                 method = "fda",
                 tuneLength = 10,
                 # tuneGrid = marsGrid,
                 metric = 'Kappa',
                 trControl = ctrl)

fda_fit

plot(fda_fit)
# plot(fdaTuned,main="FDA, degree = 1 and nprune = 6")
# test
fda_pred = predict(fda_fit, newdata = testPredictors)
## The function 'postResample' can be used to get the test set
## perforamnce values
postResample(pred = fda_pred, obs = testClasses)

confusionMatrix(data = fda_pred,
                reference = testClasses)

########################
# KNN Model
########################
set.seed(100)
# Data splitting
trainingRows = createDataPartition(Bankrupt., p = .80, list= FALSE)
# Subset the data into objects for training using
# integer sub-setting
trainPredictors = predictors[trainingRows, ]
trainClasses = Bankrupt.[trainingRows]
# Do the same for the test set using negative integers.
testPredictors =predictors[-trainingRows, ]
testClasses = Bankrupt.[-trainingRows]
# train
knn_fit = train(trainPredictors, trainClasses,
                method = "knn",
                preProc = c("center", "scale"),
                tuneLength = 10,
                metric = 'Kappa',
                trControl = ctrl)
# tuneGrid = data.frame(k = 1:10)
knn_fit
plot(knn_fit)
# test
knn_Pred = predict(knn_fit, newdata = testPredictors)
postResample(pred = knn_Pred, obs = testClasses)
confusionMatrix(data = knn_Pred,
                reference = testClasses)

########################
# Naive Bayes
########################
nb_fit = train(trainPredictors, trainClasses,
               method = "nb",
               # preProc = c("center", "scale"),
               tuneGrid = data.frame(.fL = 2, .usekernel = FALSE, .adjust = TRUE),
               metric = 'Kappa',
               trControl = ctrl)
nb_fit
# test
nb_Pred = predict(nb_fit, newdata = testPredictors)
postResample(pred = nb_Pred, obs = testClasses)
confusionMatrix(data = nb_Pred,
                reference = testClasses)








