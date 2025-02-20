######################################
#####First 46 predictors#####
##import libraries
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

data = read.csv('dataset/data.csv')
attach(data)
str(data)

data_kbw = data
data_kbw_ints = data_kbw[,sapply(data_kbw,is.integer)]
data_kbw_nums = data_kbw[,!sapply(data_kbw,is.integer)]
str(data_kbw_ints)
unique(Liability.Assets.Flag)
table(data[,'Liability.Assets.Flag'])
table(data[,'Net.Income.Flag'])

# investigate class variable
# Create the bar plot
# Create a table of counts for the variable

# Generate the bar plot and capture the bar positions
barplot(table(Bankrupt.),
                         main = "Response variable",
                         xlab = "(0) not bankrupt (1) bankrupt",
                         ylab = "Frequency",
                         col = "lightpink",
                         border = "black")



str(data_kbw_nums)
attach(data_kbw_nums)
#Relationship between predictors
## calculate the correlations between predictor variables:
correlations = cor(data_kbw_nums)
dim(correlations)
correlations[90:92, 88:90]

# visually examine the correlation structure of the data
corrplot(correlations, order = "hclust", tl.pos='n')

#filter based on correlations, the findCorrelation function will apply the
#algorithm in Sect. 3.5. For a given threshold of pairwise correlations, the function
#returns column numbers denoting the predictors that are recommended
#for deletion:
highCorr = findCorrelation(correlations, cutoff = .75)
length(highCorr)
highCorr
#remove high correlated predictors
data_kbw_nums = data_kbw_nums[, -highCorr]
length(data_kbw_nums)

#check relatioship after removing highly correlated predictors
correlations_filter = cor(data_kbw_nums)
dim(correlations_filter)
# visually examine the correlation structure of the data
corrplot(correlations_filter, order = "hclust", tl.pos='n')

data_kbw_ints = subset(data_kbw_ints, select = -c(Bankrupt.))
# Create bar plots for each categorical variable
par(mfrow = c(1,2))
for (i in 1:ncol(data_kbw_ints)) {
  barplot(table(data_kbw_ints[, i]), main = colnames(data_kbw_ints)[i], col = "lightpink")
}

#check for most frequency
for (col in names(data_kbw_ints)) {
  print(paste("Frequencies for", col))
  freq_table = table(data_kbw_ints[[col]])
  freq_pct = prop.table(freq_table) * 100
  combine_freq_table = cbind(Frequency = freq_table, freq_pct = round(freq_pct, 2))
  print(combine_freq_table)
  cat("\n")  #new line for better readability
}

#ditribution for countinous variables
par(mfrow = c(3,3))
for (i in 1:ncol(data_kbw_nums)) {
  hist(data_kbw_nums[, i],main = colnames(data_kbw_nums)[i],
       breaks=30,xlab = '', col = "lightpink", border = "black")
}
# Reset the plot layout
par(mfrow = c(1, 1))

#calculate skewness
skew_vals = apply(data_kbw_nums, 2, skewness)
skew_vals  ##skewValues



table(data[,"Bankrupt."])


#visualize outliers using boxplots
# Create box plots first 25
par(mfrow=c(3, 3))  
for(i in 1:ncol(data_kbw_nums)) {
  boxplot(data_kbw_nums[,i], main=colnames(data_kbw_nums)[i], col="lightpink")
}
# Reset the plot layout
par(mfrow = c(1, 1))

Quick.Assets.Current.Liability

############################################################################
##Transformations################3
############################################################################

# approximately symetric predictors
app_sym = names(skew_vals[skew_vals > -0.5 & skew_vals < 0.5])
app_sym
data_app_sym = data_kbw_nums[, app_sym, drop = FALSE]
str(data_app_sym)
# predictors to be transformed
data_pre_boxcox = data_kbw_nums[,!names(data_kbw_nums) %in% app_sym]
str(data_pre_boxcox)
str(app_sym)
# Add constants before box cox transformation
add_constant=function(col){
  col = col + abs(min(col)) + 1
  return(col)
}
data_pre_boxcox_select = as.data.frame(lapply(data_pre_boxcox, add_constant))
data_app_sym = as.data.frame(lapply(data_app_sym, add_constant))
str(data_app_sym)
str(data_pre_boxcox_select)

# Apply boxcox
data_pre = preProcess(data_pre_boxcox_select, method = c("BoxCox"))
#check lambdas
data_pre
# Apply
data_boxcox_trans = predict(data_pre, data_pre_boxcox_select)

#calculate skewness after transformation
skew_vals = apply(data_boxcox_trans, 2, skewness)
skew_vals  ##skewValues
# approximately symetric predictors after transformation
app_sym = names(skew_vals[skew_vals > -0.5 & skew_vals < 0.5])
app_sym
# Add the approximately symmetric predictors back
data_boxcox_fin = cbind(data_boxcox_trans,data_app_sym)
str(data_boxcox_fin)

# check distribution afterboxplot
#ditribution for countinous variables
data_boxcox_fin_samp = subset(data_boxcox_fin, select = c(Current.Assets.Total.Assets,Cash.Total.Assets,
                                                          Quick.Assets.Current.Liability,Cash.Current.Liability,
                                                          Current.Liability.to.Assets,Inventory.Working.Capital,
                                                          Inventory.Current.Liability,Long.term.Liability.to.Current.Assets,
                                                          Retained.Earnings.to.Total.Assets))
par(mfrow = c(3,3))
for (i in 1:ncol(data_boxcox_fin_samp)) {
  hist(data_boxcox_fin_samp[, i],main = colnames(data_boxcox_fin_samp)[i],
       breaks=50,xlab = '', col = "lightpink", border = "black")
}
# Reset the plot layout
par(mfrow = c(1, 1))

#apply spatial sign to minimize effects of outliers
data_pre_spatial = preProcess(data_boxcox_fin, method = c("center", "scale", "spatialSign"))
data_pre_spatial
# Apply the transformations:
data_post_spatial = predict(data_pre_spatial, data_boxcox_fin)

str(data_post_spatial)

#ditribution for countinous variables
par(mfrow = c(4,4))
for (i in 1:ncol(data_post_spatial)) {
  hist(data_post_spatial[, i],main = colnames(data_post_spatial)[i],
       breaks=30,xlab = '', col = "lightpink", border = "black")
}
# Reset the plot layout
par(mfrow = c(1, 1))

#visualize after outliers using boxplots after spatial
par(mfrow=c(3, 3))  
for(i in 1:ncol(data_post_spatial)) {
  boxplot(data_post_spatial[,i], main=colnames(data_post_spatial)[i], col="lightpink")
}
# Reset the plot layout
par(mfrow = c(1, 1))

# PCA for dimension reduction
data_pre_pca = preProcess(data_post_spatial, method = c('pca'))
data_pre_pca 
data_post_pca = predict(data_pre_spatial, data_boxcox_fin)
data_post_pca



# Perform PCA
pca_result <- prcomp(data_post_spatial, Center = TRUE, scale. = TRUE)

# Extract the variance explained by each principal component
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Create an elbow plot
plot(
  x = 1:length(explained_variance),
  y = cumsum(explained_variance),
  type = "b",  # Plot points and lines
  xlab = "Principal Components",
  ylab = "Cumulative Proportion of Variance Explained",
)




















