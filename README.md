# Bankruptcy-Prediction-
Taiwan bankruptcy prediction using statistical machine learning methods

```
Kepha Barasa and Muneendra Magani

```
## Abstract
Accurate bankruptcy prediction is essential for financial institutions to make
better lending decisions. The effectiveness of such predictions mainly depends on the
data used and the methods applied. While many studies focus on refining prediction
techniques, few have examined which types of data are most useful for predicting
bankruptcy.Research indicates that both financial data and corporate governance
factors are important in predicting bankruptcy. However, few studies have explored the
combined effect of these two data types on prediction accuracy. Additionally, the choice
of data varies across studies, with no clear agreement on which factors are most
critical.This study aims to assess the impact of combining various financial and
governance data on bankruptcy prediction. Using data from Taiwan, the results show
that certain financial and governance characteristics are most influential in predicting
bankruptcy.

## Background
Bankruptcy or business failure can harm both the company involved and the
wider economy. For years, business experts, investors, governments, and researchers
have been trying to find ways to spot the risk of a company failing, to help reduce the
financial damage caused by bankruptcy (Balleisen, 2001; Zywicki, 2008).
In the United States, approximately 18,926 bankruptcy cases were filed in 2023,
marking a 40% increase from the previous year (US Courts, 2024). These bankruptcy
cases can lead to significant economic disruptions in communities that rely on
businesses for services and household income. When a company goes bankrupt, it
often results in layoffs, contributing to an increase in unemployment and a decline in
economic activity in affected areas. Additionally, investors are impacted as they are less
likely to receive the full value of their investments in the company, further exacerbating
the financial strain on the community.
Recent research has witnessed the emergence of innovative statistical and
machine learning methods aimed at predicting business bankruptcy earlier, with the
goal of mitigating the impact of business failures. The Altman Z-score model (Anjum,
2012), introduced in 1968, remains one of the earliest and most influential predictive
models in this field. Its enduring relevance has inspired contemporary academic
researchers to build upon its foundation, striving to enhance predictive capabilities and
develop more novel bankruptcy forecasting tools.
A significant challenge in developing these models is identifying relevant financial
indicators that contribute to bankruptcies. The most common approach involves using
financial ratios, which are readily accessible through quarterly financial statements
published by public companies. These ratios provide insights into a company's financial
health and potential risk of bankruptcy, allowing researchers to create more accurate
prediction models.

## Data Preprocessing

The dataset obtained from Kaggle had already undergone initial preprocessing,
with categorical predictors encoded and no missing samples present. This allowed us to
proceed directly to subsequent preprocessing steps, which are detailed in the following
sections.

### A. Near-zero variance
```
We identified three predictors with near-zero variance: current liability to
current assets, interest coverage ratio to EBIT, and Net income flag. These
variables were removed from the dataset and we retained 92 predictors.
```

### B. Transformations

```
We investigated the dataset further and found highly significant skewed
predictors. Most predictors were negatively skewed as seen in table 1 illustrating
the first nine predictors’ skewness. To alleviate the effects of skewness on
models, we implemented a box cox transformation on all continuous predictors to
improve the symmetry.
```

## Predictors Skewness

```
After tax net interest rate -52.97 -49.
Non industry income and expenditure 39.62 -6.
Figures 1 and 2 illustrate a significant improvement in symmetry following
the application of the Box-Cox transformation. Table 1 presents the skewness
coefficients for selected predictors, with variables such as Non-industry income
and expenditure, operating gross margin, and realized sales gross margin
showing improvements in their distributional symmetry. Although other predictors
have slight improvements, the overall reduction in skewness across the dataset
is expected to contribute positively to model performance.
```
### C. Correlations

```
A correlation matrix was employed to investigate the relationships among
predictors. Figure 3 illustrates both negative and positive correlations between
variables, an expected outcome given that many financial ratios share common
denominators. To address multicollinearity, a correlation coefficient cutoff of 0.
was implemented. Predictors with correlations greater than the cutoff were
eliminated from the dataset.
This process resulted in the removal of 28 predictors, leaving 64 variables.
```

## Data Spending

The dataset shows high class imbalance, with bankruptcy cases representing
less than 4% of the sample, as illustrated in Figure 4. To address this imbalance and
minimize potential model bias, stratified random sampling was implemented. This
sampling technique ensures sufficient representation of the minority class in both
training and testing sets. The dataset was split, allocating 80% of the sample to the
training set and 20% to the test set, while maintaining the original class distribution
within each subset.

### A. Resampling and Performance Metrics

```
In the model building, a 5-fold cross-validation resampling technique was
employed. This method involves splitting the dataset into five folds, with four
folds utilized for model training and the remaining fold for testing. This process is
repeated five times, ensuring each fold serves as the test set. Given the
significant class imbalance in the dataset, the Kappa statistic was used as the
primary metric for model evaluation and selection. In this case, Kappa is suitable
for imbalanced datasets as it provides a more robust measure of model
performance by accounting for the agreement between predicted and actual
classifications while adjusting for agreement that could occur by chance.
```
