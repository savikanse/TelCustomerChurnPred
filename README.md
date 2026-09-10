# Telecom Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset. The project covers exploratory data analysis, data preprocessing, feature engineering, benchmarking of multiple classification models, and hyperparameter tuning.

## Project Overview

Customer churn prediction enables businesses to identify customers who are more likely to discontinue their services and understand the factors associated with churn.

This project analyzes **7,043 telecom customers** and evaluates multiple machine learning approaches for predicting customer churn.

Exploratory analysis highlights factors such as **contract type and customer tenure** as important characteristics associated with churn.

## Dataset

The dataset contains 7,043 customer records with information covering:

- Demographics
- Customer tenure
- Contract type
- Phone and internet services
- Online security and support services
- Payment method
- Monthly charges
- Total charges
- Churn status

The original dataset contains **21 columns**.

## Data Preprocessing

The preprocessing pipeline included:

- Removing the `customerID` identifier.
- Converting `TotalCharges` from string to numeric format.
- Identifying 11 records with missing `TotalCharges`.
- Removing customers with `tenure = 0`, corresponding to the missing `TotalCharges` records.
- Label encoding binary categorical variables.
- Standardizing numerical variables using `StandardScaler`.

## Exploratory Data Analysis

Plotly, Matplotlib, and Seaborn were used to investigate customer characteristics and their relationship with churn.

The analysis explored:

- Contract type
- Customer tenure
- Monthly charges
- Total charges
- Payment methods
- Internet services
- Technical support
- Customer demographics
- Other service-related variables

## Machine Learning

The project benchmarks **10 classification models**:

1. Logistic Regression
2. Linear SVM
3. Kernel SVM
4. K-Nearest Neighbours
5. Gaussian Naive Bayes
6. Decision Tree
7. Random Forest
8. AdaBoost
9. Gradient Boosting
10. Voting Classifier

### Model Comparison

Using 10-fold cross-validation:

| Model | ROC-AUC | Accuracy |
|---|---:|---:|
| Voting Classifier | 84.82% | 79.95% |
| Gradient Boosting | 84.62% | 79.36% |
| AdaBoost | 84.39% | 79.93% |
| Logistic Regression | 84.30% | 74.64% |
| Random Forest | 83.01% | 78.77% |
| Linear SVM | 82.94% | 79.07% |
| Gaussian Naive Bayes | 82.19% | 75.38% |
| Kernel SVM | 79.68% | 79.34% |
| KNN | 77.23% | 75.86% |
| Decision Tree | 65.85% | 72.94% |

## Hyperparameter Tuning

Gradient Boosting was further optimized using `GridSearchCV`.

The search evaluated:

- 20 parameter combinations
- 10-fold cross-validation
- **200 total model fits**
- ROC-AUC as the optimization metric

The best configuration was:

```text
GradientBoostingClassifier(
    max_depth=1,
    n_estimators=300
)
