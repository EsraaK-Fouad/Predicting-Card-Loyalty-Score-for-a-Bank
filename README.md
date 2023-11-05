# Predicting-Card-Loyalty-Score-for-a-Bank
 The main objective of this project is to predict user loyalty scores based on customer behaviour, information about merchants, and other significant features using regression analysis.

# Background
This project involves analyzing a dataset provided by a bank in Brazil, which includes information on bank card transactions and user loyalty. The primary objective of this analysis is to predict a loyalty score for each card ID represented in the "userscore.csv" file. The dataset consists of four main files: "userscore.csv," "merchants.csv," "historical transactions.csv," and "new merchant transactions.csv." Each of these files contains specific information that is essential for this analysis.
# Data Files :
## columns in merchant.csv
- merchant_id: Unique merchant identifier
- merchant_group_id: Merchant group (anonymized )
- merchant_category_id: Unique identifier for merchant category (anonymized )
- subsector_id: Merchant category group (anonymized )
- numerical_1: anonymized measure
- numerical_2: anonymized measure
- category_1: anonymized category
- most_recent_sales_range: Range of revenue (monetary units) in last active month --> A > B > C > D > E
- most_recent_purchases_range: Range of quantity of transactions in last active month --> A > B > C > D > E
- avg_sales_lag3: Monthly average of revenue in last 3 months divided by revenue in last active month
- avg_purchases_lag3: Monthly average of transactions in last 3 months divided by transactions in last active month
- active_months_lag3: Quantity of active months within last 3 months
- avg_sales_lag6: Monthly average of revenue in last 6 months divided by revenue in last active month
- avg_purchases_lag6: Monthly average of transactions in last 6 months divided by transactions in last active month
- active_months_lag6: Quantity of active months within last 6 months
- avg_sales_lag12: Monthly average of revenue in last 12 months divided by revenue in last active month
- avg_purchases_lag12: Monthly average of transactions in last 12 months divided by transactions in last active month
- active_months_lag12: Quantity of active months within last 12 months
- category_4: anonymized category
- city_id City: identifier (anonymized )
- state_id: State identifier (anonymized )
- category_2: anonymized category

## columns in new_merchant_period.csv
- card_id: Card identifier
- month_lag: month lag to reference date
- purchase_date: Purchase date
- authorized_flag: Y' if approved, 'N' if denied
- category_3: anonymized category
- installments: number of installments of purchase
- category_1: anonymized category
- merchant_category_id: Merchant category identifier (anonymized )
- subsector_id: Merchant category group identifier (anonymized )
- merchant_id: Merchant identifier (anonymized)
- purchase_amount: Normalized purchase amount
- city_id: City identifier (anonymized )
- state_id: State identifier (anonymized )
- category_2: anonymized category

## columns in history.csv
- card_id: Card identifier
- month_lag: month lag to reference date
- purchase_date: Purchase date
- authorized_flag: Y' if approved, 'N' if denied
- category_3: anonymized category
- installments: number of installments of purchase
- category_1: anonymized category
- merchant_category_id: Merchant category identifier (anonymized )
- subsector_id: Merchant category group identifier (anonymized )
- merchant_id: Merchant identifier (anonymized)
- purchase_amount: Normalized purchase amount
- city_id: City identifier (anonymized )
- state_id: State identifier (anonymized )
- category_2: anonymized category

## columns in userscore.csv
- card_id: Unique card identifier
- first_active_mont: 'YYYY-MM', month of first purchase
- feature_1: Anonymized card categorical feature
- feature_2: Anonymized card categorical feature
- feature_3: Anonymized card categorical feature
- target: Loyalty numerical score calculated 2 months after historical and evaluation period


# Project Objectives:
The main objectives of this project are as follows:

1) Data Quality Analysis: Evaluate the quality of the dataset, identify missing data, outliers, and inconsistencies, and perform necessary data preprocessing.

2) Statistical Analysis: Conduct statistical analysis on the dataset to gain insights into card loyalty and merchant information. This analysis may include summary statistics, distribution plots, and correlation analysis.

3) Regression Analysis: Develop a regression model to predict the loyalty score for each card ID. Use features from the various data files to train and test the model.

4) Model Evaluation: Assess the performance of the regression model using relevant evaluation metrics. Ensure the model's ability to accurately predict card loyalty scores.

# Summary : 
Overall, this project aims to leverage the provided dataset to gain valuable insights into customer loyalty for the bank's cardholders in Brazil and to develop a predictive model that can estimate loyalty scores based on various features and transaction data.