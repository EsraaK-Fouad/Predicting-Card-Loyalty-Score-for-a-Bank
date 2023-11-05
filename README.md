# Predicting-Card-Loyalty-Score-for-a-Bank
 The primary objective of this project is to predict user loyalty scores based on customer behaviour, information about merchants, and other significant features.

 Topics in Data Analytics S23 - Assignment 1
# Background
The data is a bank card transaction and user (card) loyalty analysis dataset provided by a bank in Brazil. This assignment aims to practice on topics covered in lectures 1-3, i.e., data quality analysis, statistical analysis, and regression analysis given the dataset. The overall purpose of the analysis is to predict a loyalty score for each card id represented in userscore.csv. The dataset contains four files. userscore.csv contain card ids and information about the card itself - the first month the card was active. It also contains the predict/analysis target, i.e., score, which is a score calculated by the bank, indicating the loyalty of each card owner. Three features are provided, all of which are anonymized card categorical features. merchants.csv contains aggregate information for each merchant id represented in the data set. merchants can be joined with the transaction sets to provide additional merchant-level information. The historical transactions.csv and new merchant transactions.csv files contain information about each card’s transactions. historical transactions.csv contains up to 3 months’ worth of transactions for every card at any of the provided merchant ids. new merchant transactions.csv contains the transactions at new merchants (merchant ids that this particular card id has not yet visited) over two months. historical transactions.csv and new merchant transactions.csv are designed to be joined with userscore.csv and merchants.csv. They contain information about transactions for each card, as described above. Given the dataset, answer the following questions. Your submission will be judged based on your answer’s relevance and your regression model’s performance.

# columns in merchant.csv
merchant_id: Unique merchant identifier
merchant_group_id: Merchant group (anonymized )
merchant_category_id: Unique identifier for merchant category (anonymized )
subsector_id: Merchant category group (anonymized )
numerical_1: anonymized measure
numerical_2: anonymized measure
category_1: anonymized category
most_recent_sales_range: Range of revenue (monetary units) in last active month --> A > B > C > D > E
most_recent_purchases_range: Range of quantity of transactions in last active month --> A > B > C > D > E
avg_sales_lag3: Monthly average of revenue in last 3 months divided by revenue in last active month
avg_purchases_lag3: Monthly average of transactions in last 3 months divided by transactions in last active month
active_months_lag3: Quantity of active months within last 3 months
avg_sales_lag6: Monthly average of revenue in last 6 months divided by revenue in last active month
avg_purchases_lag6: Monthly average of transactions in last 6 months divided by transactions in last active month
active_months_lag6: Quantity of active months within last 6 months
avg_sales_lag12: Monthly average of revenue in last 12 months divided by revenue in last active month
avg_purchases_lag12: Monthly average of transactions in last 12 months divided by transactions in last active month
active_months_lag12: Quantity of active months within last 12 months
category_4: anonymized category
city_id City: identifier (anonymized )
state_id: State identifier (anonymized )
category_2: anonymized category

# columns in new_merchant_period.csv
card_id: Card identifier
month_lag: month lag to reference date
purchase_date: Purchase date
authorized_flag: Y' if approved, 'N' if denied
category_3: anonymized category
installments: number of installments of purchase
category_1: anonymized category
merchant_category_id: Merchant category identifier (anonymized )
subsector_id: Merchant category group identifier (anonymized )
merchant_id: Merchant identifier (anonymized)
purchase_amount: Normalized purchase amount
city_id: City identifier (anonymized )
state_id: State identifier (anonymized )
category_2: anonymized category

# columns in history.csv
card_id: Card identifier
month_lag: month lag to reference date
purchase_date: Purchase date
authorized_flag: Y' if approved, 'N' if denied
category_3: anonymized category
installments: number of installments of purchase
category_1: anonymized category
merchant_category_id: Merchant category identifier (anonymized )
subsector_id: Merchant category group identifier (anonymized )
merchant_id: Merchant identifier (anonymized)
purchase_amount: Normalized purchase amount
city_id: City identifier (anonymized )
state_id: State identifier (anonymized )
category_2: anonymized category

# columns in userscore.csv
card_id: Unique card identifier
first_active_mont: 'YYYY-MM', month of first purchase
feature_1: Anonymized card categorical feature
feature_2: Anonymized card categorical feature
feature_3: Anonymized card categorical feature
target: Loyalty numerical score calculated 2 months after historical and evaluation period
