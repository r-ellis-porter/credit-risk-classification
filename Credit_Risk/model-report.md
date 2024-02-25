# Credit Risk Classification Model Report

## Overview of the Analysis

* The purpose of this analysis is to help select the best machine learning model to use to predict whether a loan applicant is high risk or not.
* The dataset used to train and test the machine learning models included information about the loan size, interest rate, borrower income, debt to income ratio, number of accounts, number of derogatory marks, total debt, and loan status.
* The target variable 'loan status' was strongly imbalanced with 97% of the data being 'healthy' and 34% being 'high risk'.
* First, I separated the 'loan status' column from the other columns to create the labels and features sets. After next splitting the data into training and testing sets, I fit the LogisticRegression model using the training data and evaluated the model's performance using the testing data to generate the accuracy score, confusion matrix, and classification report.
* After evaluating the initial model, I used RandomOverSampler to balance the target value counts by resampling our initial imbalanced dataset. I then evaluated the model's performance once more using the resampled data.

## Results

* Machine Learning Model 1:
  (imbalanced)

  * Accuracy score: 0.9531209979209979
  * Healthy loan: 
        * Precision: 1.00
        * Recall: 1.00
  * High risk loan:
        * Precision: 0.88
        * Recall: 0.91

* Machine Learning Model 2:
  (resampled)

  * Accuracy score: 0.9947758409296871
  * Healthy loan: 
        * Precision: 1.00
        * Recall: 0.99
  * High risk loan:
        * Precision: 0.99
        * Recall: 1.00

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* ML model 2 performed better than ML model 1 with a higher accuracy score and better predictions for high risk loans. Balancing the dataset helped improve the second model's ability to predict high risk loans.
* Since the purpose of this analysis was to help identify high risk loans and model 2 outperformed model 1 in predicting high risk loans, I recommend using model 2.
