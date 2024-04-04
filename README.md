# Credit Risk Classification Project

## Overview

This project aims to develop a machine learning model to predict the creditworthiness of borrowers using historical lending data from a peer-to-peer lending services company. The dataset includes various features such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, number of derogatory marks, total debt, and loan status (healthy or high-risk).

## Problem Statement

The primary objective is to build a model that can accurately classify loans as either healthy or high-risk based on the provided features. Due to the imbalanced nature of the dataset, where the majority of loans are labeled as healthy, the challenge is to develop a model that can effectively identify high-risk loans despite the class imbalance.

## Approach

1. **Data Preprocessing**
    - The dataset is first loaded into a Pandas DataFrame.
    - The features and labels are separated, with 'loan status' as the target variable.
    - The data is split into training and testing sets to train and evaluate the model.
    
2. **Model Development**
    - A Logistic Regression model is trained using the original imbalanced data.
    - Model performance metrics such as accuracy, precision, recall, and F1-score are calculated.
    - To address the imbalance issue, RandomOverSampler is employed to balance the dataset by resampling the minority class.
    - Another Logistic Regression model is trained using the resampled data, and its performance is evaluated.
    
3. **Evaluation**
    - The performance of both models is compared based on accuracy, precision, recall, and F1-score.
    - Recommendations are made regarding the selection of the best-performing model for predicting high-risk loans.

## Important Findings

**Model 1 (Imbalanced Data)**
- Accuracy score: 95.31%
- Precision for healthy loans: 100%
- Recall for healthy loans: 100%
- Precision for high-risk loans: 88%
- Recall for high-risk loans: 91%

**Model 2 (Resampled Data)**
- Accuracy score: 99.48%
- Precision for healthy loans: 100%
- Recall for healthy loans: 99%
- Precision for high-risk loans: 99%
- Recall for high-risk loans: 100%

## Summary

- The analysis highlights the effectiveness of balancing the dataset through resampling in improving model performance.
- Model 2, trained on the resampled data, outperforms Model 1 in accurately predicting high-risk loans.
- Considering the project's objective of identifying high-risk loans, Model 2 is recommended for deployment due to its superior performance.

## Technical Details

**Dependencies**
- Python 3.x
- Pandas
- NumPy
- scikit-learn
- imbalanced-learn

**How to Run the Code**
1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd credit-risk-classification`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Jupyter Notebook: `jupyter notebook credit_risk_classification.ipynb`
5. Follow the instructions in the notebook to execute the code cells and analyze the results.

## Conclusion

This project demonstrates the importance of addressing class imbalance in machine learning tasks, particularly in credit risk classification. By employing appropriate techniques such as resampling, it is possible to enhance model performance and make more accurate predictions, thereby aiding financial institutions in making informed lending decisions.
