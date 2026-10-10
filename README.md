# Loan_Approval_Prediction_Analysis
🏦 Loan Approval Prediction Model

📌 Project Overview

This project builds a supervised machine learning pipeline to predict whether a borrower's loan application will be approved or rejected based on applicant features. The primary focus is on handling real-world data challenges such as missing values, categorical encoding, class imbalance, and translating model metrics into actionable business decisions.

🎯 Objectives

Data Preprocessing: Build robust pipelines using scikit-learn to handle missing values, scale numerical features, and One-Hot Encode categorical variables.

Class Imbalance: Apply SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model doesn't become biased toward the majority class (typically approved loans).

Model Comparison: Train and compare Logistic Regression and Random Forest classifiers.

Business Evaluation: Evaluate models using Precision, Recall, F1-Score, and ROC-AUC, and provide recommendations on deployment thresholds to minimize financial risk.

📊 Dataset

The analysis uses the Loan Approval Prediction dataset from Kaggle.

Source: Kaggle - Loan Approval Prediction by Bhanu Pratap Biswas

🛠️ Tech Stack & Libraries

Language: Python

Environment: Jupyter Notebook / Google Colab

Libraries:

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn, imbalanced-learn (SMOTE)

Visualization: matplotlib, seaborn

📈 Visualizations Included

Approval Distributions: EDA charts showing overall approval rates and the critical impact of Credit History on loan success.

ROC-AUC Curve: A comparative performance curve for Logistic Regression vs. Random Forest.

Feature Importance: A bar chart visualizing the top 10 borrower attributes driving the Random Forest model's decisions.

💼 Business Insights & Recommendations

Precision vs. Recall Trade-off: In lending, False Positives (approving a bad loan) cost capital, while False Negatives (rejecting a good loan) cost opportunity. The models were evaluated with this financial trade-off in mind.

Key Drivers: Credit History and Applicant Income consistently rank as the top determining factors for approval.

Deployment Strategy: To minimize risk and defaults, it is recommended to raise the classification decision threshold from the default 0.50 to 0.65 or 0.70. This ensures a more conservative approval process, protecting the institution's capital at the expense of a slight drop in total loan volume.

📂 Repository Structure

loan_approval_pipeline.ipynb: The main notebook containing the preprocessing, SMOTE application, model training, and visualizations.

README.md: Project documentation.

💻 How to Run This Project

Clone this repository:

git clone https://github.com/your-username/loan-approval-prediction.git


Download the dataset from Kaggle and place it in the project root folder.

Install required packages (especially imbalanced-learn):

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn


Open the Jupyter Notebook or Colab file and run the cells sequentially.
