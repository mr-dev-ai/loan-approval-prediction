# Loan Approval Prediction 💳

## About This Project

This project predicts whether a loan application will be Approved or Rejected using Machine Learning.

I built this project to understand how classification works in real-world problems. I used Logistic Regression to train the model and tested its accuracy.

---

## Dataset Details

The dataset includes:

- no_of_dependents
- education
- self_employed
- income_annum
- loan_amount
- loan_term
- cibil_score
- residential_assets_value
- commercial_assets_value
- luxury_assets_value
- bank_asset_value
- loan_status (Approved / Rejected)

I converted the loan_status column into:
- Approved → 1
- Rejected → 0

---

## What I Did in This Project

- Loaded and cleaned the dataset using Pandas
- Handled categorical features using one-hot encoding
- Scaled numerical features using StandardScaler
- Split data into training and testing sets
- Trained a Logistic Regression model
- Evaluated the model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Predicted loan approval for a new applicant
- Visualized CIBIL score vs loan approval

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## How to Run

1. Install required libraries:

   pip install -r requirements.txt

2. Run the file:

   python loan_prediction.py

---

## What I Learned

- How classification problems work
- Data preprocessing steps
- Feature scaling
- Model evaluation using different metrics
- Applying ML in a finance-related problem

This project helped me understand how machine learning can be used in real-world applications.
