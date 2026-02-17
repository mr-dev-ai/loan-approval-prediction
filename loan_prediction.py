import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\Manivel R\Documents\GitHub\python-basics-practice\loan-approval-prediction\loan_data.csv")
df.columns = df.columns.str.strip()

print("Dataset Preview:")
print(df.head())

df["loan_status"] = df["loan_status"].str.strip().str.capitalize()
y = df["loan_status"].map({"Approved": 1, "Rejected": 0})

X = df[[
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]]

X = pd.get_dummies(X, columns=["education", "self_employed"], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_applicant = pd.DataFrame(
    [[2, "Graduate", "No", 9600000, 29900000, 12, 778, 2400000, 17600000, 22700000, 8000000]],
    columns=[
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]
)
new_applicant = pd.get_dummies(new_applicant, columns=["education", "self_employed"], drop_first=True)
new_applicant = new_applicant.reindex(columns=X.columns, fill_value=0)
new_applicant_scaled = scaler.transform(new_applicant)
prediction = model.predict(new_applicant_scaled)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

plt.scatter(df["cibil_score"], y, c=y, cmap="bwr")
plt.xlabel("CIBIL Score")
plt.ylabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.title("CIBIL Score vs Loan Approval")
plt.show()