import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("loan_data.csv")

print("Dataset Preview:")
print(df.head())

X = df[["Income", "Loan_Amount", "Credit_Score", "Employment_Status"]]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

new_applicant = [[42000, 160000, 710, 1]]
prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

plt.scatter(df["Credit_Score"], df["Loan_Status"])
plt.xlabel("Credit Score")
plt.ylabel("Loan Status (0 = Rejected, 1 = Approved)")
plt.title("Credit Score vs Loan Approval")
plt.show()
