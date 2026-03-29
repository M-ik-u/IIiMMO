import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("lab1/winequality_processed.csv")

print(f"Size {df.head()}"
      f"\n Columns: \n{df.columns}")

X_clf = df.drop(["quality","quality_category_high"], axis=1)
y_reg = df["quality_category_high"]

X_train, X_val, y_train, y_val = train_test_split(
    X_clf, y_reg, test_size=0.2, random_state=42
)


logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=["Low", "High"]))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))