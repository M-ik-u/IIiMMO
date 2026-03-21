import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('lab1/train_processed.csv')
df_test = pd.read_csv('lab1/test_processed.csv')

df_train.drop('Name', axis=1, inplace=True)
df_train.drop('Ticket', axis=1, inplace=True)

df_test.drop('Name', axis=1, inplace=True)
df_test.drop('Ticket', axis=1, inplace=True)

y = df_train['Survived']
X = df_train.drop('Survived', axis=1)

df_test = df_test.drop('Survived', axis=1)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=["Not Survived", "Survived"]))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

pred_test = logreg.predict(df_test)

df_result = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': pred_test
})

df_result.to_csv('lab2/result.csv', index=False)