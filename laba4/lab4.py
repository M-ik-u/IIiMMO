import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("lab1/winequality_processed.csv")

cols = [_ for _ in df.columns if _ not in ["quality","quality_category_high"]]
X = df[cols]
y = df["quality_category_high"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42,stratify=y)


#2

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)
rf2 = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf2.fit(X_train, y_train)

print(f"RANDOM FOREST MODEL\n"
      f"OOB ACCURACY: {rf.oob_score_:.3f}"
      f"\nOOB ERROR: {1 - rf.oob_score_:.3f}")
print(f"RANDOM FOREST MODEL 2\n"
      f"OOB ACCURACY: {rf2.oob_score_:.3f}"
      f"\nOOB ERROR: {1 - rf2.oob_score_:.3f}")


#3

base_model = DecisionTreeClassifier(max_depth=1, random_state=42)

ada = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)
y_proba_ada = ada.predict_proba(X_test)[:,1]

acc_ada = accuracy_score(y_test, y_pred_ada)
fpr_ada,trp_ada,thr_ada = roc_curve(y_test, y_proba_ada)
auc_ada = auc(fpr_ada, trp_ada)
print(f"ADA BOOST\n"
      f"Accuracy: {acc_ada:.3f}"
      f"\nAUC: {auc_ada:.3f}")

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:,1]

acc_gb = accuracy_score(y_test, y_pred_gb)
fpr_gb,trp_gb,thr_gb = roc_curve(y_test, y_proba_gb)
auc_gb = auc(fpr_gb, trp_gb)

print(f"GB BOOST\n"
      f"Accuracy: {acc_gb:.3f}"
      f"\nAUC: {auc_gb:.3f}"
)

plt.figure()
plt.plot(fpr_ada,trp_ada, marker = "*",label="ADA")
plt.plot(fpr_gb,trp_gb, marker = "^",label="GB")

plt.plot([0,1],[0,1], "k--", label = "Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc = "lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab4.png")
