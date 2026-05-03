import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, accuracy_score, auc, mean_absolute_error, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

df = pd.read_csv("lab1/winequality_processed.csv")
print(df.head())

cols = [_ for _ in df.columns if _ not in ['quality','quality_category_high']]

X = df[cols]
y_reg = df[['quality']]

X_train,X_test,y_train,y_test = train_test_split(X,y_reg,test_size=0.3,random_state=42)

reg_model = DecisionTreeRegressor(random_state=42)
reg_model.fit(X_train,y_train)

y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'DecisionTreeRegressor'
      f'\nMSE: {mse}'
      f'\nRMSE: {rmse}'
      f'\nMAE: {mae}')



#3

y_cls = df["quality_category_high"]
print(y_cls.value_counts())

X_train_c, X_test_c, y_train_c,y_test_c = train_test_split(X,y_cls,test_size=0.3,
                                                           random_state=42,stratify=y_cls)

model = DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

model.fit(X_train_c,y_train_c)

y_pred_c = model.predict(X_test_c)
y_proba_c = model.predict_proba(X_test_c)[:,1]

accuracy = accuracy_score(y_test_c,y_pred_c)
print("Accuracy: ",accuracy)


#4

fpr, tpr, thresholds = metrics.roc_curve(y_test_c,y_proba_c)
roc_auc = auc(fpr, tpr)

print("ROC AUC: ",roc_auc)

plt.figure()
plt.plot(fpr,tpr,marker='o',label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.grid(True)
plt.title('ROC Curve')
plt.savefig("lab3/roc_curve.png")

