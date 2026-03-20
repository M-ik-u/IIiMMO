import pandas as pd
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('lab1/train.csv')
test = pd.read_csv('lab1/test.csv')

print(f"first 5 lines of train:"
      f"\n{train.head()}"
      f"\n info:"
      f"\n{train.info()}"
      f"\n statistics:"
      f"\n{train.describe()}")

print(f"last 5 lines of test:"
      f"\n{test.head()}"
      f"\n info:"
      f"\n{test.info()}"
      f"\n statistics:"
      f"\n{test.describe()}"
)

missing_train = train.isnull().sum()
print(f"missing train: \n{missing_train}")

"""
missing train: 
Age            177
Cabin          687
Embarked         2
"""

missing_test = test.isnull().sum()
print(f"missing test: \n{missing_test}")

"""
missing test: 
Age             86
Fare             1
Cabin          327
"""


train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

train.drop('Cabin', axis=1, inplace=True)

test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

test.drop('Cabin', axis=1, inplace=True)


num_col = ['Age', 'Fare', 'SibSp', 'Parch']

scaler = MinMaxScaler()

train[num_col] = scaler.fit_transform(train[num_col])

test[num_col] = scaler.transform(test[num_col])


category_col = ['Sex','Embarked','Pclass']

train = pd.get_dummies(train,columns = category_col, drop_first = True)
test = pd.get_dummies(test,columns = category_col, drop_first = True)

missing_col = set(train.columns) - set(test.columns)
for col in missing_col:
    test[col] = 0

test = test[test.columns]


train.to_csv("lab1/train_processed.csv", index=False)
test.to_csv("lab1/test_processed.csv", index=False)
