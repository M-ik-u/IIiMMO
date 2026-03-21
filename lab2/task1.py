import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("lab2/train_processed.csv")

print(f"Size {df.head()}"
      f"\n Columns: \n{df.columns}")

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42, stratify=y)
