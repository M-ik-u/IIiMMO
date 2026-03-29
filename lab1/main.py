import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('lab1/winequality-red.csv')


print(f"first 5 lines of train:"
      f"\n{df.head()}"
      f"\n info:"
      f"\n{df.info()}"
      f"\n statistics:"
      f"\n{df.describe()}")


missing_train = df.isnull().sum()
print(f"missing train: \n{missing_train}")

norm_cols = [col for col in df.columns if col != 'quality']

scaler = MinMaxScaler()

df[norm_cols] = scaler.fit_transform(df[norm_cols])

df['quality_category'] = pd.cut(df['quality'],
                                          bins=[0, 5, 10],
                                          labels=['low', 'high'])

df = pd.get_dummies(df, columns=['quality_category'], drop_first=True)

df.to_csv('lab1/winequality_processed.csv', index=False)
