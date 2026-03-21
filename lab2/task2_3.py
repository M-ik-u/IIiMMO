from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


df = pd.read_csv("lab1/train_processed.csv")

'''
ValueError: could not convert string to float: 'Partner, Mr. Austen'
возникла такая ошибка, поэтому дропаем из фрейма Name and Ticket
так как они нам  не нужны и, прогуглив, выснил, что при применении
к подобным данным OHE будет переобучение т.к создаться много-много столбцов
'''

df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)

X_reg = df.drop(['Age','Survived'], axis=1)
y_reg = df['Age']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg,
                                                                    test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_reg_train, y_reg_train)
y_reg_predict = lin_reg.predict(X_reg_test)

mse = mean_squared_error(y_reg_test, y_reg_predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_predict)

print(f'Linear Reggresion'
      f'\nMSE: {mse}\t '
      f'\nRMSE: {rmse}'
      f'\nMAE: {mae}')

#L1
lasso_model = Lasso()
lasso_model.fit(X_reg_train, y_reg_train)
y_lasso_predict = lasso_model.predict(X_reg_test)

mse_lasso = mean_squared_error(y_reg_test, y_lasso_predict)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_reg_test, y_lasso_predict)

print(f'Lasso Regression'
      f'\nMSE: {mse_lasso}'
      f'\nRMSE: {rmse_lasso}'
      f'\nMAE: {mae_lasso}')

#L2
ridge_model = Ridge()
ridge_model.fit(X_reg_train, y_reg_train)
y_ridge_predict = ridge_model.predict(X_reg_test)

mse_ridge = mean_squared_error(y_reg_test, y_ridge_predict)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_reg_test, y_ridge_predict)

print(f'Ridge Regression'
      f'\nMSE: {mse_ridge}'
      f'\nRMSE: {rmse_ridge}'
      f'\nMAE: {mae_ridge}')

#ElasticNet
elastic_net = ElasticNet()
elastic_net.fit(X_reg_train, y_reg_train)
y_elastic_predict = elastic_net.predict(X_reg_test)

mse_elastic = mean_squared_error(y_reg_test, y_elastic_predict)
rmse_elastic = np.sqrt(mse_elastic)
mae_elastic = mean_absolute_error(y_reg_test, y_elastic_predict)

print(f'Elastic Net'
      f'\nMSE: {mse_elastic}'
      f'\nRMSE: {rmse_elastic}'
      f'\nMAE: {mae_elastic}'
)

print("\n" + "="*80)
print(f"{'Metric':<15} {'Linear':<15} {'Lasso':<15} {'Ridge':<15} {'ElasticNet':<15}")
print("="*80)
print(f"{'MSE':<15} {mse:<15.6f} {mse_lasso:<15.6f} {mse_ridge:<15.6f} {mse_elastic:<15.6f}")
print(f"{'RMSE':<15} {rmse:<15.6f} {rmse_lasso:<15.6f} {rmse_ridge:<15.6f} {rmse_elastic:<15.6f}")
print(f"{'MAE':<15} {mae:<15.6f} {mae_lasso:<15.6f} {mae_ridge:<15.6f} {mae_elastic:<15.6f}")
print("="*80)


