import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')

print(len(df))
df.head()

date_list = []
time_list = []
for dt in list(df['Datetime']):
  l = dt.split(' ')
  date_list.append(l[0])
  time_list.append(l[1])

df['Date'] = date_list
df['Time'] = time_list

train_df = df[['Date','Time','Count']].copy()

train_df.head()

date_dict = {}
for i, d in enumerate(train_df['Date'].unique()):
  date_dict[d] = i

time_dict = {}
for i, t in enumerate(train_df['Time'].unique()):
  time_dict[t] = i

train_df['Date'] = [date_dict[i] for i in list(train_df['Date'])]
train_df['Time'] = [time_dict[i] for i in list(train_df['Time'])]

train_df.head()

train_df.tail()

X, y = np.array(train_df[['Date','Time']]), np.array(train_df['Count'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

## **LINEAR REGRESSION**

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
print(f'MSE: {MSE}, RMSE: {np.sqrt(MSE)}')

## **RIDGE**

reg = Ridge(alpha=.5)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
print(f'MSE: {MSE}, RMSE: {np.sqrt(MSE)}')

## **LASSO**

reg = Lasso(alpha=.5)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
print(f'MSE: {MSE}, RMSE: {np.sqrt(MSE)}')