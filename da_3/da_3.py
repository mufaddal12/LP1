import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/drive/MyDrive/data_da_3.csv')
df_test = pd.read_csv('/content/drive/MyDrive/test_da_3.csv')
df

df.describe()

for col in df.columns:
  print(f'{col}: {len(df[col].unique())}, NaN count: {df[col].isna().sum()}')

for col in df_test.columns:
  print(f'{col}: {len(df_test[col].unique())}, NaN count: {df_test[col].isna().sum()}')

df['Outlet_Size'].value_counts()

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x = np.expand_dims(df['Item_Weight'], axis = -1)
df['Item_Weight'] = np.reshape(imp_mean.fit_transform(x), (len(x)))

x = np.expand_dims(df_test['Item_Weight'], axis = -1)
df_test['Item_Weight'] = np.reshape(imp_mean.fit_transform(x), (len(x)))

x = df['Outlet_Size']
for i in range(len(x)):
  if x[i] is np.nan: x[i] = 'Medium'
df['Outlet_Size'] = x

x = df_test['Outlet_Size']
for i in range(len(x)):
  if x[i] is np.nan: x[i] = 'Medium'
df_test['Outlet_Size'] = x

for col in df.columns:
  print(f'{col}: {len(df[col].unique())}, NaN count: {df[col].isna().sum()}')

for col in df_test.columns:
  print(f'{col}: {len(df_test[col].unique())}, NaN count: {df_test[col].isna().sum()}')

X = df.drop(['Item_Identifier', 'Item_Outlet_Sales'], axis=1)
y = df['Item_Outlet_Sales']

X_sub = df_test.drop(['Item_Identifier'], axis=1)

linear_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP']

X_scaled = X.copy()
scaler = MinMaxScaler()
for col in X.columns:
  if col in linear_cols:
    x = np.expand_dims(X[col], axis=-1)
    X_scaled[col] = np.reshape(scaler.fit_transform(x), (len(x)))
  else:
    x = X[col]
    X_scaled.drop(col, axis=1, inplace=True)
    X_scaled = X_scaled.join(pd.get_dummies(x, prefix=col))

X_scaled

X_sub_scaled = X_sub.copy()
for col in X_sub.columns:
  if col in linear_cols:
    x_sub = np.expand_dims(X_sub[col], axis=-1)
    X_sub_scaled[col] = np.reshape(scaler.fit_transform(x_sub), (len(x_sub)))
  else:
    x_sub = X_sub[col]
    X_sub_scaled.drop(col, axis=1, inplace=True)
    X_sub_scaled = X_sub_scaled.join(pd.get_dummies(x_sub, prefix=col))

X_sub_scaled

x = np.expand_dims(y, axis=-1)
y_scaled = np.reshape(scaler.fit_transform(x), (len(x)))
scaler.data_min_, scaler.data_max_

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

## **SKLEARN SOLUTION**

reg = LinearRegression().fit(X_scaled, y)

preds = reg.predict(X_sub_scaled)

for i in range(len(preds)):
  preds[i] = max(0, preds[i])

mse = mean_squared_error(y_test, preds)
print(f'MSE: {mse}, RMSE: {np.sqrt(mse)}')

sub_df = df_test[['Item_Identifier', 'Outlet_Identifier']]

sub_df['Item_Outlet_Sales'] = preds

sub_df.to_csv('submission.csv', index=False)

## **NEURAL NET SOLUTION**

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

X_train = tf.convert_to_tensor(X_scaled)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y)
y_test = tf.convert_to_tensor(y_test)

model.fit(X_train, y_train, batch_size=128, epochs=100)

preds = model.predict(X_sub_scaled)

for i in range(len(preds)):
  preds[i] = max(0, preds[i])

sub_df = df_test[['Item_Identifier', 'Outlet_Identifier']]

sub_df['Item_Outlet_Sales'] = preds

sub_df.to_csv('submission.csv', index=False)

mse = mean_squared_error(y_test, preds)
print(f'MSE: {mse}, RMSE: {np.sqrt(mse)}')