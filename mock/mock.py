import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/drive/MyDrive/lp1/data_da_3.csv')
df.head()
# REFER OUTPUT 1

df.describe()
# REFER OUTPUT 2
# VERY FEW NUMERICAL COLUMNS, MUST CONVERT THE REST

# GETTING TOTAL NULL VALUES
for col in df.columns:
  print(f'{col}: {len(df[col].unique())}, NaN count: {df[col].isna().sum()}')

"""
Item_Identifier: 1559, NaN count: 0
Item_Weight: 416, NaN count: 1463
Item_Fat_Content: 5, NaN count: 0
Item_Visibility: 7880, NaN count: 0
Item_Type: 16, NaN count: 0
Item_MRP: 5938, NaN count: 0
Outlet_Identifier: 10, NaN count: 0
Outlet_Establishment_Year: 9, NaN count: 0
Outlet_Size: 4, NaN count: 2410
Outlet_Location_Type: 3, NaN count: 0
Outlet_Type: 4, NaN count: 0
Item_Outlet_Sales: 3493, NaN count: 0
"""

# GETTING OUTLET DISTRIBUTION
df['Outlet_Size'].value_counts()

"""
Medium    2793
Small     2388
High       932
Name: Outlet_Size, dtype: int64
"""

# FILLING MISSING WEIGHTS BY MEAN
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x = np.expand_dims(df['Item_Weight'], axis = -1)
df['Item_Weight'] = np.reshape(imp_mean.fit_transform(x), (len(x)))

# FILLING MISSING OUTLETS BY MEDIUM
x = df['Outlet_Size']
for i in range(len(x)):
  if x[i] is np.nan: x[i] = 'Medium'
df['Outlet_Size'] = x

# CHECKING NEW NULL VALUES
for col in df.columns:
  print(f'{col}: {len(df[col].unique())}, NaN count: {df[col].isna().sum()}')

"""
Item_Identifier: 1559, NaN count: 0
Item_Weight: 416, NaN count: 0
Item_Fat_Content: 5, NaN count: 0
Item_Visibility: 7880, NaN count: 0
Item_Type: 16, NaN count: 0
Item_MRP: 5938, NaN count: 0
Outlet_Identifier: 10, NaN count: 0
Outlet_Establishment_Year: 9, NaN count: 0
Outlet_Size: 3, NaN count: 0
Outlet_Location_Type: 3, NaN count: 0
Outlet_Type: 4, NaN count: 0
Item_Outlet_Sales: 3493, NaN count: 0
"""

# DROPPING UNNECESSARY AND LABEL COLUMNS
X = df.drop(['Item_Identifier', 'Item_Outlet_Sales'], axis=1)
# GETTING LABEL COLUMN
y = df['Item_Outlet_Sales']

# GETTING NUMERICAL COLUMNS AND NORMALIZING VALUES
linear_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP']

X_scaled = X.copy()
scaler = MinMaxScaler()
for col in X.columns:
  if col in linear_cols:
    x = np.expand_dims(X[col], axis=-1)
    X_scaled[col] = np.reshape(scaler.fit_transform(x), (len(x)))
  # CONVERTING CATEGORICAL TO ONE-HOT
  else:
    x = X[col]
    X_scaled.drop(col, axis=1, inplace=True)
    X_scaled = X_scaled.join(pd.get_dummies(x, prefix=col))

X_scaled.head()
# REFER OUTPUT 3

# SPLITTING DATA INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# USING LINEAR REGRESSION MODEL
reg = LinearRegression().fit(X_train, y_train)

preds = reg.predict(X_test)

for i in range(len(preds)):
  preds[i] = max(0, preds[i])

# GETTING LOSS OF TEST SET
mse = mean_squared_error(y_test, preds)
print(f'MSE: {mse}, RMSE: {np.sqrt(mse)}')

"""
MSE: 1133465.5367624972, RMSE: 1064.6433847831381
"""