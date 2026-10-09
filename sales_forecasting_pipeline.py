python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# 1. Load raw data extracted via SQL script
# (Assumes a structured CSV dataset containing dates and baseline sales totals)
df = pd.read_csv('simulated_sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.groupby('date')['sales'].sum().reset_index()

# 2. Feature Engineering (Extracting Time-Series Characteristics)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Target Variables (Lags 1 through 7 to track cyclical auto-correlation)
for i in range(1, 8):
    df[f'sales_lag_{i}'] = df['sales'].shift(i)

# Drop missing rows resulting from shifting window values
df = df.dropna().reset_index(drop=True)

# 3. Separate Features and Target Array
features = ['year', 'month', 'day', 'dayofweek', 'is_weekend', 
            'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4', 
            'sales_lag_5', 'sales_lag_6', 'sales_lag_7']
X = df[features]
y = df['sales']

# Time-based train/test splitting to mimic production corporate parameters (80/20 Split)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 4. Train Model using XGBoost Gradient Boosting Architecture
model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate Forecast Performance Against Test Set
predictions = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"XGBoost Time-Series Forecast Performance Metrics:")
print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} units")
