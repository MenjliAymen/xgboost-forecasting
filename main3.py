import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Set plot styles
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Load and preprocess the data
df = pd.read_csv('PT min.csv')
df.columns = ['', '', '', '', '', 'date', 'value', '', '', '']
df = df[['date', 'value']]
df = df.drop([0, 1, 2])
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df['value'] = pd.to_numeric(df['value'])

# Plot the full data
df.plot(style='-', figsize=(15, 5), color=color_pal[9], title='PT Data')
plt.show()

# Train-test split
train = df.loc[df.index < '2023-10-01']
test = df.loc[df.index >= '2023-10-01']

# Plot train-test split
fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.legend(['Training Set', 'Test Set'])
plt.show()


# Feature creation function
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['Hour'] = df.index.hour
    df.to_csv('df.csv', index=True)
    return df


# Apply feature creation to data
train = create_features(train)
test = create_features(test)

# Define features and target
FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'value'

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

# Train the XGBoost model with TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

reg = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    n_estimators=5000,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    max_depth=12,
    learning_rate=0.001
)

# Cross-validation performance
train_rmse = []
test_rmse = []
for train_index, val_index in tscv.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    reg.fit(X_train_fold, y_train_fold, eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            verbose=False)

    train_rmse.append(np.sqrt(mean_squared_error(y_train_fold, reg.predict(X_train_fold))))
    test_rmse.append(np.sqrt(mean_squared_error(y_val_fold, reg.predict(X_val_fold))))

# Average RMSE across splits
print(f"Average Train RMSE: {np.mean(train_rmse):.4f}")
print(f"Average Test RMSE: {np.mean(test_rmse):.4f}")

# RMSE plots for cross-validation
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', marker='o', color='blue')
plt.plot(range(1, len(test_rmse) + 1), test_rmse, label='Test RMSE', marker='o', color='red')
plt.title('RMSE for Each Fold (Cross-Validation)')
plt.xlabel('Fold Number')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

# Feature importance plot
fi = pd.DataFrame(data=reg.feature_importances_, index=reg.feature_names_in_, columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

# Generate predictions and evaluate on test set
test['prediction'] = reg.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, test['prediction'])
mse = mean_squared_error(y_test, test['prediction'])
rmse = np.sqrt(mse)

print(f"Test Set Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Plot actual vs predicted for test set
ax = test[['value']].plot(figsize=(15, 5), title='Actual vs Predicted - Test Set')
test['prediction'].plot(ax=ax, style='-')
plt.legend(['Truth Data', 'Predictions'])
plt.show()

# Future forecasting
future_start_date = df.index[-1] + pd.DateOffset(days=1)
future_end_date = future_start_date + pd.DateOffset(months=4) - pd.DateOffset(days=1)
future_dates = pd.date_range(future_start_date, future_end_date, freq='1h')
future_df = pd.DataFrame(index=future_dates)

future_df = create_features(future_df)
X_future = future_df[FEATURES]
future_df['prediction'] = reg.predict(X_future)

# Save future predictions to CSV
future_df.to_csv('forecast_results.csv', index=True)

# Plot historical data and future predictions
ax = df[['value']].plot(figsize=(15, 5), title='Historical Data and Future Predictions')
future_df['prediction'].plot(ax=ax, style='-', color='red')
plt.legend(['Historical Data', 'Future Predictions'])
plt.show()

# Print future forecast overview
print(future_df.head())
