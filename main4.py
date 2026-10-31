import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
df_pt = pd.read_csv('pt.csv')
df_pt.columns = ['' ,'', '' , '' ,'','date','pt','','','']
df_pt = df_pt[['date','pt']]
df_pt =df_pt.drop([0, 1, 2])
df_pt['date'] = pd.to_datetime(df_pt['date'])
df_pt = df_pt.set_index('date')
df_pt['pt'] = pd.to_numeric(df_pt['pt'])
df_pt['pt'] = df_pt['pt']

df_qt = pd.read_csv('qt.csv')
df_qt.columns = ['' ,'', '' , '' ,'','date','qt','','','']
df_qt = df_qt[['date','qt']]
df_qt =df_qt.drop([0, 1, 2])
df_qt['date'] = pd.to_datetime(df_qt['date'])
df_qt = df_qt.set_index('date')
df_qt['qt'] = pd.to_numeric(df_qt['qt'])
df_qt['qt'] = df_qt['qt']
df = df_pt.join(df_qt, how='inner')

df_st = pd.read_csv('st.csv')
df_st.columns = ['' ,'', '' , '' ,'','date','st','','','']
df_st = df_st[['date','st']]
df_st =df_st.drop([0, 1, 2])
df_st['date'] = pd.to_datetime(df_st['date'])
df_st = df_st.set_index('date')
df_st['st'] = pd.to_numeric(df_st['st'])
df_st['st'] = df_st['st']
df = df.join(df_st, how='inner')
df.index = pd.to_datetime(df.index)
print(df)




train = df.loc[df.index < '2023-12-01']
test = df.loc[df.index >= '2023-12-01']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.legend(['Training Set', 'Test Set'])
plt.show()

#df.loc[(df.index > '01-08-2010') & (df.index < '01-15-2010')] .plot(figsize=(15, 5), title='Week Of Data')
#plt.show()


# Create features function
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['Day'] = df.index.day_name()
    df['Hour'] = df.index.hour
    df['pt_times_qt_lagged'] = df['pt'] * df['qt'].shift(1)
    rolling_window = 24
    df['pt_qt_difference'] = df['pt'] - df['qt']
    df['pt_qt_ratio'] = df['pt'] / df['qt']
    df['qt_lagged_1'] = df['qt'].shift(1)
    df['pt_lagged_1'] = df['pt'].shift(1)
    df['pt_times_qt'] = df['pt'] * df['qt']
    df['qt_roll_std'] = df['qt'].rolling(rolling_window).std()
    df['pt_times_qt_squared'] = (df['pt'] * df['qt']) ** 2
    df['pt_qt_difference_lagged'] = df['pt_lagged_1'] - df['qt_lagged_1']
    df['pt_rolling_cov'] = df['pt'].rolling(rolling_window).cov(df['qt'])
    df['pt_rolling_corr'] = df['pt'].rolling(rolling_window).corr(df['qt'])
    df['pt_roll_mean'] = df['pt'].rolling(rolling_window).mean()
    df['pt_rolling_max'] = df['pt'].rolling(rolling_window).max()
    df['pt_rolling_min'] = df['pt'].rolling(rolling_window).min()
    df['pt_rolling_skew'] = df['pt'].rolling(rolling_window).skew()
    df['pt_rolling_kurt'] = df['pt'].rolling(rolling_window).kurt()
    df['pt_rolling_q25'] = df['pt'].rolling(rolling_window).quantile(0.25)
    df['pt_rolling_q75'] = df['pt'].rolling(rolling_window).quantile(0.75)
    df['pt_rolling_iqr'] = df['pt_rolling_q75'] - df['pt_rolling_q25']
    df['st'] = np.sqrt(df['pt'] ** 2 + df['qt'] ** 2)
    df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')
    df = pd.get_dummies(df, columns=['Day', 'Day_Night'])

    return df

# Apply create_features function
df = create_features(df)



# Split data into train and test sets
train = df.loc[df.index < '2023-12-01']
test = df.loc[df.index >= '2023-12-01']

# Create XGBoost model for pt
reg_pt = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=8 ,
                       learning_rate=0.001)

# Define features and target for pt
FEATURES_PT = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
FEATURES_qt = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year','st','pt_qt_difference', 'pt_qt_ratio', 'pt_lagged_1', 'qt_lagged_1',
             'pt_times_qt', 'qt_roll_std', 'pt_times_qt_squared',
             'pt_qt_difference_lagged','pt_rolling_cov','pt_rolling_corr','pt_roll_mean','pt_rolling_max','pt_rolling_skew','pt_rolling_kurt','pt_rolling_q25','pt_rolling_q75','pt_rolling_iqr']
FEATURES_PT += [col for col in train.columns if col.startswith('Day_')]
TARGET_PT = 'pt'

X_train_pt = train[FEATURES_PT]
y_train_pt = train[TARGET_PT]
X_test_pt = test[FEATURES_PT]

# Train the model for pt
reg_pt.fit(X_train_pt, y_train_pt,
        eval_set=[(X_train_pt, y_train_pt)],
        verbose=100)

# Make predictions for pt
test['pt_prediction'] = reg_pt.predict(X_test_pt)
train['pt_prediction'] = reg_pt.predict(X_train_pt)
# Create XGBoost model for qt
reg_qt = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=10000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=15,
                       learning_rate=0.001)

# Define features and target for qt
FEATURES_QT = FEATURES_qt + ['pt_prediction']  # Include pt_prediction as a feature
TARGET_QT = 'qt'

X_train_qt = train[FEATURES_QT]
y_train_qt = train[TARGET_QT]
X_test_qt = test[FEATURES_QT]
X_train_qt = X_train_qt.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_qt = X_test_qt.replace([np.inf, -np.inf], np.nan).fillna(0)
# Train the model for qt
reg_qt.fit(X_train_qt, y_train_qt,
        eval_set=[(X_train_qt, y_train_qt)],
        verbose=100)

# Make predictions for qt
test['qt_prediction'] = reg_qt.predict(X_test_qt)

pt_mse = mean_squared_error(test['pt'], test['pt_prediction'])
qt_mse = mean_squared_error(test['qt'], test['qt_prediction'])


# Visualize predictions
plt.figure(figsize=(15, 5))
plt.plot(test.index, test['pt'], label='Actual Pt', color='blue')
plt.plot(test.index, test['pt_prediction'], label='Predicted Pt', color='red')
plt.title('Active Power Prediction (MSE: {:.2f})'.format(pt_mse))
plt.xlabel('Date')
plt.ylabel('Active Power')
plt.legend()
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(test.index, test['qt'], label='Actual Qt', color='blue')
plt.plot(test.index, test['qt_prediction'], label='Predicted Qt', color='red')
plt.title('Reactive Power Prediction (MSE: {:.2f})'.format(qt_mse))
plt.xlabel('Date')
plt.ylabel('Reactive Power')
plt.legend()
plt.show()

# Step 1: Create Features for the Future Period
# Define the date range for future predictions
future_start_date = df.index[-1] + pd.DateOffset(days=1)
future_end_date = future_start_date + pd.DateOffset(months=3)  # Adjust as needed
future_dates = pd.date_range(future_start_date, future_end_date, freq='1t')  # Adjust frequency as needed

# Create a DataFrame with future dates
future_df = pd.DataFrame(index=future_dates)
future_df['pt'] = np.nan  # Placeholder for predicted pt
future_df['qt'] = np.nan
# Apply the create_features function to add relevant features
future_df = create_features(future_df)

# Step 2: Predict Active Power (pt) for the Future Period
# Define features for pt prediction
FEATURES_PT = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
FEATURES_PT += [col for col in future_df.columns if col.startswith('Day_')]

# Predict pt for the future period
future_pt_prediction = reg_pt.predict(future_df[FEATURES_PT])

# Add the predicted pt to the future DataFrame
future_df['pt_prediction'] = future_pt_prediction

# Step 3: Use Predicted Active Power for Reactive Power (qt) Prediction
# Define features for qt prediction
FEATURES_QT = FEATURES_qt + ['pt_prediction']

# Predict qt for the future period
future_qt_prediction = reg_qt.predict(future_df[FEATURES_QT])

# Add the predicted qt to the future DataFrame
future_df['qt_prediction'] = future_qt_prediction
future_df.to_csv('forecast_results1.csv', index=False)
# Plot predicted values of active power (pt) for the future period
plt.figure(figsize=(15, 5))
plt.plot(future_df.index, future_df['pt_prediction'], label='Predicted Pt', color='red')
plt.title('Future Prediction of Active Power (Pt)')
plt.xlabel('Date')
plt.ylabel('Active Power')
plt.legend()
plt.show()

# Plot predicted values of reactive power (qt) for the future period
plt.figure(figsize=(15, 5))
plt.plot(future_df.index, future_df['qt_prediction'], label='Predicted Qt', color='blue')
plt.title('Future Prediction of Reactive Power (Qt)')
plt.xlabel('Date')
plt.ylabel('Reactive Power')
plt.legend()
plt.show()
