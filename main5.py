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
df = pd.read_csv('qt.csv')
df.columns = ['' ,'', '' , '' ,'','date','qt','','','']
df = df[['date','qt']]
df =df.drop([0, 1, 2])
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df['qt'] = pd.to_numeric(df['qt'])
df['qt'] = df['qt']

df.index = pd.to_datetime(df.index)
print(df)
df.plot(style='-',
        figsize=(15, 5),
        color=color_pal[9],
        title='power ')

plt.show()



train = df.loc[df.index < '2023-12-01']
test = df.loc[df.index >= '2023-12-01']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.legend(['Training Set', 'Test Set'])
plt.show()

#df.loc[(df.index > '01-08-2010') & (df.index < '01-15-2010')] .plot(figsize=(15, 5), title='Week Of Data')
#plt.show()


def create_features(df):

    df = df.copy()
    df['date'] = df.index
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['Day'] = df.index.day_name()
    df['Hour'] = df.index.hour
    rolling_window = 24
    df['qt_lagged_1'] = df['qt'].shift(1)
    df['qt_roll_std'] = df['qt'].rolling(rolling_window).std()
    df['qt_rolling_corr'] = df['qt'].rolling(rolling_window).corr(df['qt'])
    df['qt_roll_mean'] = df['qt'].rolling(rolling_window).mean()
    df['qt_rolling_max'] = df['qt'].rolling(rolling_window).max()
    df['qt_rolling_min'] = df['qt'].rolling(rolling_window).min()
    df['qt_rolling_skew'] = df['qt'].rolling(rolling_window).skew()
    df['qt_rolling_kurt'] = df['qt'].rolling(rolling_window).kurt()
    df['qt_rolling_q25'] = df['qt'].rolling(rolling_window).quantile(0.25)
    df['qt_rolling_q75'] = df['qt'].rolling(rolling_window).quantile(0.75)
    df['qt_rolling_iqr'] = df['qt_rolling_q75'] - df['qt_rolling_q25']
    df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')
    df = pd.get_dummies(df, columns=['Day', 'Day_Night'])

    return df

df = create_features(df)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='Hour', y='qt',palette='Blues')
ax.set_title('MW by Hour')
#plt.show()

train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year','qt_lagged_1','qt_roll_std','qt_rolling_corr','qt_roll_mean',
            'qt_rolling_max','qt_rolling_min','qt_rolling_skew','qt_rolling_kurt','qt_rolling_q25','qt_rolling_q75','qt_rolling_iqr']
FEATURES += [col for col in train.columns if col.startswith('Day_')]
TARGET = 'qt'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1500,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=12 ,
                       learning_rate=0.001)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['qt']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='-')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

# Step 1: Create Features for the Next Month
future_start_date = df.index[-1] + pd.DateOffset(days=1)
future_end_date = future_start_date + pd.DateOffset(months=4) - pd.DateOffset(days=1)
future_dates = pd.date_range(future_start_date, future_end_date, freq='1h')
future_df = pd.DataFrame(index=future_dates)
future_df['qt'] = np.nan

# The first 'qt' value is the last 'qt' value from the training data
future_df.loc[future_start_date, 'qt'] = df.loc[df.index[-1], 'qt']

# Create all features for the future dataframe
future_df = create_features(future_df)

for d in future_df.index[1:]:
    # Update the 'qt' and 'qt_lagged_1' values
    future_df.loc[d, 'qt'] = reg.predict(future_df.loc[d, FEATURES].values.reshape(1, -1))
    future_df.loc[d, 'qt_lagged_1'] = future_df.loc[d - pd.DateOffset(hours=1), 'qt']




FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year','qt_lagged_1','qt_roll_std','qt_rolling_corr','qt_roll_mean',
            'qt_rolling_max','qt_rolling_min','qt_rolling_skew','qt_rolling_kurt','qt_rolling_q25','qt_rolling_q75','qt_rolling_iqr']
FEATURES += [col for col in train.columns if col.startswith('Day_')]

# Select relevant features for prediction
X_future = future_df[FEATURES]

# Use the trained XGBoost model to predict the next month
# Note: The 'qt' values in the future dataframe are now the predicted values
future_df['prediction'] = future_df['qt']

# Print the future dataframe
print(future_df)

# Save the future predictions to a CSV file
future_df.to_csv('forecast2_results.csv', index=False)

# Step 4: Visualize the Predictions
ax = df[['qt']].plot(figsize=(15, 5))
future_df['prediction'].plot(ax=ax, style='-', color='red')
plt.legend(['Historical Data', 'Future Predictions'])
plt.title('Historical Data and Future Predictions')
plt.show()


