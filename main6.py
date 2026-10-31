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
df_st = pd.read_csv('st.csv')
df_st.columns = ['' ,'', '' , '' ,'','date','st','','','']
df_st = df_st[['date','st']]
df_st =df_st.drop([0, 1, 2])
df_st['date'] = pd.to_datetime(df_st['date'])
df_st = df_st.set_index('date')
df_st['st'] = pd.to_numeric(df_st['st'])
df_st['st'] = df_st['st']

df_pt = pd.read_csv('pt.csv')
df_pt.columns = ['' ,'', '' , '' ,'','date','pt','','','']
df_pt = df_pt[['date','pt']]
df_pt =df_pt.drop([0, 1, 2])
df_pt['date'] = pd.to_datetime(df_pt['date'])
df_pt = df_pt.set_index('date')
df_pt['pt'] = pd.to_numeric(df_pt['pt'])
df_pt['pt'] = df_pt['pt']
df = df_pt.join(df_st, how='inner')

df_qt = pd.read_csv('qt.csv')
df_qt.columns = ['' ,'', '' , '' ,'','date','qt','','','']
df_qt = df_qt[['date','qt']]
df_qt =df_qt.drop([0, 1, 2])
df_qt['date'] = pd.to_datetime(df_qt['date'])
df_qt = df_qt.set_index('date')
df_qt['qt'] = pd.to_numeric(df_qt['qt'])
df_qt['qt'] = df_qt['qt']
df = df.join(df_qt, how='inner')

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
    df['minute']=df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    #df['st'] = np.sqrt(df['pt'] ** 2 + df['qt'] ** 2)
    df['Day'] = df.index.day_name()
    df['Hour'] = df.index.hour
    winter_months = [1, 2, 3, 4, 11,12]
    summer_months = [5,6, 7, 8,9,10]

    df['Day_Night'] = np.where(
        (df['month'].isin(winter_months) & (df['Hour'] >= 6) & (df['Hour'] < 16)) |
        (df['month'].isin(summer_months) & (df['Hour'] >= 5) & (df['Hour'] < 18)),
        'Day',
        'Night'
    )
    df = pd.get_dummies(df, columns=['Day', 'Day_Night'])

    return df

df = create_features(df)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='Hour', y='st',palette='Blues')
ax.set_title('MW by Hour')
#plt.show()

train = create_features(train)
test = create_features(test)

reg_pt = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=8 ,
                       learning_rate=0.001)

# Define features and target for pt
FEATURES_PT = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
FEATURES_PT += [col for col in train.columns if col.startswith('Day_')]
TARGET_PT = 'pt'

X_train_pt = train[FEATURES_PT]
y_train_pt = train[TARGET_PT]
X_test_pt = test[FEATURES_PT]
y_test_pt = test[TARGET_PT]
reg_pt.fit(X_train_pt, y_train_pt,
        eval_set=[(X_train_pt, y_train_pt), (X_test_pt, y_test_pt)],
        verbose=100)

test['pt_prediction'] = reg_pt.predict(X_test_pt)
df = df.merge(test[['pt_prediction']], how='left', left_index=True, right_index=True)
ax = df[['pt']].plot(figsize=(15, 5))
df['pt_prediction'].plot(ax=ax, style='-')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year','minute']
FEATURES += [col for col in train.columns if col.startswith('Day_')]
TARGET = 'st'
#print(df['st'])
X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=2000,
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
ax = df[['st']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='-')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

# Step 1: Create Features for the Next Month
future_start_date = df.index[-1] + pd.DateOffset(days=1)
future_end_date = future_start_date + pd.DateOffset(months=4) - pd.DateOffset(days=1)
future_dates = pd.date_range(future_start_date, future_end_date, freq='1t')
future_df = pd.DataFrame(index=future_dates)

previous_year_start_date = future_start_date - pd.DateOffset(years=1)
previous_year_end_date = future_end_date - pd.DateOffset(years=1)
print(previous_year_start_date)
print(previous_year_end_date)
# Filter 'qt' values for the same period from the previous year
previous_year_qt = df.loc[previous_year_start_date:previous_year_end_date, 'qt']
print(previous_year_qt)
# Create a new DataFrame with 'qt' values for the next 3 months
previous_year_qt_resampled = previous_year_qt.resample('1T').ffill()

# Replicate the resampled 'qt' values to match the length of future_df
replicated_qt = pd.concat([previous_year_qt_resampled] * (len(future_df) // len(previous_year_qt_resampled)) + [previous_year_qt_resampled.iloc[:len(future_df) % len(previous_year_qt_resampled)]], ignore_index=True)

# Subtract 1 from all values in the 'qt' column
replicated_qt = replicated_qt.sub(1)

# Assign the replicated 'qt' values to the future_df
future_df['qt'] = replicated_qt.values


future_df['pt'] = np.nan
#future_df['st'] = np.nan
# Use the create_features function to add relevant features
future_df = create_features(future_df)

FEATURES_PT = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
FEATURES_PT += [col for col in future_df.columns if col.startswith('Day_')]

# Predict pt for the future period
future_pt_prediction = reg_pt.predict(future_df[FEATURES_PT])

# Add the predicted pt to the future DataFrame
future_df['pt'] = future_pt_prediction
print(future_df['pt'])
print(future_df['qt'])
#print(future_df['st'])
# Step 3: Predict with the Trained Model
FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year','minute']
FEATURES += [col for col in train.columns if col.startswith('Day_')]
# Select relevant features for prediction
#df['st']= np.sqrt(df['pt'] ** 2 + df['qt'] ** 2)
X_future = future_df[FEATURES]
#print(future_df['st'])
# Use the trained XGBoost model to predict the next month
future_df['prediction'] = reg.predict(X_future)
print(future_df)
future_df.to_csv('forecast_results.csv', index=False)
# Step 4: Visualize the Predictions
ax = df[['st']].plot(figsize=(15, 5))
future_df['prediction'].plot(ax=ax, style='-', color='red')
plt.legend(['Historical Data', 'Future Predictions'])
plt.title('Historical Data and Future Predictions')
plt.show()


