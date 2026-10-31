import os

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error


class THDAForecasting:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.train = None
        self.test = None
        self.model = None
        self.FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
        self.TARGET = 'value'
        self.model_file = 'xgboostTHDA_model.pkl'
        self.threshold = None
    def load_and_process_data(self):
        self.df = pd.read_csv(self.csv_file)
        self.df.columns = ['', '', '', '', '', 'date', 'value', '', '', '']
        self.df = self.df[['date', 'value']]
        self.df = self.df.drop([0, 1, 2])
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.set_index('date')
        self.df['value'] = pd.to_numeric(self.df['value'])

        # Remove outliers
        rows_to_remove = self.df[(self.df['value'] > 75) | (self.df['value'] < 10)]
        self.df.drop(rows_to_remove.index, inplace=True)

        filtered_df = self.df[self.df['value'] > 35]


        if not filtered_df.empty:

            avg_value_above_35 = filtered_df['value'].mean()


            self.threshold = avg_value_above_35 * 1.10

            print("Calculated average (values > 35):", avg_value_above_35)
            print("Calculated alarm threshold:", self.threshold)
        else:
            print("No values above 35 found in the dataset.")

    def split_data(self):
        self.train = self.df.loc[self.df.index < '2023-12-01']
        self.test = self.df.loc[self.df.index >= '2023-12-01']



    @staticmethod
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
        df['minute'] = df.index.minute

        winter_months = [1, 2, 3, 4, 11, 12]
        summer_months = [5, 6, 7, 8, 9, 10]

        df['Day_Night'] = np.where(
            (df['month'].isin(winter_months) & (df['Hour'] >= 6) & (df['Hour'] < 16)) |
            (df['month'].isin(summer_months) & (df['Hour'] >= 5) & (df['Hour'] < 18)),
            'Day',
            'Night')
        df = pd.get_dummies(df, columns=['Day', 'Day_Night'])



        return df


    def feature_engineering(self):
        self.train = self.create_features(self.train)
        self.test = self.create_features(self.test)

        self.FEATURES += [col for col in self.train.columns if col.startswith('Day_')]



    def train_model(self):
        X_train = self.train[self.FEATURES]
        y_train = self.train[self.TARGET]

        X_test = self.test[self.FEATURES]
        y_test = self.test[self.TARGET]

        self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                      n_estimators=50000, early_stopping_rounds=50,
                                      objective='reg:squarederror', max_depth=12,
                                      learning_rate=0.001,
                                      device='cuda', tree_method='hist')
        self.reg.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), (X_test, y_test)],
                       verbose=100)

        self.test['prediction'] = self.reg.predict(X_test)

        joblib.dump(self.reg, self.model_file)




    def load_model(self):
        # Train once if no saved model exists yet
        if not os.path.exists(self.model_file):
            self.train_model()
        self.reg = joblib.load(self.model_file)

    def forecast(self, months=1):
        future_start_date = self.df.index[-1].floor('h') + pd.Timedelta(hours=1)
        future_end_date = future_start_date + pd.DateOffset(months=months) - pd.Timedelta(hours=1)
        future_dates = pd.date_range(future_start_date, future_end_date, freq='1h')
        future_df = pd.DataFrame(index=future_dates)

        future_df = self.create_features(future_df)

        X_future = future_df[self.FEATURES]
        future_df['prediction'] = self.reg.predict(X_future)
        future_df_formatted = future_df[['prediction']].reset_index()

        future_df_formatted['date'] = future_df_formatted['index'].dt.strftime('%Y-%m-%d %H:%M:%S')

        future_df_formatted = future_df_formatted[['date', 'prediction']]

        future_data_json = future_df_formatted.to_dict(orient='records')
        # Flag points exceeding the alarm threshold (no alarms if no threshold could be computed)
        for data_point in future_data_json:
            data_point['alarm'] = bool(self.threshold is not None and data_point['prediction'] > self.threshold)
            data_point['threshold'] = self.threshold
        return future_data_json,self.threshold
