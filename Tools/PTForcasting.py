import json
import os

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS


class PTForcasting:
    def __init__(self):
        self.influxdb_url = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.environ.get("INFLUXDB_TOKEN", "")
        self.org = os.environ.get("INFLUXDB_ORG", "project")
        self.bucket = "Predictions"
        self.df = None
        self.train = None
        self.test = None
        self.reg = None
        self.FEATURES = ['dayofyear', 'Hour', 'dayofweek', 'quarter', 'month', 'year']
        self.TARGET = 'value'
        self.client = InfluxDBClient(url=self.influxdb_url, token=self.token, org=self.org)
        self.model_file = 'xgboostPT_model.pkl'
        self.threshold = None  # Set dynamically based on historical data
    def load_and_process_data(self, id,field):
        query = f'from(bucket: "{self.bucket}") |> range(start: -1y) |> filter(fn: (r) => r._measurement == "{id}" and r._field == "{field}")'
        result = self.client.query_api().query(org=self.org, query=query)

        data = []
        for table in result:
            for record in table.records:
                data.append((record.get_time(), record.get_value()))

        df = pd.DataFrame(data, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        self.df = df
        print(self.df)
        self.df.to_csv('df-formatted.csv', index=True)

        filtered_df = df[df['value'] > 4]

        # Check if there are any values above 4 to avoid division by zero or empty dataframe
        if not filtered_df.empty:
            # Calculate the average of values above 4
            avg_value_above_4 = filtered_df['value'].mean()

            # Calculate the alarm threshold (10% above the average of values above 4)
            self.threshold = avg_value_above_4 * 1.10

            print("Calculated average (values > 4):", avg_value_above_4)
            print("Calculated alarm threshold:", self.threshold)
        else:
            print("No values above 4 found in the dataset.")

    def split_data(self):
        self.train = self.df.loc[self.df.index < '2023-12-01']
        self.test = self.df.loc[self.df.index >= '2023-12-01']

    @staticmethod
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
        df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')
        df = pd.get_dummies(df, columns=['Day', 'Day_Night'])
        print("df= ", df.columns)


        return df



    def feature_engineering(self):
        self.train = self.create_features(self.train)
        self.test = self.create_features(self.test)
        print("train= ",self.train)
        self.FEATURES += [col for col in self.test.columns if col.startswith('Day_')]
        print("features= ",self.FEATURES)

    def train_model(self):
        X_train = self.train[self.FEATURES]
        y_train = self.train[self.TARGET]
        X_test = self.test[self.FEATURES]
        y_test = self.test[self.TARGET]
        self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                    n_estimators=1000,
                                    early_stopping_rounds=50,
                                    objective='reg:squarederror',
                                    max_depth=8,
                                    learning_rate=0.001,
                                    # nthread=4,
                                    device='cuda',
                                    tree_method="hist"
                                    )
        self.reg.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_test, y_test)],
                     verbose=100)

        self.test['prediction'] = self.reg.predict(X_test)
        # Save the model
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

        # Converting 'date' column to string format
        future_df_formatted['date'] = future_df_formatted['index'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Selecting only 'date' and 'prediction' columns
        future_df_formatted = future_df_formatted[['date', 'prediction']]
        # Converting DataFrame to list of dictionaries
        future_data_json = future_df_formatted.to_dict(orient='records')

        # Flag points exceeding the alarm threshold (no alarms if no threshold could be computed)
        for data_point in future_data_json:
            data_point['alarm'] = bool(self.threshold is not None and data_point['prediction'] > self.threshold)
            data_point['threshold'] = self.threshold

        return future_data_json, self.threshold
