import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Database Connection
conn = sqlite3.connect('ferry_database.db')

# SQL Query to join tables and retrieve data
sql_query = """
SELECT financial_data.revenue, daily_operations.operation_date
FROM financial_data
JOIN daily_operations ON financial_data.operation_id = daily_operations.operation_id;
"""

# Data Retrieval and Preprocessing
df = pd.read_sql_query(sql_query, conn)
df['operation_date'] = pd.to_datetime(df['operation_date'])
df.set_index('operation_date', inplace=True)
df.sort_index(inplace=True)

# Train/Test Split
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:]

# Model Selection with auto_arima
stepwise_model = auto_arima(train, start_p=1, start_q=1,
                            max_p=5, max_q=5, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore', suppress_warnings=True,
                            stepwise=True)
print(stepwise_model.aic())

# Model Fitting with SARIMAX due to seasonality
model = SARIMAX(train, order=stepwise_model.order, seasonal_order=stepwise_model.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Forecasting
forecast = model_fit.get_forecast(steps=len(test))
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

# Visualizing Results
plt.plot(train.index, train, label='observed')
plt.plot(mean_forecast.index, mean_forecast.values, color='r', label='forecast')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink')
plt.plot(test.index, test, label='true test values')
plt.legend()
plt.title('ARIMA Time Series Forecasting')
plt.show()

# Evaluation
mse = mean_squared_error(test, mean_forecast)
rmse = sqrt(mse)
print("Mean Squared Error: " + str(mse) + "\nRoot Mean Squared Error: " + str(rmse))
