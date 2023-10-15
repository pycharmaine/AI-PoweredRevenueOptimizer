import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Database Connection
conn = sqlite3.connect('ferry_database.db')

# SQL Query to Retrieve Data
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

# Normalizing Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Train/Test Split
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[0:train_size], scaled_data[train_size:]

# Function to Create Sequences
def create_sequences(data, step):
    X, y = [], []
    for i in range(len(data)-step-1):
        X.append(data[i:(i+step), 0])
        y.append(data[i + step, 0])
    return np.array(X), np.array(y)

step = 3  # Considering 3 time steps
X_train, y_train = create_sequences(train, step)
X_test, y_test = create_sequences(test, step)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse Transformation
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))

print('Train RMSE: {train_rmse}\nTest RMSE: {test_rmse}')

# Plotting
plt.figure(figsize=(15,6))
plt.plot(df, label='Original Data')
plt.plot(np.arange(step, len(train_predict)+step), train_predict, label='Train Predictions')
plt.plot(np.arange(len(train_predict)+(2*step)+1, len(df)-1), test_predict, label='Test Predictions')
plt.legend()
plt.show()
