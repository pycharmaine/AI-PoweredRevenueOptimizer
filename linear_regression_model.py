import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# SQL Query
sql_query = """
SELECT 
    f.revenue,
    d.operation_date,
    d.passenger_count,
    r.distance,
    e.weather_condition,
    e.wave_height,
    fd.fuel_type,
    fd.fuel_quantity,
    fd.fuel_price,
    t.ticket_type,
    t.price AS ticket_price,
    t.is_weekend_or_holiday
FROM 
    financial_data f
JOIN 
    daily_operations d ON f.operation_id = d.operation_id
JOIN 
    route_data r ON d.route_id = r.route_id
JOIN 
    environmental_conditions e ON d.operation_id = e.operation_id
JOIN 
    fuel_data fd ON d.operation_id = fd.operation_id
JOIN 
    ticket_data t ON d.operation_id = t.operation_id;
"""

# Connecting to the SQLite database
# Replace 'your_database.db' with your SQLite database path
conn = sqlite3.connect('ferry_database.db')

# Using pandas to read SQL query into a DataFrame
df = pd.read_sql_query(sql_query, conn)


# Features and Target Variable
features = ['passenger_count', 'expenses', 'distance', 'wave_height', 'fuel_quantity', 'fuel_price', 'price']
target = 'revenue'

# Basic Data Preprocessing
# Encoding the categorical variables, and scaling might be needed depending on the model

# Model
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluation
predictions = linear_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Visualization
plt.scatter(y_test, predictions)
plt.plot([min(y_test), max(y_test)], [min(predictions), max(predictions)], color='red')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Model Details and Metrics Output
print("Linear Regression MSE: " + str(mse))
print("Linear Regression R2 Score: " + str(r2))

# Closing the connection
conn.close()
