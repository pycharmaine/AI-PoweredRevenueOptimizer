import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Database Connection
conn = sqlite3.connect('ferry_database.db')  # Replace with your SQLite database.

# SQL Query to join necessary tables and retrieve data
sql_query = """
SELECT *
FROM financial_data
JOIN daily_operations ON financial_data.operation_id = daily_operations.operation_id
JOIN ferry_details ON daily_operations.ferry_id = ferry_details.ferry_id
JOIN route_data ON daily_operations.route_id = route_data.route_id
JOIN environmental_conditions ON daily_operations.operation_id = environmental_conditions.operation_id
JOIN fuel_data ON daily_operations.operation_id = fuel_data.operation_id
JOIN ticket_data ON daily_operations.operation_id = ticket_data.operation_id;
"""

# Retrieving data
df = pd.read_sql_query(sql_query, conn)

# Data Preparation: Feature Selection & Encoding
features = [
    'capacity', 'passenger_count', 'distance', 'weather_condition',
    'wave_height', 'fuel_quantity', 'fuel_price', 'price', 'is_weekend_or_holiday'
]
target_variable = 'revenue'

X = df[features]
y = df[target_variable]

categorical_features = ['weather_condition']
numerical_features = X.columns.difference(categorical_features).tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Model Training
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error: " + str(mse) + "\nRoot Mean Squared Error: " + str(rmse) + "\nR^2 Score: " + str(r2))

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()

# Model Tuning
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Displaying best parameters
print(f"Best Parameters: ", grid_search.best_params_)
