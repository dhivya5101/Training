import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set the MLflow tracking URI to the MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Load the Online Retail dataset
df = pd.read_csv('online_retail_dataset.csv')  
df = df.dropna(subset=['Quantity', 'UnitPrice', 'SalesChannel', 'Country'])  # Clean data

# Feature engineering
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = pd.get_dummies(df, columns=['SalesChannel', 'Country'])

# Features and Target
X = df[['Quantity', 'UnitPrice', 'TotalPrice']]  # Example features
y = df['TotalPrice']  # Example target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model using R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

# Log to MLflow if R2 score is above 0.7
if r2 >= 0.7:
    mlflow.start_run()  # Start an MLflow run
    mlflow.log_param("R2_Score", r2)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Optional: log any other parameters or metrics you might need
    mlflow.log_metric("r2_score", r2)

    print("Model is deployed in MLflow!")
    mlflow.end_run()  # End the run
else:
    print("R2 score is below the threshold, model is not deployed.")
