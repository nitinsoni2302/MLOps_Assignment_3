# train.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

print("Starting model training...")

# 1. Load the California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

# 3. Train a Linear Regression model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Evaluate the model (optional, but good for verification)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R^2 score: {train_score:.4f}")
print(f"Test R^2 score: {test_score:.4f}")

# 5. Save the trained model using joblib
model_path = 'linear_regression_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

print("Training script finished.")