# predict.py

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

print("Starting prediction script for container verification...")

# Define the path where the model is expected to be
model_path = 'linear_regression_model.joblib'

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Make sure it's available in the container.")
    exit(1) # Exit with an error code

# Load the trained model
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1) # Exit with an error code


# Load a small part of the California Housing dataset for prediction
# This simulates receiving new data for prediction
try:
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    _, X_test, _, _ = train_test_split(X, y, test_size=0.01, random_state=42) # Use a very small test set
    sample_data = X_test.iloc[[0]] # Take the first sample from the test set
    print(f"Sample data for prediction (shape {sample_data.shape}):\n{sample_data.head()}")
except Exception as e:
    print(f"Error preparing sample data: {e}")
    exit(1) # Exit with an error code


# Make a prediction
try:
    prediction = model.predict(sample_data)
    print(f"Prediction for sample data: {prediction[0]:.2f}")
    print("Prediction script finished successfully.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit(1) # Exit with an error code