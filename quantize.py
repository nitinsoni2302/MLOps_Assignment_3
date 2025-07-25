# quantize.py

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# --- Configuration ---
MODEL_PATH = 'linear_regression_model.joblib'
UNQUANT_PARAMS_PATH = 'unquant_params.joblib'
QUANT_PARAMS_PATH = 'quant_params.joblib'

print("Starting model conversion and quantization...")

#### Part B: Load Scikit-learn Model and Extract Parameters ####
print(f"\n1. Loading scikit-learn model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Scikit-learn model file not found at {MODEL_PATH}. Make sure it exists.")
    exit(1)

sklearn_model = joblib.load(MODEL_PATH)
print("Scikit-learn model loaded successfully.")

# Extract learned parameters (coefficients and intercept)
original_coef = sklearn_model.coef_
original_intercept = sklearn_model.intercept_

print(f"Original Coefficients (first 5): {original_coef[:5]}")
print(f"Original Intercept: {original_intercept}")

# Store the unquantized parameters in a dictionary
unquantized_params = {
    'coef': original_coef,
    'intercept': original_intercept
}

# Save the unquantized parameters
joblib.dump(unquantized_params, UNQUANT_PARAMS_PATH)
print(f"Unquantized parameters saved to {UNQUANT_PARAMS_PATH}")

#### Part C: Manual 8-bit Unsigned Quantization Logic ####
print("\n2. Performing manual 8-bit unsigned quantization...")

# Determine the min/max values across both coefficients and intercept for a unified scale
# This is important for consistency in quantization.
all_params = np.concatenate((original_coef.flatten(), [original_intercept]))
min_val = np.min(all_params)
max_val = np.max(all_params)

# Calculate Scale and Zero-Point for 8-bit unsigned (0-255 range)
Q_MIN = 0
Q_MAX = 255
SCALE = (max_val - min_val) / (Q_MAX - Q_MIN)
# Zero-point calculation, clamped to 0-255 range
ZERO_POINT = Q_MIN - round(min_val / SCALE)
# Ensure zero_point is within the 8-bit unsigned integer range
ZERO_POINT = max(Q_MIN, min(Q_MAX, ZERO_POINT))

print(f"Quantization Scale: {SCALE}")
print(f"Quantization Zero Point: {ZERO_POINT}")

# Quantize coefficients and intercept
quantized_coef = np.round(original_coef / SCALE + ZERO_POINT).astype(np.uint8)
quantized_intercept = np.round(original_intercept / SCALE + ZERO_POINT).astype(np.uint8)

print(f"Quantized Coefficients (first 5): {quantized_coef[:5]}")
print(f"Quantized Intercept: {quantized_intercept}")

# Store the quantized parameters
quantized_params = {
    'coef': quantized_coef,
    'intercept': quantized_intercept,
    'scale': SCALE,
    'zero_point': ZERO_POINT
}

# Save the quantized parameters
joblib.dump(quantized_params, QUANT_PARAMS_PATH)
print(f"Quantized parameters saved to {QUANT_PARAMS_PATH}")


#### Part D: PyTorch Model Creation and Inference with De-quantized Weights ####
print("\n3. Creating PyTorch model and performing inference with de-quantized weights...")

# Define a simple Linear Regression model in PyTorch
class LinearRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Load the California Housing dataset for inference
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# De-quantize the parameters for use in PyTorch model
dequantized_coef = (quantized_coef.astype(np.float32) - ZERO_POINT) * SCALE
dequantized_intercept = (quantized_intercept.astype(np.float32) - ZERO_POINT) * SCALE

print(f"De-quantized Coefficients (first 5): {dequantized_coef[:5]}")
print(f"De-quantized Intercept: {dequantized_intercept}")

# Initialize PyTorch model
input_dim = X_test.shape[1]
pytorch_model = LinearRegressionPyTorch(input_dim)

# Set the weights and bias of the PyTorch model
# Ensure dimensions match: coef for weights should be [output_dim, input_dim]
# intercept for bias should be [output_dim]
with torch.no_grad():
    pytorch_model.linear.weight.copy_(torch.tensor(dequantized_coef.reshape(1, -1), dtype=torch.float32))
    pytorch_model.linear.bias.copy_(torch.tensor(dequantized_intercept.reshape(1), dtype=torch.float32))

print("PyTorch model initialized with de-quantized weights.")

# Perform inference with the de-quantized PyTorch model
pytorch_model.eval() # Set model to evaluation mode
with torch.no_grad():
    y_pred_pytorch_tensor = pytorch_model(X_test_tensor)

y_pred_pytorch = y_pred_pytorch_tensor.numpy().flatten()

# Calculate R2 score for the de-quantized PyTorch model
r2_pytorch_quantized = r2_score(y_test, y_pred_pytorch)
print(f"R2 Score for De-quantized PyTorch Model: {r2_pytorch_quantized}")

#### Part E: Analysis and Reporting (To be implemented next - mostly manual calculation) ####
print("\n4. Analyzing and reporting results...")
# This section will guide you on how to calculate R2 scores and model sizes for comparison.

print("\nQuantization script finished.")