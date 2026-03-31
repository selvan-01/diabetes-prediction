"""
Test Script for Diabetes Prediction Model

This file:
1. Loads the trained model (JSON + H5)
2. Makes predictions on dataset
3. Compares predicted vs actual values
"""

# =========================
# Import Required Libraries
# =========================
import numpy as np
from keras.models import model_from_json

# =========================
# Load Dataset
# =========================
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y)
X = dataset[:, 0:8]   # Features
y = dataset[:, 8]     # Actual labels

# =========================
# Load Model from Files
# =========================
# Load model architecture
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Recreate model
model = model_from_json(loaded_model_json)

# Load weights into model
model.load_weights("model.h5")

print("✅ Model loaded successfully!")

# =========================
# Compile Model (IMPORTANT)
# =========================
# Needed before prediction/evaluation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# =========================
# Make Predictions
# =========================
predictions = model.predict(X)

# Convert probabilities to 0 or 1
predictions = (predictions > 0.5).astype(int)

# =========================
# Display Results
# =========================
print("\n🔍 Sample Predictions:\n")

# Show results for rows 5 to 9
for i in range(5, 10):
    print(f"Input: {X[i].tolist()}")
    print(f"Predicted: {predictions[i][0]} | Actual: {int(y[i])}")
    print("-" * 50)