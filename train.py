"""
Diabetes Prediction using Neural Network (Keras)

Dataset Features:
1. Number of times pregnant
2. Plasma glucose concentration (2 hours)
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. Serum insulin (mu U/ml)
6. Body mass index (BMI)
7. Diabetes pedigree function
8. Age (years)

Target:
9. Class (0 = No Diabetes, 1 = Diabetes)
"""

# =========================
# Import Required Libraries
# =========================
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# =========================
# Load Dataset
# =========================
# Load CSV file
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y)
X = dataset[:, 0:8]   # First 8 columns = features
y = dataset[:, 8]     # Last column = target

# =========================
# Build Neural Network Model
# =========================
model = Sequential()

# Input layer + Hidden layer
model.add(Dense(12, input_dim=8, activation='relu'))

# Hidden layer
model.add(Dense(8, activation='relu'))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# =========================
# Compile Model
# =========================
model.compile(
    loss='binary_crossentropy',   # For binary classification
    optimizer='adam',             # Optimizer
    metrics=['accuracy']          # Evaluation metric
)

# =========================
# Train Model
# =========================
model.fit(X, y, epochs=50, batch_size=10)

# =========================
# Evaluate Model
# =========================
loss, accuracy = model.evaluate(X, y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# =========================
# Save Model (Optional)
# =========================
# Save model architecture
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("model.h5")

print("✅ Model saved successfully!")