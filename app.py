"""
Streamlit App for Diabetes Prediction
"""

# =========================
# Import Libraries
# =========================
import streamlit as st
import numpy as np
from keras.models import load_model

# =========================
# Load Model
# =========================
model = load_model("diabetes_model.h5")

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# =========================
# Title
# =========================
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# =========================
# Input Fields
# =========================
pregnancies = st.number_input("Pregnancies", min_value=0.0)
glucose = st.number_input("Glucose Level", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0.0)

# =========================
# Predict Button
# =========================
if st.button("Predict"):

    # Convert input to numpy array
    input_data = np.array([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi,
        pedigree, age
    ]])

    # Prediction
    prediction = model.predict(input_data)
    result = int(prediction[0][0] > 0.5)

    # Output
    if result == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")