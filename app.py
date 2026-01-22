import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Breast Cancer Predictor 💙")

# Input fields
radius = st.number_input("Radius Mean")
texture = st.number_input("Texture Mean")
perimeter = st.number_input("Perimeter Mean")
area = st.number_input("Area Mean")
smoothness = st.number_input("Smoothness Mean")

if st.button("Predict"):
    features = np.array([[radius, texture, perimeter, area, smoothness]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    st.success("Prediction: " + ("Malignant" if pred == 0 else "Benign"))
