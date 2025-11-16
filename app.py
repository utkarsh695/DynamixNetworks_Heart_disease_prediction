import streamlit as st
import pickle
import numpy as np

# Load Model and Scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict whether they are at risk of heart disease.")

# UI inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise-Induced Angina (1=Yes, 0=No)", [1,0])
oldpeak = st.number_input("ST Depression", value=1.0)
slope = st.selectbox("Slope of ST Segment (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1,2,3)", [1,2,3])

# Predict Button
if st.button("Predict"):
    sample = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
    sample = scaler.transform(sample)
    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("❌ High chances of Heart Disease. Please Consult a Doctor.")
    else:
        st.success("✔ No Heart Disease Detected. Keep maintaining a healthy lifestyle!")
