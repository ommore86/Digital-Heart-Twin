import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load the trained model
model = joblib.load("heart_model.pkl")

# Title
st.title("Digital Twin of the Heart")
st.subheader("AI Simulation for Predicting Drug Suitability")

# Input fields
age = st.slider("Age", 20, 90, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])
heart_rate = st.slider("Max Heart Rate", 60, 200, 120)
ecg = st.selectbox("ECG Result", ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"])
drug = st.selectbox("Drug", ["Atenolol: Beta blocker", "Lisinopril: ACE inhibitor", "Amlodipine: Calcium channel blocker", "Losartan: Angiotensin II blocker", "Hydrochlorothiazide: Diuretic"])

# Button
if st.button("Simulate Drug Response"):
    # Encode inputs manually
    sex_encoded = 1 if sex == "Male" else 0
    bp_encoded = {"Low": 0, "Normal": 1, "High": 2}[bp]
    chol_encoded = {"Normal": 0, "High": 1}[cholesterol]
    ecg_encoded = {"Normal": 0, "ST-T abnormality": 1, "Left Ventricular Hypertrophy": 2}[ecg]
    drug_encoded = {"Atenolol: Beta blocker": 0, "Lisinopril: ACE inhibitor": 1, "Amlodipine: Calcium channel blocker": 2, "Losartan: Angiotensin II blocker": 3, "Hydrochlorothiazide: Diuretic": 4}[drug]

    # Prepare input
    input_data = pd.DataFrame([[
    age, sex_encoded, bp_encoded, chol_encoded, heart_rate, ecg_encoded, drug_encoded
    ]], columns=["Age", "Sex", "BP", "Cholesterol", "HeartRate", "ECG", "Drug"])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result with virtual twin
    if prediction == 1:
        st.success("✅ Drug is **suitable** for the patient.")
        # st.image("assets/good_heart.gif", caption="Healthy Heart Response", use_column_width=True)
    else:
        st.error("❌ Drug is **not suitable** for the patient.")
        # st.image("assets/bad_heart.gif", caption="Unhealthy Heart Response", use_column_width=True)

    predicted_class = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][predicted_class]

    st.write(f"💡 Confidence Score: **{confidence:.2%}**")