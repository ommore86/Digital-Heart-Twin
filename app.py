import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import re
from unidecode import unidecode

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = Path("heart_risk_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()

# Drug name normalization
def norm_drug_name(name: str) -> str:
    x = unidecode((name or "").lower())
    x = re.split(r"[,;|]", x)[0]
    x = re.sub(r"\b(pamoate|hydrochloride|benzathine|sodium|succinate|acetate|phosphate)\b", "", x)
    x = re.sub(r"[^a-z0-9]+", " ", x).strip()
    return x

# Prediction function with default values for missing columns
def predict_safety(patient: dict, drug_name: str, threshold: float = 0.5) -> dict:
    # Normalize drug name
    drug_key = norm_drug_name(drug_name)
    
    # Prepare features with defaults for missing columns
    features = {
        "age": patient["age"],
        "hr": patient["hr"],
        "qtc": patient["qtc"],
        "ef": patient["ef"],
        "sbp": patient["sbp"],
        "dbp": patient["dbp"],
        "k": patient["k"],
        "mg": patient["mg"],
        "creatinine": patient["creatinine"],
        "hx_arrhythmia": patient["hx_arrhythmia"],
        "hx_hf": patient["hx_hf"],
        "hx_cad": patient["hx_cad"],
        "sex": patient["sex"],
        # Default values for missing columns
        "drug_base_risk": 0.3,  # Default medium risk
        "kw_flag": 1 if any(kw in drug_key for kw in ["qt", "arrhythm", "brady", "tachy", "heart"]) else 0
    }
    
    # Convert to DataFrame with correct column order
    feature_cols = [
        'age', 'hr', 'qtc', 'ef', 'sbp', 'dbp', 'k', 'mg', 
        'creatinine', 'hx_arrhythmia', 'hx_hf', 'hx_cad',
        'drug_base_risk', 'kw_flag', 'sex'
    ]
    X = pd.DataFrame([features])[feature_cols]
    
    # Make prediction
    prob = float(model.predict_proba(X)[0, 1])
    decision = "unsafe" if prob >= threshold else "safe"
    
    # Generate explanation
    explanation = "Risk factors: "
    risk_factors = []
    
    if patient["qtc"] > 450: risk_factors.append(f"prolonged QTc ({patient['qtc']}ms)")
    if patient["ef"] < 40: risk_factors.append(f"low EF ({patient['ef']}%)")
    if patient["k"] < 3.5: risk_factors.append(f"low potassium ({patient['k']}mmol/L)")
    if patient["mg"] < 1.7: risk_factors.append(f"low magnesium ({patient['mg']}mg/dL)")
    if patient["hx_arrhythmia"] == 1: risk_factors.append("history of arrhythmia")
    if patient["hx_hf"] == 1: risk_factors.append("history of heart failure")
    if patient["age"] > 65: risk_factors.append(f"age >65 ({patient['age']}y)")
    if features["kw_flag"] == 1: risk_factors.append("high-risk drug type")
    if features["drug_base_risk"] > 0.5: risk_factors.append("high-risk medication")
    
    explanation += ", ".join(risk_factors) if risk_factors else "no major risk factors identified"
    
    return {
        "drug_key": drug_key,
        "prob_unsafe": round(prob, 3),
        "decision": decision,
        "explanation": explanation,
        "risk_factors": risk_factors
    }

# Demo cases
DEMO_CASES = {
    "High Risk Patient": {
        "sex": "F", "age": 67, "hr": 54, "qtc": 482, "ef": 35,
        "sbp": 118, "dbp": 72, "k": 3.2, "mg": 1.6, "creatinine": 1.8,
        "hx_arrhythmia": 1, "hx_hf": 1, "hx_cad": 0
    },
    "Low Risk Patient": {
        "sex": "M", "age": 30, "hr": 75, "qtc": 410, "ef": 65,
        "sbp": 120, "dbp": 80, "k": 4.2, "mg": 2.0, "creatinine": 0.8,
        "hx_arrhythmia": 0, "hx_hf": 0, "hx_cad": 0
    }
}

# Streamlit UI
st.title("💊 Cardiac Drug Safety Predictor")
st.markdown("Predict if a drug is safe for patients with cardiac risk factors")

# Sidebar for test cases
with st.sidebar:
    st.header("Quick Test Cases")
    case = st.selectbox("Load example patient:", list(DEMO_CASES.keys()))
    if st.button("Load Selected Patient"):
        st.session_state.patient_data = DEMO_CASES[case]

# Main form
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        sex = st.radio("Sex", ["M", "F"], horizontal=True)
        age = st.slider("Age", 18, 100, 50)
        hx_arrhythmia = st.checkbox("History of Arrhythmia")
        hx_hf = st.checkbox("History of Heart Failure")
        hx_cad = st.checkbox("History of CAD")
    
    with col2:
        st.subheader("Vital Signs")
        hr = st.number_input("Heart Rate (bpm)", 40, 140, 75)
        qtc = st.number_input("QTc Interval (ms)", 300, 600, 420)
        ef = st.number_input("Ejection Fraction (%)", 15, 80, 55)
        sbp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
        dbp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
        k = st.number_input("Potassium (mmol/L)", 2.5, 6.0, 4.2)
        mg = st.number_input("Magnesium (mg/dL)", 1.0, 3.0, 2.0)
        creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 5.0, 1.0)
    
    drug_name = st.text_input("Drug Name (e.g., Tigan, Dofetilide)", "Tigan")
    
    # Add manual override for drug risk factors
    with st.expander("Advanced Drug Settings"):
        drug_base_risk = st.slider("Estimated Drug Risk", 0.0, 1.0, 0.3)
        kw_flag = st.checkbox("Drug has cardiac risk keywords (QT, arrhythmia, etc.)")
    
    if st.form_submit_button("Predict Safety"):
        patient_data = {
            "sex": sex, "age": age, "hr": hr, "qtc": qtc, "ef": ef,
            "sbp": sbp, "dbp": dbp, "k": k, "mg": mg, "creatinine": creatinine,
            "hx_arrhythmia": int(hx_arrhythmia),
            "hx_hf": int(hx_hf),
            "hx_cad": int(hx_cad)
        }
        
        try:
            # Include manual overrides in prediction
            result = predict_safety(
                patient_data, 
                drug_name,)
            
            st.subheader("Results")
            risk_color = "red" if result["decision"] == "unsafe" else "green"
            st.markdown(f"**Decision:** <span style='color:{risk_color};font-size:20px'>"
                       f"{result['decision'].upper()}</span>", 
                       unsafe_allow_html=True)
            
            st.metric("Risk Probability", f"{result['prob_unsafe']*100:.1f}%")
            
            with st.expander("Detailed Explanation"):
                st.write(result["explanation"])
                if result["risk_factors"]:
                    st.write("**Key Risk Factors:**")
                    for factor in result["risk_factors"]:
                        st.write(f"- {factor}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

st.caption("Common cardiac-risk drugs: Tigan (trimethobenzamide), Dofetilide, Sotalol, Dronedarone, Cisapride")