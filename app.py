import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import re
from unidecode import unidecode
import json

st.set_page_config(page_title="Cardiac Drug Safety Predictor", page_icon="💊", layout="wide")

# =========================
# Helpers (match your train)
# =========================
FEATURE_ORDER = [
    'age', 'hr', 'qtc', 'ef', 'sbp', 'dbp', 'k', 'mg',
    'creatinine', 'hx_arrhythmia', 'hx_hf', 'hx_cad',
    'drug_base_risk', 'kw_flag', 'sex'
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [unidecode(c).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).map(lambda x: unidecode(x).lower().strip())
    return df

def norm_drug_name(name: str) -> str:
    x = unidecode((name or "").lower())
    x = re.split(r"[,;|]", x)[0]
    x = re.sub(r"\b(pamoate|hydrochloride|benzathine|sodium|succinate|acetate|phosphate|fumarate|tartrate|mesylate)\b", "", x)
    x = re.sub(r"[^a-z0-9]+", " ", x).strip()
    return x

def pick_key_rowwise(row, fields):
    for f in fields:
        if f in row and isinstance(row[f], str) and row[f] not in ("", "nan"):
            x = re.split(r"[,;|]", row[f])[0].strip()
            x = re.sub(r"\b(pamoate|hydrochloride|benzathine|sodium|succinate|acetate|phosphate|fumarate|tartrate|mesylate)\b", "", x)
            x = re.sub(r"[^a-z0-9]+", " ", x).strip()
            if x:
                return x
    return ""

# Optional: derive base risk from dictrank-like fields (lightweight; not hard-coded map)
SEVERITY_MAP = {
    "none": 0.0, "na": 0.0, "mild": 0.2, "moderate": 0.4, "severe": 0.7, "less": 0.15, "more": 0.5
}
KEYWORD_PATTERNS = {
    "qt": r"\bqt|torsade|torsades|prolong",
    "arrhythmia": r"arrhythm|palpitation",
    "brady": r"\bbrady",
    "tachy": r"\btachy",
    "hf": r"heart failure|reduced ejection|cardiomyopathy",
}
def kw_score(s: str) -> float:
    if not isinstance(s, str): return 0.0
    sc = 0.0
    for pat in KEYWORD_PATTERNS.values():
        if re.search(pat, s):
            sc += 0.15
    return min(sc, 0.6)

def estimate_base_risk(row) -> float:
    tox = (row.get("cardiotoxicity") or "")
    if any(k in str(tox) for k in ["arrhythm", "qt"]): tox_base = 0.45
    elif str(tox) in ("", "na", "no", "none"):         tox_base = 0.05
    else:                                              tox_base = 0.2
    sev = (row.get("dic_severity_level") or "")
    sev_w = SEVERITY_MAP.get(str(sev), 0.1)
    kw = (row.get("keywords") or "")
    kw_w = kw_score(kw)
    label_sec = (row.get("label_section") or "")
    label_boost = 0.2 if any(t in str(label_sec) for t in ["wp", "warning", "precaution"]) else 0.0
    r = tox_base + sev_w + kw_w + label_boost
    return float(max(0.0, min(1.0, r)))

def compute_kw_flag(keywords: str) -> int:
    if not isinstance(keywords, str): return 0
    return 1 if re.search(r"qt|arrhythm|brady|tachy|heart failure|cardiomyopathy|reduced ejection", keywords) else 0

def approx_match(query_key: str, candidates: pd.Index) -> str:
    if query_key in candidates: 
        return query_key
    for k in candidates:
        if k.startswith(query_key) or query_key.startswith(k):
            return k
    qtok = set(query_key.split())
    best, bestj = "", 0.0
    for k in candidates:
        ktok = set(k.split())
        j = len(qtok & ktok) / max(1, len(qtok | ktok))
        if j > bestj:
            best, bestj = k, j
    return best or query_key

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model():
    p = Path("heart_model.pkl")
    if not p.exists():
        st.error("`heart_model.pkl` not found in the app directory.")
        st.stop()
    try:
        return joblib.load(p)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# =========================
# Optional: Load drug index
# =========================
@st.cache_data
def build_drug_index(file_bytes: bytes | None):
    """
    Returns a DataFrame with columns: drug_key, keywords, drug_base_risk, kw_flag, cardio_related(bool)
    If file not provided, returns empty DataFrame (frontend will fallback to defaults).
    """
    if file_bytes is None:
        return pd.DataFrame(columns=["drug_key","keywords","drug_base_risk","kw_flag","cardio_related"])
    try:
        df_raw = pd.read_csv(file_bytes)
        df = normalize_cols(df_raw)
        name_fields = ["trade_name", "generic_proper_name_s", "active_ingredient_s", "brand_names", "related_drugs", "activity"]
        df["drug_key"] = df.apply(lambda r: pick_key_rowwise(r, name_fields), axis=1)
        df = df[df["drug_key"] != ""].drop_duplicates("drug_key")

        # estimate base risk (lightweight heuristic derived from columns; not fixed table)
        if "cardiotoxicity" not in df.columns:
            df["cardiotoxicity"] = ""
        if "dic_severity_level" not in df.columns:
            df["dic_severity_level"] = ""
        if "label_section" not in df.columns:
            df["label_section"] = ""
        if "keywords" not in df.columns:
            df["keywords"] = ""

        df["drug_base_risk"] = df.apply(estimate_base_risk, axis=1)
        df["kw_flag"] = df["keywords"].fillna("").apply(lambda s: 1 if kw_score(s) > 0 else 0)
        df["cardio_related"] = (df["kw_flag"] == 1) | (df["drug_base_risk"] > 0.2)
        return df[["drug_key","keywords","drug_base_risk","kw_flag","cardio_related"]]
    except Exception as e:
        st.warning(f"Could not parse uploaded drug list: {e}")
        return pd.DataFrame(columns=["drug_key","keywords","drug_base_risk","kw_flag","cardio_related"])

# =========================
# UI – Sidebar
# =========================
st.sidebar.title("Settings")
uploaded = st.sidebar.file_uploader("Optionally upload dictrank_dataset.csv", type=["csv"])
drug_index = build_drug_index(uploaded) if uploaded else build_drug_index(None)

st.sidebar.markdown("**Model:** XGBoost pipeline (heart_model.pkl)")

# =========================
# UI – Demo cases
# =========================
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

with st.sidebar:
    st.header("Quick Test")
    case = st.selectbox("Load example patient:", list(DEMO_CASES.keys()))
    if st.button("Load Selected Patient"):
        st.session_state["patient_data"] = DEMO_CASES[case]

# =========================
# UI – Main
# =========================
st.title("💊 Cardiac Drug Safety Predictor")
st.caption("Predict safe / unsafe for a patient–drug pair. If the drug is not cardio-related, you’ll see **NOT APPLICABLE**.")

# Restore loaded patient if present
if "patient_data" in st.session_state:
    defaults = st.session_state["patient_data"]
else:
    defaults = DEMO_CASES["Low Risk Patient"]

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient")
        sex = st.radio("Sex", ["M","F"], horizontal=True, index=0 if defaults["sex"]=="M" else 1)
        age = st.slider("Age", 18, 100, int(defaults["age"]))
        hx_arrhythmia = st.checkbox("History of Arrhythmia", bool(defaults["hx_arrhythmia"]))
        hx_hf = st.checkbox("History of Heart Failure", bool(defaults["hx_hf"]))
        hx_cad = st.checkbox("History of CAD", bool(defaults["hx_cad"]))

    with col2:
        st.subheader("Vitals & Labs")
        hr = st.number_input("Heart Rate (bpm)", 40, 140, int(defaults["hr"]))
        qtc = st.number_input("QTc Interval (ms)", 300, 600, int(defaults["qtc"]))
        ef  = st.number_input("Ejection Fraction (%)", 15, 80, int(defaults["ef"]))
        sbp = st.number_input("Systolic BP (mmHg)", 80, 200, int(defaults["sbp"]))
        dbp = st.number_input("Diastolic BP (mmHg)", 40, 120, int(defaults["dbp"]))
        k   = st.number_input("Potassium (mmol/L)", 2.5, 6.0, float(defaults["k"]))
        mg  = st.number_input("Magnesium (mg/dL)", 1.0, 3.0, float(defaults["mg"]))
        creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 5.0, float(defaults["creatinine"]))

    drug_name = st.text_input("Drug Name", "bisoprolol")

    with st.expander("Advanced Drug Settings (override if needed)"):
        use_auto = st.checkbox("Auto-derive drug features from dataset (recommended)", value=True, help="If dictrank_dataset.csv is uploaded, the app will estimate drug_base_risk and kw_flag.")
        manual_base = st.slider("Manual drug_base_risk (if override)", 0.0, 1.0, 0.15, 0.01)
        manual_kw = st.checkbox("Manual kw_flag (cardio keywords present?)", value=False)
        not_app_rule = st.checkbox("Enable NOT APPLICABLE when drug not cardio-related", value=True, help="If kw_flag=0 and drug_base_risk≤0.15")

    submitted = st.form_submit_button("Predict Safety")

# =========================
# Prediction
# =========================
def get_drug_features_from_index(drug_name: str):
    """
    Resolve a drug against uploaded index and return (drug_key, drug_base_risk, kw_flag, cardio_related, source_row)
    If not found, fallback to defaults.
    """
    if drug_index is None or drug_index.empty:
        dk = norm_drug_name(drug_name)
        return dk, 0.15, 0, False, None

    q = norm_drug_name(drug_name)
    key = approx_match(q, drug_index["drug_key"])
    row = drug_index.loc[drug_index["drug_key"] == key]
    if row.empty:
        return q, 0.15, 0, False, None
    r = row.iloc[0]
    return key, float(r["drug_base_risk"]), int(r["kw_flag"]), bool(r["cardio_related"]), r

def make_prediction(patient: dict, drug_name: str, threshold: float = 0.5, auto=True, manual=(0.15,0), na_rule=True):
    # derive drug features
    if auto:
        drug_key, base_risk, kw_flag, cardio_related, src = get_drug_features_from_index(drug_name)
    else:
        drug_key = norm_drug_name(drug_name)
        base_risk, kw_flag = manual
        cardio_related = (kw_flag == 1) or (base_risk > 0.2)
        src = None

    # Build single-row DataFrame with correct columns
    feats = {
        "age": patient["age"],
        "hr": patient["hr"],
        "qtc": patient["qtc"],
        "ef": patient["ef"],
        "sbp": patient["sbp"],
        "dbp": patient["dbp"],
        "k": patient["k"],
        "mg": patient["mg"],
        "creatinine": patient["creatinine"],
        "hx_arrhythmia": int(patient["hx_arrhythmia"]),
        "hx_hf": int(patient["hx_hf"]),
        "hx_cad": int(patient["hx_cad"]),
        "drug_base_risk": float(base_risk),
        "kw_flag": int(kw_flag),
        "sex": patient["sex"],
    }
    X = pd.DataFrame([feats])[FEATURE_ORDER]

    # Model probability
    prob = float(model.predict_proba(X)[0, 1])
    decision = "unsafe" if prob >= threshold else "safe"

    # NOT APPLICABLE rule (separate from model; transparent)
    not_app = False
    if na_rule and kw_flag == 0 and base_risk <= 0.15:
        not_app = True
        decision = "not applicable"

    # Build explanation (transparent, patient-specific — not hard-coded drug map)
    reasons = []
    if feats["qtc"] > 450: reasons.append(f"prolonged QTc ({feats['qtc']}ms)")
    if feats["ef"] < 40: reasons.append(f"low EF ({feats['ef']}%)")
    if feats["k"] < 3.5: reasons.append(f"low potassium ({feats['k']})")
    if feats["mg"] < 1.7: reasons.append(f"low magnesium ({feats['mg']})")
    if feats["hx_arrhythmia"] == 1: reasons.append("history of arrhythmia")
    if feats["hx_hf"] == 1: reasons.append("history of heart failure")
    if feats["age"] > 65: reasons.append(f"age >65 ({feats['age']})")
    if kw_flag == 1: reasons.append("cardio-related drug signals (keywords)")
    if base_risk > 0.5: reasons.append("elevated label-derived base risk")

    return {
        "drug_key": drug_key,
        "prob_unsafe": round(prob, 3),
        "decision": decision,
        "not_applicable_flag": not_app,
        "explanation": "risk drivers: " + (", ".join(reasons) if reasons else "none prominent"),
        "features_used": feats
    }

if submitted:
    patient_data = {
        "sex": sex, "age": age, "hr": hr, "qtc": qtc, "ef": ef,
        "sbp": sbp, "dbp": dbp, "k": k, "mg": mg, "creatinine": creatinine,
        "hx_arrhythmia": int(hx_arrhythmia),
        "hx_hf": int(hx_hf),
        "hx_cad": int(hx_cad)
    }
    result = make_prediction(
        patient=patient_data,
        drug_name=drug_name,
        threshold=0.5,
        auto=use_auto,
        manual=(manual_base, int(manual_kw)),
        na_rule=not_app_rule
    )

    st.subheader("Results")
    if result["decision"] == "unsafe":
        risk_color = "red"
    elif result["decision"] == "safe":
        risk_color = "green"
    else:
        risk_color = "gray"

    st.markdown(
        f"**Decision:** <span style='color:{risk_color};font-size:22px'>{result['decision'].upper()}</span>",
        unsafe_allow_html=True
    )
    st.metric("Risk Probability (model)", f"{result['prob_unsafe']*100:.1f}%")

    with st.expander("Details"):
        st.write(result["explanation"])
        st.json(result, expanded=False)

st.caption("Tip: upload dictrank_dataset.csv in the sidebar to auto-derive drug features (base risk + keywords).")
