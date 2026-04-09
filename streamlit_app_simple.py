import streamlit as st
import pickle
import pandas as pd

# Load the model directly
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# ------------ TITLE ------------
st.markdown("""
<div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem;'>
    <h1 style='color: #fff; font-size: 2.5rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
        Cardiovascular Health Assessment System
    </h1>
    <p style='color: #4fc3f7; font-size: 1.2rem; margin-top: 0; font-weight: 300;'>
        Advanced AI-Powered Heart Disease Risk Prediction
    </p>
    <div style='width: 100px; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem auto; border-radius: 2px;'></div>
</div>
""", unsafe_allow_html=True)

# ------------ FORM ------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)
        Sex = st.selectbox("Gender", ["M", "F"])
        ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        RestingBP = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=250, value=120)
        Cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        MaxHR = st.number_input("Maximum Heart Rate (bpm)", min_value=60, max_value=220, value=150)
        ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
        Oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        ST_Slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Analyze Cardiovascular Risk")

# ------------ PREDICTION ------------
if submitted:
    try:
        input_data = {
            "Age": int(Age),
            "Sex": Sex,
            "ChestPainType": ChestPainType,
            "RestingBP": int(RestingBP),
            "Cholesterol": int(Cholesterol),
            "FastingBS": int(FastingBS),
            "RestingECG": RestingECG,
            "MaxHR": int(MaxHR),
            "ExerciseAngina": ExerciseAngina,
            "Oldpeak": float(Oldpeak),
            "ST_Slope": ST_Slope
        }
        
        # Create DataFrame and predict
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        result = int(prediction)

        if result == 0:
            st.success("✅ Normal — No signs of heart disease detected.")
            st.info("Your heart health looks good! Maintain a healthy lifestyle.")
        else:
            st.error("⚠️ Please consult with a doctor — chances of heart disease detected.")
            st.warning("Early detection can help prevent serious complications.")
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.error("Please check your input values and try again.")
