import streamlit as st
import requests
import pandas as pd
import base64

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# For now, use a simple color scheme without background image
# You can add your own image path later
img_path = None  # Set to your image path if you want background
base64_img = None

if img_path:
    try:
        base64_img = get_base64(img_path)
    except:
        base64_img = None

# ------------ ANIMATED BACKGROUND WITH PARTICLES ------------
page_bg = """
<style>
/* Animated Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0a0e27, #1a1a3e, #2d1b4e, #0f2027, #1a2a4a, #2a1a3e);
    background-size: 600% 600%;
    animation: gradientShift 20s ease infinite;
    position: relative;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    25% { background-position: 50% 100%; }
    50% { background-position: 100% 50%; }
    75% { background-position: 50% 0%; }
    100% { background-position: 0% 50%; }
}

/* Animated Particle Effects */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 40% 20%, rgba(79, 195, 247, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 70% 30%, rgba(255, 107, 107, 0.1) 0%, transparent 40%);
    animation: floatParticles 15s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes floatParticles {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -30px) scale(1.1); }
    66% { transform: translate(-20px, 20px) scale(0.9); }
}

/* Glowing Orbs Effect */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 800px;
    height: 800px;
    background: radial-gradient(circle, rgba(102, 126, 234, 0.08) 0%, transparent 70%);
    animation: pulseGlow 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes pulseGlow {
    0%, 100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
    50% { opacity: 0.6; transform: translate(-50%, -50%) scale(1.2); }
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}

/* Modern Professional Form Container */
div.block-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    padding: 2.5rem;
    box-shadow: 0 4px 24px 0 rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #fff;
    position: relative;
    z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
}

/* Beautiful Input Fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #fff !important;
    padding: 8px 12px !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stNumberInput > div > div > input:focus {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 15px rgba(79, 195, 247, 0.5) !important;
    background: rgba(255, 255, 255, 0.12) !important;
}

/* Labels */
label {
    color: #4fc3f7 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

/* Professional Submit Button */
.stButton>button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%) !important;
    background-size: 200% 200% !important;
    color: white !important;
    border-radius: 50px !important;
    padding: 1.2rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) !important;
    border: 3px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4),
                0 0 0 0 rgba(240, 147, 251, 0.5),
                inset 0 -3px 8px rgba(0, 0, 0, 0.2) !important;
    width: 100% !important;
    max-width: 500px !important;
    margin: 0 auto !important;
    position: relative !important;
    overflow: hidden !important;
    animation: gradientFlow 3s ease infinite !important;
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stButton>button::before {
    content: '' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: 0 !important;
    height: 0 !important;
    border-radius: 50% !important;
    background: rgba(255, 255, 255, 0.3) !important;
    transform: translate(-50%, -50%) !important;
    transition: width 0.7s, height 0.7s !important;
}

.stButton>button:hover::before {
    width: 500px !important;
    height: 500px !important;
}

.stButton>button:hover {
    transform: translateY(-6px) scale(1.03) !important;
    box-shadow: 0 15px 40px rgba(245, 87, 108, 0.6),
                0 0 30px rgba(79, 172, 254, 0.5),
                0 0 60px rgba(240, 147, 251, 0.3) !important;
    border-color: rgba(255, 255, 255, 0.6) !important;
    letter-spacing: 1.5px !important;
}

.stButton>button:active {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 8px 20px rgba(245, 87, 108, 0.5),
                0 0 20px rgba(79, 172, 254, 0.4) !important;
    transition: all 0.1s ease !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 39, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Success/Error Boxes */
.stSuccess {
    background: rgba(0, 176, 155, 0.2) !important;
    border: 1px solid rgba(0, 176, 155, 0.5) !important;
    border-radius: 15px !important;
}

.stError {
    background: rgba(255, 65, 108, 0.2) !important;
    border: 1px solid rgba(255, 65, 108, 0.5) !important;
    border-radius: 15px !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

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
    # Personal Information Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(79, 195, 247, 0.15) 0%, rgba(79, 195, 247, 0.05) 100%); padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #4fc3f7; box-shadow: 0 4px 12px rgba(79, 195, 247, 0.2);'>
        <h3 style='color: #4fc3f7; margin: 0; font-size: 1.15rem; font-weight: 600; letter-spacing: 0.5px;'>PERSONAL INFORMATION</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, help="Enter your age in years")
        Sex = st.selectbox("Gender", ["M - Male", "F - Female"], help="Select your biological sex")
        ChestPainType = st.selectbox("Chest Pain Type", 
                                     ["ATA - Atypical Angina", "NAP - Non-Anginal Pain", "ASY - Asymptomatic", "TA - Typical Angina"],
                                     help="Type of chest pain experienced")

    with col2:
        RestingBP = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=250, value=120, help="Normal range: 90-120 mmHg")
        Cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=200, help="Normal range: < 200 mg/dL")
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                 ["0 - No (Normal)", "1 - Yes (Elevated)"],
                                 help="Is fasting blood sugar greater than 120 mg/dl?")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Cardiac Measurements Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 75, 75, 0.05) 100%); padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #ff4b4b; box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);'>
        <h3 style='color: #ff4b4b; margin: 0; font-size: 1.15rem; font-weight: 600; letter-spacing: 0.5px;'>CARDIAC MEASUREMENTS</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)

    with col3:
        RestingECG = st.selectbox("Resting Electrocardiogram", 
                                  ["Normal - Normal", "ST - ST-T Wave Abnormality", "LVH - Left Ventricular Hypertrophy"],
                                  help="Results of resting ECG")
        MaxHR = st.number_input("Maximum Heart Rate (bpm)", min_value=60, max_value=220, value=150, help="Maximum heart rate achieved during exercise")

    with col4:
        ExerciseAngina = st.selectbox("Exercise Induced Angina", 
                                      ["N - No", "Y - Yes"],
                                      help="Chest pain during exercise")
        Oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise")
        ST_Slope = st.selectbox("ST Segment Slope", 
                                ["Up - Upsloping", "Flat - Flat", "Down - Downsloping"],
                                help="Slope of the peak exercise ST segment")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    submitted = st.form_submit_button("Analyze Cardiovascular Risk")

# ------------ API CALL ------------
if submitted:
    # Extract clean values from the formatted inputs
    sex_value = Sex.split(" - ")[0]
    chest_pain_value = ChestPainType.split(" - ")[0]
    fasting_bs_value = int(FastingBS.split(" - ")[0])
    ecg_value = RestingECG.split(" - ")[0]
    angina_value = ExerciseAngina.split(" - ")[0]
    st_slope_value = ST_Slope.split(" - ")[0]
    
    input_data = {
        "Age": int(Age),
        "Sex": sex_value,
        "ChestPainType": chest_pain_value,
        "RestingBP": int(RestingBP),
        "Cholesterol": int(Cholesterol),
        "FastingBS": fasting_bs_value,
        "RestingECG": ecg_value,
        "MaxHR": int(MaxHR),
        "ExerciseAngina": angina_value,
        "Oldpeak": float(Oldpeak),
        "ST_Slope": st_slope_value
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()["prediction"]
            
            if result == 0:
                # Normal Result - Attractive Display
                st.markdown("""
                <div style="background: linear-gradient(135deg, #00b09b, #96c93d); padding: 2rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <h2 style="color: white; text-align: center; margin-bottom: 0.5rem;">🎉 Excellent News!</h2>
                    <h3 style="color: white; text-align: center; margin-top: 0;">✅ Normal — No signs of heart disease detected.</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                ### 💚 Your Heart Health Status: GOOD
                
                **Risk Level:** 🟢 Low Risk
                
                **What This Means:**
                - Your current health indicators show no significant signs of heart disease
                - Your cardiovascular system appears to be functioning well
                - Continue maintaining your healthy lifestyle
                
                **Recommendations to Stay Healthy:**
                - 🏃‍♂️ Maintain regular physical activity (150 mins/week moderate exercise)
                - 🥗 Continue eating a balanced, heart-healthy diet
                - 😴 Ensure 7-9 hours of quality sleep daily
                - 🧘‍♀️ Manage stress through relaxation techniques
                - 🩺 Schedule regular health check-ups annually
                - 🚭 Avoid smoking and limit alcohol consumption
                
                **Healthy Ranges to Maintain:**
                - Blood Pressure: < 120/80 mmHg
                - Cholesterol: < 200 mg/dL
                - Resting Heart Rate: 60-100 bpm
                - Fasting Blood Sugar: < 100 mg/dL
                """)
                
            else:
                # High Risk Result - Attractive Display
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ff416c, #ff4b2b); padding: 2rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <h2 style="color: white; text-align: center; margin-bottom: 0.5rem;">⚠️ Attention Required</h2>
                    <h3 style="color: white; text-align: center; margin-top: 0;">🚨 Please consult with a doctor — chances of heart attack detected.</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                ### ❤️‍🩹 Your Heart Health Status: AT RISK
                
                **Risk Level:** 🔴 High Risk
                
                **What This Means:**
                - Your health indicators suggest potential heart disease risk factors
                - Immediate medical consultation is strongly recommended
                - Early detection and intervention can significantly improve outcomes
                
                **Immediate Actions to Take:**
                - 🏥 Schedule an appointment with a cardiologist within 1-2 weeks
                - 📋 Bring this report and your complete medical history
                - 💊 Follow your doctor's recommendations for further testing
                - 📊 Monitor your blood pressure and heart rate regularly
                
                **Lifestyle Changes to Implement:**
                - 🚭 Quit smoking immediately (if applicable)
                - 🥗 Adopt a heart-healthy diet (Mediterranean or DASH diet)
                - 🏃‍♂️ Start light exercise only after doctor's approval
                - 😴 Prioritize quality sleep (7-9 hours)
                - 🧘‍♂️ Practice stress reduction techniques daily
                - 🍷 Limit alcohol consumption
                
                **Warning Signs to Watch For:**
                - Chest pain or discomfort
                - Shortness of breath
                - Pain in neck, jaw, throat, upper abdomen or back
                - Nausea, lightheadedness, or cold sweats
                - Unusual fatigue or weakness
                
                **If you experience any of these symptoms, seek emergency medical attention immediately!**
                
                **Target Health Goals:**
                - Blood Pressure: < 130/80 mmHg
                - Cholesterol: < 180 mg/dL
                - Resting Heart Rate: 60-100 bpm
                - Fasting Blood Sugar: < 100 mg/dL
                - Weight: Maintain healthy BMI (18.5-24.9)
                """)
                
        else:
            st.warning("⚠️ Something went wrong. Try again later.")
    except Exception as e:
        st.error(f"❌ Error connecting to API: {e}")
