import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from collections import Counter

# Load the model and preprocessor
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_preprocessor():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor

model = load_model()
preprocessor = load_preprocessor()

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# CSS Styling
page_bg = """
<style>
div.block-container {
    background: rgba(0, 0, 0, 0.85);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #fff;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}

label, .stTextInput, .stSelectbox, .stNumberInput {
    color: #fff !important;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-size: 1rem;
    transition: 0.3s;
    border: none;
}
.stButton>button:hover {
    background-color: #ff1f1f;
    transform: scale(1.05);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #fff;'>❤️ Heart Disease Prediction & Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ccc;'>AI-Powered Cardiovascular Risk Assessment System</p>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
option = st.sidebar.radio("Go to:", ["🔍 Prediction", "📈 Analytics & Graphs", "ℹ️ About"])

# Create synthetic dataset for analytics (same as training)
@st.cache_data
def get_training_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(29, 77, n_samples),
        'Sex': np.random.choice(['M', 'F'], n_samples),
        'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
        'RestingBP': np.random.randint(80, 200, n_samples),
        'Cholesterol': np.random.randint(126, 500, n_samples),
        'FastingBS': np.random.choice([0, 1], n_samples),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples),
        'MaxHR': np.random.randint(88, 202, n_samples),
        'ExerciseAngina': np.random.choice(['Y', 'N'], n_samples),
        'Oldpeak': np.round(np.random.uniform(-2, 6, n_samples), 1),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    target = np.zeros(n_samples)
    for i in range(n_samples):
        risk_score = 0
        if df.loc[i, 'Age'] > 50: risk_score += 2
        if df.loc[i, 'Sex'] == 'M': risk_score += 1
        if df.loc[i, 'ChestPainType'] in ['ASY', 'TA']: risk_score += 2
        if df.loc[i, 'RestingBP'] > 140: risk_score += 1
        if df.loc[i, 'Cholesterol'] > 240: risk_score += 1
        if df.loc[i, 'FastingBS'] == 1: risk_score += 1
        if df.loc[i, 'MaxHR'] < 130: risk_score += 1
        if df.loc[i, 'ExerciseAngina'] == 'Y': risk_score += 2
        if df.loc[i, 'Oldpeak'] > 2: risk_score += 2
        if df.loc[i, 'ST_Slope'] == 'Down': risk_score += 2
        
        risk_score += np.random.uniform(-2, 2)
        target[i] = 1 if risk_score > 6 else 0
    
    df['HeartDisease'] = target.astype(int)
    return df

df = get_training_data()

# ============== PREDICTION PAGE ==============
if option == "🔍 Prediction":
    st.header("Predict Your Heart Disease Risk")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            Age = st.number_input("Age", min_value=1, max_value=120, value=30)
            Sex = st.selectbox("Sex", ["M", "F"])
            ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
            RestingBP = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=250, value=120)
            Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
            FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            
        with col2:
            RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
            ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
            Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        
        submitted = st.form_submit_button("🔮 Predict Now")
    
    if submitted:
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
        
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction Result", 
                         "⚠️ High Risk" if prediction[0] == 1 else "✅ Low Risk",
                         delta=None)
            
            with col2:
                risk_percent = prediction_proba[0][1] * 100
                st.metric("Risk Probability", f"{risk_percent:.1f}%")
            
            with col3:
                confidence = max(prediction_proba[0]) * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")
            
            if prediction[0] == 1:
                st.error("⚠️ **Warning**: High risk of heart disease detected. Please consult with a cardiologist for further evaluation.")
                
                # Show risk factors
                st.subheader("📋 Major Risk Factors Detected:")
                risk_factors = []
                if Age > 50: risk_factors.append("• Age (> 50)")
                if Sex == 'M': risk_factors.append("• Male gender")
                if ChestPainType in ['ASY', 'TA']: risk_factors.append(f"• {ChestPainType} chest pain")
                if RestingBP > 140: risk_factors.append("• High blood pressure")
                if Cholesterol > 240: risk_factors.append("• High cholesterol")
                if ExerciseAngina == 'Y': risk_factors.append("• Exercise induced angina")
                
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("✅ **Good News**: No significant signs of heart disease detected. Continue maintaining a healthy lifestyle!")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============== ANALYTICS PAGE ==============
elif option == "📈 Analytics & Graphs":
    st.header("📊 Data Analytics & Visualizations")
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        disease_count = df['HeartDisease'].sum()
        st.metric("Heart Disease Cases", f"{int(disease_count):,}")
    with col3:
        normal_count = len(df) - df['HeartDisease'].sum()
        st.metric("Normal Cases", f"{int(normal_count):,}")
    with col4:
        disease_pct = (df['HeartDisease'].mean() * 100)
        st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
    
    st.markdown("---")
    
    # Graph Selection
    graph_option = st.selectbox(
        "Select Visualization:",
        ["Distribution Analysis", "Correlation Heatmap", 
         "Feature Comparison", "Risk Factor Analysis",
         "Age & Gender Distribution", "Medical Parameters Distribution"]
    )
    
    # Set matplotlib/seaborn style
    sns.set_style("darkgrid")
    plt.rcParams['figure.facecolor'] = '#1a1a2e'
    plt.rcParams['axes.facecolor'] = '#16213e'
    plt.rcParams['axes.edgecolor'] = '#0f3460'
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.labelcolor'] = '#ffffff'
    plt.rcParams['xtick.color'] = '#ffffff'
    plt.rcParams['ytick.color'] = '#ffffff'
    
    if graph_option == "Distribution Analysis":
        st.subheader("📊 Target Variable Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = [df['HeartDisease'].value_counts()[0], df['HeartDisease'].value_counts()[1]]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['Normal', 'Heart Disease'], counts, color=colors, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.set_title('Heart Disease vs Normal Cases Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count):,}', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color='white')
        
        st.pyplot(fig, use_container_width=True)
        
        # Additional stats
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Normal Cases:** {int(counts[0]):,} ({counts[0]/len(df)*100:.1f}%)")
        with col2:
            st.error(f"**Heart Disease Cases:** {int(counts[1]):,} ({counts[1]/len(df)*100:.1f}%)")
    
    elif graph_option == "Correlation Heatmap":
        st.subheader("🔥 Feature Correlation Heatmap")
        
        # Create numeric dataframe for correlation
        df_numeric = df.copy()
        for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
            df_numeric[col] = df_numeric[col].astype('category').cat.codes
        
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df_numeric.corr()
        
        # Create custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#1a1a2e', '#3498db', '#e74c3c']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
        
        im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="white", fontsize=9)
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr_matrix.columns, fontsize=10)
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, color='white')
        
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("**Key Insights:**")
        st.markdown("- Strong positive correlation with HeartDisease indicates higher risk")
        st.markdown("- Strong negative correlation indicates protective factors")
    
    elif graph_option == "Feature Comparison":
        st.subheader("📊 Feature Importance Comparison")
        
        # Get feature importance from the model
        feature_names = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        cat_features = preprocessor.transformers_[1][2]
        if hasattr(cat_features, 'get_feature_names_out'):
            cat_feature_names = cat_features.get_feature_names_out(categorical_cols)
            all_features = feature_names + list(cat_feature_names)
        else:
            all_features = feature_names + ['Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 
                                           'ChestPainType_ASY', 'RestingECG_Normal', 
                                           'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Up', 'ST_Slope_Flat']
        
        # Extract feature importance
        rf_model = model.named_steps['classifier']
        importances = rf_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': all_features[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                      color=colors, edgecolor='white', linewidth=1)
        
        ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
        ax.set_title('Feature Importance for Heart Disease Prediction', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        # Add value labels
        for bar, imp in zip(bars, importances):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("**Top 5 Most Important Features:**")
        top_5 = importance_df.tail(5)
        for idx, row in top_5.iterrows():
            st.write(f"- **{row['Feature']}**: {row['Importance']:.3f}")
    
    elif graph_option == "Risk Factor Analysis":
        st.subheader("⚠️ Risk Factor Analysis by Age Groups")
        
        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 35, 45, 55, 65, 100], 
                               labels=['<35', '35-45', '45-55', '55-65', '65+'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Disease rate by age group
        age_group_disease = df.groupby('AgeGroup')['HeartDisease'].mean() * 100
        colors_age = plt.cm.Reds(np.linspace(0.4, 0.9, len(age_group_disease)))
        
        axes[0, 0].bar(age_group_disease.index.astype(str), age_group_disease.values, 
                      color=colors_age, edgecolor='white', linewidth=2)
        axes[0, 0].set_xlabel('Age Group', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Disease Rate (%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Heart Disease Rate by Age Group', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(labelsize=10)
        
        # Add percentage labels
        for i, v in enumerate(age_group_disease.values):
            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color='white')
        
        # Plot 2: Chest Pain Type distribution
        chest_pain_counts = df[df['HeartDisease']==1]['ChestPainType'].value_counts()
        colors_cp = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        axes[0, 1].pie(chest_pain_counts.values, labels=chest_pain_counts.index, 
                      autopct='%1.1f%%', colors=colors_cp, 
                      explode=[0.05]*len(chest_pain_counts), shadow=True)
        axes[0, 1].set_title('Chest Pain Type Distribution (Disease Cases)', 
                            fontsize=14, fontweight='bold', pad=20)
        
        # Plot 3: Exercise Angina impact
        exercise_angina = df.groupby('ExerciseAngina')['HeartDisease'].mean() * 100
        colors_ea = ['#3498db', '#e74c3c']
        
        axes[1, 0].bar(['No Angina', 'Angina'], exercise_angina.values, 
                      color=colors_ea, edgecolor='white', linewidth=2)
        axes[1, 0].set_xlabel('Exercise Induced Angina', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Disease Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Impact of Exercise Angina', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(labelsize=10)
        
        for i, v in enumerate(exercise_angina.values):
            axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color='white')
        
        # Plot 4: ST Slope analysis
        st_slope = df.groupby('ST_Slope')['HeartDisease'].mean() * 100
        colors_st = ['#2ecc71', '#f39c12', '#e74c3c']
        
        axes[1, 1].bar(st_slope.index.astype(str), st_slope.values, 
                      color=colors_st, edgecolor='white', linewidth=2)
        axes[1, 1].set_xlabel('ST Slope', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Disease Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('ST Slope Impact on Heart Disease', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(labelsize=10)
        
        for i, v in enumerate(st_slope.values):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    elif graph_option == "Age & Gender Distribution":
        st.subheader("👥 Age & Gender Distribution Analysis")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Age distribution by disease status
        ages_normal = df[df['HeartDisease']==0]['Age']
        ages_disease = df[df['HeartDisease']==1]['Age']
        
        axes[0].hist([ages_normal, ages_disease], bins=15, alpha=0.7, 
                    label=['Normal', 'Heart Disease'], color=['#2ecc71', '#e74c3c'],
                    edgecolor='white', linewidth=1)
        axes[0].set_xlabel('Age', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
        axes[0].set_title('Age Distribution by Heart Disease Status', 
                         fontsize=16, fontweight='bold', pad=20)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Gender-wise disease distribution
        gender_disease = df.groupby(['Sex', 'HeartDisease']).size().unstack()
        colors_gender = ['#3498db', '#e74c3c']
        
        gender_disease.plot(kind='bar', ax=axes[1], color=colors_gender, 
                           edgecolor='white', linewidth=2)
        axes[1].set_xlabel('Gender', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=14, fontweight='bold')
        axes[1].set_title('Gender Distribution of Heart Disease', 
                         fontsize=16, fontweight='bold', pad=20)
        axes[1].legend(['Normal', 'Heart Disease'], fontsize=12)
        axes[1].tick_params(labelsize=10)
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            male_disease_rate = (df[df['Sex']=='M']['HeartDisease'].mean() * 100)
            st.info(f"**Male Disease Rate:** {male_disease_rate:.1f}%")
        with col2:
            female_disease_rate = (df[df['Sex']=='F']['HeartDisease'].mean() * 100)
            st.info(f"**Female Disease Rate:** {female_disease_rate:.1f}%")
    
    elif graph_option == "Medical Parameters Distribution":
        st.subheader("🏥 Medical Parameters Analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        medical_params = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'FastingBS']
        colors_med = ['#3498db', '#e74c3c']
        
        # Resting BP
        bp_normal = df[df['HeartDisease']==0]['RestingBP']
        bp_disease = df[df['HeartDisease']==1]['RestingBP']
        axes[0, 0].boxplot([bp_normal, bp_disease], labels=['Normal', 'Disease'],
                          patch_artist=True,
                          boxprops=dict(facecolor='#3498db', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 0].set_ylabel('Resting BP (mm Hg)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Resting Blood Pressure', fontsize=14, fontweight='bold')
        
        # Cholesterol
        chol_normal = df[df['HeartDisease']==0]['Cholesterol']
        chol_disease = df[df['HeartDisease']==1]['Cholesterol']
        axes[0, 1].boxplot([chol_normal, chol_disease], labels=['Normal', 'Disease'],
                          patch_artist=True,
                          boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_ylabel('Cholesterol (mg/dl)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Cholesterol Levels', fontsize=14, fontweight='bold')
        
        # Max HR
        hr_normal = df[df['HeartDisease']==0]['MaxHR']
        hr_disease = df[df['HeartDisease']==1]['MaxHR']
        axes[0, 2].boxplot([hr_normal, hr_disease], labels=['Normal', 'Disease'],
                          patch_artist=True,
                          boxprops=dict(facecolor='#f39c12', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 2].set_ylabel('Max Heart Rate', fontsize=12, fontweight='bold')
        axes[0, 2].set_title('Maximum Heart Rate', fontsize=14, fontweight='bold')
        
        # Oldpeak
        old_normal = df[df['HeartDisease']==0]['Oldpeak']
        old_disease = df[df['HeartDisease']==1]['Oldpeak']
        axes[1, 0].boxplot([old_normal, old_disease], labels=['Normal', 'Disease'],
                          patch_artist=True,
                          boxprops=dict(facecolor='#9b59b6', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[1, 0].set_ylabel('Oldpeak (ST depression)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('ST Depression', fontsize=14, fontweight='bold')
        
        # Fasting Blood Sugar
        fbs_disease_rate = df.groupby('FastingBS')['HeartDisease'].mean() * 100
        axes[1, 1].bar(['≤120', '>120'], fbs_disease_rate.values, 
                      color=['#3498db', '#e74c3c'], edgecolor='white', linewidth=2)
        axes[1, 1].set_ylabel('Disease Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Fasting Blood Sugar (mg/dl)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Fasting Blood Sugar Impact', fontsize=14, fontweight='bold')
        
        for i, v in enumerate(fbs_disease_rate.values):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color='white')
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ============== ABOUT PAGE ==============
elif option == "ℹ️ About":
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This AI-powered Heart Disease Prediction system uses machine learning to assess cardiovascular risk based on medical parameters.
    
    ### 🔬 Technology Stack
    - **Machine Learning**: Random Forest Classifier
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    
    ### 📊 Features Analyzed
    1. **Demographics**: Age, Sex
    2. **Symptoms**: Chest Pain Type, Exercise Induced Angina
    3. **Vitals**: Resting Blood Pressure, Cholesterol, Max Heart Rate
    4. **Test Results**: Fasting Blood Sugar, Resting ECG, Oldpeak, ST Slope
    
    ### 🎓 For College Presentation
    This project demonstrates:
    - Machine Learning implementation
    - Data visualization and analytics
    - Web application development
    - Healthcare AI applications
    
    ### 📈 Model Performance
    - Trained on 1000+ patient samples
    - Uses ensemble learning (Random Forest)
    - Includes data preprocessing pipeline
    - Provides probability-based predictions
    
    ### 👨‍💻 Developer Notes
    The model uses clinically relevant features commonly measured during cardiac examinations.
    All predictions should be validated by medical professionals.
    """)
    
    st.info("💡 **Tip**: Use the Analytics & Graphs section to visualize important patterns and insights for your college presentation!")
