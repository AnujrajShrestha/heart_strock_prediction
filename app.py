import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("""
This application uses a Machine Learning model (Logistic Regression) trained on a heart disease dataset to predict the likelihood of a patient having heart disease based on clinical parameters.
""")

# Paths for pickled files
MODEL_PATH = "Logistic Regression_heart.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

model, scaler = load_model_and_scaler()

# Fallback: Train model automatically if pickles don't exist
if model is None or scaler is None:
    st.warning("⚠️ Pre-trained model or scaler not found in the directory. Let's train the model now if you have the 'heart.csv' dataset!")
    uploaded_file = st.file_uploader("Upload 'heart.csv' to train the model", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing matching the notebook exactly
            ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
            df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean).round(2)
            
            resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
            df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean).round(2)
            
            df_encode = pd.get_dummies(df, drop_first=True).astype(int)
            
            X = df_encode.drop('HeartDisease', axis=1)
            y = df_encode['HeartDisease']
            
            # Numeric scaling applied exactly as per notebook (Oldpeak is intentionally omitted from scaling)
            numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']
            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numerical_cols] = scaler.fit_transform(X_scaled[numerical_cols])
            
            model = LogisticRegression()
            model.fit(X_scaled, y)
            
            # Save the trained model assets
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            st.success("🎉 Model trained and saved successfully! Refreshing application...")
            st.rerun()
        except Exception as e:
            st.error(f"Error training model: {e}")
    else:
        st.info("Please place 'Logistic Regression_heart.pkl' and 'scaler.pkl' in the directory, or upload 'heart.csv' above.")
        st.stop()

# Sidebar layout for user input
st.sidebar.header("📋 Patient Clinical Data Input")

age = st.sidebar.slider("Age", min_value=1, max_value=120, value=54, step=1)

sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
sex_M = 1 if sex == "Male" else 0

cp_type = st.sidebar.selectbox(
    "Chest Pain Type", 
    options=["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"]
)
cp_ata = 1 if cp_type == "Atypical Angina (ATA)" else 0
cp_nap = 1 if cp_type == "Non-Anginal Pain (NAP)" else 0
cp_ta = 1 if cp_type == "Typical Angina (TA)" else 0

resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
cholesterol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=650, value=220)

fasting_bs_input = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
fasting_bs = 1 if fasting_bs_input == "Yes" else 0

rest_ecg = st.sidebar.selectbox(
    "Resting Electrocardiogram Results", 
    options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy (LVH)"]
)
ecg_normal = 1 if rest_ecg == "Normal" else 0
ecg_st = 1 if rest_ecg == "ST-T wave abnormality" else 0

max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=140)

ex_angina = st.sidebar.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
ex_angina_Y = 1 if ex_angina == "Yes" else 0

oldpeak = st.sidebar.slider("Oldpeak (ST depression induced by exercise)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)

st_slope = st.sidebar.selectbox(
    "The Slope of the Peak Exercise ST Segment", 
    options=["Upsloping", "Flat", "Downsloping"]
)
slope_flat = 1 if st_slope == "Flat" else 0
slope_up = 1 if st_slope == "Upsloping" else 0

# Construct prediction row DataFrame matching feature order exactly
input_data = pd.DataFrame([{
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': sex_M,
    'ChestPainType_ATA': cp_ata,
    'ChestPainType_NAP': cp_nap,
    'ChestPainType_TA': cp_ta,
    'RestingECG_Normal': ecg_normal,
    'RestingECG_ST': ecg_st,
    'ExerciseAngina_Y': ex_angina_Y,
    'ST_Slope_Flat': slope_flat,
    'ST_Slope_Up': slope_up
}])

# Main Dashboard Layout splits screen into info summary & prediction results
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Summary of Inputted Patient Information")
    summary_df = pd.DataFrame({
        "Parameter": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", "Fasting BS > 120", "Resting ECG", "Max HR", "Exercise Angina", "Oldpeak", "ST Slope"],
        "Value": [age, sex, cp_type, f"{resting_bp} mm Hg", f"{cholesterol} mg/dl", fasting_bs_input, rest_ecg, max_hr, ex_angina, oldpeak, st_slope]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Prediction Result")
    
    # Scale only the designated numerical columns matching the notebook's approach
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']
    input_scaled = input_data.copy()
    input_scaled[numerical_cols] = scaler.transform(input_scaled[numerical_cols])
    
    # Run the prediction pipeline
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    # Display probability metric
    st.metric(label="Heart Disease Probability", value=f"{prediction_proba*100:.1f}%")
    
    if prediction == 1:
        st.error("🚨 **High Risk**: The model predicts a high likelihood of Heart Disease.")
    else:
        st.success("✅ **Low Risk**: The model predicts a low likelihood of Heart Disease.")

st.markdown("---")
st.subheader("Model Performance Context (From Notebook Evaluation)")
metrics_data = {
    'Model Name': ['Logistic Regression', 'KNN', 'Decision Tree', 'SVM', 'Naive Bayes'],
    'Accuracy': ['86.96%', '85.87%', '78.80%', '85.33%', '84.78%'],
    'F1 Score': ['88.46%', '87.74%', '80.40%', '87.32%', '86.14%']
}
st.table(pd.DataFrame(metrics_data))
st.caption("Note: This app leverages Logistic Regression as it yielded the highest overall performance on cross-validation sets.")