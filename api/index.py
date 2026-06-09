from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse  # <-- ADD THIS IMPORT
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(title="Heart Disease Prediction API")

handler = app

# Configure CORS Middleware (Kept for flexibility if testing externally)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute directory path of the current file (api/ folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to the root directory where the .pkl binaries are
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "Logistic Regression_heart.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "scaler.pkl")

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Binaries missing.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


class PatientData(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Age of the patient", example=54)
    Sex: str = Field(..., description="Sex of the patient ('Male' or 'Female')", example="Male")
    ChestPainType: str = Field(..., description="Chest Pain Type ('TA', 'ATA', 'NAP', 'ASY')", example="ATA")
    RestingBP: float = Field(..., ge=50, le=250, description="Resting Blood Pressure in mm Hg", example=130.0)
    Cholesterol: float = Field(..., ge=50, le=650, description="Serum Cholesterol in mg/dl", example=220.0)
    FastingBS: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", example=0)
    RestingECG: str = Field(..., description="Resting ECG results ('Normal', 'ST', 'LVH')", example="Normal")
    MaxHR: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved", example=140)
    ExerciseAngina: str = Field(..., description="Exercise-induced angina ('Y' or 'N')", example="N")
    Oldpeak: float = Field(..., ge=-3.0, le=7.0, description="Oldpeak - ST depression", example=0.0)
    ST_Slope: str = Field(..., description="The slope of the peak exercise ST segment ('Up', 'Flat', 'Down')", example="Up")


MODEL_METRICS = {
    "algorithm": "Logistic Regression",
    "accuracy": 86.96,
    "f1_score": 88.46,
    "total_features": 15
}

# ---- UPDATE THIS ENDPOINT TO SERVE YOUR HTML ----
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the index.html frontend page directly at the root URL."""
    html_path = os.path.join(ROOT_DIR, "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Frontend index.html template file not found.")
        
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/metrics")
def get_model_metrics():
    """Returns classification training metrics generated during the notebook pipeline run."""
    return MODEL_METRICS


@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model assets are uninitialized.")

    try:
        # 1. Map string-based categorical parameters to One-Hot encoded binary positions
        sex_M = 1 if patient.Sex.strip().upper() == "MALE" else 0
        
        cp_ata = 1 if patient.ChestPainType.strip().upper() == "ATA" else 0
        cp_nap = 1 if patient.ChestPainType.strip().upper() == "NAP" else 0
        cp_ta = 1 if patient.ChestPainType.strip().upper() == "TA" else 0
        
        ecg_normal = 1 if patient.RestingECG.strip().upper() == "NORMAL" else 0
        ecg_st = 1 if patient.RestingECG.strip().upper() == "ST" else 0
        
        ex_angina_Y = 1 if patient.ExerciseAngina.strip().upper() == "Y" else 0
        
        slope_flat = 1 if patient.ST_Slope.strip().upper() == "FLAT" else 0
        slope_up = 1 if patient.ST_Slope.strip().upper() == "UP" else 0

        # 2. Re-create the evaluation layout containing exactly 15 columns in the exact training order
        features_dict = {
            'Age': patient.Age,
            'RestingBP': patient.RestingBP,
            'Cholesterol': patient.Cholesterol,
            'FastingBS': patient.FastingBS,
            'MaxHR': patient.MaxHR,
            'Oldpeak': patient.Oldpeak,
            'Sex_M': sex_M,
            'ChestPainType_ATA': cp_ata,
            'ChestPainType_NAP': cp_nap,
            'ChestPainType_TA': cp_ta,
            'RestingECG_Normal': ecg_normal,
            'RestingECG_ST': ecg_st,
            'ExerciseAngina_Y': ex_angina_Y,
            'ST_Slope_Flat': slope_flat,
            'ST_Slope_Up': slope_up
        }
        
        input_df = pd.DataFrame([features_dict])

        # 3. Apply standard scaling on the continuous numeric columns
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']
        input_scaled = input_df.copy()
        input_scaled[numerical_cols] = scaler.transform(input_scaled[numerical_cols])

        # 4. Generate classifications and probabilities from Logistic Regression
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = float(model.predict_proba(input_scaled)[0][1])

        return {
            "heart_disease_predicted": True if prediction == 1 else False,
            "probability_percentage": round(prediction_proba * 100, 2),
            "risk_assessment": "HIGH RISK" if prediction == 1 else "LOW RISK"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing feature data vector: {str(e)}")