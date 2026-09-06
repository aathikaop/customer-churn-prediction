import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from .schemas import CustomerFeatures, PredictionResponse, HealthResponse

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether an e-commerce customer will churn.",
    version="1.0.0"
)

MODEL_PATH = "models/model.pkl"
METADATA_PATH = "models/metadata.json"

model = None
model_metadata = {}


@app.on_event("startup")
def load_model():
    global model, model_metadata

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            model_metadata = json.load(f)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_version=str(model_metadata.get("version")) if model_metadata else None
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([customer.dict()])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    if probability >= 0.7:
        risk = "High"
    elif probability >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return PredictionResponse(
        churn_prediction=int(prediction),
        churn_probability=round(float(probability), 4),
        risk_level=risk
    )