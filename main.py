from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback

app = FastAPI(title="Loan ML API", version="1.0")

# Load model safely
try:
    model = joblib.load("loan_pipeline1.pkl")
except Exception as e:
    model = None
    print("Model loading failed:", e)


@app.get("/")
def home():
    return {
        "message": "Loan ML API is running",
        "status": "success"
    }


class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.post("/predict")
def predict(data: LoanInput):

    try:
        if model is None:
            return {"error": "Model not loaded on server"}

        # Convert input safely
        input_data = data.dict()

        features = pd.DataFrame([input_data])

        # Prediction
        prediction = model.predict(features)[0]

        # Confidence (safe handling)
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(features)[0])
        else:
            confidence = 0.0

        return {
            "prediction": int(prediction),
            "result": "Approved" if prediction == 1 else "Rejected",
            "confidence": float(confidence)
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
