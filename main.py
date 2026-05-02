from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("loan_pipeline.pkl")


# ✅ ADDED ROOT ROUTE (THIS FIXES YOUR ERROR)
@app.get("/")
def home():
    return {"message": "Loan ML API is running"}


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

    features = pd.DataFrame([{
        "Gender": data.Gender,
        "Married": data.Married,
        "Dependents": data.Dependents,
        "Education": data.Education,
        "Self_Employed": data.Self_Employed,
        "ApplicantIncome": data.ApplicantIncome,
        "CoapplicantIncome": data.CoapplicantIncome,
        "LoanAmount": data.LoanAmount,
        "Loan_Amount_Term": data.Loan_Amount_Term,
        "Credit_History": data.Credit_History,
        "Property_Area": data.Property_Area
    }])

    prediction = model.predict(features)[0]

    confidence = max(model.predict_proba(features)[0])

    return {
        "prediction": int(prediction),
        "result": "Approved" if prediction == 1 else "Rejected",
        "confidence": float(confidence)
    }
