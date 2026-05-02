import streamlit as st
import requests

st.title("🏦 Loan Approval System")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income")
coapp_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Status"):

    url = "http://127.0.0.1:8001/predict"

    payload = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.success(f"Approved ✅ (Confidence: {result['confidence']:.2f})")
            else:
                st.error(f"Rejected ❌ (Confidence: {result['confidence']:.2f})")

        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Connection Error: {e}")