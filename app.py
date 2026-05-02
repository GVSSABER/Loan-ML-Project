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

# API URL
url = "https://loan-ml-project.onrender.com/predict"

if st.button("Predict Loan Status"):

    payload = {
        "Gender": str(gender),
        "Married": str(married),
        "Dependents": str(dependents),
        "Education": str(education),
        "Self_Employed": str(self_employed),
        "ApplicantIncome": float(app_income),
        "CoapplicantIncome": float(coapp_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history),
        "Property_Area": str(property_area)
    }

    try:
        response = requests.post(url, json=payload)

        # 🔥 DEBUG (IMPORTANT)
        st.write("Status Code:", response.status_code)
        st.write("Response:", response.text)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")

            if result["prediction"] == 1:
                st.success(f"✅ Approved (Confidence: {result['confidence']:.2f})")
            else:
                st.error(f"❌ Rejected (Confidence: {result['confidence']:.2f})")

        else:
            st.error("❌ API failed (check status code above)")

    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
