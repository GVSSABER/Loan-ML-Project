import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv("loan.csv")
df = df.drop("Loan_ID", axis=1)

# ------------------------
# SPLIT
# ------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})

# ------------------------
# FEATURES
# ------------------------
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

cat_cols = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area", "Credit_History"]

# ------------------------
# PIPELINES
# ------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ------------------------
# SPLIT DATA
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 🚀 FIXED MLFLOW (IMPORTANT CHANGE)
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

print("MLflow tracking folder:", MLRUNS_DIR)

# ✔ USE ONLY ONE CONSISTENT PATH
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.set_experiment("Loan_Approval_Experiment")

# ------------------------
# TRAIN + LOG
# ------------------------
with mlflow.start_run():

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print("Accuracy:", accuracy)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    print("MLflow run logged successfully ✅")

# ------------------------
# SAVE MODEL
# ------------------------
joblib.dump(model, "loan_pipeline.pkl")

print("Local model saved ✅")

print(mlflow.get_tracking_uri())