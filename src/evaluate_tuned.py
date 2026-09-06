import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Configuration

MODEL_PATH = "models/xgboost_tuned.pkl"

PROCESSED_DATA_DIR = "data/processed"

X_TEST_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "X_test.csv"
)

Y_TEST_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "y_test.csv"
)


# Load Test Data

X_test = pd.read_csv(X_TEST_PATH)

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze()

print(f"Test data loaded: {X_test.shape}")

mlflow.set_experiment("customer_churn_prediction")

# Load Tuned Model

print("\nLoading tuned XGBoost model...")

tuned_model = joblib.load(
    MODEL_PATH
)


# Make Predictions

y_pred = tuned_model.predict(
    X_test
)

y_proba = tuned_model.predict_proba(
    X_test
)[:, 1]


# Evaluate Tuned Model

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_proba
)



# Display Results

print("TUNED XGBOOST : TEST SET RESULTS")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


with mlflow.start_run(run_name="xgboost_tuned"):
    mlflow.log_param("model_name", "xgboost_tuned")
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })
    mlflow.log_artifact(MODEL_PATH)

    mlflow.sklearn.log_model(
        tuned_model,
        artifact_path="model",
        serialization_format="pickle"
    )