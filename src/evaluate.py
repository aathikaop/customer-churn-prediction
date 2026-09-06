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

MODEL_DIR = "models"
PROCESSED_DATA_DIR = "data/processed"
REPORT_DIR = "reports"

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
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print(f"Test data loaded: {X_test.shape}")


# Models to Evaluate

model_names = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost"
]

mlflow.set_experiment("customer_churn_prediction")

# Evaluate Models

results = []

for model_name in model_names:

    print(f"\nEvaluating: {model_name}")

    # Load trained pipeline
    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    pipeline = joblib.load(model_path)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability predictions for ROC-AUC
    y_proba = pipeline.predict_proba(X_test)[:, 1]


    # Evaluation metrics

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

    # Store results
    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    # Print results
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        mlflow.log_artifact(model_path)


# Create Results DataFrame

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)

best_model_name = results_df.iloc[0]["model"]
print(f"\nBest baseline model (by F1): {best_model_name}")

best_model_path = os.path.join(MODEL_DIR, f"{best_model_name}.pkl")
best_pipeline = joblib.load(best_model_path)

with mlflow.start_run(run_name=f"{best_model_name}_best_baseline"):
    mlflow.log_param("model_name", best_model_name)
    mlflow.log_metrics(results_df.iloc[0][
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ].to_dict())
    mlflow.sklearn.log_model(best_pipeline, artifact_path="model",serialization_format="pickle")


# Save Evaluation Results

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

results_path = os.path.join(
    REPORT_DIR,
    "evaluation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# Final Comparison

print("MODEL COMPARISON")

print(
    results_df.to_string(index=False)
)

print("\nEvaluation results saved to:")
print(results_path)