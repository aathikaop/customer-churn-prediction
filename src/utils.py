import json
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import pandas as pd


def load_data(file_path):
    """
    Load the dataset from the given file path.

    Parameters:
        file_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def promote_production_model(model_name="churn_model", alias="production"):
    """
    Pulls the model tagged with the given alias from the MLflow Model Registry
    and saves it locally as models/model.pkl + models/metadata.json.
    Run this once whenever a new model is promoted to production in MLflow.
    """
    client = mlflow.MlflowClient()

    model_version = client.get_model_version_by_alias(model_name, alias)
    print(f"Found {model_name} version {model_version.version} (run_id: {model_version.run_id})")

    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)

    joblib.dump(model, "models/model.pkl")
    print("Saved to models/model.pkl")

    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    metadata = {
        "model_name": model_name,
        "version": model_version.version,
        "alias": alias,
        "source_run_id": model_version.run_id,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics
    }

    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved metadata to models/metadata.json")
    print(json.dumps(metadata, indent=2))

    return model, metadata