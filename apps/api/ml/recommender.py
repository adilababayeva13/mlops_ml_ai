import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio12345")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


# --------------------------------------------------
# MLflow setup
# --------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "meal_similarity_recommender"

_client = MlflowClient()
_model = None
_model_version = None
_model_run_id = None

# --------------------------------------------------
# Model loader
# --------------------------------------------------
def get_model():
    """
    Loads the latest sklearn model AND fetches its MLflow run_id.
    Cached in memory after first load.
    """
    global _model, _model_version, _model_run_id

    if _model is None:
        # 1. Fetch latest registered model version
        versions = _client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise RuntimeError(f"No registered versions for model '{MODEL_NAME}'")

        latest = versions[0]  # highest version number
        _model_version = latest.version
        _model_run_id = latest.run_id

        # 2. Load sklearn pipeline (for predict_proba)
        _model = mlflow.sklearn.load_model(
            model_uri=f"models:/{MODEL_NAME}/{_model_version}"
        )

    return _model, _model_version, _model_run_id


# --------------------------------------------------
# Public helpers
# --------------------------------------------------
def get_model_version():
    """
    Returns MLflow run_id (NOT sklearn metadata).
    """
    _, _, run_id = get_model()
    return run_id


def predict_meal(features: dict, top_k: int = 3):
    """
    Predict meal with top-k probabilities.
    """
    model, _, _ = get_model()

    X = pd.DataFrame([features])

    # Predict class
    pred = model.predict(X)[0]

    # Predict probabilities
    proba = model.predict_proba(X)[0]
    classes = model.named_steps["clf"].classes_

    # Top-k
    k = int(top_k)
    top_idx = np.argsort(proba)[::-1][:k]

    top = [
        {"meal": str(classes[i]), "prob": float(proba[i])}
        for i in top_idx
    ]

    return {
        "recommended_meal": str(pred),
        "top_k": top,
    }
