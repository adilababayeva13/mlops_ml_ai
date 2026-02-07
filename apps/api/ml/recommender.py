import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# =========================
# Configuration
# =========================

MODEL_NAME = "meal_similarity_recommender"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",  # works inside k8s
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_client = MlflowClient()

_model = None
_model_version = "unregistered"


# =========================
# Model loading
# =========================

def _load_latest_model_fallback():
    """
    Load model in this priority order:

    1) MLflow Model Registry (if exists)
    2) Latest MLflow run artifact (fallback)

    This works even when:
    - Registry is disabled
    - Python 3.13 is used
    """
    global _model, _model_version

    # ---- 1. Try registry (safe try) ----
    try:
        versions = _client.get_latest_versions(MODEL_NAME)
        if versions:
            v = versions[0]
            _model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{MODEL_NAME}/{v.version}"
            )
            _model_version = v.run_id
            return
    except Exception:
        pass  # registry not available → fallback

    # ---- 2. Fallback: latest run artifact ----
    runs = mlflow.search_runs(
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            "No MLflow runs found. Train a model before serving predictions."
        )

    run_id = runs.iloc[0]["run_id"]

    _model = mlflow.pyfunc.load_model(
        model_uri=f"runs:/{run_id}/model"
    )
    _model_version = run_id


def get_model():
    global _model
    if _model is None:
        _load_latest_model_fallback()
    return _model, _model_version


def get_model_version():
    _, v = get_model()
    return v


# =========================
# Prediction
# =========================

def predict_meal(features: dict, top_k: int = 3):
    """
    Predict meal using trained sklearn pipeline.

    Parameters
    ----------
    features : dict
        Quiz answers (categorical strings)
    top_k : int
        Reserved for future ranking models

    Returns
    -------
    dict
    """
    model, _ = get_model()

    df = pd.DataFrame([features])

    # LogisticRegression pipeline → single label
    prediction = model.predict(df)[0]

    return {
        "recommended_meal": prediction,
        "top_k": top_k,
    }
