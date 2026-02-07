import os
import tempfile
import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "meal_similarity_recommender")  # keep name if you want
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "what_to_eat_classifier")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_client = MlflowClient()
_model = None
_model_version = "unloaded"
_model_run_id = None

def _best_latest_run_id():
    # Find the newest FINISHED run that has our artifact tag
    exp = _client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT}. Train job must run first.")

    runs = _client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    for r in runs:
        if r.data.tags.get("model_artifact") == "model/model.joblib":
            return r.info.run_id

    raise RuntimeError("No successful trained model found (FINISHED run with model/model.joblib).")

def _load_model_from_mlflow():
    global _model, _model_version, _model_run_id

    run_id = _best_latest_run_id()
    _model_run_id = run_id

    # Download artifact from MLflow artifact store (MinIO)
    # Requires boto3 inside this API container and S3 env vars in deployment.
    with tempfile.TemporaryDirectory() as td:
        local_path = _client.download_artifacts(run_id, "model/model.joblib", dst_path=td)
        _model = joblib.load(local_path)

    _model_version = run_id

def get_model():
    global _model
    if _model is None:
        _load_model_from_mlflow()
    return _model, _model_version

def get_model_version():
    _, v = get_model()
    return v

def predict_meal(features: dict, top_k: int = 3):
    model, _ = get_model()
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]

    return {
        "recommended_meal": str(pred),
        "top_k": int(top_k),
    }
