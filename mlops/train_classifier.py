import os
import tempfile
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.getenv("DATA_PATH", "/app/data/synth_meals.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "what_to_eat_classifier")
TARGET = os.getenv("TARGET", "label_meal")

# S3/MinIO artifact config (MLflow uses boto3 under the hood)
# These must exist in Job env.
# MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY are required.
# (AWS_DEFAULT_REGION is optional but fine.)
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    cat_cols = X.columns.tolist()
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    clf = LogisticRegression(max_iter=3000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("model", "logreg_onehot")
        mlflow.log_param("rows", len(df))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # ✅ SUPER RELIABLE: log plain artifact (no registry, no logged-models endpoint)
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.joblib")
            joblib.dump(pipe, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

        # Tag where the artifact is, so API can find it.
        mlflow.set_tag("model_artifact", "model/model.joblib")
        mlflow.set_tag("model_kind", "sklearn_pipeline_joblib")

        print(
            {
                "run_id": run.info.run_id,
                "accuracy": acc,
                "f1_weighted": f1,
                "artifact": "model/model.joblib",
            }
        )

if __name__ == "__main__":
    main()
