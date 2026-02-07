import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.getenv("DATA_PATH", "data/synth_meals.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://35.229.140.231:5000")
EXPERIMENT = "what_to_eat_classifier"

TARGET = "label_meal"
LOCAL_MODEL_PATH = "artifacts/model.joblib"

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

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=3000))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Save model locally
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pipe, LOCAL_MODEL_PATH)

        # Log as artifact (NO registry)
        mlflow.log_artifact(LOCAL_MODEL_PATH, artifact_path="model")

        print("Run ID:", run.info.run_id)
        print("Model saved to:", LOCAL_MODEL_PATH)

if __name__ == "__main__":
    main()
