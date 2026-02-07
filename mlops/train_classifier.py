import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.getenv("DATA_PATH", "data/synth_meals.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "what_to_eat_classifier"
MODEL_NAME = "meal_similarity_recommender"  # keep your existing name to avoid breaking API

TARGET = "label_meal"

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

    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("model", "logreg_onehot")
        mlflow.log_param("rows", len(df))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print({"accuracy": acc, "f1": f1})

if __name__ == "__main__":
    main()
