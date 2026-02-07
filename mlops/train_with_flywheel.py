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

BASE_DATA = "data/synth_meals.csv"
FEEDBACK_DATA = "data/feedback_rows.csv"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "what_to_eat_classifier"
MODEL_NAME = "meal_similarity_recommender"

TARGET = "label_meal"
FEATURES = [
    "meal_time","spicy","diet","gluten_free","dairy_free",
    "cuisine","budget","prep_time","protein_pref","health_goal"
]

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    base = pd.read_csv(BASE_DATA)
    base["weight"] = 1.0

    if os.path.exists(FEEDBACK_DATA):
        fb = pd.read_csv(FEEDBACK_DATA)

        fb["weight"] = fb["accepted"].map({0: 3.0, 1: 2.0})
        fb = fb[FEATURES + [TARGET, "weight"]]

        data = pd.concat([base, fb], ignore_index=True)
    else:
        data = base

    X = data[FEATURES]
    y = data[TARGET]
    sample_weight = data["weight"]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )

    clf = LogisticRegression(max_iter=3000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run():
        pipe.fit(X_train, y_train, clf__sample_weight=w_train)
        # Log class labels order used by predict_proba
        classes = pipe.named_steps["clf"].classes_.tolist()
        mlflow.log_dict({"classes_": classes}, "classes.json")


        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("base_rows", len(base))
        mlflow.log_param("feedback_rows", len(data) - len(base))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print("Retrained model logged")

if __name__ == "__main__":
    main()
