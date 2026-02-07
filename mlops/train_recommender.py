import os
import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "what_to_eat_recommender"
MODEL_NAME = "meal_similarity_recommender"
DATA_PATH = "/home/adilababayeva/codes/what-to-eat/data.csv"

FEATURES = ["Calories", "Proteins", "Carbs", "Fats"]

class MealRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.df = pd.read_csv(context.artifacts["meals"])

    def predict(self, context, model_input):
        meal_name = model_input["meal_name"].iloc[0]
        top_n = int(model_input.get("top_n", [5])[0])

        if meal_name not in self.df["meal_name"].values:
            return {"error": "Meal not found"}

        X_scaled = self.scaler.transform(self.df[FEATURES])
        sim = cosine_similarity(X_scaled)

        idx = self.df.index[self.df["meal_name"] == meal_name][0]
        scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)

        rec_idx = [i[0] for i in scores[1: top_n + 1]]
        return self.df.iloc[rec_idx].to_dict(orient="records")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)

    # Meal name fix (your logic — correct)
    df["meal_name"] = (
        df["cooking_method"].fillna("Unknown").astype(str) + " " +
        df["diet_type"].fillna("Balanced").astype(str) + " " +
        df["meal_type"].fillna("Meal").astype(str)
    ).str.title()

    df = df.dropna(subset=FEATURES)
    df = df.drop_duplicates(subset=["meal_name"])

    scaler = StandardScaler()
    scaler.fit(df[FEATURES])

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    df.to_csv("artifacts/meals.csv", index=False)

    with mlflow.start_run():
        mlflow.log_param("type", "content_based_cosine")
        mlflow.log_param("features", FEATURES)
        mlflow.log_metric("n_meals", len(df))

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MealRecommender(),
            artifacts={
                "scaler": "artifacts/scaler.pkl",
                "meals": "artifacts/meals.csv",
            },
            registered_model_name=MODEL_NAME,
        )

if __name__ == "__main__":
    main()
