import pandas as pd

BASELINE_PATH = "data/synth_meals.csv"

FEATURES = [
    "meal_time","spicy","diet","gluten_free","dairy_free",
    "cuisine","budget","prep_time","protein_pref","health_goal"
]

def load_baseline():
    df = pd.read_csv(BASELINE_PATH)
    return {f: df[f] for f in FEATURES}
