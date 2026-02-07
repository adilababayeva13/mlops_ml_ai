import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

MEALS = [
    "Vegan Buddha Bowl",
    "Spicy Chicken Bowl",
    "Keto Salmon Salad",
    "Margherita Pizza",
    "Beef Burrito",
]

CATEGORIES = {
    "meal_time": ["breakfast", "lunch", "dinner", "snack"],
    "spicy": ["low", "medium", "high"],
    "diet": ["none", "vegetarian", "vegan", "keto"],
    "gluten_free": ["no", "yes"],
    "dairy_free": ["no", "yes"],
    "cuisine": ["mediterranean", "mexican", "asian", "american", "middle_eastern"],
    "budget": ["low", "medium", "high"],
    "prep_time": ["quick", "medium", "long"],
    "protein_pref": ["low", "medium", "high"],
    "health_goal": ["lose", "maintain", "gain"],
}

# Rule-based score function to create learnable patterns (with noise)
def score_meals(row):
    s = {m: 0.0 for m in MEALS}

    # Vegan Buddha Bowl
    if row["diet"] == "vegan": s["Vegan Buddha Bowl"] += 4
    if row["health_goal"] == "lose": s["Vegan Buddha Bowl"] += 2
    if row["prep_time"] in ["quick", "medium"]: s["Vegan Buddha Bowl"] += 1
    if row["cuisine"] in ["mediterranean", "middle_eastern"]: s["Vegan Buddha Bowl"] += 1
    if row["protein_pref"] == "medium": s["Vegan Buddha Bowl"] += 1
    if row["dairy_free"] == "yes": s["Vegan Buddha Bowl"] += 1

    # Spicy Chicken Bowl
    if row["spicy"] == "high": s["Spicy Chicken Bowl"] += 4
    if row["diet"] in ["none", "keto"]: s["Spicy Chicken Bowl"] += 1
    if row["protein_pref"] == "high": s["Spicy Chicken Bowl"] += 2
    if row["cuisine"] == "asian": s["Spicy Chicken Bowl"] += 2
    if row["meal_time"] in ["lunch", "dinner"]: s["Spicy Chicken Bowl"] += 1

    # Keto Salmon Salad
    if row["diet"] == "keto": s["Keto Salmon Salad"] += 4
    if row["protein_pref"] == "high": s["Keto Salmon Salad"] += 2
    if row["gluten_free"] == "yes": s["Keto Salmon Salad"] += 1
    if row["health_goal"] in ["lose", "maintain"]: s["Keto Salmon Salad"] += 1
    if row["prep_time"] in ["quick", "medium"]: s["Keto Salmon Salad"] += 1
    if row["cuisine"] in ["mediterranean", "american"]: s["Keto Salmon Salad"] += 1

    # Margherita Pizza
    if row["budget"] == "low": s["Margherita Pizza"] += 2
    if row["prep_time"] == "quick": s["Margherita Pizza"] += 2
    if row["diet"] in ["none", "vegetarian"]: s["Margherita Pizza"] += 2
    if row["dairy_free"] == "no": s["Margherita Pizza"] += 2
    if row["cuisine"] == "mediterranean": s["Margherita Pizza"] += 1
    if row["health_goal"] == "gain": s["Margherita Pizza"] += 1

    # Beef Burrito
    if row["cuisine"] == "mexican": s["Beef Burrito"] += 3
    if row["protein_pref"] in ["medium", "high"]: s["Beef Burrito"] += 2
    if row["budget"] in ["low", "medium"]: s["Beef Burrito"] += 1
    if row["meal_time"] in ["lunch", "dinner"]: s["Beef Burrito"] += 1
    if row["spicy"] in ["medium", "high"]: s["Beef Burrito"] += 1
    if row["gluten_free"] == "no": s["Beef Burrito"] += 1

    # Add noise so it’s not perfectly deterministic
    for k in s:
        s[k] += RNG.normal(0, 0.8)

    return s

def generate(n_rows=20000):
    rows = []
    for _ in range(n_rows):
        row = {k: RNG.choice(v) for k, v in CATEGORIES.items()}
        scores = score_meals(row)
        label = max(scores, key=scores.get)
        row["label_meal"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate(50000)
    df.to_csv("data/synth_meals.csv", index=False)
    print("Saved:", df.shape, "-> data/synth_meals.csv")
    print(df["label_meal"].value_counts())
