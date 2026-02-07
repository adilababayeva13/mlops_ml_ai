import os
import pandas as pd
from sqlalchemy import create_engine

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://app:apppass@localhost:5432/what_to_eat"
)

OUT_PATH = "data/feedback_rows.csv"

engine = create_engine(DB_URL)

QUERY = """
SELECT
    qa.session_id,
    qa.feature_name,
    qa.feature_value,
    uf.chosen_meal AS label_meal,
    uf.accepted
FROM quiz_answers qa
JOIN user_feedback uf
    ON qa.session_id = uf.session_id
"""

def main():
    df = pd.read_sql(QUERY, engine)

    if df.empty:
        print("No feedback data yet.")
        return

    # Pivot answers → columns
    X = (
        df.pivot_table(
            index=["session_id", "label_meal", "accepted"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
        .reset_index()
    )

    X.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(X)} feedback rows → {OUT_PATH}")

if __name__ == "__main__":
    main()
