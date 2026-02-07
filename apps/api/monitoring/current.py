import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

FEATURES = [
    "meal_time","spicy","diet","gluten_free","dairy_free",
    "cuisine","budget","prep_time","protein_pref","health_goal"
]

def load_current(db: Session, limit: int = 500):
    query = text("""
        SELECT session_id, feature_name, feature_value
        FROM quiz_answers
        ORDER BY session_id DESC
        LIMIT :limit
    """)

    rows = db.execute(query, {"limit": limit}).fetchall()
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["session_id","feature","value"])

    pivot = (
        df.pivot_table(
            index="session_id",
            columns="feature",
            values="value",
            aggfunc="first"
        )
        .reset_index(drop=True)
    )

    return {f: pivot[f].dropna() for f in FEATURES if f in pivot}
