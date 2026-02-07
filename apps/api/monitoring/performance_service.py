import pandas as pd
from sqlalchemy.orm import Session
from apps.api.monitoring.performance_queries import BASE_QUERY
from apps.api.monitoring.confidence import confidence_bucket

def compute_performance(db: Session):
    rows = db.execute(BASE_QUERY).fetchall()
    if not rows:
        return {"status": "no_data"}

    df = pd.DataFrame(rows, columns=["model_version","payload","accepted","day"])

    # Extract top-1 confidence
    df["confidence"] = df["payload"].apply(
        lambda p: float(p["top_k"][0]["prob"]) if p and "top_k" in p else None
    )

    df["bucket"] = df["confidence"].apply(confidence_bucket)

    overall = {
        "acceptance_rate": round(df["accepted"].mean(), 4),
        "samples": len(df),
    }

    by_bucket = (
        df.groupby("bucket")
        .agg(
            acceptance_rate=("accepted", "mean"),
            samples=("accepted", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    by_day = (
        df.groupby("day")
        .agg(
            acceptance_rate=("accepted", "mean"),
            samples=("accepted", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    by_model = (
        df.groupby("model_version")
        .agg(
            acceptance_rate=("accepted", "mean"),
            samples=("accepted", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "overall": overall,
        "by_confidence_bucket": by_bucket,
        "by_day": by_day,
        "by_model_version": by_model,
    }
