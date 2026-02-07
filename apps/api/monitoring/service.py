from sqlalchemy.orm import Session
from apps.api.monitoring.baseline import load_baseline
from apps.api.monitoring.current import load_current
from apps.api.monitoring.drift_utils import categorical_psi, psi_status

def compute_drift(db: Session, sample_size: int = 500):
    baseline = load_baseline()
    current = load_current(db, sample_size)

    if not current:
        return {"status": "no_data"}

    results = {}
    overall = "stable"

    for feature, base_series in baseline.items():
        if feature not in current:
            continue

        psi = categorical_psi(base_series, current[feature])
        status = psi_status(psi)

        if status == "drift":
            overall = "drift"
        elif status == "warning" and overall != "drift":
            overall = "warning"

        results[feature] = {
            "psi": round(psi, 4),
            "status": status,
        }

    return {
        "overall_status": overall,
        "features": results,
    }
