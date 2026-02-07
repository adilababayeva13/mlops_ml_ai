from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from apps.api.auth.dependencies import get_current_user, get_db
from apps.api.monitoring.service import compute_drift
from apps.api.db.models import User

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/drift")
def drift(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    sample_size: int = 500,
):
    return compute_drift(db, sample_size)
