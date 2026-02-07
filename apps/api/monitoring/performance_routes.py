from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from apps.api.auth.dependencies import get_current_user, get_db
from apps.api.monitoring.performance_service import compute_performance
from apps.api.db.models import User

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/performance")
def performance(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return compute_performance(db)
