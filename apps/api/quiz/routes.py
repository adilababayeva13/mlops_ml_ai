from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.auth.dependencies import get_current_user, get_db
from apps.api.db.models import (
    QuizSession,
    QuizAnswer,
    Recommendation,
    User,
    UserFeedback
)
from apps.api.quiz.questions import QUESTIONS
from apps.api.ml.recommender import predict_meal, get_model_version

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.get("/questions")
def get_questions(user: User = Depends(get_current_user)):
    """
    Return quiz questions.
    Protected endpoint.
    """
    return QUESTIONS


@router.post("/submit")
def submit_quiz(
    answers: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit quiz answers, run ML model, store everything.

    Flow:
    1) Validate answers
    2) Create quiz session (store model version)
    3) Persist raw answers (for audit + drift)
    4) Call ML model
    5) Persist recommendation
    6) Return result to UI
    """

    # 1. Validate input strictly (NO silent bugs)
    expected_features = {q["id"] for q in QUESTIONS}
    received_features = set(answers.keys())

    missing = expected_features - received_features
    extra = received_features - expected_features

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing quiz answers for: {sorted(missing)}",
        )

    if extra:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown quiz fields: {sorted(extra)}",
        )

    # Ensure all values are strings (ML pipeline expects categorical strings)
    answers = {k: str(v) for k, v in answers.items()}

    # 2. Load model version (lazy-loaded, safe)
    model_version = get_model_version()

    # 3. Create quiz session
    session = QuizSession(
        user_id=user.id,
        mode="ml",
        model_version=model_version,
    )
    db.add(session)
    db.flush()  # session.id becomes available

    # 4. Persist raw quiz answers (flywheel + drift-ready)
    for feature_name, feature_value in answers.items():
        db.add(
            QuizAnswer(
                session_id=session.id,
                feature_name=feature_name,
                feature_value=feature_value,
            )
        )

    # 5. Run ML model (classifier)
    try:
        prediction = predict_meal(features=answers, top_k=3)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ML prediction failed: {str(e)}",
        )


    # 6. Persist recommendation
    db.add(
        Recommendation(
            session_id=session.id,
            source="ml",
            payload=prediction,
        )
    )

    # 7. Commit everything atomically
    db.commit()

    # 8. Return response to UI
    return {
        "session_id": session.id,
        "model_version": model_version,
        "result": prediction,
    }



@router.post("/feedback")
def feedback(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # payload: { "session_id": 123, "chosen_meal": "...", "accepted": true/false }
    db.add(UserFeedback(
        session_id=int(payload["session_id"]),
        chosen_meal=str(payload["chosen_meal"]),
        accepted=1 if payload.get("accepted") else 0
    ))
    db.commit()
    return {"status": "ok"}
