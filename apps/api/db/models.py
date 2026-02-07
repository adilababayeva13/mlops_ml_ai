from sqlalchemy import String, DateTime, func , ForeignKey, Integer, JSON,Column
from sqlalchemy.orm import Mapped, mapped_column , relationship
from apps.api.db.session import Base


class ChatRecommendation(Base):
    __tablename__ = "chat_recommendations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("quiz_sessions.id"), nullable=True)

    prompt = Column(String, nullable=False)
    response = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

# -----------------------------------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("quiz_sessions.id"), index=True)
    chosen_meal: Mapped[str] = mapped_column(String(100))
    accepted: Mapped[int] = mapped_column(Integer)  # 1 accepted, 0 override
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

# ------------------------------------------------------------------------------------------------------

class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    mode: Mapped[str] = mapped_column(String(20))  # "ml" or "ai"
    model_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    answers = relationship("QuizAnswer", back_populates="session")


class QuizAnswer(Base):
    __tablename__ = "quiz_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("quiz_sessions.id"))
    feature_name: Mapped[str] = mapped_column(String(100))
    feature_value: Mapped[str] = mapped_column(String(100))

    session = relationship("QuizSession", back_populates="answers")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("quiz_sessions.id"))
    source: Mapped[str] = mapped_column(String(20))  # "ml" | "ai"
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
