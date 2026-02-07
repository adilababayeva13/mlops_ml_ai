from fastapi import FastAPI, Depends
from apps.api.db.session import Base, engine
from apps.api.db import models  # noqa
from apps.api.auth.routes import router as auth_router
from apps.api.auth.dependencies import get_current_user
from apps.api.db.models import User
from apps.api.quiz.routes import router as quiz_router
from apps.api.monitoring.routes import router as monitoring_router
from apps.api.monitoring.performance_routes import router as performance_router
from apps.api.llm.routes import router as llm_router

app = FastAPI(title="What To Eat API")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(quiz_router)
app.include_router(monitoring_router)
app.include_router(performance_router)
app.include_router(llm_router)


@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email}

@app.get("/health")
def health():
    return {"status": "ok"}
