import os
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.auth.dependencies import get_current_user, get_db
from apps.api.db.models import User, ChatRecommendation

from openai import OpenAI

router = APIRouter(prefix="/llm", tags=["llm"])

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "meal": {"type": "string"},
        "reason": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["meal", "reason", "tags", "confidence"],
    "additionalProperties": False,
}


@router.post("/recommend")
def llm_recommend(
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in environment",
        )

    client = OpenAI(api_key=api_key)

    user_text = str(payload.get("message", "")).strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message is required")

    system_prompt = (
        "You recommend meals. Return ONLY valid JSON matching this schema:\n"
        f"{OUTPUT_SCHEMA}\n"
        "No markdown. No extra text."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    db.add(
        ChatRecommendation(
            user_id=user.id,
            prompt=user_text,
            response=parsed,
        )
    )
    db.commit()

    return parsed
