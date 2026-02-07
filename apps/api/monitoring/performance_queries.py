from sqlalchemy import text

BASE_QUERY = """
SELECT
    qs.model_version,
    r.payload,
    uf.accepted,
    qs.created_at::date AS day
FROM quiz_sessions qs
JOIN recommendations r ON r.session_id = qs.id
JOIN user_feedback uf ON uf.session_id = qs.id
"""
