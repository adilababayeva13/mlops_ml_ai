FROM python:3.13-slim


WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# copy pyproject files FIRST (cache-friendly)
COPY pyproject.toml ./
COPY poetry.lock* ./

# install deps (no virtualenv inside container)
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root


# copy application code
COPY apps ./apps
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
