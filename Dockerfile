FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PULSEWATCH_MODEL_PATH=/app/artifacts/random_forest.joblib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir '.[service]'

EXPOSE 8000

CMD ["uvicorn", "pulsewatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
