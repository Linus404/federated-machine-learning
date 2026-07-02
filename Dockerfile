FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/fml-venv \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    KERAS_BACKEND=tensorflow \
    FML_DATA_DIR=/app/data \
    FML_ARTIFACT_DIR=/app/artifacts \
    RAY_memory_usage_threshold=0.99 \
    MALLOC_ARENA_MAX=2 \
    PYTHONWARNINGS=ignore

WORKDIR /app

COPY pyproject.toml uv.lock dashboard.py ./
RUN uv sync --frozen --no-dev

COPY src ./src

RUN mkdir -p /app/data /app/artifacts

CMD ["uv", "run", "--no-sync", "flwr", "run", ".", "--stream", "--federation-config", "num-supernodes=4"]