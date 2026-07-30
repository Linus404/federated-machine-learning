FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim@sha256:837853d0f4703dbada56fd0807ea1db01eccca02db0af3b1f0cee5e902077107

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/fml-venv \
    HOME=/tmp \
    PYTHONHASHSEED=0 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_DETERMINISTIC_OPS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    KERAS_BACKEND=tensorflow \
    FML_PUBLIC_ARTIFACT_DIR=/app/artifacts/public \
    FML_SERVER_ARTIFACT_DIR=/app/artifacts/server \
    FML_CLIENT_COUNT=4 \
    RAY_memory_usage_threshold=0.99 \
    MALLOC_ARENA_MAX=2 \
    PYTHONWARNINGS=ignore

WORKDIR /app

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=app:app dashboard.py ./
COPY --chown=app:app src ./src
COPY --chown=app:app docs/scientific-protocol-v1.toml ./docs/

RUN mkdir -p /app/artifacts/public /app/artifacts/server /app/state \
    && chown -R app:app /app/artifacts /app/state

USER 1000:1000

CMD ["sh", "-c", "exec uv run --no-sync flwr run . --stream --federation-config \"num-supernodes=${FML_CLIENT_COUNT}\" --run-config \"expected-client-count=${FML_CLIENT_COUNT}\""]
