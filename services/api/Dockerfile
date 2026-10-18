# syntax=docker/dockerfile:1.7
# Build from the repository root: docker compose build  (or docker build -f services/api/Dockerfile .)
ARG APP=schemalens

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# third-party dependencies, cached until uv.lock changes
FROM base AS deps
ARG APP
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/
COPY services/api/pyproject.toml services/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package ${APP}-api --no-install-workspace

# models baked into the image (the runtime works offline)
FROM deps AS models
COPY scripts/download_models.py scripts/
RUN /app/.venv/bin/python scripts/download_models.py /models

# our code
FROM deps AS build
ARG APP
COPY packages/core packages/core
COPY services/api services/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package ${APP}-api --no-editable

FROM python:3.12-slim AS runtime-base
ARG APP
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    APP_MODULE=${APP}_api.main:app \
    OCR_MODEL_DIR=/models/easyocr EMBED_MODEL_DIR=/models/embed PRESETS_DIR=/app/presets TEMPLATES_DIR=/app/templates \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False
WORKDIR /app
COPY --from=models /models /models
# schema presets, read from PRESETS_DIR
COPY presets /app/presets
# few-shot examples per preset (read-only)
COPY templates /app/templates
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

# dev: editable install + hot reload (code mounted by compose)
FROM runtime-base AS dev
ARG APP
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./
COPY packages packages
COPY services/api services/api
RUN uv sync --frozen --no-dev --package ${APP}-api && chown -R app:app /app
USER app
CMD ["sh", "-c", "exec uvicorn $APP_MODULE --host 0.0.0.0 --port 8000 --no-access-log --reload --reload-dir /app/packages --reload-dir /app/services"]

# runtime (the default target)
FROM runtime-base AS runtime
COPY --from=build /app/.venv /app/.venv
USER app
CMD ["sh", "-c", "exec uvicorn $APP_MODULE --host 0.0.0.0 --port 8000 --no-access-log"]

# AWS Lambda (container image, arm64): the same code, models and uvicorn as locally. The Lambda
# Web Adapter (an extension) turns Function URL invocations into plain HTTP and streams the
# response (RESPONSE_STREAM), so `?stream=true` progress arrives live. Non-HTTP events (the
# EventBridge warm-up ping) are posted to AWS_LWA_PASS_THROUGH_PATH.
# Lambda can only write to /tmp; retrieval is off (the demo takes short documents).
FROM runtime-base AS lambda
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=build /app/.venv /app/.venv
ENV HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
    RAG_ENABLED=false FEWSHOT_ENABLED=false ACTIVE_LLM_PROFILE=groq LAMBDA_REQUEST_CONTEXT=true \
    AWS_LWA_PORT=8000 AWS_LWA_INVOKE_MODE=response_stream \
    AWS_LWA_READINESS_CHECK_PATH=/healthz AWS_LWA_PASS_THROUGH_PATH=/internal/warm-up
CMD ["sh", "-c", "exec uvicorn $APP_MODULE --host 127.0.0.1 --port 8000 --no-access-log"]
