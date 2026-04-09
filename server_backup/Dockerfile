FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && rm -rf /var/lib/apt/lists/*

# Fast installer
RUN pip install --no-cache-dir uv

# Cache dependencies layer
COPY pyproject.toml .
RUN uv pip install --system \
    fastapi \
    uvicorn[standard] \
    pydantic \
    openai \
    websockets

# Copy all source
COPY . .

# Env vars - HF_TOKEN intentionally has NO default
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100

# HF Spaces health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
