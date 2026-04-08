FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy full project
COPY . .

# Run uv sync to install dependencies into a virtual environment
RUN uv sync

# Environment Variables
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100
# Ensure Uvicorn and other dependencies installed by uv sync are accessible
ENV PATH="/app/.venv/bin:$PATH"

# Healthcheck for /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Run uvicorn server.app:app
CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
