# Incident Commander Environment - Hugging Face Spaces Docker image.
#
# HF Spaces (sdk: docker) requires this Dockerfile at the repo root.
# Single-stage, slim runtime: only the env server deps are installed
# (openenv-core, fastapi, uvicorn, pydantic, gradio). The training stack
# (torch / unsloth / trl) is NOT installed here - that lives in the
# Colab notebook. This keeps the deployed image small.

FROM python:3.11-slim

WORKDIR /app

# curl is needed for the HEALTHCHECK. build-essential is needed by
# pydantic-core's optional Rust shims on slim Python images.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install runtime deps first (better Docker layer caching).
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project source into the image.
COPY . /app/env
WORKDIR /app/env

# Make `import server.app`, `import models`, etc. resolve to /app/env.
ENV PYTHONPATH=/app/env
ENV PORT=8000

# /health is auto-registered by openenv's create_app().
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# uvicorn imports server.app:app from /app/env (PYTHONPATH).
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
