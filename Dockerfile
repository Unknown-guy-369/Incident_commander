# Incident Commander Environment — Docker Image
# Multi-stage build using a standard public Python base image.
# Compatible with Hugging Face Spaces (Docker SDK) and local Docker builds.
#
# NOTE: HuggingFace Spaces (sdk: docker) requires the Dockerfile to be at the
# root of the Space repository. This file lives at the repo root and mirrors
# the build defined in server/Dockerfile.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy project files
COPY . /app/env

WORKDIR /app/env

# Install dependencies into a venv using pyproject.toml
# Use uv.lock if it exists for reproducible builds; otherwise resolve fresh.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment source code
COPY --from=builder /app/env /app/env

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so all relative imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Hugging Face Spaces uses port 7860 by default; fall back to 8000 locally.
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose both the HF default port and the local dev port
EXPOSE 7860
EXPOSE 8000

# Launch the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
