# Use the official uv image for build optimizations
FROM ghcr.io/astral-sh/uv:latest AS uv_setup

# Base image with Python 3.13 (slim)
FROM python:3.13-slim AS builder

# Copy uv binary
COPY --from=uv_setup /uv /uvx /bin/

# Set env variables for uv to optimize build performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install build-essential in builder in case binary dependencies require compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Mount uv cache and configuration for faster download & build
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable

# Copy the rest of the project
COPY . /app

# Sync the whole project (including the app itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable


# Runtime Stage
FROM python:3.13-slim AS runner

# Install runtime system dependencies (ffmpeg is required for yt-dlp/whisper audio extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code and other required configuration files
COPY app/ /app/app/
COPY pyproject.toml README.md /app/

# Environment configurations
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose FastAPI default port
EXPOSE 8000

# Create a non-root system user and change ownership of working directory for hardening
RUN groupadd -r app && useradd -r -g app app && chown -R app:app /app
USER app

# Liveness/Readiness health check using internal api route
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
