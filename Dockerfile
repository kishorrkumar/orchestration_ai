# Multi-stage production Dockerfile for PersonaPlex Voice Agent Platform
# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsox-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# Stage 2: Final Lean Non-Root Runtime
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser orchestration /app/orchestration
COPY --chown=appuser:appuser models /app/models
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser pyproject.toml /app/

# Create data directories with correct ownership
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "orchestration.gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
