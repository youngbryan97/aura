# ============================================================================
# Aura Cognitive Runtime — Production Dockerfile
# ============================================================================
# Build:  docker build -t aura:latest .
# Run:    docker run -p 8000:8000 -v aura-data:/app/data aura:latest
# Verify: docker run --rm aura:latest make doctor
# ============================================================================

FROM python:3.12-slim AS base

LABEL maintainer="security@aura-project.dev" \
      org.opencontainers.image.title="Aura Cognitive Runtime" \
      org.opencontainers.image.description="Locally-deployed autonomous AI cognitive agent" \
      org.opencontainers.image.source="https://github.com/youngbryan97/aura" \
      org.opencontainers.image.licenses="MIT"

# ── System dependencies (pinned versions for reproducibility) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.* \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ── Non-root user ────────────────────────────────────────────────────────
RUN groupadd -r aura && useradd -r -g aura -d /app -s /sbin/nologin aura

WORKDIR /app

# ── Python dependencies (fail-closed: no fallback installs) ──────────────
COPY requirements/core.txt requirements/core.txt
COPY requirements.txt requirements.txt
COPY requirements_lock.txt requirements_lock.txt

# Production builds MUST succeed with the lockfile. No best-effort fallback.
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements/core.txt

# ── Copy source ──────────────────────────────────────────────────────────
COPY . .

# ── Data directories + permissions ───────────────────────────────────────
RUN mkdir -p data logs artifacts/current \
    && chown -R aura:aura /app

# ── Drop to non-root ────────────────────────────────────────────────────
USER aura

# ── Runtime configuration ───────────────────────────────────────────────
# AURA_MODE is read by core/runtime/mode.py. AURA_HOST and AURA_PORT are not
# read anywhere; the port is aura_main's --port default of 8000, which the
# EXPOSE and the healthcheck below both assume.
ENV AURA_MODE=production \
    AURA_ENVIRONMENT=container \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# ── Health check ────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# ── Entrypoint ──────────────────────────────────────────────────────────
CMD ["python", "aura_main.py", "--headless"]
