FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r aura && useradd -r -g aura -d /app -s /sbin/nologin aura

WORKDIR /app

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir \
    fastapi uvicorn aiohttp aiofiles pydantic \
    numpy psutil pillow sqlalchemy websockets httpx

# Copy source
COPY . .

# Data directories
RUN mkdir -p data logs \
    && chown -R aura:aura /app

USER aura

ENV AURA_HOST=0.0.0.0 \
    AURA_PORT=8000 \
    AURA_ENVIRONMENT=container \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "aura_main.py", "--headless"]
