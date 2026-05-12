#!/bin/bash
# scripts/eternal_start.sh
# ── Aura Zenith Production Launch ────────────────────
#
# CRITICAL: Must launch via aura_main.py, NOT via bare uvicorn.
# The server lifespan defers orchestrator boot to aura_main.
# Launching uvicorn directly gives you a web server with NO cognitive engine.

echo "🚀 Launching Aura Zenith Infinite Runtime..."

# Check for Apple Silicon
if [[ $(sysctl -n hw.optional.arm64) -ne 1 ]]; then
    echo "⚠️ Warning: Apple Silicon not detected. MLX performance will be degraded."
fi

# Activate Venv
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ venv or .venv not found. Run scripts/setup_arm64.sh first."
    exit 1
fi

# Environment Setup
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export AURA_PRODUCTION=1

# Launch via aura_main.py --headless which:
#   1. Boots the full orchestrator (cognitive engine, agency, consciousness)
#   2. Starts orchestrator.run() (the main mind-tick / autonomous loop)
#   3. Then starts the uvicorn API server
# Using --headless for server-only (no desktop GUI window)
python aura_main.py --headless --port 8000
