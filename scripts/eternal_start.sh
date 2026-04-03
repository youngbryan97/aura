#!/bin/bash
# scripts/eternal_start.sh
# ── Aura Zenith Production Launch ────────────────────

echo "🚀 Launching Aura Zenith Infinite Runtime..."

# Check for Apple Silicon
if [[ $(sysctl -n hw.optional.arm64) -ne 1 ]]; then
    echo "⚠️ Warning: Apple Silicon not detected. MLX performance will be degraded."
fi

# Activate Venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ venv not found. Run scripts/setup_arm64.sh first."
    exit 1
fi

# Environment Setup
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export AURA_PRODUCTION=1

# Restore latest snapshot if exists
python -c "import asyncio; from core.resilience.state_manager import StateManager; asyncio.run(StateManager().restore_latest())" || echo "Fresh start: No previous state found."

# Launch FastAPI via Uvicorn with P-core optimization
# Using 2 workers to leverage M1 Pro performance without overtaxing memory
# The --reload flag is removed for production stability
python -m uvicorn interface.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-config core/logging_config.py \
    --timeout-keep-alive 65
