#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "🧠 Aura model fetcher"
echo "   Backend: ${AURA_LOCAL_BACKEND:-llama_cpp}"
echo ""

# Keep this legacy helper as a thin wrapper so older setup flows still land on
# the current managed-runtime fetcher instead of reviving the retired MLX-only path.
if [[ -x ".venv/bin/python" ]]; then
  exec .venv/bin/python scripts/fetch_models.py
fi

exec python3 scripts/fetch_models.py
