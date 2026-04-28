#!/usr/bin/env bash
# Deploy Aura to a remote Hetzner server.
# Usage: ./scripts/deploy_hetzner.sh <user@host> [dest=/opt/aura]

set -euo pipefail

REMOTE=${1:-}
DEST=${2:-/opt/aura}

if [ -z "$REMOTE" ]; then
  echo "Usage: $0 user@host [dest]"
  exit 2
fi

echo "Deploying to $REMOTE -> $DEST"

# Ensure local repo root
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
echo "Repo root: $ROOT_DIR"

echo "Creating destination on remote (may require sudo)."
ssh "$REMOTE" "sudo mkdir -p '$DEST' && sudo chown $(whoami):$(whoami) '$DEST'"

echo "Rsyncing repository (excludes .git, venv)..."
rsync -az --delete --exclude='.git' --exclude='.venv' --exclude='venv' --exclude='__pycache__' --progress "$ROOT_DIR/" "$REMOTE:$DEST/"

echo "Installing Python venv and requirements on remote"
ssh "$REMOTE" bash -lc "'
set -e
cd '$DEST'
python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip setuptools wheel || true
if [ -f requirements.txt ]; then
  pip install -r requirements.txt || true
fi
'"

echo "Creating systemd env and unit (requires sudo)"
# Create env file and systemd unit on remote
ENV_FILE=/etc/systemd/system/aura.env
UNIT_FILE=/etc/systemd/system/aura.service

ssh "$REMOTE" sudo tee "$ENV_FILE" > /dev/null <<'ENV'
# Example Aura env file - edit values and secure this file
AURA_API_TOKEN=$(openssl rand -hex 32)
AURA_CLARITY_THRESHOLD=0.35
PYTHONPATH=/opt/aura
ENV

ssh "$REMOTE" sudo tee "$UNIT_FILE" > /dev/null <<'UNIT'
[Unit]
Description=Aura Autonomous Service
After=network.target

[Service]
Type=simple
EnvironmentFile=/etc/systemd/system/aura.env
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$DEST
ExecStart=$DEST/.venv/bin/python -m uvicorn interface.server:app --host 127.0.0.1 --port 8000 --log-level info
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

echo "Reloading systemd and starting aura.service"
ssh "$REMOTE" sudo systemctl daemon-reload
ssh "$REMOTE" sudo systemctl enable --now aura.service

echo "Deployment finished. Check logs with: sudo journalctl -u aura.service -f"
