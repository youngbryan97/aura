#!/usr/bin/env bash
# ============================================================================
# Deploy Aura to Oracle Cloud VM
# ============================================================================
# Usage:
#   ./cloud/deploy.sh <VM_IP>              # First deploy + install deps
#   ./cloud/deploy.sh <VM_IP> --update     # Code update only (faster)
#
# What it does:
#   1. Syncs Aura source code to /opt/aura on the VM
#   2. Installs/updates Python dependencies
#   3. Restarts the Aura service
# ============================================================================
set -euo pipefail

VM_IP="${1:?Usage: ./cloud/deploy.sh <VM_IP> [--update]}"
UPDATE_ONLY="${2:-}"
SSH_USER="ubuntu"
SSH_KEY="${HOME}/.ssh/aura-oracle.key"
REMOTE_DIR="/opt/aura"

# ---------------------------------------------------------------------------
# Host verification
#
# CP126 (critical): "Deployment disables SSH host verification."
# StrictHostKeyChecking=no accepts ANY host key without complaint, so this
# script would happily rsync the whole source tree, write the .env — which
# carries live secrets — and run sudo commands on whatever machine answered
# at that IP. On a cloud provider IPs are recycled constantly, so the
# machine at a remembered address genuinely may not be yours.
#
# `accept-new` keeps the ergonomics that motivated the original setting: a
# first deploy to a freshly provisioned VM still works without a manual
# known_hosts dance. What it does NOT do is stay silent when a host key
# CHANGES — which is the case that matters, because that is either a
# reprovision you should confirm or someone else holding the address.
#
# Set AURA_DEPLOY_SSH_STRICT=yes for a fully pinned deploy, or
# AURA_DEPLOY_KNOWN_HOSTS to use a checked-in host-key file.
# ---------------------------------------------------------------------------
SSH_STRICT="${AURA_DEPLOY_SSH_STRICT:-accept-new}"
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=${SSH_STRICT}"
if [[ -n "${AURA_DEPLOY_KNOWN_HOSTS:-}" ]]; then
  SSH_OPTS="${SSH_OPTS} -o UserKnownHostsFile=${AURA_DEPLOY_KNOWN_HOSTS}"
fi

echo "╔══════════════════════════════════╗"
echo "║   DEPLOYING AURA → ${VM_IP}     "
echo "╚══════════════════════════════════╝"

# Get the project root (parent of cloud/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ">>> Source: ${PROJECT_ROOT}"
echo ">>> Target: ${SSH_USER}@${VM_IP}:${REMOTE_DIR}"

# ------------------------------------------------------------------
# 1. Sync source code (excludes build artifacts & local data)
# ------------------------------------------------------------------
echo ">>> [1/3] Syncing source code..."
rsync -avz --progress -e "ssh ${SSH_OPTS}" \
    --exclude='.git/' \
    --exclude='dist/' \
    --exclude='build/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.aura/' \
    --exclude='data/memory/' \
    --exclude='data/knowledge_graph/' \
    --exclude='data/vector_store/' \
    --exclude='data/error_logs/' \
    --exclude='data/backups/' \
    --exclude='logs/' \
    --exclude='*.log' \
    --exclude='*.wav' \
    --exclude='*.dmg' \
    --exclude='*.app' \
    --exclude='aura_desktop.py' \
    --exclude='Aura.spec' \
    --exclude='build_app.sh' \
    --exclude='aura_icon.*' \
    --exclude='node_modules/' \
    --exclude='venv/' \
    --exclude='.env' \
    --exclude='config_creds.json' \
    --exclude='cloud/' \
    --exclude='memory_store/' \
    --exclude='storage/' \
    --exclude='workspace/' \
    --exclude='golden_dataset.json' \
    --exclude='aura_source_bundle.txt' \
    --exclude='verify_*.py' \
    --exclude='test_*.py' \
    --exclude='debug_*.py' \
    --exclude='*.spec' \
    "${PROJECT_ROOT}/" \
    "${SSH_USER}@${VM_IP}:/tmp/aura_deploy/"

# Move files into place (need sudo for /opt/aura ownership)
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo rsync -a /tmp/aura_deploy/ ${REMOTE_DIR}/ && sudo chown -R aura:aura ${REMOTE_DIR}/"

# ------------------------------------------------------------------
# 2. Deploy .env for cloud
# ------------------------------------------------------------------
echo ">>> [2/3] Setting up cloud environment..."
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo tee ${REMOTE_DIR}/.env > /dev/null" << 'ENV'
# Aura Cloud Configuration
LLM_PROVIDER=local
LLM_MODEL=qwen2.5:14b
AURA_LLM_PROVIDER=local
AURA_LLM_MODEL=qwen2.5:14b
AURA_LLM_BASE_URL=http://localhost:11434
AURA_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
ENV

ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo chown aura:aura ${REMOTE_DIR}/.env"

# ------------------------------------------------------------------
# 3. Install deps + restart
# ------------------------------------------------------------------
if [ "${UPDATE_ONLY}" != "--update" ]; then
    echo ">>> [3/3] Installing Python dependencies..."
    ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo -u aura ${REMOTE_DIR}/venv/bin/pip install --upgrade pip && sudo -u aura ${REMOTE_DIR}/venv/bin/pip install -r ${REMOTE_DIR}/requirements.txt && sudo -u aura ${REMOTE_DIR}/venv/bin/pip install websockets"
else
    echo ">>> [3/3] Skipping pip install (--update mode)"
fi

echo ">>> Restarting Aura service..."
ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo systemctl restart aura"

# Wait and verify
echo ">>> Waiting 15s for Aura to start..."
sleep 15

STATUS=$(ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "curl -s http://127.0.0.1:8000/api/state 2>/dev/null" || echo "FAILED")

if echo "${STATUS}" | grep -q "cycle"; then
    CYCLE=$(echo "${STATUS}" | python3 -c "import sys,json; print(json.load(sys.stdin)['cycle'])" 2>/dev/null || echo "?")
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   DEPLOYMENT SUCCESSFUL ✓                ║"
    echo "║   Cycle: ${CYCLE}                        "
    echo "║   URL: http://${VM_IP}                   "
    echo "╚══════════════════════════════════════════╝"
else
    echo ""
    echo " Service may still be starting. Check with:"
    echo "   ssh ${SSH_USER}@${VM_IP} 'sudo journalctl -u aura -f'"
    echo "   ssh ${SSH_USER}@${VM_IP} 'curl -s http://127.0.0.1:8000/api/state'"
fi
