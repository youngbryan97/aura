#!/usr/bin/env python3
"""Write the OCI retry launch script cleanly (avoids shell heredoc corruption)."""
import os

SCRIPT = r'''#!/usr/bin/env bash
# Aura Cloud - OCI ARM Instance Auto-Retry Launcher
# Retries until ARM capacity opens up in us-sanjose-1
#
# Usage:  nohup ./oci_retry_launch.sh &
# Logs:   $TMPDIR/oci_retry.log
# PID:    $TMPDIR/oci_retry.pid
# Stop:   kill $(cat "$TMPDIR/oci_retry.pid")

set -euo pipefail

# Define variables (from Env)
COMPARTMENT="${OCI_COMPARTMENT_ID:-}"
AD="RcOb:US-SANJOSE-1-AD-1"
SHAPE="VM.Standard.A1.Flex"
OCPUS=4
MEMORY_GB=24
# SUBNET/IMAGE logic moved to runtime env check
SUBNET="${OCI_SUBNET_ID:-}"
IMAGE="${OCI_IMAGE_ID:-}"
DISPLAY_NAME="aura-cloud"
BOOT_VOL_GB=200
SSH_KEY_FILE="$HOME/.ssh/aura-oracle.key.pub"
TMP_DIR="${TMPDIR:-/tmp}"

RETRY_INTERVAL=120
MAX_ATTEMPTS=720
LOG_FILE="${TMP_DIR%/}/oci_retry.log"
PID_FILE="${TMP_DIR%/}/oci_retry.pid"
RESULT_FILE="${TMP_DIR%/}/oci_launch_result.json"
SHAPE_JSON="${TMP_DIR%/}/oci_shape.json"
META_JSON="${TMP_DIR%/}/oci_meta.json"
export SHAPE_JSON META_JSON

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

notify_success() {
    osascript -e "display notification \"Instance launched!\" with title \"Aura Cloud LIVE\" sound name \"Glass\"" 2>/dev/null || true
    say "Aura cloud server launched successfully" 2>/dev/null || true
}

cleanup() { rm -f "$PID_FILE"; log "Script stopped (PID $$)"; }

# Write JSON params via Python (safe escaping)
write_json_params() {
    python3 << 'PYEOF'
import json
import os
with open(os.environ["SHAPE_JSON"], "w") as f:
    json.dump({"ocpus": 4, "memoryInGBs": 24}, f)
key_file = os.path.expanduser("~/.ssh/aura-oracle.key.pub")
key = open(key_file).read().strip()
with open(os.environ["META_JSON"], "w") as f:
    json.dump({"ssh_authorized_keys": key}, f)
PYEOF
}

# Pre-flight checks
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Already running (PID $OLD_PID). Kill first: kill $OLD_PID"
        exit 1
    fi
fi

command -v oci &>/dev/null || { echo "ERROR: oci CLI not found"; exit 1; }
[ -f "$SSH_KEY_FILE" ] || { echo "ERROR: SSH key not found at $SSH_KEY_FILE"; exit 1; }

write_json_params

# Check if instance already exists
for ST in RUNNING PROVISIONING; do
    EX=$(oci compute instance list --compartment-id "$COMPARTMENT" \
        --display-name "$DISPLAY_NAME" --lifecycle-state "$ST" \
        --query 'data[0].id' --raw-output 2>/dev/null || echo "")
    if [ -n "$EX" ] && [ "$EX" != "None" ] && [ "$EX" != "null" ]; then
        log "Instance already $ST: $EX"; exit 0
    fi
done

# Main retry loop
trap cleanup EXIT
echo $$ > "$PID_FILE"
log "======================================================="
log "  Aura Cloud OCI Retry Launcher"
log "  $SHAPE ($OCPUS OCPU / ${MEMORY_GB}GB) in us-sanjose-1"
log "  Every ${RETRY_INTERVAL}s, up to $MAX_ATTEMPTS attempts"
log "  PID: $$"
log "======================================================="

ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "Attempt $ATTEMPT/$MAX_ATTEMPTS..."

    RESPONSE=$(oci compute instance launch \
        --compartment-id "$COMPARTMENT" \
        --availability-domain "$AD" \
        --shape "$SHAPE" \
        --shape-config "file://$SHAPE_JSON" \
        --subnet-id "$SUBNET" \
        --image-id "$IMAGE" \
        --display-name "$DISPLAY_NAME" \
        --assign-public-ip true \
        --boot-volume-size-in-gbs "$BOOT_VOL_GB" \
        --metadata "file://$META_JSON" \
        2>&1) || true

    echo "$RESPONSE" > "$RESULT_FILE"

    if echo "$RESPONSE" | grep -q '"lifecycle-state"'; then
        ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null || echo "?")
        log "======================================================="
        log "  SUCCESS on attempt $ATTEMPT!"
        log "  Instance: $ID"
        log "======================================================="

        log "Waiting for public IP..."
        for i in $(seq 1 30); do
            sleep 10
            VNIC=$(oci compute vnic-attachment list --compartment-id "$COMPARTMENT" \
                --instance-id "$ID" --query 'data[0]."vnic-id"' --raw-output 2>/dev/null || echo "")
            if [ -n "$VNIC" ] && [ "$VNIC" != "None" ] && [ "$VNIC" != "null" ]; then
                IP=$(oci network vnic get --vnic-id "$VNIC" \
                    --query 'data."public-ip"' --raw-output 2>/dev/null || echo "")
                if [ -n "$IP" ] && [ "$IP" != "None" ] && [ "$IP" != "null" ]; then
                    log "Public IP: $IP"
                    log "SSH: ssh -i ~/.ssh/aura-oracle.key opc@$IP"
                    echo "$IP" > "${TMP_DIR%/}/aura_cloud_ip.txt"
                    notify_success
                    exit 0
                fi
            fi
        done
        log "Launched but no IP yet. Check OCI console."
        notify_success
        exit 0
    fi

    if echo "$RESPONSE" | grep -q "Out of host capacity"; then
        log "  -> Out of capacity"
    elif echo "$RESPONSE" | grep -q "TooManyRequests"; then
        log "  -> Rate limited, waiting 5min"; sleep 300; continue
    elif [ -z "$RESPONSE" ]; then
        log "  -> Empty/timeout"
    else
        MSG=$(echo "$RESPONSE" | grep -o '"message": "[^"]*"' | head -1)
        log "  -> $MSG"
    fi

    H=$(date +%H)
    if [ "$H" -ge 2 ] && [ "$H" -le 6 ]; then sleep 60; else sleep "$RETRY_INTERVAL"; fi
done

log "All $MAX_ATTEMPTS attempts exhausted."
osascript -e "display notification \"All attempts exhausted\" with title \"Aura Cloud\" sound name \"Basso\"" 2>/dev/null || true
exit 1
'''

path = os.path.expanduser("~/.gemini/antigravity/scratch/autonomy_engine/cloud/oci_retry_launch.sh")
with open(path, 'w') as f:
    f.write(SCRIPT)
os.chmod(path, 0o755)
print(f"Written: {path} ({os.path.getsize(path)} bytes)")
