#!/usr/bin/env bash
# ============================================================================
# Oracle Cloud ARM Instance Auto-Retry Grabber
# ============================================================================
# Keeps trying to create a VM.Standard.A1.Flex instance until capacity opens up.
# Typically succeeds within 1-6 hours. Run it and walk away.
#
# Usage:
#   ./cloud/oci_grab_instance.sh
#
# Prerequisites:
#   1. OCI CLI installed: pip3 install oci-cli
#   2. OCI CLI configured: oci setup config
#   3. Fill in the variables below from your Oracle Cloud console
# ============================================================================
set -uo pipefail

# ============================================================================
# FILL THESE IN (from Oracle Cloud Console)
# ============================================================================

# Compartment OCID — find at: Identity → Compartments → root compartment
# Starts with: ocid1.tenancy.oc1..
COMPARTMENT_ID="${OCI_COMPARTMENT_ID:-}"

# Availability Domain — e.g., "Enyx:US-ASHBURN-AD-1" or "Enyx:UK-LONDON-1-AD-1"
# Find at: Compute → Instances → Create Instance → look at "Availability domain"
# Or run: oci iam availability-domain list --query 'data[].name' --raw-output
AVAILABILITY_DOMAIN="${OCI_AVAILABILITY_DOMAIN:-RcOb:US-SANJOSE-1-AD-1}"

# Subnet OCID — after creating instance once (even if failed), a VCN+subnet was created
# Find at: Networking → Virtual Cloud Networks → click your VCN → Subnets → copy OCID
SUBNET_ID="${OCI_SUBNET_ID:-}"

# SSH Public Key — paste the FULL public key content
# This should match the key you downloaded earlier
SSH_PUBLIC_KEY="${OCI_SSH_PUBLIC_KEY:-ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDGYl9CbsgGn6QDzhhAInlf09cp1ABCQ2zaofpiSolzW2lQb9/K1d8QkBK7UqLDXoVricua+lhKQ7JEZSxq8ybc7NWM5qvABOa7omYaqWz56R/1MbYohbUNyfrUipN9iGKyGK7MPgDNKfwaL5kSfYnMdSC4WXeliYZ3L6lsZMW1evx4u95RHHGHXsZtAXYjIccntVFlN24fRSy5xCNW71fnjAD07RStbHou0cOko3xYGiU0Vd4lLJq2ESZXHNfkBYf9Z4yuZg5k9cvMagfcrpGckaWmQbwxUFw684GVj7Yg37K2L4J2VFsfeSz3Qn90OncU69A+2HXime+UIAxvvrCn ssh-key-2026-02-09}"

# ============================================================================
# Instance Configuration (defaults are max Always Free)
# ============================================================================
DISPLAY_NAME="aura-cloud"
SHAPE="VM.Standard.A1.Flex"
OCPUS=4
MEMORY_GB=24
BOOT_VOLUME_GB=200
IMAGE_ID=""  # Will be auto-detected if empty

# Retry settings
RETRY_INTERVAL=60  # seconds between attempts
MAX_ATTEMPTS=0     # 0 = infinite (keep trying until success)

# ============================================================================
# Script Logic — Don't modify below unless you know what you're doing
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ORACLE CLOUD ARM INSTANCE GRABBER                 ║"
echo "║   Will retry until capacity is available             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# --- Validate required fields ---
MISSING=0
if [ -z "$COMPARTMENT_ID" ]; then error "COMPARTMENT_ID is empty — fill it in the script"; MISSING=1; fi
if [ -z "$AVAILABILITY_DOMAIN" ]; then error "AVAILABILITY_DOMAIN is empty — fill it in the script"; MISSING=1; fi
if [ -z "$SUBNET_ID" ]; then error "SUBNET_ID is empty — fill it in the script"; MISSING=1; fi
if [ -z "$SSH_PUBLIC_KEY" ]; then error "SSH_PUBLIC_KEY is empty — fill it in the script"; MISSING=1; fi
if [ $MISSING -eq 1 ]; then
    echo ""
    echo "To find these values:"
    echo "  COMPARTMENT_ID:      Identity → Compartments → root → copy OCID"
    echo "  AVAILABILITY_DOMAIN: Run: oci iam availability-domain list --query 'data[].name' --raw-output"
    echo "  SUBNET_ID:           Networking → VCNs → your VCN → Subnets → copy OCID"
    echo "  SSH_PUBLIC_KEY:      cat ~/.ssh/aura-oracle.key.pub (or the downloaded .pub file)"
    exit 1
fi

# --- Auto-detect image if not provided ---
if [ -z "$IMAGE_ID" ]; then
    log "Auto-detecting latest Ubuntu 24.04 ARM image..."
    IMAGE_ID=$(oci compute image list \
        --compartment-id "$COMPARTMENT_ID" \
        --operating-system "Canonical Ubuntu" \
        --operating-system-version "24.04" \
        --shape "$SHAPE" \
        --sort-by TIMECREATED \
        --sort-order DESC \
        --limit 1 \
        --query 'data[0].id' \
        --raw-output 2>/dev/null)

    if [ -z "$IMAGE_ID" ] || [ "$IMAGE_ID" = "None" ]; then
        error "Could not auto-detect image. Please set IMAGE_ID manually."
        echo "  Run: oci compute image list --compartment-id $COMPARTMENT_ID --operating-system 'Canonical Ubuntu' --shape $SHAPE --query 'data[].{name:\"display-name\",id:id}'"
        exit 1
    fi
    success "Found image: $IMAGE_ID"
fi

# --- Summary ---
echo ""
log "Configuration:"
echo "  Shape:          $SHAPE ($OCPUS OCPUs, ${MEMORY_GB}GB RAM)"
echo "  Boot Volume:    ${BOOT_VOLUME_GB}GB"
echo "  AD:             $AVAILABILITY_DOMAIN"
echo "  Image:          ${IMAGE_ID:0:40}..."
echo "  Retry interval: ${RETRY_INTERVAL}s"
echo ""
log "Starting instance creation loop. Press Ctrl+C to stop."
echo ""

# ============================================================================
# Reconciliation — CP126: "The script enters an unbounded launch loop without
# first querying for an existing running, provisioning, or previously-created
# instance by idempotency token or durable instance ID. Restarting after a
# lost local receipt can create another VM and boot volume."
#
# The receipt this script wrote was /tmp/aura_cloud_ip.txt, which does not
# survive a reboot. Every rerun therefore started from "I have no instance"
# and asked for another one — and because the loop retries for hours, an
# operator who re-ran it after closing their laptop got a second VM and a
# second 200GB boot volume against a tenancy whose Always Free allowance is
# exactly one. The instances that resulted looked identical and neither was
# obviously the spare.
#
# Two independent defences, because they fail in different situations:
#   1. reconcile_existing — ask the cloud, not the local disk, what exists.
#      Handles the lost-receipt case.
#   2. --opc-retry-token — OCI's idempotency key. Handles the case the query
#      cannot: a launch that succeeded server-side but whose response was
#      lost in transit, which is precisely what a flaky retry loop produces.
# ============================================================================

# A token stable across runs for the same requested instance, so a retried
# launch is recognised by OCI as the same request rather than a new one.
# Derived from the identity of what is being asked for; changing the shape
# or the AD is a different request and correctly gets a different token.
RETRY_TOKEN="$(printf '%s|%s|%s|%s|%s' \
    "$COMPARTMENT_ID" "$AVAILABILITY_DOMAIN" "$DISPLAY_NAME" "$SHAPE" "$SUBNET_ID" \
    | shasum -a 256 | cut -c1-32)"

reconcile_existing() {
    # Print the OCID of an instance that already exists for this request, if
    # any. PROVISIONING and STARTING count: an instance mid-creation is a
    # commitment already made, and launching beside it is the exact
    # double-spend this guards against.
    oci compute instance list \
        --compartment-id "$COMPARTMENT_ID" \
        --display-name "$DISPLAY_NAME" \
        --query "data[?\"lifecycle-state\"=='RUNNING' || \"lifecycle-state\"=='PROVISIONING' || \"lifecycle-state\"=='STARTING'].id | [0]" \
        --raw-output 2>/dev/null | grep -v '^None$' || true
}

report_instance() {
    # Everything the success path reports, for an instance we did not just
    # create. An adopted instance and a fresh one must be equally usable, or
    # operators will re-run the script to get "a proper one".
    local instance_id="$1"
    log "Instance ID: $instance_id"
    local public_ip
    public_ip=$(oci compute instance list-vnics \
        --instance-id "$instance_id" \
        --query 'data[0]."public-ip"' \
        --raw-output 2>/dev/null)
    if [ -n "$public_ip" ] && [ "$public_ip" != "None" ]; then
        echo ""
        success "PUBLIC IP: $public_ip"
        echo ""
        echo "  SSH:    ssh -i ~/.ssh/aura-oracle.key ubuntu@${public_ip}"
        echo "  Deploy: ./cloud/deploy.sh ${public_ip}"
        echo ""
        echo "$public_ip" > /tmp/aura_cloud_ip.txt
    else
        warn "Instance exists but has no public IP yet. Check the Oracle Console."
    fi
}

log "Checking whether an instance already exists for this request..."
EXISTING_ID="$(reconcile_existing)"
if [ -n "$EXISTING_ID" ]; then
    echo ""
    success "═══════════════════════════════════════════"
    success "  INSTANCE ALREADY EXISTS — NOT LAUNCHING"
    success "═══════════════════════════════════════════"
    echo ""
    warn "Found '$DISPLAY_NAME' already running or provisioning."
    warn "Launching again would create a second VM and boot volume."
    report_instance "$EXISTING_ID"
    echo "  To deliberately create another, use a different DISPLAY_NAME."
    echo ""
    exit 0
fi
success "No existing instance found — proceeding."

# --- Retry loop ---
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))

    if [ $MAX_ATTEMPTS -gt 0 ] && [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
        error "Max attempts ($MAX_ATTEMPTS) reached. Giving up."
        exit 1
    fi

    log "Attempt #${ATTEMPT} — requesting instance..."

    RESULT=$(oci compute instance launch \
        --opc-retry-token "$RETRY_TOKEN" \
        --compartment-id "$COMPARTMENT_ID" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --shape "$SHAPE" \
        --shape-config "{\"ocpus\": $OCPUS, \"memoryInGBs\": $MEMORY_GB}" \
        --display-name "$DISPLAY_NAME" \
        --image-id "$IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --assign-public-ip true \
        --boot-volume-size-in-gbs "$BOOT_VOLUME_GB" \
        --metadata "{\"ssh_authorized_keys\": \"$SSH_PUBLIC_KEY\"}" \
        2>&1)

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        success "═══════════════════════════════════════════"
        success "  INSTANCE CREATED SUCCESSFULLY!"
        success "═══════════════════════════════════════════"
        echo ""

        # Extract the instance ID
        INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null)

        if [ -n "$INSTANCE_ID" ]; then
            log "Instance ID: $INSTANCE_ID"
            log "Waiting for instance to reach RUNNING state (up to 10 min)..."
            
            oci compute instance get --instance-id "$INSTANCE_ID" \
                --wait-for-state RUNNING \
                --wait-interval-seconds 15 \
                --max-wait-seconds 600 >/dev/null 2>&1 || true
            
            log "Fetching public IP..."
            sleep 10

            PUBLIC_IP=$(oci compute instance list-vnics \
                --instance-id "$INSTANCE_ID" \
                --query 'data[0]."public-ip"' \
                --raw-output 2>/dev/null)

            if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
                echo ""
                success "PUBLIC IP: $PUBLIC_IP"
                echo ""
                echo "  SSH:    ssh -i ~/.ssh/aura-oracle.key ubuntu@${PUBLIC_IP}"
                echo "  Setup:  scp cloud/setup_server.sh ubuntu@${PUBLIC_IP}:~/"
                echo "          ssh -i ~/.ssh/aura-oracle.key ubuntu@${PUBLIC_IP} 'bash ~/setup_server.sh'"
                echo "  Deploy: ./cloud/deploy.sh ${PUBLIC_IP}"
                echo ""

                # Save IP for convenience
                echo "$PUBLIC_IP" > /tmp/aura_cloud_ip.txt
                success "IP saved to /tmp/aura_cloud_ip.txt"
            else
                warn "Could not fetch public IP yet. Check Oracle Console."
            fi
        fi

        # Play a sound to alert (macOS)
        afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &

        exit 0
    fi

    # Check if it's an "out of capacity" error (expected) vs a real error
    if echo "$RESULT" | grep -qi "capacity\|InternalError"; then
        warn "Out of host capacity. Retrying in ${RETRY_INTERVAL}s... (attempt #${ATTEMPT})"
    elif echo "$RESULT" | grep -qi "limit\|quota"; then
        error "Service limit or quota error:"
        echo "$RESULT" | tail -5
        error "You may need to request a service limit increase."
        exit 1
    else
        warn "Unknown error. Retrying in ${RETRY_INTERVAL}s..."
        echo "$RESULT" | tail -3
    fi

    # An error here does not prove nothing was created. A launch can succeed
    # server-side and still return a failure to us — a dropped connection,
    # an SDK timeout, a signal. Re-check before asking for another one; the
    # retry token covers the same case at the API, and this covers it when
    # the token has been rotated by a config change mid-run.
    RECHECK_ID="$(reconcile_existing)"
    if [ -n "$RECHECK_ID" ]; then
        echo ""
        success "Instance appeared despite the reported error — adopting it."
        report_instance "$RECHECK_ID"
        exit 0
    fi

    sleep "$RETRY_INTERVAL"
done
