#!/usr/bin/env python3
"""
OCI ARM Instance Launcher — Python SDK version
Avoids CLI "Aborted!" issues. Run and walk away.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import oci

# ─── Configuration ──────────────────────────────────────────
# Configuration (from Environment)
COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID", "")
IMAGE_ID = os.environ.get("OCI_IMAGE_ID", "")
SUBNET_ID = os.environ.get("OCI_SUBNET_ID", "")

if not COMPARTMENT_ID or not SUBNET_ID:
    print("WARNING: OCI configuration missing. Please set OCI_COMPARTMENT_ID and OCI_SUBNET_ID.")

AVAILABILITY_DOMAIN = "RcOb:US-SANJOSE-1-AD-1"
SSH_KEY_FILE = os.path.expanduser("~/.ssh/aura-oracle.key.pub")

SHAPE = "VM.Standard.A1.Flex"
OCPUS = 4
MEMORY_GB = 24
BOOT_VOLUME_GB = 200
DISPLAY_NAME = "aura-cloud"

RETRY_INTERVAL = 60   # seconds between attempts
MAX_ATTEMPTS = 0       # 0 = infinite
CLOUD_IP_FILE = Path(tempfile.gettempdir()) / "aura_cloud_ip.txt"

# ─── Setup ──────────────────────────────────────────────────
config = oci.config.from_file()
compute = oci.core.ComputeClient(config)
network = oci.core.VirtualNetworkClient(config)

# Read SSH key
with open(SSH_KEY_FILE) as f:
    ssh_key = f.read().strip()

print("╔══════════════════════════════════════════════════════╗")
print("║   OCI ARM INSTANCE LAUNCHER (Python SDK)            ║")
print("╚══════════════════════════════════════════════════════╝")
print()

# ─── Find latest Ubuntu 24.04 ARM image ────────────────────
print("[*] Finding latest Ubuntu 24.04 ARM image...")
images = compute.list_images(
    compartment_id=COMPARTMENT_ID,
    operating_system="Canonical Ubuntu",
    operating_system_version="24.04",
    shape=SHAPE,
    sort_by="TIMECREATED",
    sort_order="DESC",
    limit=1
).data

if not images:
    print("[!] No Ubuntu 24.04 ARM image found!")
    sys.exit(1)

IMAGE_ID = images[0].id
print(f"[✓] Image: {images[0].display_name}")
print(f"    ID: {IMAGE_ID[:60]}...")
print()

# ─── Launch configuration ──────────────────────────────────
launch_details = oci.core.models.LaunchInstanceDetails(
    compartment_id=COMPARTMENT_ID,
    availability_domain=AVAILABILITY_DOMAIN,
    display_name=DISPLAY_NAME,
    shape=SHAPE,
    shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
        ocpus=float(OCPUS),
        memory_in_gbs=float(MEMORY_GB)
    ),
    source_details=oci.core.models.InstanceSourceViaImageDetails(
        image_id=IMAGE_ID,
        boot_volume_size_in_gbs=BOOT_VOLUME_GB
    ),
    create_vnic_details=oci.core.models.CreateVnicDetails(
        subnet_id=SUBNET_ID,
        assign_public_ip=True
    ),
    metadata={
        "ssh_authorized_keys": ssh_key
    }
)

print(f"[*] Config: {SHAPE} ({OCPUS} OCPUs, {MEMORY_GB}GB RAM, {BOOT_VOLUME_GB}GB disk)")
print(f"[*] AD: {AVAILABILITY_DOMAIN}")
print(f"[*] Retry interval: {RETRY_INTERVAL}s")
print()
print("[*] Starting launch loop. Ctrl+C to stop.")
print()

# ─── Retry loop ────────────────────────────────────────────
attempt = 0
while True:
    attempt += 1
    if MAX_ATTEMPTS > 0 and attempt > MAX_ATTEMPTS:
        print(f"[!] Max attempts ({MAX_ATTEMPTS}) reached. Giving up.")
        sys.exit(1)

    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] Attempt #{attempt}...", end=" ", flush=True)

    try:
        response = compute.launch_instance(launch_details)
        instance = response.data

        print()
        print()
        print("═" * 55)
        print("  ✓ INSTANCE CREATED SUCCESSFULLY!")
        print("═" * 55)
        print()
        print(f"  Instance ID: {instance.id}")
        print(f"  State: {instance.lifecycle_state}")
        print()

        # Wait for RUNNING
        print("[*] Waiting for instance to reach RUNNING state...")
        get_response = oci.wait_until(
            compute,
            compute.get_instance(instance.id),
            'lifecycle_state',
            'RUNNING',
            max_interval_seconds=15,
            max_wait_seconds=600
        )
        print("[✓] Instance is RUNNING!")

        # Get public IP
        time.sleep(10)
        vnics = compute.list_vnic_attachments(
            compartment_id=COMPARTMENT_ID,
            instance_id=instance.id
        ).data

        public_ip = None
        for vnic_att in vnics:
            if vnic_att.lifecycle_state == "ATTACHED":
                vnic = network.get_vnic(vnic_att.vnic_id).data
                if vnic.public_ip:
                    public_ip = vnic.public_ip
                    break

        if public_ip:
            print()
            print(f"  ✓ PUBLIC IP: {public_ip}")
            print()
            print(f"  SSH: ssh -i ~/.ssh/aura-oracle.key ubuntu@{public_ip}")
            print()

            # Save IP
            with open(CLOUD_IP_FILE, "w") as f:
                f.write(public_ip)
            print(f"  IP saved to {CLOUD_IP_FILE}")
        else:
            print("[!] Could not fetch public IP yet. Check Oracle Console.")

        # Play sound (macOS)
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sys.exit(0)

    except oci.exceptions.ServiceError as e:
        if e.status == 500 and ("capacity" in str(e.message).lower() or "InternalError" in str(e.code)):
            print(f"Out of capacity. Retrying in {RETRY_INTERVAL}s...")
        elif e.status == 429:
            wait_time = RETRY_INTERVAL * 2
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time - RETRY_INTERVAL)  # extra wait
        elif "limit" in str(e.message).lower() or "quota" in str(e.message).lower():
            print()
            print(f"[!] Service limit error: {e.message}")
            sys.exit(1)
        else:
            print(f"Error ({e.status}): {e.message[:100]}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    time.sleep(RETRY_INTERVAL)
