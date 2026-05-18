#!/usr/bin/env python3
"""
OCI ARM Instance Launcher — Multi-Region Edition
Cycles through regions to find Ampere A1 capacity.
Creates networking on-the-fly in each region if needed.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import oci

# ─── Configuration ──────────────────────────────────────────
# Configuration
COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID", "")
SSH_KEY_FILE = os.path.expanduser("~/.ssh/aura-oracle.key.pub")

SHAPE = "VM.Standard.A1.Flex"
OCPUS = 4
MEMORY_GB = 24
BOOT_VOLUME_GB = 200
DISPLAY_NAME = "aura-cloud"

RETRY_INTERVAL = 45   # seconds between attempts (faster with multi-region)
STATE_FILE = Path(tempfile.gettempdir()) / "oci_multi_region_state.json"
CLOUD_IP_FILE = Path(tempfile.gettempdir()) / "aura_cloud_ip.txt"

# Regions most likely to have free-tier A1 capacity
# Ordered by typical availability (less popular = more capacity)
REGIONS = [
    "us-sanjose-1",
    "us-phoenix-1",
    "us-ashburn-1",
    "ca-toronto-1",
    "ca-montreal-1",
    "eu-frankfurt-1",
    "eu-amsterdam-1",
    "uk-london-1",
    "ap-tokyo-1",
    "ap-osaka-1",
    "ap-sydney-1",
    "ap-melbourne-1",
    "sa-saopaulo-1",
    "me-jeddah-1",
    "af-johannesburg-1",
    "ap-singapore-1",
    "ap-seoul-1",
    "eu-marseille-1",
    "eu-zurich-1",
    "eu-milan-1",
]

# ─── Load/save state (track which regions have networking set up) ───
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"networks": {}, "total_attempts": 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ─── Setup ──────────────────────────────────────────────────
config = oci.config.from_file()

with open(SSH_KEY_FILE) as f:
    ssh_key = f.read().strip()

state = load_state()

print("╔══════════════════════════════════════════════════════╗")
print("║   OCI ARM LAUNCHER — MULTI-REGION ROTATION          ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"  Regions: {len(REGIONS)}")
print(f"  Previous attempts: {state['total_attempts']}")
print(f"  Regions with networking: {len(state['networks'])}")
print()

def get_clients(region):
    """Get OCI clients for a specific region."""
    cfg = dict(config)
    cfg["region"] = region
    return (
        oci.core.ComputeClient(cfg),
        oci.core.VirtualNetworkClient(cfg),
        oci.identity.IdentityClient(cfg),
        cfg
    )

def ensure_networking(region, vn_client, identity_client):
    """Create VCN + subnet in a region if not already done."""
    if region in state["networks"]:
        # Verify it still exists
        try:
            vn_client.get_subnet(state["networks"][region]["subnet_id"])
            return state["networks"][region]
        except oci.exceptions.ServiceError:
            print("    [!] Cached network gone, recreating...")
            del state["networks"][region]

    print(f"    [*] Creating networking in {region}...")

    # Create VCN
    vcn = vn_client.create_vcn(
        oci.core.models.CreateVcnDetails(
            compartment_id=COMPARTMENT_ID,
            cidr_block="10.0.0.0/16",
            display_name=f"aura-vcn-{region}",
        )
    ).data

    # Wait for VCN
    vcn = oci.wait_until(vn_client, vn_client.get_vcn(vcn.id), 'lifecycle_state', 'AVAILABLE').data

    # Create internet gateway
    igw = vn_client.create_internet_gateway(
        oci.core.models.CreateInternetGatewayDetails(
            compartment_id=COMPARTMENT_ID,
            vcn_id=vcn.id,
            display_name="aura-igw",
            is_enabled=True,
        )
    ).data
    igw = oci.wait_until(vn_client, vn_client.get_internet_gateway(igw.id), 'lifecycle_state', 'AVAILABLE').data

    # Update default route table to use IGW
    rt = vn_client.get_route_table(vcn.default_route_table_id).data
    vn_client.update_route_table(
        rt.id,
        oci.core.models.UpdateRouteTableDetails(
            route_rules=[
                oci.core.models.RouteRule(
                    destination="0.0.0.0/0",
                    destination_type="CIDR_BLOCK",
                    network_entity_id=igw.id,
                )
            ]
        )
    )

    # Update default security list to allow SSH + HTTP
    sl = vn_client.get_security_list(vcn.default_security_list_id).data
    vn_client.update_security_list(
        sl.id,
        oci.core.models.UpdateSecurityListDetails(
            ingress_security_rules=[
                oci.core.models.IngressSecurityRule(
                    protocol="6", source="0.0.0.0/0",
                    tcp_options=oci.core.models.TcpOptions(
                        destination_port_range=oci.core.models.PortRange(min=22, max=22))
                ),
                oci.core.models.IngressSecurityRule(
                    protocol="6", source="0.0.0.0/0",
                    tcp_options=oci.core.models.TcpOptions(
                        destination_port_range=oci.core.models.PortRange(min=8000, max=8000))
                ),
                oci.core.models.IngressSecurityRule(
                    protocol="6", source="0.0.0.0/0",
                    tcp_options=oci.core.models.TcpOptions(
                        destination_port_range=oci.core.models.PortRange(min=443, max=443))
                ),
                oci.core.models.IngressSecurityRule(
                    protocol="6", source="0.0.0.0/0",
                    tcp_options=oci.core.models.TcpOptions(
                        destination_port_range=oci.core.models.PortRange(min=80, max=80))
                ),
            ],
            egress_security_rules=[
                oci.core.models.EgressSecurityRule(
                    protocol="all", destination="0.0.0.0/0")
            ]
        )
    )

    # Get availability domain
    ads = identity_client.list_availability_domains(compartment_id=COMPARTMENT_ID).data
    ad_name = ads[0].name  # Use first AD

    # Create subnet
    subnet = vn_client.create_subnet(
        oci.core.models.CreateSubnetDetails(
            compartment_id=COMPARTMENT_ID,
            vcn_id=vcn.id,
            cidr_block="10.0.1.0/24",
            display_name="aura-subnet",
            availability_domain=ad_name,
            route_table_id=vcn.default_route_table_id,
            security_list_ids=[vcn.default_security_list_id],
        )
    ).data
    subnet = oci.wait_until(vn_client, vn_client.get_subnet(subnet.id), 'lifecycle_state', 'AVAILABLE').data

    info = {
        "vcn_id": vcn.id,
        "subnet_id": subnet.id,
        "ad_name": ad_name,
        "igw_id": igw.id,
    }
    state["networks"][region] = info
    save_state(state)
    print(f"    [✓] Networking ready in {region} (AD: {ad_name})")
    return info

def find_image(compute_client, region):
    """Find Ubuntu 24.04 ARM image in region."""
    try:
        images = compute_client.list_images(
            compartment_id=COMPARTMENT_ID,
            operating_system="Canonical Ubuntu",
            operating_system_version="24.04",
            shape=SHAPE,
            sort_by="TIMECREATED",
            sort_order="DESC",
            limit=1
        ).data
        if images:
            return images[0].id
        # Fallback to 22.04
        images = compute_client.list_images(
            compartment_id=COMPARTMENT_ID,
            operating_system="Canonical Ubuntu",
            operating_system_version="22.04",
            shape=SHAPE,
            sort_by="TIMECREATED",
            sort_order="DESC",
            limit=1
        ).data
        return images[0].id if images else None
    except Exception:
        return None

def try_launch(region):
    """Attempt to launch instance in a specific region."""
    compute, vn_client, identity, cfg = get_clients(region)

    # Ensure networking
    try:
        net = ensure_networking(region, vn_client, identity)
    except oci.exceptions.ServiceError as e:
        if "NotAuthorizedOrNotFound" in str(e) or "limit" in str(e.message).lower():
            print(f"    [!] Region {region} not subscribed or limited. Skipping.")
            return "skip"
        raise

    # Find image
    image_id = find_image(compute, region)
    if not image_id:
        print(f"    [!] No ARM image in {region}. Skipping.")
        return "skip"

    # Build launch details
    launch_details = oci.core.models.LaunchInstanceDetails(
        compartment_id=COMPARTMENT_ID,
        availability_domain=net["ad_name"],
        display_name=DISPLAY_NAME,
        shape=SHAPE,
        shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
            ocpus=float(OCPUS),
            memory_in_gbs=float(MEMORY_GB)
        ),
        source_details=oci.core.models.InstanceSourceViaImageDetails(
            image_id=image_id,
            boot_volume_size_in_gbs=BOOT_VOLUME_GB
        ),
        create_vnic_details=oci.core.models.CreateVnicDetails(
            subnet_id=net["subnet_id"],
            assign_public_ip=True
        ),
        metadata={"ssh_authorized_keys": ssh_key}
    )

    response = compute.launch_instance(launch_details)
    instance = response.data

    print()
    print("═" * 55)
    print(f"  ✓ INSTANCE CREATED IN {region.upper()}!")
    print("═" * 55)
    print(f"  Instance ID: {instance.id}")

    # Wait for RUNNING
    print("  [*] Waiting for RUNNING state...")
    oci.wait_until(
        compute, compute.get_instance(instance.id),
        'lifecycle_state', 'RUNNING',
        max_interval_seconds=15, max_wait_seconds=600
    )
    print("  [✓] Instance is RUNNING!")

    # Get public IP
    time.sleep(10)
    vnics = compute.list_vnic_attachments(
        compartment_id=COMPARTMENT_ID, instance_id=instance.id
    ).data
    public_ip = None
    for va in vnics:
        if va.lifecycle_state == "ATTACHED":
            vnic = vn_client.get_vnic(va.vnic_id).data
            if vnic.public_ip:
                public_ip = vnic.public_ip
                break

    if public_ip:
        print(f"\n  ✓ PUBLIC IP: {public_ip}")
        print(f"  ✓ REGION: {region}")
        print(f"\n  SSH: ssh -i ~/.ssh/aura-oracle.key ubuntu@{public_ip}\n")
        with open(CLOUD_IP_FILE, "w") as f:
            f.write(f"{public_ip}\n{region}\n")
    else:
        print("  [!] Could not get public IP. Check console.")

    subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "success"

# ─── Main rotation loop ────────────────────────────────────
print("[*] Starting multi-region rotation. Ctrl+C to stop.\n")

skip_regions = set()
region_idx = 0

while True:
    region = REGIONS[region_idx % len(REGIONS)]
    region_idx += 1

    if region in skip_regions:
        continue

    state["total_attempts"] += 1
    attempt = state["total_attempts"]
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] #{attempt} → {region}", end=" ", flush=True)

    try:
        result = try_launch(region)
        if result == "success":
            save_state(state)
            sys.exit(0)
        elif result == "skip":
            skip_regions.add(region)

    except oci.exceptions.ServiceError as e:
        if e.status == 500 and "capacity" in str(e.message).lower():
            print("Out of capacity.")
        elif e.status == 429:
            print("Rate limited. Extra wait...")
            time.sleep(RETRY_INTERVAL)
        elif "limit" in str(e.message).lower() or "quota" in str(e.message).lower():
            print(f"Limit reached: {e.message[:80]}")
            skip_regions.add(region)
        elif "NotAuthorizedOrNotFound" in str(e.code):
            print("Not subscribed. Skipping.")
            skip_regions.add(region)
        else:
            print(f"Error ({e.status}): {e.message[:80]}")

    except Exception as e:
        print(f"Error: {str(e)[:80]}")

    save_state(state)

    # Shorter sleep when rotating (we're hitting different regions)
    if len(REGIONS) - len(skip_regions) > 3:
        time.sleep(RETRY_INTERVAL // len(REGIONS) * 3 + 5)  # ~10-15s between regions
    else:
        time.sleep(RETRY_INTERVAL)
