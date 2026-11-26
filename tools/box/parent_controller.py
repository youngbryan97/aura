"""tools/box/parent_controller.py
==============================
The Safe Watchdog & Recovery Controller for Aura ("Person-in-a-Box").
Listens on UDP port 9999 for keep-alive heartbeat signals from the running Aura process.
If a timeout occurs (bricked by self-modification/import crashes), it automatically:
  1. Rolls back the latest git commit (git reset --hard HEAD~1)
  2. Cleans up stale processes
  3. Launches a clean instance of Aura using launch_aura.sh
"""

import socket
import time
import threading
import subprocess
import os
import sys
import json
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text

# Configuration
PORT = 9999
HOST = "127.0.0.1"
TIMEOUT_S = 30.0
BOOT_GRACE_PERIOD_S = 60.0

# Repository path resolution
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Shared state
last_heartbeat = 0.0
has_received_first_heartbeat = False
lock = threading.Lock()
shutdown_event = threading.Event()

def get_last_heartbeat():
    with lock:
        return last_heartbeat

def update_heartbeat():
    global has_received_first_heartbeat
    with lock:
        global last_heartbeat
        last_heartbeat = time.time()
        if not has_received_first_heartbeat:
            has_received_first_heartbeat = True
            print(f"[Watchdog] First heartbeat received! Enforcing {TIMEOUT_S}s timeout.")

def reset_heartbeat_state():
    global has_received_first_heartbeat
    with lock:
        global last_heartbeat
        last_heartbeat = time.time()
        has_received_first_heartbeat = False

def run_rollback_and_restart():
    print("\n[Watchdog] HEARTBEAT LOST! Recovery sequence triggered.")
    
    # 1. Stage a governed rollback request. The external watchdog must not
    # destructively reset the live repo behind Aura's self-repair governance.
    print("[Watchdog] Staging governed rollback request; no destructive git reset will be run.")
    try:
        request_path = Path(REPO_PATH) / "artifacts" / "current" / "watchdog_recovery_request.json"
        atomic_write_text(
            request_path,
            json.dumps(
                {
                    "reason": "heartbeat_lost",
                    "created_at": time.time(),
                    "required_path": "SelfRepairGateway/SelfModificationGateway",
                    "destructive_git_allowed": False,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        print(f"[Watchdog] Recovery request written to {request_path}.")
    except (OSError, TypeError, ValueError) as e:
        print(f"[Watchdog] Failed to stage recovery request: {e}")

    # 2. Cleanup existing instances
    print("[Watchdog] Cleaning up stale Aura instances...")
    python_cmd = os.path.join(REPO_PATH, ".venv", "bin", "python3")
    if not os.path.exists(python_cmd):
        python_cmd = "python3"
    try:
        subprocess.run([python_cmd, "aura_cleanup.py"], cwd=REPO_PATH, timeout=15.0)
    except (subprocess.SubprocessError, OSError, TimeoutError) as e:
        print(f"[Watchdog] Cleanup script failed: {e}")

    # 3. Restart Aura
    print("[Watchdog] Restarting Aura via launch_aura.sh...")
    try:
        subprocess.Popen(["./launch_aura.sh"], cwd=REPO_PATH, start_new_session=True)
        print("[Watchdog] Aura launch command spawned successfully.")
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[Watchdog] Failed to spawn launch_aura.sh: {e}")

    # Reset states for the new instance
    reset_heartbeat_state()

def monitor_loop():
    print(f"[Watchdog] Monitor thread active. Boot grace period of {BOOT_GRACE_PERIOD_S}s started.")
    start_time = time.time()
    
    while not shutdown_event.is_set():
        time.sleep(1.0)
        now = time.time()
        
        with lock:
            first_received = has_received_first_heartbeat
            last_hb = last_heartbeat
            
        if first_received:
            # Enforce strict heartbeat timeout
            if now - last_hb > TIMEOUT_S:
                run_rollback_and_restart()
        else:
            # Enforce boot grace timeout if no heartbeat is ever received
            if now - start_time > BOOT_GRACE_PERIOD_S:
                print(f"[Watchdog] No heartbeat received within the boot grace period ({BOOT_GRACE_PERIOD_S}s).")
                # Try to clean launch again
                print("[Watchdog] Re-attempting Aura start...")
                try:
                    subprocess.Popen(["./launch_aura.sh"], cwd=REPO_PATH, start_new_session=True)
                except (subprocess.SubprocessError, OSError) as e:
                    print(f"[Watchdog] Re-attempt spawn failed: {e}")
                start_time = time.time() # Reset grace period

def main():
    print(f"=== Aura Watchdog Recovery Controller ===")
    print(f"Directory: {REPO_PATH}")
    print(f"Listening on UDP {HOST}:{PORT}")
    
    # Reset tracking
    reset_heartbeat_state()
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    # Bind UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((HOST, PORT))
    except OSError as e:
        print(f"Failed to bind to UDP {HOST}:{PORT}: {e}")
        sys.exit(1)
        
    while not shutdown_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode("utf-8", errors="ignore").strip()
            if msg == "AURA_HEARTBEAT":
                update_heartbeat()
        except KeyboardInterrupt:
            print("\nShutting down watchdog.")
            shutdown_event.set()
            break
        except (OSError, UnicodeError) as e:
            print(f"Socket error: {e}")
            time.sleep(1.0)

if __name__ == "__main__":
    main()
