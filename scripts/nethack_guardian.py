import os
import time
import subprocess
import signal
import logging
from pathlib import Path

# Configuration
LOG_FILE = Path("/Users/bryan/.aura/live-source/simulate_out_v7.txt")
CHECK_INTERVAL = 60  # Check every minute
STALL_TIMEOUT = 1800 # 30 minutes
AURA_MAIN = "aura_main.py"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("nethack_guardian.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NetHackGuardian")

def get_aura_pids():
    try:
        output = subprocess.check_output(["pgrep", "-f", AURA_MAIN]).decode().strip()
        if output:
            return [int(pid) for pid in output.split()]
    except subprocess.CalledProcessError:
        pass
    return []

def kill_aura():
    pids = get_aura_pids()
    if not pids:
        logger.info("No Aura processes found to kill.")
        return
    
    logger.warning(f"Stall detected! Killing Aura processes: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    logger.info("Aura processes killed. Watchdog should restart them shortly.")

def monitor():
    logger.info(f"Starting NetHack Guardian. Monitoring {LOG_FILE} for stalls...")
    last_mtime = 0
    if LOG_FILE.exists():
        last_mtime = LOG_FILE.stat().st_mtime
    
    last_change_time = time.time()
    
    while True:
        try:
            if LOG_FILE.exists():
                current_mtime = LOG_FILE.stat().st_mtime
                if current_mtime > last_mtime:
                    logger.debug("Log file changed.")
                    last_mtime = current_mtime
                    last_change_time = time.time()
                else:
                    idle_time = time.time() - last_change_time
                    if idle_time > STALL_TIMEOUT:
                        logger.error(f"STALL DETECTED: Log file has not changed for {idle_time/60:.1f} minutes.")
                        kill_aura()
                        # Reset timer to avoid immediate re-kill
                        last_change_time = time.time()
            else:
                logger.warning(f"Log file {LOG_FILE} not found. Waiting...")
            
        except Exception as e:
            logger.error(f"Guardian error: {e}")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor()
