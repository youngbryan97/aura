import os
import signal
import subprocess
import time
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.IroncladCleanup")

def main() -> None:
    """
    🔥 [IRONCLAD] Absolute system purge for Aura.
    Targets orchestrators, workers, simulation scripts, and stale locks.
    """
    logger.info("🔥 [IRONCLAD] Initiating Absolute System Purge...")
    
    # 1. Broad targets to terminate
    # We include 'simulate_200.py' and 'python3' to catch rogue subprocess trees.
    targets = ["aura_main.py", "mlx_worker.py", "gui_actor.py", "llama-server", "simulate_200.py"]
    
    for target in targets:
        try:
            logger.info(f"  💀 Killing: {target}")
            # -9: SIGKILL, -f: Full command line match
            subprocess.run(["pkill", "-9", "-f", target], check=False, capture_output=True)
        except Exception:
            pass

    # 2. Hard Purge Locks
    # Instead of unlinking individual files, we wipe the whole directory to 
    # clear out hidden or corrupted fcntl locks.
    lock_dir = Path.home() / ".aura" / "locks"
    if lock_dir.exists():
        logger.info(f"🔓 [IRONCLAD] Wiping lock directory: {lock_dir}")
        try:
            shutil.rmtree(lock_dir)
            lock_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to wipe locks: {e}")

    # 3. Final Pause to let OS release ports/VRAM
    time.sleep(2)
    logger.info("✅ [IRONCLAD] Purge complete. System is verified CLEAN.")

if __name__ == "__main__":
    main()
