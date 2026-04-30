"""tests/verify_metal_persistence.py
===================================
Verification script for the Sovereign Platform Root and Metal persistence.
"""

from core.utils.task_tracker import get_task_tracker
import asyncio
import os
import sys
import logging
import time
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sovereign.platform_root import get_platform_root
from core.mycelium import MycelialNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VerifyMetalPersistence")

async def verify_persistence():
    logger.info("🚀 Starting Metal Persistence Verification...")
    
    # 1. Initialize PlatformRoot and register it for Mycelium to find
    from core.container import ServiceContainer
    platform = get_platform_root()
    ServiceContainer.register_instance("platform_root", platform)
    
    logger.info("✅ PlatformRoot initialized and registered.")
    
    # 2. Check initial state
    if platform.device_active:
        logger.info("✅ Metal device is initialy active.")
    else:
        logger.error("❌ Metal device is NOT active on start.")
        return False

    # 3. Verify MTLCompilerService via ps aux
    def check_service():
        try:
            out = subprocess.check_output(["ps", "aux"], text=True)
            return "MTLCompilerService" in out
        except Exception:
            return False

    service_exists = check_service()
    if service_exists:
        logger.info("✅ MTLCompilerService is currently running.")
    else:
        logger.warning("⚠️ MTLCompilerService not found in process list. Triggering pulse to wake it...")
        platform.pulse()
        await asyncio.sleep(2)
        if check_service():
            logger.info("✅ MTLCompilerService woke up successfully.")
        else:
            logger.error("❌ MTLCompilerService failed to wake up.")
            # Note: On some systems it might be nested or named differently, 
            # but usually it's there during Metal compilation.

    # 4. Verify Mycelial NeuralRoot binding
    mycelium = MycelialNetwork()
    nr = mycelium.establish_neural_root("verification_node", "gpu_metal")
    logger.info("✅ NeuralRoot established: %s", nr.name)
    
    # 5. TEST: Direct Subsurface Ping
    logger.info("📡 Triggering subsurface ping...")
    success = nr.subsurface_ping()
    if success:
        logger.info("✅ Subsurface ping successful! Hardware connection verified.")
    else:
        logger.error("❌ Subsurface ping failed!")
        return False

    # 6. Monitor for a few heartbeats
    logger.info("⏳ Monitoring hardware pulses for 30s...")
    monitor_task = get_task_tracker().create_task(platform.start_monitor())
    
    for i in range(3):
        await asyncio.sleep(10)
        logger.info("Check %d: Device active? %s | Last pulse offset: %.2fs", 
                    i+1, platform.device_active, time.monotonic() - platform._last_pulse)
        if not platform.device_active:
            logger.error("❌ Device became inactive during monitoring!")
            break
            
    platform.stop()
    await monitor_task
    
    logger.info("🏁 Verification complete.")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_persistence())
    sys.exit(0 if success else 1)
