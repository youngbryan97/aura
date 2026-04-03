#!/usr/bin/env python3
"""
Aura 3D Launcher for macOS
==========================

This script launches Aura with the 3D VirtualBody viewer enabled.
On macOS, MuJoCo's passive viewer requires the 'mjpython' wrapper to 
handle the Cocoa event loop correctly.

Usage:
    mjpython launch_aura_3d.py
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.3DLauncher")

def check_mjpython():
    """Try to find mjpython in PATH or local .venv."""
    # 1. Check PATH
    try:
        subprocess.run(["mjpython", "--version"], capture_output=True, check=True)
        return "mjpython"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
        
    # 2. Check local .venv
    venv_mjpy = os.path.join(os.getcwd(), ".venv", "bin", "mjpython")
    if os.path.exists(venv_mjpy):
        return venv_mjpy
        
    return None

def main():
    if sys.platform == "darwin":
        mjpython_bin = check_mjpython()
        
        # Check if we are already running under mjpython
        if "mjpython" not in sys.executable and not mjpython_bin:
            logger.error("❌ 'mjpython' not found. Please install it with: pip install mujoco")
            logger.info("Once installed, run this script with: mjpython launch_aura_3d.py")
            sys.exit(1)
            
        if "mjpython" not in sys.executable:
            logger.info(f"🚀 Re-launching Aura with '{mjpython_bin}' for 3D support...")
            cmd = [mjpython_bin, __file__] + sys.argv[1:]
            os.execv(mjpython_bin, cmd)

    logger.info("✨ Starting Aura 3D Interface...")
    
    # Set environment variables
    os.environ["AURA_SHOW_3D"] = "1"
    
    try:
        import asyncio
        import time
        from core.container import ServiceContainer
        from core.state.state_repository import StateRepository
        from core.soma.virtual_body import VirtualBody
        
        async def is_aura_running(repo):
            """Check if another Aura instance is active via state timestamp."""
            latest = await repo._load_latest()
            if latest and (time.time() - latest.updated_at) < 10:
                return True
            return False

        async def sync_mode(repo, body):
            """Viewer-only mode: follow the primary Aura process."""
            logger.info("🔗 [Sync Mode] Connected to live Aura process. Tracking movements...")
            body.show_viewer = True
            body._start_viewer()
            
            while True:
                state = await repo._load_latest()
                if state and state.soma and body.data:
                    # Sync joint positions
                    body.data.qpos[:] = state.soma.qpos
                    # Sync sensors for consistency if needed
                    body.sensors.update(state.soma.sensors)
                await asyncio.sleep(0.05) # 20fps sync

        async def run_launcher():
            # 1. Initialize State Repository to check for life
            repo = StateRepository()
            await repo.initialize()
            
            # 2. Setup VirtualBody (Viewer only if syncing)
            body = VirtualBody(show_viewer=False) 
            
            if await is_aura_running(repo):
                # 3a. Sync Mode (Viewer only)
                await sync_mode(repo, body)
            else:
                # 3b. Dual Mode (Start Engine + Viewer)
                logger.info("🌱 [Dual Mode] No active Aura found. Starting full engine...")
                from core.agency_core import AgencyCore
                aura = AgencyCore()
                await aura.start()
                # VirtualBody in Dual mode handles its own viewer if started via AgencyCore
                while True:
                    await asyncio.sleep(3600)
                
        asyncio.run(run_launcher())
    except ImportError as e:
        logger.error(f"❌ Could not import Aura core: {e}")
        logger.info("Ensure you are running from the project root and PYTHONPATH is set.")
    except Exception as e:
        logger.error(f"💥 Runtime error: {e}")

if __name__ == "__main__":
    main()
