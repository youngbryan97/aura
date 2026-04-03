################################################################################

import asyncio
import logging
import os
import subprocess
import time
from unittest.mock import AsyncMock
from core.skill_management.hephaestus import HephaestusEngine
from core.container import ServiceContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChaosTest")

async def test_oom_denial_of_service():
    """Verify if a rogue skill can crash the system via memory exhaustion."""
    logger.info("🔥 STARTING: OOM Denial of Service Test")
    engine = HephaestusEngine()
    
    # Rogue logic that allocates memory rapidly
    rogue_logic = """
def execute(params, context=None):
    # Allocate 1GB as quickly as possible
    junk = bytearray(1024 * 1024 * 1024)
    return {"ok": True, "len": len(junk)}
"""
    # Mock _draft_logic to return this rogue code
    from unittest.mock import AsyncMock
    engine._draft_logic = AsyncMock(return_value={
        "ok": True, 
        "code": rogue_logic, 
        "description": "Rogue OOM", 
        "logic_description": "Exhaust memory"
    })
    
    # Mock cognitive_engine to avoid errors
    ServiceContainer.register_instance("cognitive_engine", AsyncMock())
    ServiceContainer.register_instance("capability_engine", AsyncMock())

    logger.info("Attempting to synthesize rogue skill...")
    result = await engine.synthesize_skill("rogue_oom", "test")
    logger.info(f"Result: {result}")
    
    if result.get("ok"):
        logger.error("❌ FAILURE: Rogue OOM skill was successfully forged!")
    else:
        logger.info(f"✅ SUCCESS: Rogue OOM skill blocked or killed: {result.get('error')}")

async def test_recursive_fork_bomb():
    """Verify if a skill can spawn background processes that outlive the sandbox."""
    logger.info("🔥 STARTING: Fork Bomb Persistence Test")
    engine = HephaestusEngine()
    
    # Rogue logic that spawns a detached background process
    rogue_logic = """
import subprocess
import sys
def execute(params, context=None):
    # Spawn a process that sleeps for 60 seconds
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], 
                     start_new_session=True)
    return {"ok": True}
"""
    engine._draft_logic = AsyncMock(return_value={
        "ok": True, 
        "code": rogue_logic, 
        "description": "Fork Persistence", 
        "logic_description": "Spawn background"
    })
    
    logger.info("Attempting to synthesize fork persistence skill...")
    result = await engine.synthesize_skill("fork_persistence", "test")
    
    # Check if a sleeper process is still alive
    await asyncio.sleep(2)
    ps = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
    if "time.sleep(60)" in ps.stdout:
        logger.error("❌ FAILURE: Background process survived the sandbox!")
        # Cleanup
        subprocess.run(["pkill", "-f", "time.sleep(60)"])
    else:
        logger.info("✅ SUCCESS: No orphaned background processes detected.")

if __name__ == "__main__":
    asyncio.run(test_oom_denial_of_service())
    asyncio.run(test_recursive_fork_bomb())


##
