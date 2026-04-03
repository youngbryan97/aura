################################################################################


import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.soul import Soul, Drive
from core.orchestrator import RobustOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SoulTest")

async def test_soul_triggers():
    logger.info("🚀 Starting Soul Autonomy Trigger Test...")
    
    # 1. Setup Orchestrator (Mocked where needed)
    orchestrator = RobustOrchestrator()
    # Ensure clean state for test - boredom might be loaded from stale snapshot
    orchestrator.boredom = 0.0
    
    orchestrator.curiosity = MagicMock()
    orchestrator.volition = MagicMock()
    orchestrator.volition.last_speak_time = 3600 # Assume we spoke an hour ago
    
    # Mock execute_tool for competence drive
    orchestrator.execute_tool = AsyncMock(return_value={"ok": True})
    
    soul = Soul(orchestrator)
    
    # 2. Test Curiosity Trigger
    logger.info("🧪 Testing Curiosity Drive Trigger...")
    curiosity_drive = Drive("Curiosity", 0.9, "Explore")
    await soul.satisfy_drive(curiosity_drive)
    
    # Verify curiosity.add_curiosity was called
    orchestrator.curiosity.add_curiosity.assert_called()
    logger.info("✅ Curiosity satisfied: add_curiosity was called.")
    
    # 3. Test Connection Trigger
    logger.info("🧪 Testing Connection Drive Trigger...")
    connection_drive = Drive("Connection", 0.9, "Connect")
    await soul.satisfy_drive(connection_drive)
    
    # Verify volition cooldown was reset (last_speak_time set to 0)
    assert orchestrator.volition.last_speak_time == 0
    logger.info("✅ Connection satisfied: Volition last_speak_time reset to 0.")
    
    # 4. Test Competence Trigger
    logger.info("🧪 Testing Competence Drive Trigger...")
    competence_drive = Drive("Competence", 0.9, "Repair")
    await soul.satisfy_drive(competence_drive)
    
    # Verify execute_tool("system_health", ...) was called
    orchestrator.execute_tool.assert_called_with("system_health", {})
    logger.info("✅ Competence satisfied: execute_tool('system_health') was called.")

    # 5. Test Dominant Drive Calculation
    logger.info("🧪 Testing Dominant Drive Calculation (Loneliness)...")
    soul.last_chat_time = 0 # Long time ago
    dominant = soul.get_dominant_drive()
    logger.info(f"Dominant drive when lonely: {dominant.name} (urgency={dominant.urgency:.2f})")
    assert dominant.name == "Connection"
    
    logger.info("🏁 Soul Autonomy Trigger Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_soul_triggers())


##
