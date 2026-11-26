
import asyncio
import logging
import sys
import os
from types import SimpleNamespace

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.soul import Soul, Drive
from core.orchestrator import RobustOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SoulTest")


class CallRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


async def test_soul_triggers():
    logger.info("Starting Soul Autonomy Trigger Test...")
    
    orchestrator = RobustOrchestrator()
    orchestrator.boredom = 0.0
    
    curiosity_add = CallRecorder()
    orchestrator.curiosity = SimpleNamespace(add_curiosity=curiosity_add)
    orchestrator.volition = SimpleNamespace()
    orchestrator.volition.last_speak_time = 3600 # Assume we spoke an hour ago
    
    execute_tool = AsyncCallRecorder(result={"ok": True})
    orchestrator.execute_tool = execute_tool
    
    soul = Soul(orchestrator)
    
    # 2. Test Curiosity Trigger
    logger.info("Testing Curiosity Drive Trigger...")
    curiosity_drive = Drive("curiosity", 0.9, "Explore")
    await soul.satisfy_drive(curiosity_drive)
    
    assert curiosity_add.calls
    logger.info("Curiosity satisfied: add_curiosity was called.")
    
    # 3. Test Connection Trigger
    logger.info("Testing Connection Drive Trigger...")
    connection_drive = Drive("connection", 0.9, "Connect")
    await soul.satisfy_drive(connection_drive)
    
    # Verify volition cooldown was reset (last_speak_time set to 0)
    assert orchestrator.volition.last_speak_time == 0
    logger.info("Connection satisfied: Volition last_speak_time reset to 0.")
    
    # 4. Test Competence Trigger
    logger.info("Testing Competence Drive Trigger...")
    competence_drive = Drive("competence", 0.9, "Repair")
    await soul.satisfy_drive(competence_drive)
    
    assert execute_tool.calls
    assert execute_tool.calls[-1].args == ("system_health", {})
    logger.info("Competence satisfied: execute_tool('system_health') was called.")

    # 5. Test Dominant Drive Calculation
    logger.info("Testing Dominant Drive Calculation (Loneliness)...")
    soul.last_chat_time = 0 # Long time ago
    dominant = soul.get_dominant_drive()
    logger.info(f"Dominant drive when lonely: {dominant.name} (urgency={dominant.urgency:.2f})")
    assert dominant.name == "connection"
    
    logger.info("Soul Autonomy Trigger Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_soul_triggers())
