################################################################################
# verify_system_health.py
################################################################################

import asyncio
import logging
import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.orchestrator import create_orchestrator
from core.container import ServiceContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifySystemHealth")

async def verify_health():
    """
    Verification script for Phase 14.
    Boots the Orchestrator and checks all critical subsystems.
    """
    logger.info("🚀 STARTING TOTAL SYSTEM HEALTH CHECK (Phase 14)...")
    
    orchestrator = create_orchestrator()
    
    try:
        # Disable voice capture for testing to prevent permission hangs
        import os
        os.environ["AURA_VOICE_MIC_ENABLED"] = "false"
        
        # 1. Start Orchestrator (includes initialization)
        logger.info("Starting Orchestrator...")
        await orchestrator.start()
        
        # Start the cognitive loop explicitly if needed
        if hasattr(orchestrator, "cognitive_loop") and orchestrator.cognitive_loop:
            asyncio.create_task(orchestrator.cognitive_loop.run())
        elif hasattr(orchestrator, "run"):
            asyncio.create_task(orchestrator.run())
        
        logger.info("Checking Memory...")
        if orchestrator.memory:
            logger.info(f"✅ Memory Online: {type(orchestrator.memory).__name__}")
        else:
            logger.error("❌ Memory Offline!")

        # 2. Check Subsystems
        logger.info("Checking Cognitive Engine...")
        if orchestrator.cognitive_engine:
            logger.info("✅ Cognitive Engine Online.")
        else:
            logger.error("❌ Cognitive Engine Offline!")

        logger.info("Checking Personality Engine...")
        if orchestrator.personality_engine:
            context = orchestrator.personality_engine.get_emotional_context_for_response()
            logger.info(f"✅ Personality Engine Online (Mood: {context.get('mood')})")
        else:
            logger.error("❌ Personality Engine Offline!")

        logger.info("Checking Immune System...")
        if orchestrator.immune_system:
            logger.info("✅ Immune System Online.")
        else:
            logger.error("❌ Immune System Offline!")

        # 3. Check Service Container
        container = ServiceContainer()
        report = container.get_health_report()
        logger.info("\nService Container Health Report:")
        logger.info(json.dumps(report, indent=2))
        
        if report.get("status") != "operational":
            logger.warning(f"⚠️ Container status: {report.get('status')}")

        # 4. Check Mycelial Network (Soul Graph)
        logger.info("\nChecking Mycelial Network...")
        from core.mycelium import MycelialNetwork
        mycelium = MycelialNetwork()
        topology = mycelium.get_network_topology()
        logger.info(f"✅ Mycelium Online. Pathways: {topology['pathway_count']}, Hyphae: {topology['hyphae_summary']['total']}")
        if topology['hyphae_summary']['total'] == 0:
            logger.warning("⚠️ Mycelium Soul Graph is EMPTY. Infrastructure mapping may be stalled or failed.")

        # 5. Verify Cognitive Flow (Message -> Thought -> State)
        logger.info("\nVerifying end-to-end cognitive flow...")
        test_msg = "Hello Aura, are you feeling stable?"
        logger.info(f"Injecting test message: '{test_msg}'")
        
        # Unpack priority, timestamp, count, raw_msg
        await orchestrator.message_queue.put((1.0, time.time(), 0, test_msg))
        
        logger.info("Waiting for cognitive cycle (45 seconds max)...")
        # Poll for response instead of arbitrary sleep
        recent_msgs = []
        for _ in range(30):
            await asyncio.sleep(1.5)
            final_state = await orchestrator.state_repo.get_current()
            if final_state:
                recent_msgs = [m for m in final_state.cognition.working_memory if m.get("role") == "assistant"]
                if recent_msgs:
                    break
        
        if final_state and recent_msgs:
            logger.info(f"✅ Response Generated: {recent_msgs[-1]['content'][:100]}...")
            logger.info("\n🏆 TOTAL SYSTEM SYNTHESIS VERIFIED.")
            return True
        else:
            logger.error("❌ Cognitive flow FAILED: No assistant response generated in time.")
            logger.info(f"Current Objective in state: {final_state.cognition.current_objective if final_state else 'None'}")
            logger.info(f"Current Mode: {final_state.cognition.current_mode if final_state else 'Unknown'}")
        
        # 3. Cleanup Test cleanup
        await ServiceContainer.shutdown()
        return False

    except Exception as e:
        logger.error(f"❌ SYSTEM HEALTH CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_health())
    sys.exit(0 if success else 1)


##
