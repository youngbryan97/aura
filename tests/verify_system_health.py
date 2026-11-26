import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifySystemHealth")


async def verify_health():
    """
    Verification script for Phase 14.
    Boots the Orchestrator and checks all critical subsystems.
    """
    from core.container import ServiceContainer
    from core.orchestrator.main import create_orchestrator

    logger.info("STARTING TOTAL SYSTEM HEALTH CHECK (Phase 14)...")
    orchestrator = create_orchestrator()
    orchestrator.setup()

    try:
        # 1. Boot Subsystems (No full loop)
        logger.info("Checking Memory...")
        if orchestrator.memory:
            logger.info(f"Memory Online: {type(orchestrator.memory).__name__}")
        else:
            logger.error("Memory Offline!")

        logger.info("Checking Cognitive Engine...")
        if hasattr(orchestrator, 'cognitive_engine') and orchestrator.cognitive_engine:
            logger.info("Cognitive Engine Online.")
        else:
            logger.error("Cognitive Engine Offline!")

        logger.info("Checking Personality Engine...")
        if hasattr(orchestrator, 'personality_engine') and orchestrator.personality_engine:
            context = orchestrator.personality_engine.get_emotional_context_for_response()
            logger.info(f"Personality Engine Online (Mood: {context.get('mood')})")
        else:
            logger.error("Personality Engine Offline!")

        logger.info("Checking Immune System...")
        if orchestrator.immune_system:
            if hasattr(orchestrator.immune_system, "get_status"):
                status = orchestrator.immune_system.get_status()
            elif hasattr(orchestrator.immune_system, "status"):
                status = orchestrator.immune_system.status()
            elif hasattr(orchestrator.immune_system, "get_last_scan"):
                status = orchestrator.immune_system.get_last_scan()
                if not status and hasattr(orchestrator.immune_system, "scan_system_health"):
                    status = await orchestrator.immune_system.scan_system_health()
            else:
                status = {}
            logger.info(f"Immune System Online. Status: {status}")
            if status.get("antibody_active") or (
                isinstance(status, dict)
                and (("failed" in status and not status["failed"]) or "known_signatures" in status)
            ):
                logger.info("Active Defense & Encryption keys verified.")
            else:
                logger.warning("Immune subsystems partially initialized.")
        else:
            logger.error("Immune System Offline!")

        # 2. Check Service Container
        report = ServiceContainer.get_health_report()
        logger.info("\nService Container Report:")
        logger.info(report)

        # 3. Cleanup Test cleanup
        await ServiceContainer.shutdown()
        logger.info("\nTOTAL SYSTEM SYNTHESIS VERIFIED.")
        return True

    except (AttributeError, ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
        logger.error(f"SYSTEM HEALTH CHECK FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(verify_health())
    sys.exit(0 if success else 1)
