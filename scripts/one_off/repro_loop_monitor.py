
import asyncio
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Repro")

# Add project root to sys.path
sys.path.append("/Users/bryan/Desktop/aura")

async def test_boot_sequence():
    from core.container import ServiceContainer
    from core.orchestrator import create_orchestrator
    
    logger.info("Starting mock boot...")
    orchestrator = create_orchestrator()
    
    # Simulate ResilientBoot.ignite() which calls _init_cognitive_architecture
    # logger.info("Initializing cognitive architecture...")
    # await orchestrator._init_cognitive_architecture()
    
    # Check if registered
    synth = ServiceContainer.get("qualia_synthesizer", default=None)
    logger.info(f"ServiceContainer['qualia_synthesizer'] after init: {synth}")
    
    if synth is None:
        logger.error("FAIL: QualiaSynthesizer not registered!")
    else:
        logger.info("SUCCESS: QualiaSynthesizer registered.")
        
    # Now start the LoopMonitor
    from core.consciousness.loop_monitor import get_loop_monitor
    monitor = get_loop_monitor(orchestrator)
    monitor.start()
    logger.info("LoopMonitor started.")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Force a check
    issues = await monitor._run_checks()
    logger.info(f"LoopMonitor issues: {issues}")
    
    has_synth_none = any(issue.get("check") == "qualia_synthesizer_registered" for issue in issues)
    if has_synth_none:
        logger.error("FAIL: LoopMonitor reports qualia_synthesizer is None!")
    else:
        logger.info("SUCCESS: LoopMonitor sees qualia_synthesizer.")

if __name__ == "__main__":
    asyncio.run(test_boot_sequence())
