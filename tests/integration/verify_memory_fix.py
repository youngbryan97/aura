################################################################################


import asyncio
import logging
import sys
import os

# Add the project root to sys.path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.container import ServiceContainer
from core.service_registration import register_all_services

async def verify():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VerifyFix")
    
    logger.info("Starting verification...")
    
    # 1. Register all services
    register_all_services()
    
    # 2. Get Orchestrator
    orchestrator = ServiceContainer.get("orchestrator")
    logger.info(f"Orchestrator retrieved: {type(orchestrator)}")
    
    # 3. Check memory property
    memory = orchestrator.memory
    logger.info(f"Orchestrator memory retrieved: {type(memory)}")
    
    # Check for retrieve_unified_context
    if hasattr(memory, 'retrieve_unified_context'):
        logger.info("SUCCESS: memory has 'retrieve_unified_context'")
    else:
        logger.error("FAILURE: memory missing 'retrieve_unified_context'")
        return False

    # Check for search_memories (compatibility)
    if hasattr(memory, 'search_memories'):
        logger.info("SUCCESS: memory has 'search_memories'")
    else:
        logger.error("FAILURE: memory missing 'search_memories'")
        return False

    # 4. Attempt call
    try:
        # Mocking sub-memories for a safe call if needed, 
        # but the facade handles None sub-memories gracefully.
        result = await memory.retrieve_unified_context("test query")
        logger.info(f"Call to retrieve_unified_context succeeded. Result length: {len(result)}")
    except Exception as e:
        logger.error(f"FAILURE: call to retrieve_unified_context failed: {e}")
        return False
        
    logger.info("Verification PASSED!")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify())
    sys.exit(0 if success else 1)


##
