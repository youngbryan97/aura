import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.kernel.aura_kernel import AuraKernel, KernelConfig
from core.state.state_repository import StateRepository
from core.container import ServiceContainer
from core.providers.cognitive_provider import register_cognitive_services

async def diagnostic():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Diagnostic")
    
    logger.info("Starting Final Hardening Diagnostic...")
    
    # 1. Register services (simulating real boot)
    container = ServiceContainer()
    register_cognitive_services(container)
    
    # 2. Setup dependencies
    config = KernelConfig()
    vault = StateRepository(db_path=":memory:", is_vault_owner=True)
    
    # 3. Boot Kernel
    kernel = AuraKernel(config=config, vault=vault)
    try:
        await kernel.boot()
        logger.info("Kernel boot successful.")
        
        # 4. Verify LLM Organ
        llm = kernel.organs.get("llm")
        if llm:
            instance = llm.get_instance()
            logger.info(f"LLM Organ Instance Class: {instance.__class__.__name__}")
            if "MockLLM" in instance.__class__.__name__:
                logger.error("❌ FAILED: LLM resolved to MockLLM")
            else:
                logger.info("✅ SUCCESS: LLM resolved correctly")
        else:
            logger.error("❌ FAILED: LLM organ stub not found")
            
        await kernel.stop()
    except Exception as e:
        logger.error(f"❌ DIAGNOSTIC FAILED: {e}", exc_info=True)
    finally:
        logger.info("Diagnostic complete.")

if __name__ == "__main__":
    asyncio.run(diagnostic())
