import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.kernel.aura_kernel import AuraKernel, KernelConfig
from core.state.state_repository import StateRepository
from core.container import ServiceContainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Genesis.Trigger")

from core.utils.singleton import acquire_instance_lock

async def trigger_genesis():
    # Enforce instance lock for genesis cycle
    acquire_instance_lock(lock_name="genesis")
    
    logger.info("🧬 [GENESIS] Initializing Kernel for ASI Cycle 0...")
    
    # 1. Setup Mock Services
    vault = StateRepository(is_vault_owner=True)
    config = KernelConfig()
    kernel = AuraKernel(config, vault)
    
    # Register the kernel itself in the container if needed
    ServiceContainer.register_instance("aura_kernel", kernel)
    
    # Mock some services needed for SME
    from core.brain.cognitive_engine import CognitiveEngine
    ce = CognitiveEngine()
    ServiceContainer.register_instance("cognitive_engine", ce)
    
    # Mock stability guardian to avoid errors
    from unittest.mock import AsyncMock
    ServiceContainer.register_instance("stability_guardian", AsyncMock())
    
    # 2. Boot the Kernel
    logger.info("🧬 [GENESIS] Booting Kernel...")
    # kernel.boot() calls _setup_phases() and _load_initial_state()
    # We hook into _initialize_organs to ensure we use stubs that don't fail
    await kernel.boot()
    
    # 3. Inject the Genesis Condition
    logger.info("🧬 [GENESIS] Injection: evolution_score = 1.0, curiosity = 0.99")
    kernel.state.identity.evolution_score = 1.0
    kernel.state.affect.curiosity = 0.99
    
    # 4. Force a research initiative
    logger.info("🧬 [GENESIS] Injecting Autotelic Research Initiative...")
    topic = kernel.state.motivation.latent_interests[0]
    kernel.state.cognition.pending_initiatives.append({
        "drive": "curiosity",
        "goal": f"Researching {topic}",
        "urgency": 1.0
    })
    
    # 5. Trigger the Tick
    logger.info("🧬 [GENESIS] Initiating Unitary Tick...")
    objective = "Autonomous Evolution: Self-optimize Reasoning Logic"
    
    try:
        # This should now hit TrueEvolutionPhase.execute()
        await kernel.tick(objective)
    except Exception as e:
        logger.error(f"❌ [GENESIS] Tick Failed: {e}", exc_info=True)
    
    logger.info("🧬 [GENESIS] Tick Complete. Check logs for SelfModification.Engine output.")

if __name__ == "__main__":
    asyncio.run(trigger_genesis())
