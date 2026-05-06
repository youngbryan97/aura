import asyncio
import logging
from core.orchestrator.main import create_orchestrator
from core.container import ServiceContainer

logging.basicConfig(level=logging.DEBUG)

async def test_boot():
    print("Initializing Orchestrator via Factory...")
    orchestrator = create_orchestrator()
    print("Running boot sequence...")
    await orchestrator._async_init_subsystems()
    print("Boot sequence complete.")
    
    # Verify key services
    experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
    print(f"Experiencer initialized: {experiencer is not None}")
    
    learner = ServiceContainer.get("continuous_learner", default=None)
    print(f"Learner initialized: {learner is not None}")
    
    phi_core = ServiceContainer.get("phi_core", default=None)
    print(f"PhiCore initialized: {phi_core is not None}")

if __name__ == "__main__":
    asyncio.run(test_boot())
