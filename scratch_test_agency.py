import asyncio
from core.orchestrator.main import Orchestrator
from core.container import ServiceContainer

async def run_test():
    orch = Orchestrator()
    await orch.start()
    
    # Simulate time passing to build up idle seconds
    orch._last_user_interaction_time = 0
    orch._last_thought_time = 0
    
    # Mock some state
    orch.liquid_state.current.curiosity = 1.0
    orch.liquid_state.current.energy = 1.0
    
    print("Forcing metabolic cycle...")
    res = await orch.metabolic_coordinator._process_metabolic_tasks(volition=3)
    print("Metabolic cycle finished. Result:", res)
    await orch.stop()

if __name__ == "__main__":
    asyncio.run(run_test())
