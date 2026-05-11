import asyncio
from aura_main import bootstrap_aura
from core.orchestrator import create_orchestrator
from core.container import ServiceContainer

async def main():
    orchestrator = create_orchestrator()
    await bootstrap_aura(orchestrator)
    print("llm_router:", ServiceContainer.get("llm_router", default=None))
    print("brain:", ServiceContainer.get("brain", default=None))

asyncio.run(main())
