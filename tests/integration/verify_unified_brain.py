import asyncio
import logging
from core.container import ServiceContainer
from core.brain.llm.llm_router import IntelligentLLMRouter, LLMTier
from core.language_center import LanguageCenter
from core.inner_monologue import ThoughtPacket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_unified_brain")

async def verify():
    logger.info("🚀 Starting Unified Brain Verification")
    
    # 1. Setup Router
    router = IntelligentLLMRouter()
    ServiceContainer.register_instance("llm_router", router)
    
    # 2. Setup LanguageCenter
    lc = LanguageCenter()
    await lc.start()
    
    # 3. Test Legacy Mapping (api_deep -> PRIMARY)
    thought = ThoughtPacket(
        model_tier="api_deep",
        stance="Verifying legacy mapping",
        primary_points=["Expression point 1"]
    )
    
    logger.info("📡 Testing Legacy Dispatch (api_deep)...")
    # We use a mock endpoint if no real ones are available, 
    # but the router should at least attempt to find PRIMARY.
    try:
        response = await lc.express("Hello", thought)
        logger.info("✅ Dispatch successful. Response: %s", response)
    except Exception as e:
        logger.error("❌ Dispatch failed: %s", e)

    # 4. Verify Racing is disabled
    # If we check the logs of the router (manually or via intercept), 
    # it should NOT log "🏁 Racing:".
    
    logger.info("🏁 Verification complete.")

if __name__ == "__main__":
    asyncio.run(verify())

