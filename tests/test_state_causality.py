import asyncio
import pytest
import logging
from core.state.aura_state import AuraState, AffectVector
from core.brain.predictive_engine import PredictiveEngine
from core.brain.metacognitive_monitor import MetacognitiveMonitor
from core.container import ServiceContainer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.CausalityTest")

def setup_module():
    """Bootstrap the ServiceContainer for testing."""
    from core.brain.llm.llm_router import IntelligentLLMRouter
    from core.container import ServiceContainer, ServiceDescriptor, ServiceLifetime
    
    router = IntelligentLLMRouter()
    ServiceContainer.register_instance("llm_router", router)
    logger.info("✅ ServiceContainer bootstrapped with llm_router")

async def _semantic_divergence(response_a: str, response_b: str, router) -> float:
    """Measure semantic divergence between two responses using LLM evaluation."""
    prompt = f"""Rate how semantically different these two responses are.
0.0 = identical meaning, 1.0 = completely different meaning/perspective.
Respond with only a float.

Response A: {response_a[:300]}
Response B: {response_b[:300]}

Divergence score:"""
    
    result = await router.think(prompt, priority=0.5, is_background=True)
    try:
        import re
        match = re.search(r"(\d+\.\d+)", result)
        if match:
            return float(match.group(1))
        return float(result.strip())
    except ValueError:
        return 0.0
