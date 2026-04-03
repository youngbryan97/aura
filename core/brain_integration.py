"""core/brain_integration.py — Intelligence pipeline integration
======================================================================================

Provides actual registration functions for the new intelligence pipeline.
"""

import logging

logger = logging.getLogger("Aura.BrainIntegration")


def setup_intelligence_layer(container) -> None:
    """Register all new intelligence layer services."""
    from core.api_adapter import get_api_adapter
    from core.cognitive_kernel import get_cognitive_kernel
    from core.inner_monologue import get_inner_monologue
    from core.language_center import get_language_center
    from core.memory_synthesizer import get_memory_synthesizer

    container.register_factory("api_adapter", get_api_adapter)
    container.register_factory("cognitive_kernel", get_cognitive_kernel)
    container.register_factory("inner_monologue", get_inner_monologue)
    container.register_factory("language_center", get_language_center)
    container.register_factory("memory_synthesizer", get_memory_synthesizer)
    logger.info("✅ Intelligence layer services registered.")


# ══════════════════════════════════════════════════════════════════
# QUICK REFERENCE: Data flow
# ══════════════════════════════════════════════════════════════════

DATA_FLOW = """
USER INPUT
    │
    ▼
CognitiveKernel.evaluate()          ← pure Python, ~2ms
    │  reads: BeliefRevisionEngine, MemorySynthesizer worldview
    │  produces: CognitiveBrief (domain, strategy, beliefs, framing)
    │
    ▼
InnerMonologue.think()              ← ~5ms (local) or ~500ms (API deepening)
    │  input: CognitiveBrief
    │  optionally calls: APIAdapter (api_fast) with structured prompt
    │  produces: ThoughtPacket (stance, points, tone, constraints)
    │
    ▼
LanguageCenter.express()            ← ~800ms (local) or ~1200ms (API)
    │  input: ThoughtPacket → to_system_prompt() → full LLM briefing
    │  routes to: local MLX | Claude Haiku | Claude Sonnet
    │  LLM is told WHAT to say. Not asked to figure it out.
    │
    ▼
RESPONSE (natural language, cleaned)

The LLM is the mouth. CognitiveKernel is the brain.
"""


if __name__ == "__main__":
    print(DATA_FLOW)
    print("Use setup_intelligence_layer(container) to register services.")
