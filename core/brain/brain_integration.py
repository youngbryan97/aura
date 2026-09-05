"""core/brain/brain_integration.py — Intelligence pipeline integration
======================================================================================

Provides actual registration functions for the new intelligence pipeline.
"""

import logging

logger = logging.getLogger("Aura.BrainIntegration")


def setup_intelligence_layer(container) -> None:
    """Register all new intelligence layer services."""
    from core.adapters.api_adapter import get_api_adapter
    from core.cognition.cognitive_kernel import get_cognitive_kernel
    from core.introspection.inner_monologue import get_inner_monologue
    from core.brain.language_center import get_language_center
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
InnerMonologue.think()              ← local cognitive briefing
    │  input: CognitiveBrief
    │  optionally calls: APIAdapter (local compatibility facade)
    │  produces: ThoughtPacket (stance, points, tone, constraints)
    │
    ▼
LanguageCenter.express()            ← managed local generation
    │  input: ThoughtPacket → to_system_prompt() → full LLM briefing
    │  routes to: Cortex | local Solver | Brainstem/Reflex recovery
    │  LLM is told WHAT to say. Not asked to figure it out.
    │
    ▼
RESPONSE (natural language, cleaned)

The LLM is the mouth. CognitiveKernel is the brain.
"""


if __name__ == "__main__":
    print(DATA_FLOW)
    print("Use setup_intelligence_layer(container) to register services.")
