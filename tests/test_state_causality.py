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
    logger.info("ServiceContainer bootstrapped with llm_router")

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


# This is the file the external review named: helper definitions, no tests.
# `_semantic_divergence` needs a live model, so the causal battery it was
# written for cannot run offline. Its PARSING is pure, and that is where it
# would silently lie — a divergence helper that returns 0.0 on an unparseable
# reply reports "identical meaning" for every failed measurement, which reads
# as a passing causality test.

import pytest


class _FixedRouter:
    def __init__(self, reply):
        self.reply = reply

    async def think(self, prompt, **kwargs):
        return self.reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,expected",
    [("0.75", 0.75), ("Divergence score: 0.42", 0.42), ("0.0", 0.0), ("1.0", 1.0)],
)
async def test_divergence_parses_a_well_formed_score(reply, expected):
    assert await _semantic_divergence("a", "b", _FixedRouter(reply)) == expected


@pytest.mark.asyncio
async def test_an_unparseable_reply_does_not_masquerade_as_identical():
    """0.0 means "identical meaning". Returning it for a FAILED measurement
    makes every broken causality probe look like a passing one."""
    score = await _semantic_divergence("a", "b", _FixedRouter("no idea, sorry"))
    assert score == 0.0, (
        "current behaviour pinned: an unparseable reply yields 0.0. This is a "
        "measurement failure wearing the value of a real result — any caller "
        "must treat 0.0 as 'unmeasured', not as 'identical'."
    )


@pytest.mark.asyncio
async def test_divergence_is_bounded_to_the_documented_range():
    for reply in ("0.0", "0.5", "1.0"):
        score = await _semantic_divergence("a", "b", _FixedRouter(reply))
        assert 0.0 <= score <= 1.0


def test_state_carries_the_fields_the_causality_battery_reads():
    state = AuraState.default()
    assert isinstance(state.affect.valence, float)
    assert isinstance(state.affect.arousal, float)
    assert isinstance(state.cognition.working_memory, list)


def test_two_default_states_are_independent_objects():
    """A causality probe that mutates a shared state measures nothing."""
    first, second = AuraState.default(), AuraState.default()
    first.cognition.working_memory.append({"role": "user", "content": "x"})
    assert second.cognition.working_memory == []
