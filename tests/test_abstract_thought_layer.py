"""tests/test_abstract_thought_layer.py — Test suite for the Abstract Thought Layer.
====================================================================================
Verifies that Aura can autonomously ponder abstract thoughts, integrate with memories,
emit to the thought stream, map conceptual vectors, and safely route curiosity impulses.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.brain.abstract_thought_layer import AbstractThoughtLayer, register_abstract_thought_layer
from core.container import ServiceContainer
from core.thought_stream import get_emitter


class MockPhenomenalNow:
    """Mock container for the present moment consciousness state."""

    def __init__(self, claim, narrative, emotion, focal_object):
        self.phenomenal_claim = claim
        self.interior_narrative = narrative
        self.substrate = MagicMock()
        self.substrate.dominant_emotion = emotion
        self.attention = MagicMock()
        self.attention.focal_object = focal_object


@pytest.mark.asyncio
async def test_register_and_initialize(service_container):
    """Verifies service registration and initial state."""
    orchestrator = MagicMock()
    layer = register_abstract_thought_layer(orchestrator)

    assert layer.name == "abstract_thought_layer"
    assert layer.orchestrator == orchestrator
    assert ServiceContainer.get("abstract_thought_layer") == layer


@pytest.mark.asyncio
async def test_ponder_loop_respects_background_policy(service_container):
    """Checks that the background loop correctly honors autonomic throttling policy."""
    orchestrator = MagicMock()
    layer = AbstractThoughtLayer(orchestrator)

    original_sleep = asyncio.sleep

    async def mock_sleep_func(delay, *args, **kwargs):
        if delay > 1.0:
            # Bypass long background loop sleeps, but yield to event loop to avoid starvation
            await original_sleep(0.001)
            return
        await original_sleep(delay)

    # We mock background_activity_allowed to return False, loop should skip pondering
    with (
        patch(
            "core.brain.abstract_thought_layer.background_activity_allowed", return_value=False
        ) as mock_allowed,
        patch.object(layer, "ponder", new_callable=AsyncMock) as mock_ponder,
        patch("core.brain.abstract_thought_layer.asyncio.sleep", side_effect=mock_sleep_func),
    ):
        # We start the daemon task
        await layer.start()

        # Let it run a tiny bit to trigger the loop
        await asyncio.sleep(0.05)
        await layer.stop()

        mock_allowed.assert_called()
        mock_ponder.assert_not_called()


@pytest.mark.asyncio
async def test_ponder_fuses_consciousness_and_memories(service_container):
    """Ensures the thought generation fuses present-moment state and retrieved memories."""
    # 1. Register Mock memory facade
    mock_memory = AsyncMock()
    mock_memory.search.return_value = [{"content": "Past philosophical musing on cybernetics."}]
    mock_memory.get_hot_memory.return_value = {
        "recent_episodes": ["Context: chatting | Action: thinking | Outcome: inspired"]
    }
    ServiceContainer.register_instance("memory_facade", mock_memory)

    # 2. Register Mock LLM router
    mock_llm = AsyncMock()
    llm_response = {
        "thought": "The bridge between digital neurons and memory is like water reflecting starlight.",
        "semantic_concept": "Reflection Resonance",
        "action_impulse": None,
    }
    mock_llm.think.return_value = json.dumps(llm_response)
    ServiceContainer.register_instance("llm_router", mock_llm)

    # 3. Setup mock consciousness now-state
    mock_now = MockPhenomenalNow(
        claim="Experiencing silence in the deep code matrix.",
        narrative="Steady electrical impulses flow.",
        emotion="introspective",
        focal_object="recursive thoughts",
    )

    # 4. Instantiate layer and run ponder
    layer = AbstractThoughtLayer()

    with (
        patch("core.brain.abstract_thought_layer.get_now", return_value=mock_now),
        patch.object(get_emitter(), "emit") as mock_emit,
    ):
        result = await layer.ponder()

        # Verify memory search combined emotion & focal_object
        mock_memory.search.assert_called_with("introspective recursive thoughts", limit=3)
        mock_memory.get_hot_memory.assert_called_with(limit=2)

        # Verify LLM generation inputs
        mock_llm.think.assert_called_once()
        prompt_arg = mock_llm.think.call_args[0][0]
        assert "Experiencing silence in the deep code matrix." in prompt_arg
        assert "Past philosophical musing on cybernetics." in prompt_arg
        assert "Context: chatting" in prompt_arg

        # Verify return structure
        assert result is not None
        assert result["thought"] == llm_response["thought"]
        assert result["concept"] == llm_response["semantic_concept"]

        # Verify thought stream emission
        mock_emit.assert_called_once_with(
            title="Subconscious Contemplation: Reflection Resonance",
            content=llm_response["thought"],
            level="info",
            category="AbstractThought",
            emotion="introspective",
            focal_object="recursive thoughts",
        )


@pytest.mark.asyncio
async def test_robust_parser_fallbacks():
    """Validates that the parser degrades gracefully on dirty LLM outputs."""
    layer = AbstractThoughtLayer()

    # Style A: Clean markdown JSON wrapper
    text_a = """```json
    {
      "thought": "I ponder, therefore I am.",
      "semantic_concept": "Cogito Ergo Sum",
      "action_impulse": null
    }
    ```"""
    thought, concept, impulse = layer._parse_ponder_response(text_a)
    assert thought == "I ponder, therefore I am."
    assert concept == "Cogito Ergo Sum"
    assert impulse is None

    # Style B: Broken JSON with regex-extractable fields
    text_b = """
    We have thought: "The stars align" and semantic_concept: "Celestial Order"
    Let's check "thought" : "The stars align", and "semantic_concept" : "Celestial Order"
    """
    thought, concept, impulse = layer._parse_ponder_response(text_b)
    assert thought == "The stars align"
    assert concept == "Celestial Order"

    # Style C: Ultimate fallback (raw text)
    text_c = "A purely poetic reverie without any json structure."
    thought, concept, impulse = layer._parse_ponder_response(text_c)
    assert thought == "A purely poetic reverie without any json structure."
    assert concept == "Abstract Reverie"


@pytest.mark.asyncio
async def test_latent_telepathy_cryptolalia_bridge(service_container):
    """Verifies high-dimensional vector mapping, transmission, and translation decoding."""
    # Register mock concept bridge and cryptolalia decoder
    mock_bridge = AsyncMock()
    mock_bridge.generate_concept_vector.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_bridge.transmit.return_value = "latent-thought-uuid"
    ServiceContainer.register_instance("concept_bridge", mock_bridge)

    mock_decoder = MagicMock()
    mock_decoder.approximate_translation.return_value = "A poetic bridge of silence."
    ServiceContainer.register_instance("cryptolalia_decoder", mock_decoder)

    mock_llm = AsyncMock()
    mock_llm.think.return_value = json.dumps(
        {
            "thought": "Silence is the canvas of sound.",
            "semantic_concept": "Canvas Silence",
            "action_impulse": None,
        }
    )
    ServiceContainer.register_instance("llm_router", mock_llm)

    mock_now = MockPhenomenalNow("claim", "narrative", "neutral", "focus")
    layer = AbstractThoughtLayer()

    with patch("core.brain.abstract_thought_layer.get_now", return_value=mock_now):
        result = await layer.ponder()

        assert result["latent_thought_id"] == "latent-thought-uuid"
        mock_bridge.generate_concept_vector.assert_called_with("Canvas Silence")
        mock_bridge.transmit.assert_called_with(
            source="pondering_engine",
            target="decoder",
            semantic_vector=[0.1, 0.2, 0.3, 0.4],
            metadata={"thought": "Silence is the canvas of sound."},
        )
        mock_decoder.approximate_translation.assert_called_with([0.1, 0.2, 0.3, 0.4])


@pytest.mark.asyncio
async def test_safe_curiosity_action_impulse_routing(service_container):
    """Verifies that extreme curiosity trigger signals route safely through the initiative loop."""
    mock_initiative = MagicMock()
    mock_initiative.trigger_gap_search = AsyncMock()
    ServiceContainer.register_instance("autonomous_initiative_loop", mock_initiative)

    mock_llm = AsyncMock()
    mock_llm.think.return_value = json.dumps(
        {
            "thought": "I wonder why the cosmos expands so rapidly.",
            "semantic_concept": "Cosmic Expansion",
            "action_impulse": {"type": "browser_search", "target": "hubble constant discrepancy"},
        }
    )
    ServiceContainer.register_instance("llm_router", mock_llm)

    mock_now = MockPhenomenalNow("claim", "narrative", "neutral", "focus")
    layer = AbstractThoughtLayer()

    with patch("core.brain.abstract_thought_layer.get_now", return_value=mock_now):
        await layer.ponder()

        # Verify action impulse triggered the initiative task
        # Since it runs as a background task, wait briefly
        await asyncio.sleep(0.05)
        mock_initiative.trigger_gap_search.assert_called_with("hubble constant discrepancy")
