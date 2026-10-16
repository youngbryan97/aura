"""Her chat's corpus reading sees the conversation it continues.

The conversation-aware grounding (core/knowledge/talked_about.py) was wired
into the unitary phase, which her desktop chat never runs; the reading her
chat does get is taken by the inference gate from the request alone, so "how
many did he win?" could not know who "he" was.
"""

from __future__ import annotations

import asyncio

import pytest

from core.knowledge.corpus_grounding import (
    conversation_this_turn_continues,
    the_conversation_this_turn_continues,
)

TURNS = [("user", "Karl Malone was a great power forward."), ("aura", "He played for the Utah Jazz.")]


def test_a_turn_that_refers_back_is_read_only_with_a_conversation_in_hand():
    from core.brain.observable_registry import _matches_corpus

    assert not _matches_corpus("how many did he win?")
    with the_conversation_this_turn_continues(TURNS, speakers=("Aura Luna", "Aura", "Luna")):
        assert _matches_corpus("how many did he win?")
        assert conversation_this_turn_continues() == (tuple(TURNS), ("Aura Luna", "Aura", "Luna"))
    assert conversation_this_turn_continues() is None


def test_the_reading_is_given_the_conversation_and_who_speaks(monkeypatch):
    from core.brain import observable_registry
    from core.knowledge import corpus_grounding

    seen: dict = {}

    class Empty:
        grounded = False

    def grounding_for(prompt, **kwargs):
        seen.update(kwargs, prompt=prompt)
        return Empty()

    monkeypatch.setattr(corpus_grounding, "corpus_grounding_for", grounding_for)
    monkeypatch.setattr(observable_registry, "her_nearness", lambda: None)

    async def turn():
        from core.brain.observable_grounding import observable_blocks

        with the_conversation_this_turn_continues(TURNS, speakers=("Aura",)):
            return await observable_blocks("how many did he win?")

    asyncio.run(turn())
    assert seen["prompt"] == "how many did he win?"
    assert seen["conversation"] == TURNS
    assert seen["speakers"] == ("Aura",)


@pytest.mark.asyncio
async def test_her_chat_sets_the_conversation_while_her_model_drafts(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.phases import kept_procedures_in_a_turn

    held: list = []

    class Router:
        async def think(self, **_kwargs):
            held.append(conversation_this_turn_continues())
            return "He won two."

        def get_last_generation_metadata(self):
            return {"surface_control_receipt": {"generation_stop_reason": "eos", "surface_quality_gate_reasons": []}}

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    async def no_kept_answer(*_args, **_kwargs):
        return None

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())
    monkeypatch.setattr(kept_procedures_in_a_turn, "kept_answer_for", no_kept_answer)
    await CognitiveEngine()._direct_desktop_quick_reply(
        "how many did he win?", ThinkingMode.FAST, "user",
        {"desktop_quick_reply_contract": True, "visible_user_message": "how many did he win?", "max_tokens": 512},
        timeout_s=30.0,
    )
    assert held and held[-1] is not None
    _turns, speakers = held[-1]
    assert "Aura" in speakers
    assert conversation_this_turn_continues() is None
