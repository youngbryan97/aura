"""Her operational self context goes in where the turn bears on it, part by part (Bryan, 11 November 2026).

Live: asked "Hey Aura, can you hear me okay?" in voice mode she answered with her memory level and thermal state, and
"What's two plus two?" read the whole block before "Four." The measure is the one her history reach uses.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.conversation import her_operational_self as her


def _verdict(relevant: bool, measured: bool = True):
    return SimpleNamespace(relevant=relevant, measured=measured)


def test_no_message_or_no_judgement_means_the_whole_block(monkeypatch):
    import core.cognition.evidence_relevance as relevance

    assert her.what_of_herself_bears("", {"state": "x"}) is None
    monkeypatch.setattr(relevance, "assess_evidence_alignments",
                        lambda request, passages: [_verdict(False, measured=False) for _ in passages])
    assert her.what_of_herself_bears("What's two plus two?", {"state": "x", "what_i_am": "y"}) is None


def test_the_parts_the_turn_bears_on_are_named(monkeypatch):
    import core.cognition.evidence_relevance as relevance

    monkeypatch.setattr(relevance, "assess_evidence_alignments",
                        lambda request, passages: [_verdict(i == 1) for i, _ in enumerate(passages)])
    assert her.what_of_herself_bears("Tell me about yourself", {"state": "a", "what_i_am": "b", "what_i_can_do_now": "c"}) == {"what_i_am"}


def _render(monkeypatch, bears):
    import core.consciousness.unified_self as unified
    import core.conversation.chat_preflight as preflight

    state = SimpleNamespace(name="Aura", current_state=SimpleNamespace(value="awake"), current_mood="calm",
                            sense_of_agency=0.5, sense_of_presence=0.5, continuity=0.9, identity_commitments=["be honest"])

    async def get_unified_self():
        return SimpleNamespace(get_state=lambda: state)

    monkeypatch.setattr(unified, "get_unified_self", get_unified_self)
    monkeypatch.setattr(preflight, "_live_internals_summary", lambda: ["Memory: 7.0GB in use by my runtime"])
    monkeypatch.setattr(preflight, "_reference_corpus_summary", lambda objective: [])
    monkeypatch.setattr(preflight, "_live_capability_affordance_block", lambda objective="": "- a_tool: does a thing")
    monkeypatch.setattr(her, "what_of_herself_bears", lambda objective, parts: bears)
    return asyncio.run(her.inject_operational_self_context("a turn"))


def test_a_turn_that_bears_on_nothing_of_her_gets_no_block(monkeypatch):
    assert _render(monkeypatch, set()) == ""


def test_her_internals_go_in_only_when_the_turn_bears_on_how_she_is(monkeypatch):
    block = _render(monkeypatch, {"what_i_am"})
    assert "What I am:" in block and "Memory: 7.0GB" not in block and "Mood:" not in block
    block = _render(monkeypatch, {"state"})
    assert "Memory: 7.0GB" in block and "Mood: calm" in block and "What I am:" not in block


def test_a_turn_that_asks_her_to_act_gets_her_live_capabilities(monkeypatch):
    block = _render(monkeypatch, {"what_i_can_do_now"})
    assert "Current actionable affordances:" in block and "a_tool" in block


def test_when_it_cannot_be_judged_the_whole_block_goes_in(monkeypatch):
    block = _render(monkeypatch, None)
    for part in ("Mood: calm", "What I am:", "Architectural capacities", "Current actionable affordances:",
                 "Current internals (live):", "My commitments in this conversation:"):
        assert part in block, part


def test_the_block_says_what_is_so_and_tells_her_nothing_to_do(monkeypatch):
    """Bryan, 11 November 2026: the binding rules were instructions in her turns; they are out."""
    block = _render(monkeypatch, None)
    assert "(binding)" not in block and "do not claim" not in block


def test_the_chat_preflight_still_offers_the_block_by_its_old_name():
    from core.conversation import chat_preflight

    assert chat_preflight.inject_operational_self_context is her.inject_operational_self_context
