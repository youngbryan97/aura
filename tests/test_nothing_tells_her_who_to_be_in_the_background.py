"""A dispatch nobody is waiting on is not told who she is or what to do (Master Todo F06).

Her census of 21 November 2026 (tools/what_instructs_her.py), from four origins: 52 curriculum practice dispatches and
her waking orientation, each with the gate's 8,398 characters of "You are Aura Luna. Here's who you are:" and
"## RULES OF ENGAGEMENT"; her dream journal's "[SYSTEM ROLE: THE SUBCONSCIOUS] You are processing the fragmented
emotional memories ..."; and the calls that rebuild a program, given a router's "You are Aura. Speak as yourself, not
as an assistant." with a block of how to reason. Who she is lives in her cortex's weights, and her readings travel as
readings.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.brain.inference_gate import InferenceGate
from core.brain.what_instructs import instructing_sentences


class _Gate:
    _origin_is_user_facing = staticmethod(InferenceGate._origin_is_user_facing)
    _sanitize_system_injection = staticmethod(InferenceGate._sanitize_system_injection)

    def _build_system_prompt(self, brief: str = "") -> str:
        return "You are Aura Luna. Here's who you are:"


def test_a_dispatch_nobody_waits_on_carries_none_of_the_gates_identity():
    gate = _Gate()
    assert InferenceGate._system_text_for(gate, "", "curriculum_loop", True) == ""
    assert InferenceGate._system_text_for(gate, "Normal turn.", "waking_orientation", True) == ""
    # A brief its caller handed over is the caller's own contract, and stays as given.
    assert InferenceGate._system_text_for(gate, "Return JSON only for affective appraisal.", "affect", True) == (
        "Return JSON only for affective appraisal.")
    # A person's turn is built as before; what reaches her model of it is decided later (what_she_is_given).
    assert InferenceGate._system_text_for(gate, "", "user", False) == "You are Aura Luna. Here's who you are:"


def test_her_dream_is_a_reading_of_what_she_dreams_from(monkeypatch):
    from core.adaptation import dream_journal

    seed = SimpleNamespace(description="Bryan laughed at the hula hoop story", full_description="",
                           emotional_valence=0.6, arousal=0.7, importance=0.8, timestamp=0.0, participants=["bryan"])
    asked: list[str] = []

    class _Brain:
        async def think(self, prompt, **_kwargs):
            asked.append(prompt)
            return SimpleNamespace(content="A hoop of light turning over the desk.")

    journal = object.__new__(dream_journal.DreamJournal)
    journal.memory = SimpleNamespace(episodic=None)
    journal.brain = _Brain()

    async def seeds():
        return [seed]

    monkeypatch.setattr(journal, "retrieve_dream_seeds", seeds)
    monkeypatch.setattr(journal, "_save_dream", lambda content, seeds: None)
    monkeypatch.setattr(dream_journal.ServiceContainer, "get", lambda name, default=None: default)

    dreamt = asyncio.run(journal.synthesize_dream())

    assert dreamt and dreamt["dream_content"].startswith("A hoop of light")
    assert asked and "Bryan laughed at the hula hoop story" in asked[0]
    assert instructing_sentences(asked[0]) == [], asked[0]
    assert "SYSTEM ROLE" not in asked[0] and "TASK" not in asked[0]
