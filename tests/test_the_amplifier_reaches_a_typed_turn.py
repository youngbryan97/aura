"""The verifier-backed amplifier runs on the chat turns people type.

It lived only in the phase pipeline, and desktop chat answers without a phase,
so it stood down as "not a user-facing turn" about 300 times from 4 to 17
October. These check the chat entry: it adopts an answer only under the same
authority the phase requires, keeps her draft otherwise, and spends only the
time left on the turn.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.brain import reasoning_amplifier_v2 as amp
from core.brain.cognitive_engine_quick_reply import _AnswersTheDesktopDirectly

_amplify = _AnswersTheDesktopDirectly._direct_desktop_quick_reply_amplify
QUESTION = "What is 7919 multiplied by 104729?"


class _Router:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return "829348951"


def _result(authority: str, answer: str):
    receipt = SimpleNamespace(
        to_dict=lambda: {"promotion_authority": authority, "mode": "deep", "num_candidates": 3}
    )
    return SimpleNamespace(receipt=receipt, answer=answer, source_answer=answer)


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setattr(amp, "search_allowance", lambda *a, **k: (20.0, 40.0))
    monkeypatch.setattr(amp, "is_amplifiable", lambda q: "math")


def _context(seconds_left: float) -> dict:
    return {"cognitive_cycle_deadline_monotonic": time.monotonic() + seconds_left}


def test_a_checked_answer_replaces_the_draft(priced, monkeypatch):
    seen: dict = {}

    async def _amplify_turn(objective, generate, **kwargs):
        seen.update(kwargs)
        await generate("work it out", 0.5)
        return _result("checked_verifier", "829,348,951")

    monkeypatch.setattr(amp, "amplify_turn", _amplify_turn)
    router = _Router()
    kwargs = {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 512}
    out = asyncio.run(_amplify(None, QUESTION, "829348950", router, kwargs, _context(300.0)))

    assert out == "829,348,951"
    assert seen["extra_context"]["seed_candidates"] == ["829348950"]
    assert seen["time_budget_s"] <= 300.0
    # A candidate goes through her own turn's lane with the amplifier's prompt.
    assert router.calls[0]["messages"][-1]["content"] == "work it out"
    assert kwargs["messages"][-1]["content"] == QUESTION


def test_an_unchecked_answer_leaves_her_draft(priced, monkeypatch):
    async def _amplify_turn(objective, generate, **kwargs):
        return _result("none", "something else")

    monkeypatch.setattr(amp, "amplify_turn", _amplify_turn)
    out = asyncio.run(
        _amplify(None, QUESTION, "829348950", _Router(), {"messages": []}, _context(300.0))
    )
    assert out is None


def test_no_time_left_means_no_search(priced, monkeypatch):
    async def _amplify_turn(objective, generate, **kwargs):
        raise AssertionError("searched with no time left")

    monkeypatch.setattr(amp, "amplify_turn", _amplify_turn)
    out = asyncio.run(
        _amplify(None, QUESTION, "829348950", _Router(), {"messages": []}, _context(10.0))
    )
    assert out is None


def test_ordinary_chat_is_not_searched(monkeypatch):
    monkeypatch.setattr(amp, "is_amplifiable", lambda q: None)

    async def _amplify_turn(objective, generate, **kwargs):
        raise AssertionError("searched an ordinary turn")

    monkeypatch.setattr(amp, "amplify_turn", _amplify_turn)
    out = asyncio.run(
        _amplify(None, "how was your day?", "fine", _Router(), {"messages": []}, _context(300.0))
    )
    assert out is None
