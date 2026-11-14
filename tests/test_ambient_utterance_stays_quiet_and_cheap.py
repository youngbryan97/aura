"""An unprompted remark must never make the person's own question slower.

``consider_utterance`` runs off the ambient observation loop — every time she
notices the screen changed, on a machine whose resident 32B is the same model
answering whatever the person just typed. So the judgment has two costs and
both have to stay small: the cheap gate must reject almost everything before
any model call, and the model call it does make must be BACKGROUND.

It was neither marked background nor bounded. It passed ``mode=ThinkingMode.
FAST``, and the router's tiering reads ``is_background`` / ``priority`` /
``prefer_tier`` — never ``mode`` — so the flag was inert and an ambient
musing rode the foreground lane against a live turn.

There was no dedicated test for this module at all, which is why.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.perception.ambient_utterance import (
    _COMPOSE_TIMEOUT_S,
    _compose,
    _worth_noticing,
    consider_utterance,
)


class _Observation:
    def __init__(self, capture: str):
        self.capture = capture

    def for_reasoning(self) -> str:
        return f"[screen] {self.capture}"


class _Router:
    """Records exactly how the ambient lane asked for cognition."""

    def __init__(self, reply: str = "The failing assertion is on line 42."):
        self.kwargs: dict = {}
        self.prompt = None
        self.reply = reply

    async def think(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return self.reply


@pytest.fixture(autouse=True)
def _a_conversation_with_him(monkeypatch):
    """Her remark is her next turn in her conversation with him; a turn of his comes before it."""
    monkeypatch.setattr("core.together.remarks.said_since",
                        lambda _began: [(1.0, "user", "I'm going to run the tests."), (2.0, "assistant", "Go ahead.")])


@pytest.fixture
def router(monkeypatch):
    instance = _Router()
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: instance if name == "llm_router" else default
        ),
    )
    return instance


# ─────────────────────────────────────────── it does not fight the foreground


def test_the_ambient_judgment_is_marked_background(router):
    """The flag the router actually reads.

    ``mode=ThinkingMode.FAST`` went into **kwargs and was never consulted, so
    the call it was meant to de-prioritise ran at full foreground priority.
    """
    asyncio.run(_compose(_Observation("traceback (most recent call last)"), "traceback"))

    assert router.kwargs.get("is_background") is True
    assert router.kwargs.get("priority", 1.0) <= 0.3
    assert router.kwargs.get("prefer_tier") is not None
    assert "mode" not in router.kwargs, (
        "the router does not read `mode`; passing it looks like tiering and is not"
    )


def test_composing_is_bounded(router):
    """It runs inside a loop. An unbounded call is a wedged loop."""
    source = inspect.getsource(_compose)

    assert "wait_for" in source
    assert "_COMPOSE_TIMEOUT_S" in source
    assert 0 < _COMPOSE_TIMEOUT_S <= 60


def test_a_slow_composition_is_silence_not_a_late_remark(monkeypatch):
    """A remark about a screen the person already left is wrong, not late."""

    class _Slow:
        async def think(self, prompt, **kwargs):
            await asyncio.sleep(5)
            return "too late to matter"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: _Slow() if name == "llm_router" else default
        ),
    )
    monkeypatch.setattr(
        "core.perception.ambient_utterance._COMPOSE_TIMEOUT_S", 0.05
    )

    assert asyncio.run(_compose(_Observation("panic: nil map"), "panic:")) == ""


def test_no_cognition_means_no_fallback_line(monkeypatch):
    """Silence beats a canned "I noticed an error on your screen!".

    A template interrupts and carries no information, and the person cannot
    tell it from a real observation until they have already been interrupted.
    """
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: default),
    )

    assert asyncio.run(_compose(_Observation("build failed"), "build failed")) == ""


# ──────────────────────────────────── the cheap gate rejects almost everything


@pytest.mark.parametrize(
    "capture",
    [
        "Traceback (most recent call last):\n  File 'x.py'",
        "AssertionError: expected 3, got 5",
        "build failed after 12 seconds",
        "panic: runtime error: index out of range",
    ],
)
def test_a_visible_fault_is_worth_noticing(capture):
    assert _worth_noticing(capture) != ""


@pytest.mark.parametrize(
    "capture",
    [
        "def handler(): pass  # ordinary code with no fault",
        "Stack Overflow — how to fix error: undefined",
        "Read the documentation about error: codes here",
        "One error: appears once in ordinary prose",
    ],
)
def test_ordinary_screens_are_not_worth_interrupting_for(capture):
    assert _worth_noticing(capture) == ""


def test_the_cheap_gate_runs_before_any_model_call(router):
    """Most ticks must cost nothing. Otherwise the loop is unaffordable."""
    import core.perception.ambient_utterance as module

    calls: list[str] = []

    class _Memory:
        def latest(self, _kind):
            calls.append("latest")
            return _Observation("def handler(): pass  # nothing wrong here at all")

    module_memory = _Memory()
    original = module._latest_observation
    module._latest_observation = lambda: module_memory.latest(None)
    try:
        assert asyncio.run(consider_utterance(object())) == ""
    finally:
        module._latest_observation = original

    assert router.kwargs == {}, "an ordinary screen reached the model"


# ─────────────────────────────────────── her own words, with nothing added to her turn


def test_her_remark_is_her_turn_with_the_screen_as_a_reading_and_nothing_else(router):
    """It asked her brainstem, under "You are Aura. You are quiet by default ...", to answer "exactly: NOTHING" unless
    it could say something specific, and not to greet him: instruction on the path to her words (F06)."""
    from core.brain.llm.llm_tier import LLMTier

    said = asyncio.run(_compose(_Observation("Traceback (most recent call last):\n  File 'x.py', line 3"), "traceback"))
    assert said == "The failing assertion is on line 42."
    assert router.prompt == "" and router.kwargs["system_prompt"] == ""
    assert router.kwargs["prefer_tier"] is LLMTier.PRIMARY, "her words, from her cortex"
    messages = router.kwargs["messages"]
    assert [m["role"] for m in messages[:2]] == ["user", "assistant"]
    assert messages[-1]["content"].startswith("## WHAT IS ON THE SCREEN\nTraceback (most recent call last)")
    for instruction in ("NOTHING", "Do not", "If — and only if", "You are Aura"):
        assert instruction not in str(messages), instruction


def test_a_remark_that_opens_with_a_greeting_is_still_hers(monkeypatch):
    """A filter threw away remarks by their opening words ("Hey", "I see you"); her words are not judged by them."""
    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(
        lambda name, default=None: _Router("Hey, that traceback is from the missing config key on line 3.")
        if name == "llm_router" else default))
    said = asyncio.run(_compose(_Observation("Traceback (most recent call last):"), "traceback"))
    assert said.startswith("Hey, that traceback")
