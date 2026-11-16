"""An unprompted remark must never make the person's own question slower.

``consider_utterance`` runs off the ambient observation loop — every time she
notices the screen changed, on a machine whose resident 27B is the same model
answering whatever the person just typed. So the judgment has two costs and
both have to stay small: the cheap gate must reject almost everything before
any model call, and the model call it does make must be BACKGROUND.

It was neither marked background nor bounded. Then, marked background and asked
of her router, it never reached her cortex: the router sends background work to
its tertiary tier (LIVE 14 November 2026, eleven remarks with nothing formed, a
traceback on the screen). It now takes the generation lane's background lease,
as her goal pass does, and her cortex's own client.
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from core.brain.her_unprompted_turn import on_her_cortex
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


class _Cortex:
    """Records exactly how the ambient lane asked her cortex's client, and which lease it held."""

    def __init__(self, reply: str = "The failing assertion is on line 42.", delay_s: float = 0.0):
        self.kwargs: dict = {}
        self.prompt = None
        self.reply = reply
        self.delay_s = delay_s
        self.leases: list[dict] = []
        self.released: list[int] = []

    async def generate_text_async(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self.reply


@pytest.fixture(autouse=True)
def _a_conversation_with_him(monkeypatch):
    """Her remark is her next turn in her conversation with him; a turn of his comes before it."""
    # In the sitting going on now: an earlier sitting is not the conversation in front of her (16 November 2026).
    monkeypatch.setattr("core.together.remarks.said_since",
                        lambda _began: [(time.time() - 60.0, "user", "I'm going to run the tests."),
                                        (time.time() - 30.0, "assistant", "Go ahead.")])


def _her_cortex_is(monkeypatch, cortex, *, lease: int | None = 7, person_waiting: bool = False):
    class _Gate:
        _mlx_client = cortex

    async def acquire(**kwargs):
        if cortex is not None:
            cortex.leases.append(kwargs)
        return lease

    def release(lease_id):
        if cortex is not None:
            cortex.released.append(lease_id)

    monkeypatch.setattr("core.runtime.service_access.resolve_inference_gate",
                        lambda default=None: _Gate() if cortex is not None else default)
    monkeypatch.setattr("core.brain.llm_health_router.acquire_external_generation_gate_lease", acquire)
    monkeypatch.setattr("core.brain.llm_health_router.release_external_generation_gate_lease", release)
    monkeypatch.setattr("core.brain.llm_health_router.a_person_is_waiting_for_her_model", lambda: person_waiting)


@pytest.fixture
def router(monkeypatch):
    instance = _Cortex()
    _her_cortex_is(monkeypatch, instance)
    return instance


# ─────────────────────────────────────────── it does not fight the foreground


def test_the_ambient_judgment_is_background_work_under_the_lanes_lease(router):
    """A person's turn asks a background lease to yield; the generation itself is marked background too."""
    asyncio.run(_compose(_Observation("traceback (most recent call last)"), "traceback"))

    assert router.kwargs.get("is_background") is True
    assert router.kwargs.get("origin") == "ambient_remark"
    from core.brain.llm.chat_format import thinking_enabled_for_request

    assert thinking_enabled_for_request("Aura-Qwen3.8-27B", cognitive_mode=router.kwargs.get("cognitive_mode")) is True, (
        "her private channel opened by name, so the decoder bounds it and its working is never handed over as a remark")
    assert len(router.leases) == 1 and not router.leases[0].get("foreground"), "a background lease"
    assert router.released == [7], "the lease is given back"
    assert "mode" not in router.kwargs


def test_it_does_not_ask_for_her_model_while_a_person_is_waiting_for_it(monkeypatch):
    cortex = _Cortex()
    _her_cortex_is(monkeypatch, cortex, person_waiting=True)
    assert asyncio.run(_compose(_Observation("panic: nil map"), "panic:")) == ""
    assert cortex.leases == [] and cortex.prompt is None


def test_no_lease_means_no_generation_and_no_remark(monkeypatch):
    cortex = _Cortex()
    _her_cortex_is(monkeypatch, cortex, lease=None)
    assert asyncio.run(_compose(_Observation("panic: nil map"), "panic:")) == ""
    assert cortex.prompt is None and cortex.released == []


def test_composing_is_bounded(router):
    """It runs inside a loop. An unbounded call is a wedged loop."""
    from core.brain.her_unprompted_turn import _on_her_lane

    source = inspect.getsource(on_her_cortex)
    lane = inspect.getsource(_on_her_lane)

    assert "wait_for" in source and "least_s" in source
    assert "least_s" in lane and "time_a_prompt_needs" in lane
    assert 0 < _COMPOSE_TIMEOUT_S <= 60


def test_a_slow_composition_is_silence_not_a_late_remark(monkeypatch):
    """A remark about a screen the person already left is wrong, not late."""
    cortex = _Cortex("too late to matter", delay_s=5)
    _her_cortex_is(monkeypatch, cortex)
    monkeypatch.setattr("core.perception.ambient_utterance._COMPOSE_TIMEOUT_S", 0.05)
    monkeypatch.setattr("core.brain.llm.mlx_client.time_a_prompt_needs", lambda _chars, _tokens: 0.0)

    assert asyncio.run(_compose(_Observation("panic: nil map"), "panic:")) == ""
    assert cortex.released == [7], "the lease is given back when the time runs out"


def test_no_cognition_means_no_fallback_line(monkeypatch):
    """Silence beats a canned "I noticed an error on your screen!".

    A template interrupts and carries no information, and the person cannot
    tell it from a real observation until they have already been interrupted.
    """
    _her_cortex_is(monkeypatch, None)

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
    said = asyncio.run(_compose(_Observation("Traceback (most recent call last):\n  File 'x.py', line 3"), "traceback"))
    assert said == "The failing assertion is on line 42."
    assert router.prompt == "" and router.kwargs["system_prompt"] == ""
    messages = router.kwargs["messages"]
    assert [m["role"] for m in messages[:2]] == ["user", "assistant"]
    assert messages[-1]["content"].startswith("## WHAT IS ON THE SCREEN\nTraceback (most recent call last)")
    for instruction in ("NOTHING", "Do not", "If — and only if", "You are Aura"):
        assert instruction not in str(messages), instruction


def test_a_remark_that_opens_with_a_greeting_is_still_hers(monkeypatch):
    """A filter threw away remarks by their opening words ("Hey", "I see you"); her words are not judged by them."""
    _her_cortex_is(monkeypatch, _Cortex("Hey, that traceback is from the missing config key on line 3."))
    said = asyncio.run(_compose(_Observation("Traceback (most recent call last):"), "traceback"))
    assert said.startswith("Hey, that traceback")


def test_a_remark_her_allowance_cut_ends_at_her_last_finished_sentence_with_its_lines(monkeypatch):
    """LIVE 14 November 2026, 15:03: her remark ended "Both ran the same command:" and an opened code fence."""
    cut = ("Here's what I can see.\n\n**Behind it:** two Terminal windows. Both crashed on line 12 of load_settings.py."
           "\n\nBoth ran the same command:\n\n```")
    _her_cortex_is(monkeypatch, _Cortex(cut))
    said = asyncio.run(_compose(_Observation("Traceback (most recent call last):"), "traceback"))
    assert said == ("Here's what I can see.\n\n**Behind it:** two Terminal windows. Both crashed on line 12 of "
                    "load_settings.py.")


def test_a_whole_remark_is_served_whole_however_long(monkeypatch):
    whole = "The traceback is a JSONDecodeError: settings.json ends after its opening brace. " * 8
    _her_cortex_is(monkeypatch, _Cortex(whole.strip()))
    assert asyncio.run(_compose(_Observation("Traceback (most recent call last):"), "traceback")) == whole.strip()
