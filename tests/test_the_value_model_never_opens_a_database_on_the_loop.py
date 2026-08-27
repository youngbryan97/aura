"""A singleton getter that does I/O is a stall waiting for the first caller.

Two things were wrong and they compounded. The first call to
``get_action_value_model`` built the model AND pulled its statistics out of
SQLite — on whatever thread arrived first, which is the event loop. And the
singleton lock was held across all of it, so every other caller queued behind
that I/O too.

LIVE, repeatedly, most recently 2026-08-27: "blocking lock
'core.reasoning.action_value.1' ... was held 278ms on the event loop thread
(limit 50ms) — the loop could not make progress for that window". The runtime
tainted itself over it every time, recorded a degradation, and went round again
on the next boot.

The async path already existed. Nothing had to be built — the first call just
had to stop taking the slow road.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import core.reasoning.action_value as action_value
from core.reasoning.action_value import ActionValueModel, get_action_value_model


@pytest.fixture(autouse=True)
def fresh_singleton(monkeypatch):
    """Each test gets an unbuilt model, and leaves the real one alone."""
    monkeypatch.setattr(action_value, "_model", None)
    monkeypatch.setattr(action_value, "_observers_installed", True)
    yield
    monkeypatch.setattr(action_value, "_model", None)


class SlowLedger:
    """A ledger that costs what a real one costs."""

    def __init__(self, seconds: float = 0.3) -> None:
        self.seconds = seconds
        self.reads = 0

    def measured_action_stats(self, by_state: bool = False):
        self.reads += 1
        time.sleep(self.seconds)
        return {}


# ── on the loop, it does not go near the ledger ──────────────────────────

def test_the_first_call_on_the_loop_does_no_reading(monkeypatch):
    ledger = SlowLedger()
    monkeypatch.setattr(
        "core.cognition.outcome_ledger.get_outcome_ledger", lambda: ledger, raising=False
    )

    async def on_the_loop():
        started = time.monotonic()
        model = get_action_value_model()
        return model, time.monotonic() - started

    model, took = asyncio.run(on_the_loop())
    assert isinstance(model, ActionValueModel)
    assert ledger.reads == 0
    assert took < 0.05


def test_and_leaves_it_stale_so_the_async_path_picks_it_up(monkeypatch):
    monkeypatch.setattr(
        "core.cognition.outcome_ledger.get_outcome_ledger",
        lambda: SlowLedger(0.0),
        raising=False,
    )

    async def on_the_loop():
        model = get_action_value_model()
        assert model.is_stale()
        return await model.refresh_if_stale()

    assert asyncio.run(on_the_loop()) is True


# ── off the loop, warming it costs the caller nothing it minds ───────────

def test_off_the_loop_it_is_warmed_at_once(monkeypatch):
    ledger = SlowLedger(0.0)
    monkeypatch.setattr(
        "core.cognition.outcome_ledger.get_outcome_ledger", lambda: ledger, raising=False
    )
    get_action_value_model()
    assert ledger.reads >= 1


# ── and the lock is never held across the reading ────────────────────────

def test_the_lock_is_free_while_the_ledger_is_being_read(monkeypatch):
    """Whoever asks second must not queue behind the first caller's I/O."""
    held: list[bool] = []

    class Watching(SlowLedger):
        def measured_action_stats(self, by_state: bool = False):
            held.append(action_value._model_lock.acquire(blocking=False))
            if held[-1]:
                action_value._model_lock.release()
            return super().measured_action_stats(by_state)

    monkeypatch.setattr(
        "core.cognition.outcome_ledger.get_outcome_ledger", lambda: Watching(0.0), raising=False
    )
    get_action_value_model()
    assert held and all(held), "the singleton lock was held while the ledger was read"


def test_the_same_model_comes_back_every_time(monkeypatch):
    monkeypatch.setattr(
        "core.cognition.outcome_ledger.get_outcome_ledger",
        lambda: SlowLedger(0.0),
        raising=False,
    )
    assert get_action_value_model() is get_action_value_model()
