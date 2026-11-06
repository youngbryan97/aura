"""What stopped her state log on the third day run (6 November 2026), and her journal that never wrote.

At 03:15:48 her terminal monitor's silent self-repair ran as a turn. The turn set the orchestrator's current origin to
"terminal_monitor" and never handed it back, because the restore asked whether the current thought task was the task
running the handler, and the handler had just replaced it with its own watchdog task. Every later thought that gave no
origin of its own took "terminal_monitor", read as background, and was refused for want of a user anchor; her journal,
the thought that wrote her state log on an unattended day, gave no origin. Before that, her journal had been refused at
its response step for the same want of a user anchor all day, silently.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from core.orchestrator.mixins.incoming_logic import turn_still_owns_context
from core.runtime import response_policy


@pytest.mark.asyncio
async def test_a_turn_whose_watchdog_task_is_current_hands_its_origin_back():
    async def handler():
        running = asyncio.current_task()
        watchdog = asyncio.ensure_future(asyncio.sleep(0))
        await watchdog
        # What the handler sees in its finally: the current thought task is its own watchdog task.
        return turn_still_owns_context(watchdog, running=running, own=watchdog), (watchdog is running)

    owns, same_task = await asyncio.create_task(handler())
    assert same_task is False  # the old test (current is running) was false here, so nothing was restored
    assert owns is True


@pytest.mark.asyncio
async def test_a_newer_turn_keeps_the_context_it_took():
    async def handler():
        running = asyncio.current_task()
        own = asyncio.ensure_future(asyncio.sleep(0))
        newer = asyncio.ensure_future(asyncio.sleep(0))
        await asyncio.gather(own, newer)
        return turn_still_owns_context(newer, running=running, own=own)

    assert await asyncio.create_task(handler()) is False


def test_no_current_task_hands_the_origin_back():
    assert turn_still_owns_context(None, running=object(), own=None) is True


def test_only_her_journal_is_waived_the_user_anchor(monkeypatch):
    seen = {}

    def fake_reason(orchestrator=None, *, profile=None, allow_no_user_anchor=False, **kwargs):
        seen["allow"] = allow_no_user_anchor
        return "" if allow_no_user_anchor else "no_user_anchor"

    monkeypatch.setattr(response_policy.background_policy, "background_activity_reason", fake_reason)
    objective = "You are writing Aura's internal journal. Reflect on these recent events."
    assert response_policy.background_response_suppression_reason(objective, origin="narrative_journal") == ""
    assert seen["allow"] is True
    assert response_policy.background_response_suppression_reason(objective, origin="system") == "no_user_anchor"
    assert seen["allow"] is False
    assert response_policy.background_response_suppression_reason(objective, origin="terminal_monitor") == "no_user_anchor"
    assert response_policy.background_response_suppression_reason(objective) == "no_user_anchor"


@pytest.mark.asyncio
async def test_her_journal_gives_its_own_origin_and_says_when_it_is_held(monkeypatch, caplog):
    from core.brain import narrative_memory

    episode = SimpleNamespace(timestamp=0.0, action="read a post", outcome="found five posts", episode_id="e1")

    class Episodic:
        async def recall_recent_async(self, limit=20):
            return [episode]

        async def delete_episodes_async(self, ids):
            raise AssertionError("episodes must be kept when no journal was written")

    calls = []

    class Brain:
        async def think(self, objective=None, context=None, mode=None, origin=None, **kwargs):
            calls.append(origin)
            return SimpleNamespace(content="", reasoning=["background_deferred"],
                                   metadata={"suppressed": True, "deferred_reason": "memory_pressure_88.0"})

    monkeypatch.setattr(narrative_memory, "get_episodic_memory", lambda: Episodic())
    engine = narrative_memory.NarrativeEngine(SimpleNamespace(cognitive_engine=Brain()))
    with caplog.at_level(logging.WARNING, logger="Cognition.Narrative"):
        await engine.consolidate_episodes()
        await engine.consolidate_episodes()
    assert calls == ["narrative_journal", "narrative_journal"]
    held = [r for r in caplog.records if "Journal held back" in r.getMessage()]
    assert len(held) == 1 and "memory_pressure_88.0" in held[0].getMessage()


def test_a_refused_thought_is_said_at_info_once_per_reason(caplog):
    from core.brain import cognitive_engine

    cognitive_engine._REFUSALS_SAID.clear()
    with caplog.at_level(logging.INFO, logger=cognitive_engine.logger.name):
        for _ in range(3):
            cognitive_engine._note_refused_thought("terminal_monitor", "no_user_anchor")
        cognitive_engine._note_refused_thought("terminal_monitor", "memory_pressure_90.0")
    said = [r.getMessage() for r in caplog.records if "refused a background thought" in r.getMessage()]
    assert len(said) == 2
    assert "origin=terminal_monitor (no_user_anchor)" in said[0]


def test_her_health_pulse_says_when_her_thoughts_stop_getting_through(monkeypatch):
    import time as _time

    from core.brain import cognitive_engine
    from core.ops import subsystem_audit

    cognitive_engine.note_thought_committed()
    assert subsystem_audit._cognition_stall_line(_time.time()) == ""
    for _ in range(5):
        cognitive_engine._note_refused_thought("terminal_monitor", "no_user_anchor")
    ledger = cognitive_engine.cognition_liveness()
    assert ledger["refused_since_commit"] == 5
    assert ledger["last_refusal"] == "terminal_monitor (no_user_anchor)"
    # Refused for a minute: not yet a stall.
    assert subsystem_audit._cognition_stall_line(_time.time() + 60) == ""
    # Refused for four hours with nothing committed: the pulse says so.
    line = subsystem_audit._cognition_stall_line(_time.time() + 4 * 3600)
    assert "5 thoughts refused over 240 min" in line and "terminal_monitor (no_user_anchor)" in line
    # One thought that commits clears it.
    cognitive_engine.note_thought_committed()
    assert subsystem_audit._cognition_stall_line(_time.time() + 4 * 3600) == ""
