"""Every skill did ONE act and returned, so nothing could watch.

The pieces all existed and none were connected. FluidExecutor.pursue closes a
perceive-decide-act loop with governance and verification. get_screen_text
reads the screen and reports where each run of text sat. perception_demand
keeps her eyes open at task cadence while she acts, instead of the 0.1Hz a
foreground generation imposes. hotkey presses keys.

What was missing was any way to ASK for the combination. Tasks that are
trivially describable — "wait for that build and tell me if it fails", "keep
pressing next until the form is done", "play this until you win" — had no path
through the system, and the failure always looked the same: she did one step
and stopped.

Nothing here is about any particular screen. A build log, a wizard, an
installer and a game are the same problem: read, decide from what was read,
act, read again.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.perception_demand import (
    perception_is_demanded,
    reset_perception_demand,
)
from core.skills import screen_pursuit as sp


@pytest.fixture(autouse=True)
def _clean():
    reset_perception_demand()
    yield
    reset_perception_demand()


@pytest.fixture
def screen(monkeypatch):
    """A screen that only changes when a key is pressed."""
    state = {"n": 0, "pressed": [], "saw_demand": False}

    async def read():
        state["saw_demand"] = state["saw_demand"] or perception_is_demanded()
        return {
            "ok": True,
            "text": "DONE" if state["n"] >= 4 else f"count {state['n']}",
            "layout": [],
            "error": "",
        }

    async def press(key):
        state["n"] += 1
        state["pressed"].append(key)
        return True

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    return state


async def _alternating(observation):
    text = observation["text"]
    n = int(text.split()[-1]) if text.startswith("count") else 0
    return {"key": "right" if n % 2 == 0 else "left", "because": f"count is {n}"}


def _run(**kw):
    defaults = {"goal": "reach DONE", "success_when": r"\bDONE\b", "narrate": False}
    return asyncio.run(sp.pursue_on_screen(**{**defaults, **kw}))


def test_she_keeps_going_until_the_screen_says_it_is_done(screen):
    result = _run(policy=_alternating, max_cycles=30)

    assert result["completed"] is True
    assert result["outcome"] == "goal_reached"
    assert screen["n"] >= 4


def test_each_move_is_decided_from_the_current_reading(screen):
    """Not from a plan: the policy sees the screen before choosing."""
    result = _run(policy=_alternating, max_cycles=30)

    assert [m["key"] for m in result["moves"]] == ["right", "left", "right", "left"]
    assert result["moves"][0]["because"] == "count is 0"


def test_perception_stays_open_for_the_whole_pursuit(screen):
    """The loop acts on what it sees; that is when sight was being throttled."""
    _run(policy=_alternating, max_cycles=30)

    assert screen["saw_demand"] is True
    assert not perception_is_demanded()


def test_an_unreachable_goal_is_bounded(screen):
    result = _run(success_when="IMPOSSIBLE_TOKEN", policy=_alternating, max_cycles=6)

    assert result["completed"] is False
    assert result["outcome"] == "out_of_cycles"
    assert result["cycles"] == 6


def test_no_policy_stalls_instead_of_spinning(screen):
    result = _run(success_when="NOPE", policy=None, max_cycles=50)

    assert result["outcome"] == "no_move_available"
    assert result["cycles"] <= 4


@pytest.mark.parametrize("forbidden", ["cmd+q", "ctrl+c", "delete", "f13", ""])
def test_a_policy_cannot_press_a_key_outside_the_allowed_set(screen, forbidden):
    """A loop that can press anything can press ⌘Q."""

    async def rogue(_observation):
        return {"key": forbidden, "because": "should never run"}

    result = _run(success_when="NOPE", policy=rogue, max_cycles=10)

    assert result["moves"] == []
    assert screen["pressed"] == []


def test_success_is_judged_on_what_was_read_not_on_what_was_done(screen):
    """An action that ran is not an action that worked."""
    assert sp.goal_reached({"text": "build FAILED"}, "FAILED") is True
    assert sp.goal_reached({"text": "still running"}, "FAILED") is False
    assert sp.goal_reached({"text": ""}, "FAILED") is False


def test_a_malformed_success_pattern_falls_back_to_plain_text():
    """An unbalanced bracket must not take the whole pursuit down."""
    assert sp.goal_reached({"text": "score [4096]"}, "[4096") is True


def test_a_wedged_capture_ends_the_cycle_rather_than_acting_blind(monkeypatch):
    async def hang():
        await asyncio.sleep(30)

    monkeypatch.setattr(sp, "read_screen", hang)
    monkeypatch.setattr(sp, "OBSERVE_TIMEOUT_S", 0.05)

    called: list[str] = []

    async def policy(_o):
        called.append("decided")
        return {"key": "right"}

    result = _run(policy=policy, max_cycles=3)

    assert called == [], "a blind loop must not keep pressing keys"
    assert result["completed"] is False


def test_a_policy_that_raises_does_not_kill_the_run(screen):
    async def broken(_observation):
        raise ValueError("policy exploded")

    result = _run(success_when="NOPE", policy=broken, max_cycles=10)

    assert result["outcome"] == "no_move_available"
