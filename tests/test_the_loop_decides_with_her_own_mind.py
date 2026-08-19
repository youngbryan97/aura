"""The judgement in a goal loop used to live outside her.

Both of Aura's pursuit loops took the decision as an injected callable. The
screen loop asked a ``policy`` lambda what to press; the plan loop asked a
``replan`` lambda what to do when it stalled. Everything around those hooks —
perception, focus, overlays, page identity, verification — was hers, and the
one step that chooses was a parameter.

These tests hold the wiring that closed it: with nothing injected the loop
reasons, predicts, checks the prediction against the next reading, and carries
what broke into the next decision.
"""
from __future__ import annotations

import pytest

from core.skills import screen_pursuit as sp


class _Store:
    """A spine and a graph that record nothing anywhere real."""

    def __init__(self):
        self.episodes = []

    def record(self, episode):
        self.episodes.append(episode)
        return f"ep_{len(self.episodes)}"

    def resolve(self, episode_id, outcome):
        pass

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        pass


@pytest.fixture
def screen(monkeypatch):
    """A screen that changes only when a move is one that works."""
    state = {"pressed": [], "text": "board 2", "works": {"up"}, "spoken": []}

    async def read(app_name=""):
        return {"ok": True, "text": state["text"], "layout": [], "bounds": []}

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        if key in state["works"]:
            state["text"] = f"board {len(state['pressed']) + 2}"
        return True

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://example.test/", "title": "board", "error": ""}

    async def narrate(line, because=""):
        state["spoken"].append(f"{line} — {because}" if because else line)

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)
    monkeypatch.setattr(sp, "_narrate", narrate)
    return state


def _thinks(*replies):
    """A mind that answers in sequence, and records what it was asked."""
    queue = list(replies)
    asked: list[list[str]] = []

    async def think(objective, evidence):
        asked.append(list(evidence))
        return queue.pop(0) if queue else queue_last(replies)

    def queue_last(all_replies):
        return all_replies[-1] if all_replies else ""

    think.asked = asked
    return think


@pytest.mark.asyncio
async def test_with_nothing_injected_the_loop_reasons_for_itself(screen):
    think = _thinks("up")
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="board 4",
        think=think,
        max_cycles=6,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert screen["pressed"], "she never moved"
    assert think.asked, "she never thought"
    assert result["moves"][0]["key"] == "up"


@pytest.mark.asyncio
async def test_every_move_carries_a_prediction_that_is_then_graded(screen):
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up"),
        max_cycles=4,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert result["attempts"], "no prediction was ever checked"
    first = result["attempts"][0]
    assert first["option"] == "up"
    assert "different after up" in first["expected"]
    assert first["held"] is True


@pytest.mark.asyncio
async def test_a_move_that_changes_nothing_is_recorded_as_not_having_worked(screen):
    screen["works"] = set()  # nothing this loop presses moves the screen
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("down"),
        max_cycles=4,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    graded = result["attempts"]
    assert graded and all(not a["held"] for a in graded)
    assert "nothing changed" in graded[0]["why"]


@pytest.mark.asyncio
async def test_what_broke_is_carried_into_the_next_decision(screen):
    screen["works"] = set()
    think = _thinks("down", "down", "down", "down")
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=think,
        max_cycles=3,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert len(think.asked) >= 2
    later = think.asked[-1]
    assert any("nothing changed" in line for line in later), later


@pytest.mark.asyncio
async def test_language_being_gone_does_not_stop_her_from_playing(screen):
    """The resident model is her language organ, not her decision organ.

    core/cognition/pre_linguistic.py holds this as an invariant: actions can
    be dispatched even when the LLM is unavailable. Measured live, a pursuit
    spent every cycle inside a forty-second model reload and ended having
    made no move.
    """

    async def unreachable(objective, evidence):
        raise RuntimeError("worker_not_alive")

    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=unreachable,
        max_cycles=4,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert screen["pressed"], "she stopped playing because she could not talk"
    assert result["moves"], "no move was made without language"
    assert result["outcome"] != "cannot_decide"


@pytest.mark.asyncio
async def test_an_injected_policy_still_wins(screen):
    """A caller with its own judgement keeps it, and no thinking happens."""
    think = _thinks("up")

    async def policy(_observation):
        return {"key": "left", "because": "the caller decided"}

    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        policy=policy,
        think=think,
        max_cycles=2,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert result["moves"][0]["key"] == "left"
    assert not think.asked


@pytest.mark.asyncio
async def test_the_moves_offered_are_the_ones_the_caller_named(screen):
    think = _thinks("tab")
    await sp.pursue_on_screen(
        goal="move through the fields",
        success_when="never happens",
        think=think,
        move_keys=("tab", "return"),
        max_cycles=2,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    offered = [line for line in think.asked[0] if line.startswith("Available move")]
    assert any("tab" in line for line in offered)
    assert not any("up" in line for line in offered)


@pytest.mark.asyncio
async def test_a_mind_that_comes_back_is_used_again(screen):
    """Language returning means she narrates in her own words again."""
    calls = {"n": 0}

    async def slow_to_wake(objective, evidence):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("worker_not_alive")
        return "up"

    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=slow_to_wake,
        max_cycles=2,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert screen["pressed"], "she gave up on a mind that was coming back"
    assert calls["n"] >= 2, "she stopped asking once language failed once"




@pytest.mark.asyncio
async def test_the_loop_itself_says_nothing(screen):
    """Narration is a separate faculty, so the loop has no voice of its own.

    It used to speak between deciding and acting, which made the next move
    wait on language. Decisions now reach the global workspace and a narrator
    — if one is running — speaks from there.
    """
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up"),
        max_cycles=2,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert screen["spoken"] == [], "the loop narrated itself instead of publishing"
    assert screen["pressed"], "and it did not play"
