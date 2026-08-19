"""A loop whose only moves are inside the task can only press harder.

Live, she played 2048 to a stall and kept pressing keys that changed nothing.
Nothing was wrong with the keys; the position was lost. What a person does
there is either start again knowing what the attempt taught, or play it out
on purpose because the ending is where the evidence is.

The runtime's remedy vocabulary had NUDGE, FORCE_NEW_STRATEGY and ASK_HUMAN —
escalations of "try harder" and then "give up". Neither of the two real ways
out was in it.

Both are moves about the task rather than inside it, both are offered only
once the in-task moves have demonstrably stopped working, and both are chosen
through the ordinary deliberation so they are decisions with reasons rather
than things that happen to her.
"""
from __future__ import annotations

import pytest

from core.runtime.stuck_detector import Remedy
from core.skills import screen_pursuit as sp
from core.skills.screen_pursuit import SEE_IT_THROUGH, START_OVER, restart_control, ways_out


class _Store:
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


BOARD = {
    "ok": True,
    "text": "2048 SCORE 744 New Game 2 4 8",
    "bounds": [],
    "layout": [
        {"text": "New Game", "center_x": 0.75, "center_y": 0.18},
        {"text": "2 4 8", "center_x": 0.5, "center_y": 0.5},
    ],
}


def test_the_vocabulary_has_both_ways_out():
    assert Remedy.START_OVER.value == "start_over"
    assert Remedy.SEE_IT_THROUGH.value == "see_it_through"


def test_a_restart_control_is_found_by_what_it_says():
    label, x, y = restart_control(BOARD)
    assert label == "New Game"
    assert (x, y) == (0.75, 0.18)


def test_a_screen_with_no_restart_control_offers_none():
    assert restart_control({"layout": [{"text": "2 4 8", "center_x": 0.5, "center_y": 0.5}]}) is None


def test_both_ways_out_are_offered_when_one_is_available():
    names = [option.name for option in ways_out(BOARD)]
    assert names == [START_OVER, SEE_IT_THROUGH]


def test_playing_it_out_is_offered_even_with_nothing_to_click():
    names = [option.name for option in ways_out({"layout": []})]
    assert names == [SEE_IT_THROUGH]


@pytest.fixture
def screen(monkeypatch):
    state = {"pressed": [], "clicked": [], "text": "2048 SCORE 744 New Game 2 4 8"}

    async def read(app_name=""):
        return dict(BOARD, text=state["text"])

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        return True

    async def click(x, y, *, expect_app="", bounds=None):
        state["clicked"].append((x, y))
        state["text"] = "2048 SCORE 0 New Game 2"
        return True

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://play2048.co/", "title": "2048", "error": ""}

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "click_normalized", click)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)
    return state


def _thinks(*replies):
    queue = list(replies)

    async def think(objective, evidence):
        think.seen = list(evidence)
        return queue.pop(0) if queue else replies[-1]

    think.seen = None
    return think


@pytest.mark.asyncio
async def test_an_ordinary_run_is_never_offered_a_restart(screen, monkeypatch):
    """Nothing gets restarted casually."""
    from core.agency import task_knowledge as tk

    monkeypatch.setattr(tk, "learn_about", _no_learning)
    think = _thinks("up")
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=think,
        max_cycles=1,
        max_seconds=5.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    offered = [line for line in think.seen if line.startswith("Available move")]
    assert not any(START_OVER in line for line in offered)


@pytest.mark.asyncio
async def test_when_nothing_works_she_can_begin_again(screen, monkeypatch):
    from core.agency import task_knowledge as tk

    monkeypatch.setattr(tk, "learn_about", _no_learning)
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up", "up", "up", "start over"),
        max_cycles=6,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert screen["clicked"], "she never took the way out"
    assert result["restarts"] >= 1
    assert result["restarted_because"]


@pytest.mark.asyncio
async def test_she_can_decide_to_play_it_out_instead(screen, monkeypatch):
    from core.agency import task_knowledge as tk

    monkeypatch.setattr(tk, "learn_about", _no_learning)
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up", "up", "up", "see it through", "up"),
        max_cycles=7,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert result.get("played_out_because") is not None
    assert not screen["clicked"], "playing it out is not restarting"
    assert screen["pressed"], "she stopped playing instead of playing on"


async def _no_learning(goal, **kw):
    from core.agency.task_knowledge import TaskKnowledge

    return TaskKnowledge(goal=goal)
