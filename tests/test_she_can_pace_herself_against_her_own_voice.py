"""Two faculties running at once will not run at the same speed.

She can act faster than she can speak. The one that falls behind is normally
invisible to the one that does not, so the commentary quietly loses moves and
nothing in her notices.

Noticing is not enough on its own — noticing without a lever is a status
line. So the gap is published, and there are three things she can actually
do about it: wait for the voice to catch up, say less per move, or carry on
and let some of it go unsaid. Any of the three is defensible, which is why it
is a decision and not a rule.
"""
from __future__ import annotations

import pytest

from core.skills import screen_pursuit as sp
from core.skills.screen_pursuit import PRESS_ON, SAY_LESS, SLOW_DOWN, pacing_options


class _Store:
    def record(self, episode):
        return "ep"

    def resolve(self, episode_id, outcome):
        pass

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        pass


def test_the_gap_is_only_a_question_when_there_is_one():
    assert pacing_options({"waiting": 0}) == []
    assert pacing_options({}) == []


def test_all_three_answers_are_offered():
    names = [option.name for option in pacing_options({"waiting": 3})]
    assert names == [SLOW_DOWN, SAY_LESS, PRESS_ON]


def test_the_option_says_how_far_behind_she_is():
    waiting = pacing_options({"waiting": 4})[0]
    assert "4 line(s) behind" in waiting.detail


def test_the_presence_reports_its_own_backlog():
    import threading
    from collections import deque

    import core.perception.ambient_presence as ap

    presence = ap.AmbientPresence.__new__(ap.AmbientPresence)
    presence._lock = threading.Lock()
    presence._pending_utterance = "Board: Up"
    presence._narration = deque(["Board: Left", "Board: Down"], maxlen=12)
    backlog = presence.narration_backlog()
    assert backlog["waiting"] == 2
    assert backlog["showing"] == 1
    assert backlog["capacity"] == 12


@pytest.fixture
def body(monkeypatch):
    state = {"pressed": [], "said": [], "backlog": 3, "text": "board 0"}

    async def read(app_name=""):
        return {"ok": True, "text": state["text"], "layout": [], "bounds": []}

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        state["text"] = f"board {len(state['pressed'])}"
        return True

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://example.test/", "title": "board", "error": ""}

    def said(key, chosen=None):
        state["said"].append((key, chosen is not None))

    async def catch_up(before, patience=4.0):
        state["backlog"] = 0

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)
    monkeypatch.setattr(sp, "_say_move", said)
    monkeypatch.setattr(sp, "narration_backlog", lambda: {"waiting": state["backlog"]})
    monkeypatch.setattr(sp, "let_the_voice_catch_up", catch_up)

    from core.agency import task_knowledge as tk

    async def no_learning(goal, **kw):
        return tk.TaskKnowledge(goal=goal)

    monkeypatch.setattr(tk, "learn_about", no_learning)
    return state


def _thinks(*replies):
    queue = list(replies)
    asked = []

    async def think(objective, evidence):
        asked.append(list(evidence))
        return queue.pop(0) if queue else replies[-1]

    think.asked = asked
    return think


@pytest.mark.asyncio
async def test_she_is_offered_the_choice_when_her_voice_is_behind(body):
    think = _thinks("press on")
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=think,
        max_cycles=2,
        max_seconds=10.0,
        narrate=True,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    offered = [line for line in think.asked[0] if line.startswith("Available move")]
    assert any(SLOW_DOWN in line for line in offered)


@pytest.mark.asyncio
async def test_choosing_to_slow_down_waits_for_the_voice(body):
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("slow down", "up"),
        max_cycles=3,
        max_seconds=10.0,
        narrate=True,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert body["backlog"] == 0, "she chose to wait and did not"
    assert result["pacing"]["waited"] >= 1


@pytest.mark.asyncio
async def test_choosing_to_say_less_keeps_moving_and_drops_the_reasoning(body):
    result = await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("say less", "up", "left"),
        max_cycles=4,
        max_seconds=10.0,
        narrate=True,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert body["pressed"], "saying less should not stop her playing"
    assert all(not carried for _key, carried in body["said"]), "the reasoning was still attached"
    assert result["pacing"]["chose"] == SAY_LESS


@pytest.mark.asyncio
async def test_pressing_on_keeps_both_the_pace_and_the_reasoning(body):
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("press on", "up"),
        max_cycles=3,
        max_seconds=10.0,
        narrate=True,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert body["pressed"]
    assert any(carried for _key, carried in body["said"])


@pytest.mark.asyncio
async def test_a_silent_run_is_never_asked_about_pacing(body):
    think = _thinks("up")
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=think,
        max_cycles=1,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    offered = [line for line in think.asked[0] if line.startswith("Available move")]
    assert not any(SLOW_DOWN in line for line in offered)


def test_effort_follows_what_rides_on_the_decision():
    """The amplifier's value is agreement between attempts and a verifier's
    opinion. That is worth the wall-clock on a hard answer and wrong on a
    move in a game — a loop acting once a second cannot spend several
    generations per decision.
    """
    from core.agency.her_reasoning import reasoning_for

    assert reasoning_for(0.2).__qualname__.startswith("quick_reasoning")
    assert reasoning_for(0.5).__qualname__.startswith("her_reasoning")
    assert reasoning_for(0.9).__qualname__.startswith("deep_reasoning")


def test_a_routine_move_is_treated_as_routine():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "min(stakes, 0.3)" in source, "every step was being paid for at full weight"
    where = source.index("min(stakes, 0.3)")
    assert "stuck(history)" in source[max(0, where - 300) : where], (
        "being stuck has to still be worth more than one pass"
    )
