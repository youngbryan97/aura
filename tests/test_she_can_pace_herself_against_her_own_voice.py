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

    def said(key, chosen=None, *, out_loud=False):
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
    """A stand-in mind whose answers are scripted for the question about a move.

    She asks two different questions of language: how to go about the task at
    all, and which move to make now. A double that answers by call order
    hands a move-shaped reply to the question about the approach, which is
    not a fixture detail — it is the same confusion the loop itself has to
    avoid.
    """
    queue = list(replies)
    asked = []

    async def think(objective, evidence):
        asked.append(list(evidence))
        if "Decide how to play toward this goal" in str(objective):
            return ""
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
    offered = [line for call in think.asked for line in call if line.startswith("Available move")]
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
    offered = [line for call in think.asked for line in call if line.startswith("Available move")]
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


def _bare_presence():
    import threading
    from collections import deque

    import core.perception.ambient_presence as ap

    presence = ap.AmbientPresence.__new__(ap.AmbientPresence)
    presence._lock = threading.Lock()
    presence._pending_utterance = ""
    presence._utterance_at = 0.0
    presence._narration = deque(maxlen=12)
    presence._mode = ap.PresenceMode.BUBBLE
    return presence


def test_a_narrated_line_moves_on_by_itself(monkeypatch):
    """LIVE: she played for a minute and the bubble showed one move the whole
    time.

    An unprompted thought waits to be dismissed, because it is an offer
    nobody asked for. A commentary is a stream, and waiting for a dismissal
    that is never coming leaves the first line on screen while everything
    after it queues up behind.
    """
    import time

    import core.perception.ambient_presence as ap

    monkeypatch.setattr(ap, "_proactivity_suppressed", lambda: False)
    presence = _bare_presence()
    for line in ("Board: Up", "Board: Left", "Board: Right"):
        presence.offer_utterance(line, requested=True)

    assert presence._pending_utterance == "Board: Up"
    presence._promote_next_narration()
    assert presence._pending_utterance == "Board: Up", "it moved on before it could be read"

    presence._utterance_at = time.time() - (ap._NARRATION_DWELL_S + 1.0)
    presence._promote_next_narration()
    assert presence._pending_utterance == "Board: Left"


def test_an_unprompted_thought_still_waits_to_be_dismissed(monkeypatch):
    import time

    import core.perception.ambient_presence as ap

    monkeypatch.setattr(ap, "_proactivity_suppressed", lambda: False)
    presence = _bare_presence()
    presence.offer_utterance("I noticed something about your screen")
    presence._utterance_at = time.time() - 60.0
    presence._promote_next_narration()
    assert presence._pending_utterance == "I noticed something about your screen"


def test_language_is_asked_where_it_changes_the_answer():
    """A board changes a little each move, so re-reasoning every one buys
    little and costs the whole cycle — measured live, a language pass took
    about eight seconds and a decision from evidence takes none, on a loop
    that needs hundreds of moves.
    """
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    where = source.index("asking = (")
    condition = source[where : where + 400]
    # The first move, a run that has stopped getting anywhere, one weighing
    # whether to start over, a fresh board, and periodically in between.
    assert "unusual" in condition
    assert "not moves" in condition
    assert "restarts[" in condition
    assert "LANGUAGE_EVERY" in condition


@pytest.mark.asyncio
async def test_skipping_language_is_not_language_failing(body):
    """A caller that deliberately did not ask is not a mind out of reach."""
    from core.agency.deliberate_action import ActionOption, deliberate

    options = [ActionOption(name="up", detail="press up")]
    decided = await deliberate(
        "reach 128", "a board", options, think=None, lived=False, spine=_Store(), graph=_Store()
    )
    assert decided.reached
    assert decided.spoke is False
    assert "could not be reached" not in decided.reason


def test_a_live_decision_carries_its_own_deadline():
    """LIVE: one generation timed out at the endpoint's own 103-second budget
    and the whole run stood still for it.

    That budget is right for a hard answer somebody is waiting on and wrong
    for a move in a game. Past its own deadline, deciding from evidence is
    not merely faster — it is the only thing still about the board in front
    of her.
    """
    import inspect

    from core.agency import her_reasoning

    source = inspect.getsource(her_reasoning)
    assert "timeout=timeout_s" in source, "the model call has no deadline of its own"
    assert "asyncio.wait_for" in source, "a client that never returns would hang the loop"
    assert her_reasoning.DECISION_BUDGET_S <= 10.0


def test_how_far_she_commits_is_what_her_predictions_say():
    """2048 generates a new tile after every move, so a plan is a bet that
    decays with each step.

    The length of a plan is not a setting — it is what her own recent
    predictions say about how predictable this position is.
    """
    from core.agency.deliberate_action import PLAN_AHEAD, Attempt, Verdict, how_far_to_commit

    def graded(held):
        return Attempt(option="up", expected="a shift", verdict=Verdict(held=held, observed_change=held))

    assert how_far_to_commit([]) == 1, "nothing measured yet is one move, then find out"
    assert how_far_to_commit([graded(True)] * 4) == PLAN_AHEAD
    assert how_far_to_commit([graded(True), graded(True), graded(False), graded(False)]) == PLAN_AHEAD // 2
    assert how_far_to_commit([graded(False)] * 4) == 1


def test_a_plan_is_read_in_the_order_she_named_it():
    from core.agency.deliberate_action import ActionOption, choose_sequence

    options = [ActionOption(name=name) for name in ("up", "down", "left", "right")]
    plan = choose_sequence("left, down, left, down", options, 4)
    assert [option.name for option in plan] == ["left", "down", "left", "down"]


def test_a_name_said_twice_running_is_one_move():
    from core.agency.deliberate_action import ActionOption, choose_sequence

    options = [ActionOption(name=name) for name in ("up", "down", "left")]
    plan = choose_sequence("left left then down", options, 4)
    assert [option.name for option in plan] == ["left", "down"]


def test_a_plan_that_disagrees_with_the_conclusion_is_not_a_plan():
    """What she concluded wins; one move is a plan of one."""
    import inspect

    from core.agency import deliberate_action

    source = inspect.getsource(deliberate_action.deliberate)
    assert "planned[0] is not chosen" in source


def test_a_pivot_is_immediate_and_a_first_attempt_is_not_retried_every_move():
    """The condition breaking is news and is worth the pass that answers it.

    Having no stated approach yet is not news. A loop that asks for one every
    cycle pays a full language pass per move for an answer that was not there
    last time either.
    """
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    where = source.index("time_to_ask = (")
    condition = source[where : where + 300]
    assert 'plan["held"] is not None' in condition, "a real pivot is answered at once"
    assert "LANGUAGE_EVERY" in condition, "a first attempt waits for the rhythm"


def test_what_she_is_doing_is_held_where_the_rest_of_her_can_read_it():
    """An approach that lives inside the loop that chose it is a gear in a box.

    Asked mid-task what she is doing, the answer has to come from what her
    body is actually working from.
    """
    from core.agency import what_she_is_doing as doing
    from core.brain.self_state_report import runtime_self_report

    doing.taking_on("reach the highest tile", where="Safari")
    doing.going_about_it(
        "keep the largest tile in one corner",
        because="it keeps a clear row above",
        watching_for="the corner losing its tile",
        alternatives=("build along the right edge",),
        lived=False,
    )
    try:
        lines = doing.as_lines()
        assert any("reach the highest tile" in line for line in lines)
        assert any("keep the largest tile in one corner" in line for line in lines)
        assert any("the corner losing its tile" in line for line in lines)
        report = runtime_self_report()
        assert "keep the largest tile in one corner" in report, "her instruments omitted her own plan"
    finally:
        doing.how_it_went(False, "test", graph=_Store())


def test_changing_her_mind_keeps_what_she_left_behind():
    from core.agency import what_she_is_doing as doing

    doing.taking_on("get somewhere")
    doing.going_about_it("the first way", lived=False)
    doing.going_about_it("the second way", because="the first stopped working", lived=False)
    try:
        current = doing.right_now()
        assert current.approach == "the second way"
        assert current.left_behind == ("the first way",)
        assert current.changes == 1
    finally:
        doing.how_it_went(False, "test", graph=_Store())


def test_an_undertaking_nobody_touched_stops_being_the_present():
    import time

    from core.agency import what_she_is_doing as doing

    doing.taking_on("something long ago")
    doing.going_about_it("a way", lived=False)
    current = doing.right_now()
    current.changed_at = time.time() - (doing.STALE_AFTER_S + 1)
    try:
        assert doing.right_now() is None, "she answered for work that stopped hours ago"
        assert doing.as_lines() == []
    finally:
        doing.how_it_went(False, "test", graph=_Store())
