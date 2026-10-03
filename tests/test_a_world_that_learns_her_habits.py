"""Whether the world answers what she keeps doing, measured rather than assumed.

Watched 2 October 2026: in two fights and a basketball breakdown, whoever kept
lunging in with the same thing kept getting caught, and a centre with one move
was schemed out of it once the other side knew it was coming. In a world that
learns, the best act becomes the worst by being the habit; in one that does not,
repeating the best act is right. See core/cognition/what_wears_out.py.
"""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from core.cognition.what_wears_out import REMEMBERED_OUTCOMES, WhatWearsOut, gap_before

pytestmark = pytest.mark.unit

ACTS = ("a", "b", "c", "d")


def _played(world, *, seed: int, n: int = 400, varied: float = 0.5) -> tuple[WhatWearsOut, list[str]]:
    """A player who mostly favours one act and sometimes takes another."""
    roll = random.Random(seed)
    wears, made = WhatWearsOut(), []
    for _ in range(n):
        act = roll.choice(ACTS) if roll.random() < varied else "a"
        wears.it_went(act, world(act, made, roll), made)
        made.append(act)
    return wears, made


def _learns(act, made, roll) -> bool:
    """Expectations hold when an act comes as a surprise, and break when it is a habit."""
    gap = gap_before(act, made)
    return roll.random() < (0.9 if gap is None or gap > 3 else 0.3)


def _does_not_learn(act, made, roll) -> bool:
    return roll.random() < {"a": 0.85, "b": 0.75, "c": 0.6, "d": 0.5}[act]


def test_the_gap_is_counted_in_her_own_acts():
    assert gap_before("left", ["left", "up", "down"]) == 3
    assert gap_before("down", ["left", "up", "down"]) == 1
    assert gap_before("right", ["left"]) is None


def test_in_a_world_that_learns_her_the_habit_carries_a_cost():
    learned = [_played(_learns, seed=seed) for seed in range(10)]
    sure = [wears.learns_her() for wears, _made in learned]
    assert sum(chance > 0.5 for chance in sure) >= 8, sure
    wears, made = learned[0]
    # Straight after making it, "a" is worth less; rested, it is not.
    assert wears.cost_of_going_again("a", made + ["a"]) > 0.2
    assert wears.cost_of_going_again("a", made + ["a", "b", "c", "d", "b", "c", "d", "b"]) == 0.0
    assert "answers what I keep doing" in wears.says()


def test_in_a_world_that_does_not_learn_her_nothing_is_marked_down():
    for seed in range(30):
        wears, made = _played(_does_not_learn, seed=seed)
        assert wears.learns_her() < 0.5, seed
        assert max(wears.worn(ACTS, made).values()) < 0.15, seed
        assert "does not answer" in wears.says()


def test_one_early_miss_is_not_read_as_rest_mattering():
    """LIVE-shaped: the same key pressed all game, missed once on its first use."""
    wears, made = WhatWearsOut(), []
    for at in range(300):
        wears.it_went("down", at > 0, made)
        made.append("down")
    assert wears.learns_her() == pytest.approx(0.5)
    assert wears.cost_of_going_again("down", made) == 0.0


def test_a_habit_that_never_varies_teaches_nothing_either_way():
    """With every gap the same there is no contrast, and the record says so."""
    wears, made = WhatWearsOut(), []
    roll = random.Random(3)
    for _ in range(200):
        wears.it_went("a", _learns("a", made, roll), made)
        made.append("a")
    assert wears.learns_her() == pytest.approx(0.5)
    assert wears.cost_of_going_again("a", made) == 0.0


def test_what_is_carried_comes_back_discounted_and_rescored():
    wears, made = _played(_learns, seed=1, n=300)
    kept = WhatWearsOut.from_memory(wears.as_memory(), trust=0.5)
    assert len(kept.record) == 150
    assert kept.record == wears.record[-150:]
    assert WhatWearsOut.from_memory(None).record == []
    bounded, _ = _played(_does_not_learn, seed=2, n=REMEMBERED_OUTCOMES + 40)
    assert len(bounded.record) == REMEMBERED_OUTCOMES


# ── wired into the play loop ─────────────────────────────────────────────────


def _a_run_that_learned_her(seed: int = 0) -> SimpleNamespace:
    wears, _made = _played(_learns, seed=seed)
    assert wears.learns_her() > 0.5
    return SimpleNamespace(wears=wears)


def test_where_she_sees_ahead_the_habit_loses_its_lead(monkeypatch):
    from core.skills import screen_pursuit
    from core.skills.screen_pursuit_decision_steps import _mark_down_what_she_keeps_doing

    said: list[str] = []
    monkeypatch.setattr(screen_pursuit, "_tell", lambda line, *_a, **_k: said.append(line))
    run = _a_run_that_learned_her()
    options = [SimpleNamespace(name=name) for name in ACTS]
    ahead = {"a": (1.0, "best on the board"), "b": (0.9, "next best"), "c": (0.2, ""), "d": (0.1, "")}
    # "a" made just now; "b" left seven acts ago, longer than this world remembers.
    moves = [{"key": key} for key in ("b", "c", "d", "c", "d", "c", "d", "a")]
    after, kept = _mark_down_what_she_keeps_doing(run, ahead, options, moves, True)
    assert max(after, key=lambda name: after[name][0]) == "b"
    assert kept == options
    assert said and "will not lean on one move" in said[0]


def test_where_she_cannot_see_ahead_the_rested_acts_come_first():
    from core.skills.screen_pursuit_decision_steps import _mark_down_what_she_keeps_doing

    run = _a_run_that_learned_her()
    options = [SimpleNamespace(name=name) for name in ACTS]
    _after, kept = _mark_down_what_she_keeps_doing(run, {}, options, [{"key": "a"}], False)
    assert kept[-1].name == "a"


def test_in_a_world_that_does_not_learn_her_the_choice_is_untouched():
    from core.skills.screen_pursuit_decision_steps import _mark_down_what_she_keeps_doing

    wears, _made = _played(_does_not_learn, seed=4)
    options = [SimpleNamespace(name=name) for name in ACTS]
    ahead = {"a": (1.0, ""), "b": (0.95, ""), "c": (0.2, ""), "d": (0.1, "")}
    after, _kept = _mark_down_what_she_keeps_doing(
        SimpleNamespace(wears=wears), ahead, options, [{"key": "a"}], False
    )
    assert max(after, key=lambda name: after[name][0]) == "a"


def test_the_graded_press_is_measured_from_the_press_before_it():
    from core.skills.screen_pursuit_decision_steps import _grade_how_soon_she_came_back

    run = SimpleNamespace(wears=WhatWearsOut())
    moves = [{"key": "left"}, {"key": "up"}, {"key": "left"}, {"key": "down"}]
    _grade_how_soon_she_came_back(run, "left", False, moves)
    assert run.wears.record[-1][:2] == ("left", 2)


def test_the_loop_grades_every_single_press_and_keeps_the_record(monkeypatch):
    """Through the real loop on a world that slides, with nothing faked inside her."""
    import itertools

    from screen_pursuit_support import patch_pursuit
    from test_she_learns_how_a_world_moves_by_playing_it import _World

    import core.skills.screen_pursuit as sp
    from core.cognition import what_wears_out

    it = _World()
    graded: list[str] = []
    real = what_wears_out.WhatWearsOut.it_went

    def spy(self, act, held, made_before):
        graded.append(act)
        return real(self, act, held, made_before)

    monkeypatch.setattr(what_wears_out.WhatWearsOut, "it_went", spy)
    kept: dict = {}

    async def read(app_name="", over=None):
        return it.drawn()

    async def press(key, *, expect_app=""):
        it.push(key)
        return True

    async def press_many(keys, *, expect_app=""):
        for key in keys:
            it.push(key)
        return len(list(keys))

    async def yes(*_a, **_k):
        return True

    async def identity():
        return {"url": "", "title": "", "error": ""}

    class _Allowed:
        allowed = True
        reason = None

    async def may_look(*_a, **_k):
        return _Allowed()

    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "press", press)
    patch_pursuit(monkeypatch, "press_many", press_many)
    patch_pursuit(monkeypatch, "_ensure_frontmost", yes)
    patch_pursuit(monkeypatch, "current_page_identity", identity)
    patch_pursuit(monkeypatch, "_bring_the_thing_back_to_the_front", yes)
    patch_pursuit(monkeypatch, "remember", lambda world, data, *a, **k: kept.update(data), raising=False)
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async", may_look
    )
    ways = itertools.cycle(("left", "up", "right", "down"))

    async def think(_prompt, **_kw):
        return f"I will press {next(ways)} to push everything to one side."

    asyncio.run(
        sp.pursue_on_screen(
            goal="make the numbers larger",
            success_when=r"\b4096\b",
            think=think,
            narrate=False,
            lived=False,
            research=False,
            target_app="TheThing",
            max_cycles=30,
            max_seconds=120.0,
        )
    )
    assert it.pressed, "she never pressed anything"
    assert graded, "no press she made was ever graded against how soon she came back to it"
    assert set(graded) <= {"left", "up", "right", "down"}
    assert "wears" in kept and len(kept["wears"]["record"]) == len(graded)
