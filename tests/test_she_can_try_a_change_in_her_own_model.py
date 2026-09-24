"""A change to how she judges is played out in her own model before she lives with it.

Measured 2026-09-19 on 2048: two properties she had kept from live trials,
played out from the same four starts with her own rule and search, took her
from 2048 in all four games to 2048 in one. A live trial compares two
stretches of a life; a rehearsal compares the same starts, the same dice, two
ways of judging.
"""

from __future__ import annotations

from core.agency.rehearsing_in_her_model import Rehearsed, rehearse
from core.perception.how_it_moves import HowItMoves, shifted_and_combined
from core.perception.what_is_there import Arrangement, Cell
from core.perception.what_the_world_does import WhatTheWorldDoes

MOVES = ("up", "down", "left", "right")


def _board(values):
    return Arrangement(
        4, 4, tuple(Cell(i // 4, i % 4, str(v), (0.0, 0.0)) for i, v in enumerate(values) if v)
    )


def _a_world_she_has_learned():
    knows, world = HowItMoves(), WhatTheWorldDoes()
    state = _board([2, 4, 0, 8, 0, 2, 4, 0, 4, 0, 0, 2, 64, 2, 0, 4])
    for move in ("left", "up", "right", "down", "left", "up", "right", "down"):
        after = shifted_and_combined(state, move)
        empty = [i for i in range(16) if after.at(i // 4, i % 4) is None]
        if empty and after.as_text() != state.as_text():
            spot = empty[0]
            after = Arrangement(4, 4, after.cells + (Cell(spot // 4, spot % 4, "2", (0.0, 0.0)),))
        knows.watched(state, move, after)
        world.watched(knows.expect(state, move), after)
        state = after
    return knows, world, state


def test_the_same_starts_are_played_both_ways():
    knows, world, start = _a_world_she_has_learned()
    asked: list[dict] = []

    def her_search(knows_, state, actions, *, toward, world, weights, depth):
        asked.append(dict(weights))
        return {action: (1.0 if action == "left" else 0.0, "") for action in actions}

    rehearsed = rehearse(
        knows, world, start, MOVES,
        weights={"room": 0.15}, trying={"a property": 0.4},
        times=2, how_far=5, choose=her_search,
    )
    assert rehearsed is not None
    assert len(rehearsed.without) == len(rehearsed.with_it) == 2
    # Judged both ways, and only the change differs.
    assert any("a property" in weights for weights in asked)
    assert any("a property" not in weights for weights in asked)


def test_getting_further_is_what_helps():
    assert Rehearsed((256.0, 512.0), (512.0, 1024.0)).helps()
    assert not Rehearsed((2048.0, 2048.0, 2048.0), (1024.0, 2048.0, 1024.0)).helps()
    assert "against" in Rehearsed((2048.0,), (1024.0,), what="x").says()


def test_with_no_model_there_is_nothing_to_rehearse_in():
    empty_knowledge = HowItMoves()
    assert rehearse(
        empty_knowledge, WhatTheWorldDoes(), _board([2, 2, 0, 0] + [0] * 12), MOVES,
        weights={}, trying={"a property": 0.4}, times=1, how_far=3,
    ) is None


def test_her_own_search_plays_it_out():
    knows, world, start = _a_world_she_has_learned()
    rehearsed = rehearse(
        knows, world, start, MOVES, weights={"room": 0.15, "order": 0.15},
        trying={"nothing it knows": 0.4}, times=1, how_far=20,
    )
    assert rehearsed is not None
    assert rehearsed.without[0] >= 64.0


class _Matters:
    def weights(self):
        return {"room": 0.15, "worse": 0.4, "better": 0.4}


class _Knows:
    def __init__(self, rules):
        self.rules = rules


import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_after_a_run_what_does_worse_in_her_model_is_let_go(monkeypatch):
    from core.agency import how_good_is_this, rehearsing_in_her_model, what_she_invented
    from core.skills import screen_pursuit

    rules, world, start = _a_world_she_has_learned()
    monkeypatch.setitem(how_good_is_this.INVENTED, "worse", object())
    monkeypatch.setitem(how_good_is_this.INVENTED, "better", object())
    forgotten, kept = [], []
    monkeypatch.setattr(how_good_is_this, "forget", lambda name: forgotten.append(name) or True)
    monkeypatch.setattr(what_she_invented, "keep", lambda: kept.append(True) or True)

    def played_out(*_args, trying, **_kwargs):
        name = next(iter(trying))
        return Rehearsed((2048.0,), (1024.0,) if name == "worse" else (2048.0 * 2,), what=name)

    monkeypatch.setattr(rehearsing_in_her_model, "rehearse", played_out)
    task = screen_pursuit._judge_what_she_judges_by_in_her_model(
        _Knows(rules), world, {"arranged": start}, _Matters(), MOVES, "2048", False
    )
    assert task is not None
    await task
    assert forgotten == ["worse"]
    assert kept == [True]


def test_with_nothing_learned_there_is_nothing_to_rehearse():
    from core.skills import screen_pursuit

    assert screen_pursuit._judge_what_she_judges_by_in_her_model(
        _Knows(HowItMoves()), WhatTheWorldDoes(), {"arranged": None}, _Matters(), MOVES, "", False
    ) is None


def test_going_nowhere_both_ways_is_no_evidence_to_let_go_on():
    """A run that ended on a finished position rehearses to nothing either way."""
    assert not Rehearsed((0.0, 0.0), (0.0, 0.0)).hurts()
    assert not Rehearsed((64.0,), (64.0,)).hurts()
    assert Rehearsed((2048.0,), (1024.0,)).hurts()


def test_it_is_played_out_from_where_the_run_began():
    from screen_pursuit_support import pursuit_source
    from source_contract import in_order

    source = pursuit_source()
    in_order(source, "going.starting_from(made)", 'pending.setdefault("first_arranged", laid_out)')
    assert 'pending.get("first_arranged") or pending.get("arranged")' in source


def test_a_rehearsal_plays_to_what_she_is_after_rather_than_a_count():
    """Two hundred moves reach about 256 on 2048 whichever way she judges.

    So every rehearsal came out level, and a property measured to cost her
    three games in four was kept after every game (2026-09-20 to 23).
    """
    knows, world, start = _a_world_she_has_learned()
    played: list[int] = []

    def her_search(knows_, state, actions, *, toward, world, weights, depth):
        played.append(1)
        return {action: (1.0 if action == "left" else 0.0, "") for action in actions}

    rehearsed = rehearse(
        knows, world, start, MOVES, weights={"room": 0.15}, trying={"a property": 0.4},
        times=1, toward="64", choose=her_search,
    )
    assert rehearsed is not None
    # Already at 64, so what she is after is reached before a move.
    assert played == []


def test_a_rehearsal_the_clock_cut_off_gives_no_verdict():
    knows, world, start = _a_world_she_has_learned()

    def her_search(knows_, state, actions, *, toward, world, weights, depth):
        import time

        time.sleep(0.02)
        return {action: (1.0 if action == "up" else 0.5, "") for action in actions}

    assert rehearse(
        knows, world, start, MOVES, weights={"room": 0.15}, trying={"a property": 0.4},
        times=1, toward="1000000", choose=her_search, within_s=0.05,
    ) is None
