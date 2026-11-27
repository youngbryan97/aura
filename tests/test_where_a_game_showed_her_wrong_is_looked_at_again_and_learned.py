"""Where a game in the world showed her wrong, looked at again for longer between games and learned from
(core/agency/where_she_was_shown_wrong.py).

Bryan, 27 November 2026: she should learn from a stronger player even while it beats her every game, from what it
does, how it answers positions, what failed and what worked. Her results against it said nothing; the places where its
answers made her judgement fall say where she was wrong.
"""
from __future__ import annotations

import random

import pytest

import core.agency.where_she_was_shown_wrong as shown_wrong
from core.agency.pieces_by_their_kind import after, moves
from core.agency.playing_pieces_by_kind import _a_place
from core.agency.searching_by_kind import _compiled, first_weights
from core.agency.what_a_kind_is_worth import LookedAtLonger, Rehearsal, _with_what_she_was_shown
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME

pytestmark = pytest.mark.unit

needs_the_compiled_search = pytest.mark.skipif(
    _compiled() is None or not hasattr(_compiled(), "seen_as_practice_sees"), reason="the compiled search is not built")


def test_a_fall_out_of_line_with_the_rest_of_the_game_marks_where_she_was_shown_wrong() -> None:
    judged = [(f"place {n}", "a move", score) for n, score in enumerate([0, 10, 5, 15, 10, -380, -370, -390, -380])]
    assert shown_wrong.places_shown_wrong(judged) == ["place 4"]
    steady = [(f"place {n}", "a move", 10 * (n % 2)) for n in range(9)]
    assert shown_wrong.places_shown_wrong(steady) == []


def test_places_wait_once_and_what_is_kept_comes_back_as_it_was() -> None:
    shown = shown_wrong.ShownWrong()
    shown.shown(["a", "b"])
    shown.shown(["b", "c"])
    assert shown.waiting == ["a", "b", "c"] and shown.games == 2
    shown.kept_from(["P", "K"], [3, 2 * 64 + 5], 12.0, -40.0, 64)
    assert shown.parts == [[3, 2 * 64 + 5]] and shown.game == [-3]
    assert shown_wrong.ShownWrong.from_bytes(shown.as_bytes()) == shown


@needs_the_compiled_search
@pytest.mark.parametrize("family", [CHESS, NOBODYS_GAME], ids=["chess", "nobody's game"])
def test_a_place_looked_at_again_keeps_the_quiet_position_its_longer_look_ends_in(monkeypatch, family) -> None:
    monkeypatch.setattr(shown_wrong, "LOOK_BACK_S", 0.3)
    rng = random.Random(4)
    places = []
    for _game in range(3):
        position = family.the_start()
        for _ply in range(rng.randint(6, 20)):
            legal = moves(family, position)
            if not legal:
                break
            position = after(family, position, rng.choice(legal))
        places.append(_a_place(position, dict(family.start), family))
    shown = shown_wrong.ShownWrong()
    shown.shown(places)
    looked = shown_wrong.look_back_on_some(shown, family, first_weights(family), dict(family.start), seconds=60.0)
    assert looked == 3 and not shown.waiting
    assert shown.parts and len(shown.parts) == len(shown.slow) == len(shown.glance) == len(shown.game)
    squares = family.rows * family.columns
    assert all(0 <= i < 2 * len(shown.kinds) * squares for p in shown.parts for i in p)


def test_how_the_pieces_stand_together_is_learned_from_what_she_was_shown_too() -> None:
    import numpy as np

    store = LookedAtLonger(kinds=["P", "K"], squares=64, columns=8, parts=[np.array([1, 2], dtype=np.int16)],
                           glance=[0.0], slow=[10.0], game=[7], scales=[200.0])
    shown = shown_wrong.ShownWrong()
    shown.kept_from(["K", "P"], [5, 2 * 64 + 9], 3.0, -50.0, 64)
    rehearsal = Rehearsal(weights=first_weights(CHESS), looked_at_longer=store, shown_wrong=shown)
    both = _with_what_she_was_shown(rehearsal)
    assert len(both.parts) == 2 and both.slow == [10.0, -50.0] and both.game == [7, -1]
    # The kind first in the shown store's order ("K") is the practice store's second.
    assert both.parts[1].tolist() == [1 * 2 * 64 + 5, 0 * 2 * 64 + 9]
    assert _with_what_she_was_shown(Rehearsal(weights=first_weights(CHESS), looked_at_longer=store)) is store


def test_what_she_was_shown_is_read_again_when_her_worlds_are() -> None:
    import json

    import core.skills.screen_pursuit_board_of_pieces as pursuit
    from core.runtime.what_she_learned import remember_beside

    uneven = {"waiting": [], "kinds": ["K"], "parts": [[1], [2]], "glance": [0.0], "slow": [1.0, 2.0],
              "game": [-1, -1], "games": 1}
    remember_beside("a-board-world", pursuit._SHOWN_WRONG, json.dumps(uneven).encode())
    found = pursuit._what_no_longer_holds_of_what_she_was_shown("a-board-world", {})
    assert any("not as many" in said for said in found)
    remember_beside("a-board-world", pursuit._SHOWN_WRONG, shown_wrong.ShownWrong().as_bytes())
    assert pursuit._what_no_longer_holds_of_what_she_was_shown("a-board-world", {}) == []
