"""Her search's habits moved by trying them in the world she is in (core/agency/how_she_looks_set_by_trying.py): only
the habits a look of a fixed length can feel, each moved toward the side of it that did better, and what a run found
kept only where the sequential test finds it better than what she has.

LIVE 26 November 2026 her search's habits stood where they were set by hand on 22 November, measured in one world,
and nothing she did moved them.
"""
from __future__ import annotations

import math

import pytest

import core.agency.how_she_looks_set_by_trying as trying
from core.agency.pieces_by_their_kind import after, moves
from core.agency.searching_by_kind import _compiled, first_weights
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME

pytestmark = pytest.mark.unit

needs_the_compiled_search = pytest.mark.skipif(
    _compiled() is None or not hasattr(_compiled(), "habits_as_set"), reason="the compiled search is not built")


def _positions() -> list:
    found, position = [], NOBODYS_GAME.the_start()
    for _ in range(12):
        legal = moves(NOBODYS_GAME, position)
        if not legal:
            break
        position = after(NOBODYS_GAME, position, legal[len(legal) // 2])
        found.append(position)
    return found


@needs_the_compiled_search
def test_only_habits_a_look_of_fixed_length_can_feel_are_moved() -> None:
    felt = trying.felt_habits(NOBODYS_GAME, _positions(), first_weights(NOBODYS_GAME), 3000)
    assert felt
    # How a second thread's look is spread, and how a clock is spent, change nothing a single fixed look does.
    assert "helpers_spread" not in felt and "steady_share" not in felt
    assert all(trying.habits_as_set()[name] > 0 for name in felt)


def test_each_habit_moves_toward_the_side_that_won(monkeypatch) -> None:
    import random

    # The side moved up wins its pair outright.
    monkeypatch.setattr(trying, "paired", lambda *args, **kwargs: 1.0)
    run = trying.Trying.begun(["a", "b", "c"], {"a": 1.0, "b": 2.0, "c": 4.0}, pairs_given=10, seed=5)
    before = dict(run.at)
    drawn = random.Random(run.seed * 1_000_003 + run.played)
    sign = {name: drawn.choice((-1.0, 1.0)) for name in run.names}
    run.one_pair(None, None, 0)
    for name in run.names:
        assert math.copysign(1.0, run.at[name] - before[name]) == sign[name]
    # A first step moves a habit by no more than the share it was tried by.
    assert all(abs(run.at[n] - before[n]) <= trying.SHARE + 1e-12 for n in run.names)


def test_a_run_that_found_what_she_has_is_no_better(monkeypatch) -> None:
    monkeypatch.setattr(trying, "paired", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(trying, "habits_as_set", lambda: {"a": 1.0})

    class _Weights:
        habits: dict = {}

    said, scores = trying.tried_against_hers(None, _Weights(), {"a": 1.0}, 0, pairs_at_most=300, spread=0.25)
    assert said == "no better"
    assert len(scores) < 300


def test_rounds_of_trying_how_she_looks_take_turns_with_rounds_of_rehearsal(monkeypatch) -> None:
    import core.agency.what_a_kind_is_worth as worth

    monkeypatch.setattr(trying, "felt_habits", lambda *args, **kwargs: ["shorten_by"])
    monkeypatch.setattr(trying, "habits_as_set", lambda: {"shorten_by": 2.25})
    monkeypatch.setattr(trying, "paired", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(trying, "_engine", lambda: type("Engine", (), {"habits_as_set": staticmethod(lambda: "{}")}))
    rehearsal = worth.a_first_rehearsal(NOBODYS_GAME)
    rehearsal.rounds = 5
    assert trying.wants_a_round(rehearsal)
    last = trying.a_round_of_trying(rehearsal, NOBODYS_GAME, seconds=0.0)
    assert last["tried"] == "how she looks" and last["kept"] is False
    assert not trying.wants_a_round(rehearsal)
    rehearsal.rounds = 6
    assert trying.wants_a_round(rehearsal)


def test_habits_a_run_finds_better_are_the_ones_she_plays_by(monkeypatch) -> None:
    import core.agency.what_a_kind_is_worth as worth

    monkeypatch.setattr(trying, "felt_habits", lambda *args, **kwargs: ["shorten_by"])
    monkeypatch.setattr(trying, "habits_as_set", lambda: {"shorten_by": 2.25})
    # Enough pairs for the test to decide: four pairs won outright cannot show a way better.
    monkeypatch.setattr(trying, "PAIRS_GIVEN", 12)

    def paired(family, weights, these, those, nodes, *, seed):
        # Whatever is tried as found, or moved up, wins its pair.
        return 1.0 if these != those and these.get("shorten_by", 0) >= those.get("shorten_by", 0) else 0.5

    monkeypatch.setattr(trying, "paired", paired)
    rehearsal = worth.a_first_rehearsal(NOBODYS_GAME)
    rehearsal.rounds = 1
    said = None
    for _ in range(40):
        last = trying.a_round_of_trying(rehearsal, NOBODYS_GAME, seconds=0.001)
        if last["kept"]:
            said = last
            break
        rehearsal.rounds += 1
    assert said is not None
    assert rehearsal.weights.habits["shorten_by"] == pytest.approx(said["habits"]["shorten_by"], abs=1e-3)
    assert rehearsal.trying_looks is None


def test_a_run_under_way_is_kept_with_her_record_and_comes_back() -> None:
    import core.agency.what_a_kind_is_worth as worth
    import core.skills.screen_pursuit_board_of_pieces as pursuit
    from core.perception.pieces_by_their_look import Looks

    from tests.fixtures.a_game_of_pieces_to_learn import CHESS

    rehearsal = worth.a_first_rehearsal(CHESS)
    rehearsal.earned = ["worth"]
    run = trying.Trying.begun(["shorten_by"], {"shorten_by": 2.25}, pairs_given=400, seed=3)
    run.played, run.scores = 2, [0.5, 0.75]
    rehearsal.trying_looks, rehearsal.looks_tried_after = run.as_dict(), 7
    pursuit._keep_what_she_learned("A Board App", Looks(), None, rehearsal)
    back = pursuit._what_she_kept("A Board App")["rehearsal"]
    assert back.looks_tried_after == 7
    assert trying.Trying.from_dict(back.trying_looks) == run
