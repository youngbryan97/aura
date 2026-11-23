"""A board with few pieces left is worked out to the end under the rules she holds, and in a game she plays it so.

Every placement of the pieces of an ending, with either side to move, is listed and worked out back from the ends her
rules give (core/agency/what_a_small_ending_comes_to.py). The rules of chess stand in for hers here, on a board of four
squares by four so the test lists a few thousand situations, not half a million.
"""
from __future__ import annotations

from dataclasses import replace

from core.agency.pieces_by_their_kind import FIRST, SECOND, Kind, Move, Position, moves
from core.agency.what_a_small_ending_comes_to import (
    Ending,
    an_ending_worked_out,
    rules_of_the_ending,
    worked_to_the_end,
)
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

SMALL = replace(CHESS, rows=4, columns=4, start={})
HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL = ((FIRST, "K"), (FIRST, "Q"), (SECOND, "K"))
#: Their royal in a corner, hers two squares below it, her strongest piece one move from shutting it in.
ONE_FROM_THE_END = {(3, 0): (SECOND, "K"), (1, 0): (FIRST, "K"), (2, 3): (FIRST, "Q")}


def test_an_ending_is_worked_out_back_from_the_ends_its_rules_give() -> None:
    ending = an_ending_worked_out(SMALL, HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL)
    here = Position(cells=dict(ONE_FROM_THE_END), turn=FIRST)
    assert ending.value_of(here) == 2, "won for her, one turn on"
    shut_in = Move((2, 3), (3, 3))
    assert worked_to_the_end({ending.pieces: ending}, SMALL, here, moves(SMALL, here)) == shut_in
    after = Position(cells={**{k: v for k, v in ONE_FROM_THE_END.items() if k != (2, 3)}, (3, 3): (FIRST, "Q")},
                     turn=SECOND)
    assert ending.value_of(after) == 1, "won for her where they stand: they have no move, their royal attacked"
    again = Ending.from_bytes(ending.as_bytes())
    assert again is not None and again.value_of(here) == 2 and again.rules == ending.rules


def test_an_ending_rests_only_on_the_kinds_in_it() -> None:
    other = replace(SMALL, kinds={**SMALL.kinds, "N": Kind("N", name="anything", steps=())})
    assert rules_of_the_ending(other, HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL) == \
        rules_of_the_ending(SMALL, HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL)
    queen = SMALL.kinds["Q"]
    along_lines_only = replace(queen.steps[0], offsets=queen.steps[0].offsets[:4])
    weaker = replace(SMALL, kinds={**SMALL.kinds, "Q": replace(queen, steps=(along_lines_only,))})
    assert rules_of_the_ending(weaker, HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL) != \
        rules_of_the_ending(SMALL, HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL)
    # Worked out under rules she no longer holds for its kinds, an ending is not played from.
    ending = an_ending_worked_out(SMALL, HER_ROYAL_AND_STRONGEST_AGAINST_A_ROYAL)
    here = Position(cells=dict(ONE_FROM_THE_END), turn=FIRST)
    assert worked_to_the_end({ending.pieces: ending}, weaker, here, moves(weaker, here)) is None


def _learned(rows: int, columns: int, start: dict):
    from core.agency.how_the_pieces_move import ROYAL_AFTER, HowThePiecesMove

    rules = HowThePiecesMove.from_the_start(rows, columns, start)
    rules.kept_off_attacked["K"] = ROYAL_AFTER + 1
    return rules


def test_the_endings_worth_working_out_are_each_royal_with_one_more_piece_of_any_kind_either_side_began_with() -> None:
    from core.agency.what_a_small_ending_comes_to import endings_worth_working_out

    endings = endings_worth_working_out(_learned(CHESS.rows, CHESS.columns, CHESS.start))
    assert len(endings) == 10 and len(set(endings)) == 10
    assert ((FIRST, "K"), (FIRST, "Q"), (SECOND, "K")) in endings
    assert ((FIRST, "K"), (SECOND, "K"), (SECOND, "R")) in endings
    assert all(sum(1 for side, kind in e if kind == "K") == 2 for e in endings)


def test_practice_works_one_ending_out_a_round_and_keeps_them_beside_her_record(monkeypatch) -> None:
    import core.agency.practising_between_games as practice
    import core.runtime.what_she_learned as what_she_learned
    from core.agency.what_a_small_ending_comes_to import endings_from_bytes

    start = {(0, 0): (FIRST, "K"), (0, 1): (FIRST, "Q"), (3, 3): (SECOND, "K"), (3, 2): (SECOND, "Q")}
    kept = {"rules": _learned(4, 4, start)}
    monkeypatch.setattr(practice, "_held", lambda name, written: kept)
    monkeypatch.setattr(practice, "_PLAYING", False)
    stored: dict = {}
    monkeypatch.setattr(what_she_learned, "remember_beside",
                        lambda world, what, payload: stored.__setitem__(what, payload) or True)
    assert practice._work_out_an_ending("A Board", (0, 0)) == 1
    assert practice._work_out_an_ending("A Board", (0, 0)) == 1
    assert practice._work_out_an_ending("A Board", (0, 0)) == 0, "both endings worth working out are worked out"
    kept_beside = endings_from_bytes(stored["worked-to-the-end"])
    assert set(kept_beside) == {((FIRST, "K"), (FIRST, "Q"), (SECOND, "K")), ((FIRST, "K"), (SECOND, "K"), (SECOND, "Q"))}
