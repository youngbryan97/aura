"""A run nobody can follow is not a narrated run.

She knows the biggest thing she has built in a place and has never said it.
Somebody watching a game came to hear exactly that: not every move, but the
moment the run passes anything she has managed here before.

Said only when it passes what she carried in, so a second attempt is quiet
until it is doing better than the first.
"""

from __future__ import annotations

from core.perception.what_is_there import Arrangement, Cell
from core.skills.screen_pursuit import _she_got_further, _the_biggest_thing_on_it


def _board(*rows: tuple[str, ...]) -> Arrangement:
    cells = [
        Cell(row=r, column=c, says=said, at=(0.0, 0.0))
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ]
    return Arrangement(rows=len(rows), columns=len(rows[0]), cells=tuple(cells))


def test_the_biggest_thing_on_it_is_the_biggest_number():
    assert _the_biggest_thing_on_it(_board(("2", "64"), ("8", "16"))) == 64


def test_a_place_that_only_reports_is_not_something_she_made():
    """A score inside the thing's outline passes every tile almost at once."""
    board = _board(("2", "4096"), ("8", "16"))
    assert _the_biggest_thing_on_it(board, [(0, 1)]) == 16


def test_words_that_are_not_numbers_are_not_counted():
    assert _the_biggest_thing_on_it(_board(("New Game", "8"))) == 8


def test_an_empty_thing_is_nothing():
    assert _the_biggest_thing_on_it(_board(("", ""))) == 0


def test_the_first_one_is_said_plainly():
    assert _she_got_further(64, 0.0) == "I have a 64 on the board."


def test_passing_what_she_managed_before_says_both():
    said = _she_got_further(128, 64)
    assert "128" in said
    assert "64" in said


def test_nothing_is_said_until_it_passes():
    assert _she_got_further(64, 64) == ""
    assert _she_got_further(32, 64) == ""


def test_the_pursuit_says_it_and_keeps_it():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "_she_got_further(made, beaten)" in source
    assert 'furthest["here"] = max(furthest["here"], made)' in source
    assert '"furthest": (' in source


def test_nothing_is_claimed_before_she_knows_what_the_thing_is():
    """Before the grid settles the reading is the whole window.

    LIVE 2026-09-04: "I have a 5292 on the board", on a board holding a 4 and
    two 2s. The 5292 was the best score printed above it.
    """
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index("_she_got_further(made, beaten)")
    assert 'if responds["lattice"].held' in source[at - 2200 : at]


def test_nothing_is_claimed_from_a_reading_of_an_ending():
    """An ending screen is not the board.

    LIVE 2026-09-04: "A 11619 — the biggest I have made here", said over a
    finished game whose best tile was 128. What is on the screen then is an
    overlay, a score and a way to begin again, and its words land in the
    board's places like anything else.
    """
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index("made = (")
    assert 'not responds["state"].nothing_answers()' in source[at : at + 900]


# ── and a record she cannot earn again is not one ────────────────────────


def test_a_carried_record_has_to_turn_up_again_before_it_counts():
    """It is written from a reading, and a reading can be wrong.

    LIVE 2026-09-04: one bad reading put 11619 in this record for a board
    whose best tile was 128, and because the record only ever goes up she
    could never say anything about her own progress in that world again.
    Everything else she carries comes back discounted for exactly this reason.
    """
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index('furthest["again"] = max(')
    nearby = source[at : at + 500]
    assert 'furthest["again"] >= furthest["here"]' in nearby
    assert "_she_got_further(made, beaten)" in nearby


def test_what_is_kept_is_the_record_this_sitting_could_stand_behind():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index('"furthest": (')
    nearby = source[at : at + 300]
    assert 'furthest["again"] >= furthest["here"]' in nearby
