"""An ordering of the cells, then a rearrangement of the positions.

The two halves of the language were solved separately and had no way to meet.
"Sorted, then rotated" is proved outside the positional language — the sources
genuinely contradict, and the proof is right — and the ordering alone cannot
say it either, because the cells do not come out in the order the values carry.
Between them they say it exactly, and neither of them alone says anything.

Undoing the move is what makes the search cheap rather than quadratic. If
after[i] = mid[f(i, n)] then mid is determined by after and f, so each candidate
move has exactly one intermediate state whose ordering has to be solved. The
search is over the moves already known, not over pairs of things.
"""

from __future__ import annotations

from core.cognition.primitive_invention import Transition as T, _index_forms
from core.cognition.value_order import solve_ordering, solve_ordering_then_move


def _sorted_then_rotated(row: tuple) -> tuple:
    ordered = tuple(sorted(row))
    return ordered[1:] + ordered[:1]


def test_neither_half_says_it_alone() -> None:
    from core.cognition.language_limits import certify

    world = [T(row, _sorted_then_rotated(row)) for row in ((3, 1, 2), (9, 5, 7), (2, 8, 4))]

    # The positional language: proved impossible, correctly.
    assert certify(world).proven_outside

    # The ordering alone: it finds a table, and the table cannot extrapolate.
    alone = solve_ordering(world)
    assert alone is None or alone.apply((40, 11, 27)) is None


def test_together_they_say_it_and_it_extrapolates() -> None:
    world = [T(row, _sorted_then_rotated(row)) for row in ((3, 1, 2), (9, 5, 7), (2, 8, 4))]
    found = solve_ordering_then_move(world, _index_forms(3))
    assert found is not None
    assert found.apply((40, 11, 27)) == _sorted_then_rotated((40, 11, 27))
    said = found.describe()
    assert "ascending" in said and "then" in said


def test_only_an_ordering_that_extrapolates_counts() -> None:
    """A table would fit whatever intermediate a move happened to produce.

    It would then be a table of that move's arithmetic rather than a claim
    about the cells, and it would answer nothing it had not been shown.
    """

    secret = [
        T(("q", "k", "z"), ("q", "k", "z")),
        T(("m", "p", "b"), ("p", "b", "m")),
    ]
    found = solve_ordering_then_move(secret, _index_forms(3))
    assert found is None or found.ordering.natural is not None


def test_the_simpler_answer_is_not_displaced() -> None:
    """A world that IS a plain sort is never explained as a sort and a move."""

    from core.cognition.sequence_induction import answer_sequence_question

    plain = answer_sequence_question(
        "[3, 1, 2] becomes [1, 2, 3]. [1, 3, 2] becomes [1, 2, 3]. "
        "[2, 1, 3] becomes [1, 2, 3]. What does [9, 4, 7] become?"
    )
    assert "[4, 7, 9]" in plain
    assert "then" not in plain

    mirrored = answer_sequence_question(
        "[1,2,3,4] becomes [4,3,2,1], [1,2,3,4,5] becomes [5,4,3,2,1], "
        "[7,8,9,10] becomes [10,9,8,7]. What does [7,8,9] become?"
    )
    assert "[9, 8, 7]" in mirrored
    assert "n-1-i" in mirrored


def test_the_live_path_composes() -> None:
    from core.cognition.sequence_induction import answer_sequence_question

    said = answer_sequence_question(
        "[3, 1, 2] becomes [2, 3, 1]. [9, 5, 7] becomes [7, 9, 5]. "
        "[2, 8, 4] becomes [4, 8, 2]. What does [40, 11, 27] become?"
    )
    assert "[27, 40, 11]" in said
    assert "ascending" in said and "then" in said
