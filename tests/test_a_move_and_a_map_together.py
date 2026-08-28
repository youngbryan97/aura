"""A positional rule and a value map, where neither alone explains it.

The two sides were solved separately and could not meet. "Mirror, then add one
to every cell" is not a rearrangement — no cell that came out ever went in — and
it is not a value map, because the cells also moved. Between them they say it
exactly, and neither of them alone says anything.

Found by measurement rather than by design: the write-axis reported this world
as creating cells, the solver returned None, and the two facts together named
the gap.
"""

from __future__ import annotations

from core.cognition.primitive_invention import Transition as T, invent_relation


def test_a_move_then_a_map() -> None:
    found = invent_relation(
        [T((1, 2, 3, 4), (5, 4, 3, 2)), T((2, 3, 4, 5), (6, 5, 4, 3))]
    )
    assert found is not None
    assert "n-1-i" in found.form and "gains 1" in found.form
    assert list(found.apply((7, 8, 9))) == [10, 9, 8]


def test_a_move_then_a_scaling() -> None:
    found = invent_relation([T((1, 2, 3), (6, 9, 3)), T((4, 5, 6), (15, 18, 12))])
    assert found is not None
    assert list(found.apply((7, 8, 9))) == [24, 27, 21]


def test_the_map_is_fitted_on_the_sorted_states() -> None:
    """The map applies to what the MOVE produced, not to the input positions.

    Reading before[i] and after[i] as a pair of the map's gave four different
    offsets for one offset. A move is a permutation, so the two multisets are
    related by the map alone and sorting both recovers the pairs it was really
    applied to.
    """

    from pathlib import Path

    body = Path("core/cognition/primitive_invention.py").read_text()
    start = body.index("def _map_then_move(")
    window = body[start : start + 2000]
    assert "sorted(item.before)" in window
    assert "sorted(item.after)" in window


def test_a_constant_is_not_usable_this_way() -> None:
    """It destroys what it was applied to, so the move underneath cannot be seen."""

    from core.cognition.primitive_invention import _invertible_value_maps

    every_value_becomes_seven = [(1, 7), (2, 7), (3, 7), (4, 7)]
    assert not _invertible_value_maps(every_value_becomes_seven)
    # An offset keeps the information and is usable.
    assert _invertible_value_maps([(1, 3), (2, 4), (3, 5)])


def test_the_simpler_answer_is_not_displaced() -> None:
    plain_mirror = invent_relation([T((1, 2, 3), (3, 2, 1)), T((4, 5, 6), (6, 5, 4))])
    assert plain_mirror is not None
    assert "and then" not in plain_mirror.form

    plain_offset = invent_relation([T((1, 2, 3), (4, 5, 6)), T((4, 5, 6), (7, 8, 9))])
    assert plain_offset is not None
    assert plain_offset.form == "every value gains 3"


def test_noise_is_still_refused() -> None:
    assert invent_relation([T((1, 2, 3), (9, 4, 7)), T((4, 5, 6), (2, 8, 1))]) is None
