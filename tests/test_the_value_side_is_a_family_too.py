"""``v -> a*v + b``, solved for rather than listed.

The value side had "becomes a constant" and "gains k" and nothing else, so
doubling every cell — as plain a rule as either of them — was silent. Adding
"times k" beside them would have been the third entry in a list that has no
end. These are three members of one family, and the family is two points and a
check: the same argument that collapsed identity, mirror and offset on the
position side.
"""

from __future__ import annotations

from core.cognition.primitive_invention import Transition as T, invent_relation


def _found(world) -> str | None:
    got = invent_relation(world)
    return None if got is None else got.form


def test_scaling_is_reached_without_being_listed() -> None:
    assert _found([T((1, 2, 3), (2, 4, 6)), T((4, 5, 6), (8, 10, 12))]) == (
        "every value becomes 2 times itself"
    )
    assert _found([T((1, 2, 3), (4, 7, 10)), T((4, 5, 6), (13, 16, 19))]) == (
        "every value becomes 3 times itself, plus 1"
    )
    assert _found([T((2, 4), (3, 7)), T((6, 8), (11, 15))]) == (
        "every value becomes 2 times itself, minus 1"
    )


def test_the_simpler_description_still_wins() -> None:
    """A plain offset is a plain offset, not a scaling with a slope of one."""

    assert _found([T((1, 2, 3), (4, 5, 6)), T((4, 5, 6), (7, 8, 9))]) == (
        "every value gains 3"
    )


def test_a_slope_that_does_not_divide_evenly_is_not_a_slope() -> None:
    """Rounding it would invent a rule that nearly works.

    Nearly is indistinguishable from right on the examples it was fitted to,
    which is where it would be checked.
    """

    assert _found([T((1, 2), (3, 4)), T((4, 7), (6, 10))]) != "every value becomes 1 times itself"


def test_noise_is_still_refused() -> None:
    assert _found([T((1, 2, 3), (9, 4, 7)), T((4, 5, 6), (2, 8, 1))]) is None
