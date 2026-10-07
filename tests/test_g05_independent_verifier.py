"""G05's independent reading of her final integer, written apart from the runner's."""

from __future__ import annotations

from tools.verify_g05_independently import last_integer, two_sided


def test_the_last_integer_is_read_with_its_thousands() -> None:
    assert last_integer("so 3 and then **2,104,802,751,450**.") == 2104802751450
    assert last_integer("476 583 is the count") == 476583
    assert last_integer("first 12, finally −7") == -7
    assert last_integer("no number") is None


def test_the_two_sided_test_is_symmetric_and_bounded() -> None:
    assert two_sided(5, 0) == two_sided(0, 5) == 2 / 32
    assert two_sided(0, 0) == 1.0 and two_sided(3, 3) == 1.0
