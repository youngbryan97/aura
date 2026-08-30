"""Inventing a kind of thing, not another way of transforming things.

Every word she could invent was an operation: places to places. A great deal of
understanding is not an operation at all — it is a new KIND of thing to have
operations about. Velocity is not an operation on positions. Once "even" and
"odd" exist, arithmetic that was hopeless over the numbers is trivial over the
two of them.

Nothing here is given: which cases belong together is read off a failure, and
the test that applies the distinction is synthesised.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.a_kind_of_thing_she_named import (
    KINDS_OF_THING,
    a_kind_of_thing_she_named,
    read_back,
    written_down,
)

FOURS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
FIVES = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3), (5, 1, 9, 3, 7)]


@pytest.fixture(autouse=True)
def _left_as_found():
    ways, named = dict(kinds.WAYS_TO_BUILD), dict(KINDS_OF_THING)
    kinds.WAYS_TO_BUILD.clear()
    KINDS_OF_THING.clear()
    try:
        yield
    finally:
        for holds, was in ((kinds.WAYS_TO_BUILD, ways), (KINDS_OF_THING, named)):
            holds.clear()
            holds.update(was)


def _one_way_for_evens_another_for_odds():
    far = kinds.WHERE_FROM["the far end"]
    along = kinds.WHERE_FROM["one along"]

    def either(at, size):
        return far(at, size) if size % 2 == 0 else along(at, size)

    return [
        (one, tuple(one[either(at, len(one)) % len(one)] for at in range(len(one))))
        for one in FOURS + FIVES
    ]


def test_she_names_the_distinction_a_failure_points_at():
    named = a_kind_of_thing_she_named(_one_way_for_evens_another_for_odds())
    assert named is not None
    assert len(named.classes) == 2


def test_it_is_parity_and_it_holds_on_sizes_she_never_saw():
    """Four and five were in front of her. Twelve and thirteen were not."""
    named = a_kind_of_thing_she_named(_one_way_for_evens_another_for_odds())
    assert named is not None
    for size in (2, 6, 8, 12, 13, 21):
        assert named.of(tuple(range(size))) == size % 2


def test_a_test_that_memorises_the_sizes_it_met_is_refused():
    """"How many there are" separates fours from fives and invents a class for six."""
    named = a_kind_of_thing_she_named(_one_way_for_evens_another_for_odds())
    assert named is not None
    seen = {named.of(tuple(range(size))) for size in range(2, 20)}
    assert len(seen) == 2, f"a kind of thing has classes, not one per size: {seen}"


def test_nothing_is_named_where_the_cases_do_not_split_that_way():
    already = [(one, tuple(reversed(one))) for one in FOURS]
    assert a_kind_of_thing_she_named(already) is None


def test_nothing_is_named_from_cases_with_no_structure():
    import random

    drawn = random.Random(11)
    noise = [
        (one, tuple(drawn.sample(list(one), len(one)))) for one in FOURS + FIVES
    ]
    named = a_kind_of_thing_she_named(noise)
    if named is not None:
        seen = {named.of(tuple(range(size))) for size in range(2, 20)}
        assert len(seen) == 2


def test_a_distinction_she_drew_survives_a_restart():
    named = a_kind_of_thing_she_named(_one_way_for_evens_another_for_odds())
    assert named is not None
    again = read_back(written_down(named))
    assert again is not None
    assert again.of((1, 2, 3, 4, 5)) == named.of((1, 2, 3, 4, 5))
    assert again.classes == named.classes


def test_nothing_outside_the_algebra_reads_back_as_a_kind():
    assert read_back({"name": "x", "tells": {"head": "os.system"}}) is None
    assert read_back("not a kind") is None


def test_it_is_kept_with_the_rest_of_the_language(tmp_path, monkeypatch):
    """A distinction that dies at process exit was never a concept."""
    from core.cognition import what_she_gave_meaning as keeping

    monkeypatch.setattr(keeping, "_KEPT_AT", tmp_path / "meanings.json")
    named = a_kind_of_thing_she_named(_one_way_for_evens_another_for_odds())
    assert named is not None
    assert keeping.keep()

    KINDS_OF_THING.clear()
    keeping.recall()
    back = list(KINDS_OF_THING.values())
    assert back
    assert back[0].of(tuple(range(13))) == 1
    assert back[0].of(tuple(range(12))) == 0
