"""A language that resets every morning has not grown.

The meanings she induces were kept across a restart and the WORDS they are
written in were not, so a meaning came back as a name with nothing behind it.
These pin the whole round trip: the words, the ways of building, and the
meanings written in them.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition import what_she_gave_meaning as keeping
from core.cognition import widening_the_language as widening
from core.cognition.a_constructor_she_built import Recipe, build
from core.cognition.an_operation_that_generalises import an_operation_that_generalises


@pytest.fixture(autouse=True)
def _kept_somewhere_of_its_own(tmp_path, monkeypatch):
    from core.cognition.a_kind_of_thing_she_named import KINDS_OF_THING

    monkeypatch.setattr(keeping, "_KEPT_AT", tmp_path / "meanings.json")
    where, what, ways, known, named = (
        dict(kinds.WHERE_FROM), dict(kinds.WHAT_OF_IT),
        dict(kinds.WAYS_TO_BUILD), dict(kinds.KINDS), dict(KINDS_OF_THING),
    )
    kinds.WAYS_TO_BUILD.clear()
    KINDS_OF_THING.clear()
    try:
        yield
    finally:
        for holds, was in (
            (kinds.WHERE_FROM, where), (kinds.WHAT_OF_IT, what),
            (kinds.WAYS_TO_BUILD, ways), (kinds.KINDS, known),
            (KINDS_OF_THING, named),
        ):
            holds.clear()
            holds.update(was)


def _restart() -> int:
    """Everything she worked out goes; what was written down comes back."""
    kept = keeping.keep()
    assert kept
    kinds.WHERE_FROM.pop("where these came from", None)
    kinds.WHAT_OF_IT.pop("how far apart", None)
    kinds.WAYS_TO_BUILD.clear()
    kinds.KINDS.clear()
    return keeping.recall()


def test_a_derived_word_comes_back_and_the_meaning_still_runs():
    states = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
    where = (2, 0, 3, 1)
    pairs = [(state, tuple(state[at] for at in where)) for state in states]

    found = widening.an_addressing_nobody_wrote(pairs, already=kinds.addressings())
    assert found is not None
    widening.widen_with_addressing("where these came from", found)
    meaning = kinds.induce_from(pairs)
    assert meaning is not None
    kinds.admit("shuffled", meaning)

    _restart()

    run = kinds.interpretation_of("shuffled")
    assert run is not None
    assert run((1, 2, 3, 4)) == (3, 1, 4, 2)


def test_a_way_she_built_comes_back_as_its_recipe():
    """A name resolves against the source. What she built is not in the source."""
    recipe = Recipe(kind="in sequence", depth=3)
    kinds.WAYS_TO_BUILD["a way she built: 3 in sequence"] = build(recipe)
    reach = len(kinds.addressings())

    _restart()

    assert "a way she built: 3 in sequence" in kinds.WAYS_TO_BUILD
    assert len(kinds.addressings()) == reach


def test_a_worked_out_operation_still_answers_a_pair_it_never_saw():
    rule = an_operation_that_generalises(
        [(7, 3, 4), (9, 2, 7), (5, 5, 0), (2, 8, 6), (11, 4, 7), (6, 1, 5)]
    )
    assert rule is not None
    kinds.WHAT_OF_IT["how far apart"] = widening.DerivedOperation(
        name="how far apart", does={(7, 3): 4}, rule=rule
    )

    _restart()

    back = kinds.WHAT_OF_IT.get("how far apart")
    assert back is not None
    assert back(100, 37) == 63


def test_the_words_come_back_before_the_meanings_written_in_them():
    """A meaning read back into a language missing its word has nothing behind it."""
    kinds.WAYS_TO_BUILD["a way she built: 2 times over"] = build(
        Recipe(kind="over and over", depth=2)
    )
    # A family only sayable with the word she built, so what is admitted is a
    # meaning she actually worked out rather than one written here.
    states = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3),
              (5, 1, 9, 3, 7), (8, 2, 6, 4, 9), (3, 7, 1, 5, 2)]
    two_along = [
        (state, tuple(state[(at + 2) % len(state)] for at in range(len(state))))
        for state in states
    ]
    made = kinds.induce_from(two_along)
    assert made is not None
    assert kinds.admit("built on a word she made", made)

    assert _restart() >= 1
    run = kinds.interpretation_of("built on a word she made")
    assert run is not None
    assert run((9, 8, 7, 6, 5)) == (7, 6, 5, 9, 8)


def test_nothing_is_kept_when_she_has_worked_nothing_out():
    """Every store it consults, empty. It consults more of them than it did."""
    from core.cognition.a_kind_of_thing_she_named import KINDS_OF_THING

    kinds.KINDS.clear()
    kinds.WAYS_TO_BUILD.clear()
    KINDS_OF_THING.clear()
    derived = [
        name
        for name, word in list(kinds.WHERE_FROM.items())
        if type(word).__name__ == "DerivedAddressing"
    ]
    for name in derived:
        kinds.WHERE_FROM.pop(name, None)
    assert keeping.keep() is False
