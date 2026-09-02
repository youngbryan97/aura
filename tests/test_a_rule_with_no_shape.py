"""A rule whose shape is a term, so the shape is not a thing somebody wrote.

`Induced` was the last authored ceiling above the floor, and a different kind
of ceiling from the heads. A rule there is always

    after[i] = what(before[g(i, n)], before[h(i, n)])

Two sources, one operation, both value-blind. Every word she derives, every
maker she writes and every head she writes fits inside that sentence, and no
amount of any of them changes it. `language_limits.certify` names the last part
exactly: where the cells themselves changed, it refuses to rule, because no
rule about where a cell came from can produce a value that was never there.

A rule here is a floor term handed the whole state, so the number of sources,
whether it reads values or positions, and whether it makes values are inside
the term.

What it reaches, measured over seven families with distinct-valued states at
lengths four to nine, fitted on three and judged on three:

    add one to every cell           5 stages of 1 source   0.8s   makes values
    twice the cell                                          0.8s   makes values
    mirror                                                  0.7s   moves cells
    cell plus its position                                  0.8s   makes values
    sum of a cell and the next      2 sources               2.4s   makes values
    three cells added                                       unsolved
    larger of a cell and its mirror                         unsolved

The two it does not reach are the honest limit and are named rather than
hidden: a third source, and a second source read at a place whose shortest
spelling is past the forty-eight the fold walks.
"""

from __future__ import annotations

import random

import pytest

from core.cognition.a_rule_with_no_shape import (
    RULES_WITH_NO_SHAPE,
    THE_CELL_AT,
    Rule,
    a_rule_she_wrote,
    read_a_rule_back,
    the_rule_written_down,
)
from core.cognition.the_floor_she_stands_on import HOW_MANY_PARTS, how_long


@pytest.fixture(autouse=True)
def _clean():
    before = dict(RULES_WITH_NO_SHAPE)
    RULES_WITH_NO_SHAPE.clear()
    yield
    RULES_WITH_NO_SHAPE.clear()
    RULES_WITH_NO_SHAPE.update(before)


def _family(rule, *, seed: int = 5, sizes=(4, 5, 6, 7, 8, 9)):
    """Before and after states only, with distinct values in each state.

    Distinct because a repeated value leaves more than one place an answer
    could have come from, and reading the sources off the data then names more
    than one place — which `_the_places_the_answers_name` refuses rather than
    guesses at.
    """
    rng = random.Random(seed)
    made = []
    for size in sizes:
        before = tuple(rng.sample(range(100), size))
        made.append((before, tuple(rule(before, at, size) for at in range(size))))
    return made


_REACHED = {
    "add one to every cell": (lambda b, at, n: b[at] + 1, True),
    "twice the cell": (lambda b, at, n: 2 * b[at], True),
    "mirror": (lambda b, at, n: b[n - 1 - at], False),
    "cell plus its position": (lambda b, at, n: b[at] + at, True),
    "sum of a cell and the next": (lambda b, at, n: b[at] + b[(at + 1) % n], True),
}


@pytest.mark.parametrize("name", sorted(_REACHED))
def test_she_writes_it_from_the_states_alone(name: str) -> None:
    rule, makes = _REACHED[name]
    found = a_rule_she_wrote(_family(rule), now_sayable=lambda: False, within=25.0)
    assert found is not None, name
    assert found.makes_new_values is makes, name
    assert set(found.fitted_at) & set(found.judged_at) == set()


@pytest.mark.parametrize("name", sorted(_REACHED))
def test_it_holds_at_a_length_it_never_saw(name: str) -> None:
    rule, _makes = _REACHED[name]
    found = a_rule_she_wrote(_family(rule), now_sayable=lambda: False, within=25.0)
    assert found is not None
    rng = random.Random(97)
    for size in (11, 13):
        before = tuple(rng.sample(range(200), size))
        want = tuple(rule(before, at, size) for at in range(size))
        assert found.read(before) == want, (name, size)


def test_a_rule_that_makes_values_is_what_language_limits_refuses_to_rule_on() -> None:
    """The case the value-blind certificate hands over rather than deciding."""
    from core.cognition.language_limits import certify

    family = _family(_REACHED["add one to every cell"][0])

    class _Seen:
        def __init__(self, before, after):
            self.before, self.after = before, after

    verdict = certify([_Seen(before, after) for before, after in family])
    assert verdict.writes == "creates"
    assert verdict.standing == "undecided"

    found = a_rule_she_wrote(family, now_sayable=lambda: False, within=25.0)
    assert found is not None and found.makes_new_values


def test_nothing_is_written_where_something_already_says_it() -> None:
    found = a_rule_she_wrote(
        _family(_REACHED["mirror"][0]), now_sayable=lambda: True, within=25.0
    )
    assert found is None


def test_a_family_with_too_few_examples_holds_nothing_back() -> None:
    found = a_rule_she_wrote(
        _family(_REACHED["mirror"][0], sizes=(4, 5)),
        now_sayable=lambda: False,
        within=25.0,
    )
    assert found is None


def test_a_state_of_something_other_than_numbers_is_refused() -> None:
    words = [(("a", "b", "c"), ("c", "b", "a")) for _ in range(4)]
    assert a_rule_she_wrote(words, now_sayable=lambda: False, within=5.0) is None


def test_it_survives_a_restart_and_still_runs() -> None:
    found = a_rule_she_wrote(
        _family(_REACHED["add one to every cell"][0]),
        now_sayable=lambda: False,
        within=25.0,
    )
    assert found is not None
    back = read_a_rule_back(the_rule_written_down(found))
    assert back is not None
    assert back.body == found.body
    assert back.makes_new_values == found.makes_new_values
    assert back.read((7, 3, 9, 1, 5, 2, 8)) == (8, 4, 10, 2, 6, 3, 9)


def test_the_rule_is_a_term_of_the_floor_and_nothing_else() -> None:
    found = a_rule_she_wrote(
        _family(_REACHED["mirror"][0]), now_sayable=lambda: False, within=25.0
    )
    assert found is not None

    def heads(code):
        yield code.head
        for part in code.parts:
            yield from heads(part)

    assert set(heads(found.body)) <= set(HOW_MANY_PARTS)
    assert set(heads(THE_CELL_AT)) <= set(HOW_MANY_PARTS)
    assert how_long(THE_CELL_AT) > 15, "a fixed point, which is why it is supplied"


def test_the_shape_is_not_a_field(  ) -> None:
    """What the module is for, as an assertion rather than a docstring.

    Nothing on the record says how many places a rule reads or what it does
    with them. A rule that reads one and a rule that reads two differ in their
    term and in nothing else.
    """
    one = a_rule_she_wrote(
        _family(_REACHED["twice the cell"][0]), now_sayable=lambda: False, within=25.0
    )
    two = a_rule_she_wrote(
        _family(_REACHED["sum of a cell and the next"][0]),
        now_sayable=lambda: False,
        within=25.0,
    )
    assert one is not None and two is not None
    assert set(Rule.__dataclass_fields__) == {
        "body",
        "fitted_at",
        "judged_at",
        "makes_new_values",
    }
    assert how_long(two.body) > how_long(one.body)


def test_it_reaches_the_ladder_and_is_kept_there() -> None:
    from core.cognition.sequence_induction import _a_rule_with_no_shape  # noqa: PLC2701

    said = _a_rule_with_no_shape(_family(_REACHED["add one to every cell"][0]))
    assert said is not None
    assert "makes values that were never there" in said
    assert len(RULES_WITH_NO_SHAPE) == 1
