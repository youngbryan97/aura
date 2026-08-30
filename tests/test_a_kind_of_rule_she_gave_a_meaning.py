"""The interpreter knew three kinds of node, and anything else meant nothing.

Apply these in turn, read the positions, read the cells — and every other kind
returned None. So she could compose programs out of the meanings she was given
and the set of meanings never grew. A node of an unknown kind had no semantics,
and acquiring one meant a person editing the interpreter.

That is the boundary between inventing expressions in a language and inventing
the language. Everything she learned lived inside a semantic algebra whose
evaluation rules were authored, however far its closure reached.

This is the other side of it: a kind whose meaning she worked out from
examples, held to transitions it was not built from, admitted to a registry the
interpreter consults exactly as it consults the three it was born with — and
composable with them, without a branch being written for it.
"""

from __future__ import annotations

import pytest

from core.cognition.an_invented_kind import (
    ENOUGH_HELD_BACK,
    KINDS,
    Induced,
    admit,
    every_meaning,
    forget,
    induce_from,
    interpretation_of,
)
from core.cognition.rule_ir import THEN, Node

#: Each pair is a state before and the state after. Nothing names the family.
PAIRWISE = [((1, 5, 2, 9), (5, 5, 9, 9)), ((3, 1, 4, 1), (3, 3, 4, 4)),
            ((7, 2, 8, 6), (7, 7, 8, 8)), ((2, 6, 1, 3), (6, 6, 3, 3))]
MIRRORED = [((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (6, 5, 4)),
            ((7, 8, 9), (9, 8, 7)), ((1, 1, 2), (2, 1, 1))]
#: Values replaced by something no reading of the places can produce.
NOTHING_SAYS_THIS = [((1, 2), (5, 5)), ((3, 4), (9, 9)),
                     ((5, 6), (1, 1)), ((7, 8), (2, 2))]


@pytest.fixture(autouse=True)
def _a_clean_registry():
    """Nothing is admitted at import, and nothing leaks between tests."""
    held = dict(KINDS)
    KINDS.clear()
    yield
    KINDS.clear()
    KINDS.update(held)


# ── nothing is here that a person put here ───────────────────────────────

def test_the_registry_starts_empty():
    assert KINDS == {}


def test_an_unknown_kind_means_nothing_until_she_works_it_out():
    assert Node(kind="pairwise").apply((4, 9, 1, 2)) is None
    assert Node(kind="pairwise").describe() == "an unreadable rule"


# ── a meaning induced from examples ──────────────────────────────────────

def test_she_works_out_a_meaning_from_before_and_after():
    found = induce_from(PAIRWISE)
    assert found is not None
    assert found.read((4, 9, 1, 2)) == (9, 9, 2, 2)


def test_and_a_different_family_gives_a_different_meaning():
    found = induce_from(MIRRORED)
    assert found is not None
    assert found.read((1, 2, 3, 4)) == (4, 3, 2, 1)


def test_a_family_nothing_in_the_space_expresses_is_refused():
    """The honest answer, and what keeps "I cannot say this" sayable."""
    assert induce_from(NOTHING_SAYS_THIS) is None


def test_too_few_examples_to_hold_anything_back_induces_nothing():
    assert induce_from(PAIRWISE[:1]) is None
    assert induce_from(PAIRWISE[: ENOUGH_HELD_BACK]) is None


def test_a_meaning_is_judged_on_what_it_was_not_built_from():
    found = induce_from(PAIRWISE)
    assert found is not None and found.held_back == 1.0
    assert found.from_examples > 0


# ── admitted, and then the interpreter runs it ───────────────────────────

def test_the_interpreter_runs_a_kind_it_was_never_told_about():
    admit("pairwise", induce_from(PAIRWISE))
    assert Node(kind="pairwise").apply((4, 9, 1, 2)) == (9, 9, 2, 2)


def test_no_branch_was_written_for_it():
    """The proof that the language grew rather than the code."""
    import inspect

    from core.cognition import rule_ir

    source = inspect.getsource(rule_ir)
    assert "pairwise" not in source
    assert "mirrored" not in source


def test_and_it_composes_with_the_kinds_that_were_always_there():
    admit("pairwise", induce_from(PAIRWISE))
    admit("mirrored", induce_from(MIRRORED))
    both = Node(kind=THEN, parts=(Node(kind="pairwise"), Node(kind="mirrored")))
    assert both.apply((4, 9, 1, 2)) == (2, 2, 9, 9)


def test_it_says_what_it_means_in_words():
    admit("pairwise", induce_from(PAIRWISE))
    said = Node(kind="pairwise").describe()
    assert "take" in said and "not built from" in said


def test_what_was_admitted_on_evidence_can_be_taken_back():
    admit("pairwise", induce_from(PAIRWISE))
    assert forget("pairwise") is True
    assert Node(kind="pairwise").apply((4, 9, 1, 2)) is None
    assert forget("pairwise") is False


# ── and nothing is admitted that was never tested ────────────────────────

def test_a_meaning_never_held_to_anything_is_not_admitted():
    """A guess with an executable body is worse than saying she cannot."""
    guess = Induced(where_from="here", and_from="here", what_of_it="as it is")
    assert admit("guessy", guess) == ""
    assert "guessy" not in KINDS


@pytest.mark.parametrize("kind", ["", "   ", None])
def test_a_kind_with_no_name_is_not_a_kind(kind):
    assert admit(kind, induce_from(PAIRWISE)) == ""


def test_something_that_is_not_a_meaning_is_not_admitted():
    assert admit("odd", object()) == ""
    assert admit("odd", None) == ""


def test_nothing_is_run_for_a_kind_she_has_no_meaning_for():
    assert interpretation_of("never heard of it") is None


# ── the space itself ─────────────────────────────────────────────────────

def test_the_space_holds_no_family_by_name():
    import inspect

    from core.cognition import an_invented_kind

    code = [
        line
        for line in inspect.getsource(an_invented_kind).splitlines()
        if not line.strip().startswith("#")
    ]
    for named in ("reverse", "rotate", "swap", "pairwise"):
        assert not any(named in line.lower() for line in code)


@pytest.mark.parametrize("meaning", list(every_meaning()))
def test_every_meaning_reads_a_state_without_falling_over(meaning):
    assert meaning.read(()) == ()
    read = meaning.read((1, 2, 3, 4))
    assert read is None or len(read) == 4


def test_a_state_of_words_is_not_an_error():
    for meaning in list(every_meaning())[:12]:
        read = meaning.read(("a", "b", "c"))
        assert read is None or len(read) == 3
