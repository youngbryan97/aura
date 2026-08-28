"""A solver holding `sorted_up` has not discovered sorting.

The induction battery names one family it cannot express: ordering cells by a
property of the values. Elsewhere in the tree, `ProcedureInducer` enumerates
programs over a pinned vocabulary, and that vocabulary contains `sorted_up` and
`unique`. Pointing the second at the first would produce a program that sorts,
and it would mean nothing at all: the answer was handed over with the
primitives.

Nobody has made that claim. This is here so that nobody can make it without
first deleting an assertion that says why it would be empty — the same
discipline the battery fingerprint applies to its own score.
"""

from __future__ import annotations

#: Operations that ARE one of the families the index language cannot express.
#: A search holding these has been told the answer, whatever it then finds.
GIVES_THE_ANSWER_AWAY = frozenset({"sorted_up", "unique"})


def test_the_procedure_vocabulary_still_contains_the_answer() -> None:
    """Recorded as a fact, not repaired.

    These are ordinary operations and that inducer is entitled to them; it was
    never claiming to invent them. Removing them would weaken a working system
    to protect a claim nobody is making. What matters is that the fact is
    written down where the claim would be made.
    """

    from core.learning.procedure_induction import PRIMITIVES

    names = {item.name for item in PRIMITIVES}
    assert GIVES_THE_ANSWER_AWAY <= names, (
        "if these are gone the note below can go too — check first that they "
        "were removed deliberately and not renamed"
    )


def test_the_index_language_does_not_contain_the_answer() -> None:
    """The one whose score is cited has nothing that orders by value.

    Every form it can build says where a cell comes from using its position and
    the length. None of them can read a cell. That is the whole reason its
    111/120 means anything, and it is what the other vocabulary would destroy
    if the two were ever confused.
    """

    from core.cognition.primitive_invention import _index_forms

    kinds = {rule.kind for _f, _d, rule in _index_forms(8)}
    assert not (kinds & GIVES_THE_ANSWER_AWAY)
    assert kinds <= {
        "identity",
        "mirror",
        "offset",
        "exchange",
        "ends",
        "grouping",
        "affine",
    }


def test_the_two_are_not_wired_together() -> None:
    """No import path from the scored language to the vocabulary holding sort."""

    from pathlib import Path

    for name in (
        "core/cognition/primitive_invention.py",
        "core/cognition/relation_language.py",
        "core/cognition/induction_battery.py",
        "core/cognition/sequence_induction.py",
        "core/cognition/language_limits.py",
    ):
        body = Path(name).read_text()
        assert "procedure_induction" not in body, name
