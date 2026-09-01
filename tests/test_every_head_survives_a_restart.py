"""Every head the interpreter runs can also be read back after a restart.

A maker she wrote is kept as its term and rebuilt from that term at boot. If a
head evaluates but does not read back, the maker is written down correctly,
refused on the way in, and logged as "a term she wrote does not read back" —
which looks like a corrupt file rather than like a missing case.

That is what "over again" did. It is the head one_algebra's own docstring calls
the one with a shape no fixed-length composition has, and it was the one head
missing from the list read_back checked against. Measured, not read: the round
trip returned None.

Two tests, because fixing the one case is not the same as making the class of
defect impossible. The first checks the round trip at every head. The second
checks that the arity table covers exactly what ``run`` dispatches on, by
reading ``run``'s source — so a head added to the interpreter and not to the
table fails here rather than at somebody's next restart.
"""

from __future__ import annotations

import inspect
import re

import pytest

from core.cognition import one_algebra
from core.cognition.one_algebra import (
    HEADS,
    HOW_MANY_PARTS,
    Term,
    read_back,
    run,
    written_down,
)

_A_LEAF = Term("where")


def _a_term_with(head: str) -> Term:
    """The shortest well-formed term at this head."""
    parts = tuple(_A_LEAF for _ in range(HOW_MANY_PARTS[head]))
    value = 3 if head in {"fixed", "hole"} else None
    return Term(head, parts=parts, value=value)


@pytest.mark.parametrize("head", sorted(HOW_MANY_PARTS))
def test_a_term_at_every_head_comes_back_as_itself(head: str) -> None:
    term = _a_term_with(head)
    assert read_back(written_down(term)) == term, (
        f"{head!r} evaluates but does not read back, so a maker she wrote with "
        "it dies at the next restart"
    )


def test_the_head_that_repeats_survives_inside_a_larger_term() -> None:
    """The case that was broken, at the depth it actually occurs."""
    walk = Term("over again", parts=(Term("many"), Term("hole", value=0)))
    nested = Term("through", parts=(Term("hole", value=1), walk))
    assert read_back(written_down(nested)) == nested


def test_the_arity_table_covers_every_head_the_interpreter_dispatches() -> None:
    """Read off ``run``'s source, so a new head cannot be added to one only.

    The defect this replaces was a second list of head names that drifted from
    the first. Any check written as a third list would drift the same way, so
    this one is derived from the interpreter itself.
    """
    source = inspect.getsource(run)
    dispatched = set(re.findall(r'head == "([^"]+)"', source))
    dispatched |= set(HEADS)
    missing = dispatched - set(HOW_MANY_PARTS)
    assert not missing, (
        f"run() evaluates {sorted(missing)} and HOW_MANY_PARTS does not list "
        "them, so read_back will refuse a term that ran perfectly well"
    )
    unknown = set(HOW_MANY_PARTS) - dispatched
    assert not unknown, (
        f"HOW_MANY_PARTS lists {sorted(unknown)} and run() cannot evaluate "
        "them, so read_back would admit a term nothing can run"
    )


def test_a_head_nobody_wrote_is_still_refused() -> None:
    assert read_back({"head": "invented by a corrupt file", "parts": []}) is None
    assert read_back({"head": "if", "parts": []}) is None
    assert read_back("not a row") is None


def test_what_reads_back_still_computes_what_it_did() -> None:
    """Structural equality is not the test; the answers are."""
    words = (lambda at, size: (at + 1) % size, lambda at, size: size - 1 - at)
    term = Term("over again", parts=(Term("fixed", value=2), Term("hole", value=0)))
    back = read_back(written_down(term))
    assert back is not None
    for size in (3, 4, 5, 7):
        for at in range(size):
            assert run(back, at, size, words) == run(term, at, size, words)


def test_the_module_still_exports_the_table() -> None:
    assert "HOW_MANY_PARTS" in one_algebra.__all__
