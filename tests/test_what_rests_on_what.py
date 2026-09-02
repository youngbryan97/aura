"""What rests on what, and what happens when the thing underneath goes.

Cascade was the least of the three. A head removal already took the words
written over it; what it did not do was set a head aside without losing it, or
try to derive what rested on it again before writing it off.

The distinction the graph has to keep straight, and the one a naive
implementation gets wrong: a WORD written over a head breaks when the head
goes, because the word's term names it. A HEAD written over another head does
not, because a floor term contains its parts rather than pointing at them — the
descendant carries a copy and keeps computing. A graph that cascaded through
that would destroy things that still work, so it is recorded and not cascaded,
and there is a test for each direction.
"""

from __future__ import annotations

import pytest

from core.cognition.a_way_of_computing_she_wrote import as_a_head
from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.one_algebra import (
    DERIVED_HEADS,
    Made,
    Term,
    forget_the_head,
    run,
    the_head_she_wrote,
)
from core.cognition.the_floor_she_stands_on import Code
from core.cognition.what_rests_on_what import (
    QUARANTINED,
    quarantine,
    rebuild,
    release,
    rests_on,
    retract,
    what_rests_on_it,
)

_HERE = lambda at, size: at  # noqa: E731
_ALONG = lambda at, size: (at + 1) % size  # noqa: E731


@pytest.fixture(autouse=True)
def _clean():
    heads, words, held = dict(DERIVED_HEADS), dict(WHERE_FROM), dict(QUARANTINED)
    DERIVED_HEADS.clear()
    QUARANTINED.clear()
    yield
    DERIVED_HEADS.clear()
    DERIVED_HEADS.update(heads)
    WHERE_FROM.clear()
    WHERE_FROM.update(words)
    QUARANTINED.clear()
    QUARANTINED.update(held)


def _a_head(name: str) -> Code:
    """A head adding what its two parts say here."""
    body = as_a_head(
        Code(
            "plus",
            parts=(
                Code("the one it was given", value=3),
                Code("the one it was given", value=2),
            ),
        )
    )
    the_head_she_wrote(name, 2, body)
    return body


def _a_word_over(head: str, name: str) -> None:
    WHERE_FROM[name] = Made(
        term=Term(head, parts=(Term("hole", value=0), Term("hole", value=1))),
        words=(WHERE_FROM["here"], WHERE_FROM["one along"]),
        built_from=("here", "one along"),
    )


# ── reading the graph ────────────────────────────────────────────────────


def test_what_a_word_rests_on_is_read_off_its_term() -> None:
    _a_head("both here")
    _a_word_over("both here", "a word over it")
    assert rests_on("a word over it") == frozenset({"both here"})
    assert rests_on("here") == frozenset()


def test_a_word_and_a_head_resting_on_it_are_told_apart() -> None:
    body = _a_head("both here")
    the_head_she_wrote("carries a copy", 2, body)
    _a_word_over("both here", "a word over it")
    words, heads = what_rests_on_it("both here")
    assert words == ("a word over it",)
    assert heads == ("carries a copy",)


# ── quarantine ───────────────────────────────────────────────────────────


def test_a_head_can_be_set_aside_without_being_lost() -> None:
    """Suspected wrong is not known wrong, and deleting the first loses the
    evidence that would have settled which it was."""
    _a_head("both here")
    assert quarantine("both here")
    assert "both here" not in DERIVED_HEADS
    assert "both here" in QUARANTINED
    with pytest.raises(ValueError):
        run(Term("both here", parts=(Term("where"), Term("many"))), 0, 5,
            (_HERE, _ALONG))


def test_releasing_puts_back_exactly_what_quarantine_took() -> None:
    body = _a_head("both here")
    quarantine("both here")
    assert release("both here")
    assert DERIVED_HEADS["both here"].body == body
    assert not QUARANTINED


def test_quarantining_something_that_is_not_there_says_so() -> None:
    assert not quarantine("nothing anybody wrote")
    assert not release("nothing anybody wrote")
    _a_head("both here")
    assert quarantine("both here")
    assert not quarantine("both here")


# ── retraction, cascade and rebuild ──────────────────────────────────────


def test_retracting_takes_the_words_and_reports_what_went() -> None:
    _a_head("both here")
    _a_word_over("both here", "a word over it")
    went = retract("both here")
    assert went.removed
    assert went.words == ("a word over it",)
    assert went.inactive == ("a word over it",)
    assert went.rebuilt == ()
    assert "a word over it" not in WHERE_FROM
    assert "here" in WHERE_FROM, "an unrelated word went"


def test_a_word_that_can_be_derived_again_is_rebuilt_rather_than_lost() -> None:
    """Deriving a word is a question about the family it came from, so the
    caller answers it. Without an answer nothing is rebuilt, which is honest
    rather than a silent success."""
    _a_head("both here")
    _a_word_over("both here", "a word over it")

    def derive(name: str):
        return WHERE_FROM["here"] if name == "a word over it" else None

    went = retract("both here", derive=derive)
    assert went.rebuilt == ("a word over it",)
    assert went.inactive == ()
    assert WHERE_FROM["a word over it"] is WHERE_FROM["here"]


def test_rebuild_with_no_way_to_derive_reports_everything_inactive() -> None:
    made, lost = rebuild(("one", "two"))
    assert made == ()
    assert lost == ("one", "two")


def test_a_head_resting_on_it_keeps_computing() -> None:
    """A floor term contains its parts rather than pointing at them.

    The opposite would be a silent wrong answer rather than a loud failure,
    which is why this is a test and not a comment.
    """
    body = _a_head("both here")
    the_head_she_wrote("carries a copy", 2, body)
    went = retract("both here")
    assert went.rests_on_it == ("carries a copy",)
    assert "carries a copy" in DERIVED_HEADS
    still = Term("carries a copy", parts=(Term("where"), Term("many")))
    assert [run(still, at, 5, (_HERE, _ALONG)) for at in range(5)] == [0, 1, 2, 3, 4]


def test_retracting_reads_the_graph_before_it_changes_it() -> None:
    """Reading afterwards reads a graph the removal has already changed."""
    import inspect

    from core.cognition import what_rests_on_what

    source = inspect.getsource(what_rests_on_what.retract)
    before = source.index("what_rests_on_it(said)")
    after = source.index("DERIVED_HEADS.pop(said")
    assert before < after


def test_forgetting_a_head_is_the_same_path() -> None:
    """One implementation, so the two cannot drift apart."""
    _a_head("both here")
    _a_word_over("both here", "a word over it")
    report = forget_the_head("both here")
    assert report["removed"]
    assert report["words"] == ["a word over it"]
    assert report["inactive"] == ["a word over it"]
