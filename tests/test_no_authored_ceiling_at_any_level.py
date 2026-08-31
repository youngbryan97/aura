"""There is no level of this that is a list somebody wrote.

She could add a word, then a way of building words, then a way of building
those. Each level grew; the set of ways to WRITE a way of building never did.
It was composition, inversion and iteration, so she could reach anything in
their closure and nothing outside it, and a family needing a maker that is not
one of those three was unreachable however long she looked. Adding a fourth to
the list moves the ceiling. It does not remove it.

A way of building is now a term with a hole in it, in the same algebra a word
is a term in. Composition, inversion, iteration and branching are four things
that can be written, not four primitives — and so is everything else the
grammar admits.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.one_algebra import (
    HEADS,
    Term,
    _computes,
    _where_each_came_from,
    as_a_maker,
    every_term,
    holes_in,
    read_back,
    run,
    the_closure_of_composing_undoing_and_repeating,
    written_down,
)

HOLE = Term("hole", value=0)
OTHER = Term("hole", value=1)
WHERE = Term("where")

COMPOSITION = Term("through", (OTHER, Term("through", (HOLE, WHERE))))
ITERATION = Term("through", (HOLE, Term("through", (HOLE, WHERE))))
INVERSION = Term("undo", (HOLE, WHERE))
BRANCHING = Term(
    "if",
    (
        Term("same as", (Term("left over", (Term("many"), Term("fixed", value=2))),
                         Term("fixed", value=0))),
        HOLE,
        OTHER,
    ),
)


@pytest.fixture(autouse=True)
def _ways_left_as_found():
    was = dict(kinds.WAYS_TO_BUILD)
    kinds.WAYS_TO_BUILD.clear()
    try:
        yield
    finally:
        kinds.WAYS_TO_BUILD.clear()
        kinds.WAYS_TO_BUILD.update(was)


# ── none of the three is a primitive ─────────────────────────────────────


@pytest.mark.parametrize(
    "named,term",
    [
        ("composition", COMPOSITION),
        ("iteration", ITERATION),
        ("inversion", INVERSION),
        ("branching", BRANCHING),
    ],
)
def test_each_of_them_is_a_term_rather_than_a_primitive(named, term):
    assert term.head not in {"composition", "iteration", "inversion", "branching"}
    assert holes_in(term) >= 1


def test_the_grammar_names_no_way_of_building_at_all():
    """What is written down is arithmetic and a branch, not a menu of ideas."""
    assert set(HEADS) == {
        "plus", "minus", "times", "over", "left over", "below", "same as"
    }


def test_a_term_with_a_hole_in_it_is_a_way_of_building_words():
    made = as_a_maker(COMPOSITION)(dict(kinds.WHERE_FROM))
    assert made
    assert all(callable(word) for word in made.values())


# ── and the frontier: outside the closure of those three ─────────────────


def test_branching_is_not_something_those_three_could_have_produced():
    """The claim stated as a check she can run, not as a sentence about her."""
    reach = the_closure_of_composing_undoing_and_repeating(
        dict(kinds.WHERE_FROM), deepest=3
    )
    far = kinds.WHERE_FROM["the far end"]
    along = kinds.WHERE_FROM["one along"]
    shape = tuple(
        run(BRANCHING, at, size, (far, along)) % size
        for size in (3, 4, 5)
        for at in range(size)
    )
    assert len(reach) > 1000
    assert shape not in reach


def test_the_words_those_three_do_reach_are_still_reached():
    reach = the_closure_of_composing_undoing_and_repeating(
        dict(kinds.WHERE_FROM), deepest=2
    )
    far = kinds.WHERE_FROM["the far end"]
    plain = tuple(far(at, size) % size for size in (3, 4, 5) for at in range(size))
    assert plain in reach


# ── written against what she can see, not searched blindly ───────────────


def test_the_correspondence_is_read_off_the_examples():
    states = [(1, 2, 3, 4), (5, 6, 7, 8)]
    reversed_ = [(one, tuple(reversed(one))) for one in states]
    assert _where_each_came_from(reversed_) == {4: (3, 2, 1, 0)}


def test_a_repeat_that_disagrees_with_itself_reads_off_nothing():
    muddled = [((1, 2, 3), (3, 2, 1)), ((4, 5, 6), (5, 4, 6))]
    assert _where_each_came_from(muddled) == {}


def test_a_term_is_checked_against_that_correspondence_directly():
    far = kinds.WHERE_FROM["the far end"]
    wanted = {4: (3, 2, 1, 0), 5: (4, 3, 2, 1, 0)}
    assert _computes(Term("through", (HOLE, WHERE)), (far,), wanted)
    assert not _computes(Term("through", (HOLE, WHERE)),
                         (kinds.WHERE_FROM["one along"],), wanted)


def test_terms_come_shortest_first():
    lengths = [
        term.how_long()
        for term in list(every_term((0, 1, 4), holes=2, deepest=2))[:400]
    ]
    assert lengths == sorted(lengths)


# ── and what she writes survives being written down ──────────────────────


def test_a_term_survives_being_written_down_and_read_back():
    assert read_back(written_down(BRANCHING)) == BRANCHING
    assert read_back(written_down(COMPOSITION)) == COMPOSITION


def test_nothing_outside_the_grammar_can_be_read_back():
    assert read_back({"head": "os.system", "parts": []}) is None
    assert read_back({"head": "if", "parts": []}) is None
    assert read_back("not a term") is None


def test_the_answering_path_writes_one_before_reaching_for_a_written_one():
    """The two makers it used to be handed are the second choice now."""
    import inspect

    from core.cognition import sequence_induction

    source = inspect.getsource(sequence_induction)
    wrote = source.index("a_maker_she_wrote(")
    handed = source.index('"one after another", one_after_another')
    assert wrote < handed


def test_a_term_can_be_applied_to_a_word_and_not_only_a_word_to_a_term():
    """Composition in both directions, which it only ever had in one.

    A hole was kept in its own list and used only as the FIRST part of
    "through" and "undo", so "that word, of this term" was sayable and "this
    term, of what that word gives you" was not — although ``run`` has always
    evaluated it. The term needed to build a maker on a word an earlier maker
    made is five symbols long, computes its family correctly, and appeared in
    none of sixty thousand candidates because nothing could produce its shape.
    """
    from core.cognition.one_algebra import Term, every_term

    wanted = Term(
        head="through",
        parts=(
            Term(head="plus", parts=(Term(head="where"), Term(head="fixed", value=1))),
            Term(head="hole", value=0),
        ),
    )
    for at, one in enumerate(every_term((0, 1, 2, 3), holes=2, deepest=4)):
        if one == wanted:
            break
        if at > 400_000:  # pragma: no cover - the assertion below is the point
            break
    else:  # pragma: no cover
        one = None
    assert one == wanted


def test_a_hole_can_stand_where_any_other_piece_stands():
    """It is a piece like any other, so it belongs everywhere pieces go."""
    from core.cognition.one_algebra import Term, every_term, holes_in

    found = False
    for at, one in enumerate(every_term((1,), holes=1, deepest=2)):
        if one.head in {"plus", "minus"} and any(
            part.head == "hole" for part in one.parts
        ):
            found = True
            assert holes_in(one) >= 1
            break
        if at > 50_000:  # pragma: no cover
            break
    assert found
