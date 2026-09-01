"""A way of computing she wrote, standing where a head stands.

The grammar was the last authored thing in one_algebra. A word is a term, a
maker is a term with a hole, and the heads those terms are built from were
seven arithmetic operations and an if-chain in `run`. `HOW_MANY_PARTS` said
how many parts each took, and adding a ninth meant editing the file.

These tests hold the property that replaced that: a head is a term on the
floor, the floor is universal, and what she can put there is everything
computable rather than a longer list. They also hold the things a new head
must not break — it must persist, it must be removable, and a term mentioning
one that is gone must fail loudly rather than quietly answer something else.
"""

from __future__ import annotations

import pytest

from core.cognition import one_algebra
from core.cognition.one_algebra import (
    DERIVED_HEADS,
    HEADS,
    HOW_MANY_PARTS,
    Term,
    every_term,
    forget_the_head,
    parts_taken_by,
    read_back,
    run,
    the_head_she_wrote,
    written_down,
)
from core.cognition.the_floor_she_stands_on import (
    A,
    FST,
    IF,
    L,
    MINUS,
    N,
    SAME,
    SND,
    TIMES,
    V,
    Y,
    build,
)

#: Reading a floor list at a position. Every head she writes needs it, because
#: what it is handed for each part is what that part says everywhere.
_NTH = Y(
    "nth",
    L(
        "xs",
        L(
            "k",
            IF(
                SAME(V("k"), N(0)),
                FST(V("xs")),
                A(V("nth"), SND(V("xs")), MINUS(V("k"), N(1))),
            ),
        ),
    ),
)

_TWICE = Y(
    "twice",
    L("k", IF(SAME(V("k"), N(0)), N(1), TIMES(N(2), A(V("twice"), MINUS(V("k"), N(1)))))),
)


def _over_everything(body):
    """Close a body over everything a head is given, outermost binder first."""
    return build(
        L("at", L("n", L("here_first", L("here_second", L("all_first", L("all_second", body))))))
    )


def _doubling_head():
    """Two to the power of what the first part says here."""
    return _over_everything(A(_TWICE, V("here_first")))


def _the_other_part_head():
    """What the second part says at the place the first part names."""
    return _over_everything(A(_NTH, V("all_second"), V("here_first")))


@pytest.fixture
def _clean_registry():
    before = dict(DERIVED_HEADS)
    DERIVED_HEADS.clear()
    yield
    DERIVED_HEADS.clear()
    DERIVED_HEADS.update(before)


_HERE = lambda at, size: at  # noqa: E731
_ALONG = lambda at, size: (at + 1) % size  # noqa: E731


def test_a_head_she_wrote_computes(_clean_registry) -> None:
    the_head_she_wrote("two to the", 2, _doubling_head())
    term = Term("two to the", parts=(Term("where"), Term("where")))
    assert [run(term, at, 5, (_HERE, _ALONG)) for at in range(5)] == [1, 2, 4, 3, 1]
    assert [run(term, at, 7, (_HERE, _ALONG)) for at in range(7)] == [1, 2, 4, 1, 2, 4, 1]


def test_a_head_can_read_a_part_somewhere_other_than_here(_clean_registry) -> None:
    """What separates a head she wrote from arithmetic over the parts.

    The parts arrive as tables — what each says at every position — so a head
    can do what ``through`` does. Handing numbers instead would have limited
    her to arithmetic on what the parts happen to say at this one place.
    """
    the_head_she_wrote("that one, over there", 2, _the_other_part_head())
    term = Term(
        "that one, over there", parts=(Term("hole", value=1), Term("hole", value=0))
    )
    got = [run(term, at, 5, (_HERE, _ALONG)) for at in range(5)]
    assert got == [(at + 1) % 5 for at in range(5)]


def test_the_grammar_is_no_longer_a_list(_clean_registry) -> None:
    before = set(HEADS) | {"where", "many", "fixed", "hole", "through", "undo",
                           "over again", "if"}
    the_head_she_wrote("two to the", 2, _doubling_head())
    assert parts_taken_by("two to the") == 2
    assert "two to the" not in HOW_MANY_PARTS
    assert "two to the" not in before


def test_a_head_she_wrote_is_offered_to_the_search(_clean_registry) -> None:
    """Reachable, or it is a registry entry rather than part of the language."""
    the_head_she_wrote("two to the", 2, _doubling_head())
    found = False
    for at, term in enumerate(every_term((0, 1, 2), holes=2, deepest=2)):
        if term.head == "two to the":
            found = True
            break
        if at > 200_000:
            break
    assert found, "a head she wrote never appears in what the search walks"


def test_a_term_using_a_head_she_wrote_survives_a_restart(_clean_registry) -> None:
    the_head_she_wrote("two to the", 2, _doubling_head())
    term = Term("two to the", parts=(Term("where"), Term("many")))
    assert read_back(written_down(term)) == term


def test_a_term_using_a_head_that_is_gone_does_not_quietly_answer(
    _clean_registry,
) -> None:
    the_head_she_wrote("two to the", 2, _doubling_head())
    term = Term("two to the", parts=(Term("where"), Term("where")))
    assert run(term, 2, 5, (_HERE, _ALONG)) == 4
    assert forget_the_head("two to the")
    with pytest.raises(ValueError):
        run(term, 2, 5, (_HERE, _ALONG))
    assert read_back(written_down(term)) is None


def test_a_head_that_will_not_stop_is_refused_rather_than_waited_on(
    _clean_registry,
) -> None:
    """A head is a term on a universal floor, so this case exists and is met."""
    forever = _over_everything(A(Y("go", L("k", A(V("go"), V("k")))), N(1)))
    the_head_she_wrote("never settles", 2, forever)
    term = Term("never settles", parts=(Term("where"), Term("where")))
    with pytest.raises(ValueError):
        run(term, 0, 5, (_HERE, _ALONG))


def test_a_head_that_asks_for_nonsense_is_refused(_clean_registry) -> None:
    the_head_she_wrote(
        "asks for the first of a number", 2, _over_everything(FST(N(1)))
    )
    term = Term("asks for the first of a number", parts=(Term("where"), Term("where")))
    with pytest.raises(ValueError):
        run(term, 0, 5, (_HERE, _ALONG))


def test_the_wrong_number_of_parts_is_refused(_clean_registry) -> None:
    the_head_she_wrote("two to the", 2, _doubling_head())
    with pytest.raises(ValueError):
        run(Term("two to the", parts=(Term("where"),)), 0, 5, (_HERE, _ALONG))


def test_the_registry_ships_empty_so_nothing_is_shipped_as_invented() -> None:
    """What she wrote is what she kept, never what was in the source.

    Read off the source rather than by reloading the module. Reloading builds
    a second copy of the registry and of the arity table while other modules
    still hold the first, which made every test that ran afterwards fail on a
    head it could no longer find — an order dependence introduced by the test
    rather than by the code.
    """
    import ast
    import inspect
    import pathlib as _pathlib

    tree = ast.parse(
        _pathlib.Path(inspect.getfile(one_algebra)).read_text(encoding="utf-8")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DERIVED_HEADS":
            assert isinstance(node.value, ast.Dict)
            assert not node.value.keys, "a head was shipped rather than written"
            return
    raise AssertionError("DERIVED_HEADS is not declared where it was")


def test_no_word_of_hers_says_the_diagonal() -> None:
    """The second witness, inside the range a position lives in.

    The growth bound rules out doubling because it leaves the range a word
    answers in. This one stays inside it: walk the words she can build and
    answer differently from the n-th at the n-th place.
    """
    from core.cognition.what_the_old_language_cannot_say import no_word_of_hers_says_it

    found = no_word_of_hers_says_it(400)
    assert found["words"] == 400
    assert found["checked"] >= 1200
    assert found["differs_from_every_one"], found["agreed"][:5]


def test_the_diagonal_is_a_place_inside_the_state() -> None:
    from core.cognition.what_the_old_language_cannot_say import (
        the_one_no_word_of_hers_says,
    )

    rule = the_one_no_word_of_hers_says()
    for size in (2, 3, 5, 8):
        for at in range(size):
            assert 0 <= rule(at, size) < size
