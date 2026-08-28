"""The crossing: reading the cells, without a list of keys to try.

A council of ten was asked how a system extends its own metalanguage. Nine
answered the same way — add `sort_by(key)`, `partition_by(predicate)` and
`filter(predicate)` — which is a person naming the missing kinds, the same act
as adding `grouping` by hand. Four of them said so when pushed.

The tenth cloned the repository and measured, and its answer does not name a
key at all. A stable ordering fixes the correspondence between before and after
even when values repeat, because equal keys keep the order they were found in.
So each transition hands over n-1 facts about the key directly, and those facts
are a graph on the values: components that must share a key, a contradiction
when a strict edge lands inside one, and a level for each component from the
longest path. Nothing is searched. The key is read off.

What the levels turn out to be says which family it was. As many levels as
values is a sort; two is a split by a property; two with the length free is a
filter. Three things that were listed as three kinds are one thing at three
settings.
"""

from __future__ import annotations

from core.cognition.primitive_invention import Transition as T
from core.cognition.value_order import solve_ordering


def test_the_three_families_are_one_mechanism() -> None:
    sortish = solve_ordering(
        [
            T((3, 1, 2), (1, 2, 3)),
            T((5, 9, 7), (5, 7, 9)),
            T((2, 8, 4), (2, 4, 8)),
            T((6, 0, 3), (0, 3, 6)),
        ]
    )
    assert sortish is not None and sortish.natural == "ascending"

    split = solve_ordering(
        [
            T((1, 2, 3, 4), (2, 4, 1, 3)),
            T((5, 6, 7, 8), (6, 8, 5, 7)),
            T((2, 1, 4, 3), (2, 4, 1, 3)),
        ]
    )
    assert split is not None and split.kind != "filtered"

    dropped = solve_ordering(
        [T((1, 2, 3, 4), (2, 4)), T((5, 6, 7, 8), (6, 8)), T((2, 3, 5, 6), (2, 6))]
    )
    assert dropped is not None and dropped.kind == "filtered"
    # Over cells it has seen, it filters.
    assert dropped.apply((2, 3, 5, 6)) == (2, 6)
    # Over cells it has not, it refuses. Which cells go was learned as the list
    # of cells that went, and a cell never shown is not on either list —
    # reading that as "keep" returned the whole state and called it an answer.
    assert dropped.apply((11, 12, 13, 14)) is None


def test_it_reaches_values_it_never_saw_and_says_when_it_cannot() -> None:
    """The order the values carry is a prior, so it is checked, not assumed."""

    seen_order = solve_ordering(
        [
            T((3, 1, 2), (1, 2, 3)),
            T((5, 9, 7), (5, 7, 9)),
            T((2, 8, 4), (2, 4, 8)),
            T((6, 0, 3), (0, 3, 6)),
        ]
    )
    assert seen_order is not None
    assert seen_order.apply((40, 11, 27)) == (11, 27, 40)

    # And it transfers to a domain sharing no values with the examples.
    words = solve_ordering(
        [
            T(("pear", "fig", "date"), ("date", "fig", "pear")),
            T(("kiwi", "apple"), ("apple", "kiwi")),
            T(("plum", "cherry", "berry"), ("berry", "cherry", "plum")),
        ]
    )
    assert words is not None
    assert words.apply(("zebra", "melon")) == ("melon", "zebra")


def test_a_secret_order_is_refused_rather_than_invented() -> None:
    """The control that makes the order-prior falsifiable.

    An ordering over values that is real, consistent, and NOT the order the
    values carry. The right kind of rule, with content that cannot be recovered.
    A mechanism that answers this is fitting a table and calling it a law.
    """

    secret = solve_ordering(
        [T(("q", "k", "z"), ("z", "q", "k")), T(("m", "p", "b"), ("p", "b", "m"))]
    )
    assert secret is not None
    assert secret.natural is None
    assert secret.apply(("w", "y", "a")) is None


def test_values_never_related_are_not_put_in_an_order() -> None:
    """Two chains the observations never joined have no order between them.

    They were being ranked anyway, from levels computed inside each chain
    independently, and the answer put 6 before 4 on values whose relative order
    had never been shown.
    """

    split = solve_ordering(
        [
            T((1, 2, 3, 4), (2, 4, 1, 3)),
            T((5, 6, 7, 8), (6, 8, 5, 7)),
            T((2, 1, 4, 3), (2, 4, 1, 3)),
        ]
    )
    assert split is not None
    assert split.apply((3, 4, 5, 6)) is None
    # Within one chain it answers.
    assert split.apply((4, 3, 2, 1)) == (2, 4, 1, 3)


def test_noise_is_refused() -> None:
    assert solve_ordering([T((1, 2, 3), (9, 4, 7)), T((4, 5, 6), (2, 8, 1))]) is None


def test_it_is_reached_only_where_the_index_language_is_proved_to_fail() -> None:
    """A mirror IS descending order for the cases shown, and it is not a sort.

    Offered beside the index language rather than behind the proof, the wider
    net would take a world the simpler rule already had right.
    """

    from core.cognition.sequence_induction import answer_sequence_question

    said = answer_sequence_question(
        "[1,2,3,4] becomes [4,3,2,1], [1,2,3,4,5] becomes [5,4,3,2,1], "
        "[7,8,9,10] becomes [10,9,8,7]. What does [7,8,9] become?"
    )
    assert "[9, 8, 7]" in said
    assert "n-1-i" in said
    assert "order of their values" not in said


def test_the_live_path_answers_a_sort() -> None:
    from core.cognition.sequence_induction import answer_sequence_question

    said = answer_sequence_question(
        "[3, 1, 2] becomes [1, 2, 3]. [1, 3, 2] becomes [1, 2, 3]. "
        "[2, 1, 3] becomes [1, 2, 3]. What does [9, 4, 7] become?"
    )
    assert "[4, 7, 9]" in said
    assert "ascending" in said


def test_a_thin_world_does_not_reach_past_the_cells_it_saw() -> None:
    """Fitting everything you were shown says nothing when you were shown one thing.

    A single observed pair claimed "ascending" and extrapolated it to every
    value in existence, and a wrong order survives one test half the time. The
    levels still hold for the cells that were shown; what is refused is the
    claim that reaches past them.
    """

    thin = solve_ordering([T((2, 1), (1, 2))])
    assert thin is not None
    assert thin.natural is None
    assert thin.apply((40, 11, 27)) is None
    # And it still answers about the cells it actually saw.
    assert thin.apply((2, 1)) == (1, 2)
