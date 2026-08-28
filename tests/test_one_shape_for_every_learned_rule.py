"""Every learned rule is ``Node(kind, parameters, [Node...])``.

Three things were learnable and each was written down differently: a
positional program, an ordering over cells, and the pair of them. Anything
that wanted "the rule this turn learned" had to know which of the three it
held, and composing one with another needed a type written for that pair —
``Composed`` exists because ordering-then-move needed one, and a third axis
would have needed three more.

That is a limit of the representation rather than a missing feature, so the
test that matters is not that a node exists. It is that an unplanned pair is
expressible without anybody planning it.
"""

from __future__ import annotations

import json

from core.cognition.primitive_invention import Transition, _index_forms
from core.cognition.rule_ir import BY_KEY, POSITIONAL, THEN, Node, as_node
from core.cognition.value_order import solve_ordering, solve_ordering_then_move

_SORT_THEN_ROTATE = [
    Transition((3, 1, 2), (2, 3, 1)),
    Transition((5, 4, 6), (5, 6, 4)),
    Transition((9, 7, 8), (8, 9, 7)),
]


def test_a_composed_rule_is_a_node_with_children() -> None:
    """The pair that needed its own type is an ordinary node now."""

    composed = solve_ordering_then_move(_SORT_THEN_ROTATE, _index_forms(3))
    assert composed is not None
    node = as_node(composed)
    assert node is not None
    assert node.kind == THEN
    assert [part.kind for part in node.parts] == [BY_KEY, POSITIONAL]


def test_the_node_answers_exactly_what_the_solver_answers() -> None:
    """A spine, not a second opinion: the solvers keep their own semantics."""

    composed = solve_ordering_then_move(_SORT_THEN_ROTATE, _index_forms(3))
    node = as_node(composed)
    for state in ((6, 4, 5), (3, 1, 2), (9, 7, 8)):
        assert node.apply(state) == composed.apply(state)


def test_it_survives_a_round_trip_whole() -> None:
    composed = solve_ordering_then_move(_SORT_THEN_ROTATE, _index_forms(3))
    node = as_node(composed)
    back = Node.from_json(json.loads(json.dumps(node.to_json())))
    assert back == node
    assert back.apply((6, 4, 5)) == node.apply((6, 4, 5))


def test_it_says_the_rule_in_the_same_words() -> None:
    """The words live in the table of forms, not on a saved program.

    Found again by value rather than copied, so there is one place they are
    written and a saved rule does not lose them.
    """

    composed = solve_ordering_then_move(_SORT_THEN_ROTATE, _index_forms(3))
    node = as_node(composed)
    assert node.describe() == composed.describe()
    assert "ascending order" in node.describe()
    assert "i+1 (mod n)" in node.describe()


def test_a_pair_nobody_wrote_a_type_for_is_expressible() -> None:
    """The point of the whole exercise.

    Two positional rules in sequence had no composed type. As nodes it is a
    node with two children, and it is composed the same way an ordering and a
    move are.
    """

    mirror = next(
        program for _n, _d, program in _index_forms(4) if program.kind == "mirror"
    )
    rotate = next(
        program
        for _n, _d, program in _index_forms(4)
        if program.kind == "offset" and program.args == (1,)
    )
    pair = Node(
        kind=THEN,
        parts=(as_node(mirror), as_node(rotate)),
    )
    # Mirror (1,2,3,4) -> (4,3,2,1); rotate left by one -> (3,2,1,4).
    assert pair.apply((1, 2, 3, 4)) == (3, 2, 1, 4)
    assert Node.from_json(json.loads(json.dumps(pair.to_json()))) == pair


def test_three_deep_is_no_harder_than_two() -> None:
    """Depth is not a type either."""

    mirror = as_node(
        next(program for _n, _d, program in _index_forms(4) if program.kind == "mirror")
    )
    deep = Node(kind=THEN, parts=(mirror, Node(kind=THEN, parts=(mirror, mirror))))
    # Three mirrors is one mirror.
    assert deep.apply((1, 2, 3, 4)) == (4, 3, 2, 1)


def test_a_rule_that_cannot_say_returns_nothing() -> None:
    """None rather than a guess, at every level of the tree."""

    ordering = solve_ordering(
        [Transition((3, 1, 2), (1, 2, 3)), Transition((9, 7, 8), (7, 8, 9))]
    )
    node = as_node(ordering)
    assert node is not None
    # Cells the ordering never ranked against each other.
    assert node.apply((8, 7, 9)) == ordering.apply((8, 7, 9))

    # And a refusal inside a composition refuses the whole composition.
    pair = Node(kind=THEN, parts=(node, node))
    if ordering.apply((8, 7, 9)) is None:
        assert pair.apply((8, 7, 9)) is None


def test_an_unknown_object_is_not_given_a_node() -> None:
    """Inventing one would put a rule in the library nothing can interpret."""

    assert as_node(None) is None
    assert as_node("a mirror, roughly") is None
    assert as_node(object()) is None
