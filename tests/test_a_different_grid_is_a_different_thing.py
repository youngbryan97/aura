"""What she counted while reading the thing wrongly is not weak evidence.

Half a game read through a grid four by seven for a board four by four. The
grid corrected itself mid-game and everything after was right, and the rule
still finished at sixty per cent because the wrong half outnumbered it — so
she looked ahead on not one move of a hundred and thirty eight, on a board she
was by then reading perfectly.
"""

from __future__ import annotations

from core.perception.how_it_moves import HowItMoves
from core.perception.what_is_there import Arrangement, Cell


def _board(rows: list[list[int]]) -> Arrangement:
    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=tuple(
            Cell(row=r, column=c, says=str(value), at=(c * 0.1, r * 0.1))
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value
        ),
        down_at=tuple(r * 0.1 for r in range(len(rows))),
        across_at=tuple(c * 0.1 for c in range(len(rows[0]))),
    )


def _nonsense(rules: HowItMoves, how_many: int = 12) -> None:
    """Pairs that no rule explains, which is what a wrong grid produces."""
    for at in range(how_many):
        before = _board([[2, 0, 4, 0, 0, 0, 8], [0, 16, 0, 0, 32, 0, 0]])
        after = _board([[0, 8, 0, 2, 0, 0, 4], [64, 0, 0, 0, 0, 16, 0]])
        rules.watched(before, "left" if at % 2 else "up", after)


def test_what_was_read_through_a_different_grid_is_dropped() -> None:
    rules = HowItMoves()
    _nonsense(rules)
    assert rules.seen > 0, "it did count them"
    rules.learned_through_a_different_reading()
    assert rules.seen == 0
    assert not rules.right
    assert rules.confidence() == 0.0


def test_a_rule_forms_afterwards_that_could_not_have_before() -> None:
    """The point of dropping it: what comes next is no longer outnumbered."""
    rules = HowItMoves()
    _nonsense(rules, how_many=30)
    rules.learned_through_a_different_reading()
    for _ in range(12):
        before = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0]])
        after = _board([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 0]])
        rules.watched(before, "left", after)
    # Which rule it names is not the claim — these boards fit more than one,
    # and naming one of them is right. The claim is that a rule forms at all,
    # which it could not while the wrong half outnumbered it.
    assert "not worked out" not in rules.says(), rules.says()
    assert "combines" in rules.says(), rules.says()
    assert rules.confidence() > 0.9, rules.says()


def test_dropping_nothing_is_quiet() -> None:
    rules = HowItMoves()
    rules.learned_through_a_different_reading()
    assert rules.seen == 0


def test_the_pursuit_drops_it_when_the_grid_changes_shape() -> None:
    """The wiring, and the condition: a change of SHAPE, which settles —
    not every rebuild, which would thrash."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "learned_through_a_different_reading()" in text
    where = text.index("learned_through_a_different_reading()")
    near = text[where - 700 : where]
    assert "built_from(" in near
    assert "was != now" in near
