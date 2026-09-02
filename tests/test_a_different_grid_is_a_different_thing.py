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


def test_the_pursuit_drops_it_in_both_places_it_has_to() -> None:
    """Twice, because either one alone leaves the hole open.

    During a run, when the grid corrects itself. And on loading, because the
    counts are written down at the end and read back at the start — so a
    sitting that began with the wrong grid poisons every sitting after it, and
    the during-a-run correction never fires because by then the grid is
    already right.
    """
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = [
        where
        for where in range(len(text))
        if text.startswith("learned_through_a_different_reading()", where)
    ]
    assert len(at) == 2, f"expected both call sites, found {len(at)}"
    around = [text[where - 800 : where] for where in at]
    assert any("read_through" in one for one in around), (
        "one of them must fire on what the remembered counts were read through"
    )
    assert any("built_from(" in one and "was != now" in one for one in around), (
        "and one when the grid changes shape during a run"
    )


def test_the_grid_it_was_read_through_survives_the_process() -> None:
    """And loading it at all is worth a test.

    This went out live and fell over on the first line of a run — the counts
    came back, the shape did not, and nothing here exercised the loading, so
    compiling and linting both passed a function that could not be called.
    """
    rules = HowItMoves()
    before = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    after = _board([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    rules.watched(before, "left", after)
    assert rules.read_through == (4, 4)

    again = HowItMoves.from_memory(rules.as_memory())
    assert again.read_through == (4, 4)
    assert again.seen == rules.seen

    # And the shapes of memory it has to survive.
    assert HowItMoves.from_memory({}).read_through == (0, 0)
    assert HowItMoves.from_memory("not a memory at all").read_through == (0, 0)
    assert HowItMoves.from_memory({"read_through": "nonsense"}).read_through == (0, 0)
    assert HowItMoves.from_memory({"read_through": [4]}).read_through == (0, 0)
    assert HowItMoves.from_memory({"read_through": ["a", "b"]}).read_through == (0, 0)


def test_a_rule_she_has_named_is_not_wiped_by_a_tighter_crop() -> None:
    """The correction for early confusion must not fire once she knows.

    LIVE 2026-09-02, the sitting after one that ended knowing the rule at
    eighty six per cent: she carried it in, looked ahead from the first move,
    and this wiped it at move twenty nine on a board she was reading
    perfectly. The sitting after that inherited the wreckage and looked ahead
    on eighteen moves of a hundred and forty seven.
    """
    rules = HowItMoves()
    for _ in range(40):
        before = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 0, 0]])
        after = _board([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 0]])
        rules.watched(before, "left", after)
    assert rules.rule() is not None, rules.says()
    named, seen = rules.says(), rules.seen

    # Now a place turns out to be furniture. The crop gets tighter; what she
    # has already established does not become wrong.
    for _ in range(12):
        before = _board([[2, 2, 0, 9], [0, 0, 0, 9], [0, 0, 0, 9], [4, 4, 0, 9]])
        after = _board([[4, 0, 0, 9], [0, 0, 0, 9], [0, 0, 0, 9], [8, 0, 0, 9]])
        rules.watched(before, "left", after)
    assert rules.rule() is not None, rules.says()
    assert rules.seen >= seen, f"{rules.says()} — was {named}"
