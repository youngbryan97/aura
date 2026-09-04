"""A rule that said the board would slide, over a board that did not, is wrong.

An act that changed nothing is agreed about by every rule that also said
nothing would change, and counting those lets a rule ride to certainty on the
many acts that did nothing. That is why the tally that judges a rule was
restricted to acts where something moved.

It threw away the other half. A rule that claimed a change, over an act that
produced none, has been refuted — and pressing a direction into a wall, over
and over, could not overturn a carried rule that claimed the board would
slide. Measured: fourteen straight contradictions of exactly that kind, and
the rule still stood.
"""

from __future__ import annotations

from core.perception.how_it_moves import HowItMoves
from core.perception.what_is_there import Arrangement, Cell


def _board(*rows: tuple[str, ...]) -> Arrangement:
    cells = [
        Cell(row=r, column=c, says=said, at=(0.0, 0.0))
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ]
    return Arrangement(rows=len(rows), columns=len(rows[0]), cells=tuple(cells))


#: Full rows with no two equals touching: every rule agrees a push along
#: them does nothing.
PACKED = _board(("2", "4", "8", "16"), ("4", "8", "16", "32"))

#: Gaps in every row: "slides and combines" says a push closes them.
GAPPED = _board(("2", "", "4", ""), ("", "8", "", "16"))


def test_a_claim_of_movement_that_did_not_happen_is_counted_against_the_rule():
    model = HowItMoves()
    for _ in range(8):
        # It says the gaps close. They do not, every time.
        model.watched(GAPPED, "right", GAPPED)
    assert model.tried_when_it_moved.get("slides and combines", 0) == 8
    assert model.right_when_it_moved.get("slides and combines", 0) == 0


def test_a_rule_that_claimed_nothing_gains_nothing_from_it():
    model = HowItMoves()
    for _ in range(8):
        model.watched(PACKED, "right", PACKED)
    # Nothing to slide and no two equals touching: every rule agrees, and
    # agreement about an act that did nothing is not evidence for any of them.
    assert model.tried_when_it_moved.get("does not move", 0) == 0
    assert model.tried_when_it_moved.get("slides and combines", 0) == 0


def test_it_can_overturn_what_she_carried_in():
    from core.perception.how_it_moves import shifted_and_combined

    model = HowItMoves()
    state = _board(("2", "", "", ""), ("", "4", "", ""), ("", "", "8", ""))
    for step in range(12):
        move = ("left", "up", "right", "down")[step % 4]
        after = shifted_and_combined(state, move)
        model.watched(state, move, after)
        state = after
    assert model.rule() is not None
    for _ in range(14):
        model.watched(GAPPED, "right", GAPPED)
    rule = model.rule()
    assert rule is None or rule.name != "slides and combines"
