"""The moves she has decided against never reach the search.

This is the wiring, not the reasoning: the reasoning is checked in
test_something_she_keeps_true.py. What is checked here is that the pursuit
loop actually takes moves off the table with it, that it will not do so on
too little evidence, that it will not do so when the rules cannot foresee
anything, and that it can never leave her with nothing to press.
"""

from __future__ import annotations

from core.skills.screen_pursuit import _moves_she_will_not_make
from tests.test_something_she_keeps_true import ACTS, _board, _moved, _watched


class _Rules:
    def __init__(self, sure: float = 1.0) -> None:
        self._sure = sure

    def expect(self, reading, act):
        return _moved(reading, act)

    def confidence(self) -> float:
        return self._sure


class _Knows:
    def __init__(self, sure: float = 1.0) -> None:
        self.rules = _Rules(sure)


HERE = _board([[2, 4, 0, 0], [0, 8, 0, 0], [0, 0, 16, 0], [0, 0, 0, 256]])


def test_the_moves_that_break_it_are_gone_before_anything_looks_ahead() -> None:
    keeps: dict = {}
    wont, said = _moves_she_will_not_make(
        keeps, _watched(), HERE, ACTS, _Knows(), turn=0
    )
    assert wont, said
    assert set(wont) < set(ACTS), "she must be left something to press"
    assert keeps["it"] is not None
    assert keeps["it"].name in said


def test_she_will_not_rule_anything_out_on_too_little_evidence() -> None:
    """One move watched for each move she could make, or she holds nothing."""
    thin = _watched(how_many=len(ACTS) - 1)
    wont, said = _moves_she_will_not_make({}, thin, HERE, ACTS, _Knows(), turn=0)
    assert not wont and not said


def test_she_will_not_rule_anything_out_on_rules_she_does_not_trust() -> None:
    wont, _ = _moves_she_will_not_make(
        {}, _watched(), HERE, ACTS, _Knows(sure=0.0), turn=0
    )
    assert not wont


def test_a_move_she_cannot_foresee_the_result_of_survives() -> None:
    """Which is why the way out and the ways of asking are never removed."""
    names = [*ACTS, "start over", "ask him"]
    wont, _ = _moves_she_will_not_make({}, _watched(), HERE, names, _Knows(), turn=0)
    assert "start over" not in wont
    assert "ask him" not in wont


def test_it_refuses_rather_than_leaving_her_stuck() -> None:
    """If every move broke it, refusing them all would not be holding
    something. It is being stuck, and she would rather move."""
    only_bad = ["left", "up"]
    keeps: dict = {}
    _moves_she_will_not_make(keeps, _watched(), HERE, ACTS, _Knows(), turn=0)
    wont, _ = _moves_she_will_not_make(
        keeps, _watched(), HERE, only_bad, _Knows(), turn=0
    )
    assert not wont


def test_what_she_holds_is_worked_out_once_a_turn() -> None:
    keeps: dict = {}
    _moves_she_will_not_make(keeps, _watched(), HERE, ACTS, _Knows(), turn=4)
    first = keeps["it"]
    _moves_she_will_not_make(keeps, [], HERE, ACTS, _Knows(), turn=4)
    assert keeps["it"] is first, "same turn, so not weighed again"


def test_when_she_cannot_hold_it_she_wants_something_nearer_instead() -> None:
    """The rebound, in the loop: not stuck, and not giving the plan up.

    A board where every move breaks what she is holding. Rather than ruling
    nothing out and playing at random, she walks back to something that would
    put it within reach and holds that.
    """
    from core.cognition.something_she_keeps_true import SomethingTrue

    keeps: dict = {
        "at": 0,
        "why": "",
        # Wanting a tile that no single move from here can produce.
        "it": SomethingTrue(
            name="it holds a 512",
            holds=lambda one: any(cell.says == "512" for cell in one.cells),
            well_when_true=9,
            times_true=9,
            well_when_false=1,
            times_false=9,
        ),
    }
    here = _board([[2, 4, 0, 0], [0, 8, 0, 0], [0, 0, 16, 0], [0, 0, 0, 256]])
    # Two of the places she has been are one move from a 512, because they
    # hold two 256s that a single slide would put together. Three are not.
    seen = [
        (_board([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 256, 256]]), True),
        (_board([[2, 4, 0, 0], [0, 8, 0, 0], [0, 0, 16, 0], [0, 0, 0, 256]]), False),
        (_board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4]]), False),
        (_board([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 256], [0, 0, 0, 256]]), True),
        (_board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 8]]), False),
    ]
    _moves_she_will_not_make(keeps, seen, here, ACTS, _Knows(), turn=0)
    assert keeps["it"].name != "it holds a 512", "she should have stepped back"
    assert "256" in keeps["it"].name, keeps["it"].name
