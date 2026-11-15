"""Whether an act had an effect and whether her claim was right are two questions.

They were the same answer while the only claim a move carried was that the
view would differ. They came apart the moment a claim could say something —
and the part of the loop that works out WHICH PART OF THE SCREEN answers to
her was reading the wrong one, so a move that moved the board and did not do
the specific thing she predicted counted as a move that did nothing.
"""

from __future__ import annotations

from screen_pursuit_support import pursuit_source

from pathlib import Path

from core.agency.deliberate_action import Expectation
from core.perception.what_is_there import arranged

SOURCE = pursuit_source()

BEFORE = arranged([(0.3, 0.2, "4"), (0.3, 0.35, "4"), (0.45, 0.2, "8"), (0.45, 0.35, "2")])
AFTER = arranged([(0.3, 0.2, "8"), (0.3, 0.35, "2"), (0.45, 0.2, "8"), (0.45, 0.35, "2")])


def test_a_move_can_change_things_and_still_break_its_claim():
    verdict = Expectation(changed=True, contains=("256",)).check_in(BEFORE, AFTER)
    assert verdict.observed_change
    assert not verdict.held


def test_the_band_learner_is_asked_whether_the_act_had_an_effect():
    # The band learner's call, found by what it is given: the state of which
    # part answers. Other things are "noticed(" too — a rung she reached is.
    # Read as a call, not as text: its `worked` grew a branch on 10 October
    # and the substring this read stopped matching while the answer held.
    import ast

    calls = [
        node for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "noticed"
        and node.args and ast.unparse(node.args[0]) == "responds['state']"
    ]
    assert calls, "the band learner's call is gone"
    for call in calls:
        worked = next(kw.value for kw in call.keywords if kw.arg == "worked")
        said = ast.unparse(worked)
        assert "attempt.verdict.observed_change" in said, said
        assert "attempt.verdict.held" not in said, said


def test_the_record_of_what_she_did_still_carries_the_claim():
    """What she predicted and whether it held belongs in the record either way."""
    assert 'moves[-1]["held"] = attempt.verdict.held' in SOURCE


def test_a_move_that_changed_nothing_is_still_no_effect():
    verdict = Expectation(changed=True).check_in(BEFORE, BEFORE)
    assert not verdict.observed_change
    assert verdict.stalled
