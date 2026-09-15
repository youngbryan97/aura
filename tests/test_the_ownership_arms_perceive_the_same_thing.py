"""The two ownership arms are checked to have perceived the same thing.

P37.4 asks that the arms which differ only in who is named as the actor see the
same percept. The worlds they end in were already compared; what they began by
perceiving was assumed. These pin the check: the open frame's perception domain
is compared, a difference is counted as a mismatch, and no comparison is not
reported as identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.agency import AgencyReport, _same_percept

pytestmark = pytest.mark.unit


class _Frame:
    def __init__(self, perceived: list[float]) -> None:
        self._perceived = np.asarray(perceived, dtype=np.float64)

    def domain(self, key: str) -> np.ndarray:
        assert key == "P"
        return self._perceived


def test_arms_that_opened_on_the_same_percept_match() -> None:
    assert _same_percept([_Frame([0.5, 1.0]), _Frame([9.0, 9.0])], [_Frame([0.5, 1.0]), _Frame([0.0, 0.0])]) is True


def test_arms_that_opened_on_different_percepts_do_not() -> None:
    assert _same_percept([_Frame([0.5, 1.0])], [_Frame([0.5, 0.9])]) is False


def test_no_frames_is_no_reading() -> None:
    assert _same_percept([], [_Frame([0.5])]) is None


def test_the_report_says_identical_only_when_every_comparison_matched() -> None:
    base = dict(self_to_action=0.1, self_to_action_floor=0.0, outcome_to_self=0.2, outcome_to_self_floor=0.1,
                action_text_changed=False, trials=4)
    assert AgencyReport(**base, percepts_matched=4, percepts_compared=4).as_dict()["percepts_identical"] is True
    assert AgencyReport(**base, percepts_matched=3, percepts_compared=4).percepts_identical is False
    assert AgencyReport(**base).percepts_identical is False
