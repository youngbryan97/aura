"""What she feels now cues which memories come back.

Her mood reached recall as one valence against another on the scale of -1 to
1, and her memories are felt across a band a fifth of that wide, so a change in
how she felt moved a recollection's score by about a hundredth. On seed 7 a
displacement that moved her internal geometry by 0.178 moved what she recalled
by 0.011, and "moves together", J*'s structure term, had nothing to correlate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory.felt_at_encoding import encode, held_now
from core.phases.memory_retrieval import _cued_by_what_she_feels

pytestmark = pytest.mark.unit


def _affect(**above: float) -> SimpleNamespace:
    return SimpleNamespace(emotions=dict(above), mood_baselines={name: 0.0 for name in above})


def test_the_share_she_holds_is_graded_by_how_strongly_she_holds_it() -> None:
    joyful = encode({"joy": 0.3})
    assert held_now({"joy": 0.4}, joyful) == pytest.approx(1.0)
    assert held_now({"joy": 0.2, "fear": 0.4}, joyful) == pytest.approx(0.5)
    assert held_now({"fear": 0.4}, joyful) == 0.0
    assert held_now({}, joyful) == 0.0
    assert held_now({"joy": 0.4}, "") == 0.0


def _ranked(affect: SimpleNamespace, salience: float = 0.66) -> list[str]:
    candidates = [
        (0.50, "[memory score=0.500] the afternoon the build finally passed"),
        (0.50, "[memory score=0.500] the evening the run was lost"),
        (0.50, "[memory score=0.500] a note about the weather"),
    ]
    felt = {
        candidates[0][1]: encode({"joy": 0.30, "pride": 0.20}),
        candidates[1][1]: encode({"sadness": 0.30, "frustration": 0.20}),
        candidates[2][1]: encode({"joy": 0.10, "sadness": 0.10}),
    }
    reread = _cued_by_what_she_feels(candidates, felt, affect, salience)
    return [text.split("] ", 1)[1] for _, text in sorted(reread, key=lambda item: item[0], reverse=True)]


def test_a_memory_made_in_what_she_feels_now_comes_back_first() -> None:
    assert _ranked(_affect(joy=0.4, pride=0.2))[0] == "the afternoon the build finally passed"
    assert _ranked(_affect(sadness=0.4, frustration=0.3))[0] == "the evening the run was lost"


def test_a_shift_the_size_of_her_ordinary_span_changes_the_order() -> None:
    """Two feelings held almost equally, then one of them by one ordinary span more."""
    span = 0.044
    level = _ranked(_affect(joy=0.30, sadness=0.30 + span / 2))
    lifted = _ranked(_affect(joy=0.30 + span, sadness=0.30))
    assert level[0] == "the evening the run was lost"
    assert lifted[0] == "the afternoon the build finally passed"


def test_nothing_changes_without_a_feeling_or_without_salience() -> None:
    candidates = [(0.4, "[memory score=0.400] a"), (0.6, "[memory score=0.600] b")]
    felt = {"[memory score=0.400] a": encode({"joy": 0.3})}
    assert _cued_by_what_she_feels(candidates, felt, _affect(), 0.66) == candidates
    assert _cued_by_what_she_feels(candidates, felt, _affect(joy=0.3), 0.0) == candidates
    assert _cued_by_what_she_feels(candidates, {}, _affect(joy=0.3), 0.66) == candidates
