"""What touching a kind of thing, or letting it by, comes to, read from the counters the game writes."""
from __future__ import annotations

import pytest

from core.agency.what_meeting_things_does import AVOID, MEET, Readouts, WhatMeetingDoes, readouts_in

pytestmark = pytest.mark.unit


def _region(text, x, y):
    return {"text": text, "center_x": x, "center_y": y, "x": x - 0.05, "y": y - 0.02, "width": 0.1, "height": 0.04}


def test_numbers_are_read_with_the_words_beside_them():
    found = readouts_in([_region("SCORE", 0.1, 0.05), _region("12", 0.2, 0.05), _region("LIVES 3", 0.9, 0.05)])
    assert {(r.label, r.value) for r in found} == {("SCORE", 12), ("LIVES", 3)}


def test_a_change_is_believed_on_its_second_reading_and_dated_to_its_first():
    counters = Readouts()
    assert counters.read([_region("SCORE 1", 0.1, 0.05)], 1.0, None) == []
    assert counters.read([_region("SCORE 2", 0.1, 0.05)], 1.5, None) == []
    [verdict] = counters.read([_region("SCORE 2", 0.1, 0.05)], 2.0, None)
    assert verdict["what"] == "gain" and verdict["at"] == 1.5 and verdict["since"] == 1.0


def test_of_two_unlabelled_scores_hers_is_on_her_side():
    counters = Readouts()
    regions = lambda left, right: [_region(str(left), 0.42, 0.05), _region(str(right), 0.58, 0.05)]  # noqa: E731
    counters.read(regions(0, 0), 1.0, her_x=0.05)
    counters.read(regions(0, 1), 1.5, her_x=0.05)
    [verdict] = counters.read(regions(0, 1), 2.0, her_x=0.05)
    assert verdict["what"] == "loss"


def _settled(meeting, event, at, verdict):
    meeting.since = 0.0
    meeting._open.append({"what": event, "kind": 7, "at": at})
    meeting.evidence[7]
    if verdict:
        meeting._verdict(dict(verdict))
    meeting._settle(at + 10.0)


def test_a_kind_whose_touch_is_followed_by_a_loss_is_avoided():
    meeting = WhatMeetingDoes()
    _settled(meeting, "touched", 5.0, {"what": "loss", "at": 5.3, "since": 4.9})
    assert meeting.stance(7) == AVOID


def test_a_kind_that_got_by_before_a_loss_is_met():
    meeting = WhatMeetingDoes()
    _settled(meeting, "passed", 5.0, {"what": "loss", "at": 5.3, "since": 4.9})
    assert meeting.stance(7) == MEET and meeting.evidence[7].meet > 0


def test_a_loss_long_after_a_touch_says_nothing_about_it():
    meeting = WhatMeetingDoes()
    _settled(meeting, "touched", 5.0, {"what": "loss", "at": 6.9, "since": 6.4})
    assert meeting.stance(7) == MEET


def test_gains_that_come_every_second_whatever_she_does_credit_nothing():
    """A score that counts time alive covers half the run with verdicts; a pass in it is chance."""
    meeting = WhatMeetingDoes()
    meeting.since = 0.0
    for second in range(1, 30):
        meeting._verdict({"what": "gain", "at": float(second), "since": second - 0.1})
    for at in (3.05, 7.95, 12.5, 18.0):
        meeting._open.append({"what": "passed", "kind": 7, "at": at})
    meeting._settle(40.0)
    assert abs(meeting.evidence[7].meet) < 2.0
