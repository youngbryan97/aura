"""What she learned of one game is kept under its name and given back whole, as plain data."""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.agency.what_meeting_things_does import AVOID, WhatMeetingDoes
from core.agency.what_she_keeps_of_a_game import kept_from, to_keep
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.how_things_move_here import HowThingsMoveHere
from core.perception.what_moves_in_the_picture import Kind

pytestmark = pytest.mark.unit


def test_a_game_played_before_starts_from_what_was_learned():
    hers = WhichIsHers()
    hers.kind = 1
    for _ in range(6):
        hers._hers.add("up", 0.0, -100.0)
    meeting = WhatMeetingDoes()
    meeting.evidence[2].touch_sum, meeting.evidence[2].touches_settled, meeting.evidence[2].immediate = -1.5, 2, 1
    meeting.told[3] = AVOID
    physics = HowThingsMoveHere()
    physics.kinds[2].meetings.append((0.5, 0.4, 1.05))
    physics.kinds[2].edges["top"].bounces = 3
    keep = {
        "kinds": [Kind(0, np.full(64, 1 / 64), 10.0, (1, 2, 3)), Kind(1, np.full(64, 1 / 64), 20.0, (4, 5, 6))],
        "hers": hers, "meeting": meeting, "physics": physics, "meeting_with": {0.7: [3, 1]},
    }
    back = kept_from(json.loads(json.dumps(to_keep(keep))))
    # Which thing is hers and what the keys do are tried again each time.
    assert "hers" not in back
    assert back["meeting"].stance(2) == AVOID and back["meeting"].told == {3: AVOID}
    assert back["physics"].kinds[2].edges["top"].bounces == 3
    assert back["meeting_with"] == {0.7: [3, 1]}
    assert [k.colour for k in back["kinds"]] == [(1, 2, 3), (4, 5, 6)]


def test_nothing_kept_gives_nothing_back():
    assert kept_from({}) == {}
