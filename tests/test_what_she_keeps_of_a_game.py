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
        "kinds": [Kind(number, np.full(64, 1 / 64), 10.0 * (number + 1), colour)
                  for number, colour in enumerate(((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))],
        "hers": hers, "meeting": meeting, "physics": physics, "meeting_with": {0.7: [3, 1]},
    }
    back = kept_from(json.loads(json.dumps(to_keep(keep))))
    # Which thing is hers and what the keys do are tried again each time.
    assert "hers" not in back
    assert back["meeting"].stance(2) == AVOID and back["meeting"].told == {3: AVOID}
    assert back["physics"].kinds[2].edges["top"].bounces == 3
    assert back["meeting_with"] == {0.7: [3, 1]}
    assert [k.colour for k in back["kinds"]] == [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]


def test_nothing_kept_gives_nothing_back():
    assert kept_from({}) == {}


def test_a_bounded_game_memory_keeps_each_kind_with_its_own_measured_effects(tmp_path, monkeypatch):
    from core.agency.what_she_keeps_of_a_game import INDEXED_TABLES
    from core.runtime import what_she_learned

    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 4_000)
    kinds = [Kind(n, np.full(64, 1 / 64), 10.0 + n, (n, 20, 40)) for n in range(80)]
    meeting, physics, hers = WhatMeetingDoes(), HowThingsMoveHere(), WhichIsHers()
    hers.kind = 79
    for n in range(80):
        meeting.evidence[n].touch_sum = -float(n)
        meeting.evidence[n].touched = n + 1
        physics.kinds[n].speeds.extend([float(n)] * 20)
    held = to_keep({"kinds": kinds, "meeting": meeting, "physics": physics, "hers": hers})
    assert what_she_learned.remember("a moving world", held)
    recalled = what_she_learned.recall("a moving world", indexed_tables=INDEXED_TABLES)
    back = kept_from(recalled)
    assert 0 < len(back["kinds"]) < 80
    assert 79 in {kind.colour[0] for kind in back["kinds"]}
    for kind in back["kinds"]:
        original = kind.colour[0]
        evidence = back["meeting"].evidence[kind.number]
        assert evidence.touch_sum == -float(original)
        assert evidence.touched == original + 1
        assert set(back["physics"].kinds[kind.number].speeds) == {float(original)}
    assert len(next(tmp_path.glob("*.json")).read_text()) <= 4_000


def test_damaged_legacy_kind_references_cannot_reach_the_fast_controller():
    assert kept_from({"kinds": [{"colour": [1, 2, 3], "size": 10.0}],
                      "evidence": {"9": [0] * 9}, "physics": {"9": {"speeds": [2.0]}}}) == {}
