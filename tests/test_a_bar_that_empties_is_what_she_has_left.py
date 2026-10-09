"""A bar that fills and empties is read as it goes: a fall of what she has left is a loss, and low, she is careful.

LIVE 2026-10-09 three heroes shared a health bar and she played as if nothing could hurt her; another game drained a
bucket of paint at every knock, and she stood still while it emptied. Nothing here is either game: a strip on a
panel that shortens in steps, a plain strip that never changes, a band of sky, and a thing crossing the picture.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.how_full_a_bar_is import BarsOnTheScreen

pytestmark = pytest.mark.unit


def _picture(left_of_bar: int, thing_x: int) -> np.ndarray:
    picture = np.full((240, 320, 3), (250, 160, 60), np.uint8)       # sky, running off both sides
    picture[150:240] = (60, 150, 60)                                  # ground
    picture[10:24, 20:140] = (90, 30, 60)                             # the bar's track
    picture[13:21, 22:22 + left_of_bar] = (255, 200, 210)             # the bar, emptying from its right end
    picture[40:48, 200:300] = (30, 30, 200)                           # a plain strip that never changes
    picture[100:130, thing_x:thing_x + 20] = (200, 30, 30)            # something crossing
    return picture


def _played(levels):
    bars, changes = BarsOnTheScreen(), []
    at = 0.0
    for level in levels:
        for _ in range(6):
            at += 0.1
            changes += bars.read(_picture(level, int(at * 40) % 280), at)
    return bars, changes


def test_a_strip_that_shortens_in_steps_and_holds_between_is_a_bar_and_nothing_else_is():
    bars, changes = _played([116, 116, 100, 100, 84, 84, 60, 60, 40])
    found = bars.bars()
    assert len(found) == 1 and found[0].anchored == "start", [(b.row, b.start, b.end) for b in bars.strips]
    # Three changes, each after the level held, make it a bar; from then on each change is told.
    assert [round(c["to"], 2) for c in changes] == [round(60 / 116, 2), round(40 / 116, 2)]
    assert all(c["to"] < c["from"] for c in changes)


def test_a_fall_in_a_bar_named_health_is_a_loss_and_low_she_is_careful():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Choosing, _read_the_bars, _Run
    from core.agency.what_meeting_things_does import WhatMeetingDoes

    run = _Run(keys=["left"], began=0.0, last_moving=0.0)
    run.regions_read = [{"text": "HEALTH", "center_x": 0.03, "center_y": 0.07}]
    meeting = WhatMeetingDoes()
    moves = SimpleNamespace(shape=(160, 240), things={}, share=lambda x, y: (x / 240, y / 160))
    hers = SimpleNamespace(thing=lambda _moves: None)
    said = []
    at = 0.0
    for level in [116, 116, 100, 100, 70, 70, 40, 40, 20]:
        for _ in range(6):
            at += 0.1
            _read_the_bars(run, meeting, hers, moves, _picture(level, 10), at, said.append)
    assert run.losses >= 2 and [v["what"] for v in meeting.verdicts] == ["loss"] * len(meeting.verdicts)
    assert run.vitals < 0.25 and meeting.vitals == run.vitals
    assert said and "HEALTH bar" in said[0]
    choosing = _Choosing(SimpleNamespace(things={}), SimpleNamespace(thing=lambda _m: None, keys_that_move_her=lambda _k: {},
                                                                     follows_pointer=False, axes=lambda _k: (False, False),
                                                                     makes={}), meeting, [])
    assert choosing.caution > 2.0
