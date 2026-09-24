"""What a place looks like is learned only from a picture of a world at rest.

Her eyes remember each look they have read, so a look that comes round again
is recognised rather than read. They learned from every picture, including
the ones taken while a tile was half way between two places or still growing
into its square, where what was read there is not what that look is. LIVE
2026-09-24: places that "would not read through a whole look" on nearly every
move, on a board a looker with nothing learned read without one unsure place.
"""
from __future__ import annotations

import time

from core.perception import what_the_pixels_show as pixels
from core.perception.what_the_pixels_show import Looker

from tests.test_the_places_are_seen_not_inferred import _growing


def _eyes(monkeypatch, frames, reads_as):
    showing = {"at": -1}

    def take():
        showing["at"] = min(showing["at"] + 1, len(frames) - 1)
        return frames[showing["at"]]

    def strip(self, image, grid_, spots):
        said = reads_as(showing["at"])
        return {(1, 1): said} if said and (1, 1) in spots else {}

    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", strip)
    return take


def test_a_look_misread_while_it_moved_is_not_learned(monkeypatch):
    frames = [_growing(scale) for scale in (0.45, 0.8, 1.0, 1.0, 1.0)]
    # Half grown it reads as a 2; at its full size it is a 256.
    take = _eyes(monkeypatch, frames, lambda at: "2" if at == 0 else ("256" if at >= 2 else ""))
    looker = Looker()
    _picture, reading, still = pixels.settled_reading(
        take, looker, wait=True, within_s=5.0, began=time.monotonic()
    )
    assert still and reading["grids"][0]["says"][1 * 4 + 1] == "256"
    assert looker.seen, "a look at rest is still learned"
    assert {one.says for one in looker.seen} == {"256"}


def test_a_look_that_never_came_to_rest_teaches_nothing(monkeypatch):
    frames = [_growing(scale) for scale in (0.2, 0.45, 0.6, 0.8, 0.9, 1.0)] * 3
    take = _eyes(monkeypatch, frames, lambda at: "2")
    looker = Looker()
    _picture, _reading, still = pixels.settled_reading(
        take, looker, wait=True, within_s=0.3, began=time.monotonic()
    )
    assert not still
    assert not looker.seen


def test_a_reading_that_does_not_wait_learns_as_it_always_did(monkeypatch):
    take = _eyes(monkeypatch, [_growing(1.0)], lambda at: "256")
    looker = Looker()
    pixels.settled_reading(take, looker, wait=False, within_s=1.0, began=time.monotonic())
    assert {one.says for one in looker.seen} == {"256"}


def test_held_lessons_are_learned_only_when_asked(monkeypatch):
    take = _eyes(monkeypatch, [_growing(1.0)], lambda at: "256")
    looker = Looker()
    looker.read(take(), learn=False)
    assert not looker.seen and looker.held_lessons
    looker.learn_what_was_read()
    assert {one.says for one in looker.seen} == {"256"} and not looker.held_lessons
