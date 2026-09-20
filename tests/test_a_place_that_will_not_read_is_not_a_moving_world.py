"""A place that will not read, twice running, is read as unknown.

LIVE 2026-09-20, in a game: "not still after 1.6s: place(s) [(0, 1), (0, 2)]
could not be read", on look after look, and her moves slowed to ten seconds
each. The window had come to rest; two of its places would not read, and
waiting longer never made them.
"""
from __future__ import annotations

from core.perception.what_the_pixels_show import it_has_come_to_rest

SAID = ("4x4", ("2", "4"))


def test_a_world_at_rest_with_nothing_to_wait_for_is_at_rest():
    assert it_has_come_to_rest(
        said=SAID, said_again=SAID, still_places=True, waiting_on=()
    )


def test_a_place_worth_waiting_for_keeps_the_look_going():
    assert not it_has_come_to_rest(
        said=SAID, said_again=SAID, still_places=True, waiting_on=((0, 1),)
    )


def test_a_window_whose_words_changed_is_not_at_rest():
    assert not it_has_come_to_rest(
        said=SAID, said_again=("4x4", ("2", "8")), still_places=True, waiting_on=()
    )


def test_a_place_still_growing_into_its_square_is_not_at_rest():
    assert not it_has_come_to_rest(
        said=SAID, said_again=SAID, still_places=False, waiting_on=()
    )


def test_a_place_that_would_not_read_through_a_look_is_not_waited_on_again(monkeypatch):
    """The live case: the same two places, look after look, 1.6 s each."""
    import time

    from core.perception import what_the_pixels_show as pixels
    from core.perception.what_the_pixels_show import Looker

    from tests.test_the_places_are_seen_not_inferred import _growing

    frames = [_growing(1.0)] * 40
    at = {"n": -1}

    def take():
        at["n"] = min(at["n"] + 1, len(frames) - 1)
        return frames[at["n"]]

    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid_, spots: {})

    looker = Looker()
    began = time.monotonic()
    _picture, _reading, still = pixels.settled_reading(
        take, looker, wait=True, within_s=0.4, began=began
    )
    assert not still
    first_look = time.monotonic() - began
    assert looker.would_not_read, "it remembers what would not read"

    began = time.monotonic()
    _picture, reading, still = pixels.settled_reading(
        take, looker, wait=True, within_s=0.4, began=began
    )
    assert still, "the second look does not wait on them again"
    assert time.monotonic() - began < first_look
    assert pixels.anything_unread(reading), "and they are still carried as unsure"
