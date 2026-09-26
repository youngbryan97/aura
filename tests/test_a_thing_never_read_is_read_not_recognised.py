"""A look near one thing she knows may be a thing she has never read.

Measured on the 2048 app, 26 Sep: a 256 sits 8.4 Lab units from a 128, inside
the 9.0 one look was allowed, while the same number at two places differs by
at most 2.5. With only 128s learned, the first 256 of a game was recognised
as a 128, and her rule for the game, which had it right, was scored as wrong:
"her rule missed on left: (2,1) said '256' saw '128'".
"""
from __future__ import annotations

import numpy as np

from core.perception.what_the_pixels_show import Looker


def _a_look(shade: float) -> np.ndarray:
    """A small picture every sample of which sits ``shade`` Lab units from grey 50."""
    look = np.full((24, 24, 3), 50.0)
    look[..., 0] += shade
    return look


def test_a_look_as_far_from_a_thing_as_a_new_thing_sits_is_read():
    looker = Looker()
    looker.learned(_a_look(0.0), "128")
    looker.learned(_a_look(2.5), "128")  # the same number at another place
    assert looker.recognised(_a_look(1.0)) == "128"
    assert looker.recognised(_a_look(8.4)) is None, "a 256 is not a 128 because nothing else is nearer"


def test_nothing_is_recognised_but_its_own_look_before_one_thing_is_read_twice():
    looker = Looker()
    looker.learned(_a_look(0.0), "2")
    assert looker.how_far_one_thing_varies() == 0.0
    assert looker.recognised(_a_look(0.0)) == "2"
    assert looker.recognised(_a_look(1.0)) is None


def test_how_far_one_thing_varies_is_learned_from_every_thing():
    looker = Looker()
    looker.learned(_a_look(0.0), "2")
    looker.learned(_a_look(1.2), "2")
    looker.learned(_a_look(30.0), "32")
    looker.learned(_a_look(32.5), "32")
    assert looker.how_far_one_thing_varies() == 2.5
    # A 2 seen as far from itself as a 32 has been is still a 2.
    assert looker.recognised(_a_look(2.0)) == "2"
