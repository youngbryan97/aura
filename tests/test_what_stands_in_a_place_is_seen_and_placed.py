"""What stands still in a place is seen by her eyes and placed by the picture: her eyes say what is there, roughly
where; what stands out from its surroundings says exactly where. Nothing here is any one game."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.what_is_in_a_place import Landmark, placed

pytestmark = pytest.mark.unit


def _hall(doors: list[int], her_at: int | None = None) -> np.ndarray:
    """A wall over a floor, doors standing on the floor (their left edges, of 480), and a figure where she stands."""
    picture = np.zeros((320, 480, 3), np.uint8)
    picture[:] = (201, 180, 138)
    picture[280:] = (107, 74, 43)
    for x in doors:
        picture[225:280, x - 3:x + 33] = (59, 36, 18)
        picture[228:280, x:x + 30] = (139, 90, 43)
    if her_at is not None:
        picture[256:280, her_at:her_at + 14] = (47, 111, 214)
    return picture


@pytest.mark.parametrize(("doors", "drawn", "her_at"), [
    ([120, 236, 386], [(0.382, 0.468), (0.582, 0.668), (0.782, 0.868)], None),   # evenly spaced, as her eyes drew them
    ([114, 255, 355], [(0.125, 0.21), (0.382, 0.468), (0.625, 0.71)], 259),      # two close together, she by one
])
def test_a_row_of_doors_drawn_roughly_is_placed_on_the_doors(doors, drawn, her_at):
    seen = [Landmark("doors", (left, 0.65, right, 0.85)) for left, right in drawn]
    lefts = sorted(round(m.box[0] * 480) for m in placed(_hall(doors, her_at), seen))
    assert lefts == pytest.approx([x - 3 for x in doors], abs=4)


def test_a_thing_with_nothing_like_it_near_is_kept_as_her_eyes_drew_it():
    seen = [Landmark("ladder", (0.1, 0.1, 0.14, 0.6))]
    assert placed(_hall([236]), seen) == seen
