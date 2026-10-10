"""A thing that heads for the pointer at its own pace follows it: a boat eased toward the mouse, never under it.

LIVE 2026-10-10 "Nothing here follows the mouse" of a boat the mouse steered, and its game was lost to GAME OVER.
"""
from __future__ import annotations

import math

import pytest

from core.agency.which_one_answers_to_her import _chases_along, _measured_along

pytestmark = pytest.mark.unit


def _sweep(n: int = 60) -> list[float]:
    """The pointer swept back and forth across the picture."""
    return [120 + 90 * math.sin(k / 6.0) for k in range(n)]


def test_a_boat_eased_toward_the_pointer_heads_for_it_though_it_is_never_under_it():
    pointer, boat, pairs = _sweep(), 120.0, []
    for x in pointer:
        boat += 0.12 * (x - boat)                       # eases a little of the way each picture
        pairs.append((x, 80.0, boat, 150.0))           # and keeps to its lane, low down
    assert _measured_along(pairs, (240.0, 160.0))[0] is not True    # not under the pointer: the old test misses it
    assert _chases_along(pairs)[0] is True


def test_a_thing_going_its_own_way_while_the_pointer_is_swept_does_not_head_for_it():
    pointer = _sweep()
    pairs = [(x, 80.0, 120 + 80 * math.sin(k / 2.3 + 1.0), 40.0) for k, x in enumerate(pointer)]
    assert _chases_along(pairs)[0] is not True
    still = [(x, 80.0, 60.0, 40.0) for x in pointer]
    assert _chases_along(still) == (None, None)
