"""Scenery going by is the view moving, not things coming and going; what moves in it is still found, and found hers.

LIVE 2026-10-09 a game slid its town past three flying heroes: in six seconds 266 things appeared and 242 went, one of
them the whole street, and in a hundred seconds she never found which hero answered to her keys. Nothing here is a
town or a hero: textured layers going by at speeds of their own, parts of a pixel a picture, and one patch the keys move.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.how_the_scenery_goes_by import how_the_view_moved
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit

FPS = 30
WAYS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _layer(seed: int, tall: int, wide: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(40, 220, size=(tall // 8 + 1, wide // 8 + 1, 3)).astype(np.uint8)
    return np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)[:tall, :wide]


def _world(seconds: float, *, keys: list[str] | None = None):
    """Pictures of two layers going by (the far one slower), and a hero the keys move, with the key held at each."""
    far, near = _layer(1, 120, 2400), _layer(2, 120, 2400)
    hero = np.array([250, 40, 200], np.uint8)
    x, y = 120.0, 60.0
    for n in range(int(seconds * FPS)):
        at = n / FPS
        turn = int(at / 0.45)
        key = keys[(turn // 2) % len(keys)] if keys and turn % 2 == 0 else ""
        dx, dy = WAYS.get(key, (0, 0))
        x = min(220.0, max(20.0, x + dx * 90 / FPS))
        y = min(100.0, max(20.0, y + dy * 90 / FPS))
        picture = np.full((240, 320, 3), (250, 160, 60), np.uint8)
        picture[0:120] = far[:, int(at * 25):int(at * 25) + 320]          # sky and hills, going by slowly
        picture[120:240] = near[:, int(at * 70):int(at * 70) + 320]       # the street, faster
        picture[int(y) - 8:int(y) + 8, int(x) - 6:int(x) + 6] = hero
        yield picture, at, key


def test_the_view_goes_by_in_layers_and_the_hero_on_its_own():
    pictures = [p for p, _at, _k in _world(0.2)]
    moved = how_the_view_moved(pictures[0], pictures[3])
    far, near = moved.across[0], moved.across[-1]
    assert far and near and min(near) < min(far) < 0, moved.across      # the near layer goes faster than the far
    assert moved.scenery is not None and moved.scenery.mean() > 0.9
    assert moved.on_its_own is not None and moved.on_its_own[: moved.on_its_own.shape[0] // 2].any()  # the hero


def test_scenery_going_by_is_not_taken_for_things_and_the_one_the_keys_move_is_hers():
    moves, hers = WhatMoves(), WhichIsHers()
    appeared = 0
    known = 0
    held = None
    for picture, at, key in _world(14.0, keys=["up", "down", "left", "right"]):
        if key != held:
            hers.holding(key, at, trying=True)
            held = key
        happened = moves.see(picture, at)
        appeared += sum(1 for h in happened if h.get("what") == "appeared")
        hers.saw(moves, happened, at)
        known += hers.number is not None
    assert appeared < 60, appeared                       # a few slivers at the edges, not the street piece by piece
    assert known > 0, "the patch the keys move was never found to be hers"
    left, right = hers.way_of("left"), hers.way_of("right")
    assert left is not None and right is not None and left[0] < 0 < right[0], (left, right)
