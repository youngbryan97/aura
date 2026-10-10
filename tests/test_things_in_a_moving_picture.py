"""Things in a moving picture are followed, found when still, and never made of floor."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit

FLOOR = (20, 60, 40)


def _picture(things: list[tuple[int, int, int, int, tuple[int, int, int]]]) -> np.ndarray:
    picture = np.zeros((120, 200, 3), dtype=np.uint8)
    picture[:, :] = FLOOR
    for x, y, w, h, colour in things:
        picture[y : y + h, x : x + w] = colour
    return picture


def _watch(frames: list[np.ndarray], every: float = 0.03) -> tuple[WhatMoves, list[dict]]:
    moves = WhatMoves()
    happened = []
    for index, frame in enumerate(frames):
        happened.extend(moves.see(frame, index * every))
    return moves, happened


def test_a_thing_crossing_the_picture_keeps_its_name_and_its_way():
    frames = [_picture([(10 + 3 * n, 50, 6, 6, (250, 250, 250))]) for n in range(50)]
    moves, happened = _watch(frames)
    moving = [t for t in moves.things.values() if t.moved]
    assert len(moving) == 1
    assert moving[0].vx > 50 and abs(moving[0].vy) < 10
    assert sum(1 for h in happened if h["what"] == "appeared") == 1


def test_a_thing_that_never_moves_is_seen_from_the_start():
    """A coin that sits still through the first look is still a coin."""
    frames = [_picture([(100, 60, 6, 6, (240, 210, 20))]) for _ in range(30)]
    moves, _ = _watch(frames)
    assert [t.colour for t in moves.things.values()] == [(240, 210, 20)]


def test_where_a_still_thing_was_is_floor_once_it_leaves():
    """The paddle stood still during the first look; when it moves, the place it left is not a thing."""
    still = [_picture([(10, 40, 4, 25, (230, 230, 230))]) for _ in range(20)]
    leaving = [_picture([(10, 40 + 2 * n, 4, 25, (230, 230, 230))]) for n in range(1, 30)]
    moves, _ = _watch(still + leaving)
    assert len(moves.things) == 1
    only = next(iter(moves.things.values()))
    assert only.colour == (230, 230, 230)


def test_a_thing_that_stops_is_still_there():
    going = [_picture([(10 + 2 * n, 40, 6, 6, (250, 80, 80))]) for n in range(30)]
    stopped = [_picture([(68, 40, 6, 6, (250, 80, 80))]) for _ in range(80)]
    moves, happened = _watch(going + stopped)
    assert len(moves.things) == 1
    assert not any(h["what"] == "gone" for h in happened)


def test_a_new_screen_starts_again():
    first = [_picture([(10 + 2 * n, 40, 6, 6, (250, 80, 80))]) for n in range(20)]
    other = np.full((120, 200, 3), 200, dtype=np.uint8)
    moves, happened = _watch(first + [other] * 10)
    assert any(h["what"] == "new screen" for h in happened)
    assert not moves.things


def test_a_thing_lost_for_a_picture_is_where_its_speed_took_it():
    """A ball missing from one picture (drawn into another thing, or not drawn) goes on, not stays put."""
    frames = [_picture([(10 + 3 * n, 50, 6, 6, (250, 250, 250))]) for n in range(30)]
    frames.append(_picture([]))
    moves = WhatMoves()
    for index, frame in enumerate(frames):
        moves.see(frame, index * 0.03)
    [ball] = [t for t in moves.things.values() if t.moved]
    assert ball.seen < 30 * 0.03
    last_seen_x = ball.path[-1][1]
    assert ball.x > last_seen_x + 0.02 * ball.vx


def test_a_kind_kept_from_another_way_of_looking_does_not_stop_play():
    """A kept kind whose colour mix has another number of bins starts its mix again."""
    from core.perception.what_moves_in_the_picture import Kind

    old = Kind(0, np.full(32, 1 / 32), 36.0, (250, 250, 250))
    moves = WhatMoves(kinds=[old])
    for index in range(20):
        moves.see(_picture([(10 + 3 * index, 50, 6, 6, (250, 250, 250))]), index * 0.03)
    assert moves.kinds[0].look.shape == (64,)


def test_an_equal_area_object_with_another_shape_does_not_inherit_a_controls_identity():
    moves, _ = _watch([_picture([(10, 20 + n, 4, 24, (230, 230, 230))]) for n in range(35)])
    [control] = [t for t in moves.things.values() if t.moved]
    # Replace it in place with the same colour and area, turned sideways.
    replacement = _picture([(0, 58, 24, 4, (230, 230, 230))])
    for n in range(20):
        moves.see(replacement, (35 + n) * 0.03)
    assert control.number not in moves.things


def test_a_thing_that_goes_behind_something_and_comes_out_is_the_same_thing():
    """A figure walking behind a pillar and out the other side is the one that went in, not a new one."""
    pillar = (90, 20, 30, 80, (90, 90, 90))
    def at(n: int) -> list:
        x = 10 + 3 * n
        hidden = 84 <= x <= 120                                  # behind the pillar: not drawn
        return [pillar] if hidden else [pillar, (x, 50, 6, 6, (250, 250, 250))]
    moves, happened = _watch([_picture(at(n)) for n in range(55)])
    walker = [t for t in moves.things.values() if t.colour == (250, 250, 250)]
    assert len(walker) == 1 and walker[0].again and walker[0].vx > 50
    first = next(h["thing"] for h in happened if h["what"] == "appeared" and h["thing"] == walker[0].number)
    assert first == walker[0].number
    assert any(h["what"] == "appeared" and h["again"] for h in happened)


def test_a_thing_of_another_kind_coming_into_sight_is_not_the_one_that_went():
    going = [_picture([(10 + 3 * n, 50, 6, 6, (250, 250, 250))]) for n in range(20)]
    gap = [_picture([]) for _ in range(15)]
    other = [_picture([(120 + 3 * n, 50, 6, 6, (40, 40, 250))]) for n in range(15)]
    moves, _ = _watch(going + gap + other)
    assert not any(t.again for t in moves.things.values())
