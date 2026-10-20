"""Keep enough measured headroom for a control trial and its braking path.

The picture's edges bound observation. A predicted crossing leaves that
measurement unavailable; coordinates are never clamped into a supposed wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class ControlForObservation:
    key: str
    number: int
    epoch: int
    axis: int
    wait_s: float


def visible_path_clearance(thing: Any, shape: tuple[int, int], segments: Any) -> float:
    """Minimum visible margin along accelerated segments, including turning points."""
    position, velocity = [float(thing.x), float(thing.y)], [float(thing.vx), float(thing.vy)]
    lower = [float(thing.w) / 2, float(thing.h) / 2]
    upper = [shape[1] - lower[0], shape[0] - lower[1]]
    clearance = min(*(position[a] - lower[a] for a in (0, 1)), *(upper[a] - position[a] for a in (0, 1)))
    for seconds, push in segments:
        for axis in (0, 1):
            points = [0.0, seconds]
            if abs(push[axis]) > 1e-9:
                turns = -velocity[axis] / push[axis]
                if 0 < turns < seconds:
                    points.append(turns)
            for when in points:
                where = position[axis] + velocity[axis] * when + push[axis] * when * when / 2
                clearance = min(clearance, where - lower[axis], upper[axis] - where)
            position[axis] += velocity[axis] * seconds + push[axis] * seconds * seconds / 2
            velocity[axis] += push[axis] * seconds
    return clearance


def _crossing(position: float, velocity: float, push: float, edge: float) -> float | None:
    delta = position - edge
    if abs(push) < 1e-9:
        return -delta / velocity if abs(velocity) > 1e-9 and -delta / velocity > 0 else None
    discriminant = velocity * velocity - 2 * push * delta
    if discriminant < 0:
        return None
    roots = [(-velocity + sign * math.sqrt(discriminant)) / push for sign in (-1, 1)]
    future = [value for value in roots if value > 1e-9]
    return min(future) if future else None


def control_before_the_next_trial(hers: Any, moves: Any, at: float, keys: Any, wait_s: float, *,
                                  measured_rest: tuple | None = None) -> ControlForObservation | None:
    """A known control needed now because waiting then braking would lose the track."""
    mine = hers.thing(moves)
    rest = measured_rest if measured_rest is not None else hers.push_of("") if hers.carried else None
    if mine is None or mine.seen != at or rest is None or wait_s <= 0:
        return None
    values = (mine.x, mine.y, mine.w, mine.h, mine.vx, mine.vy, *rest, at, wait_s, *moves.shape)
    if not all(math.isfinite(value) for value in values) or min(mine.w, mine.h, *moves.shape) <= 0:
        return None
    going, position = (mine.vx, mine.vy), (mine.x, mine.y)
    size, limits = (mine.w, mine.h), (moves.shape[1], moves.shape[0])
    crossing = []
    for axis in (0, 1):
        for direction, edge in ((-1, size[axis] / 2), (1, limits[axis] - size[axis] / 2)):
            when = _crossing(position[axis], going[axis], rest[axis], edge)
            if when is not None:
                crossing.append((when, axis, direction))
    if not crossing:
        return None
    _when, axis, direction = min(crossing)
    controls = {key: push for key in ["", *keys] if (push := hers.push_of(key)) is not None
                and all(math.isfinite(value) for value in push) and direction * push[axis] < -1e-9}
    if not controls:
        return None
    choices = []
    for key, push in controls.items():
        after_wait = direction * (going[axis] + rest[axis] * wait_s)
        brake_after = max(0., after_wait) / (-direction * push[axis])
        if visible_path_clearance(mine, moves.shape, [(wait_s, rest), (brake_after, push)]) > 0:
            return None
        brake_now = max(0., direction * going[axis]) / (-direction * push[axis])
        clearance = visible_path_clearance(mine, moves.shape, [(max(wait_s, brake_now), push)])
        if clearance > 0:
            choices.append((clearance, key))
    if not choices:
        return None
    return ControlForObservation(max(choices)[1], mine.number, hers.identification.epoch, axis, wait_s)


@invariant("agency.trial_paths_do_not_clamp_into_the_picture", scope="agency",
           owner="core/agency/keeping_a_trial_observable.py", observational=False)
def _unclamped_trial_path_invariant() -> tuple:
    from types import SimpleNamespace

    track = SimpleNamespace(x=50., y=90., vx=0., vy=40., w=10., h=10.)
    assert visible_path_clearance(track, (100, 100), [(.5, (0., 20.))]) < 0
    assert visible_path_clearance(track, (100, 100), [(.5, (0., -200.))]) > 0
    return ()


@invariant("agency.observation_viability_uses_measured_control", scope="agency",
           owner="core/agency/keeping_a_trial_observable.py", observational=False)
def _observed_viability_invariant() -> tuple:
    from types import SimpleNamespace

    track = SimpleNamespace(number=1, seen=1., x=120., y=170., vx=0., vy=100., w=14., h=14.)
    pushes = {"": (0., 90.), "a": (0., -110.)}
    hers = SimpleNamespace(carried=True, thing=lambda _moves: track, push_of=pushes.get,
                            identification=SimpleNamespace(epoch=2))
    moves = SimpleNamespace(shape=(240, 240))
    choice = control_before_the_next_trial(hers, moves, 1., ["a"], .65)
    assert choice is not None and choice.key == "a" and choice.number == 1
    pushes.pop("")
    assert control_before_the_next_trial(hers, moves, 1., ["a"], .65) is None
    return ()
