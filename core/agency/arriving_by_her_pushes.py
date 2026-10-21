"""A carried body needs room to stop before the place it is going to.

Offline 2026-10-12 the same measured pushes that chose a braking key in
one picture could not bring an invented body to rest at its destination.
A fixed long press stopped short or ran past. A feedback path can begin
with a short push and brake as its remaining room shrinks.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from core.verify import invariant

#: Leave part of the measured braking room for error in the measured push.
BRAKING_SHARE = .7


def braking_pulse_holds(pushes: Mapping[str, tuple], step: float, horizon: float) -> tuple:
    """Bounded push, opposing push and neutral candidates at the executable cadence.

    A single control interval can overshoot a small displacement. Two held
    accelerations can move and remove speed before the remaining forecast.
    Every candidate still requires ordinary model scoring and fresh feedback.
    """
    if (not isinstance(pushes, Mapping) or '' not in pushes or len(pushes) > 16
            or not all(type(v) in (int, float) and math.isfinite(v) for v in (step, horizon))
            or not 0 < 2*step < horizon
            or any(not isinstance(key, str) or not isinstance(push, tuple) or len(push) != 2
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in push)
                   for key, push in pushes.items())):
        return ()
    candidates = []
    for first in sorted(key for key in pushes if key):
        for second in sorted(key for key in pushes if key and key != first):
            a, b = pushes[first], pushes[second]
            opposition = a[0]*b[0]+a[1]*b[1]
            if math.isfinite(opposition) and opposition < 0:
                candidates.append(((0., first), (step, second), (2*step, '')))
    return tuple(candidates[:32])


def arrival_key(position: tuple, going: tuple, goal: tuple, pushes: Mapping[str, tuple],
                step: float, held: str = "", *, near: tuple = (0., 0.), gentle: float = 0.) -> str | None:
    """Choose a measured push toward a speed that can be stopped in the room left.

    An axis needs pushes in both directions to promise a stop. Close to the
    goal, leave three control turns for correction. Unknown or nonfinite
    motion cannot supply an arrival path.
    """
    if (not pushes or "" not in pushes or not math.isfinite(step) or step <= 0
            or not all(math.isfinite(v) for v in (*position, *going))
            or not all(math.isfinite(v) for push in pushes.values() for v in push)
            or not all(v is None or math.isfinite(v) for v in goal)
            or not all(math.isfinite(v) and v >= 0 for v in (*near, gentle))):
        return None
    wanted = []
    for axis, target in enumerate(goal):
        if target is None:
            continue
        low, high = min(p[axis] for p in pushes.values()), max(p[axis] for p in pushes.values())
        if low >= 0 or high <= 0:
            continue
        error = target - position[axis]
        braking = -low if error >= 0 else high
        room = max(0., abs(error) - near[axis])
        speed = min(BRAKING_SHARE * math.sqrt(gentle ** 2 + 2 * braking * room), abs(error) / (3 * step))
        wanted.append((axis, math.copysign(speed, error), (high - low) * step))
    if not wanted:
        return None
    choices = sorted(pushes, key=lambda key: key != held)
    return min(choices, key=lambda key: sum(((going[axis] + pushes[key][axis] * step - speed) / span) ** 2
                                           for axis, speed, span in wanted))


def arrival_holds(position: tuple, going: tuple, goal: tuple, pushes: Mapping[str, tuple],
                  times: Sequence[float], control_step: float, held: str, *,
                  near: tuple = (0., 0.), gentle: float = 0.) -> tuple | None:
    """Recompute the braking push once per control turn along a prospective path."""
    if not times or not math.isfinite(control_step) or control_step <= 0:
        return None
    x, y = position
    vx, vy = going
    previous, change_at = 0., 0.
    key, holds = held, []
    for when in times:
        if previous >= change_at - 1e-9:
            chosen = arrival_key((x, y), (vx, vy), goal, pushes, control_step, key, near=near, gentle=gentle)
            if chosen is None:
                return None
            if not holds or chosen != key:
                holds.append((previous, chosen))
            key, change_at = chosen, previous + control_step
        step = float(when) - previous
        ax, ay = pushes[key]
        vx, vy = vx + ax * step, vy + ay * step
        x, y = x + vx * step, y + vy * step
        previous = float(when)
    return tuple(holds)


def stopping_distance(going: tuple, goal: tuple, pushes: Mapping[str, tuple]) -> float:
    """Room still needed to remove the terminal speed along the goal's axes."""
    room = []
    for axis, target in enumerate(goal):
        if target is None or going[axis] == 0:
            continue
        opposing = max((-math.copysign(1., going[axis]) * p[axis] for p in pushes.values()), default=0.)
        if opposing > 0:
            room.append(going[axis] ** 2 / (2 * opposing))
    return math.hypot(*room)


def arrival_excess(position: tuple, going: tuple, goal: tuple, pushes: Mapping[str, tuple],
                   near: tuple, gentle: float) -> float:
    """Speed that cannot be removed before first contact, beyond a plan's horizon."""
    excess = []
    for axis, target in enumerate(goal):
        if target is None:
            continue
        error = target - position[axis]
        direction = math.copysign(1., error if error else going[axis])
        approaching = going[axis] * direction if error else abs(going[axis])
        opposing = max((-direction * p[axis] for p in pushes.values()), default=0.)
        if opposing <= 0 or approaching <= 0:
            continue
        allowed = math.sqrt(gentle ** 2 + 2 * opposing * max(0., abs(error) - near[axis]))
        excess.append(max(0., approaching - allowed))
    return math.hypot(*excess)


@invariant("agency.arrivals_need_measured_braking_room", scope="agency",
           owner="core/agency/arriving_by_her_pushes.py", observational=False)
def _arrival_braking_invariant() -> tuple:
    pushes = {"": (0., 0.), "a": (-100., 0.), "b": (100., 0.)}
    assert arrival_key((100., 100.), (20., 0.), (200., None), {"b": (100., 0.)}, .1) is None
    assert arrival_key((100., 100.), (20., 0.), (None, None), pushes, .1) is None
    assert arrival_key((100., 100.), (math.nan, 0.), (200., None), pushes, .1) is None
    assert arrival_excess((100., 100.), (200., 0.), (200., None), pushes, (24., 24.), 12.) > 70.
    assert arrival_excess((100., 100.), (20., 0.), (200., None), pushes, (24., 24.), 12.) == 0.
    return ()


@invariant('agency.braking_pulses_keep_the_executable_control_cadence', scope='agency',
           owner='core/agency/arriving_by_her_pushes.py', observational=False)
def _braking_pulse_invariant() -> tuple:
    pushes = {'': (0., 0.), 'a': (-80., 0.), 'b': (80., 0.)}
    candidates = braking_pulse_holds(pushes, .1, 1.)
    assert candidates == (((0., 'a'), (.1, 'b'), (.2, '')), ((0., 'b'), (.1, 'a'), (.2, '')))
    assert not braking_pulse_holds({'a': (-80., 0.), 'b': (80., 0.)}, .1, 1.)
    assert not braking_pulse_holds(pushes, .1, .2)
    return ()
