"""Rank repeated control experiments by their measured visible paths.

A single pulse may suggest which experiment has more room. It does not
establish a response or change causal identity. The caller still owns the
finite per-control allowance and the physical input deadline.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

from core.agency.a_response_worth_measuring_again import _push
from core.agency.calibration_for_a_current_body import MAX_CALIBRATION_CONTROLS
from core.agency.keeping_a_trial_observable import visible_path_clearance
from core.perception.translation_from_visible_support import motion_was_measured
from core.verify import invariant


def _push_envelope(record: Any, key: str, born: float, at: float, history_s: float) -> tuple | None:
    measured = []
    for press, samples in record.timed.items():
        name, began = press
        if (name != key or not born <= began <= at or press in record.interrupted
                or not samples or not at - history_s <= samples[-1][0] <= at):
            continue
        estimate = _push(samples)
        if estimate is not None:
            measured.append(estimate)
    if not measured:
        return None
    return tuple((min(push[axis] - error for push, error in measured),
                  max(push[axis] + error for push, error in measured)) for axis in (0, 1))


def _corners(envelope: tuple) -> tuple:
    return tuple(set(itertools.product(*envelope)))


def trials_with_room(keys: Sequence[str], hers: Any, moves: Any, at: float, seconds: float) -> list[str]:
    """Prefer the observed pulse/rest path with more clearance when all candidates are measured.

    Fit residuals and observed variation widen each acceleration interval.
    These projections rank experiments; they do not guarantee future motion.
    Absent, stale or non-current evidence preserves the caller's order.
    """
    order = list(keys)
    if not 2 <= len(order) <= MAX_CALIBRATION_CONTROLS or not getattr(hers, 'carried', False):
        return order
    mine = hers.thing(moves)
    if (mine is None or mine.seen != at or not motion_was_measured(mine, at)
            or not all(math.isfinite(v) for v in (at, seconds, mine.born, mine.x, mine.y,
                                                 mine.vx, mine.vy, mine.w, mine.h, *moves.shape))
            or seconds <= 0 or min(mine.w, mine.h, *moves.shape) <= 0):
        return order
    record = hers._hers
    if record is None:
        return order
    history_s = 8 * len(order) * seconds
    rest = _push_envelope(record, '', mine.born, at, history_s)
    estimates = {key: _push_envelope(record, key, mine.born, at, history_s) for key in order}
    if rest is None or any(value is None for value in estimates.values()):
        return order
    clearance = {key: min(visible_path_clearance(mine, moves.shape, [(seconds, pulse), (seconds, neutral)])
                         for pulse in _corners(estimates[key]) for neutral in _corners(rest)) for key in order}
    if not all(math.isfinite(value) for value in clearance.values()):
        return order
    return sorted(order, key=lambda key: -clearance[key])


@invariant('agency.trial_priority_cannot_certify_a_response', scope='agency',
           owner='core/agency/trials_with_room_to_observe.py', observational=False)
def _trial_priority_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    hers.number, hers.kind, hers.carried = 1, 1, True
    track = SimpleNamespace(number=1, kind=1, born=0., seen=10., x=60., y=120.,
                            vx=-40., vy=0., w=14., h=14., motion_evidence='direct')
    moves = SimpleNamespace(things={1: track}, shape=(240, 240))
    record = hers._hers
    for key, began, push in [('q', 1., -110.), ('r', 3., 110.), ('', 5., 0.)]:
        for dt in (.2, .3, .4, .5):
            record.add(key, dt * push, 0., press=began, when=began + dt)
    assert trials_with_room(['q', 'r'], hers, moves, 10., .65) == ['r', 'q']
    assert hers.push_of('r') is None
    assert hers.number == 1 and not hers.identification.receipts
    return ()
