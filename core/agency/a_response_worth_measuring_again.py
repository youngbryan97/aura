"""Repeat a measured response when ordinary exploration would lose its track.

A first pulse and rest can suggest an input worth testing again. This chooses
that next experiment from their acceleration measurements. It gives no body
identity: another independent pulse must establish whether the response repeats.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.agency.keeping_a_trial_observable import visible_path_clearance
from core.verify import invariant


@dataclass(frozen=True)
class ResponseToRepeat:
    """A pulse and its following rest on one observed track in one experiment."""

    number: int
    epoch: int
    key: str
    pulse_at: float
    rest_at: float
    pulse_push: tuple[float, float]
    rest_push: tuple[float, float]
    pulse_error: float
    rest_error: float


def _push(samples: Any) -> tuple[tuple[float, float], float] | None:
    if len(samples) < 4:
        return None
    try:
        samples = [tuple(float(value) for value in sample) for sample in samples]
    # not a failure: samples that are not numbers give none
    except (TypeError, ValueError):
        return None
    if any(len(sample) != 3 or not all(math.isfinite(value) for value in sample) for sample in samples):
        return None
    times = [sample[0] for sample in samples]
    span = times[-1] - times[0]
    if span < 0.15 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        return None
    middle = statistics.fmean(times)
    spread = sum((at - middle) ** 2 for at in times)
    average = [statistics.fmean(sample[axis] for sample in samples) for axis in (1, 2)]
    push = tuple(sum((sample[0] - middle) * sample[axis] for sample in samples) / spread for axis in (1, 2))
    residual = max(math.hypot(*(sample[axis + 1] - average[axis] - push[axis] * (sample[0] - middle)
                               for axis in (0, 1))) for sample in samples)
    return (push[0], push[1]), 2 * residual / span


def _clearance(thing: Any, shape: tuple[int, int], pushes: Sequence[tuple[float, float]], seconds: float) -> float:
    """The least visible clearance along the measured accelerated path."""
    return visible_path_clearance(thing, shape, [(seconds, push) for push in pushes])


def response_worth_repeating(keys: Sequence[str], things: Iterable[Any], hers: Any, shape: tuple[int, int],
                            at: float, neutral_since: float | None, *, seconds: float, steps: int,
                            effect_over: float) -> ResponseToRepeat | None:
    """A fresh pulse/rest response that keeps one track in sight through another test.

    Unknown rest, stale samples and competing responding tracks leave this
    experiment unchosen. A suggested response never changes causal identity.
    """
    if neutral_since is None or steps < 2 or seconds <= 0:
        return None
    if not all(math.isfinite(value) for value in (at, neutral_since, seconds, effect_over)):
        return None
    candidates = []
    epoch = hers.identification.epoch
    for thing in things:
        record = hers._by_thing.get(thing.number)
        if record is None or thing.seen != at:
            continue
        rest = record.timed.get(("", neutral_since))
        if not rest or rest[-1][0] != at:
            continue
        neutral = _push(rest)
        if neutral is None:
            continue
        neutral_push, neutral_error = neutral
        if _clearance(thing, shape, [neutral_push] * steps, seconds) >= 0:
            continue
        for (key, began), samples in record.timed.items():
            if key not in keys or began >= neutral_since or thing.born > began or not samples:
                continue
            if samples[-1][0] > neutral_since or neutral_since - samples[-1][0] > seconds:
                continue
            pulse = _push(samples)
            if pulse is None:
                continue
            pulse_push, pulse_error = pulse
            if math.dist(pulse_push, neutral_push) <= effect_over + pulse_error + neutral_error:
                continue
            path = [pulse_push if phase % 2 == 0 else neutral_push for phase in range(steps)]
            clearance = _clearance(thing, shape, path, seconds)
            if clearance > 0:
                witness = ResponseToRepeat(thing.number, epoch, key, began, neutral_since, pulse_push, neutral_push,
                                           pulse_error, neutral_error)
                candidates.append((clearance, witness))
    if len({witness.number for _clearance, witness in candidates}) != 1:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


@invariant("agency.provisional_responses_do_not_establish_identity", scope="agency",
           owner="core/agency/a_response_worth_measuring_again.py", observational=False)
def _provisional_response_invariant() -> tuple:
    from types import SimpleNamespace

    pulse = [(t, 0.0, 40.0 - 110.0 * t) for t in (.2, .25, .3, .35, .4)]
    rest = [(t, 0.0, 40.0 + 90.0 * (t - .5)) for t in (.7, .75, .8, .85, .9)]
    track = dict(seen=.9, born=-1., x=120., y=190., vx=0., vy=40., w=12., h=12.)
    first, second = SimpleNamespace(number=1, **track), SimpleNamespace(number=2, **track)
    record = SimpleNamespace(timed={("a", .0): pulse, ("", .5): rest})
    ledger = SimpleNamespace(epoch=2, receipts=[])
    hers = SimpleNamespace(number=None, identification=ledger, _by_thing={1: record})
    arguments = dict(seconds=.45, steps=4, effect_over=40.)
    witness = response_worth_repeating(["a"], [first], hers, (240, 240), .9, .5, **arguments)
    assert witness is not None and witness.number == 1 and witness.epoch == 2
    assert hers.number is None and ledger.receipts == []
    assert response_worth_repeating(["a"], [first], hers, (240, 240), .9, None, **arguments) is None
    hers._by_thing[2] = record
    assert response_worth_repeating(["a"], [first, second], hers, (240, 240), .9, .5, **arguments) is None
    assert hers.number is None and ledger.receipts == []
    return ()
