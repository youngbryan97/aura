"""A bounded control experiment once fresh unheld motion can be measured.

The first picture often has no motion history. That picture cannot decide
whether there will be motion to counter. Ordinary trials continue while this
experiment waits for an observed neutral interval.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


def _measured_without_a_control(thing: Any, at: float, since: float, began: float) -> bool:
    """Whether a visible track has two fresh positions after the control settled."""
    try:
        measured = float(thing.seen)
        born = float(thing.born)
        values = [float(thing.vx), float(thing.vy), float(thing.w), float(thing.h)]
        places = [(float(t), float(x), float(y)) for t, x, y in thing.path if since <= float(t) <= at]
    # not a failure: a thing whose motion is not numbers is not countered
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in [measured, born, *values]):
        return False
    return (measured == at and born <= began and len(places) >= 2
            and all(math.isfinite(v) for place in places for v in place)
            and places[-1][0] - places[0][0] > 1e-6)


@dataclass
class CounteringMotion:
    """One finite pulse/rest experiment, with its identity and delivered phase."""

    first_at: float | None = None
    search_epoch: int | None = None
    until: float | None = None
    key: str = ""
    epoch: int | None = None
    phase: int = 0
    done: bool = False
    response: Any = None
    phase_seconds: float = 0.0
    last_deadline_began: float | None = None

    def choose(self, keys: Sequence[str], things: Iterable[Any], at: float, *, held: str,
               held_since: float | None, epoch: int, settle_s: float, step_s: float,
               steps: int, wait_s: float, against: Callable[[Sequence[str], Iterable[Any]], str],
               measured: Callable[[Sequence[str], Iterable[Any]], Any] | None = None) -> str | None:
        """The trial's next control; None leaves the ordinary trials available.

        A blank control is a measured rest in an active trial. No measurement
        leaves the search pending until its finite observation budget ends.
        """
        numbers = (at, settle_s, step_s, wait_s)
        if self.done or not all(math.isfinite(value) for value in numbers):
            return None
        if settle_s < 0 or step_s <= 0 or wait_s <= 0 or steps < 2:
            return None
        if self.first_at is None:
            self.first_at = at
            self.search_epoch = epoch
        if epoch != self.search_epoch:
            self.done = True
            return None
        if self.key:
            if held_since is None or not math.isfinite(held_since):
                self.done = True
                return None
            if self.response is not None and not any(thing.number == self.response.number and thing.seen == at for thing in things):
                self.done = True
                return None
            return self._continue(at, held, held_since, epoch, steps)
        if at - self.first_at >= wait_s:
            self.done = True
            return None
        if held_since is None or not math.isfinite(held_since) or held or at - held_since < settle_s:
            return None
        observed = [thing for thing in things if _measured_without_a_control(thing, at, held_since + settle_s, held_since)]
        response = measured(keys, observed) if measured is not None else None
        if response is not None and (response.epoch != epoch or not any(thing.number == response.number for thing in observed)):
            return None
        key = response.key if response is not None else against(keys, observed)
        if not key or key not in keys:
            return None
        self.response = response
        self.key, self.epoch, self.phase = key, epoch, 0
        self.phase_seconds = step_s
        self.until = at + 2 * step_s * steps
        return key

    def _continue(self, at: float, held: str, held_since: float, epoch: int,
                  steps: int) -> str | None:
        expected = self.key if self.phase % 2 == 0 else ""
        if epoch != self.epoch or self.until is None or at >= self.until or held != expected:
            self.done = True
            return None
        if at - held_since < self.phase_seconds:
            return expected
        self.phase += 1
        if self.phase >= steps:
            self.done = True
            return None
        return self.key if self.phase % 2 == 0 else ""

    def released_at_deadline(self, key: str, began: float, ended: float, epoch: int, *, steps: int) -> bool:
        """A leased pulse can finish into its preplanned rest without another picture."""
        if (self.done or not self.key or key != self.key or epoch != self.epoch or self.phase % 2
                or not all(math.isfinite(value) for value in (began, ended, self.phase_seconds))
                or self.phase_seconds <= 0 or ended - began + 1e-9 < self.phase_seconds or steps < 2
                or self.last_deadline_began is not None and began <= self.last_deadline_began):
            return False
        self.last_deadline_began = began
        self.phase += 1
        if self.phase >= steps:
            self.done = True
        return True


@invariant("agency.counter_trials_wait_for_measured_neutral_motion", scope="agency",
           owner="core/agency/countering_motion_as_it_appears.py", observational=False)
def _counter_trial_measurement_invariant() -> tuple:
    trial = CounteringMotion()
    def never(_keys: Any, _things: Any) -> str:
        return ""
    options = dict(held="", held_since=0.0, epoch=1, settle_s=0.2, step_s=0.5,
                   steps=4, wait_s=2.0, against=never)
    assert trial.choose(["a"], [], 0.0, **options) is None
    assert not trial.done
    assert trial.choose(["a"], [], 2.0, **options) is None
    assert trial.done
    return ()
