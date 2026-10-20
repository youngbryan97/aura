"""Retain the predicates that actually chose a motion family.

These are the existing classifier's summaries. They do not establish that
its two statistics are comparable, or that its selected family is correct.
An unvisited predicate remains unavailable in the record.
"""
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class MotionFamilyReading:
    accelerated: bool
    reason: str
    push_statistic: float
    push_effect: float
    pace_statistic: float | None = None
    stops_on_release: bool | None = None
    prediction: Any = None


def read_motion_family(speeds: Any, *, statistic_over: float, effect_over: float) -> MotionFamilyReading:
    """Retain the initial gates and prefer qualified forward evidence when available."""
    from dataclasses import replace

    reading = _initial_motion_family(speeds, statistic_over=statistic_over, effect_over=effect_over)
    models = getattr(getattr(speeds, 'intervals', None), 'families', None)
    if models is None:
        return reading
    current_statistic, current_effect = speeds.intervals.contrast(owner=models.owner) if models.owner is not None else (0., 0.)
    qualified = current_statistic >= statistic_over and current_effect >= effect_over
    prediction = models.read(admitted_acceleration=reading.accelerated, qualified_push=qualified)
    if prediction.selected is None:
        if models.owner is not None and reading.accelerated and not qualified:
            return replace(reading, accelerated=False, reason='current_material_push_unresolved', prediction=prediction)
        return replace(reading, prediction=prediction)
    return replace(reading, accelerated=prediction.selected, reason=prediction.reason, prediction=prediction)


def _initial_motion_family(speeds: Any, *, statistic_over: float, effect_over: float) -> MotionFamilyReading:
    """The original ordered predicates remain the initial admission evidence."""
    statistic, effect = speeds.by_push()
    if not statistic >= statistic_over:
        return MotionFamilyReading(False, 'push_statistic_unresolved', statistic, effect)
    if not effect >= effect_over:
        return MotionFamilyReading(False, 'push_effect_unresolved', statistic, effect)
    pace = speeds._by_pace_alone()
    if not statistic > pace:
        return MotionFamilyReading(False, 'pace_statistic_wins', statistic, effect, pace)
    stops = speeds._stops_when_let_go()
    return MotionFamilyReading(not stops, 'stops_on_release' if stops else 'push_statistic_wins',
                               statistic, effect, pace, stops)


@invariant('agency.motion_family_records_only_its_visited_predicates', scope='agency',
           owner='core/agency/why_a_motion_history_is_called_carried.py', observational=False)
def _motion_family_reading_invariant() -> tuple:
    from types import SimpleNamespace

    speeds = SimpleNamespace(by_push=lambda: (30., 40.), _by_pace_alone=lambda: 50.,
                             _stops_when_let_go=lambda: (_ for _ in ()).throw(AssertionError('unvisited')))
    reading = read_motion_family(speeds, statistic_over=20., effect_over=20.)
    assert not reading.accelerated and reading.reason == 'pace_statistic_wins'
    assert reading.pace_statistic == 50. and reading.stops_on_release is None
    return ()
