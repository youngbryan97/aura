"""Give a control enough pictures to show its response.

Offline 2026-10-12 short trials ended with fewer pictures than a push
measurement requires. The picture cadence says how much time to reserve
after the input settles. Each delivered trial keeps its own deadline;
later changes in cadence cannot move that deadline indefinitely.
"""
from __future__ import annotations

import math
from collections.abc import Reversible
from itertools import islice

from core.verify import invariant


def reserved_phase_seconds(observed_s: float, planned_s: float) -> float:
    """A new phase keeps both its observation reservation and its planned minimum."""
    if not all(math.isfinite(value) and value > 0 for value in (observed_s, planned_s)):
        raise ValueError('a control phase requires finite positive observation and plan durations')
    return max(observed_s, planned_s)


@invariant('agency.new_phases_keep_their_observation_reservation', scope='agency',
           owner='core/agency/time_to_observe_a_control.py', observational=False)
def _phase_reservation_invariant() -> tuple:
    assert reserved_phase_seconds(1.18, .734) == 1.18
    assert reserved_phase_seconds(.6, .734) == .734
    return ()


def observed_response_seconds(intervals: Reversible[float], *, least_s: float, settle_s: float,
                              samples: int, span_s: float, longest_s: float) -> float | None:
    """Reserve samples at the slowest recent valid cadence, within the stated limit.

    This estimates time, not whether a body was measured. Absent cadence or
    a demand beyond the limit remains unknown. Current motion, independent
    input receipts and the estimator's sample requirements still decide
    whether the response was learned.
    """
    values = (least_s, settle_s, span_s, longest_s)
    if (not all(math.isfinite(value) for value in values) or least_s <= 0 or settle_s < 0
            or span_s < 0 or samples < 2 or longest_s < least_s):
        return None
    recent = [value for value in islice(reversed(intervals), 12) if math.isfinite(value) and 0 < value <= .35]
    if len(recent) < 3:
        return None
    cadence = max(recent)
    seconds = max(least_s, settle_s + max(samples * cadence, span_s + cadence))
    return seconds if seconds <= longest_s else None


@invariant("agency.control_trials_reserve_their_observation_samples", scope="agency",
           owner="core/agency/time_to_observe_a_control.py", observational=False)
def _observation_time_invariant() -> tuple:
    options = dict(least_s=.45, settle_s=.2, samples=4, span_s=.15, longest_s=1.8)
    assert math.isclose(observed_response_seconds([.1] * 6, **options), .6)
    assert observed_response_seconds([], **options) is None
    assert observed_response_seconds([.5] * 6, **options) is None
    assert observed_response_seconds([.35] * 6, **options) <= 1.8
    assert math.isclose(observed_response_seconds([.1] * 11 + [.3], **options), 1.4)
    return ()
