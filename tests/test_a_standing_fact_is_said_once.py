"""Saying a true thing 436 times is not 436 times as useful.

LIVE, 2026-08-18. "Disk is 92% full. Worth cleaning up before it causes
problems." had been fired 436 times — 265 at 92%, 109 at 93%, 54 at 94%, 8 at
99%. Every reading was true when it was taken, so nothing was wrong with the
measurement. The same standing fact was simply re-announced every cycle it
remained true.

The daily initiation budget did not prevent it, because the budget counts
INITIATIONS and not SUBJECTS. One unchanging condition spent the whole
allowance, and everything else she might have raised was crowded out by it. A
cap on how often she speaks is not a cap on how often she repeats herself.
"""

from __future__ import annotations

import threading

import pytest

from core.fictional.jarvis import ProactiveAnticipationEngine


@pytest.fixture()
def engine() -> ProactiveAnticipationEngine:
    made = ProactiveAnticipationEngine.__new__(ProactiveAnticipationEngine)
    made._lock = threading.RLock()
    made._condition_last_value = {}
    return made


def _announced(engine, key, readings):
    return [v for v in readings if engine._condition_is_worth_restating(key, v)]


def test_the_live_sequence_collapses_to_what_was_news(engine):
    """The real readings, in the order they were taken."""
    readings = [92.0] * 3 + [93.0] * 2 + [94.0] * 2 + [99.0] * 2
    assert _announced(engine, "disk_pressure", readings) == [92.0, 99.0]


def test_an_unchanged_condition_is_not_restated(engine):
    assert engine._condition_is_worth_restating("disk_pressure", 92.0)
    for _ in range(50):
        assert not engine._condition_is_worth_restating("disk_pressure", 92.0)


def test_a_material_worsening_is_still_worth_hearing(engine):
    """Quiet is not the same as silent — it must still escalate."""
    assert engine._condition_is_worth_restating("disk_pressure", 92.0)
    assert not engine._condition_is_worth_restating("disk_pressure", 94.0)
    assert engine._condition_is_worth_restating("disk_pressure", 99.0)


def test_a_cleared_condition_re_arms(engine):
    """The point is to say a thing once while it is true, not once ever."""
    assert engine._condition_is_worth_restating("disk_pressure", 92.0)
    assert not engine._condition_is_worth_restating("disk_pressure", 92.0)
    engine.note_condition_cleared("disk_pressure")
    assert engine._condition_is_worth_restating("disk_pressure", 92.0)


def test_conditions_do_not_shadow_each_other(engine):
    """Keyed on the condition, so a busy disk does not silence a hot CPU."""
    assert engine._condition_is_worth_restating("cpu_pressure", 95.0)
    assert engine._condition_is_worth_restating("memory_pressure", 95.0)
    assert engine._condition_is_worth_restating("disk_pressure", 95.0)


def test_rewording_does_not_restart_the_nagging(engine):
    """Keyed on the condition rather than the sentence.

    Keying on message text would let any change of phrasing — a different
    percentage inside the same sentence — count as something new, which is
    precisely how 436 near-identical sentences got through.
    """
    assert engine._condition_is_worth_restating("disk_pressure", 92.0)
    assert not engine._condition_is_worth_restating("disk_pressure", 93.0)
    assert not engine._condition_is_worth_restating("disk_pressure", 94.0)


def test_an_unkeyed_alert_is_never_suppressed(engine):
    """One-off observations are not standing conditions and must still fire."""
    for _ in range(5):
        assert engine._condition_is_worth_restating("", None)


def test_the_standing_alerts_are_keyed():
    """The mechanism is worth nothing if the alerts do not use it."""
    import inspect

    source = inspect.getsource(ProactiveAnticipationEngine._check_system_anomalies)
    for key in ("cpu_pressure", "memory_pressure", "disk_pressure"):
        assert f'condition="{key}"' in source, key


def test_a_repeat_does_not_spend_the_budget():
    """It is checked before the reservation, which is the whole failure."""
    import inspect

    source = inspect.getsource(ProactiveAnticipationEngine._fire_initiation)
    dedup = source.index("_condition_is_worth_restating")
    reserve = source.index("_reserve_initiation")
    assert dedup < reserve, "a repeat still consumes a daily initiation slot"
