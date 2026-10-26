"""Contracts for K6 typed conditions and the A4 escalation-rate cap.

K6: managed components expose Ready/Progressing/Degraded with a reason
and a last-TRANSITION time (steady state shows its true age; flapping
shows recent transitions).

A4: one repeating fault on a fail-closed subsystem must not become a
CRITICAL storm — the first N identical escalations pass with full force,
repeats stay visible but neither re-escalate nor raise (FM-FCL-001).
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.conditions import (
    ConditionType,
    all_conditions_report,
    get_component_conditions,
    reset_conditions_for_test,
)
from core.runtime.errors import get_escalation_governor, record_degradation

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_state():
    reset_conditions_for_test()
    get_escalation_governor().reset_for_test()
    yield
    reset_conditions_for_test()
    get_escalation_governor().reset_for_test()


class TestConditions:
    def test_transition_time_moves_only_on_status_flip(self, monkeypatch):
        import core.runtime.conditions as cond_mod

        class Clock:
            now = 1_000.0

            def time(self):
                return self.now

        clock = Clock()
        monkeypatch.setattr(cond_mod, "time", clock)

        conditions = get_component_conditions("test_lane")
        first = conditions.set(ConditionType.READY, True, reason="Warm")
        assert first.last_transition_at == 1_000.0

        clock.now = 1_100.0
        refreshed = conditions.set(ConditionType.READY, True, reason="StillWarm")
        assert refreshed.last_transition_at == 1_000.0, "no flip → transition keeps its age"
        assert refreshed.last_update_at == 1_100.0
        assert refreshed.reason == "StillWarm"

        clock.now = 1_200.0
        flipped = conditions.set(ConditionType.READY, False, reason="Down")
        assert flipped.last_transition_at == 1_200.0, "flip → transition moves"

    def test_registry_report_shape(self):
        get_component_conditions("lane_a").set(
            ConditionType.DEGRADED, True, reason="CrashLoopBackOff", message="trip=2"
        )
        report = all_conditions_report()
        entry = report["lane_a"]["Degraded"]
        assert entry["status"] is True
        assert entry["reason"] == "CrashLoopBackOff"
        assert "trip=2" in entry["message"]

    def test_reconciler_publishes_lane_conditions(self):
        from core.runtime.lane_reconciler import CrashLoopBreaker, LaneReconciler

        async def spawn():
            return True

        async def evict(path):
            return True

        rec = LaneReconciler(
            observe_lanes=lambda: [],
            primary_alive=lambda: False,
            primary_key=lambda: "/m/cortex",
            primary_age_s=lambda: 0.0,
            spawn_primary=spawn,
            evict_lane=evict,
            foreground_active=lambda: False,
            breaker=CrashLoopBreaker(),
        )
        asyncio.run(rec.reconcile_once())
        conditions = get_component_conditions("cortex_lane")
        ready = conditions.get(ConditionType.READY)
        progressing = conditions.get(ConditionType.PROGRESSING)
        assert ready is not None and ready.status is False
        assert ready.reason == "PrimaryDown"
        assert progressing is not None and progressing.status is True
        assert progressing.reason == "WarmupRequested"
        assert "conditions" in rec.snapshot()

    def test_reconciler_names_crash_loop_in_ready_condition(self):
        from core.runtime.lane_reconciler import CrashLoopBreaker, LaneReconciler

        breaker = CrashLoopBreaker()
        for _ in range(3):
            breaker.note_death("/m/cortex", lifetime_s=10.0, reason="stall")

        async def spawn():
            return True

        async def evict(path):
            return True

        rec = LaneReconciler(
            observe_lanes=lambda: [],
            primary_alive=lambda: False,
            primary_key=lambda: "/m/cortex",
            primary_age_s=lambda: 0.0,
            spawn_primary=spawn,
            evict_lane=evict,
            foreground_active=lambda: False,
            breaker=breaker,
        )
        asyncio.run(rec.reconcile_once())
        ready = get_component_conditions("cortex_lane").get(ConditionType.READY)
        assert ready is not None and ready.status is False
        assert ready.reason == "CrashLoopBackOff"
        assert "crash_loop_backoff" in ready.message


class TestEscalationCap:
    @pytest.fixture
    def fail_closed_unit(self, monkeypatch):
        from core.container import ServiceContainer

        monkeypatch.setenv("AURA_MODE", "live")
        ServiceContainer.register_instance(
            "storm_unit", object(), failure_policy="fail-closed"
        )
        yield "storm_unit"
        ServiceContainer.clear()

    def test_first_escalations_raise_with_full_force(self, fail_closed_unit):
        with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
            record_degradation(
                fail_closed_unit,
                RuntimeError("pool child died"),
                severity="degraded",
                action="unit test",
            )

    def test_repeats_past_cap_record_but_do_not_raise(self, fail_closed_unit):
        for _ in range(3):
            with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
                record_degradation(
                    fail_closed_unit,
                    RuntimeError("pool child died"),
                    severity="degraded",
                    action="unit test",
                )
        # The 4th identical fault in the window: visible, NOT a raise.
        record = record_degradation(
            fail_closed_unit,
            RuntimeError("pool child died"),
            severity="degraded",
            action="unit test",
        )
        assert record.severity == "degraded", "caller severity preserved, no re-escalation"
        assert get_escalation_governor().snapshot(), "suppression is observable"

    def test_distinct_error_types_are_independent(self, fail_closed_unit):
        for _ in range(3):
            with pytest.raises(RuntimeError):
                record_degradation(
                    fail_closed_unit,
                    RuntimeError("pool child died"),
                    severity="degraded",
                    action="unit test",
                )
        # A DIFFERENT fault class on the same subsystem still fails closed.
        with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
            record_degradation(
                fail_closed_unit,
                ValueError("state checksum mismatch"),
                severity="degraded",
                action="unit test",
            )

    def test_kill_switch_disables_suppression(self, fail_closed_unit, monkeypatch):
        monkeypatch.setenv("AURA_ESCALATION_CAP", "0")
        for _ in range(5):
            with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
                record_degradation(
                    fail_closed_unit,
                    RuntimeError("pool child died"),
                    severity="degraded",
                    action="unit test",
                )

    def test_fresh_window_escalates_again(self, fail_closed_unit, monkeypatch):
        from core.runtime.errors import get_escalation_governor

        monkeypatch.setenv("AURA_ESCALATION_CAP_WINDOW_S", "0.05")
        # The test moves the governor's clock. On the wall clock, four calls
        # under a loaded suite outlasted the 50 ms window before the fourth.
        now = [1_000.0]
        monkeypatch.setattr(get_escalation_governor(), "_clock", lambda: now[0])

        for _ in range(3):
            with pytest.raises(RuntimeError):
                record_degradation(
                    fail_closed_unit,
                    RuntimeError("pool child died"),
                    severity="degraded",
                    action="unit test",
                )
        record = record_degradation(
            fail_closed_unit,
            RuntimeError("pool child died"),
            severity="degraded",
            action="unit test",
        )
        assert record.severity == "degraded"
        now[0] += 0.06
        with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
            record_degradation(
                fail_closed_unit,
                RuntimeError("pool child died"),
                severity="degraded",
                action="unit test",
            )

    def test_the_window_runs_on_the_experiment_clock(self):
        # A subject-core arm installs its own clock as time.time and a restore
        # rewinds it. A window on any other clock differs between arms.
        from core.runtime.errors import get_escalation_governor
        from core.subject.clock import ExperimentClock

        with ExperimentClock(1.0, start=5_000.0):
            assert get_escalation_governor()._clock() == 5_000.0
