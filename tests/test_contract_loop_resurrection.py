"""Dead contract loops must be revivable from health-pulse threads.

Observed live 2026-07-05: the runtime sat DEGRADED for 84 minutes with three
'important' contract entries dead (event_loop_monitor, mind_tick,
unified_runtime_pressure). Two had working restart machinery that silently
no-oped because health pulses run on plain threads where asyncio task
creation raises; the third was a phantom requirement with no provider
anywhere in the tree.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import core.runtime.runtime_pressure as rp
from core.runtime.runtime_pressure import UnifiedRuntimePressure


class TestUnifiedRuntimePressure:
    def test_snapshot_shape_and_nominal_alive(self, monkeypatch):
        pressure = UnifiedRuntimePressure()
        snap = pressure.runtime_pressure_snapshot()
        for key in (
            "loop_lag_s",
            "memory_pct",
            "thermal_level",
            "model_resource_lifecycle",
            "model_load_active",
            "red_zones",
            "pressure_ok",
        ):
            assert key in snap, f"snapshot missing {key}"
        assert isinstance(snap["red_zones"], list)

    def test_model_loading_is_observed_without_importing_the_model_module(
        self, monkeypatch
    ):
        monkeypatch.delitem(
            __import__("sys").modules,
            "core.brain.llm.mlx_client",
            raising=False,
        )

        snap = UnifiedRuntimePressure().runtime_pressure_snapshot()

        assert snap["model_resource_lifecycle"] == "cold"
        assert snap["model_load_active"] is False

    def test_live_uninitialized_lane_is_reported_as_model_loading(
        self, monkeypatch
    ):
        import sys

        class _Process:
            @staticmethod
            def is_alive():
                return True

        fake_client = SimpleNamespace(
            _lane_state="warming",
            _process=_Process(),
            _init_done=False,
            _warmup_in_flight=True,
        )
        # The probe reads the registry through clients_snapshot(), not by
        # touching _CLIENTS: iterating the dict directly raised "dictionary
        # changed size during iteration" live on 2026-08-03, and because the
        # inference gate is fail-closed that RuntimeError held the runtime
        # DEGRADED across health pulses.
        #
        # A double that exposes only _CLIENTS therefore looks like an EMPTY
        # registry, and the probe correctly reports "cold" — so this test was
        # asserting against a fake that no longer resembles the real module.
        # clients_snapshot returns a list of (name, client) pairs.
        fake_module = SimpleNamespace(
            _CLIENTS={"resident": fake_client},
            clients_snapshot=lambda: [("resident", fake_client)],
        )
        monkeypatch.setitem(
            sys.modules,
            "core.brain.llm.mlx_client",
            fake_module,
        )

        snap = UnifiedRuntimePressure().runtime_pressure_snapshot()

        assert snap["model_resource_lifecycle"] == "model_loading"
        assert snap["model_load_active"] is True
        assert snap["model_lane_count"] == 1

    def test_red_zone_memory_makes_it_not_alive(self, monkeypatch):
        from core.runtime.resource_observation import SimulatedResourceObserver

        observer = SimulatedResourceObserver(
            scenario_id="critical-memory",
            memory_percent=97.0,
        )
        pressure = UnifiedRuntimePressure(observer=observer)
        assert pressure.is_alive() is False
        assert pressure.get_status()["observation_source"] == "simulated"
        snap = pressure.get_status()
        assert any(zone.startswith("memory_") for zone in snap["red_zones"])

    def test_is_alive_never_raises(self, monkeypatch):
        pressure = UnifiedRuntimePressure()
        monkeypatch.setattr(
            pressure, "runtime_pressure_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert pressure.is_alive() is False

    def test_singleton_accessor(self):
        assert rp.get_unified_runtime_pressure() is rp.get_unified_runtime_pressure()

    def test_stale_lag_is_retained_as_history_but_not_current_pressure(self, monkeypatch):
        from core.runtime.resource_observation import SimulatedResourceObserver

        monitor = SimpleNamespace(
            get_status=lambda: {
                "alive": True,
                "last_lag_s": 3.638,
                "last_sample_at_unix": time.time() - 20.0,
                "sample_age_s": 20.0,
                "sample_fresh": False,
            }
        )
        monkeypatch.setattr(
            "core.runtime.service_registry.get_runtime_service",
            lambda name, default=None: monitor if name == "event_loop_monitor" else default,
        )
        observer = SimulatedResourceObserver(
            scenario_id="stale-loop-sample",
            memory_percent=40.0,
        )

        snapshot = UnifiedRuntimePressure(observer=observer).runtime_pressure_snapshot()

        assert snapshot["loop_lag_s"] == 0.0
        assert snapshot["last_observed_loop_lag_s"] == 3.638
        assert snapshot["loop_lag_sample_fresh"] is False
        assert "loop_lag_observation_stale" in snapshot["red_zones"]


class TestEventLoopMonitorThreadRevival:
    def test_healthy_tick_replaces_current_lag_without_erasing_breach_history(self):
        from core.utils.concurrency import EventLoopMonitor

        class _RunningTask:
            def done(self):
                return False

        monitor = EventLoopMonitor(threshold=0.5)
        monitor._task = _RunningTask()
        monitor._started_at = time.perf_counter()
        monitor._capture_lag_sample(3.638)
        monitor._last_breach_lag = 3.638
        monitor._last_breach_at = time.time()

        monitor._capture_lag_sample(0.012)
        status = monitor.get_status()

        assert status["alive"] is True
        assert status["sample_fresh"] is True
        assert status["last_lag_s"] == 0.012
        assert status["peak_lag_s"] == 3.638
        assert status["last_breach_lag_s"] == 3.638

    def test_foreign_thread_hands_restart_to_owner_loop(self):
        from core.utils.concurrency import EventLoopMonitor

        class _DoneTask:
            def done(self):
                return True

        scheduled: list = []

        class _FakeLoop:
            def is_closed(self):
                return False

            def call_soon_threadsafe(self, fn, *args):
                scheduled.append(fn)

        monitor = EventLoopMonitor(threshold=0.5)
        monitor._task = _DoneTask()
        monitor._owner_loop = _FakeLoop()

        def _raising_start():
            raise RuntimeError("no running event loop")

        monitor.start = _raising_start

        # Called from a plain (non-asyncio) test thread: must not raise, must
        # schedule the restart onto the owning loop, and must stay honest
        # (False) until the restart actually lands.
        assert monitor.ensure_running() is False
        assert scheduled == [_raising_start]

    def test_owner_loop_captured_on_start(self):
        from core.utils.concurrency import EventLoopMonitor

        async def _scenario():
            monitor = EventLoopMonitor(threshold=0.5)
            monitor.start()
            try:
                assert monitor._owner_loop is asyncio.get_running_loop()
            finally:
                await monitor.stop()

        asyncio.run(_scenario())


class TestMindTickThreadRevival:
    def test_repair_from_foreign_thread_schedules_onto_owner_loop(self):
        from core.mind_tick import MindTick

        scheduled: list = []

        class _FakeLoop:
            def is_closed(self):
                return False

            def call_soon_threadsafe(self, fn, *args):
                scheduled.append(fn)

        class _DoneTask:
            def done(self):
                return True

            def exception(self):
                return None

        tick = MindTick.__new__(MindTick)
        tick._running = True
        tick._task = _DoneTask()
        tick._owner_loop = _FakeLoop()
        tick._consecutive_loop_failures = 0
        tick._last_liveness_repair_at = 0.0
        tick._started_at = 1.0

        # Plain thread: no running loop here. The repair must hand itself to
        # the owning loop instead of returning False and leaving the mind dead.
        assert tick._attempt_liveness_repair() is False
        assert len(scheduled) == 1

    def test_background_kernel_tick_yields_under_foreground_inference(self, monkeypatch):
        # A soak's back-to-back turns saturate the generation gate; the
        # background kernel tick must yield instead of blocking (which froze
        # the iteration and falsely marked mind_tick dead, 2026-07-06).
        import core.runtime.backpressure as bp
        from core.mind_tick import MindTick

        monkeypatch.setattr(bp, "foreground_inference_active", lambda: True)
        tick = MindTick.__new__(MindTick)
        tick.orchestrator = SimpleNamespace(_flow_controller=None)
        assert tick._background_reasoning_pause_reason() == "foreground_inference_active"


class TestGuiDegradedReadyKeepsUI:
    def test_degraded_ready_when_conversation_works_but_loop_degraded(self):
        from interface.gui_actor import _heartbeat_response_state

        class _Resp:
            status_code = 503

            @staticmethod
            def json():
                return {
                    "healthy": False,
                    "status": "booting",
                    "runtime_probe_healthy": True,
                    "conversation_ready": True,
                    "boot_phase": "kernel_warming",
                    "blockers": ["healthy", "runtime_contract_healthy", "important:mind_tick"],
                    "required_probes": {"all_passed": True},
                }

        # Conversation works; only a background loop is degraded → keep the UI.
        assert _heartbeat_response_state(_Resp()) == "degraded_ready"

    def test_a_turn_in_flight_is_working_not_a_miss(self):
        # Live 2026-09-15: the payload during a 636s answer.
        from interface.gui_actor import _heartbeat_response_state

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "healthy": False,
                    "status": "working",
                    "runtime_probe_healthy": True,
                    "runtime_status": "working",
                    "conversation_ready": False,
                    "conversation_busy": True,
                    "boot_phase": "conversation_working",
                    "blockers": [],
                    "required_probes": {"all_passed": True},
                }

        assert _heartbeat_response_state(_Resp()) == "working"

    def test_truly_down_is_still_unhealthy(self):
        from interface.gui_actor import _heartbeat_response_state

        class _Resp:
            status_code = 503

            @staticmethod
            def json():
                return {
                    "healthy": False,
                    "status": "booting",
                    "runtime_probe_healthy": False,
                    "conversation_ready": False,
                    "boot_phase": "kernel_booting",
                    "blockers": ["healthy", "inference"],
                    "required_probes": {"all_passed": False},
                }

        assert _heartbeat_response_state(_Resp()) == "unhealthy"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
