"""Boot flight recorder: a slow boot must name its slow phase.

A live desktop boot once took 13 minutes with zero evidence in the logs of
which phase ate the time. The profiler records phase durations as the boot
runs, warns on slow phases in real time, and persists a JSON artifact at
ready.
"""
from __future__ import annotations

import json
import logging
import time

from core.runtime.boot_profile import (
    PHASE_WARN_S,
    SLOW_PHASE_WARN_S,
    BootProfiler,
    get_boot_profiler,
    reset_boot_profiler,
)


class TestMarkApi:
    def test_marks_attribute_elapsed_time_to_named_phases(self):
        prof = BootProfiler()
        time.sleep(0.02)
        prof.mark("phase_a")
        time.sleep(0.01)
        prof.mark("phase_b")

        phases = prof.phases()
        assert [p["name"] for p in phases] == ["phase_a", "phase_b"]
        assert phases[0]["duration_s"] >= 0.015
        assert phases[1]["duration_s"] >= 0.005
        # Offsets are cumulative from boot start.
        assert phases[1]["offset_s"] >= phases[0]["duration_s"]

    def test_phase_context_manager_records_even_on_error(self):
        prof = BootProfiler()
        try:
            with prof.phase("failing_phase"):
                raise RuntimeError("boot step exploded")
        except RuntimeError:
            pass
        assert [p["name"] for p in prof.phases()] == ["failing_phase"]

    def test_slow_phase_warns_in_real_time(self, caplog, monkeypatch):
        prof = BootProfiler()
        with caplog.at_level(logging.WARNING, logger="Aura.BootProfile"):
            monkeypatch.setattr(
                "core.runtime.boot_profile.SLOW_PHASE_WARN_S", 0.0
            )
            # _record reads the module constant at call time via the class —
            # patch the comparison threshold by calling _record directly.
            prof._record("model_load", SLOW_PHASE_WARN_S + 1.0, 0.0)
        slow = [r for r in caplog.records if "🐢" in r.getMessage()]
        assert len(slow) == 1
        assert "model_load" in slow[0].getMessage()

    def test_resident_orchestrator_uses_its_measured_phase_budget(self, caplog):
        prof = BootProfiler()
        with caplog.at_level(logging.WARNING, logger="Aura.BootProfile"):
            prof._record(
                "orchestrator_runtime_start",
                PHASE_WARN_S["orchestrator_runtime_start"] - 1.0,
                0.0,
            )
        assert not [r for r in caplog.records if "orchestrator_runtime_start" in r.getMessage()]

    def test_readiness_settlement_uses_its_bounded_wait_budget(self, caplog):
        prof = BootProfiler()
        with caplog.at_level(logging.WARNING, logger="Aura.BootProfile"):
            prof._record(
                "readiness_health_snapshot",
                PHASE_WARN_S["readiness_health_snapshot"] - 1.0,
                0.0,
            )
        assert not [
            record
            for record in caplog.records
            if "readiness_health_snapshot" in record.getMessage()
        ]

    def test_summary_names_slowest_phases_first(self):
        prof = BootProfiler()
        prof._record("fast", 0.01, 0.0)
        prof._record("slowest", 12.5, 0.0)
        prof._record("medium", 3.0, 0.0)
        summary = prof.summary(top=2)
        assert "slowest=12.5s" in summary
        assert "medium=3.0s" in summary
        assert "fast" not in summary
        assert "3 phases" in summary

    def test_empty_profile_summary_is_safe(self):
        assert "no phases recorded" in BootProfiler().summary()


class TestReportAndArtifact:
    def test_report_schema(self):
        prof = BootProfiler()
        prof.mark("only_phase")
        report = prof.to_report()
        assert report["schema"] == "aura.boot_profile.v1"
        assert report["phases"][0]["name"] == "only_phase"
        assert report["total_s"] >= 0.0

    def test_artifact_written_atomically_and_parseable(self, tmp_path):
        prof = BootProfiler()
        prof.mark("phase_x")
        target = tmp_path / "boot_profile.json"
        written = prof.write_artifact(target)
        assert written == target
        payload = json.loads(target.read_text())
        assert payload["phases"][0]["name"] == "phase_x"

    def test_artifact_write_failure_never_raises(self, monkeypatch):
        prof = BootProfiler()
        prof.mark("p")
        monkeypatch.setattr(
            "core.runtime.file_write_gateway.get_file_write_gateway",
            lambda: (_ for _ in ()).throw(RuntimeError("gateway down")),
        )
        assert prof.write_artifact() is None


class TestSingleton:
    def test_get_returns_same_instance(self):
        reset_boot_profiler()
        assert get_boot_profiler() is get_boot_profiler()

    def test_reset_returns_fresh_instance(self):
        first = get_boot_profiler()
        second = reset_boot_profiler()
        assert first is not second
        assert get_boot_profiler() is second


def test_boot_spine_is_instrumented():
    """The orchestrator boot must keep its phase marks."""
    import inspect

    from core.orchestrator import boot

    source = inspect.getsource(boot)
    for phase in (
        "inference_gate",
        "cognitive_architecture",
        "skill_system",
        "cognitive_core",
    ):
        assert f'boot_profiler.mark("{phase}")' in source, (
            f"boot spine lost its '{phase}' profiler mark"
        )


def test_boot_profile_persistence_is_awaited_off_loop():
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "aura_main.py").read_text())
    boot = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_boot_runtime_orchestrator")
    writes = [n for n in ast.walk(boot) if isinstance(n, ast.Attribute)
              and n.attr == "write_artifact"]
    assert len(writes) == 1
    assert any(
        isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "asyncio.to_thread"
        and writes[0] in n.value.args
        for n in ast.walk(boot)
    )
