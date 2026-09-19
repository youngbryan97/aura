import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.orchestrator import RobustOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_setup_registers_output_gate_for_runtime_health_contract():
    ServiceContainer.clear()
    try:
        orchestrator = RobustOrchestrator()
        orchestrator.setup()

        assert ServiceContainer.get("output_gate", default=None) is orchestrator.output_gate
    finally:
        ServiceContainer.clear()


def test_final_boot_complete_uses_fresh_runtime_health_check():
    boot_source = (PROJECT_ROOT / "core" / "orchestrator" / "boot.py").read_text(
        encoding="utf-8"
    )
    final_boot_slice = boot_source.split("# ── Final Success State", 1)[1].split(
        "except (ImportError, AttributeError, RuntimeError) as e:",
        1,
    )[0]

    assert "self.status.healthy = bool(self.health_check())" in final_boot_slice
    assert "final runtime health check failed" in final_boot_slice
    assert "_final_boot_health_log(" in final_boot_slice


#: How a bounded wait on the scheduler's start is spelled.
#:
#: It was `await asyncio.wait_for(scheduler.start(), timeout=5.0)`. It is
#: now `await_while_the_task_moves(scheduler.start(), stall_s=5.0, ...)`,
#: which cancels on a STALL rather than on a deadline — a start that is
#: still making progress is no longer killed at five seconds. The property
#: these tests hold is that the start is bounded at all, so they read the
#: call and the bound rather than the name of the waiter.
_STARTS_THE_SCHEDULER = "scheduler.start()"
_AND_BOUNDS_IT = ("timeout=5.0", "stall_s=5.0")


def _starts_the_scheduler_under_a_bound(source: str) -> bool:
    for line in source.splitlines():
        if _STARTS_THE_SCHEDULER in line and any(b in line for b in _AND_BOUNDS_IT):
            return True
    return False


def test_boot_phase_health_contract_does_not_emit_runtime_critical_summary():
    boot_source = (PROJECT_ROOT / "core" / "orchestrator" / "boot.py").read_text(
        encoding="utf-8"
    )
    health_slice = boot_source.split("# ── Runtime Health Contract", 1)[1].split(
        "# ── Startup Validation",
        1,
    )[0]

    assert "runtime_ready_for_health_log" in health_slice
    assert "log_health_report() if runtime_ready_for_health_log else evaluate_health()" in health_slice
    assert "HEALTH CONTRACT DETAIL: boot pending critical liveness" in health_slice
    scheduler_slice = boot_source.split("# ── Canonical Scheduler Heartbeat", 1)[1].split(
        "# ── Runtime Health Contract",
        1,
    )[0]
    assert "Scheduler heartbeat disabled for foreground-only boot" not in scheduler_slice
    assert _starts_the_scheduler_under_a_bound(scheduler_slice)


def test_canonical_boot_refreshes_health_before_manifest():
    aura_main = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    boot_slice = aura_main.split("async def _boot_runtime_orchestrator", 1)[1].split(
        "def _refresh_orchestrator_health_before_manifest",
        1,
    )[0]

    assert "await _enforce_boot_probes(ready_label)" in boot_slice
    assert "activate_live_mind_runtime" in boot_slice
    assert "await _activate_ulysses_covenant_for_boot()" in boot_slice
    assert boot_slice.index("activate_live_mind_runtime") < boot_slice.index(
        "ServiceContainer.lock_registration()"
    )
    assert boot_slice.index("await _activate_ulysses_covenant_for_boot()") < boot_slice.index(
        "await orchestrator.start()"
    )
    assert boot_slice.index("await _activate_ulysses_covenant_for_boot()") < boot_slice.index(
        "ServiceContainer.lock_registration()"
    )
    assert "readiness_snapshot = await _settle_orchestrator_health_before_manifest(" in boot_slice
    assert "ServiceContainer.write_service_ownership_manifest," in boot_slice
    assert "await asyncio.to_thread(\n        _write_runtime_manifest," in boot_slice
    assert "readiness_snapshot=readiness_snapshot" in boot_slice
    assert "_schedule_runtime_manifest_ready_refresh(" in boot_slice
    assert "initial_readiness=readiness_snapshot" in boot_slice
    assert boot_slice.index("_settle_orchestrator_health_before_manifest") < boot_slice.index(
        "_write_runtime_manifest,"
    )


@pytest.mark.asyncio
async def test_ulysses_covenant_boot_helper_materializes_status_off_loop(monkeypatch):
    import threading

    import aura_main
    import core.sovereignty.ulysses as ulysses

    event_loop_thread = threading.current_thread()
    observed: dict[str, object] = {}

    class Covenant:
        @staticmethod
        def status():
            observed["status_thread"] = threading.current_thread()
            return {
                "active_contracts": 3,
                "hard": 1,
                "integrity": 1.0,
                "chain_length": 3,
            }

    def boot():
        observed["boot_thread"] = threading.current_thread()
        return Covenant()

    monkeypatch.setattr(ulysses, "boot_ulysses_covenant", boot)
    monkeypatch.setattr(aura_main, "_env_flag", lambda _name, _default: True)

    status = await aura_main._activate_ulysses_covenant_for_boot()

    assert status == {
        "active_contracts": 3,
        "hard": 1,
        "integrity": 1.0,
        "chain_length": 3,
    }
    assert observed["boot_thread"] is not event_loop_thread
    assert observed["status_thread"] is not event_loop_thread


def test_runtime_manifest_pre_ready_snapshot_gets_bounded_refresh_task():
    aura_main = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    refresh_slice = aura_main.split("def _schedule_runtime_manifest_ready_refresh", 1)[1].split(
        "def _register_runtime_singletons",
        1,
    )[0]

    assert "if bool(initial_readiness.get(\"ready\")):" in refresh_slice
    assert "AURA_RUNTIME_MANIFEST_READY_REFRESH_SECONDS" in refresh_slice
    assert "_refresh_runtime_manifest_until_ready(" in refresh_slice
    assert "runtime_manifest.ready_refresh" in refresh_slice
    assert "if bool(snapshot.get(\"ready\")):" in refresh_slice
    assert "readiness_snapshot=snapshot" in refresh_slice
    assert "snapshot = await asyncio.to_thread(" in refresh_slice
    assert "await asyncio.to_thread(\n                _write_runtime_manifest," in refresh_slice


@pytest.mark.asyncio
async def test_runtime_manifest_refresh_keeps_owner_loop_responsive(monkeypatch, tmp_path):
    import aura_main

    def slow_health(_orchestrator, _ready_label):
        time.sleep(0.08)
        return {"ready": True, "status": "healthy", "required_probe_blockers": []}

    def slow_write(**_kwargs):
        time.sleep(0.08)

    monkeypatch.setattr(
        aura_main,
        "_refresh_orchestrator_health_before_manifest",
        slow_health,
    )
    monkeypatch.setattr(aura_main, "_write_runtime_manifest", slow_write)
    monkeypatch.setattr(aura_main, "is_shutdown_requested", lambda: False)

    refresh = asyncio.create_task(
        aura_main._refresh_runtime_manifest_until_ready(
            orchestrator=object(),
            profile="desktop",
            ready_label="Desktop",
            artifact_root=tmp_path,
            timeout_s=1.0,
            interval_s=0.01,
        )
    )
    owner_loop_ticks = 0
    while not refresh.done():
        owner_loop_ticks += 1
        await asyncio.sleep(0.005)
    await refresh

    assert owner_loop_ticks >= 20


def test_runtime_manifest_unready_refresh_logs_on_change_not_every_poll(monkeypatch, caplog):
    # caplog, not capsys: asserting on stdout couples this test to whichever
    # root logging handlers earlier tests happened to install (observed as an
    # order-dependence failure in chunked suite runs).
    import logging

    import aura_main
    import core.runtime.health_contract as health_contract

    class Orchestrator:
        @staticmethod
        def health_check():
            return False

    def report():
        return {
            "status": "critical",
            "failures": {
                "critical": [{"container_key": "inference_gate"}],
                "important": [],
            },
        }

    monkeypatch.setattr(health_contract, "runtime_health_report", report)
    monkeypatch.setattr(health_contract, "required_probe_status", lambda contract: {})
    monkeypatch.setattr(
        health_contract,
        "required_probe_blockers",
        lambda status: ["runtime_required_probes", "probe:inference"],
    )
    monkeypatch.setattr(aura_main, "_MANIFEST_UNREADY_LOG_INTERVAL_S", 9999.0)
    aura_main._MANIFEST_UNREADY_LOG_STATE.clear()

    with caplog.at_level(logging.WARNING, logger="Aura.Main"):
        first = aura_main._refresh_orchestrator_health_before_manifest(
            Orchestrator(),
            "Server",
        )
        second = aura_main._refresh_orchestrator_health_before_manifest(
            Orchestrator(),
            "Server",
        )

    assert first["ready"] is False
    assert second["ready"] is False
    unready_warnings = [
        record
        for record in caplog.records
        if "Runtime health still not clean before manifest" in record.getMessage()
    ]
    assert len(unready_warnings) == 1


def test_runtime_manifest_reuses_fresh_successful_health_receipt(monkeypatch):
    import aura_main

    class Orchestrator:
        status = SimpleNamespace(initialized=True, running=True)
        _last_health_check_result = True
        _last_health_check_monotonic = time.monotonic()

        @staticmethod
        def health_check():
            raise AssertionError("fresh boot health must not be recomputed")

    result = aura_main._refresh_orchestrator_health_before_manifest(
        Orchestrator(),
        "Desktop",
    )

    assert result["ready"] is True
    assert result["source"] == "fresh_orchestrator_health_receipt"
    assert result["receipt_age_s"] < 0.05


@pytest.mark.asyncio
async def test_runtime_manifest_settle_waits_for_transient_inference_warmup(monkeypatch):
    import aura_main

    snapshots = iter(
        [
            {"ready": False, "status": "warming", "required_probe_blockers": ["probe:inference"]},
            {"ready": True, "status": "healthy", "required_probe_blockers": []},
        ]
    )
    calls = []

    def refresh(orchestrator, ready_label, *, log_unready=True):
        calls.append(log_unready)
        return next(snapshots)

    monkeypatch.setattr(aura_main, "_refresh_orchestrator_health_before_manifest", refresh)
    monkeypatch.setenv("AURA_MANIFEST_HEALTH_SETTLE_SECONDS", "1")
    monkeypatch.setenv("AURA_MANIFEST_HEALTH_SETTLE_INTERVAL_SECONDS", "0.001")

    result = await aura_main._settle_orchestrator_health_before_manifest(object(), "Desktop")

    assert result["ready"] is True
    assert calls == [False, False]


@pytest.mark.asyncio
async def test_runtime_manifest_settle_logs_only_final_persistent_failure(monkeypatch):
    import aura_main

    calls = []

    def refresh(orchestrator, ready_label, *, log_unready=True):
        calls.append(log_unready)
        return {
            "ready": False,
            "status": "critical",
            "required_probe_blockers": ["probe:inference"],
        }

    monkeypatch.setattr(aura_main, "_refresh_orchestrator_health_before_manifest", refresh)
    monkeypatch.setenv("AURA_MANIFEST_HEALTH_SETTLE_SECONDS", "0")

    result = await aura_main._settle_orchestrator_health_before_manifest(object(), "Desktop")

    assert result["ready"] is False
    assert calls == [False, True]


def test_runtime_manifest_records_pre_ready_boot_contract_snapshot(tmp_path):
    from core.runtime.runtime_manifest import build_runtime_manifest

    manifest = build_runtime_manifest(
        profile="desktop",
        ready_label="Server",
        project_root=PROJECT_ROOT,
        artifact_root=tmp_path,
        readiness_snapshot={
            "ready": False,
            "status": "booting",
            "critical": ["inference_gate"],
            "important": [],
            "required_probe_blockers": ["probe:inference"],
        },
    )

    assert manifest["readiness_snapshot"]["ready"] is False
    assert manifest["readiness_snapshot"]["critical"] == ["inference_gate"]
    assert manifest["readiness_snapshot"]["required_probe_blockers"] == ["probe:inference"]
    assert manifest["launch_provenance"]["schema"] == "aura.launch_provenance.v1"
    assert manifest["launch_provenance"]["required"] is False


def test_runtime_manifest_does_not_mark_registered_unready_role_healthy(tmp_path):
    from core.runtime.runtime_manifest import build_runtime_manifest

    class UnreadyOutputGate:
        @staticmethod
        def is_ready():
            return False

    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance(
            "output_gate",
            UnreadyOutputGate(),
            required=False,
        )
        manifest = build_runtime_manifest(
            profile="desktop",
            ready_label="Server",
            project_root=PROJECT_ROOT,
            artifact_root=tmp_path,
        )

        service = manifest["services"]["output_gate"]
        role = manifest["service_roles"]["output_gate"]
        assert service["health_status"] == "liveness_failed"
        assert role["health_status"] == "liveness_failed"
        assert role["health_evidence"]["output_gate"]["liveness"] == "failed"
        assert "output_gate" in manifest["disabled_subsystems"]
    finally:
        ServiceContainer.clear()


def test_runtime_manifest_snapshot_never_invokes_arbitrary_service_status(tmp_path):
    from core.runtime.runtime_manifest import build_runtime_manifest

    class BlockingStatusService:
        @staticmethod
        def get_status():
            raise AssertionError("runtime manifest observation invoked service code")

    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance(
            "manifest_observation_probe",
            BlockingStatusService(),
            required=False,
        )
        manifest = build_runtime_manifest(
            profile="desktop",
            ready_label="Server",
            project_root=PROJECT_ROOT,
            artifact_root=tmp_path,
            readiness_snapshot={
                "ready": True,
                "status": "healthy",
                "required_probe_blockers": [],
            },
        )

        service = manifest["services"]["manifest_observation_probe"]
        assert service["health_status"] == "registered_unchecked"
    finally:
        ServiceContainer.clear()


def test_foreground_start_keeps_scheduler_heartbeat_alive():
    main_source = (PROJECT_ROOT / "core" / "orchestrator" / "main.py").read_text(
        encoding="utf-8"
    )
    scheduler_section = main_source.split(
        "# HARDENING: Register Periodic Metabolic/Substrate Tasks",
        1,
    )[1]
    foreground_slice = scheduler_section.split("if _foreground_only_runtime():", 1)[1].split(
        "else:",
        1,
    )[0]

    assert "heartbeat remains active for runtime health" in foreground_slice
    assert "if not scheduler.is_alive():" in foreground_slice
    assert _starts_the_scheduler_under_a_bound(foreground_slice)


def test_foreground_boot_defers_mycelium_infrastructure_mapping():
    boot_source = (PROJECT_ROOT / "core" / "orchestrator" / "boot.py").read_text(
        encoding="utf-8"
    )
    assert boot_source.count("mycelium.setup()") == 1
    assert "orchestrator.mycelium.background_mapping" not in boot_source
    assert "mapping_scheduled = mycelium.setup()" in boot_source
    assert "mycelium.get_infrastructure_report()[\"mapping_state\"]" in boot_source


def test_orchestrator_main_loop_refreshes_watchdog_heartbeat():
    main_source = (PROJECT_ROOT / "core" / "orchestrator" / "main.py").read_text(
        encoding="utf-8"
    )
    loop_slice = main_source.split(
        "logger.info(\"🚩 [ORCHESTRATOR] Main Heartbeat Active",
        1,
    )[1].split("await asyncio.sleep(0.05)", 1)[0]

    assert "self.status.cycle_count += 1" in loop_slice
    assert "self._update_heartbeat()" in loop_slice
    assert "watchdog.heartbeat(\"orchestrator_loop\")" in loop_slice
