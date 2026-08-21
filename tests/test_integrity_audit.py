"""Tests for the consolidated system-integrity audit."""
from __future__ import annotations

import threading
import time

import core.runtime.integrity_audit as ia
from core.health.read_model import HealthReadModelConfig, HealthSnapshotReadModel


def test_audit_aggregates_signals_and_reports_structure():
    report = ia.run_integrity_audit(log=False)
    assert set(report) >= {"healthy", "concerns", "advisory", "degradations", "crsm_loop", "caa_readiness"}
    assert isinstance(report["concerns"], list)
    # CRSM loop + CAA readiness are real on this repo → expect them surfaced
    assert report["crsm_loop"] and report["caa_readiness"]


def test_caa_and_crsm_are_advisory_not_health_blocking():
    report = ia.run_integrity_audit(log=False)
    # Operational proof facts surface as ADVISORY when they remain open, never
    # as runtime-health concerns, so they cannot make launch report "degraded".
    if report["crsm_loop"].get("state") == "open":
        assert any("CRSM" in c for c in report["advisory"])
    else:
        assert not any("CRSM" in c for c in report["advisory"])
    assert report["caa_readiness"]["level"] in {"production", "validated", "mixed", "bootstrap"}
    if report["caa_readiness"].get("below_design_capacity"):
        assert any("CAA steering" in c for c in report["advisory"])
    assert not any("CAA steering" in c for c in report["concerns"])
    assert not any("CRSM" in c for c in report["concerns"])


def test_maybe_run_is_throttled(monkeypatch):
    ia._last_run = 0.0
    first = ia.maybe_run(interval_s=10_000)
    assert first is not None
    # immediate second call within the interval returns the cached report, no re-run
    ran = {"n": 0}
    orig = ia.run_integrity_audit
    monkeypatch.setattr(ia, "run_integrity_audit", lambda **k: ran.__setitem__("n", ran["n"] + 1) or orig(**k))
    ia.maybe_run(interval_s=10_000)
    assert ran["n"] == 0                      # throttled — did not re-run


def test_integrity_read_model_never_joins_blocked_collector(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def collect():
        started.set()
        assert release.wait(1.0)
        return {
            "healthy": True,
            "concerns": [],
            "advisory": [],
            "crsm_loop": {"state": "closed"},
            "caa_readiness": {"level": "production"},
        }

    model = HealthSnapshotReadModel(
        collect,
        ia._integrity_snapshot_fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=1.0,
            max_stale_s=2.0,
            collection_timeout_s=0.5,
            metadata_key="integrity_read_model",
            worker_name_prefix="AuraIntegritySnapshotTest",
            incident_prefix="integrity-refresh",
            log_label="Integrity snapshot test",
        ),
    )
    monkeypatch.setattr(ia, "_INTEGRITY_READ_MODEL", model)

    started_at = time.monotonic()
    report = ia.read_integrity_audit()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.05
    assert report["healthy"] is False
    assert report["concerns"] == ["integrity_snapshot_initializing"]
    assert report["integrity_read_model"]["serving"] == "initializing"
    assert report["integrity_read_model"]["refresh_in_flight"] is True
    assert started.wait(0.2)
    release.set()


def test_integrity_snapshot_expiry_blocks_proof_not_runtime_health():
    from interface.routes.system import _runtime_integrity_public_payload

    payload = _runtime_integrity_public_payload(
        {
            "healthy": True,
            "concerns": [],
            "advisory": [],
            "crsm_loop": {"state": "closed"},
            "caa_readiness": {"level": "production"},
            "integrity_read_model": {
                "expired": True,
                "captured_at_unix": 100.0,
            },
        }
    )

    assert payload["healthy"] is True
    assert payload["proof_readiness"] is False
    assert "integrity:integrity_snapshot_expired" in payload["proof_blockers"]


def test_crsm_bridge_status_reuses_integrity_snapshot(monkeypatch, tmp_path):
    import core.consciousness.crsm_loop_monitor as monitor_module
    import core.consciousness.crsm_lora_bridge as bridge_module

    monkeypatch.setattr(
        bridge_module,
        "PERSIST_PATH",
        tmp_path / "crsm_lora_buffer.jsonl",
    )
    monkeypatch.setattr(
        ia,
        "read_integrity_audit",
        lambda: {
            "crsm_loop": {"state": "closed", "verified_consumption": True},
            "integrity_read_model": {"serving": "fresh", "expired": False},
        },
    )

    def fail_if_scanned():
        raise AssertionError("ordinary bridge status must not scan CRSM state")

    monkeypatch.setattr(monitor_module, "get_crsm_loop_monitor", fail_if_scanned)
    status = bridge_module.CRSMLoraBridge().get_status()

    assert status["loop"]["state"] == "closed"
    assert status["loop_read_model"]["serving"] == "fresh"


def test_strict_mode_reflected(monkeypatch):
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    assert ia.strict_mode() is True
    assert ia.run_integrity_audit(log=False)["strict_mode"] is True


def test_report_names_failure_pressure_feeders(monkeypatch):
    """During a lockdown the integrity report must name the pressure's top
    contributing subsystems — no log archaeology to find the feeder."""
    import core.runtime.integrity_audit as ia

    monkeypatch.setattr(
        "core.health.degraded_events.get_unified_failure_state",
        lambda limit=25: {
            "pressure": 0.85,
            "count": 12,
            "critical": 2,
            "top_subsystems": ["mlx_warmup", "chat.cognitive_engine_reply"],
        },
    )

    report = ia.run_integrity_audit(log=False)

    assert report["failure_state"]["pressure"] == 0.85
    assert report["failure_state"]["top_subsystems"][0] == "mlx_warmup"
    assert any("failure pressure 0.85" in a for a in report["advisory"])


def test_concern_verdict_uses_trailing_window_not_lifetime_counts(monkeypatch):
    """A long-lived runtime must recover once a degradation storm passes:
    old records outside the window must not hold the runtime unhealthy."""
    import time

    import core.runtime.integrity_audit as ia
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    try:
        old_ts = time.time() - ia._DEGRADATION_CONCERN_WINDOW_S - 60.0
        for i in range(ia._DEGRADATION_CONCERN + 5):
            tracker.record(
                DegradationRecord(
                    subsystem="stormy_subsystem",
                    severity="warning",
                    error_type="TimeoutError",
                    error_message=f"old event {i}",
                    action="recovered",
                    timestamp=old_ts,
                )
            )

        report = ia.run_integrity_audit(log=False)
        assert not any("stormy_subsystem" in c for c in report["concerns"]), (
            "stale records outside the window must not mark the runtime unhealthy"
        )

        # A fresh burst inside the window still trips the concern.
        for i in range(ia._DEGRADATION_CONCERN):
            tracker.record(
                DegradationRecord(
                    subsystem="stormy_subsystem",
                    severity="warning",
                    error_type="TimeoutError",
                    error_message=f"fresh event {i}",
                    action="recovered",
                    timestamp=time.time(),
                )
            )
        report = ia.run_integrity_audit(log=False)
        assert any("stormy_subsystem" in c for c in report["concerns"])
    finally:
        tracker.reset()


def test_concern_verdict_excludes_records_from_previous_process_epoch(monkeypatch):
    """A clean runtime cannot inherit a prior runtime's recent concern count."""
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    process_started_at = time.time() - 30.0
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", process_started_at)
    try:
        for i in range(ia._DEGRADATION_CONCERN + 5):
            tracker.record(
                DegradationRecord(
                    subsystem="cognitive_contract",
                    severity="warning",
                    error_type="RuntimeError",
                    error_message=f"prior runtime violation {i}",
                    action="recorded contract violation",
                    timestamp=process_started_at - 1.0,
                )
            )

        report = ia.run_integrity_audit(log=False)

        assert report["healthy"] is True
        assert report["degradations"]["recent_counts_by_subsystem"] == {}
        assert report["degradations"]["recent_scope"]["process_started_at"] == (
            process_started_at
        )
        assert not any("cognitive_contract" in item for item in report["concerns"])
    finally:
        tracker.reset()


def test_concern_verdict_keeps_current_process_contract_violations(monkeypatch):
    """Epoch scoping must preserve real violations produced by this runtime."""
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    process_started_at = time.time() - 30.0
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", process_started_at)
    try:
        for i in range(ia._DEGRADATION_CONCERN):
            tracker.record(
                DegradationRecord(
                    subsystem="cognitive_contract",
                    severity="warning",
                    error_type="RuntimeError",
                    error_message=f"current runtime violation {i}",
                    action="recorded contract violation",
                    timestamp=process_started_at + 1.0,
                )
            )

        report = ia.run_integrity_audit(log=False)

        assert report["healthy"] is False
        assert report["degradations"]["recent_counts_by_subsystem"] == {
            "cognitive_contract": {"warning": ia._DEGRADATION_CONCERN}
        }
        assert any(
            f"cognitive_contract: {ia._DEGRADATION_CONCERN} degradations" in item
            for item in report["concerns"]
        )
    finally:
        tracker.reset()


def test_integrity_concern_logs_one_incident_not_every_refresh(monkeypatch, caplog):
    """A level-triggered health concern must not become a warning storm."""
    import logging

    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    ia._reset_integrity_incident()
    process_started_at = time.time() - 30.0
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", process_started_at)
    increments: list[str] = []

    class Metrics:
        def increment_counter(self, name):
            increments.append(name)

    monkeypatch.setattr("core.observability.metrics.get_metrics", lambda: Metrics())
    try:
        for i in range(ia._DEGRADATION_CONCERN):
            tracker.record(
                DegradationRecord(
                    subsystem="cognitive_engine",
                    severity="warning",
                    error_type="RuntimeError",
                    error_message=f"failure {i}",
                    action="recorded",
                    timestamp=time.time(),
                )
            )

        with caplog.at_level(logging.INFO, logger="Aura.IntegrityAudit"):
            first = ia.run_integrity_audit()
            second = ia.run_integrity_audit()
            third = ia.run_integrity_audit()

        warnings = [
            record
            for record in caplog.records
            if record.levelno >= logging.WARNING
            and "runtime concern incident" in record.getMessage()
        ]
        assert len(warnings) == 1
        assert increments == ["integrity_concern_total"]
        assert first["integrity_incident"]["warning_required"] is True
        assert second["integrity_incident"]["warning_required"] is False
        assert third["integrity_incident"]["warning_required"] is False
        assert all(report["healthy"] is False for report in (first, second, third))
        assert all(
            report["integrity_incident"]["active"] is True
            for report in (first, second, third)
        )
    finally:
        tracker.reset()
        ia._reset_integrity_incident()


def test_integrity_incident_relogs_material_worsening_and_new_subsystem(
    monkeypatch, caplog
):
    import logging

    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    ia._reset_integrity_incident()
    process_started_at = time.time() - 30.0
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", process_started_at)

    def add(subsystem: str, count: int) -> None:
        for i in range(count):
            tracker.record(
                DegradationRecord(
                    subsystem=subsystem,
                    severity="warning",
                    error_type="RuntimeError",
                    error_message=f"{subsystem} failure {i}",
                    action="recorded",
                    timestamp=time.time(),
                )
            )

    try:
        add("cognitive_engine", ia._DEGRADATION_CONCERN)
        with caplog.at_level(logging.WARNING, logger="Aura.IntegrityAudit"):
            first = ia.run_integrity_audit()
            add("cognitive_engine", ia._DEGRADATION_CONCERN - 1)
            stable = ia.run_integrity_audit()
            add("mlx_client", ia._DEGRADATION_CONCERN)
            expanded = ia.run_integrity_audit()
            add("cognitive_engine", ia._DEGRADATION_CONCERN)
            worsened = ia.run_integrity_audit()

        warnings = [
            record
            for record in caplog.records
            if "runtime concern incident" in record.getMessage()
        ]
        assert len(warnings) == 3
        assert first["integrity_incident"]["new_subsystems"] == ["cognitive_engine"]
        assert stable["integrity_incident"]["warning_required"] is False
        assert expanded["integrity_incident"]["new_subsystems"] == ["mlx_client"]
        assert worsened["integrity_incident"]["materially_worsened_subsystems"] == [
            "cognitive_engine"
        ]
    finally:
        tracker.reset()
        ia._reset_integrity_incident()


def test_integrity_incident_reports_recovery_once_and_rearms(monkeypatch, caplog):
    import logging

    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    ia._reset_integrity_incident()
    process_started_at = time.time() - 30.0
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", process_started_at)
    try:
        for i in range(ia._DEGRADATION_CONCERN):
            tracker.record(
                DegradationRecord(
                    subsystem="cognitive_engine",
                    severity="warning",
                    error_type="RuntimeError",
                    error_message=f"failure {i}",
                    action="recorded",
                    timestamp=time.time(),
                )
            )
        with caplog.at_level(logging.INFO, logger="Aura.IntegrityAudit"):
            ia.run_integrity_audit()
            tracker.reset()
            recovered = ia.run_integrity_audit()
            stable = ia.run_integrity_audit()
            for i in range(ia._DEGRADATION_CONCERN):
                tracker.record(
                    DegradationRecord(
                        subsystem="cognitive_engine",
                        severity="warning",
                        error_type="RuntimeError",
                        error_message=f"recurrence {i}",
                        action="recorded",
                        timestamp=time.time(),
                    )
                )
            recurrence = ia.run_integrity_audit()

        recovery_messages = [
            record.getMessage()
            for record in caplog.records
            if "runtime concern recovered" in record.getMessage()
        ]
        warning_messages = [
            record.getMessage()
            for record in caplog.records
            if "runtime concern incident" in record.getMessage()
        ]
        assert len(recovery_messages) == 1
        assert len(warning_messages) == 2
        assert recovered["integrity_incident"]["fully_recovered"] is True
        assert stable["integrity_incident"]["fully_recovered"] is False
        assert recurrence["integrity_incident"]["warning_required"] is True
    finally:
        tracker.reset()
        ia._reset_integrity_incident()


def test_unavailable_process_epoch_does_not_hide_recent_violations(monkeypatch):
    """Missing epoch evidence falls back to the real trailing window."""
    from core.runtime.errors import DegradationRecord, get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", 0.0)
    try:
        for i in range(ia._DEGRADATION_CONCERN):
            tracker.record(
                DegradationRecord(
                    subsystem="cognitive_contract",
                    severity="warning",
                    error_type="RuntimeError",
                    error_message=f"unscoped current violation {i}",
                    action="recorded contract violation",
                    timestamp=time.time() - 1.0,
                )
            )

        report = ia.run_integrity_audit(log=False)

        assert report["healthy"] is False
        assert report["degradations"]["recent_scope"]["kind"] == (
            "trailing_window_epoch_unavailable"
        )
        assert any("cognitive_contract" in item for item in report["concerns"])
    finally:
        tracker.reset()


def test_integrity_session_epoch_advances_only_after_lifespan_restart(monkeypatch):
    class ReadModel:
        def __init__(self):
            self.starts = 0
            self.closes = 0

        def start(self):
            self.starts += 1
            return True

        def close(self):
            self.closes += 1

    model = ReadModel()
    monkeypatch.setattr(ia, "_INTEGRITY_READ_MODEL", model)
    monkeypatch.setattr(ia, "_PROCESS_STARTED_AT", 100.0)
    monkeypatch.setattr(ia, "_SESSION_STARTED_AT", None)
    monkeypatch.setattr(ia, "_SESSION_ACTIVE", False)
    monkeypatch.setattr(ia, "_SESSION_GENERATION", 0)

    assert ia.start_integrity_read_model() is True
    assert ia._runtime_epoch_started_at() == 100.0
    ia._observe_integrity_incident({"cognitive_engine": ia._DEGRADATION_CONCERN})
    assert ia._ACTIVE_CONCERN_COUNTS == {
        "cognitive_engine": ia._DEGRADATION_CONCERN
    }

    # A duplicate prewarm belongs to the same lifespan and keeps its evidence.
    assert ia.start_integrity_read_model() is True
    assert ia._runtime_epoch_started_at() == 100.0
    assert ia._ACTIVE_CONCERN_COUNTS

    ia.stop_integrity_read_model()
    monkeypatch.setattr(ia.time, "time", lambda: 250.0)
    assert ia.start_integrity_read_model() is True
    assert ia._runtime_epoch_started_at() == 250.0
    assert ia._ACTIVE_CONCERN_COUNTS == {}
    assert model.starts == 3
    assert model.closes == 1
