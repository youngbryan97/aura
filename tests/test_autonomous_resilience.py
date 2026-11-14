from __future__ import annotations

from types import SimpleNamespace

from core.adaptation.autonomous_resilience import (
    IntegrationAuditor,
    RuntimeWatchdogAuditor,
    StaticFaultAuditor,
    VerifierGuidedRepairPipeline,
)


class _AutopoiesisStub:
    def __init__(self):
        self._health_fns = {}
        self.handlers = {}

    def register_component(self, name, health_fn):
        self._health_fns[name] = health_fn

    def register_repair_handler(self, strategy, component, handler):
        self.handlers[(strategy.value, component)] = handler


class _ServiceStub:
    def __init__(self):
        self.cache_cleared = False
        self.restarted = False

    def get_status(self):
        return {"overall_healthy": True}

    def clear_cache(self):
        self.cache_cleared = True

    def restart(self):
        self.restarted = True


class _ContainerStub:
    _aliases = {}
    _services = {
        "demo_service": SimpleNamespace(dependencies=["missing_dep"]),
    }


def test_static_fault_auditor_detects_zero_division_and_async_blocking(tmp_path):
    source = tmp_path / "core" / "demo_async.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "import time\n\n"
        "def ratio(total):\n"
        "    return total / 0\n\n"
        "async def runner():\n"
        "    while True:\n"
        "        time.sleep(1)\n",
        encoding="utf-8",
    )

    auditor = StaticFaultAuditor(tmp_path)
    findings = auditor.audit_file(source)
    kinds = {finding.kind for finding in findings}

    assert "definite_zero_division" in kinds
    assert "async_blocking_time_sleep" in kinds
    assert "async_busy_loop" in kinds


def test_integration_auditor_finds_dependency_gaps_and_auto_wires():
    autopoiesis = _AutopoiesisStub()
    service = _ServiceStub()

    def resolver(name: str):
        if name == "autopoiesis":
            return autopoiesis
        if name == "demo_service":
            return service
        return None

    auditor = IntegrationAuditor(
        service_resolver=resolver,
        container_cls=_ContainerStub,
    )

    service_report = auditor.audit_service_graph()
    assert service_report["finding_count"] == 1
    assert service_report["dependency_gaps"][0]["metadata"]["dependency"] == "missing_dep"

    wire_report = auditor.auto_wire_autopoiesis()
    assert "demo_service" in wire_report["health_probes_added"]
    assert autopoiesis._health_fns["demo_service"]() == 1.0
    assert ("clear_cache", "demo_service") in autopoiesis.handlers
    assert ("restart", "demo_service") in autopoiesis.handlers


def test_runtime_watchdog_marks_missing_stability_guardian_unhealthy():
    auditor = RuntimeWatchdogAuditor(service_resolver=lambda _name: None)

    snapshot = auditor._stability_snapshot()

    assert snapshot["healthy"] is False
    assert snapshot["status"] == "unavailable"
    assert snapshot["required_probe_missing"] is True


def test_runtime_watchdog_marks_stability_guardian_without_report_unhealthy():
    guardian = SimpleNamespace(
        _report_history=[],
        get_health_summary=lambda: {
            "status": "initializing",
            "healthy": True,
            "message": "legacy optimistic startup",
        },
    )
    auditor = RuntimeWatchdogAuditor(
        service_resolver=lambda name: guardian if name == "stability_guardian" else None
    )

    snapshot = auditor._stability_snapshot()

    assert snapshot["healthy"] is False
    assert snapshot["status"] == "initializing"
    assert snapshot["required_probe_missing"] is True


def test_runtime_watchdog_marks_malformed_stability_report_unhealthy():
    guardian = SimpleNamespace(
        _report_history=[
            SimpleNamespace(
                checks=[
                    SimpleNamespace(name="kernel", healthy=True, message="ok"),
                ]
            )
        ],
    )
    auditor = RuntimeWatchdogAuditor(
        service_resolver=lambda name: guardian if name == "stability_guardian" else None
    )

    snapshot = auditor._stability_snapshot()

    assert snapshot["healthy"] is False
    assert snapshot["status"] == "malformed_report"
    assert snapshot["required_probe_missing"] is True


class _CodeRepairStub:
    def __init__(self):
        self.calls = []

    async def repair_bug(self, file_path, line_number, diagnosis):
        self.calls.append((file_path, line_number, diagnosis))
        fix = SimpleNamespace(confidence="high")
        return True, fix, {"success": True}


class _SelfModifierStub:
    def __init__(self):
        self.code_repair = _CodeRepairStub()
        self.applied = []

    async def apply_fix(self, proposal, force=False, test_results=None):
        self.applied.append((proposal, force, test_results))
        return True


def test_verifier_guided_patch_pipeline_uses_self_modifier(tmp_path):
    target = tmp_path / "core" / "module.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("value = 1\n", encoding="utf-8")

    modifier = _SelfModifierStub()
    pipeline = VerifierGuidedRepairPipeline(
        base_dir=tmp_path,
        service_resolver=lambda name: modifier if name == "self_modification_engine" else None,
    )

    result = __import__("asyncio").run(
        pipeline.attempt_repair(
            error_signature="ZeroDivisionError",
            stack_trace=f'Traceback\n  File "{target}", line 1, in demo\n',
            context={"message": "division exploded"},
        )
    )

    assert result["attempted"] is True
    assert result["applied"] is True
    assert modifier.code_repair.calls[0][0] == "core/module.py"
    assert modifier.code_repair.calls[0][1] == 1
    assert modifier.applied


# ── CP126 remediation regressions ───────────────────────────────────────────


def test_static_auditor_refuses_paths_outside_the_repository(tmp_path):
    """An absolute path used to be returned as-is, letting the auditor scan —
    and mint repair candidates for — files outside the repo entirely."""
    import pytest

    repo = tmp_path / "repo"
    (repo / "core").mkdir(parents=True)
    outside = tmp_path / "outside" / "secret.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("x = 1\n", encoding="utf-8")

    auditor = StaticFaultAuditor(repo)

    with pytest.raises(ValueError, match="escapes the repository"):
        auditor._resolve_path(outside)
    with pytest.raises(ValueError, match="escapes the repository"):
        auditor._resolve_path("../outside/secret.py")
    # A legitimate in-repo path still resolves.
    assert auditor._resolve_path("core/thing.py") == (repo / "core" / "thing.py").resolve()


def test_repair_target_from_context_is_confined_to_the_repository(tmp_path):
    """A caller-supplied '../../' target was inserted with no containment check
    at all, and an absolute outside path raised an uncaught ValueError."""
    repo = tmp_path / "repo"
    repo.mkdir()
    pipeline = VerifierGuidedRepairPipeline(
        base_dir=repo, service_resolver=lambda name: None
    )

    escaped = pipeline._locate_target(
        "", context={"file_path": "../../etc/passwd", "line_number": 3}
    )
    assert escaped is None

    absolute = pipeline._locate_target(
        "", context={"file_path": str(tmp_path / "outside.py"), "line_number": 3}
    )
    assert absolute is None

    inside = pipeline._locate_target(
        "", context={"file_path": "core/mod.py", "line_number": 7}
    )
    assert inside == ("core/mod.py", 7)


def test_unavailable_watchdog_telemetry_reports_unknown_not_zero_risk():
    """Zero counts from unreadable instrumentation suppressed every finding, so
    the mesh reported a clean bill of health exactly when it had gone blind."""

    def _resolver(name):
        raise RuntimeError(f"{name} unavailable")

    auditor = RuntimeWatchdogAuditor(service_resolver=_resolver)
    snapshot = auditor._lock_watchdog_snapshot()

    assert snapshot["observed"] is False
    assert snapshot["active_count"] == 0   # still zero...
    # ...but the audit now says so out loud instead of staying silent.
    report = auditor.audit()
    kinds = {finding["kind"] for finding in report["findings"]}
    assert "telemetry_unavailable" in kinds
    assert report["threat_score"] > 0.0


def test_uninstrumented_service_is_not_scored_as_healthy():
    """0.55 sat in the healthy band, so a service exposing no health field was
    indistinguishable from a working one."""
    import core.adaptation.autonomous_resilience as module

    class _Opaque:
        def get_status(self):
            return {"some_unrelated_field": 1}

    module._reported_uninstrumented.clear()
    probe = IntegrationAuditor._health_probe_for(_Opaque())

    assert probe is not None
    assert probe() == module._UNKNOWN_HEALTH_SCORE
    assert module._UNKNOWN_HEALTH_SCORE < 0.55
    # The gap was surfaced rather than silently absorbed.
    assert "_Opaque" in module._reported_uninstrumented


def test_auto_wiring_never_builds_a_service_on_her_loop():
    """It runs every tick on her event loop. Live, 13 November 2026, resolving every registered name built the semantic
    runtime there; it wires only what is already built, and a service built later on the tick after."""
    autopoiesis = _AutopoiesisStub()
    built = {"demo_service": None}
    asked_to_build = []

    class _Container(_ContainerStub):
        @staticmethod
        def peek(name, default=None):
            return built.get(name) or default

    def resolver(name: str):
        if name == "autopoiesis":
            return autopoiesis
        asked_to_build.append(name)
        return _ServiceStub()

    auditor = IntegrationAuditor(service_resolver=resolver, container_cls=_Container)
    assert "demo_service" not in auditor.auto_wire_autopoiesis()["health_probes_added"]
    assert asked_to_build == []

    built["demo_service"] = _ServiceStub()
    assert "demo_service" in auditor.auto_wire_autopoiesis()["health_probes_added"]
