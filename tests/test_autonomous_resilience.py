from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.adaptation.autonomous_resilience import (
    IntegrationAuditor,
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
