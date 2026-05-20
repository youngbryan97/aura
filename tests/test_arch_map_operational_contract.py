from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO

from tools import arch_map


def test_operational_surface_scanner_maps_sensitive_authority_calls(tmp_path, monkeypatch):
    root = tmp_path
    core = root / "core"
    work = core / "agency"
    work.mkdir(parents=True)
    source = work / "runner.py"
    source.write_text(
        "\n".join(
            [
                "import requests",
                "import subprocess",
                "from core.will import get_will",
                "",
                "async def run(memory_facade, llm):",
                "    will_decision = get_will().decide(content='x', source='test')",
                "    await memory_facade.add_memory('durable memory write')",
                "    await llm.generate('model request')",
                "    subprocess.run(['true'])",
                "    requests.get('https://example.com')",
                "    return will_decision",
            ]
        ),
        encoding="utf-8",
    )
    owner = core / "will.py"
    owner.write_text(
        "def owner_path(will):\n    return will.decide(content='x', source='owner')\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(arch_map, "ROOT", root)
    monkeypatch.setattr(arch_map, "CORE", core)

    calls = arch_map.analyze_operational_surfaces(source)
    owner_calls = arch_map.analyze_operational_surfaces(owner)

    surfaces = {call.surface for call in calls}
    assert {
        "will_decision",
        "memory_write",
        "tool_execution",
        "llm_call",
        "external_io",
    }.issubset(surfaces)
    assert any(call.call == "requests.get" for call in calls)
    assert any(call.call == "subprocess.run" for call in calls)
    assert all(not call.owner_path for call in calls)
    assert any(call.surface == "will_decision" and call.owner_path for call in owner_calls)


def test_operational_authority_report_prints_reviewable_locations():
    calls = [
        arch_map.OperationalCall(
            surface="external_io",
            file="core/agency/runner.py",
            line=9,
            subsystem="agency",
            call="subprocess.run",
            source="subprocess.run(['true'])",
            owner_path=False,
        )
    ]

    output = StringIO()
    with redirect_stdout(output):
        arch_map.print_operational_authority_map(calls)

    rendered = output.getvalue()
    assert "Operational Authority Map" in rendered
    assert "External I/O" in rendered
    assert "Direct-call review candidates: 1" in rendered
    assert "core/agency/runner.py:9" in rendered


def test_architecture_report_is_machine_readable_and_persisted(tmp_path, monkeypatch):
    root = tmp_path
    core = root / "core"
    agency = core / "agency"
    runtime = core / "runtime"
    skills = root / "skills"
    agency.mkdir(parents=True)
    runtime.mkdir(parents=True)
    skills.mkdir(parents=True)
    (core / "__init__.py").write_text("", encoding="utf-8")
    (agency / "runner.py").write_text(
        "\n".join(
            [
                "import subprocess",
                "from core.runtime.errors import record_degradation",
                "from core.container import ServiceContainer",
                "",
                "def run(memory_facade):",
                "    ServiceContainer.get('inference_gate', default=None)",
                "    memory_facade.add_memory('durable memory write')",
                "    subprocess.run(['true'])",
                "    record_degradation('agency', RuntimeError('x'))",
            ]
        ),
        encoding="utf-8",
    )
    (runtime / "errors.py").write_text("def record_degradation(*a, **k): pass\n", encoding="utf-8")
    (core / "container.py").write_text(
        "class ServiceContainer:\n"
        "    @classmethod\n"
        "    def get(cls, name, default=None): return default\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(arch_map, "ROOT", root)
    monkeypatch.setattr(arch_map, "CORE", core)

    report = arch_map.build_architecture_report()

    assert report["schema"] == arch_map.ARCH_MAP_SCHEMA
    assert report["totals"]["subsystems"] >= 2
    assert report["service_container"]["get_call_count"] == 1
    assert report["operational_surfaces"]["memory_write"]["review_candidate_count"] == 1
    assert report["operational_surfaces"]["tool_execution"]["review_candidate_count"] == 1
    assert report["degradation"]["total_calls"] >= 1
    assert any(call["file"] == "core/agency/runner.py" for call in report["degradation"]["calls"])

    out = arch_map.write_report_artifacts(report, tmp_path / "artifacts" / "architecture")
    payload = json.loads((tmp_path / "artifacts" / "architecture" / "latest.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "artifacts" / "architecture" / "latest.md").read_text(encoding="utf-8")

    assert payload["schema"] == arch_map.ARCH_MAP_SCHEMA
    assert out["json"].endswith("latest.json")
    assert "Operational Authority Map" in markdown
