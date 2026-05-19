from __future__ import annotations

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
