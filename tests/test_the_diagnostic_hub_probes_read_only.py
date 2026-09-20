"""The diagnostic hub's linters are read-only probes, launched as such.

LIVE 2026-09-16: every fix the self-repair loop proposed failed at its own
sandbox — "Pyright guard unavailable: offline subprocess tooling bypass
denied while live governance is active: maintenance_tooling:diagnostic_hub".
The hub asked the gateway for the offline-tooling bypass, which exists for
CLI wrappers running outside the live runtime and fails closed inside it.
Ruff and pyright over one file change nothing; they are read-only probes,
and the gateway has a lane for those.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.resilience.diagnostic_hub import DiagnosticHub


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_until_its_work_is_done(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})

        class _Done:
            stdout = ""
            returncode = 0

        return _Done()


@pytest.mark.asyncio
async def test_ruff_and_pyright_are_launched_as_read_only_probes(monkeypatch, tmp_path):
    gateway = _Gateway()
    monkeypatch.setattr("core.resilience.diagnostic_hub.get_subprocess_gateway", lambda: gateway)
    hub = DiagnosticHub.__new__(DiagnosticHub)
    hub.code_base = Path(tmp_path)

    results = await hub.run_deep_diagnostic("probe.py")

    assert results["ruff"]["ok"] and results["pyright"]["ok"]
    assert len(gateway.calls) == 2
    for call in gateway.calls:
        assert call["read_only"] is True
        assert "offline_tooling" not in call
        assert call["source"].startswith("diagnostic_hub.")
        assert call["cpu_budget_s"] > 0
    assert "--output-format" in gateway.calls[0]["argv"], "ruff 0.15 has no --format"


def test_the_hub_no_longer_asks_for_the_offline_bypass() -> None:
    source = (Path(__file__).resolve().parents[1] / "core/resilience/diagnostic_hub.py").read_text(encoding="utf-8")
    assert "offline_tooling" not in source
    assert "maintenance_tooling:diagnostic_hub" not in source
