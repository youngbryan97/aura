from __future__ import annotations

import sys
import types

from core.ops.hot_reload import HotReloader


def _install_modules(monkeypatch, *names: str) -> None:
    for name in names:
        module = types.ModuleType(name)
        module.__file__ = f"/tmp/{name.replace('.', '/')}.py"
        monkeypatch.setitem(sys.modules, name, module)


def test_hot_reload_all_skips_runtime_owned_modules(monkeypatch):
    _install_modules(
        monkeypatch,
        "core.phases.dialogue",
        "core.memory.episodic_memory",
        "core.state.state_repository",
        "core.bus.local_pipe_bus",
        "core.senses.interaction_signals",
        "core.resilience.metrics_exporter",
        "core.executive.executive_core",
    )

    reloaded: list[str] = []

    def _reload(module):
        reloaded.append(module.__name__)
        return module

    monkeypatch.setattr("core.ops.hot_reload.importlib.reload", _reload)

    result = HotReloader(project_root="/tmp").reload_scope("all")

    assert result.ok
    assert "core.phases.dialogue" in reloaded
    assert "core.memory.episodic_memory" in reloaded
    assert "core.state.state_repository" not in reloaded
    assert "core.bus.local_pipe_bus" not in reloaded
    assert "core.senses.interaction_signals" not in reloaded
    assert "core.resilience.metrics_exporter" not in reloaded
    assert "core.executive.executive_core" not in reloaded


def test_hot_reload_explicit_scope_still_respects_protected_modules(monkeypatch):
    _install_modules(
        monkeypatch,
        "core.resilience.health_probe",
        "core.resilience.metrics_exporter",
        "core.resilience.circuit_breaker",
    )

    reloaded: list[str] = []

    def _reload(module):
        reloaded.append(module.__name__)
        return module

    monkeypatch.setattr("core.ops.hot_reload.importlib.reload", _reload)

    result = HotReloader(project_root="/tmp").reload_scope("resilience")

    assert result.ok
    assert "core.resilience.health_probe" in reloaded
    assert "core.resilience.metrics_exporter" not in reloaded
    assert "core.resilience.circuit_breaker" not in reloaded
    assert "core.resilience.metrics_exporter" in result.skipped
    assert "core.resilience.circuit_breaker" in result.skipped
