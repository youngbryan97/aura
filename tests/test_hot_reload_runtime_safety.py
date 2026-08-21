from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.ops.hot_reload import HotReloader


def _install_modules(monkeypatch, *names: str) -> None:
    for name in names:
        module = types.ModuleType(name)
        module.__file__ = str(Path(tempfile.gettempdir()) / f"{name.replace('.', '/')}.py")
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
    # This contract exercises scope protection. Prefix validity has a separate
    # contract and depends on the real import graph, which this test replaces
    # with deliberately spec-less module doubles.
    monkeypatch.setattr(HotReloader, "_unmatched_prefixes", lambda self, scopes: [])

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


@pytest.mark.asyncio
async def test_kernel_hot_reboot_without_receipted_changes_is_a_true_noop(monkeypatch):
    from core.kernel.aura_kernel import AuraKernel

    cancelled = []
    kernel = AuraKernel.__new__(AuraKernel)
    kernel._background_tasks = [SimpleNamespace(cancel=lambda: cancelled.append(True))]
    monkeypatch.setattr(
        "core.runtime.backpressure.primary_inference_active",
        lambda: False,
    )

    result = await kernel.hot_reboot()

    assert result["reason"] == "no_applied_source_changes"
    assert result["reloaded"] == []
    assert cancelled == []


@pytest.mark.asyncio
async def test_kernel_hot_reboot_defers_while_primary_lane_is_active(monkeypatch):
    from core.kernel.aura_kernel import AuraKernel

    cancelled = []
    kernel = AuraKernel.__new__(AuraKernel)
    kernel._background_tasks = [SimpleNamespace(cancel=lambda: cancelled.append(True))]
    monkeypatch.setattr(
        "core.runtime.backpressure.primary_inference_active",
        lambda: True,
    )

    result = await kernel.hot_reboot(changed_files=("core/phases/dialogue.py",))

    assert result["reason"] == "primary_inference_active"
    assert result["restart_required"] is True
    assert cancelled == []


@pytest.mark.asyncio
async def test_kernel_hot_reboot_does_not_cancel_background_tasks_for_skipped_change(
    monkeypatch,
):
    from core.kernel.aura_kernel import AuraKernel

    cancelled = []
    kernel = AuraKernel.__new__(AuraKernel)
    kernel._background_tasks = [SimpleNamespace(cancel=lambda: cancelled.append(True))]
    monkeypatch.setattr(
        "core.runtime.backpressure.primary_inference_active",
        lambda: False,
    )
    monkeypatch.setattr(
        "core.ops.hot_reload.get_hot_reloader",
        lambda: SimpleNamespace(
            reload_file=lambda path: SimpleNamespace(
                reloaded=[],
                skipped=["core.kernel.aura_kernel"],
                orphan_risks=[],
                failed=[],
            )
        ),
    )

    result = await kernel.hot_reboot(changed_files=("core/kernel/aura_kernel.py",))

    assert result["restart_required"] is True
    assert result["reloaded"] == []
    assert cancelled == []
