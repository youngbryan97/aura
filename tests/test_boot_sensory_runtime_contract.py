import sys
import types
from pathlib import Path

import pytest


class Service:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False

    async def start(self):
        self.started = True


def test_boot_sensory_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/orchestrator/mixins/boot/boot_sensory.py")) == []


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _patch_container(monkeypatch):
    import core.orchestrator.mixins.boot.boot_sensory as boot_sensory

    registered = {}

    def _register_instance(name, instance, *args, **kwargs):
        registered[name] = instance

    monkeypatch.setattr(boot_sensory.ServiceContainer, "register_instance", staticmethod(_register_instance))
    return registered


@pytest.mark.asyncio
async def test_sensory_boot_records_failed_modality_and_keeps_other_lanes(monkeypatch):
    from core.orchestrator.mixins.boot.boot_sensory import BootSensoryMixin

    class BrokenEars:
        def __init__(self):
            reason = "audio device unavailable"
            raise RuntimeError(reason)

    class ReasoningQueue(Service):
        async def start(self):
            self.started = True

    _install_module(monkeypatch, "core.senses.ears", SovereignEars=BrokenEars)
    _install_module(monkeypatch, "core.senses.screen_vision", LocalVision=Service)
    _install_module(monkeypatch, "core.terminal_monitor", get_terminal_monitor=lambda: Service())
    _install_module(monkeypatch, "core.adaptation.immune_system", ImmuneSystem=Service)
    _install_module(monkeypatch, "core.utils.sanitizer", BloodBrainBarrier=Service)
    _install_module(monkeypatch, "core.brain.reasoning_queue", get_reasoning_queue=lambda: ReasoningQueue())
    _install_module(monkeypatch, "core.senses.sensory_instincts", SensoryInstincts=Service)
    registered = _patch_container(monkeypatch)
    host = type("Host", (BootSensoryMixin,), {})()

    await host._init_sensory_systems()

    report = host.sensory_boot
    assert "ears" in report["degraded"]
    assert set(report["completed"]) >= {
        "vision",
        "terminal_monitor",
        "immune_barriers",
        "reasoning_queue",
        "sensory_instincts",
    }
    assert "vision_engine" in registered
    assert "terminal_monitor" in registered
    assert host.terminal_monitor is registered["terminal_monitor"]
    assert host.instincts is not None


@pytest.mark.asyncio
async def test_voice_boot_runs_inline_when_task_tracker_fails(monkeypatch):
    from core.orchestrator.mixins.boot.boot_sensory import BootSensoryMixin

    class Voice(Service):
        async def ensure_tts_async(self):
            self.started = True

    class BrokenTracker:
        def track(self, task, *, name):
            reason = f"{name}:tracker unavailable"
            raise RuntimeError(reason)

    _install_module(monkeypatch, "core.senses.voice_engine", get_voice_engine=lambda: Voice())
    _install_module(monkeypatch, "core.brain.multimodal_orchestrator", MultimodalOrchestrator=Service)
    _install_module(monkeypatch, "core.utils.task_tracker", get_task_tracker=lambda: BrokenTracker())
    registered = _patch_container(monkeypatch)
    host = type("Host", (BootSensoryMixin,), {})()

    await host._init_voice_subsystem()

    report = host.sensory_boot
    assert "voice_task_tracker" in report["degraded"]
    assert "voice_engine" in report["completed"]
    assert "multimodal_orchestrator" in report["completed"]
    assert registered["voice_engine"].started is True
    assert "multimodal_orchestrator" in registered
