import asyncio
import builtins
from pathlib import Path

from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState
from tools.audit_degradation import analyze_file


class DummyKernel:
    cycle_count = 1


def test_cognitive_integration_degradation_audit_is_clean():
    assert analyze_file(Path("core/phases/cognitive_integration_phase.py")) == []


def test_subsystem_failure_is_written_into_tick_state():
    tracker = get_degradation_tracker()
    tracker.reset()
    phase = CognitiveIntegrationPhase(DummyKernel())
    state = AuraState()

    class BrokenHomeostat:
        def __init__(self):
            self.calls = 0

        def get_energy(self):
            self.calls += 1
            raise RuntimeError("homeostat unavailable")

    phase._homeostatic_rl = BrokenHomeostat()

    asyncio.run(phase._run_homeostatic_rl(state))

    degraded = state.response_modifiers["cognitive_integration_degraded"]
    assert degraded[-1]["method"] == "_run_homeostatic_rl"
    assert degraded[-1]["action"] == "continued tick with prior affect energy after homeostatic RL failed"
    recent = tracker.recent(subsystem="cognitive_integration_phase", limit=1)
    assert recent
    assert recent[0].severity == "degraded"
    assert recent[0].action == "continued tick with prior affect energy after homeostatic RL failed"


def test_service_resolution_failure_retries_on_future_ticks(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    phase = CognitiveIntegrationPhase(DummyKernel())

    real_import = builtins.__import__

    def fail_container_import(name, *args, **kwargs):
        if name == "core.container":
            raise ImportError("container not ready")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_container_import)

    phase._resolve_services()

    assert phase._resolved is False
    recent = tracker.recent(subsystem="cognitive_integration_phase", limit=1)
    assert recent
    assert recent[0].action == "deferred cognitive service resolution and skipped unresolved systems this tick"
