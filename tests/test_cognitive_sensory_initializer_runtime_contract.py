import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

LEARNED_SERVICE_MODULES = {
    "sentiment_tracker": ("core.cognitive.sentiment_tracker", "get_sentiment_tracker"),
    "anomaly_detector": ("core.cognitive.anomaly_detector", "AnomalyDetector"),
    "strange_loop": ("core.cognitive.strange_loop", "get_strange_loop"),
    "homeostatic_rl": ("core.cognitive.homeostatic_rl", "get_homeostatic_rl"),
    "topology_evolution": ("core.cognitive.topology_evolution", "TopologyEvolution"),
    "autopoiesis": ("core.cognitive.autopoiesis", "get_autopoiesis_engine"),
    "adaptive_immune_system": ("core.adaptation.adaptive_immunity", "get_adaptive_immune_system"),
    "autonomous_resilience_mesh": ("core.adaptation.autonomous_resilience", "get_autonomous_resilience_mesh"),
    "criticality_regulator": ("core.consciousness.criticality_regulator", "get_criticality_regulator"),
    "alife_dynamics": ("core.consciousness.alife_dynamics", "ALifeDynamics"),
    "alife_extensions": ("core.consciousness.alife_extensions", "ALifeExtensions"),
    "endogenous_fitness": ("core.consciousness.endogenous_fitness", "get_endogenous_fitness"),
}


class Service:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.initialized = False

    async def start(self):
        self.started = True

    async def initialize(self):
        self.initialized = True

    def setup_hooks(self, orchestrator):
        self.hooked = orchestrator


class SelfModelService(Service):
    @classmethod
    async def load(cls):
        return cls()


def test_cognitive_sensory_initializer_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/orchestrator/initializers/cognitive_sensory.py")) == []


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_success_modules(monkeypatch, *, will_engine_cls=Service):
    _install_module(monkeypatch, "core.self_model", SelfModel=SelfModelService)
    _install_module(monkeypatch, "core.brain.identity", IdentityService=Service)
    _install_module(monkeypatch, "core.soul", Soul=Service)
    _install_module(
        monkeypatch,
        "core.fictional_ai_synthesis",
        register_all_fictional_engines=lambda orchestrator: {"registered": True},
    )
    _install_module(monkeypatch, "core.brain.personality_engine", PersonalityEngine=Service)
    _install_module(monkeypatch, "core.managers.drive_controller", DriveController=Service)
    _install_module(monkeypatch, "core.senses.voice_engine", get_voice_engine=lambda: Service())
    _install_module(monkeypatch, "core.brain.multimodal_orchestrator", MultimodalOrchestrator=Service)
    _install_module(monkeypatch, "core.brain.composer_node", ComposerNode=Service)
    _install_module(monkeypatch, "core.guardians.memory_guard", MemoryGuard=Service)
    _install_module(monkeypatch, "core.soma.resilience_engine", ResilienceEngine=Service)
    _install_module(monkeypatch, "core.identity.drift_monitor", IdentityDriftMonitor=Service)
    _install_module(monkeypatch, "core.identity.spine", SpiritualSpine=Service)
    _install_module(monkeypatch, "core.self_modification.growth_ladder", GrowthLadder=Service)
    _install_module(monkeypatch, "core.memory.sovereign_pruner", SovereignPruner=Service)
    _install_module(monkeypatch, "core.guardians.governor", SystemGovernor=Service)
    _install_module(monkeypatch, "core.self.will_engine", WillEngine=will_engine_cls)
    _install_module(monkeypatch, "core.state.cellular_substrate", CellularSubstrate=Service)

    for _service_name, (module_path, factory_name) in LEARNED_SERVICE_MODULES.items():
        if factory_name[:1].isupper():
            _install_module(monkeypatch, module_path, **{factory_name: Service})
        else:
            _install_module(monkeypatch, module_path, **{factory_name: lambda: Service()})


def _patch_container(monkeypatch):
    import core.orchestrator.initializers.cognitive_sensory as cognitive_sensory

    registered = {}

    def _register_instance(name, instance, *args, **kwargs):
        registered[name] = instance

    def _get(name, default=None):
        return default

    monkeypatch.setattr(cognitive_sensory.ServiceContainer, "register_instance", staticmethod(_register_instance))
    monkeypatch.setattr(cognitive_sensory.ServiceContainer, "get", staticmethod(_get))
    return registered


@pytest.mark.asyncio
async def test_cognitive_sensory_initializer_returns_complete_boot_report(monkeypatch):
    from core.orchestrator.initializers.cognitive_sensory import init_cognitive_sensory_layer

    _install_success_modules(monkeypatch)
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert report["degraded"] == {}
    assert report["learned_services"] == {"registered": len(LEARNED_SERVICE_MODULES), "expected": len(LEARNED_SERVICE_MODULES)}
    assert "identity_personality" in report["completed"]
    assert "cellular_substrate" in report["completed"]
    assert registered["self_model"] is orchestrator.self_model
    assert registered["drive_engine"] is orchestrator.affect.drive_controller
    assert registered["cellular_substrate"] is orchestrator.cellular_substrate


@pytest.mark.asyncio
async def test_cognitive_sensory_initializer_continues_after_will_engine_failure(monkeypatch):
    from core.orchestrator.initializers.cognitive_sensory import init_cognitive_sensory_layer

    class BrokenWillEngine(Service):
        async def initialize(self):
            reason = "will engine unavailable"
            raise RuntimeError(reason)

    _install_success_modules(monkeypatch, will_engine_cls=BrokenWillEngine)
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert "will_engine" in report["degraded"]
    assert report["degraded"]["will_engine"]["severity"] == "critical"
    assert "cellular_substrate" in report["completed"]
    assert "cellular_substrate" in registered
    assert "will_engine" not in registered
