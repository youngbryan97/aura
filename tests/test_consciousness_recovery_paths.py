from __future__ import annotations

import asyncio
import sys
import types
from collections import deque

import numpy as np

from core.consciousness import (
    affective_steering,
    aura_protocol,
    closed_loop,
    consciousness_bridge,
    endogenous_fitness,
    executive_closure,
    experience_consolidator,
    free_energy,
    global_workspace,
    heartbeat,
    liquid_substrate,
    loop_monitor,
    mesh_cognition,
    mhaf_field,
    neural_mesh,
    neurochemical_system,
    parallel_branches,
    phi_core,
    precision_sampler,
    resource_stakes,
    stdp_learning,
    substrate_authority,
    system,
    time_dilation,
    unified_field,
)
from core.consciousness.affective_steering import SteeringVectorLibrary, SubstrateSyncThread
from core.consciousness.aura_protocol import AuraProtocolClient, build_message_from_state
from core.consciousness.closed_loop import OutputReceptor
from core.consciousness.consciousness_bridge import ConsciousnessBridge
from core.consciousness.endogenous_fitness import EndogenousFitness
from core.consciousness.executive_closure import ExecutiveClosureEngine
from core.consciousness.experience_consolidator import ExperienceConsolidator
from core.consciousness.free_energy import FreeEnergyEngine
from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace
from core.consciousness.heartbeat import CognitiveHeartbeat
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.loop_monitor import ConsciousnessLoopMonitor
from core.consciousness.mesh_cognition import MeshCognition
from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.parallel_branches import BranchManager
from core.consciousness.phi_core import PhiCore
from core.consciousness.precision_sampler import ActiveInferenceSampler
from core.consciousness.resource_stakes import ResourceStakesEngine
from core.consciousness.stdp_learning import STDPLearningEngine
from core.consciousness.structural_opacity import StructuralOpacityMonitor
from core.consciousness.substrate_authority import (
    ActionCategory,
    AuthorizationDecision,
    SubstrateAuthority,
    SubstrateVerdict,
)
from core.consciousness.system import ConsciousnessSystem
from core.consciousness.unified_field import FieldConfig, UnifiedField


class _FailingCallable:
    def __init__(self, message: str):
        self.message = message
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError(self.message)


def test_neurochemical_optional_links_emit_degradation_receipts(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        neurochemical_system,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    broken_container = types.SimpleNamespace(
        get=_FailingCallable("drive registry unavailable")
    )

    monkeypatch.setattr("core.container.ServiceContainer", broken_container)

    ncs = NeurochemicalSystem()
    ncs.on_success()
    ncs.on_novelty(0.4)

    assert recorded == [
        ("neurochemical_system", "RuntimeError"),
        ("neurochemical_system", "RuntimeError"),
    ]


def test_adaptive_mood_failures_fall_back_with_receipts(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        neurochemical_system,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    broken_adaptive_mood = types.SimpleNamespace(
        predict=_FailingCallable("mood predictor unavailable"),
        update_from_outcome=_FailingCallable("mood learner unavailable"),
    )

    monkeypatch.setattr(
        "core.consciousness.adaptive_mood.get_adaptive_mood",
        lambda: broken_adaptive_mood,
    )

    ncs = NeurochemicalSystem()

    mood = ncs.get_mood_vector()
    update = ncs.learn_mood_from_outcome({"valence": 0.2})

    assert {"valence", "arousal", "motivation", "sociality", "stress", "calm", "wakefulness"} <= set(mood)
    assert update == {}
    assert recorded == [
        ("neurochemical_system", "RuntimeError"),
        ("neurochemical_system", "RuntimeError"),
    ]


def test_consciousness_bridge_prediction_hook_records_failures(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        consciousness_bridge,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _Predictor:
        async def tick(self, **_kwargs):
            return "tick-ok"

        def get_surprise_signal(self):
            return 0.9

    bridge = ConsciousnessBridge.__new__(ConsciousnessBridge)
    bridge.neurochemical = types.SimpleNamespace(
        on_prediction_error=_FailingCallable("prediction coupling unavailable")
    )
    bridge._cs = types.SimpleNamespace(self_prediction=_Predictor())

    bridge._hook_neurochemical_events()

    result = asyncio.run(bridge._cs.self_prediction.tick())

    assert result == "tick-ok"
    assert recorded == [("consciousness_bridge", "RuntimeError")]


def test_consciousness_bridge_stop_handles_sync_and_async_components(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        consciousness_bridge,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _SyncStop:
        def __init__(self):
            self.calls = 0

        def stop(self):
            self.calls += 1

    class _AsyncFailingStop:
        def __init__(self):
            self.calls = 0

        async def stop(self):
            self.calls += 1
            raise RuntimeError("stop unavailable")

    sync_stop = _SyncStop()
    failing_stop = _AsyncFailingStop()
    bridge = ConsciousnessBridge.__new__(ConsciousnessBridge)
    bridge._running = True
    bridge._task = None
    bridge.unified_will = sync_stop
    bridge.substrate_evolution = failing_stop
    bridge.unified_field = None
    bridge.oscillatory_binding = None
    bridge.interoception = None
    bridge.neurochemical = None
    bridge.neural_mesh = None

    asyncio.run(bridge.stop())

    assert bridge._running is False
    assert sync_stop.calls == 1
    assert failing_stop.calls == 1
    assert recorded == [("consciousness_bridge", "RuntimeError")]


def test_consciousness_bridge_somatic_gate_hook_is_idempotent_and_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        consciousness_bridge,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        consciousness_bridge.ServiceContainer,
        "get",
        lambda _name, default=None: default,
    )

    class _Workspace:
        def __init__(self):
            self.submits = 0

        async def submit(self, _candidate):
            self.submits += 1
            return True

    class _Authority:
        def __init__(self):
            self.calls = 0

        def authorize(self, **_kwargs):
            self.calls += 1
            raise RuntimeError("authority unavailable")

    workspace = _Workspace()
    authority = _Authority()
    bridge = ConsciousnessBridge.__new__(ConsciousnessBridge)
    bridge._cs = types.SimpleNamespace(global_workspace=workspace)
    bridge.substrate_authority = authority

    bridge._hook_somatic_into_gwt()
    installed_submit = workspace.submit
    bridge._hook_somatic_into_gwt()
    candidate = types.SimpleNamespace(
        source="curiosity_engine",
        content="inspect an unexpected signal",
        effective_priority=0.8,
        priority=0.8,
    )

    assert workspace.submit is installed_submit
    assert asyncio.run(workspace.submit(candidate)) is True
    assert workspace.submits == 1
    assert authority.calls == 1
    assert recorded == [("consciousness_bridge", "RuntimeError")]


def test_consciousness_bridge_neurochemical_hook_supports_sync_tick_once():
    class _Predictor:
        def __init__(self):
            self.tick_calls = 0

        def tick(self, **_kwargs):
            self.tick_calls += 1
            return "tick-ok"

        def get_surprise_signal(self):
            return 0.5

    class _Neurochemical:
        def __init__(self):
            self.prediction_errors: list[float] = []

        def on_prediction_error(self, surprise):
            self.prediction_errors.append(surprise)

    predictor = _Predictor()
    neurochemical = _Neurochemical()
    bridge = ConsciousnessBridge.__new__(ConsciousnessBridge)
    bridge.neurochemical = neurochemical
    bridge._cs = types.SimpleNamespace(self_prediction=predictor)

    bridge._hook_neurochemical_events()
    installed_tick = predictor.tick
    bridge._hook_neurochemical_events()

    assert predictor.tick is installed_tick
    assert asyncio.run(predictor.tick()) == "tick-ok"
    assert predictor.tick_calls == 1
    assert neurochemical.prediction_errors == [0.5]


def test_consciousness_bridge_status_reports_authority_and_will_layers():
    class _Status:
        def get_status(self):
            return {"status": "online"}

    bridge = ConsciousnessBridge.__new__(ConsciousnessBridge)
    bridge.neural_mesh = _Status()
    bridge.neurochemical = _Status()
    bridge.interoception = _Status()
    bridge.oscillatory_binding = _Status()
    bridge.somatic_gate = _Status()
    bridge.unified_field = _Status()
    bridge.substrate_evolution = _Status()
    bridge.substrate_authority = _Status()
    bridge.unified_will = _Status()
    bridge._running = False
    bridge._tick_count = 3
    bridge._start_time = 0.0
    bridge._boot_errors = []

    status = bridge.get_status()

    assert status["layers_active"] == 9
    assert status["layers_total"] == 9
    assert status["components"]["substrate_authority"] == {"status": "online"}
    assert status["components"]["unified_will"] == {"status": "online"}


def test_consciousness_bridge_lookup_and_dispatch_failures_are_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        consciousness_bridge,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        consciousness_bridge.ServiceContainer,
        "get",
        _FailingCallable("service lookup unavailable"),
    )

    class _ClosedLoop:
        def __init__(self):
            self.calls = 0

        def is_running(self):
            return True

        def call_soon_threadsafe(self, _callback):
            self.calls += 1
            raise RuntimeError("loop closed")

    loop = _ClosedLoop()
    bridge = ConsciousnessBridge.__new__(ConsciousnessBridge)
    bridge._cs = types.SimpleNamespace(
        liquid_substrate="substrate-fallback",
        global_workspace="workspace-fallback",
    )
    bridge.substrate_evolution = types.SimpleNamespace()
    bridge._loop = loop

    assert bridge._get_substrate() == "substrate-fallback"
    assert bridge._get_workspace() == "workspace-fallback"
    bridge._dispatch_micro_evolve("coherence_collapse", 0.9)

    assert recorded == [
        ("consciousness_bridge", "RuntimeError"),
        ("consciousness_bridge", "RuntimeError"),
        ("consciousness_bridge", "RuntimeError"),
    ]
    assert loop.calls == 1


def test_free_energy_entropy_fallback_records_degradation(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        free_energy,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        free_energy.psutil,
        "cpu_percent",
        _FailingCallable("cpu telemetry unavailable"),
    )

    engine = FreeEnergyEngine()

    assert engine._compute_system_entropy() == 0.3
    assert recorded == [("free_energy", "RuntimeError")]


def test_executive_closure_substrate_reads_record_recoverable_failures(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        executive_closure,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    def service_get(name, default=None):
        services = {
            "unified_field": types.SimpleNamespace(
                get_experiential_quality=_FailingCallable("field unavailable")
            ),
            "neurochemical_system": types.SimpleNamespace(
                get_mood_vector=_FailingCallable("chemistry unavailable")
            ),
            "embodied_interoception": types.SimpleNamespace(
                get_body_budget=_FailingCallable("body budget unavailable")
            ),
        }
        return services.get(name, default)

    monkeypatch.setattr(executive_closure.ServiceContainer, "get", service_get)

    state = types.SimpleNamespace(
        motivation=types.SimpleNamespace(budgets={}),
        soma=types.SimpleNamespace(hardware={}),
        affect=types.SimpleNamespace(social_hunger=0.5),
    )
    pressures = ExecutiveClosureEngine()._compute_pressures(
        state,
        homeostasis_status={"will_to_live": 1.0, "metabolism": 1.0},
        closed_loop_status={"free_energy": 0.1},
        prediction_error=0.2,
    )

    assert {"stability", "integrity", "curiosity", "social", "growth"} <= set(pressures)
    assert recorded == [
        ("executive_closure", "RuntimeError"),
        ("executive_closure", "RuntimeError"),
        ("executive_closure", "RuntimeError"),
    ]


def test_executive_closure_completion_checks_record_failures(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        executive_closure,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        executive_closure.ServiceContainer,
        "get",
        lambda name, default=None: types.SimpleNamespace(
            get_all_active=_FailingCallable("verifier unavailable")
        )
        if name == "task_commitment_verifier"
        else default,
    )

    class _Cognition:
        @property
        def modifiers(self):
            self._seen = True
            raise RuntimeError("modifiers unavailable")

    state = types.SimpleNamespace(cognition=_Cognition())

    assert ExecutiveClosureEngine()._task_completion_observed(state) is False
    assert recorded == [
        ("executive_closure", "RuntimeError"),
        ("executive_closure", "RuntimeError"),
    ]


def test_affective_steering_source_fallback_records_degradation(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        affective_steering,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    library = SteeringVectorLibrary.__new__(SteeringVectorLibrary)
    library._cache_dir = types.SimpleNamespace(
        resolve=_FailingCallable("vector path unavailable")
    )

    assert library._infer_source() == "configured_caa"
    assert recorded == [("affective_steering", "RuntimeError")]


def test_affective_steering_live_source_annotation_failures_are_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        affective_steering,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: types.SimpleNamespace(
            get_mood_vector=lambda: {"arousal": 0.4, "coherence": 0.8}
        )
        if name == "neurochemical_system"
        else default,
    )

    class _ReadOnlySourceHook:
        def update_substrate(self, moods):
            self.moods = moods

        @property
        def substrate_source(self):
            return ""

        @substrate_source.setter
        def substrate_source(self, value):
            self.last_attempted_source = value
            raise RuntimeError("source annotation unavailable")

    hook = _ReadOnlySourceHook()
    thread = SubstrateSyncThread(
        [hook],
        types.SimpleNamespace(
            governor=types.SimpleNamespace(compute_alpha=lambda *_args: 0.2),
            telemetry=types.SimpleNamespace(alpha=0.0),
        ),
    )
    thread._running = True

    def stop_after_one_sleep(_seconds):
        thread._running = False

    monkeypatch.setattr(affective_steering.time, "sleep", stop_after_one_sleep)

    thread._loop()

    assert hook.moods == {"arousal": 0.4, "coherence": 0.8}
    assert hook.last_attempted_source == "live_mood"
    assert recorded == [("affective_steering", "RuntimeError")]


def test_neural_mesh_foreground_lane_failure_records_and_allows_plasticity(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        neural_mesh,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        _FailingCallable("service registry unavailable"),
    )

    assert neural_mesh.NeuralMesh._foreground_request_active() is False
    assert recorded == [("neural_mesh", "RuntimeError")]


def test_aura_protocol_identity_read_failure_records_and_preserves_message(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        aura_protocol,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    def service_get(name, default=None):
        if name == "unified_will":
            return types.SimpleNamespace(get_status=_FailingCallable("will unavailable"))
        return default

    monkeypatch.setattr(aura_protocol.ServiceContainer, "get", service_get)

    message = build_message_from_state("coordinate continuity", identity_name="FallbackAura")

    assert message.intent == "coordinate continuity"
    assert message.source_identity == "FallbackAura"
    assert recorded == [("aura_protocol", "RuntimeError")]


def test_aura_protocol_client_disconnect_records_writer_close_failure(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        aura_protocol,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _Writer:
        def close(self):
            self.closed = True

        async def wait_closed(self):
            self.waited = True
            raise RuntimeError("close handshake failed")

    client = AuraProtocolClient()
    client._writer = _Writer()

    asyncio.run(client.disconnect())

    assert client._writer is None
    assert recorded == [("aura_protocol", "RuntimeError")]


def test_parallel_branch_event_publish_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        parallel_branches,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    import core.event_bus as event_bus

    monkeypatch.setattr(event_bus, "get_event_bus", _FailingCallable("event bus unavailable"))

    BranchManager()._publish_event("branch.test", {"branch_id": "br_test"})

    assert recorded == [("parallel_branches", "RuntimeError")]


def test_structural_opacity_uses_weight_topology_as_readout():
    monitor = StructuralOpacityMonitor(neuron_count=16, n_perturbations=5)
    x = np.linspace(-0.4, 0.4, 16)
    weights = np.eye(16) * 0.2

    signature = monitor.measure(x, weights)

    assert 0.0 <= signature.opacity_index <= 1.0
    assert 0.0 <= signature.causal_depth <= 1.0
    assert monitor._measurement_count == 1


def test_endogenous_fitness_sampling_failures_keep_safe_defaults(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        endogenous_fitness,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _UnavailableSubstrate:
        idx_energy = 5

        @property
        def x(self):
            self.read_attempted = True
            raise RuntimeError("substrate energy unavailable")

    class _UnavailableFreeEnergy:
        @property
        def _current(self):
            self.read_attempted = True
            raise RuntimeError("free energy unavailable")

    class _UnavailableHomeostasis:
        compute_vitality = _FailingCallable("vitality unavailable")

        @property
        def curiosity(self):
            self.read_attempted = True
            raise RuntimeError("curiosity unavailable")

    def service_get(name, default=None):
        services = {
            "liquid_substrate": _UnavailableSubstrate(),
            "homeostasis": _UnavailableHomeostasis(),
            "anomaly_detector": types.SimpleNamespace(
                get_threat_level=_FailingCallable("threat unavailable")
            ),
            "free_energy_engine": _UnavailableFreeEnergy(),
            "phi_core": types.SimpleNamespace(
                get_status=_FailingCallable("phi status unavailable")
            ),
            "affective_steering": types.SimpleNamespace(
                get_status=_FailingCallable("affect status unavailable")
            ),
        }
        if name == "ice_layer":
            return None
        return services.get(name, default)

    monkeypatch.setattr(endogenous_fitness.ServiceContainer, "get", service_get)

    fitness = EndogenousFitness()
    sampled = fitness._sample_system_state()
    vector = fitness._get_behavioral_state_vector()

    assert sampled == {
        "energy": 50.0,
        "vitality": 0.8,
        "threat_level": 0.0,
        "free_energy": 0.3,
        "entropy": 4.0,
        "phi": 1.0,
    }
    assert vector.shape == (7,)
    assert recorded == [
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
        ("endogenous_fitness", "RuntimeError"),
    ]


def test_substrate_authority_reader_and_audit_failures_are_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        substrate_authority,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    authority = SubstrateAuthority()

    class _BrokenChemistry:
        def __init__(self):
            self.calls: list[str] = []

        @property
        def chemicals(self):
            self.calls.append("chemicals")
            raise RuntimeError("chemistry snapshot unavailable")

        def on_frustration(self, *_args, **_kwargs):
            self.calls.append("on_frustration")
            raise RuntimeError("feedback unavailable")

    broken_chemistry = _BrokenChemistry()

    authority._field_ref = types.SimpleNamespace(
        get_coherence=_FailingCallable("field offline")
    )
    authority._somatic_ref = types.SimpleNamespace(
        evaluate=_FailingCallable("somatic gate offline")
    )
    authority._neurochemical_ref = broken_chemistry

    assert authority._get_field_coherence() == 0.5
    assert authority._get_somatic_state("content", "source", 0.5) == (0.0, 0.0, True)
    assert authority._get_neurochemical_constraints(ActionCategory.RESPONSE) == ("normal", [])
    authority._neurochemical_feedback(AuthorizationDecision.BLOCK, ActionCategory.RESPONSE)

    import core.consciousness.authority_audit as authority_audit

    monkeypatch.setattr(
        authority_audit,
        "get_audit",
        _FailingCallable("audit unavailable"),
    )
    authority._record(
        SubstrateVerdict(
            decision=AuthorizationDecision.ALLOW,
            reason="test",
            field_coherence=0.6,
            somatic_approach=0.0,
            somatic_confidence=0.0,
            neurochemical_state="normal",
            body_budget_available=True,
            constraints=[],
        )
    )

    assert recorded == [
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
    ]


def test_phi_core_degradation_receipts_for_metric_and_surrogate_failures(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        phi_core,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    metrics_module = types.ModuleType("core.observability.metrics")
    metrics_module.get_metrics = _FailingCallable("metrics unavailable")
    monkeypatch.setitem(sys.modules, "core.observability.metrics", metrics_module)

    phi = PhiCore.__new__(PhiCore)
    assert phi._detect_disconnected_graph(np.zeros((3, 3))) == (False, 3, [1, 1, 1])

    phi._last_result = None
    phi._state_history = deque([0] * 20)
    phi.compute_surrogate_phi = _FailingCallable("surrogate unavailable")

    assert phi.get_live_phi() == 0.0
    assert recorded == [
        ("phi_core", "RuntimeError"),
        ("phi_core", "RuntimeError"),
    ]


def test_precision_sampler_top_p_fallback_records_mhaf_failure(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        precision_sampler,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(mhaf_field, "get_mhaf", _FailingCallable("mhaf unavailable"))

    sampler = ActiveInferenceSampler()

    assert sampler._compute_top_p() == 0.85
    assert recorded == [("precision_sampler", "RuntimeError")]


def test_stdp_nonfinite_metric_failure_is_recorded(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        stdp_learning,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(stdp_learning.np.linalg, "norm", lambda _matrix, ord=None: 0.0)

    metrics_module = types.ModuleType("core.observability.metrics")
    metrics_module.get_metrics = _FailingCallable("metrics unavailable")
    monkeypatch.setitem(sys.modules, "core.observability.metrics", metrics_module)

    engine = STDPLearningEngine(n_neurons=4)
    updated = engine.apply_to_connectivity(
        np.full((4, 4), np.nan),
        np.zeros((4, 4)),
    )

    assert np.isfinite(updated).all()
    assert recorded == [("stdp_learning", "RuntimeError")]


def test_experience_consolidator_deferral_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        experience_consolidator,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        _FailingCallable("container unavailable"),
    )

    consolidator = ExperienceConsolidator(cognitive_engine=None)

    assert consolidator._background_should_defer() is False
    assert recorded == [("experience_consolidator", "RuntimeError")]


def test_experience_consolidator_collects_metacognition_and_reflections(monkeypatch):
    crsm_module = types.ModuleType("core.consciousness.crsm")
    crsm_module.get_crsm = lambda: types.SimpleNamespace(_history=[])
    monkeypatch.setitem(sys.modules, "core.consciousness.crsm", crsm_module)

    bridge_module = types.ModuleType("core.consciousness.crsm_lora_bridge")
    bridge_module.get_crsm_lora_bridge = lambda: types.SimpleNamespace(_buffer=[])
    monkeypatch.setitem(sys.modules, "core.consciousness.crsm_lora_bridge", bridge_module)

    hot_module = types.ModuleType("core.consciousness.hot_engine")
    hot_module.get_hot_engine = lambda: types.SimpleNamespace(_history=[])
    monkeypatch.setitem(sys.modules, "core.consciousness.hot_engine", hot_module)

    class _Assessment:
        def to_dict(self):
            return {"task": "calibration", "confidence": 0.82}

    reflection = types.SimpleNamespace(
        content="A useful private synthesis.",
        impact_score=0.7,
        source_id="source-1",
        timestamp=123.0,
    )
    metacognition = types.SimpleNamespace(
        monitor=types.SimpleNamespace(reasoning_history=[_Assessment()]),
        reflector=types.SimpleNamespace(reflections=[reflection]),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: metacognition if name == "metacognition" else default,
    )

    reflection_module = types.ModuleType("core.conversation_reflection")
    reflection_module.get_reflector = lambda: types.SimpleNamespace(
        get_recent_reflections=lambda _count: [
            {"text": "Recent conversation insight.", "timestamp": 456.0, "mood": "focused"}
        ]
    )
    monkeypatch.setitem(sys.modules, "core.conversation_reflection", reflection_module)

    material = ExperienceConsolidator(cognitive_engine=None)._gather_material()

    assert material["metacognition"] == [{"task": "calibration", "confidence": 0.82}]
    assert [item["source"] for item in material["reflections"]] == [
        "source-1",
        "conversation_reflection",
    ]


def test_time_dilation_signal_failures_return_safe_defaults(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        time_dilation,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _BrokenFreeEnergy:
        def __init__(self):
            self.read_attempted = False

        @property
        def current(self):
            self.read_attempted = True
            raise RuntimeError("free energy unavailable")

    class _BrokenWorldState:
        def __init__(self):
            self.read_attempted = False

        @property
        def user_idle_seconds(self):
            self.read_attempted = True
            raise RuntimeError("world state unavailable")

    services = {
        "free_energy_engine": _BrokenFreeEnergy(),
        "drive_engine": types.SimpleNamespace(
            get_drive_vector=_FailingCallable("drive unavailable")
        ),
        "world_state": _BrokenWorldState(),
        "homeostatic_coupling": types.SimpleNamespace(
            get_modifiers=_FailingCallable("homeostasis unavailable")
        ),
    }
    monkeypatch.setattr(
        time_dilation.ServiceContainer,
        "get",
        lambda name, default=None: services.get(name, default),
    )

    signals = time_dilation.TimeDilationEngine()._gather_signals()

    assert signals.free_energy == 0.0
    assert signals.drive_urgency == 0.0
    assert signals.user_waiting is False
    assert signals.critical_maintenance is False
    assert recorded == [
        ("time_dilation", "RuntimeError"),
        ("time_dilation", "RuntimeError"),
        ("time_dilation", "RuntimeError"),
        ("time_dilation", "RuntimeError"),
    ]


def test_unified_field_gamma_failure_and_projection_recovery(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        unified_field,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _BrokenBinding:
        def __init__(self):
            self.phase_read_attempted = False

        @property
        def _gamma_phase(self):
            self.phase_read_attempted = True
            raise RuntimeError("gamma phase unavailable")

        @property
        def _gamma_amplitude(self):
            return 1.0

    field = UnifiedField(
        FieldConfig(
            dim=8,
            mesh_input_dim=2,
            chem_input_dim=2,
            binding_input_dim=2,
            intero_input_dim=2,
            substrate_input_dim=2,
            plasticity_interval=100,
        )
    )
    field._binding_ref = _BrokenBinding()
    field._tick()

    field.W_mesh = np.zeros((1, 1), dtype=np.float32)
    predictions = field.get_world_model_predictions()

    assert predictions["mesh"].shape == (2,)
    assert np.allclose(predictions["mesh"], 0.0)
    assert recorded == [
        ("unified_field", "RuntimeError"),
        ("unified_field", "ValueError"),
    ]


def test_mesh_cognition_signal_failures_are_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mesh_cognition,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _BrokenAffect:
        def __init__(self):
            self.read_attempted = False

        @property
        def valence(self):
            self.read_attempted = True
            raise RuntimeError("affect unavailable")

    services = {
        "liquid_substrate": types.SimpleNamespace(
            get_substrate_affect=lambda: {"valence": object()}
        ),
        "resource_stakes": types.SimpleNamespace(
            action_envelope=_FailingCallable("stakes envelope unavailable"),
        ),
        "global_workspace": types.SimpleNamespace(
            current_winner=_FailingCallable("workspace unavailable"),
        ),
    }
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: services.get(name, default),
    )

    signals = MeshCognition()._gather_signals(types.SimpleNamespace(affect=_BrokenAffect()))

    assert signals == {}
    assert recorded == [
        ("mesh_cognition", "RuntimeError"),
        ("mesh_cognition", "TypeError"),
        ("mesh_cognition", "RuntimeError"),
        ("mesh_cognition", "RuntimeError"),
    ]


def test_resource_stakes_signal_failures_are_visible(monkeypatch, tmp_path):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        resource_stakes,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda _name, default=None: types.SimpleNamespace(
            apply_event=_FailingCallable("neurochemical event unavailable")
        ),
    )

    stakes = ResourceStakesEngine(data_dir=tmp_path)
    stakes._signal_reward("unit")
    stakes._signal_stress("unit", 0.8)

    assert recorded == [
        ("resource_stakes", "RuntimeError"),
        ("resource_stakes", "RuntimeError"),
    ]


def test_liquid_substrate_affect_and_chaos_recovery_are_visible(monkeypatch, tmp_path):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        liquid_substrate,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    substrate = LiquidSubstrate(
        SubstrateConfig(neuron_count=4, state_file=tmp_path / "substrate_state.npy")
    )
    substrate.x = np.array([], dtype=np.float64)

    assert substrate.get_substrate_affect() == {
        "valence": 0.0,
        "arousal": 0.3,
        "dominance": 0.0,
        "energy": 0.5,
        "volatility": 0.0,
    }

    substrate.x = np.zeros(4, dtype=np.float64)
    substrate.W = np.zeros((4, 4), dtype=np.float64)
    substrate._chaos_engine = types.SimpleNamespace(
        tick=_FailingCallable("chaos unavailable")
    )
    substrate._step_torch_math(0.01)

    assert recorded == [
        ("liquid_substrate", "IndexError"),
        ("liquid_substrate", "RuntimeError"),
    ]


def test_liquid_substrate_gate_scar_failures_are_visible(monkeypatch, tmp_path):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        liquid_substrate,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: types.SimpleNamespace(
            authorize=_FailingCallable("authority unavailable")
        ) if name == "substrate_authority" else default,
    )

    scar_module = types.ModuleType("core.memory.scar_formation")
    scar_module.ScarDomain = types.SimpleNamespace(AUTHORITY_GATE_FAILURE="authority")
    scar_module.get_scar_formation = _FailingCallable("scar unavailable")
    monkeypatch.setitem(sys.modules, "core.memory.scar_formation", scar_module)

    substrate = LiquidSubstrate(
        SubstrateConfig(neuron_count=4, state_file=tmp_path / "substrate_state.npy")
    )
    asyncio.run(substrate.inject_stimulus(np.ones(4), weight=1.0))

    assert recorded == [
        ("liquid_substrate", "RuntimeError"),
        ("liquid_substrate", "RuntimeError"),
    ]


def test_closed_loop_output_lookup_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        closed_loop,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        _FailingCallable("container unavailable"),
    )

    result = OutputReceptor().receive_output("A wonderful curious signal with enough affect.")

    assert result is None
    assert recorded == [("closed_loop", "RuntimeError")]


def test_global_workspace_theory_arbitration_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        global_workspace,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(global_workspace.ServiceContainer, "get", lambda *_args, **_kwargs: None)

    peripheral_module = types.ModuleType("core.consciousness.peripheral_awareness")
    peripheral_module.get_peripheral_awareness_engine = lambda: types.SimpleNamespace(
        process_workspace_results=lambda **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "core.consciousness.peripheral_awareness", peripheral_module)

    unity_module = types.ModuleType("core.unity")
    unity_module.get_unity_runtime = lambda: types.SimpleNamespace(
        record_workspace_competition=lambda *_args, **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "core.unity", unity_module)

    emitter_module = types.ModuleType("core.thought_stream")
    emitter_module.get_emitter = lambda: types.SimpleNamespace(
        emit=lambda **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "core.thought_stream", emitter_module)

    arbitration_module = types.ModuleType("core.consciousness.theory_arbitration")
    arbitration_module.get_theory_arbitration = _FailingCallable("arbitration unavailable")
    monkeypatch.setitem(sys.modules, "core.consciousness.theory_arbitration", arbitration_module)

    async def run_workspace():
        workspace = GlobalWorkspace()
        await workspace.submit(CognitiveCandidate(content="ignite", source="unit", priority=1.0))
        return await workspace.run_competition()

    winner = asyncio.run(run_workspace())

    assert winner is not None
    assert recorded == [("global_workspace", "RuntimeError")]


def test_loop_monitor_stale_cache_heal_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        loop_monitor,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _StickyHeartbeat:
        def __init__(self):
            self._qualia_cache = None
            self.delete_attempts: list[str] = []

        def __delattr__(self, name):
            self.delete_attempts.append(name)
            raise RuntimeError("cache cannot be cleared")

    service_container = types.SimpleNamespace(
        get=lambda name, default=None: _StickyHeartbeat() if name == "heartbeat" else default
    )

    healed = ConsciousnessLoopMonitor()._try_heal_stale_cache(service_container, object())

    assert healed is False
    assert recorded == [("loop_monitor", "RuntimeError")]


def test_heartbeat_time_dilation_failure_records_and_uses_fixed_interval(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        heartbeat,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        heartbeat.ServiceContainer,
        "get",
        lambda name, default=None: types.SimpleNamespace(
            evaluate=_FailingCallable("time dilation unavailable")
        )
        if name == "time_dilation"
        else default,
    )

    hb = CognitiveHeartbeat.__new__(CognitiveHeartbeat)

    assert hb._evaluate_tick_interval() == 1.0
    assert recorded == [("heartbeat", "RuntimeError")]


def test_heartbeat_mind_model_sync_invokes_live_pulse_and_records_failure(
    monkeypatch,
):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        heartbeat,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _MindModel:
        def __init__(self):
            self.calls = 0

        async def pulse(self):
            self.calls += 1

    hb = CognitiveHeartbeat.__new__(CognitiveHeartbeat)
    mind_model = _MindModel()
    asyncio.run(hb._sync_mind_model(mind_model, tick=1))
    asyncio.run(
        hb._sync_mind_model(
            types.SimpleNamespace(pulse=_FailingCallable("mind model unavailable")),
            tick=1,
        )
    )

    assert mind_model.calls == 1
    assert recorded == [("heartbeat", "RuntimeError")]


def test_heartbeat_predictive_feedback_closes_free_energy_loop(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        heartbeat,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _Predictive:
        def __init__(self):
            self.feedback = None

        async def accept_feedback(self, feedback):
            self.feedback = feedback

    hb = CognitiveHeartbeat.__new__(CognitiveHeartbeat)
    predictive = _Predictive()

    async def send_feedback():
        delivered = await hb._send_predictive_feedback(
            predictive,
            types.SimpleNamespace(
                free_energy=0.42,
                dominant_action="explore",
                valence=-0.2,
            ),
            surprise=0.31,
        )
        failed = await hb._send_predictive_feedback(
            types.SimpleNamespace(
                accept_feedback=_FailingCallable("feedback unavailable")
            ),
            types.SimpleNamespace(free_energy=0.1),
            surprise=0.2,
        )
        return delivered, failed

    delivered, failed = asyncio.run(send_feedback())

    assert delivered is True
    assert predictive.feedback == {
        "free_energy": 0.42,
        "dominant_action": "explore",
        "surprise": 0.31,
        "valence": -0.2,
    }
    assert failed is False
    assert recorded == [("heartbeat", "RuntimeError")]


def test_heartbeat_phi_fallback_records_core_failure_and_uses_substrate(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        heartbeat,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    def service_get(name, default=None):
        if name == "phi_core":
            return types.SimpleNamespace(
                get_live_phi=_FailingCallable("phi unavailable")
            )
        if name == "liquid_substrate":
            return types.SimpleNamespace(_current_phi=0.37)
        return default

    monkeypatch.setattr(heartbeat.ServiceContainer, "get", service_get)

    hb = CognitiveHeartbeat.__new__(CognitiveHeartbeat)

    assert hb._resolve_live_phi() == 0.37
    assert recorded == [("heartbeat", "RuntimeError")]


def test_heartbeat_qualia_metrics_failure_records_safe_empty(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        heartbeat,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        heartbeat.ServiceContainer,
        "get",
        _FailingCallable("liquid state unavailable"),
    )

    hb = CognitiveHeartbeat.__new__(CognitiveHeartbeat)
    hb.orch = types.SimpleNamespace()

    state = asyncio.run(hb._gather_state())

    assert state["qualia_metrics"] == {}
    assert recorded == [("heartbeat", "RuntimeError")]


def test_consciousness_system_required_substrate_start_failure_allows_retry(
    monkeypatch,
):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        system,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _BrokenSubstrate:
        def __init__(self):
            self.start_calls = 0

        async def start(self):
            self.start_calls += 1
            raise RuntimeError("substrate start unavailable")

    substrate = _BrokenSubstrate()
    cs = ConsciousnessSystem.__new__(ConsciousnessSystem)
    cs._running = False
    cs.liquid_substrate = substrate

    try:
        asyncio.run(cs.start())
    except RuntimeError as exc:
        assert str(exc) == "substrate start unavailable"
    else:
        raise AssertionError("required substrate start failure was not raised")

    assert cs._running is False
    assert substrate.start_calls == 1
    assert recorded == [("system", "RuntimeError")]


def test_consciousness_system_stop_records_shutdown_failure_and_resets_running(
    monkeypatch,
):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        system,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    class _BrokenSubstrate:
        def __init__(self):
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            raise RuntimeError("substrate stop unavailable")

    substrate = _BrokenSubstrate()
    cs = ConsciousnessSystem.__new__(ConsciousnessSystem)
    cs._running = True
    cs._task = None
    cs.heartbeat = types.SimpleNamespace(stop=lambda: None)
    cs.bridge = None
    cs.closed_loop = None
    cs.branch_manager = None
    cs.aura_protocol = None
    cs.liquid_substrate = substrate

    asyncio.run(cs.stop())

    assert cs._running is False
    assert substrate.stop_calls == 1
    assert recorded == [("system", "RuntimeError")]
