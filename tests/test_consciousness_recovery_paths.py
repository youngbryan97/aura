from __future__ import annotations

import asyncio
import sys
import types
from collections import deque

import numpy as np

from core.consciousness import (
    affective_steering,
    aura_protocol,
    consciousness_bridge,
    executive_closure,
    free_energy,
    neural_mesh,
    neurochemical_system,
    parallel_branches,
    phi_core,
    substrate_authority,
)
from core.consciousness.affective_steering import SteeringVectorLibrary, SubstrateSyncThread
from core.consciousness.aura_protocol import AuraProtocolClient, build_message_from_state
from core.consciousness.consciousness_bridge import ConsciousnessBridge
from core.consciousness.executive_closure import ExecutiveClosureEngine
from core.consciousness.free_energy import FreeEnergyEngine
from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.parallel_branches import BranchManager
from core.consciousness.phi_core import PhiCore
from core.consciousness.structural_opacity import StructuralOpacityMonitor
from core.consciousness.substrate_authority import (
    ActionCategory,
    AuthorizationDecision,
    SubstrateAuthority,
    SubstrateVerdict,
)


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
