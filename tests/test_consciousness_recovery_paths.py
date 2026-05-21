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
    counterfactual_engine,
    endogenous_fitness,
    executive_closure,
    experience_consolidator,
    free_energy,
    global_workspace,
    heartbeat,
    homeostatic_coupling,
    liquid_substrate,
    liquid_substrate_bridge,
    loop_monitor,
    mesh_cognition,
    mhaf_field,
    multiple_drafts,
    neologism_engine,
    neural_mesh,
    neurochemical_system,
    oscillatory_binding,
    parallel_branches,
    phi_core,
    precision_sampler,
    resource_stakes,
    somatic_marker_gate,
    stdp_learning,
    substrate_authority,
    substrate_evolution,
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
from core.consciousness.mhaf import phi_estimator
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


def test_neurochemical_run_loop_survives_tick_failure(monkeypatch):
    recorded: list[tuple[str, str, str, dict | None]] = []

    def record(module, exc, **kwargs):
        recorded.append(
            (
                module,
                type(exc).__name__,
                kwargs.get("action", ""),
                kwargs.get("extra"),
            )
        )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(neurochemical_system, "record_degradation", record)
    monkeypatch.setattr(neurochemical_system.asyncio, "sleep", no_sleep)

    ncs = NeurochemicalSystem()
    attempts = {"count": 0}

    def fail_once_then_stop():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("metabolic tick unavailable")
        ncs._running = False

    ncs._metabolic_tick = fail_once_then_stop
    ncs._running = True

    asyncio.run(ncs._run_loop())

    assert attempts["count"] == 2
    assert ncs._running is False
    assert any(
        action
        == "kept NeurochemicalSystem loop alive after tick failure and normalized chemistry"
        and extra == {"consecutive_tick_failures": 1}
        for _, _, action, extra in recorded
    )


def test_neurochemical_corrupted_state_is_repaired(monkeypatch):
    recorded: list[tuple[str, str, str]] = []

    def record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs.get("action", "")))

    monkeypatch.setattr(neurochemical_system, "record_degradation", record)

    ncs = NeurochemicalSystem()
    dopamine = ncs.chemicals["dopamine"]
    dopamine.level = float("nan")
    dopamine.tonic_level = float("inf")
    dopamine.phasic_burst = -1.0
    dopamine.receptor_sensitivity = float("nan")

    ncs._metabolic_tick()

    assert 0.0 <= dopamine.level <= 1.0
    assert 0.0 <= dopamine.tonic_level <= 1.0
    assert 0.0 <= dopamine.phasic_burst <= 0.5
    assert 0.3 <= dopamine.receptor_sensitivity <= 2.0
    assert np.all(np.isfinite(ncs.get_mesh_modulation()))
    assert any(
        action == "normalized chemical state before interaction"
        for _, _, action in recorded
    )


def test_neurochemical_modulation_push_failure_is_structured(monkeypatch):
    recorded: list[tuple[str, str, str, dict | None]] = []

    def record(module, exc, **kwargs):
        recorded.append(
            (
                module,
                type(exc).__name__,
                kwargs.get("action", ""),
                kwargs.get("extra"),
            )
        )

    class _Mesh:
        def __init__(self):
            self.calls = 0

        def set_modulatory_state(self, *_args):
            self.calls += 1
            raise RuntimeError("mesh offline")

    monkeypatch.setattr(neurochemical_system, "record_degradation", record)

    ncs = NeurochemicalSystem()
    mesh = _Mesh()
    ncs._mesh_ref = mesh

    ncs._push_modulation()

    assert mesh.calls == 1
    assert any(
        action == "kept chemistry alive when mesh modulation push failed"
        and extra is not None
        and {"gain", "plasticity", "noise"} <= set(extra)
        for _, _, action, extra in recorded
    )


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


def test_somatic_marker_gate_recovery_paths_are_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        somatic_marker_gate,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    gate = somatic_marker_gate.SomaticMarkerGate()
    gate._mesh_ref = object()
    gate._outcome_patterns.extend(
        somatic_marker_gate.OutcomeRecord(
            pattern=np.zeros(gate._PATTERN_DIM, dtype=np.float32),
            outcome_valence=0.1,
            timestamp=1.0,
            source=f"source-{idx}",
        )
        for idx in range(3)
    )
    gate._get_executive_state = _FailingCallable("executive mesh unavailable")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        _FailingCallable("state repository unavailable"),
    )

    assert gate._gut_feeling("act", "unit") == (0.0, 0.1)
    assert gate._foreground_commitment_active() is False
    assert recorded == [
        ("somatic_marker_gate", "RuntimeError"),
        ("somatic_marker_gate", "RuntimeError"),
    ]


def test_counterfactual_llm_simulation_failure_records_fallback(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        counterfactual_engine,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    router = types.SimpleNamespace(
        think=_FailingCallable("router unavailable"),
    )
    engine = counterfactual_engine.CounterfactualEngine()

    result = asyncio.run(
        engine._llm_simulate(
            action_type="search",
            description="look for evidence",
            context={"valence": 0.1, "curiosity": 0.8},
            router=router,
        )
    )

    assert result == "Expected outcome of search"
    assert recorded == [("counterfactual_engine.llm_simulate", "RuntimeError")]


def test_homeostatic_entropy_floor_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        homeostatic_coupling,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        homeostatic_coupling.ServiceContainer,
        "get",
        _FailingCallable("free energy unavailable"),
    )

    async def read_drives():
        return {"energy": 0.8, "curiosity": 0.7, "persistence": 0.7}

    async def read_affect():
        return {"valence": 0.1, "engagement": 0.5, "arousal": 0.3}

    coupling = homeostatic_coupling.HomeostaticCoupling.__new__(
        homeostatic_coupling.HomeostaticCoupling
    )
    coupling._lock = None
    coupling.substrate = None
    coupling._modifiers = homeostatic_coupling.CognitiveModifiers()
    coupling._last_update = 0.0
    coupling._prospective_dread = 0.0
    coupling._cpu_stress = 0.0
    coupling._mem_stress = 0.0
    coupling._stress_timestamp = 0.0
    coupling._read_drives = read_drives
    coupling._read_affect = read_affect
    coupling._pulse_root = lambda *_args, **_kwargs: None

    mods = asyncio.run(coupling.update())

    assert isinstance(mods, homeostatic_coupling.CognitiveModifiers)
    assert recorded == [("homeostatic_coupling", "RuntimeError")]


def test_homeostatic_constructor_preserves_substrate_default_on_lookup_failure(
    monkeypatch,
):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        homeostatic_coupling,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        homeostatic_coupling.ServiceContainer,
        "get",
        _FailingCallable("substrate unavailable"),
    )

    coupling = homeostatic_coupling.HomeostaticCoupling(types.SimpleNamespace())

    assert coupling.substrate is None
    assert recorded == [("homeostatic_coupling", "RuntimeError")]


def test_liquid_substrate_bridge_affect_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        liquid_substrate_bridge,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    substrate = types.SimpleNamespace(
        running=True,
        config=types.SimpleNamespace(neuron_count=4),
        get_substrate_affect=_FailingCallable("substrate affect unavailable"),
        inject_stimulus=lambda *_args, **_kwargs: None,
    )
    orchestrator = types.SimpleNamespace(
        enqueue_message=lambda _message: None,
        _update_liquid_pacing=lambda: None,
        _gather_agentic_context=lambda _message: {},
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )

    liquid_substrate_bridge.bridge_to_orchestrator(orchestrator)

    assert orchestrator.get_substrate_affect() == {
        "valence": 0.0,
        "arousal": 0.3,
        "dominance": 0.0,
        "energy": 0.5,
        "volatility": 0.0,
    }
    assert recorded == [("liquid_substrate_bridge", "RuntimeError")]


def test_substrate_evolution_stability_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        substrate_evolution,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    engine = substrate_evolution.SubstrateEvolution()
    engine._mesh_ref = types.SimpleNamespace(
        columns=[types.SimpleNamespace(x=np.array([1.0], dtype=np.float32))],
        get_global_synchrony=lambda: 0.5,
    )
    genome = substrate_evolution.Genome(
        id=1,
        inter_weights=np.eye(2, dtype=np.float32),
    )

    assert asyncio.run(engine._evaluate_fitness(genome)) == 0.0
    assert recorded == [("substrate_evolution", "ValueError")]


def test_multiple_drafts_mesh_lookup_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        multiple_drafts,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        _FailingCallable("mesh unavailable"),
    )

    engine = multiple_drafts.MultipleDraftsEngine()

    assert engine._get_mesh() is None
    assert recorded == [("multiple_drafts", "RuntimeError")]


def test_multiple_drafts_partial_mesh_failures_keep_finite_drafts(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        multiple_drafts,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    class _PartialMesh:
        def get_column_summary(self, column):
            if column % 3 == 0:
                raise RuntimeError("column unavailable")
            if column % 3 == 1:
                return {"energy": float("nan"), "mean_activation": 0.2}
            return {"energy": 0.8, "mean_activation": 0.25}

    state = types.SimpleNamespace(
        affect=types.SimpleNamespace(
            valence="not-a-number",
            arousal=float("nan"),
            curiosity=2.0,
        )
    )
    engine = multiple_drafts.MultipleDraftsEngine()
    engine._mesh_ref = _PartialMesh()

    drafts = engine.submit_input("How should I interpret this?", state)

    assert len(drafts) == 3
    assert all(np.isfinite(d.valence) for d in drafts)
    assert all(np.isfinite(d.urgency) for d in drafts)
    assert all(0.0 <= d.urgency <= 1.0 for d in drafts)
    assert any(
        kwargs.get("action") == "used neutral affect value for draft generation"
        for _args, kwargs in recorded
    )
    assert any(
        kwargs.get("action") == "ignored unavailable neural mesh column during draft generation"
        for _args, kwargs in recorded
    )


def test_neologism_push_state_normalizes_vectors(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(neologism_engine, "_LEXICON_PATH", tmp_path / "lexicon.json")
    monkeypatch.setattr(
        neologism_engine,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    engine = neologism_engine.NeologismEngine()
    engine.push_state(
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([np.inf, -np.inf], dtype=np.float32),
    )

    assert len(engine._state_buffer) == 1
    assert engine._state_buffer[0].shape == (48,)
    assert np.isfinite(engine._state_buffer[0]).all()
    assert any(
        kwargs.get("action") == "replaced non-finite neologism state values"
        for _args, kwargs in recorded
    )


def test_neologism_generation_falls_back_without_brain(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(neologism_engine, "_LEXICON_PATH", tmp_path / "lexicon.json")
    monkeypatch.setattr(
        neologism_engine,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda _name, default=None: default,
    )

    engine = neologism_engine.NeologismEngine()
    centroid = np.linspace(-0.5, 0.5, 48, dtype=np.float32)

    data = asyncio.run(engine._generate_neologism(centroid, count=4))

    assert data is not None
    assert data["word"].startswith("velm")
    assert data["source"] == "deterministic_fallback"
    assert data["occurrence_count"] == 4
    assert any(
        kwargs.get("action") == "used deterministic neologism fallback without brain service"
        for _args, kwargs in recorded
    )


def test_mhaf_phi_logdet_failure_is_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        phi_estimator,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    monkeypatch.setattr(
        phi_estimator.np.linalg,
        "slogdet",
        _FailingCallable("logdet unavailable"),
    )

    assert phi_estimator._safe_logdet(np.eye(2, dtype=np.float64)) == float("-inf")
    assert recorded == [("mhaf_phi_estimator", "RuntimeError")]


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


def test_neural_mesh_run_loop_survives_tick_failure_with_structured_action(monkeypatch):
    recorded: list[tuple[str, str, str, dict | None]] = []

    def record(module, exc, **kwargs):
        recorded.append(
            (
                module,
                type(exc).__name__,
                kwargs.get("action", ""),
                kwargs.get("extra"),
            )
        )

    monkeypatch.setattr(neural_mesh, "record_degradation", record)

    cfg = neural_mesh.MeshConfig(
        total_neurons=64,
        columns=4,
        neurons_per_column=16,
        sensory_end=1,
        association_end=3,
        update_hz=1000.0,
        projection_dim=8,
    )
    mesh = neural_mesh.NeuralMesh(cfg)
    attempts = {"count": 0}

    def fail_once_then_stop():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("tick exploded")
        mesh._running = False

    mesh._tick = fail_once_then_stop
    mesh._running = True

    asyncio.run(mesh._run_loop())

    assert attempts["count"] == 2
    assert mesh._running is False
    assert any(
        action == "kept NeuralMesh loop alive after tick failure and damped field"
        and extra == {"consecutive_tick_failures": 1}
        for _, _, action, extra in recorded
    )


def test_neural_mesh_injection_and_modulators_sanitize_nonfinite(monkeypatch):
    recorded: list[tuple[str, str, str]] = []

    def record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs.get("action", "")))

    monkeypatch.setattr(neural_mesh, "record_degradation", record)

    cfg = neural_mesh.MeshConfig(
        total_neurons=64,
        columns=4,
        neurons_per_column=16,
        sensory_end=1,
        association_end=3,
        projection_dim=8,
    )
    mesh = neural_mesh.NeuralMesh(cfg)

    mesh.inject_sensory([np.nan, np.inf, -np.inf, 5.0])
    mesh.inject_association(np.full(40, np.nan, dtype=np.float32))
    mesh.set_modulatory_state(gain=np.nan, plasticity=np.inf, noise=-3.0)
    mesh._tick()

    assert mesh._sensory_buffer is None
    assert mesh._association_buffer is None
    assert mesh._modulatory_gain == 1.0
    assert mesh._modulatory_plasticity == 1.0
    assert mesh._modulatory_noise == 0.0
    assert np.all(np.isfinite(mesh.get_field_state()))
    assert np.all(np.isfinite(mesh.get_executive_projection()))
    actions = {action for _, _, action in recorded}
    assert "sanitized sensory ingress before NeuralMesh injection" in actions
    assert "sanitized association ingress before NeuralMesh injection" in actions
    assert "normalized NeuralMesh modulatory state before applying it" in actions


def test_neural_mesh_invalid_column_summary_is_observable(monkeypatch):
    recorded: list[tuple[str, str, str]] = []

    def record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs.get("action", "")))

    monkeypatch.setattr(neural_mesh, "record_degradation", record)

    cfg = neural_mesh.MeshConfig(
        total_neurons=64,
        columns=4,
        neurons_per_column=16,
        sensory_end=1,
        association_end=3,
        projection_dim=8,
    )
    mesh = neural_mesh.NeuralMesh(cfg)

    summary = mesh.get_column_summary(99)

    assert summary["tier"] == "UNKNOWN"
    assert summary["energy"] == 0.0
    assert recorded == [
        (
            "neural_mesh",
            "IndexError",
            "returned empty NeuralMesh column summary for invalid index",
        )
    ]


def test_oscillatory_binding_run_loop_survives_tick_failure(monkeypatch):
    recorded: list[tuple[str, str, str, dict | None]] = []

    def record(module, exc, **kwargs):
        recorded.append(
            (
                module,
                type(exc).__name__,
                kwargs.get("action", ""),
                kwargs.get("extra"),
            )
        )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(oscillatory_binding, "record_degradation", record)
    monkeypatch.setattr(oscillatory_binding.asyncio, "sleep", no_sleep)

    cfg = oscillatory_binding.BindingConfig(internal_rate=200.0, output_rate=200.0)
    binding = oscillatory_binding.OscillatoryBinding(cfg)
    attempts = {"count": 0}

    def fail_once_then_stop():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("phase step unavailable")
        binding._running = False

    binding._oscillator_step = fail_once_then_stop
    binding._running = True

    asyncio.run(binding._run_loop())

    assert attempts["count"] == 2
    assert binding._running is False
    assert any(
        action
        == (
            "kept OscillatoryBinding loop alive after tick failure "
            "and reset to neutral phase state"
        )
        and extra == {"consecutive_tick_failures": 1}
        for _, _, action, extra in recorded
    )


def test_oscillatory_binding_sanitizes_bad_phase_and_mesh_inputs(monkeypatch):
    recorded: list[tuple[str, str, str]] = []

    def record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs.get("action", "")))

    monkeypatch.setattr(oscillatory_binding, "record_degradation", record)

    binding = oscillatory_binding.OscillatoryBinding()
    binding.receive_mesh_energy(
        {"sensory": np.nan, "association": 4.0, "executive": -1.0}
    )
    binding.report_phase("", 1.0)
    binding.report_phase("invalid", np.nan)
    binding.report_phase("valid_a", 0.0)
    binding.report_phase("valid_b", 2 * np.pi)
    binding._compute_synchronization()
    phase = binding.compute_subsystem_phase(np.inf, "computed")

    assert 0.0 <= binding._mesh_sensory_energy <= 1.0
    assert 0.0 <= binding._mesh_association_energy <= 1.0
    assert 0.0 <= binding._mesh_executive_energy <= 1.0
    assert "invalid" not in binding._phase_reports
    assert 0.0 <= phase < 2 * np.pi
    assert 0.0 <= binding.get_psi() <= 1.0
    assert np.isfinite(binding.get_gamma_amplitude())
    actions = {action for _, _, action in recorded}
    assert "normalized mesh energy before oscillator modulation" in actions
    assert "ignored phase report with invalid source" in actions
    assert "ignored invalid phase report" in actions
    assert "normalized subsystem activation before phase computation" in actions


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


def test_parallel_branch_failure_is_isolated_with_structured_receipt(monkeypatch):
    recorded: list[tuple[str, str, str, dict | None]] = []

    def record(module, exc, **kwargs):
        recorded.append(
            (
                module,
                type(exc).__name__,
                kwargs.get("action", ""),
                kwargs.get("extra"),
            )
        )

    async def failing_work(branch):
        branch.context.progress = 0.25
        raise RuntimeError("branch cognition failed")

    monkeypatch.setattr(parallel_branches, "record_degradation", record)

    manager = BranchManager()
    manager._publish_event = lambda *_args, **_kwargs: None
    branch = parallel_branches.CognitiveBranch(
        branch_id="br_test",
        name="test_branch",
        origin=parallel_branches.BranchOrigin.SYSTEM,
    )

    asyncio.run(manager._run_branch(branch, failing_work))

    assert branch.state == parallel_branches.BranchState.FAILED
    assert manager.get_status()["total_failed"] == 1
    assert recorded == [
        (
            "parallel_branches",
            "RuntimeError",
            "isolated failed branch without poisoning branch manager",
            {"branch_id": "br_test", "name": "test_branch"},
        )
    ]


def test_parallel_branch_spawn_fails_closed_without_will(monkeypatch):
    recorded: list[tuple[str, str, str]] = []

    def record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs.get("action", "")))

    async def work(_branch):
        return "should not run"

    monkeypatch.setattr(parallel_branches, "record_degradation", record)
    monkeypatch.setattr("core.will.get_will", _FailingCallable("will offline"))

    manager = BranchManager()
    branch = asyncio.run(
        manager.spawn(
            name="unauthorized_branch",
            origin=parallel_branches.BranchOrigin.SYSTEM,
            task_description="try to run without will",
            work_fn=work,
        )
    )

    assert branch is None
    assert recorded == [
        (
            "parallel_branches",
            "RuntimeError",
            "rejected branch spawn because Unified Will was unavailable",
        )
    ]


def test_parallel_branch_yield_enforces_budget(monkeypatch):
    sleeps: list[float] = []

    async def tracked_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(parallel_branches.asyncio, "sleep", tracked_sleep)

    branch = parallel_branches.CognitiveBranch(
        branch_id="br_budget",
        name="budget_branch",
        origin=parallel_branches.BranchOrigin.SYSTEM,
        cpu_budget_ms=1.0,
    )
    branch.context.metadata["_last_yield_monotonic"] = (
        parallel_branches.time.monotonic() - 1.0
    )

    asyncio.run(BranchManager.branch_yield(branch))

    assert 0.001 in sleeps
    assert 0 in sleeps
    assert branch.cpu_used_ms == 0.0


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


def test_experience_consolidator_sanitizes_llm_identity_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experience_consolidator,
        "NARRATIVE_PATH",
        tmp_path / "self_narrative.json",
    )
    consolidator = ExperienceConsolidator(cognitive_engine=None)
    consolidator._narrative = experience_consolidator.IdentityNarrative(version=2)

    narrative = consolidator._narrative_from_mapping(
        {
            "signature_phrase": "steady " * 80,
            "stable_traits": ["curious", "curious", "", 42, "reflective"],
            "learned_preferences": "quiet rooms",
            "growth_edges": ["patience", None, "precision", "overflow"],
        }
    )

    assert narrative.version == 3
    assert len(narrative.signature_phrase) <= 280
    assert narrative.stable_traits == ["curious", "42", "reflective"]
    assert narrative.learned_preferences == ["quiet rooms"]
    assert narrative.growth_edges == ["patience", "precision", "overflow"]


def test_experience_consolidator_home_vector_rejects_bad_hidden_states(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experience_consolidator,
        "NARRATIVE_PATH",
        tmp_path / "self_narrative.json",
    )
    crsm = types.SimpleNamespace(
        _history=deque(
            [
                {"hidden": [1.0, 2.0, 3.0], "prediction_error": 0.3},
                {"hidden": [float("nan"), 2.0, 3.0], "prediction_error": 0.4},
                {"hidden": [2.0, 3.0, 4.0], "prediction_error": 0.5},
                {"hidden": [9.0, 9.0], "prediction_error": 0.6},
                {"hidden": [3.0, 4.0, 5.0], "prediction_error": 0.7},
            ]
        ),
        home_vector=np.ones(2),
    )
    crsm_module = types.ModuleType("core.consciousness.crsm")
    crsm_module.get_crsm = lambda: crsm
    monkeypatch.setitem(sys.modules, "core.consciousness.crsm", crsm_module)

    narrative = experience_consolidator.IdentityNarrative()
    ExperienceConsolidator(cognitive_engine=None)._update_crsm_home_vector(narrative)

    assert np.allclose(crsm.home_vector, np.array([0.2, 0.3, 0.4]))
    assert narrative.home_vector_delta == [2.0, 3.0, 4.0]


def test_experience_consolidator_failure_backoff_records_recovery_action(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        experience_consolidator,
        "NARRATIVE_PATH",
        tmp_path / "self_narrative.json",
    )
    monkeypatch.setattr(
        experience_consolidator,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    consolidator = ExperienceConsolidator(cognitive_engine=None)
    backoff = consolidator._handle_consolidation_failure(RuntimeError("llm offline"))

    assert backoff == experience_consolidator._BACKOFF_BASE_SECS
    assert consolidator._consecutive_failures == 1
    assert consolidator._next_allowed_run > 0.0
    assert recorded
    assert "scheduled exponential backoff" in recorded[0][1]["action"]
    assert recorded[0][1]["receipt_required"] is True


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


def test_unified_field_run_loop_survives_tick_failure(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def _record(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(unified_field, "record_degradation", _record)

    field = UnifiedField(
        FieldConfig(
            dim=8,
            mesh_input_dim=2,
            chem_input_dim=2,
            binding_input_dim=2,
            intero_input_dim=2,
            substrate_input_dim=2,
            update_hz=1000.0,
            plasticity_interval=100,
        )
    )
    calls = {"count": 0}

    def _flaky_tick():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("field tick failed once")
        field._running = False

    monkeypatch.setattr(field, "_tick", _flaky_tick)
    field._running = True

    asyncio.run(field._run_loop())

    assert calls["count"] == 2
    assert field._running is False
    assert recorded
    assert recorded[0][0] == "unified_field"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "kept UnifiedField loop alive" in str(recorded[0][2]["action"])


def test_unified_field_sanitizes_non_finite_inputs_and_state(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        unified_field,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

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
    field.receive_mesh(np.array([np.nan, np.inf, 5.0], dtype=np.float32))
    field.receive_chemicals(object())
    field.F[0] = np.nan
    field._tick()

    state = field.get_field_state()

    assert np.all(np.isfinite(state))
    assert np.all(np.abs(state) <= 1.0)
    assert any(item == ("unified_field", "ValueError") for item in recorded)


def test_unified_field_surprise_uses_binding_input():
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
    field.W_bind = np.zeros_like(field.W_bind)
    field.receive_binding(np.ones(2, dtype=np.float32))

    surprise = field.compute_world_model_surprise()

    assert surprise > 0.5


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
        lambda module, exc, **_kwargs: recorded.append((module, type(exc).__name__)),
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


def test_loop_monitor_service_lookup_failure_records_structured_action(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        loop_monitor,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    service_container = types.SimpleNamespace(
        get=_FailingCallable("container lookup failed")
    )

    result = ConsciousnessLoopMonitor()._get(service_container, "affect_engine")

    assert result is None
    assert recorded
    assert recorded[0][1]["action"] == "treated missing service container lookup as absent service"
    assert recorded[0][1]["receipt_required"] is True


def test_loop_monitor_end_to_end_probe_awaits_async_bridge():
    class _Synth:
        _tick = 1

    class _Affect:
        def __init__(self):
            self.calls = 0

        async def receive_qualia_echo(self, *, q_norm, pri, trend):
            self.calls += 1
            assert q_norm == 0.001
            assert pri == 0.001
            assert trend == 0.0

    affect = _Affect()

    class _Container:
        @staticmethod
        def get(name, default=None):
            if name == "qualia_synthesizer":
                return _Synth()
            if name == "affect_engine":
                return affect
            return default

    monitor = ConsciousnessLoopMonitor()
    monitor._consecutive_healthy = 4
    monitor._get_service_container = lambda: _Container

    issues = asyncio.run(monitor._run_checks())

    assert issues == []
    assert affect.calls == 1


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


def test_heartbeat_run_survives_tick_failure_with_structured_action(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        heartbeat,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    async def _scenario():
        hb = CognitiveHeartbeat.__new__(CognitiveHeartbeat)
        hb.tick_count = 0
        hb._TICK_RATE_HZ = 1000.0
        hb._stop_event = asyncio.Event()
        hb._consecutive_tick_failures = 0

        calls = 0

        async def tick():
            nonlocal calls
            calls += 1
            hb.tick_count += 1
            if calls == 1:
                raise RuntimeError("workspace unavailable")
            hb.stop()

        hb._tick = tick
        hb._evaluate_tick_interval = lambda: 0.0

        async def no_sleep(_delay):
            return None

        monkeypatch.setattr(heartbeat.asyncio, "sleep", no_sleep)
        await hb.run()
        return calls, hb._consecutive_tick_failures

    calls, consecutive = asyncio.run(_scenario())

    assert calls == 2
    assert consecutive == 0
    assert recorded
    assert "kept heartbeat loop alive" in recorded[0][1]["action"]
    assert recorded[0][1]["receipt_required"] is True


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
        lambda module, exc, **_kwargs: recorded.append((module, type(exc).__name__)),
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
    assert cs.layer_status["liquid_substrate"] == "degraded"
    assert "liquid_substrate" in cs._degraded_layers
    assert recorded == [("system", "RuntimeError")]


def test_consciousness_system_stop_records_shutdown_failure_and_resets_running(
    monkeypatch,
):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        system,
        "record_degradation",
        lambda module, exc, **_kwargs: recorded.append((module, type(exc).__name__)),
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
