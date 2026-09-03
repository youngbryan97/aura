from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_ops_health_monitor_import_and_error_rate_contract():
    from core.ops.health_monitor import HealthMonitor

    monitor = HealthMonitor(max_consecutive_errors=3)
    assert monitor.is_healthy()
    monitor.track_error(RuntimeError("one"))
    assert monitor.error_rate >= 0.0
    monitor.record_success()
    assert monitor.is_healthy()


def test_precision_engine_fhn_step_mutates_local_state_without_gateway():
    from core.pneuma.precision_engine import FHNOscillator

    oscillator = FHNOscillator(dt=0.05)
    before = (oscillator.state.v, oscillator.state.w, oscillator.state.t)
    after = oscillator.step(i_ext=0.7)

    assert after is oscillator.state
    assert (after.v, after.w, after.t) != before
    assert -10.0 < after.v < 10.0
    assert -10.0 < after.w < 10.0


def test_resilience_engine_exposes_soma_snapshot_contract():
    from core.soma.resilience_engine import ResilienceEngine

    engine = ResilienceEngine()
    engine.record_failure("tool_execution", severity=0.8, stakes=0.9)

    snapshot = engine.get_body_snapshot()
    status = engine.get_status()

    assert snapshot["soma"]["thermal_load"] >= 0.0
    assert snapshot["soma"]["resource_anxiety"] >= 0.0
    assert snapshot["affects"]["stress"] >= 0.0
    assert snapshot["affects"]["fatigue"] >= 0.0
    assert status["soma"]["thermal_load"] == pytest.approx(snapshot["soma"]["thermal_load"])
    assert status["soma"]["resource_anxiety"] == pytest.approx(snapshot["soma"]["resource_anxiety"])
    assert status["affects"]["stress"] == pytest.approx(snapshot["affects"]["stress"])
    assert status["affects"]["fatigue"] == pytest.approx(snapshot["affects"]["fatigue"])


def test_motivation_engine_drive_vector_contract():
    from core.motivation.engine import MotivationEngine

    engine = MotivationEngine()
    engine.budgets["energy"].level = 42.0

    vector = engine.get_drive_vector()

    assert "energy" in vector
    assert "curiosity" in vector
    assert all(0.0 <= value <= 1.0 for value in vector.values())
    assert engine.get_dominant_motivation() in vector or engine.get_dominant_motivation() == "at_rest"


@pytest.mark.asyncio
async def test_motivation_engine_liveness_requires_running_loop(monkeypatch):
    from core.container import ServiceContainer
    from core.motivation import engine as motivation_module

    ServiceContainer.clear()
    ServiceContainer.register_instance("orchestrator", SimpleNamespace(), required=False)
    monkeypatch.setattr(
        motivation_module,
        "_background_autonomy_block_reason",
        lambda _orchestrator: "unit_test_hold",
    )
    engine = motivation_module.MotivationEngine()

    assert engine.is_alive() is False

    await engine.start()
    try:
        await asyncio.sleep(0)
        assert engine.is_alive() is True
    finally:
        await engine.stop()
        await asyncio.sleep(0)

    assert engine.is_alive() is False


def test_provider_constructors_accept_boot_time_defaults():
    from core.curiosity_engine import CuriosityEngine
    from core.ops.singularity_monitor import SingularityMonitor

    curiosity = CuriosityEngine()
    monitor = SingularityMonitor()

    assert curiosity.proactive_comm.get_boredom_level() == 0.0
    assert monitor.get_status()["status"] == "STABLE"


def test_live_boot_paths_do_not_use_generic_exception_boundaries():
    project_root = Path(__file__).resolve().parent.parent
    aura_main = (project_root / "aura_main.py").read_text(encoding="utf-8")
    boot_identity = (
        project_root / "core" / "orchestrator" / "mixins" / "boot" / "boot_identity.py"
    ).read_text(encoding="utf-8")

    boundary_tuple = aura_main.split("_AURA_MAIN_BOUNDARY_ERRORS = (", 1)[1].split(")", 1)[0]

    assert "Exception," not in boundary_tuple
    assert "except Exception" not in aura_main
    assert "except BaseException" not in aura_main
    assert "except Exception" not in boot_identity
    assert "except BaseException" not in boot_identity


def test_memory_provider_registers_usable_knowledge_graph_and_dreamer(tmp_path, monkeypatch):
    from core.config import Paths
    from core.container import ServiceContainer
    from core.providers.memory_provider import register_memory_services

    ServiceContainer.clear()
    monkeypatch.setattr(Paths, "_runtime_home_cache", tmp_path)
    ServiceContainer.register_instance("cognitive_engine", SimpleNamespace(think=lambda *_args, **_kwargs: None))

    try:
        register_memory_services(ServiceContainer)

        kg = ServiceContainer.require("knowledge_graph")
        dreamer = ServiceContainer.require("dreamer_v2")
        ledger = ServiceContainer.require("knowledge_ledger")
        vault = ServiceContainer.require("blackhole_vault")
        cold = ServiceContainer.require("cold_store")

        assert kg.db_path.endswith("knowledge_graph/knowledge.db")
        assert dreamer.kg is kg
        assert callable(ledger.get_ledger)
        assert vault is ServiceContainer.require("memory_vector")
        assert cold.is_ready() is True
        assert cold.db_path.name == "cold_store.db"
    finally:
        ServiceContainer.clear()


def test_memory_provider_migrates_legacy_knowledge_graph_file(tmp_path, monkeypatch):
    import contextlib
    import sqlite3

    from core.config import Paths
    from core.container import ServiceContainer
    from core.providers.memory_provider import register_memory_services

    # contextlib.closing, not a bare `with sqlite3.connect(...)`: the
    # connection's own context manager commits a transaction and leaves the
    # handle open. That left knowledge.db open past teardown and the hermetic
    # sandbox reported it as a leak — the test failed for its own plumbing
    # rather than for the migration it checks.
    legacy_path = tmp_path / "data" / "knowledge_graph"
    legacy_path.parent.mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(legacy_path)) as conn:
        conn.execute("CREATE TABLE migration_marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO migration_marker(value) VALUES ('preserved')")
        conn.commit()

    ServiceContainer.clear()
    monkeypatch.setattr(Paths, "_runtime_home_cache", tmp_path)
    ServiceContainer.register_instance("cognitive_engine", SimpleNamespace(think=lambda *_args, **_kwargs: None))

    try:
        register_memory_services(ServiceContainer)

        kg = ServiceContainer.require("knowledge_graph")

        canonical_db = legacy_path / "knowledge.db"
        assert kg.db_path == str(canonical_db)
        assert legacy_path.is_dir()
        with contextlib.closing(sqlite3.connect(canonical_db)) as conn:
            assert conn.execute("SELECT value FROM migration_marker").fetchone() == (
                "preserved",
            )
        assert not list(legacy_path.parent.glob(".knowledge_graph.migration-*"))
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_foundation_cognition_validation_samples_new_diagnostics_first():
    import core.runtime.foundations as foundations
    from core.organism.model_validation import reset_validation_for_test

    reset_validation_for_test()
    middleware = await foundations._activate_middleware(foreground_only=True)
    cognition = await foundations._activate_cognition(foreground_only=True)

    assert middleware.ok is True
    assert cognition.ok is True
    assert cognition.data["suite_outcome"]["failed"] == 0
    assert cognition.data["problem_tests"] == []


@pytest.mark.asyncio
async def test_desktop_cognition_validation_runs_off_loop():
    import threading

    import core.runtime.foundations as foundations

    foundations.reset_foundations_for_test()
    entered = threading.Event()
    release = threading.Event()

    def _validation():
        entered.set()
        release.wait()
        return {
            "passed": 7,
            "failed": 0,
            "errored": 0,
            "not_measured": 1,
            "applicable": 8,
            "measured": 7,
        }

    try:
        assert foundations._schedule_cognition_validation(_validation) is True
        assert foundations._schedule_cognition_validation(_validation) is False
        assert await asyncio.to_thread(entered.wait, 1.0) is True
        assert foundations.cognition_validation_status()["state"] == "running"
        release.set()

        for _ in range(100):
            status = foundations.cognition_validation_status()
            if status["state"] == "completed":
                break
            await asyncio.sleep(0.01)
        assert status["outcome"] == {
            "passed": 7,
            "failed": 0,
            "errored": 0,
            "not_measured": 1,
            "applicable": 8,
            "measured": 7,
        }
    finally:
        release.set()
        foundations.reset_foundations_for_test()


@pytest.mark.asyncio
async def test_desktop_cognition_activation_registers_without_running_suite(monkeypatch):
    import core.knowledge.metta as metta
    import core.organism.model_validation as model_validation
    import core.runtime.foundations as foundations
    from core.container import ServiceContainer

    calls = []
    monkeypatch.setattr(metta, "install_runtime_rules", lambda: ["rule"])
    monkeypatch.setattr(metta, "get_metta", lambda: object())
    monkeypatch.setattr(metta, "metta_report", lambda: {"grounded_ops": [1, 2, 3]})
    monkeypatch.setattr(
        model_validation,
        "install_runtime_validation",
        lambda: {"claims": 2, "tests": ["a", "b"]},
    )
    monkeypatch.setattr(
        model_validation,
        "get_suite",
        lambda: SimpleNamespace(unsupported_claims=lambda: []),
    )
    monkeypatch.setattr(model_validation, "run_validation", lambda: calls.append("run"))
    monkeypatch.setattr(ServiceContainer, "register_instance", lambda *_args, **_kwargs: None)

    result = await foundations._activate_cognition(foreground_only=False)

    assert result.ok is True
    assert result.data["suite_outcome"] == {"state": "pending_start"}
    assert calls == []
    assert foundations.cognition_validation_status()["state"] == "pending_start"


@pytest.mark.asyncio
async def test_desktop_empirical_validation_starts_after_last_foundation(monkeypatch):
    import core.runtime.foundations as foundations

    order = []

    async def _activation(*, foreground_only):
        assert foreground_only is False
        order.append("activation")
        return foundations.ActivationResult(name="only", ok=True)

    def _schedule(_run_validation):
        order.append("validation")
        return True

    foundations.reset_foundations_for_test()
    monkeypatch.setattr(foundations, "_ACTIVATORS", [("only", _activation)])
    monkeypatch.setattr(foundations, "_schedule_cognition_validation", _schedule)

    report = await foundations.activate_foundations(foreground_only=False)

    assert report["ok"] is True
    assert foundations.foundations_report()["activated"] is True
    assert order == ["activation", "validation"]


@pytest.mark.asyncio
async def test_boot_identity_reuses_existing_fictional_engines(monkeypatch):
    import core.agency.latent_distiller as latent_distiller_module
    import core.fictional_ai_synthesis as fictional_synthesis_module
    import core.memory.snap_kv_evictor as snap_evictor_module
    import core.self_modification.shadow_ast_healer as shadow_healer_module
    from core.orchestrator.mixins.boot.boot_identity import BootIdentityMixin

    engines = {"jarvis": object()}
    harness = SimpleNamespace(fictional_engines=engines)
    duplicate_registration_calls = 0
    component_constructions = 0

    def record_duplicate_registration(*_args, **_kwargs):
        nonlocal duplicate_registration_calls
        duplicate_registration_calls += 1
        return {}

    def component():
        nonlocal component_constructions
        component_constructions += 1
        return object()

    monkeypatch.setattr(
        fictional_synthesis_module,
        "register_all_fictional_engines",
        record_duplicate_registration,
    )
    monkeypatch.setattr(shadow_healer_module, "ShadowASTHealer", lambda *_args, **_kwargs: component())
    monkeypatch.setattr(snap_evictor_module, "SnapKVEvictor", lambda *_args, **_kwargs: component())
    monkeypatch.setattr(
        latent_distiller_module,
        "LatentSpaceDistiller",
        lambda *_args, **_kwargs: component(),
    )

    await BootIdentityMixin._init_fictional_synthesis(harness)
    await BootIdentityMixin._init_fictional_synthesis(harness)

    assert harness.fictional_engines is engines
    assert duplicate_registration_calls == 0
    assert component_constructions == 3


def test_final_engines_create_persistence_dirs_without_generated_gateways(tmp_path):
    from core.final_engines import NarrativeIdentityEngine, WorldModelEngine

    world_path = tmp_path / "world" / "beliefs.json"
    identity_path = tmp_path / "identity" / "narrative.json"

    world = WorldModelEngine(world_path)
    identity = NarrativeIdentityEngine(identity_path)
    world.add_belief("Aura boot contracts are durable", 0.9, source_id="test")
    identity.append_chapter("Boot", "Runtime contracts stayed intact.")

    assert world_path.exists()
    assert identity_path.exists()


def test_world_model_confidence_is_finite_and_bounded(tmp_path):
    from core.final_engines import WorldModelEngine

    world = WorldModelEngine(tmp_path / "world" / "beliefs.json")
    world.add_belief("NaN is not evidence", float("nan"))
    world.add_belief("Overconfidence is bounded", 7.0)
    world.add_belief("Negative confidence is bounded", -3.0)

    assert world.beliefs["nan is not evidence"].confidence == 0.0
    assert world.beliefs["overconfidence is bounded"].confidence == 1.0
    assert world.beliefs["negative confidence is bounded"].confidence == 0.0


def test_scaffolds_and_null_telemetry_are_operational():
    from core.pipeline.prompt_scaffold import PromptScaffold
    from core.runtime.telemetry_exporter import MetricSample, NullExporter

    prompt = PromptScaffold().build_structured_prompt("solve it", context="ctx")
    exporter = NullExporter()
    exporter.emit_metric(MetricSample(name="boot.contract", value=1.0))
    exporter.flush()

    assert "solve it" in prompt
    assert exporter.metrics[0].name == "boot.contract"


@pytest.mark.asyncio
async def test_av_production_local_renderer_creates_artifact(tmp_path):
    from core.perception.sensory_integration import AVProductionSystem

    av = AVProductionSystem(output_dir=str(tmp_path))
    result = await av.create_image("a boot-safe local renderer", style="diagnostic")

    assert result["source"] in {"local_renderer", "manifest_fallback"}
    assert result["path"]


@pytest.mark.asyncio
async def test_performance_guard_start_uses_task_tracker():
    from core.runtime.performance_guard import PerformanceGuard

    guard = PerformanceGuard()
    await guard.start(interval=3600.0)
    assert guard._task is not None
    await guard.stop()


@pytest.mark.asyncio
async def test_performance_guard_persists_reports_off_event_loop():
    import core.runtime.performance_guard as performance_guard_module
    from core.runtime.performance_guard import PerformanceGuard

    guard = PerformanceGuard()
    release = asyncio.Event()
    to_thread_calls = 0

    async def _fake_to_thread(func, row):
        nonlocal to_thread_calls
        to_thread_calls += 1
        assert row["kind"] == "report"
        release.set()
        return func(row)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(performance_guard_module.asyncio, "to_thread", _fake_to_thread)
        monkeypatch.setattr(guard, "_persist", lambda _row: None)
        await guard.start(interval=3600.0)
        await asyncio.wait_for(release.wait(), timeout=1.0)
        await guard.stop()
    finally:
        monkeypatch.undo()

    assert to_thread_calls == 1


def test_consciousness_augmentor_exposes_status():
    from core.consciousness.integration import ConsciousnessAugmentor

    core = SimpleNamespace(get_status=lambda: {"integration_active": True})
    augmentor = ConsciousnessAugmentor(core)

    data = augmentor.get_augmentation("check launch")

    assert data["integration_active"] is True
    assert "check launch" in data["objective_hint"]


def test_soma_status_contract_exposes_homeostasis_shape(monkeypatch):
    from core.senses.soma import Soma

    soma = Soma()
    soma.state.cpu_percent = 35.0
    soma.state.ram_percent = 50.0
    soma.state.stress_level = 0.2

    status = soma.get_status()

    assert "soma" in status
    assert "affects" in status
    assert status["soma"]["thermal_load"] >= 0.0
    assert status["soma"]["resource_anxiety"] >= 0.0


def test_liquid_substrate_velocity_contract(tmp_path):
    from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

    substrate = LiquidSubstrate(SubstrateConfig(state_file=tmp_path / "substrate_state.npy"))
    substrate.v[0] = 0.5

    velocity = substrate.compute_cognitive_velocity()

    assert 0.0 <= velocity <= 1.0


def test_system_state_monitor_initializes_health_history():
    from core.ops.system_monitor import SystemStateMonitor

    monitor = SystemStateMonitor()

    assert monitor.health_history == []


@pytest.mark.asyncio
async def test_heartbeat_telemetry_clamps_negative_runtime_metrics(monkeypatch):
    import core.consciousness.heartbeat as heartbeat_module
    from core.consciousness.heartbeat import CognitiveHeartbeat

    published = {}

    class EventBus:
        def publish_threadsafe(self, topic, payload):
            published["topic"] = topic
            published["payload"] = payload

    async def get_narrative():
        return "steady"

    hb = object.__new__(CognitiveHeartbeat)
    hb.homeostasis = SimpleNamespace(
        get_modifiers=lambda: SimpleNamespace(overall_vitality=-1.0)
    )
    hb.temporal = SimpleNamespace(get_narrative=get_narrative)
    hb.orch = SimpleNamespace(
        liquid_state=SimpleNamespace(
            current=SimpleNamespace(
                energy=-1.0,
                curiosity=-0.25,
                frustration=-0.5,
                focus=-0.75,
            )
        )
    )
    hb.attention = SimpleNamespace(coherence=-0.5)
    hb._integrity_cache = None

    monkeypatch.setattr(heartbeat_module, "get_event_bus", lambda: EventBus())
    monkeypatch.setattr(
        heartbeat_module.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: default),
    )

    await hb._emit_telemetry(winner=None, state={}, tick=1, surprise=-2.0)

    assert published["topic"] == "telemetry"
    payload = published["payload"]
    assert payload["energy"] == 0.0
    assert payload["curiosity"] == 0.0
    assert payload["frustration"] == 0.0
    assert payload["confidence"] == 0.0
    assert payload["coherence"] == 0.0
    assert payload["vitality"] == 0.0
    assert payload["surprise"] == 0.0
