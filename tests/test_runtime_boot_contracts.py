from __future__ import annotations

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


def test_provider_constructors_accept_boot_time_defaults():
    from core.curiosity_engine import CuriosityEngine
    from core.ops.singularity_monitor import SingularityMonitor

    curiosity = CuriosityEngine()
    monitor = SingularityMonitor()

    assert curiosity.proactive_comm.get_boredom_level() == 0.0
    assert monitor.get_status()["status"] == "STABLE"


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


def test_scaffolds_and_null_telemetry_are_operational():
    from core.pipeline.prompt_scaffold import PromptScaffold
    from core.runtime.telemetry_exporter import NullExporter, MetricSample

    prompt = PromptScaffold().build_structured_prompt("solve it", context="ctx")
    exporter = NullExporter()
    exporter.emit_metric(MetricSample(name="boot.contract", value=1.0))
    exporter.flush()

    assert "solve it" in prompt
    assert exporter.metrics[0].name == "boot.contract"


@pytest.mark.asyncio
async def test_av_production_local_renderer_creates_artifact(tmp_path):
    from core.sensory_integration import AVProductionSystem

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


@pytest.mark.asyncio
async def test_heartbeat_telemetry_clamps_negative_runtime_metrics(monkeypatch):
    from core.consciousness.heartbeat import CognitiveHeartbeat
    import core.consciousness.heartbeat as heartbeat_module

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
