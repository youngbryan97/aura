"""tests/test_morphogenesis_runtime.py

Comprehensive test suite for the morphogenetic self-organisation layer.

Tests cover:
  1. Cell activation and repair signal emission
  2. Organ stabilisation from repeated co-activation
  3. Runtime status JSON safety
  4. Field diffusion across tissue edges
  5. Metabolism backpressure and energy budgeting
  6. Cell lifecycle transitions (quarantine, hibernate, apoptosis)
  7. Immunity bridge routing
  8. Hook influence (metabolic modulation, routing advice, initiative suppression)
  9. Registry persistence round-trip
  10. Runtime start/stop lifecycle
"""
from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from core.morphogenesis.cell import MorphogenCell
from core.morphogenesis.field import MorphogenField
from core.morphogenesis.metabolism import MetabolismManager, ResourceSnapshot
from core.morphogenesis.organs import OrganStabilizer
from core.morphogenesis.registry import MorphogenesisRegistry
from core.morphogenesis.runtime import MorphogeneticRuntime
from core.morphogenesis.types import (
    CellLifecycle,
    CellManifest,
    CellRole,
    MorphogenesisConfig,
    MorphogenSignal,
    SignalKind,
    clamp01,
    json_safe,
    stable_digest,
)

# ---------------------------------------------------------------------------
# 1. Cell activation and repair signal emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_morphogenesis_cell_activates_and_emits_repair():
    """A cell receiving a danger signal should activate and emit a repair signal."""
    manifest = CellManifest(
        name="test_repair",
        role=CellRole.REPAIR,
        subsystem="resilience",
        capabilities=["repair"],
        consumes=[SignalKind.ERROR.value, SignalKind.EXCEPTION.value],
        emits=[SignalKind.REPAIR.value],
        protected=True,
        criticality=0.9,
        baseline_energy=0.5,
        activation_threshold=0.15,
    )
    cell = MorphogenCell(manifest)
    field = MorphogenField()
    field.perturb("resilience", "danger", 0.7)

    error_signal = MorphogenSignal(
        kind=SignalKind.ERROR,
        source="test",
        subsystem="resilience",
        intensity=0.8,
        payload={"error": "test_error"},
        ttl_ticks=5,
    )
    result = await cell.tick(signals=[error_signal], field=field, global_energy=1.0)

    assert result.activated, "Cell should activate on danger signal"
    assert result.success
    assert any(s.kind == SignalKind.REPAIR for s in result.emitted_signals), (
        "Activated cell should emit a repair signal"
    )
    assert cell.state.activation_count == 1


# ---------------------------------------------------------------------------
# 2. Organ stabilisation from repeated co-activation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_morphogenesis_organ_stabilizes_from_coactivation():
    """Repeated co-activation of cells should discover an organ."""
    stabilizer = OrganStabilizer(
        min_coactivations=3,
        min_members=2,
        edge_threshold=0.5,
    )
    for _ in range(5):
        stabilizer.observe_activation(
            ["cell_alpha", "cell_beta"],
            success=True,
            task_signature="test_task",
            subsystem="resilience",
        )

    organs = stabilizer.discover()
    assert len(organs) >= 1, "Repeated co-activation should discover at least one organ"
    assert "cell_alpha" in organs[0].members
    assert "cell_beta" in organs[0].members
    assert organs[0].confidence > 0.0


# ---------------------------------------------------------------------------
# 3. Runtime status is JSON-safe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_morphogenesis_status_is_json_safe():
    """Runtime status must be fully JSON-serialisable."""
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False))
    status = rt.status()
    serialised = json.dumps(status, default=str)
    assert "tick" in serialised


@pytest.mark.asyncio
async def test_morphogenesis_stop_is_single_flight_and_persists_once():
    class _Registry:
        def __init__(self) -> None:
            self.save_calls = 0

        def save(self) -> None:
            self.save_calls += 1

    registry = _Registry()
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(enabled=False),
        registry=registry,  # type: ignore[arg-type]
    )

    await asyncio.gather(rt.stop(), rt.stop())

    assert registry.save_calls == 1
    assert rt._task is None
    assert rt.shutdown_timeout_s >= 5.0


# ---------------------------------------------------------------------------
# 4. Field diffusion across tissue edges
# ---------------------------------------------------------------------------

def test_field_diffusion_propagates_across_edges():
    """Danger in 'memory' should diffuse to 'cognition' via a tissue edge."""
    field = MorphogenField(diffusion=0.5, decay=0.0)  # No decay for test clarity
    field.register_edge("memory", "cognition", weight=1.0)

    field.perturb("memory", "danger", 0.8)
    assert field.sample("cognition")["danger"] == 0.0, "Before diffusion, cognition has no danger"

    field.diffuse_step()
    cognition_danger = field.sample("cognition")["danger"]
    assert cognition_danger > 0.0, f"After diffusion, cognition should have danger; got {cognition_danger}"
    assert cognition_danger <= 0.8, "Diffused value should not exceed source"


def test_field_decay_reduces_values():
    """Values should decay over time."""
    field = MorphogenField(diffusion=0.0, decay=0.5)
    field.perturb("test_subsystem", "danger", 1.0)

    field.diffuse_step()
    after_decay = field.sample("test_subsystem")["danger"]
    assert after_decay < 1.0, f"Danger should decay; got {after_decay}"
    assert after_decay > 0.0, "One step of 50% decay should not reach zero"


def test_field_signal_ingestion_maps_correctly():
    """Ingesting an ERROR signal should perturb danger, damage, and repair."""
    field = MorphogenField()
    sig = MorphogenSignal(
        kind=SignalKind.ERROR,
        source="test",
        subsystem="resilience",
        intensity=0.6,
    )
    field.ingest_signal(sig)
    sample = field.sample("resilience")
    assert sample["danger"] > 0.0
    assert sample["damage"] > 0.0
    assert sample["repair"] > 0.0


# ---------------------------------------------------------------------------
# 5. Metabolism backpressure and energy budgeting
# ---------------------------------------------------------------------------

def test_metabolism_budget_spending():
    """Spending energy should reduce the cell budget and global energy."""
    mgr = MetabolismManager(global_energy=1.0)
    mgr.ensure_budget("cell_a", priority=0.5, baseline=0.5)

    assert mgr.spend("cell_a", 0.2), "Should be able to spend within budget"
    budget = mgr._budgets["cell_a"]
    assert budget.energy < 0.5, "Budget energy should decrease after spending"
    assert mgr.global_energy < 1.0, "Global energy should decrease after spending"


def test_metabolism_denies_overspend():
    """Spending more than available should be denied."""
    mgr = MetabolismManager(global_energy=0.05)  # Very low global
    mgr.ensure_budget("cell_a", priority=0.5, baseline=0.5)

    # Try to spend more than global * 0.35 permits
    result = mgr.spend("cell_a", 0.5)
    assert not result, "Overspend should be denied when global energy is too low"


def test_metabolism_pulse_recovers_energy(monkeypatch):
    """Pulse should recover global energy and cell budgets."""
    mgr = MetabolismManager(global_energy=0.5, recovery_per_tick=0.1)
    mgr.ensure_budget("cell_a", priority=0.8, baseline=0.3)
    mgr._budgets["cell_a"].energy = 0.1  # Simulate depleted cell

    monkeypatch.setattr(
        MetabolismManager,
        "sample_resources",
        lambda self: ResourceSnapshot(pressure=0.0),
    )
    snap = mgr.pulse()

    assert mgr.global_energy > 0.5, "Global energy should recover"
    assert mgr._budgets["cell_a"].energy > 0.1, "Cell budget should recover"
    assert snap.pressure == 0.0


# ---------------------------------------------------------------------------
# 6. Cell lifecycle transitions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cell_quarantine_after_failures():
    """A non-protected cell should quarantine after 3 failures."""
    manifest = CellManifest(
        name="fragile",
        role=CellRole.SENSOR,
        subsystem="test",
        capabilities=["test"],
        consumes=[SignalKind.TASK.value],
        protected=False,
        criticality=0.3,
        activation_threshold=0.01,  # Very low so it always activates
    )

    handler_calls = []

    def _failing_handler(cell, signals, field_state):
        handler_calls.append((cell, signals, field_state))
        raise RuntimeError("test failure")

    cell = MorphogenCell(manifest, handler=_failing_handler)
    field = MorphogenField()
    field.perturb("test", "task_pressure", 0.9)

    task_signal = MorphogenSignal(
        kind=SignalKind.TASK, source="test", subsystem="test", intensity=0.9,
    )
    for _ in range(3):
        await cell.tick(signals=[task_signal], field=field, global_energy=1.0)

    assert len(handler_calls) == 3
    assert cell.lifecycle == CellLifecycle.QUARANTINED, (
        f"Expected QUARANTINED after 3 failures, got {cell.lifecycle}"
    )


@pytest.mark.asyncio
async def test_protected_cell_cannot_die():
    """A protected cell should never enter APOPTOTIC or DEAD state."""
    manifest = CellManifest(
        name="core_service",
        role=CellRole.GOVERNOR,
        subsystem="resilience",
        capabilities=["governance"],
        consumes=[SignalKind.DANGER.value],
        protected=True,
        criticality=0.95,
    )
    cell = MorphogenCell(manifest)
    cell.apoptosis(reason="test_kill")
    assert cell.lifecycle != CellLifecycle.APOPTOTIC, "Protected cell must not enter apoptosis"
    assert cell.lifecycle != CellLifecycle.DEAD, "Protected cell must not die"


@pytest.mark.asyncio
async def test_cell_hibernates_under_low_energy():
    """Cell should hibernate when energy is below threshold."""
    manifest = CellManifest(
        name="hibernator",
        role=CellRole.SENSOR,
        subsystem="test",
        capabilities=["test"],
        consumes=[SignalKind.TASK.value],
        protected=False,
        criticality=0.3,
        hibernation_threshold=0.3,
        activation_threshold=0.01,
    )
    cell = MorphogenCell(manifest)
    cell.state.energy = 0.1  # Below hibernation threshold
    field = MorphogenField()
    field.perturb("test", "task_pressure", 0.5)

    result = await cell.tick(
        signals=[MorphogenSignal(kind=SignalKind.TASK, source="t", subsystem="test", intensity=0.5)],
        field=field,
        global_energy=1.0,
    )
    assert not result.activated, "Cell should not activate under low energy"
    assert cell.lifecycle == CellLifecycle.HIBERNATING


# ---------------------------------------------------------------------------
# 7. Immunity bridge routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_immunity_bridge_routes_high_danger_signals(monkeypatch):
    """Signals above 0.55 intensity should be bridged to adaptive immunity."""
    import core.adaptation.adaptive_immunity as adaptive_immunity_module

    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(
            enabled=True,
            adaptive_immunity_bridge=True,
        )
    )

    class RecordingImmuneSystem:
        def __init__(self):
            self.events = []

        def observe_event(self, event):
            self.events.append(event)

    immune = RecordingImmuneSystem()
    monkeypatch.setattr(adaptive_immunity_module, "get_adaptive_immune_system", lambda: immune)

    danger_signal = MorphogenSignal(
        kind=SignalKind.ERROR,
        source="test",
        subsystem="resilience",
        intensity=0.85,
        payload={"error": "critical_failure"},
    )

    await rt._bridge_signals_to_immunity([danger_signal])
    await rt.wait_for_immunity_idle(timeout_s=1.0)

    assert len(immune.events) == 1
    event = immune.events[0]
    assert event["type"] == SignalKind.ERROR.value
    assert event["danger"] >= 0.85
    assert rt.status()["immunity_bridge"]["processed"] == 1
    await rt.stop()


@pytest.mark.asyncio
async def test_immunity_bridge_ignores_low_intensity(monkeypatch):
    """Signals below 0.55 should NOT be bridged to immunity."""
    import core.adaptation.adaptive_immunity as adaptive_immunity_module

    rt = MorphogeneticRuntime(config=MorphogenesisConfig(adaptive_immunity_bridge=True))

    class RecordingImmuneSystem:
        def __init__(self):
            self.events = []

        def observe_event(self, event):
            self.events.append(event)

    immune = RecordingImmuneSystem()
    monkeypatch.setattr(adaptive_immunity_module, "get_adaptive_immune_system", lambda: immune)

    low_signal = MorphogenSignal(
        kind=SignalKind.ERROR,
        source="test",
        subsystem="test",
        intensity=0.3,  # Below threshold
    )

    await rt._bridge_signals_to_immunity([low_signal])

    assert immune.events == []
    await rt.stop()


@pytest.mark.asyncio
async def test_resource_pressure_requires_true_high_pressure_and_is_environmental(
    monkeypatch,
):
    import core.adaptation.adaptive_immunity as adaptive_immunity_module

    class RecordingImmuneSystem:
        def __init__(self):
            self.events = []

        def observe_event(self, event):
            self.events.append(event)

    immune = RecordingImmuneSystem()
    monkeypatch.setattr(
        adaptive_immunity_module,
        "get_adaptive_immune_system",
        lambda: immune,
    )
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(adaptive_immunity_bridge=True)
    )

    rt._emit_system_signals(resource_pressure=0.70)
    await rt._bridge_signals_to_immunity(rt._consume_signals())
    assert immune.events == []

    rt._emit_system_signals(resource_pressure=0.90)
    await rt._bridge_signals_to_immunity(rt._consume_signals())
    await rt.wait_for_immunity_idle(timeout_s=1.0)

    resource_events = [
        event
        for event in immune.events
        if event["type"] == SignalKind.RESOURCE_PRESSURE.value
    ]
    assert len(resource_events) == 1
    assert resource_events[0]["source_domain"] == "environment"
    assert resource_events[0]["observation_class"] == "resource_telemetry"
    await rt.stop()


@pytest.mark.asyncio
async def test_immunity_bridge_is_async_deduplicated_and_bounded(monkeypatch):
    """Slow immunity work must not block ticks or multiply requeued signals."""
    import core.adaptation.adaptive_immunity as adaptive_immunity_module

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowImmuneSystem:
        def __init__(self):
            self.events = []

        async def observe_event(self, event):
            self.events.append(event)
            started.set()
            await release.wait()

    immune = SlowImmuneSystem()
    monkeypatch.setattr(adaptive_immunity_module, "get_adaptive_immune_system", lambda: immune)
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(
            adaptive_immunity_bridge=True,
            immunity_bridge_queue_capacity=2,
        )
    )
    signal = MorphogenSignal(
        kind=SignalKind.EXCEPTION,
        source="test",
        subsystem="resilience",
        intensity=0.9,
        payload={"exception_type": "TimeoutError"},
    )

    await asyncio.wait_for(rt._bridge_signals_to_immunity([signal]), timeout=0.1)
    await asyncio.wait_for(started.wait(), timeout=0.5)
    await asyncio.wait_for(rt._bridge_signals_to_immunity([signal]), timeout=0.1)

    status = rt.status()["immunity_bridge"]
    assert status["inflight_signal_id"] == signal.signal_id
    assert status["deduplicated"] == 1
    assert len(immune.events) == 1

    release.set()
    await rt.wait_for_immunity_idle(timeout_s=1.0)
    await rt.stop()


@pytest.mark.asyncio
async def test_immunity_bridge_queue_pressure_is_visible_and_nonfatal(monkeypatch):
    import core.adaptation.adaptive_immunity as adaptive_immunity_module

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowImmuneSystem:
        async def observe_event(self, _event):
            started.set()
            await release.wait()

    monkeypatch.setattr(
        adaptive_immunity_module,
        "get_adaptive_immune_system",
        lambda: SlowImmuneSystem(),
    )
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(
            adaptive_immunity_bridge=True,
            immunity_bridge_queue_capacity=1,
            immunity_bridge_max_enqueue_per_tick=8,
        )
    )

    def signal(index: int) -> MorphogenSignal:
        return MorphogenSignal(
            kind=SignalKind.ERROR,
            source=f"source-{index}",
            subsystem="resilience",
            intensity=0.9 - index * 0.01,
            payload={"error": f"failure-{index}"},
        )

    await rt._bridge_signals_to_immunity([signal(0)])
    await asyncio.wait_for(started.wait(), timeout=0.5)
    await rt._bridge_signals_to_immunity([signal(1), signal(2)])

    status = rt.status()["immunity_bridge"]
    assert status["queue_depth"] == 1
    assert status["dropped"] == 1
    assert status["worker_running"] is True

    release.set()
    await rt.wait_for_immunity_idle(timeout_s=1.0)
    await rt.stop()


@pytest.mark.asyncio
async def test_immunity_bridge_contains_timeout_and_processes_next_signal(monkeypatch):
    import core.adaptation.adaptive_immunity as adaptive_immunity_module
    import core.morphogenesis.runtime as runtime_module

    class FlakyImmuneSystem:
        def __init__(self):
            self.calls = 0

        async def observe_event(self, _event):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("slow numerical projection")

    immune = FlakyImmuneSystem()
    monkeypatch.setattr(adaptive_immunity_module, "get_adaptive_immune_system", lambda: immune)
    monkeypatch.setattr(runtime_module, "record_degradation", lambda *_args, **_kwargs: None)
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(adaptive_immunity_bridge=True))
    signals = [
        MorphogenSignal(
            kind=SignalKind.ERROR,
            source=f"source-{index}",
            subsystem="resilience",
            intensity=0.9,
            payload={"error": f"failure-{index}"},
        )
        for index in range(2)
    ]

    await rt._bridge_signals_to_immunity(signals)
    await rt.wait_for_immunity_idle(timeout_s=1.0)

    status = rt.status()["immunity_bridge"]
    assert immune.calls == 2
    assert status["failures"] == 1
    assert status["processed"] == 1
    assert status["worker_running"] is True
    await rt.stop()


# ---------------------------------------------------------------------------
# 8. Hook influence tests
# ---------------------------------------------------------------------------

def test_routing_advice_nominal():
    """When morphogenesis is offline, routing advice should be neutral."""
    from core.morphogenesis.hooks import get_morphogenesis_routing_advice
    advice = get_morphogenesis_routing_advice()
    assert not advice["recommend_downgrade"], "No morphogenesis = no downgrade"


def test_initiative_suppression_default():
    """When morphogenesis is offline, initiative should NOT be suppressed."""
    from core.morphogenesis.hooks import should_suppress_autonomous_initiative
    assert not should_suppress_autonomous_initiative()


@pytest.mark.asyncio
async def test_self_healing_hook_uses_atomic_runtime_restart(monkeypatch):
    import core.runtime.self_healing as self_healing_module
    from core.container import ServiceContainer
    from core.morphogenesis.hooks import register_self_healing_watch

    class Runtime:
        def __init__(self):
            self.config = types.SimpleNamespace(enabled=True, tick_interval_s=0.5)
            self.restart_count = 0

        async def restart_async(self):
            self.restart_count += 1

        async def stop(self):
            raise AssertionError("atomic restart_async should own stop")

        async def start(self):
            raise AssertionError("atomic restart_async should own start")

        def status(self):
            return {"running": self.restart_count > 0}

    class Healer:
        def watch(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs

    runtime = Runtime()
    healer = Healer()
    ServiceContainer.clear()
    ServiceContainer.register_instance("morphogenetic_runtime", runtime, required=False)
    monkeypatch.setattr(self_healing_module, "get_healer", lambda: healer)
    try:
        assert register_self_healing_watch() is True
        assert healer.name == "morphogenesis_runtime"
        assert healer.kwargs["expected_interval_s"] == 5.0
        await healer.kwargs["restart_async"]()
        assert runtime.restart_count == 1
    finally:
        ServiceContainer.clear()


def test_initiative_suppression_under_danger(monkeypatch):
    """When morphogenetic field shows high danger, initiative should be suppressed."""
    import core.container as container_module
    from core.morphogenesis.hooks import should_suppress_autonomous_initiative

    rt = MorphogeneticRuntime()
    rt.field.perturb("global", "danger", 0.8)

    class ServiceContainerFixture:
        @staticmethod
        def get(name, default=None):
            return rt if name == "morphogenetic_runtime" else default

    monkeypatch.setattr(container_module, "ServiceContainer", ServiceContainerFixture)
    assert should_suppress_autonomous_initiative(), "High danger should suppress initiative"


def test_metabolic_modulation_under_pressure(monkeypatch):
    """Under high danger, metabolic energy refill rate should decrease."""
    import core.container as container_module
    from core.morphogenesis.hooks import modulate_metabolic_energy

    rt = MorphogeneticRuntime()
    rt.field.perturb("global", "danger", 0.9)

    class MetabolicCoordinatorFixture:
        pass

    coord = MetabolicCoordinatorFixture()
    coord._energy_refill_rate = 0.05

    class ServiceContainerFixture:
        @staticmethod
        def get(name, default=None):
            if name == "morphogenetic_runtime":
                return rt
            if name == "metabolic_coordinator":
                return coord
            return default

    monkeypatch.setattr(container_module, "ServiceContainer", ServiceContainerFixture)
    modifier = modulate_metabolic_energy()

    assert modifier is not None
    assert modifier < 1.0, f"Under danger, modifier should be < 1.0; got {modifier}"
    assert coord._energy_refill_rate < 0.05, "Refill rate should be reduced under danger"


def test_cell_capability_boost(monkeypatch):
    """Active healthy cells should boost matching tool names."""
    import core.container as container_module
    from core.morphogenesis.hooks import get_cell_capability_boost

    rt = MorphogeneticRuntime()
    manifest = CellManifest(
        name="browser_tool",
        role=CellRole.EFFECTOR,
        subsystem="tools",
        capabilities=["sovereign_browser", "web_search"],
        consumes=[],
    )
    rt.registry.register_cell(manifest)

    class ServiceContainerFixture:
        @staticmethod
        def get(name, default=None):
            return rt if name == "morphogenetic_runtime" else default

    monkeypatch.setattr(container_module, "ServiceContainer", ServiceContainerFixture)
    boost = get_cell_capability_boost("sovereign_browser")
    assert boost > 0.0, "Active cell with matching capability should give boost"

    no_boost = get_cell_capability_boost("nonexistent_tool")
    assert no_boost == 0.0, "No matching cell = no boost"


# ---------------------------------------------------------------------------
# 9. Registry persistence round-trip
# ---------------------------------------------------------------------------

def test_registry_roundtrip(tmp_path):
    """Registry save/load should preserve cell and organ state."""
    config = MorphogenesisConfig()
    reg = MorphogenesisRegistry(config=config, root=tmp_path / "morphogenesis")

    manifest = CellManifest(
        name="persistent_cell",
        role=CellRole.REPAIR,
        subsystem="test",
        capabilities=["persistence"],
        consumes=[],
        criticality=0.75,
    )
    reg.register_cell(manifest)
    assert len(reg.active_cells()) == 1

    reg.save()

    # Create a new registry and load
    reg2 = MorphogenesisRegistry(config=config, root=tmp_path / "morphogenesis")
    reg2.load()
    assert len(reg2.active_cells()) == 1
    assert reg2.active_cells()[0].manifest.name == "persistent_cell"


# ---------------------------------------------------------------------------
# 10. Runtime start/stop lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runtime_start_stop_lifecycle():
    """Runtime should start, tick, and stop cleanly."""
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(
        enabled=True,
        tick_interval_s=0.05,
    ))

    # Register a test cell
    manifest = CellManifest(
        name="lifecycle_test",
        role=CellRole.SENSOR,
        subsystem="test",
        capabilities=["lifecycle"],
        consumes=[SignalKind.HEARTBEAT.value],
        activation_threshold=0.01,
    )
    rt.registry.register_cell(manifest)

    await rt.start()
    assert rt.status()["running"]

    # Run a manual tick
    result = await rt.tick()
    assert result["tick"] >= 1

    await rt.stop()
    assert not rt.status()["running"]


@pytest.mark.asyncio
async def test_runtime_defers_heavy_tick_during_foreground_quiet_window(tmp_path, monkeypatch):
    """Foreground conversation quiet windows should pause heavy morphogenesis ticks."""
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(
            enabled=True,
            tick_interval_s=0.01,
        ),
        registry=MorphogenesisRegistry(config=MorphogenesisConfig(), root=tmp_path / "morphogenesis"),
    )

    class TickRecorder:
        def __init__(self):
            self.await_count = 0

        async def __call__(self):
            self.await_count += 1

    tick_recorder = TickRecorder()
    rt.tick = tick_recorder
    monkeypatch.setattr(MorphogeneticRuntime, "_foreground_quiet_window_active", staticmethod(lambda: True))

    await rt.start()
    await asyncio.sleep(0.05)
    await rt.stop()

    assert tick_recorder.await_count == 0


@pytest.mark.asyncio
async def test_runtime_disabled_does_not_start():
    """If config.enabled=False, start() should be a no-op."""
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False))
    await rt.start()
    assert not rt.status()["running"]


@pytest.mark.asyncio
async def test_runtime_start_falls_back_when_task_tracker_unavailable(monkeypatch, tmp_path):
    """Task tracker failure should not prevent morphogenesis from coming alive."""
    import core.runtime.task_ownership as task_ownership_module

    tracker_lookups = 0

    def unavailable_tracker():
        nonlocal tracker_lookups
        tracker_lookups += 1
        return None

    monkeypatch.setattr(task_ownership_module, "_get_tracker", unavailable_tracker)

    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(enabled=True, tick_interval_s=0.05),
        registry=MorphogenesisRegistry(config=MorphogenesisConfig(), root=tmp_path / "morphogenesis"),
    )

    await rt.start()
    assert rt.status()["running"]
    assert tracker_lookups == 2
    assert rt._task is not None
    assert not rt._task.done()
    assert rt._immunity_task is not None
    assert not rt._immunity_task.done()
    await rt.stop()


@pytest.mark.asyncio
async def test_runtime_loop_failure_records_signal_and_backoff(monkeypatch):
    """A failed tick should become a causal error signal, not a silent loop."""
    import core.morphogenesis.runtime as runtime_module

    recorded: list[tuple[str, str, dict[str, object]]] = []

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(runtime_module, "record_degradation", record_degradation)
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=True, tick_interval_s=0.01))

    async def failing_tick():
        rt._stopping.set()
        raise RuntimeError("tick failed")

    rt.tick = failing_tick
    await rt._run_loop()

    status = rt.status()
    assert status["consecutive_tick_failures"] == 1
    assert "tick failed" in status["last_tick_error"]
    assert any(signal.kind == SignalKind.ERROR for signal in rt._signals)
    assert "backed off" in str(recorded[0][2]["action"])


@pytest.mark.asyncio
async def test_runtime_loop_contains_timeout_and_remains_restartable(monkeypatch):
    import core.morphogenesis.runtime as runtime_module

    recorded = []
    monkeypatch.setattr(
        runtime_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, exc, kwargs)),
    )
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(enabled=True, tick_interval_s=0.01)
    )

    async def timed_out_tick():
        rt._stopping.set()
        raise TimeoutError("immune observer exceeded its local budget")

    rt.tick = timed_out_tick
    await rt._run_loop()

    assert rt.status()["consecutive_tick_failures"] == 1
    assert "TimeoutError" in rt.status()["last_tick_error"]
    assert recorded


@pytest.mark.asyncio
async def test_stop_contains_failure_from_already_finished_owned_task(monkeypatch):
    import core.morphogenesis.runtime as runtime_module

    recorded = []
    monkeypatch.setattr(
        runtime_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, exc, kwargs)),
    )
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False))

    async def fail():
        raise TimeoutError("prior runtime failure")

    rt._task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    await rt.stop()

    assert rt._task is None
    assert any(isinstance(item[1], TimeoutError) for item in recorded)


@pytest.mark.asyncio
async def test_start_morphogenesis_runtime_marks_hook_failure_and_emits_signal(monkeypatch):
    """Hook wiring failure should be visible in runtime state and signal flow."""
    import core.morphogenesis.integration as integration_module

    recorded: list[tuple[str, str, dict[str, object]]] = []

    async def wire_all_hooks():
        attempted = True
        assert attempted
        raise RuntimeError("hooks offline")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    hooks_module = types.ModuleType("core.morphogenesis.hooks")
    hooks_module.wire_all_hooks = wire_all_hooks
    monkeypatch.setitem(sys.modules, "core.morphogenesis.hooks", hooks_module)
    monkeypatch.setattr(integration_module, "record_degradation", record_degradation)

    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False))
    returned = await integration_module.start_morphogenesis_runtime(rt)

    assert returned is rt
    assert rt.status()["hooks_wired"] is False
    assert any(signal.kind == SignalKind.ERROR for signal in rt._signals)
    assert recorded[0][0] == "morphogenesis.integration"
    assert "hook-wiring error signal" in str(recorded[0][2]["action"])


# ---------------------------------------------------------------------------
# 11. Observe exception flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observe_exception_emits_signal():
    """observe_exception should create an EXCEPTION signal and ingest it."""
    rt = MorphogeneticRuntime()
    sig = rt.observe_exception(
        subsystem="test",
        exc=RuntimeError("boom"),
        source="test_source",
        danger=0.8,
    )
    assert sig.kind == SignalKind.EXCEPTION
    assert sig.intensity == 0.8
    assert "boom" in sig.payload.get("message", "")
    assert rt.field.sample("test")["danger"] > 0.0


# ---------------------------------------------------------------------------
# 12. Signal TTL decay
# ---------------------------------------------------------------------------

def test_signal_ttl_decays_on_consume():
    """Consumed signals should re-queue with reduced TTL and intensity."""
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(
        signal_decay_per_tick=1,
        max_signals_per_tick=1,  # Limit to 1 so we can observe re-queue
    ))

    sig = MorphogenSignal(
        kind=SignalKind.TASK,
        source="test",
        subsystem="test",
        intensity=0.5,
        ttl_ticks=3,
    )
    rt._signals.append(sig)  # Direct append to avoid field.ingest_signal
    consumed = rt._consume_signals()
    assert len(consumed) == 1

    # After consume, the re-queued signal should have reduced TTL
    assert len(rt._signals) >= 1
    requeued = rt._signals[0]
    assert requeued.ttl_ticks == 2  # 3 - 1
    assert requeued.intensity < 0.5  # decayed by 0.92 factor


# ---------------------------------------------------------------------------
# 13. JSON safety helpers
# ---------------------------------------------------------------------------

def test_json_safe_handles_edge_cases():
    """json_safe should handle nested dicts, lists, and non-serialisable types."""
    data = {
        "normal": "string",
        "number": 42,
        "nested": {"inner": [1, 2, 3]},
        "set_value": {1, 2, 3},  # Sets are not JSON-safe
    }
    safe = json_safe(data)
    serialised = json.dumps(safe, default=str)
    assert "normal" in serialised


def test_stable_digest_deterministic():
    """Same inputs should produce the same digest."""
    d1 = stable_digest("a", "b", "c")
    d2 = stable_digest("a", "b", "c")
    assert d1 == d2
    assert len(d1) == 16  # default length

    d3 = stable_digest("a", "b", "c", length=8)
    assert len(d3) == 8


def test_clamp01():
    assert clamp01(1.5) == 1.0
    assert clamp01(-0.5) == 0.0
    assert clamp01(0.5) == 0.5


def test_a_full_bridge_that_drains_is_backpressure_and_a_stuck_one_is_a_degradation(monkeypatch, caplog):
    """Four QueueFull degradations per boot on a loaded host, each feeding the
    resilience engine, for a queue that was draining. Said at info while the
    observer keeps up; recorded only when nothing has drained since the last
    report."""
    import logging

    from core.morphogenesis import runtime as runtime_module
    from core.morphogenesis import runtime_immunity as bridge_module

    recorded: list[dict] = []
    monkeypatch.setattr(
        runtime_module,
        "_record_morphogenesis_runtime_degradation",
        lambda error, **kwargs: recorded.append(kwargs),
    )
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(
            adaptive_immunity_bridge=True,
            immunity_bridge_queue_capacity=1,
            immunity_bridge_degradation_interval_s=1.0,
        )
    )
    rt._last_immunity_degradation_at = 0.0
    extra = {"queue_depth": 1, "queue_capacity": 1, "signal_kind": "error"}

    with caplog.at_level(logging.INFO, logger=bridge_module.logger.name):
        # The observer processed something since the last report: draining.
        rt._immunity_processed = 3
        rt._record_immunity_bridge_degradation(asyncio.QueueFull(), action="x", extra=extra)
    assert recorded == []
    assert any("not stuck" in r.getMessage() for r in caplog.records)

    # Nothing processed since: stuck, and recorded.
    rt._last_immunity_degradation_at = 0.0
    rt._record_immunity_bridge_degradation(asyncio.QueueFull(), action="x", extra=extra)
    assert len(recorded) == 1 and recorded[0]["extra"]["stuck"] is True
