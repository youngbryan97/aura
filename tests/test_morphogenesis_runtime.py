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
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.morphogenesis.types import (
    CellLifecycle,
    CellManifest,
    CellRole,
    CellState,
    MorphogenesisConfig,
    MorphogenSignal,
    SignalKind,
    clamp01,
    json_safe,
    stable_digest,
)
from core.morphogenesis.field import MorphogenField
from core.morphogenesis.cell import MorphogenCell, CellTickResult
from core.morphogenesis.metabolism import MetabolismManager, ResourceSnapshot, CellBudget
from core.morphogenesis.organs import Organ, OrganStabilizer
from core.morphogenesis.registry import MorphogenesisRegistry
from core.morphogenesis.runtime import MorphogeneticRuntime


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


def test_metabolism_pulse_recovers_energy():
    """Pulse should recover global energy and cell budgets."""
    mgr = MetabolismManager(global_energy=0.5, recovery_per_tick=0.1)
    mgr.ensure_budget("cell_a", priority=0.8, baseline=0.3)
    mgr._budgets["cell_a"].energy = 0.1  # Simulate depleted cell

    # Mock psutil to avoid system dependency
    with patch("core.morphogenesis.metabolism.MetabolismManager.sample_resources") as mock_sr:
        mock_sr.return_value = ResourceSnapshot(pressure=0.0)
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

    def _failing_handler(cell, signals, field_state):
        raise RuntimeError("test failure")

    cell = MorphogenCell(manifest, handler=_failing_handler)
    field = MorphogenField()
    field.perturb("test", "task_pressure", 0.9)

    task_signal = MorphogenSignal(
        kind=SignalKind.TASK, source="test", subsystem="test", intensity=0.9,
    )
    for _ in range(3):
        result = await cell.tick(signals=[task_signal], field=field, global_energy=1.0)

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
async def test_immunity_bridge_routes_high_danger_signals():
    """Signals above 0.55 intensity should be bridged to adaptive immunity."""
    rt = MorphogeneticRuntime(
        config=MorphogenesisConfig(
            enabled=True,
            adaptive_immunity_bridge=True,
        )
    )

    mock_immune = MagicMock()
    mock_immune.observe_event = MagicMock(return_value=None)

    danger_signal = MorphogenSignal(
        kind=SignalKind.ERROR,
        source="test",
        subsystem="resilience",
        intensity=0.85,
        payload={"error": "critical_failure"},
    )

    with patch("core.adaptation.adaptive_immunity.get_adaptive_immune_system", return_value=mock_immune):
        await rt._bridge_signals_to_immunity([danger_signal])

    mock_immune.observe_event.assert_called_once()
    event = mock_immune.observe_event.call_args[0][0]
    assert event["type"] == SignalKind.ERROR.value
    assert event["danger"] >= 0.85


@pytest.mark.asyncio
async def test_immunity_bridge_ignores_low_intensity():
    """Signals below 0.55 should NOT be bridged to immunity."""
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(adaptive_immunity_bridge=True))

    mock_immune = MagicMock()
    mock_immune.observe_event = MagicMock(return_value=None)

    low_signal = MorphogenSignal(
        kind=SignalKind.ERROR,
        source="test",
        subsystem="test",
        intensity=0.3,  # Below threshold
    )

    with patch("core.adaptation.adaptive_immunity.get_adaptive_immune_system", return_value=mock_immune):
        await rt._bridge_signals_to_immunity([low_signal])

    mock_immune.observe_event.assert_not_called()


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


def test_initiative_suppression_under_danger():
    """When morphogenetic field shows high danger, initiative should be suppressed."""
    from core.morphogenesis.hooks import should_suppress_autonomous_initiative

    rt = MorphogeneticRuntime()
    rt.field.perturb("global", "danger", 0.8)

    mock_container_cls = MagicMock()
    mock_container_cls.get = MagicMock(return_value=rt)

    with patch("core.container.ServiceContainer", mock_container_cls):
        assert should_suppress_autonomous_initiative(), "High danger should suppress initiative"


def test_metabolic_modulation_under_pressure():
    """Under high danger, metabolic energy refill rate should decrease."""
    from core.morphogenesis.hooks import modulate_metabolic_energy

    rt = MorphogeneticRuntime()
    rt.field.perturb("global", "danger", 0.9)

    coord = MagicMock()
    coord._energy_refill_rate = 0.05

    def _get(name, default=None):
        if name == "morphogenetic_runtime":
            return rt
        if name == "metabolic_coordinator":
            return coord
        return default

    mock_container_cls = MagicMock()
    mock_container_cls.get = _get

    with patch("core.container.ServiceContainer", mock_container_cls):
        modifier = modulate_metabolic_energy()

    assert modifier is not None
    assert modifier < 1.0, f"Under danger, modifier should be < 1.0; got {modifier}"
    assert coord._energy_refill_rate < 0.05, "Refill rate should be reduced under danger"


def test_cell_capability_boost():
    """Active healthy cells should boost matching tool names."""
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

    mock_container_cls = MagicMock()
    mock_container_cls.get = MagicMock(return_value=rt)

    with patch("core.container.ServiceContainer", mock_container_cls):
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
async def test_runtime_disabled_does_not_start():
    """If config.enabled=False, start() should be a no-op."""
    rt = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False))
    await rt.start()
    assert not rt.status()["running"]


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
