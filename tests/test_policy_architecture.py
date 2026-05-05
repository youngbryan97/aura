import pytest
from core.environment.policy import CandidateGenerator, ActionRanker, TacticalPolicy, StrategicPolicy
from core.environment.asset import AssetModel, Asset
from core.environment.spatial import TopologicalMap, MapNode
from core.environment.hazard import HazardModel, Hazard
from core.environment.resources import ResourceCalendar, TimeSeriesPoint
from core.environment.strategy import HTNPlanner, TaskNode
from core.environment.outcome import SemanticDiffLearner, CausalLink
from core.environment.lifecycle_manager import LifecycleManager, TerminalState
from core.environment.boundary_guard import BoundaryGuard, BoundaryViolationError
from core.environment.benchmark_runner import BenchmarkRunner
from core.environments.terminal_grid.nethack_adapter import NetHackTerminalGridAdapter, EnvironmentMode
from core.environment.adapter import EnvironmentUnavailableError


def test_imports_and_instantiation():
    assert CandidateGenerator()
    assert ActionRanker()
    assert TacticalPolicy()
    assert StrategicPolicy()
    
    assert AssetModel()
    assert TopologicalMap()
    assert HazardModel()
    assert ResourceCalendar()
    assert HTNPlanner()
    assert SemanticDiffLearner()
    assert LifecycleManager("test_run")
    assert BoundaryGuard()
    
    # We can pass a dummy factory
    runner = BenchmarkRunner(lambda: None)
    assert runner


def test_boundary_guard():
    guard = BoundaryGuard()
    # Allowed
    guard.check_operation("print", "stdout")
    
    # Blocked operation
    with pytest.raises(BoundaryViolationError):
        guard.check_operation("inspect_memory")
        
    # Blocked channel
    with pytest.raises(BoundaryViolationError):
        guard.check_operation("print", "internal_ipc")


@pytest.mark.asyncio
async def test_strict_real_mode():
    # If we request strict real but pass a fake path, it should raise
    adapter = NetHackTerminalGridAdapter(
        nethack_path="/fake/path/does/not/exist",
        mode=EnvironmentMode.STRICT_REAL
    )
    
    with pytest.raises(EnvironmentUnavailableError):
        await adapter.start(run_id="test_strict")
        
    # Simulated mode should not raise
    adapter_sim = NetHackTerminalGridAdapter(
        nethack_path="/fake/path/does/not/exist",
        mode=EnvironmentMode.SIMULATED
    )
    await adapter_sim.start(run_id="test_sim")
    assert adapter_sim._simulated is True
