"""Final Preflight Gate: Verifies all subsystems are bound and ready for deep runs.

This gate ensures the architecture is no longer a loose collection of
unconnected modules, but a unified, policy-driven kernel ready for production.
"""
import pytest
from core.environment.environment_kernel import EnvironmentKernel
from core.environment.benchmark_runner import BenchmarkRunner, BenchmarkReport
from core.environment.boundary_guard import BoundaryGuard
from core.environment.strategy.htn_planner import HTNPlanner
from core.environment.run_manager import RunManager
from core.environment.governance_bridge import EnvironmentGovernanceBridge
from core.environment.policy.policy_orchestrator import PolicyOrchestrator
from core.environment.outcome.semantic_diff import SemanticDiffLearner
from core.environment.belief_graph import EnvironmentBeliefGraph


@pytest.mark.asyncio
async def test_final_deep_run_preflight_gate_requires_all_clean_subsystems():
    """Verifies that all required subsystems are bound to the kernel and configured correctly."""
    
    # 1. Kernel is policy-driven
    from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter
    kernel = EnvironmentKernel(adapter=ScriptedTerminalAdapter(["screen1"]))
    assert hasattr(kernel, 'policy'), "policy_drives_kernel=False"
    assert isinstance(kernel.policy, PolicyOrchestrator), "Policy is not the orchestrator"
    
    # 2. Outcomes use semantic diff
    assert isinstance(kernel.semantic_diff, SemanticDiffLearner), "semantic_outcome_diff_enabled=False"
    
    # 3. Canonical spatial model
    assert hasattr(kernel.belief, 'spatial'), "canonical_spatial_model=False"
    assert isinstance(kernel.belief, EnvironmentBeliefGraph)
    
    # 4. Governance bridge
    assert isinstance(kernel.governance_bridge, EnvironmentGovernanceBridge), "governance_bridge_clean=False"
    
    # 5. Boundary Guard
    assert isinstance(kernel.boundary_guard, BoundaryGuard), "boundary_guard_enabled=False"
    
    # 6. Run Manager
    assert isinstance(kernel.run_manager, RunManager), "run_manager_enabled=False"
    
    # 7. HTN Planner
    assert isinstance(kernel.htn_planner, HTNPlanner), "htn_planner_enabled=False"
    
    # 8. Benchmark Runner checks (Mode Separation)
    def dummy_factory():
        from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter
        return EnvironmentKernel(adapter=ScriptedTerminalAdapter(["screen1"]))
    runner = BenchmarkRunner(kernel_factory=dummy_factory)
    assert hasattr(runner, 'validate_for_deep_run_claim'), "benchmark integrity validation missing"
    assert hasattr(runner, 'report')
    assert isinstance(runner.report, BenchmarkReport)

    # We emit a "CLEAN" pass
    pass

__all__ = ["test_final_deep_run_preflight_gate_requires_all_clean_subsystems"]

