from core.environment.action_budget import ActionBudget
from core.environment.crisis import CrisisManager
from core.environment.cognition_router import CognitionRequest, CognitionRouter


def test_crisis_forces_allowed_options_and_blocks_progress():
    crisis = CrisisManager().assess(critical_resources=["health"], unknown_modal=True)
    assert crisis.active
    assert "STABILIZE_RESOURCE" in crisis.forced_options
    assert "progress" in crisis.forbidden_actions
    assert crisis.exit_conditions


def test_action_budget_enforces_repeated_failures_and_irreversible_limits():
    budget = ActionBudget(10, 0, 1, 2, 5, 1.0)
    budget.record(action_name="submit", irreversible=True)
    budget.record(action_name="move", failed=True)
    budget.record(action_name="move", failed=True)
    budget.record(action_name="move", failed=True)
    reasons = budget.exhausted_reasons()
    assert "max_irreversible_actions" in reasons
    assert "max_repeated_failures" in reasons


def test_model_tier_router_escalates_high_risk():
    route = CognitionRouter().route(CognitionRequest("choice", urgency=0.5, risk=0.9, uncertainty=0.2, token_budget=1000, context={}))
    assert route.model_tier == "cortex"


def test_subsystem_auto_recovery():
    import time
    from core.runtime.errors import SubsystemRegistry

    registry = SubsystemRegistry()
    health = registry.register("test_subsystem")
    
    # Check that initial state is healthy
    assert health.status == "healthy"
    
    # Mark it degraded
    health.mark_degraded("simulated failure")
    assert health.status == "degraded"
    
    # Auto-recovery checking with timeout should not recover if not enough time passed
    recovered = registry.auto_recover_subsystems(timeout_seconds=300.0)
    assert len(recovered) == 0
    assert health.status == "degraded"
    
    # Backdate last_failed_at to 10 minutes ago
    health.last_failed_at = time.time() - 600.0
    recovered = registry.auto_recover_subsystems(timeout_seconds=300.0)
    assert "test_subsystem" in recovered
    assert health.status == "healthy"

