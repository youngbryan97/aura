import time
from types import SimpleNamespace

from core.autonomy.research_cycle import ResearchCycle


def test_research_cycle_respects_background_policy_gate_before_starting(monkeypatch):
    cycle = ResearchCycle.__new__(ResearchCycle)
    cycle.orchestrator = SimpleNamespace(_last_user_interaction_time=0.0, status=SimpleNamespace(is_processing=False))
    cycle._last_cycle_mono = 0.0
    cycle._get_state = lambda: SimpleNamespace(
        motivation=SimpleNamespace(budgets={"energy": {"level": 100.0}}),
        affect=SimpleNamespace(curiosity=0.8),
        cognition=SimpleNamespace(pending_initiatives=[{"goal": "Research continuity"}]),
    )

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "no_user_anchor",
    )

    assert cycle._should_run() is False
