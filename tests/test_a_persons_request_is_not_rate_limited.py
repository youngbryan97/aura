"""The escalation is for a runaway agent, not for the person at the keyboard.

Three approved medium-risk actions in sixty seconds escalate to requiring
confirmation. It counted every approved MEDIUM decision in the window,
including the background loops that run constantly — curiosity, dreaming and
auto-refactor reach three between them easily — and then the next thing
escalated was whatever a person asked for next.

Live on 2026-08-29: "use that library to record this" reached the sandbox and
came back "WILL DEFERRED: response_generation_user/tool_execution --
permission_model_requires_confirmation: Escalated due to rapid MEDIUM
actions", asking the person to confirm the thing they had just asked for in
those words.
"""

from __future__ import annotations

from core.capabilities.permission_model import PermissionRiskModel


def _burst(model: PermissionRiskModel, count: int, origin: str) -> None:
    for index in range(count):
        model.check_permission(
            "code_repl",
            f"{origin}-{index}",
            {"origin": origin},
            effect_scope="sandboxed_compute",
            execution_risk="medium",
        )


def test_a_person_is_not_escalated_by_autonomous_volume() -> None:
    model = PermissionRiskModel()
    _burst(model, 6, "curiosity")
    decision = model.check_permission(
        "code_repl",
        "the thing they asked for",
        {"origin": "desktop_quick_user", "a_person_is_waiting": True},
        effect_scope="sandboxed_compute",
        execution_risk="medium",
    )
    assert decision.approved is True, decision.reason
    assert decision.requires_confirmation is False


def test_a_runaway_agent_still_escalates() -> None:
    """The case it was written for, unchanged."""

    model = PermissionRiskModel()
    _burst(model, 6, "curiosity")
    decision = model.check_permission(
        "code_repl",
        "more of its own",
        {"origin": "curiosity"},
        effect_scope="sandboxed_compute",
        execution_risk="medium",
    )
    assert decision.approved is False
    assert decision.requires_confirmation is True
    assert "rapid MEDIUM" in decision.reason


def test_a_persons_own_actions_do_not_fill_the_window() -> None:
    """Asking for several things quickly is not a runaway agent either."""

    model = PermissionRiskModel()
    _burst(model, 6, "desktop_quick_user")
    decision = model.check_permission(
        "code_repl",
        "one more",
        {"origin": "desktop_quick_user"},
        effect_scope="sandboxed_compute",
        execution_risk="medium",
    )
    assert decision.approved is True, decision.reason


def test_the_decision_records_who_asked() -> None:
    model = PermissionRiskModel()
    asked = model.check_permission(
        "code_repl", "x", {"origin": "user"},
        effect_scope="sandboxed_compute", execution_risk="medium",
    )
    unasked = model.check_permission(
        "code_repl", "y", {"origin": "dream_cycle"},
        effect_scope="sandboxed_compute", execution_risk="medium",
    )
    assert asked.asked_for is True
    assert unasked.asked_for is False
