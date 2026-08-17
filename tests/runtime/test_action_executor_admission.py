from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor


class _Decision:
    def __init__(self, *, approved: bool, reason: str = "") -> None:
        self._approved = approved
        self.reason = reason
        self.receipt_id = "will-test-receipt"

    def is_approved(self) -> bool:
        return self._approved


def test_action_executor_owns_one_context_bound_admission(monkeypatch):
    calls = []

    class Will:
        def decide(self, **kwargs):
            calls.append(kwargs)
            return _Decision(approved=True, reason="approved")

    monkeypatch.setattr("core.will.get_will", lambda: Will())

    admission = ActionExecutor.authorize_action(
        domain=ActionDomain.TOOL_EXECUTION,
        action_name="web_search",
        params={"query": "public octopus cognition"},
        source="curiosity",
        priority=0.7,
        context={"standing_authority_token": "signed-test-token"},
    )

    assert admission.approved is True
    assert admission.receipt_id == "will-test-receipt"
    assert len(calls) == 1
    assert calls[0]["domain"] is ActionDomain.TOOL_EXECUTION
    assert calls[0]["source"] == "curiosity"
    assert calls[0]["priority"] == 0.7
    # The token is the contract; the rest of the context is provenance the
    # executor is free to add. Exact-dict equality made this test fail the
    # moment action_executor_action_name and action_executor_source were
    # recorded alongside it — a strictly better receipt breaking a test that
    # only cared about one key.
    context = calls[0]["context"]
    assert context["standing_authority_token"] == "signed-test-token"
    assert context.get("action_executor_action_name") == "web_search"
    assert "web_search" in calls[0]["content"]


def test_action_executor_admission_fails_closed_on_malformed_decision(monkeypatch):
    monkeypatch.setattr(
        "core.will.get_will",
        lambda: SimpleNamespace(decide=lambda **_kwargs: SimpleNamespace()),
    )

    admission = ActionExecutor.authorize_action(
        domain="tool_execution",
        action_name="status",
        params={},
    )

    assert admission.approved is False
    assert admission.receipt_id == ""


@pytest.mark.parametrize("priority", [float("nan"), float("inf"), -0.1, 1.1, "bad"])
def test_action_executor_admission_rejects_invalid_priority(priority):
    with pytest.raises(ValueError, match="priority"):
        ActionExecutor.authorize_action(
            domain="tool_execution",
            action_name="status",
            params={},
            priority=priority,
        )
