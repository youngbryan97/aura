"""A lease must validate the way it was issued.

LIVE 2026-08-19, from the runtime log: computer_use, desktop_task,
self_evolution and read_screen_text all refused with

    denied_by_default: tool_execution requires validated scoped authority
    (standing_authority_effect_scope_mismatch)

Her self-directed action could not execute a tool at all, and a screen read
came back to the person as an executive veto.

issue_child_lease derives the effect scope from the tool when the caller does
not pass one, and classifies the risk the same way. validate_context did
neither. A caller that omitted them presented "" for the scope — and
normalize_risk("") returns "critical", not "" — against a lease recorded as
"status" and "low". Neither could ever match: not a policy decision, a
guaranteed inequality.

Nothing is loosened. The comparison against the RECORDED values is unchanged,
and the tool and arguments the fallbacks derive from are bound separately, so
they can only reproduce the recorded values for the same tool and arguments
the lease was issued for. A lease with no token is still refused.
"""

from __future__ import annotations

import asyncio

from core.executive.standing_authority import (
    get_standing_authority_manager,
    normalize_risk,
)


def _issue_then_validate(strip: bool) -> tuple[bool, str]:
    """Issue a lease and validate it in the SAME task the token is bound to."""

    async def run() -> tuple[bool, str]:
        manager = get_standing_authority_manager()
        decision = await manager.issue_child_lease(
            "clock", {}, origin="desktop_ui", user_authorized=True
        )
        context = dict(decision.context or {})
        if strip:
            context.pop("effect_scope", None)
            context.pop("risk_level", None)
        valid, reason, _record = manager.validate_context(
            context, tool_name="clock", arguments={}, origin="desktop_ui"
        )
        return valid, reason

    return asyncio.run(run())


def test_a_lease_validates_with_the_scope_and_risk_it_was_issued_with() -> None:
    valid, reason = _issue_then_validate(strip=False)

    assert valid is True, reason


def test_the_same_lease_validates_when_the_caller_omits_them() -> None:
    """The live failure: every omission was a refusal nobody could fix."""
    valid, reason = _issue_then_validate(strip=True)

    assert valid is True, reason
    assert "effect_scope_mismatch" not in reason
    assert "risk_mismatch" not in reason


def test_an_empty_risk_is_not_silently_critical() -> None:
    """Why testing the normalised value could not find the gap."""
    assert normalize_risk("") == "critical"


def test_a_missing_token_is_still_refused() -> None:
    manager = get_standing_authority_manager()
    valid, reason, record = manager.validate_context({}, tool_name="clock")

    assert valid is False
    assert reason == "signed_standing_authority_lease_missing"
    assert record is None
