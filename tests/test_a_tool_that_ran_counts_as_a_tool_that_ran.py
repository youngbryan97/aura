"""A tool receipt on any lane must reach the guard that reads the turn ledger.

Two stores hold the same fact. `record_tool_receipt` writes turn evidence
custody; the honesty guards read the turn EFFECT ledger. Only
`record_verified_effects` wrote both, and it has exactly one caller — the
desktop-task lane — so a tool run on any other lane left the effect ledger
empty and the guard concluded nothing had happened.

LIVE 2026-09-07: asked to compute a sum by actually running code, `code_repl`
was dispatched from the conversational lane, completed in 521ms, and the answer
was right. The reply was served with a correction appended saying to treat it
as not done. She had done it, said so truthfully, and was contradicted by a
guard that had never been told.
"""

from __future__ import annotations

import pytest

from core.conversation.surface_disposition import record_tool_receipt
from core.epistemics.turn_effects import turn_has_verified_effect
from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    current_turn_evidence_custody,
)
from core.runtime.turn_outcome import TurnOutcome, bind_turn, current_turn


@pytest.fixture
def bound_turn():
    with bind_turn(TurnOutcome(turn_id="test-turn")) as outcome:
        yield outcome


@pytest.fixture
def admitted_custody():
    with bind_turn_evidence_custody(session_id="test-session", turn_id="test-turn"):
        assert current_turn_evidence_custody() is not None
        yield


def test_an_observed_tool_effect_reaches_the_turn_ledger(bound_turn, admitted_custody):
    recorded = record_tool_receipt(
        "code_repl",
        ok=True,
        action="run_code",
        object_ref="sum(range(1,101))",
        effect_observed=True,
        verification="postcondition_verified",
        evidence="stdout: 5050",
    )
    assert recorded is True
    assert turn_has_verified_effect() is True, (
        "a tool that ran, on a lane that is not desktop_task, still did not "
        "count as having run"
    )


def test_a_tool_that_only_returned_success_is_not_observed(bound_turn, admitted_custody):
    """Asserted is not observed, which is what makes a false claim catchable."""
    record_tool_receipt(
        "web_search",
        ok=True,
        action="search",
        effect_observed=False,
        evidence="returned rows",
    )
    assert turn_has_verified_effect() is False


def test_the_desktop_lane_does_not_record_the_same_effect_twice(bound_turn, admitted_custody):
    from core.epistemics.turn_effects import record_verified_effects, turn_effect_evidence

    record_verified_effects(
        [{"action": "write_file", "ok": True, "effect_evidence": "file exists"}],
        lane="desktop_task",
    )
    evidence = turn_effect_evidence()
    names = [str(item) for item in (evidence.get("effects") or evidence.get("names") or [])]
    duplicated = [name for name in names if name.startswith("tool:")]
    assert not duplicated, f"the same effect landed twice: {names}"


def test_nothing_is_recorded_without_a_bound_turn(admitted_custody):
    """A background tick has no user-facing reply to keep honest."""
    assert current_turn() is None
    assert record_tool_receipt("code_repl", ok=True, action="run_code") is True
    assert turn_has_verified_effect() is False
