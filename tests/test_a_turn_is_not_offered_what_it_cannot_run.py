"""The offered tools are the ones this turn is allowed to use.

LIVE, 2026-08-25. Asked to diagnose a project with a symptom and no traceback,
the turn was offered diagnose_repo, code_repl and file_operation. It reached for
file_operation — state_mutation, above this turn's sandboxed_compute ceiling —
and the executive vetoed it. Then for code_repl, which the permission model
refused for want of a confirmation nobody could give. Both of the turn's two
tool calls went to tools that could never have run, and the one that would have
answered was never called. The reply was "I couldn't get to an answer I'd stand
behind".

The rule was already written down in inference_gate: "Offering a capability the
dispatch then refuses is worse than not offering it". The ceiling was computed
on the line above and assigned to `_ceiling`, then discarded.
"""

from __future__ import annotations

from core.brain.inference_gate import _tools_within_reach

_SANDBOX_CEILING = frozenset({"status", "pure_compute", "read_only", "sandboxed_compute"})


def test_a_tool_above_the_ceiling_is_not_offered() -> None:
    kept, withheld = _tools_within_reach({"diagnose_repo": 1, "file_operation": 2}, _SANDBOX_CEILING)
    assert "file_operation" in withheld, "state_mutation was offered under a sandbox ceiling"
    assert "diagnose_repo" in kept


def test_a_tool_needing_a_confirmation_nobody_can_give_is_not_offered() -> None:
    """code_repl passes the ceiling and still cannot run.

    Running code the model just wrote is correctly high risk. Offering it spends
    a tool call on a refusal.
    """
    kept, withheld = _tools_within_reach({"diagnose_repo": 1, "code_repl": 2}, _SANDBOX_CEILING)
    assert "code_repl" in withheld
    assert "diagnose_repo" in kept


def test_the_tool_that_answers_survives_the_filter() -> None:
    """diagnose_repo runs the project's own code, not the model's."""
    kept, _ = _tools_within_reach(
        {"diagnose_repo": 1, "code_repl": 2, "file_operation": 3, "web_search": 4},
        _SANDBOX_CEILING,
    )
    assert sorted(kept) == ["diagnose_repo", "web_search"]


def test_an_unrated_skill_is_still_offered() -> None:
    """Withholding what has not been rated would hide every new skill.

    That failure is worse than one refusal, and it is how a skill built for a
    request ends up never being called.
    """
    kept, withheld = _tools_within_reach({"brand_new_skill": 1}, _SANDBOX_CEILING)
    assert "brand_new_skill" in kept
    assert not withheld


def test_no_ceiling_withholds_nothing() -> None:
    """An empty permission set means the caller did not compute one."""
    tools = {"diagnose_repo": 1, "file_operation": 2}
    kept, withheld = _tools_within_reach(tools, frozenset())
    assert kept == tools
    assert not withheld


def test_the_ceiling_for_a_diagnosis_request_admits_the_diagnosis_tool() -> None:
    """End to end on the shape of the live request."""
    from core.phases.response_contract import requested_effect_ceiling
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    ceiling, allowed = requested_effect_ceiling(
        "Something weird is happening in a little project of mine. There's no error "
        "and no failing test, but the second invoice comes out with the first one's "
        "lines in it. What's actually going on?"
    )
    assert SKILL_EFFECT_SCOPES["diagnose_repo"] in allowed, (
        f"a {ceiling} ceiling excludes the tool that answers the question"
    )
