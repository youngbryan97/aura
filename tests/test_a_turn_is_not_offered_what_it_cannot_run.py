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


_READ_ONLY_CEILING = frozenset({"status", "pure_compute", "read_only"})


def test_a_tool_whose_every_action_is_out_of_reach_is_not_offered() -> None:
    """diagnose_repo is sandboxed_compute all the way down."""
    kept, withheld = _tools_within_reach({"diagnose_repo": 1}, _READ_ONLY_CEILING)
    assert "diagnose_repo" in withheld
    assert not kept


def test_a_tool_is_offered_for_the_safe_actions_it_has() -> None:
    """A skill's blanket rating is its most dangerous action, not its only one.

    LIVE, 2026-08-27: "read the docs at <path>, then actually use it" ran under
    a sandboxed_compute ceiling, and file_operation — rated state_mutation
    because it can delete — was withheld, so the only tool that could read the
    file was gone. Its read, list and exists actions mutate nothing, and the
    dispatch is handed the ceiling and refuses each call on its own scope.
    """
    from core.brain.inference_gate import _reachable_scope

    assert _reachable_scope("file_operation", "state_mutation", _SANDBOX_CEILING) == "read_only"
    kept, withheld = _tools_within_reach({"file_operation": 1}, _SANDBOX_CEILING)
    assert "file_operation" in kept
    assert not withheld
    # Still offered under a read-only ceiling, for the same reason.
    kept, _withheld = _tools_within_reach({"file_operation": 1}, _READ_ONLY_CEILING)
    assert "file_operation" in kept


def test_a_tool_needing_a_confirmation_nobody_can_give_is_not_offered() -> None:
    """A tool whose risk is high whatever the arguments spends a call on a refusal."""
    kept, withheld = _tools_within_reach(
        {"diagnose_repo": 1, "test_generator": 2}, _SANDBOX_CEILING
    )
    assert "test_generator" in withheld
    assert "diagnose_repo" in kept


def test_a_tool_rated_from_its_arguments_is_offered_before_they_exist() -> None:
    """code_repl and run_code are rated from the snippet, and there is none yet.

    Withholding them on the worst case meant they could never be called, so
    "read these docs and actually use the library" had no execution lane at all
    and the turn ended in "I couldn't get to an answer I'd stand behind".
    Dispatch sees the real snippet and refuses there.
    """
    kept, withheld = _tools_within_reach({"code_repl": 1, "run_code": 2}, _SANDBOX_CEILING)
    assert sorted(kept) == ["code_repl", "run_code"]
    assert not withheld


def test_the_tool_that_answers_survives_the_filter() -> None:
    """diagnose_repo runs the project's own code, not the model's."""
    kept, withheld = _tools_within_reach(
        {"diagnose_repo": 1, "test_generator": 2, "file_operation": 3, "web_search": 4},
        _SANDBOX_CEILING,
    )
    assert "diagnose_repo" in kept
    assert "test_generator" in withheld, "an unconditionally high tool was offered"


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


def test_what_offers_a_tool_reads_its_scope_as_the_dispatch_does(monkeypatch) -> None:
    """LIVE 19 November 2026, 22:34, "Can you remind me in 20 minutes to stretch my back?": reminder is in no table of
    scopes, so it was offered at the turn's sandboxed_compute ceiling, and the dispatch, which reads the scope the
    skill declares, refused it. Her reply said the system refused the reminder; none was set."""
    from types import SimpleNamespace

    from core.container import ServiceContainer
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES, effect_scope_of

    assert "reminder" not in SKILL_EFFECT_SCOPES
    declared = SimpleNamespace(effect_scope="state_mutation")
    monkeypatch.setattr(ServiceContainer, "get",
                        lambda name, default=None: SimpleNamespace(skills={"reminder": declared})
                        if name == "capability_engine" else default)
    kept, withheld = _tools_within_reach({"reminder": 1}, _SANDBOX_CEILING)
    assert withheld == ("reminder",) and not kept
    # The skill's own declaration wins over the table, which holds only older declarations.
    assert effect_scope_of("file_operation", SimpleNamespace(effect_scope="read_only")) == "read_only"
    assert effect_scope_of("file_operation", SimpleNamespace()) == SKILL_EFFECT_SCOPES["file_operation"]
    assert effect_scope_of("no_such_skill") is None


def test_a_request_for_a_reminder_may_open_what_setting_one_does() -> None:
    """Setting a reminder writes one record the person asked for and removes nothing; a request may open that."""
    from core.phases.response_contract import _REQUESTABLE_SCOPES
    from core.skills.catalog_policy import effect_scope_of
    from core.skills.reminder import ReminderSkill

    assert effect_scope_of("reminder", ReminderSkill) in _REQUESTABLE_SCOPES
