"""The router and the tool loop cannot disagree about what a turn needs.

LIVE, 2026-08-19. Two mechanisms decided this independently. Asked to read a
repository and find a failing test, the capability router nominated
``uplink_local`` — whose description mentions a state-repository — and omitted
``file_operation`` entirely, while the tool handoff had the right five. The
router sent the turn to a skill that could not do the job while the loop knew
which one could.

Both now call one selector, so the disagreement is not a bug to be fixed again
but a state that cannot be reached.
"""

from __future__ import annotations

import pytest

from core.intent.capability_selection import select_capabilities


class _Meta:
    def __init__(self, description: str, scope: str, module="", cls=""):
        self.description = description
        self.effect_scope = scope
        self.enabled = True
        self.skill_class = None
        self.instance = None
        self.module_path = module
        self.class_name = cls


@pytest.fixture
def skills():
    return {
        "file_operation": _Meta(
            "Read, write, append, or list files in the allowed workspace.",
            "state_mutation",
            "core.skills.file_operation",
            "FileOperationSkill",
        ),
        "code_repl": _Meta("Execute Python code in a real-time, sandboxed REPL", "sandboxed_compute"),
        "web_search": _Meta("Search the web for current information", "read_only"),
        "uplink_local": _Meta(
            "Verify local persistence for real: state-repository health", "read_only"
        ),
        "sovereign_terminal": _Meta("Execute shell commands, launch system apps", "privileged_mutation"),
    }


def test_a_file_task_gets_the_file_reader(skills):
    chosen = select_capabilities(
        "there's a python project at /tmp/ledger - one of its tests is failing. "
        "read the code, work out why",
        skills,
        ceiling="sandboxed_compute",
        admissible_scopes=frozenset({"read_only", "sandboxed_compute", "pure_compute", "status"}),
    )
    assert "file_operation" in chosen


def test_a_skill_needing_more_authority_than_the_turn_has_is_not_offered(skills):
    """sovereign_terminal is privileged_mutation and declares no safe action."""
    chosen = select_capabilities(
        "run the tests in /tmp/ledger",
        skills,
        ceiling="sandboxed_compute",
        admissible_scopes=frozenset({"read_only", "sandboxed_compute", "pure_compute", "status"}),
    )
    assert "sovereign_terminal" not in chosen


def test_conversation_is_offered_nothing(skills):
    for talk in ("how are you feeling today", "tell me a story about the sea"):
        assert (
            select_capabilities(
                talk,
                skills,
                ceiling="sandboxed_compute",
                admissible_scopes=frozenset({"read_only", "sandboxed_compute"}),
            )
            == []
        ), talk


def test_the_router_and_the_contract_ask_the_same_function():
    """Structural, because agreement by coincidence is what broke."""
    import inspect

    from core.capability_engine import CapabilityEngine
    from core.phases import response_contract

    router = inspect.getsource(CapabilityEngine._foundational_candidates)
    contract = inspect.getsource(response_contract.derive_capability_set)
    assert "select_capabilities" in router
    assert "select_capabilities" in contract


def test_the_limit_is_shared_so_the_sets_are_the_same_length():
    import inspect

    from core.capability_engine import CapabilityEngine

    assert "DEFAULT_CAPABILITY_SET" in inspect.getsource(
        CapabilityEngine._foundational_candidates
    )
