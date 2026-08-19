"""Every capability can reach the tool loop, not only search.

LIVE, 2026-08-19. Asked to run a little Python with ``code_repl`` READY, she
produced a function and the line ``Output: 94867200.0``. Nothing executed and
the number was wrong.

The runtime has a complete tool-calling loop: it parses a call out of the
generation, binds the arguments to the tool's advertised JSON schema, executes
it and feeds the result back for another turn. Reaching it requires
``should_force_tool_handoff``, which required ``contract.requires_search``.
``ResponseContract.required_skill`` was a general field assigned exactly one
value in the whole codebase::

    required_skill="web_search" if requires_search else None

So the loop served search and nothing else, and roughly sixty other registered
capabilities could not be called from a foreground turn under any phrasing.
"""

from __future__ import annotations

from core.brain.llm.runtime_wiring import should_force_tool_handoff
from core.phases.response_contract import (
    _SELF_SERVICE_EFFECT_SCOPES,
    ResponseContract,
)


def test_a_search_turn_still_hands_off():
    """The behaviour that already worked must not change."""
    contract = ResponseContract(requires_search=True, required_skill="web_search")
    assert should_force_tool_handoff(contract, is_background=False)


def test_a_non_search_capability_now_hands_off_too():
    """The live miss: executing code could never reach the tool loop."""
    contract = ResponseContract(requires_search=False, required_skill="code_repl")
    assert should_force_tool_handoff(contract, is_background=False)


def test_a_turn_needing_no_capability_does_not_hand_off():
    contract = ResponseContract(requires_search=False, required_skill=None)
    assert not should_force_tool_handoff(contract, is_background=False)


def test_evidence_already_in_hand_ends_the_obligation():
    contract = ResponseContract(
        requires_search=False,
        required_skill="code_repl",
        tool_evidence_available=True,
    )
    assert not should_force_tool_handoff(contract, is_background=False)


def test_background_work_is_never_forced_into_a_handoff():
    contract = ResponseContract(requires_search=False, required_skill="code_repl")
    assert not should_force_tool_handoff(contract, is_background=True)


def test_only_recoverable_effects_are_entered_unasked():
    """A request that merely sounds like a capability must not reach the world.

    Sandboxed computation and reading are recoverable. Writing, external I/O
    and driving the desktop are the person's call, and each already has its
    own explicit path through the chat route.
    """
    assert "sandboxed_compute" in _SELF_SERVICE_EFFECT_SCOPES
    assert "pure_compute" in _SELF_SERVICE_EFFECT_SCOPES
    assert "read_only" in _SELF_SERVICE_EFFECT_SCOPES
    for dangerous in (
        "external_io",
        "state_mutation",
        "read_write_artifacts",
        "foreground_desktop_control",
        "foreground_browser_dialogue",
        "write",
        "external",
    ):
        assert dangerous not in _SELF_SERVICE_EFFECT_SCOPES, dangerous


def test_no_capability_engine_means_no_inferred_skill():
    """Contract building must never depend on a service being up."""
    from core.phases.response_contract import derive_required_skill

    assert derive_required_skill("") is None
