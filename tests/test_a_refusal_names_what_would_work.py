"""A refusal that names no alternative is a dead end.

LIVE, 2026-08-27: file_operation was leased read_only for the turn, and the
model asked it to WRITE a script so it could exercise a library — twice. Both
calls came back "denied_by_default: tool_execution requires validated scoped
authority", which says what went wrong and nothing about what would work, while
code_repl sat offered on the same turn, able to import that library and run it.

This is the complement of not offering a tool that will be refused: when one is
refused anyway, say what remains.
"""

from __future__ import annotations

from core.capability_engine import _name_what_is_still_available

_VETO = {
    "ok": False,
    "error": "Executive veto: denied_by_default: tool_execution requires validated scoped authority",
    "status": "blocked_by_executive",
}


def test_the_refusal_names_the_other_tools_this_turn_has() -> None:
    told = _name_what_is_still_available(
        _VETO, "file_operation", {"required_skills": ["file_operation", "code_repl"]}
    )
    assert told["available_instead"] == ["code_repl"]
    assert "Still available on this turn: code_repl." in told["error"]
    # The original reason survives; this adds, it does not replace.
    assert "denied_by_default" in told["error"]


def test_the_refused_tool_is_not_offered_back() -> None:
    told = _name_what_is_still_available(
        _VETO, "file_operation", {"required_skills": ["file_operation"]}
    )
    assert "available_instead" not in told
    assert told["error"] == _VETO["error"]


def test_a_turn_with_nothing_else_says_nothing_else() -> None:
    for context in ({}, None, {"required_skills": "not a list"}):
        assert _name_what_is_still_available(_VETO, "file_operation", context) == _VETO


def test_the_denial_is_not_mutated() -> None:
    """The caller's payload is theirs."""
    original = dict(_VETO)
    _name_what_is_still_available(
        _VETO, "file_operation", {"required_skills": ["file_operation", "code_repl"]}
    )
    assert _VETO == original
