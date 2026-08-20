"""A tool call is an execution turn and cannot be clamped below its own size.

LIVE, 2026-08-19. Every tool-using turn came back:

    ⚠️ [WORKER] Generation produced 1 token(s) but no text survived to the
    caller — discarded downstream, not a decode failure

The prompt was correct and ended in an open assistant turn. The tools were
offered, the schema was valid, the routing was right. The budget was one
token, because under unified-memory pressure the cap is applied per contract:
a desktop execution turn keeps a 1024-token plan floor, a clean user surface
keeps its completion floor, and everything else takes the raw pressure cap.

A tool generation matched none of them. Days of "she will not use her tools"
was a token budget, and the comment beside the desktop floor already stated
the principle: clamped below its plan, she cannot express the steps, so
nothing runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import (
    _TOOL_CALL_TOKEN_FLOOR,
    _apply_memory_pressure_generation_controls,
)


def _under_pressure(cap: int = 1):
    return SimpleNamespace(max_token_cap=cap)


def test_a_tool_turn_keeps_room_for_a_call():
    options = {"tools": {"file_operation": {}}, "max_tokens": 4096}
    applied = _apply_memory_pressure_generation_controls(options, _under_pressure())
    assert applied["max_tokens"] >= _TOOL_CALL_TOKEN_FLOOR


def test_the_floor_is_big_enough_for_a_name_and_arguments():
    """Below this a call cannot finish, and a half-call reads as no call."""
    assert _TOOL_CALL_TOKEN_FLOOR >= 128


def test_a_turn_with_no_tools_is_still_clamped():
    """The floor is for calls, not a way around memory pressure."""
    applied = _apply_memory_pressure_generation_controls({"max_tokens": 4096}, _under_pressure())
    assert applied["max_tokens"] == 1


def test_the_desktop_plan_floor_is_untouched():
    applied = _apply_memory_pressure_generation_controls(
        {"desktop_execution_contract": True, "max_tokens": 4096}, _under_pressure()
    )
    assert applied["max_tokens"] >= 1024


def test_without_pressure_the_full_budget_survives():
    applied = _apply_memory_pressure_generation_controls(
        {"tools": {"x": {}}, "max_tokens": 4096}, SimpleNamespace(max_token_cap=None)
    )
    assert applied["max_tokens"] == 4096


@pytest.mark.parametrize("cap", [1, 8, 64, 319])
def test_any_cap_below_the_floor_is_raised_for_a_tool_turn(cap: int):
    applied = _apply_memory_pressure_generation_controls(
        {"tools": {"x": {}}, "max_tokens": 4096}, _under_pressure(cap)
    )
    assert applied["max_tokens"] >= _TOOL_CALL_TOKEN_FLOOR


def test_a_cap_above_the_floor_is_left_alone():
    """The floor raises a budget that is too small; it does not lower one."""
    applied = _apply_memory_pressure_generation_controls(
        {"tools": {"x": {}}, "max_tokens": 4096}, _under_pressure(2048)
    )
    assert applied["max_tokens"] >= 2048
